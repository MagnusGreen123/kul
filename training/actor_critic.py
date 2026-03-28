"""
Actor-critic training via imagination rollouts through the world model.
- Imagines H=15 steps using learned RSSM dynamics
- Computes lambda-returns (lambda=0.95) with proper bootstrap
- Uses target critic (EMA / Polyak) for stable value targets
- Trains actor with backprop through rollout + entropy regularization
- Trains critic with MSE on lambda-return targets from target critic
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")

from models.actor import Actor
from models.critic import Critic
from training.world_model import WorldModel, symlog, symexp


def compute_lambda_returns(rewards, values, bootstrap, gamma: float = 0.997,
                           lambda_: float = 0.95, continuations=None):
    """Compute lambda-returns for a sequence.

    Args:
        rewards:       (B, H) predicted rewards from imagination
        values:        (B, H) value estimates for states s_1..s_H
        bootstrap:     (B,) value estimate for state s_{H+1} (beyond horizon)
        gamma:         discount factor
        lambda_:       trace decay
        continuations: (B, H) predicted P(continue) per step, or None (all 1)

    Returns:
        (B, H) lambda-return targets
    """
    B, H = rewards.shape
    returns = torch.zeros_like(rewards)

    for t in reversed(range(H)):
        next_val = bootstrap if t == H - 1 else values[:, t + 1]
        next_ret = bootstrap if t == H - 1 else returns[:, t + 1]
        # Discount by gamma * cont — at terminal states cont→0, zeroing bootstrap
        cont = 1.0 if continuations is None else continuations[:, t]
        returns[:, t] = rewards[:, t] + gamma * cont * (
            (1 - lambda_) * next_val + lambda_ * next_ret
        )

    return returns


class ActorCriticTrainer:
    def __init__(self, actor: Actor, critic: Critic, world_model: WorldModel,
                 cfg: dict, device: torch.device):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.world_model = world_model  # already on device, frozen during AC training
        self.device = device

        # Target critic — slow-moving copy for stable bootstrap targets
        self.target_critic = copy.deepcopy(critic).to(device)
        self.target_critic.requires_grad_(False)
        self.critic_ema_decay = cfg.get("critic_ema_decay", 0.98)

        self.horizon = cfg.get("horizon", 15)
        self.gamma = cfg.get("gamma", 0.997)
        self.lambda_ = cfg.get("lambda_", 0.95)
        self.use_symlog = cfg.get("use_symlog", True)
        self.entropy_coeff = cfg.get("entropy_coeff", 3e-3)

        actor_lr = cfg.get("actor_lr", cfg.get("learning_rate", 3e-4))
        critic_lr = cfg.get("critic_lr", cfg.get("learning_rate", 3e-4))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        self.max_grad_norm = cfg.get("max_grad_norm", 100.0)

        # Return normalization — EMA of 5th/95th percentiles (DreamerV3)
        self._return_ema_low = None
        self._return_ema_high = None
        self._return_ema_decay = 0.99

    def _update_target_critic(self):
        """Polyak / EMA update: target_critic slowly tracks the live critic."""
        tau = 1.0 - self.critic_ema_decay  # e.g. 0.02
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.lerp_(p.data, tau)

    def _normalize_returns(self, returns):
        """Normalize returns using EMA of 5th/95th percentiles (DreamerV3).

        Detaches normalization stats so gradients flow through raw returns only.
        Puts returns into roughly [0, 1] range, making entropy_coeff meaningful.
        """
        with torch.no_grad():
            low = torch.quantile(returns, 0.05)
            high = torch.quantile(returns, 0.95)
            if self._return_ema_low is None:
                self._return_ema_low = low
                self._return_ema_high = high
            else:
                decay = self._return_ema_decay
                self._return_ema_low = decay * self._return_ema_low + (1 - decay) * low
                self._return_ema_high = decay * self._return_ema_high + (1 - decay) * high
            scale = (self._return_ema_high - self._return_ema_low).clamp(min=1.0)
            offset = self._return_ema_low
        return (returns - offset) / scale

    def imagine_rollout(self, initial_h, initial_z):
        """Roll out in imagination using actor policy through RSSM.

        Uses target_critic for value estimates (stable targets),
        and cont_pred for continuation probabilities (done masking).

        Returns:
            h_seq:          (B, H, hidden_dim)
            z_seq:          (B, H, stoch_dim)
            actions:        (B, H, act_dim)
            rewards:        (B, H)
            values:         (B, H)  — from target_critic (no grad)
            entropies:      (B, H)
            continuations:  (B, H)  — P(continue) from cont_pred (no grad)
            bootstrap:      (B,) — target_critic value for state beyond horizon
        """
        h, z = initial_h, initial_z
        h_list, z_list, action_list = [], [], []
        reward_list, value_list, entropy_list, cont_list = [], [], [], []

        for _ in range(self.horizon):
            action, _, entropy = self.actor.get_action(h, z)
            h, z, _ = self.world_model.rssm.imagine_step(h, z, action)

            reward = self.world_model.reward_pred(h, z)
            with torch.no_grad():
                value = self.target_critic(h, z)
                cont = torch.sigmoid(self.world_model.cont_pred(h, z))

            h_list.append(h)
            z_list.append(z)
            action_list.append(action)
            reward_list.append(reward)
            value_list.append(value)
            entropy_list.append(entropy)
            cont_list.append(cont)

        # Bootstrap: value of state one step beyond horizon (target critic)
        boot_action, _, _ = self.actor.get_action(h, z)
        h_boot, z_boot, _ = self.world_model.rssm.imagine_step(h, z, boot_action)
        with torch.no_grad():
            bootstrap = self.target_critic(h_boot, z_boot)

        return (
            torch.stack(h_list, dim=1),
            torch.stack(z_list, dim=1),
            torch.stack(action_list, dim=1),
            torch.stack(reward_list, dim=1),
            torch.stack(value_list, dim=1),
            torch.stack(entropy_list, dim=1),
            torch.stack(cont_list, dim=1),
            bootstrap,
        )

    def train_step(self, h_seq, z_seq, dones=None):
        """Train actor and critic from world model states.

        Args:
            h_seq: (B, T, hidden_dim) from world model observe pass
            z_seq: (B, T, stoch_dim) from world model observe pass
            dones: (B, T) float32 done flags, used to avoid starting
                   imagination from terminal states (optional)

        Returns:
            dict of losses
        """
        B, T, _ = h_seq.shape

        # Pick random starting states, avoiding terminal steps and last step
        if dones is not None:
            # Valid starts: non-terminal, not the last timestep (leave room)
            valid = (dones < 0.5) & (torch.arange(T, device=self.device).unsqueeze(0) < T - 1)
            # Per-batch: sample from valid indices; fall back to all if none valid
            t_indices = torch.zeros(B, dtype=torch.long, device=self.device)
            for b in range(B):
                valid_t = valid[b].nonzero(as_tuple=False).squeeze(-1)
                if len(valid_t) == 0:
                    valid_t = torch.arange(T - 1, device=self.device)
                t_indices[b] = valid_t[torch.randint(len(valid_t), (1,))]
        else:
            t_indices = torch.randint(0, T - 1, (B,), device=self.device)
        start_h = h_seq[torch.arange(B), t_indices].detach()
        start_z = z_seq[torch.arange(B), t_indices].detach()

        # Freeze world model: gradients still flow to actor via inputs
        # (straight-through), but RSSM/reward_pred params don't accumulate grads
        for p in self.world_model.parameters():
            p.requires_grad_(False)

        # Single imagination rollout (values/bootstrap from target_critic)
        h_imag, z_imag, actions, rewards, values, entropies, conts, bootstrap = \
            self.imagine_rollout(start_h, start_z)

        if self.use_symlog:
            rewards = symexp(rewards)

        # Lambda-returns with stable bootstrap from target critic,
        # discounted by predicted continuation (done masking)
        lambda_returns = compute_lambda_returns(
            rewards, values, bootstrap,
            self.gamma, self.lambda_, continuations=conts
        )

        # ── Actor loss ── (maximise normalized returns + entropy bonus)
        # Normalization puts returns in ~[0,1], making entropy_coeff meaningful
        normed_returns = self._normalize_returns(lambda_returns)
        actor_loss = -(normed_returns.mean() + self.entropy_coeff * entropies.mean())

        self.actor_opt.zero_grad()
        actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        # Unfreeze world model for its own training
        for p in self.world_model.parameters():
            p.requires_grad_(True)

        # ── Critic loss ── targets from target_critic, train live critic
        critic_values = self.critic(h_imag.detach(), z_imag.detach())
        critic_loss = F.mse_loss(critic_values, lambda_returns.detach())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_opt.step()

        # ── EMA update target critic ──
        self._update_target_critic()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "imagined_reward": rewards.mean().item(),
            "imagined_value": values.mean().item(),
            "entropy": entropies.mean().item(),
            "cont_mean": conts.mean().item(),
            "actor_grad_norm": actor_grad.item(),
            "critic_grad_norm": critic_grad.item(),
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = {
        "obs_channels": 4,
        "embed_dim": 512,
        "hidden_dim": 256,
        "stoch_dim": 32,
        "act_dim": 4,
        "cnn_depth": 32,
        "kl_weight": 0.1,
        "kl_balance": 0.8,
        "free_bits": 1.0,
        "use_symlog": True,
        "learning_rate": 1e-4,
        "actor_lr": 8e-5,
        "critic_lr": 8e-5,
        "mixed_precision": False,
        "max_grad_norm": 100.0,
        "horizon": 15,
        "gamma": 0.997,
        "lambda_": 0.95,
        "entropy_coeff": 3e-3,
        "critic_ema_decay": 0.98,
    }

    wm = WorldModel(cfg).to(device)
    actor = Actor(hidden_dim=256, stoch_dim=32, act_dim=4, discrete=True)
    critic = Critic(hidden_dim=256, stoch_dim=32)
    ac_trainer = ActorCriticTrainer(actor, critic, wm, cfg, device)

    # Verify target critic exists and is separate
    assert ac_trainer.target_critic is not ac_trainer.critic
    for p in ac_trainer.target_critic.parameters():
        assert not p.requires_grad

    # Fake world model states
    B, T = 4, 20
    h_seq = torch.randn(B, T, 256, device=device)
    z_seq = torch.randn(B, T, 32, device=device)

    for step in range(5):
        losses = ac_trainer.train_step(h_seq, z_seq)
        print(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in losses.items()))

    # Verify target critic params diverge from live critic (EMA is updating)
    for p, tp in zip(ac_trainer.critic.parameters(), ac_trainer.target_critic.parameters()):
        assert not torch.equal(p.data, tp.data), "Target should lag behind live critic"

    print("Smoke test passed!")
