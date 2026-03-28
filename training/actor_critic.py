"""
Actor-critic training via imagination rollouts through the world model.
- Imagines H=15 steps using learned RSSM dynamics
- Computes lambda-returns (lambda=0.95) with proper bootstrap
- Trains actor with backprop through rollout + entropy regularization
- Trains critic with MSE on lambda-return targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")

from models.actor import Actor
from models.critic import Critic
from training.world_model import WorldModel, symlog, symexp


def compute_lambda_returns(rewards, values, bootstrap, gamma: float = 0.997,
                           lambda_: float = 0.95):
    """Compute lambda-returns for a sequence.

    Args:
        rewards:   (B, H) predicted rewards from imagination
        values:    (B, H) value estimates for states s_1..s_H
        bootstrap: (B,) value estimate for state s_{H+1} (beyond horizon)
        gamma:     discount factor
        lambda_:   trace decay

    Returns:
        (B, H) lambda-return targets
    """
    B, H = rewards.shape
    returns = torch.zeros_like(rewards)

    for t in reversed(range(H)):
        next_val = bootstrap if t == H - 1 else values[:, t + 1]
        next_ret = bootstrap if t == H - 1 else returns[:, t + 1]
        returns[:, t] = rewards[:, t] + gamma * (
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

        self.horizon = cfg.get("horizon", 15)
        self.gamma = cfg.get("gamma", 0.997)
        self.lambda_ = cfg.get("lambda_", 0.95)
        self.use_symlog = cfg.get("use_symlog", True)
        self.entropy_coeff = cfg.get("entropy_coeff", 1e-3)

        actor_lr = cfg.get("actor_lr", cfg.get("learning_rate", 3e-4))
        critic_lr = cfg.get("critic_lr", cfg.get("learning_rate", 3e-4))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        self.max_grad_norm = cfg.get("max_grad_norm", 100.0)

    def imagine_rollout(self, initial_h, initial_z):
        """Roll out in imagination using actor policy through RSSM.

        Returns:
            h_seq:     (B, H, hidden_dim)
            z_seq:     (B, H, stoch_dim)
            actions:   (B, H, act_dim)
            rewards:   (B, H)
            values:    (B, H)
            entropies: (B, H)
            bootstrap: (B,) — value estimate for state beyond horizon
        """
        h, z = initial_h, initial_z
        h_list, z_list, action_list = [], [], []
        reward_list, value_list, entropy_list = [], [], []

        for _ in range(self.horizon):
            action, _, entropy = self.actor.get_action(h, z)
            h, z, _ = self.world_model.rssm.imagine_step(h, z, action)

            reward = self.world_model.reward_pred(h, z)
            value = self.critic(h, z)

            h_list.append(h)
            z_list.append(z)
            action_list.append(action)
            reward_list.append(reward)
            value_list.append(value)
            entropy_list.append(entropy)

        # Bootstrap: value of state one step beyond horizon
        boot_action, _, _ = self.actor.get_action(h, z)
        h_boot, z_boot, _ = self.world_model.rssm.imagine_step(h, z, boot_action)
        bootstrap = self.critic(h_boot, z_boot)

        return (
            torch.stack(h_list, dim=1),
            torch.stack(z_list, dim=1),
            torch.stack(action_list, dim=1),
            torch.stack(reward_list, dim=1),
            torch.stack(value_list, dim=1),
            torch.stack(entropy_list, dim=1),
            bootstrap,
        )

    def train_step(self, h_seq, z_seq):
        """Train actor and critic from world model states.

        Args:
            h_seq: (B, T, hidden_dim) from world model observe pass
            z_seq: (B, T, stoch_dim) from world model observe pass

        Returns:
            dict of losses
        """
        B, T, _ = h_seq.shape

        # Pick random starting states from the observed sequence
        t_indices = torch.randint(0, T, (B,), device=self.device)
        start_h = h_seq[torch.arange(B), t_indices].detach()
        start_z = z_seq[torch.arange(B), t_indices].detach()

        # Freeze world model: gradients still flow to actor via inputs
        # (straight-through), but RSSM/reward_pred params don't accumulate grads
        for p in self.world_model.parameters():
            p.requires_grad_(False)

        # Single imagination rollout
        h_imag, z_imag, actions, rewards, values, entropies, bootstrap = \
            self.imagine_rollout(start_h, start_z)

        if self.use_symlog:
            rewards = symexp(rewards)

        # Lambda-returns with proper bootstrap from state beyond horizon
        # Values detached so critic serves as a fixed baseline
        lambda_returns = compute_lambda_returns(
            rewards, values.detach(), bootstrap.detach(),
            self.gamma, self.lambda_
        )

        # ── Actor loss ── (maximise lambda-returns + entropy bonus)
        actor_loss = -(lambda_returns.mean() + self.entropy_coeff * entropies.mean())

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        # Unfreeze world model for its own training
        for p in self.world_model.parameters():
            p.requires_grad_(True)

        # ── Critic loss ── (detached states, no imagination graph needed)
        critic_values = self.critic(h_imag.detach(), z_imag.detach())
        critic_loss = F.mse_loss(critic_values, lambda_returns.detach())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_opt.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "imagined_reward": rewards.mean().item(),
            "imagined_value": values.mean().item(),
            "entropy": entropies.mean().item(),
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
        "actor_lr": 3e-5,
        "critic_lr": 3e-5,
        "mixed_precision": False,
        "max_grad_norm": 100.0,
        "horizon": 15,
        "gamma": 0.997,
        "lambda_": 0.95,
        "entropy_coeff": 1e-3,
    }

    wm = WorldModel(cfg).to(device)
    actor = Actor(hidden_dim=256, stoch_dim=32, act_dim=4, discrete=True)
    critic = Critic(hidden_dim=256, stoch_dim=32)
    ac_trainer = ActorCriticTrainer(actor, critic, wm, cfg, device)

    # Fake world model states
    B, T = 4, 20
    h_seq = torch.randn(B, T, 256, device=device)
    z_seq = torch.randn(B, T, 32, device=device)

    for step in range(5):
        losses = ac_trainer.train_step(h_seq, z_seq)
        print(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in losses.items()))

    print("Smoke test passed!")
