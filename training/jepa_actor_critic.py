"""
Actor-critic training via imagination rollouts through the JEPA predictor.

Key differences from RSSM-based actor_critic.py:
  - Imagination uses the transformer predictor (not RSSM.imagine_step)
  - Truncated history window (history_size=3, like LeWM) for efficiency
  - Actor/critic take a single embedding (no h/z split)
  - No prior sampling — predictor is deterministic given context
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")

from models.actor import Actor
from models.critic import Critic
from training.jepa_world_model import JEPAWorldModel


def compute_lambda_returns(rewards, values, gamma: float = 0.997,
                           lambda_: float = 0.95, continuations=None):
    """Compute lambda-returns for a sequence.

    Args:
        rewards:       (B, H) predicted rewards from imagination
        values:        (B, H) value estimates
        gamma:         discount factor
        lambda_:       trace decay
        continuations: (B, H) predicted P(continue) per step, or None

    Returns:
        (B, H) lambda-return targets
    """
    B, H = rewards.shape
    returns = torch.zeros_like(rewards)
    next_return = values[:, -1]

    for t in reversed(range(H)):
        cont = 1.0 if continuations is None else continuations[:, t]
        returns[:, t] = rewards[:, t] + gamma * cont * (
            (1 - lambda_) * values[:, t] + lambda_ * next_return
        )
        next_return = returns[:, t]

    return returns


class JEPAActorCriticTrainer:
    def __init__(self, actor: Actor, critic: Critic, world_model: JEPAWorldModel,
                 cfg: dict, device: torch.device):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.world_model = world_model  # already on device, frozen during AC training
        self.device = device

        # Target critic
        self.target_critic = copy.deepcopy(critic).to(device)
        self.target_critic.requires_grad_(False)
        self.critic_ema_decay = cfg.get("critic_ema_decay", 0.98)

        self.horizon = cfg.get("horizon", 15)
        self.gamma = cfg.get("gamma", 0.997)
        self.lambda_ = cfg.get("lambda_", 0.95)
        self.entropy_coeff = cfg.get("entropy_coeff", 3e-3)
        self.history_size = cfg.get("history_size", 3)

        actor_lr = cfg.get("actor_lr", cfg.get("learning_rate", 1e-4))
        critic_lr = cfg.get("critic_lr", cfg.get("learning_rate", 1e-4))
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=actor_lr, eps=1e-5, weight_decay=5e-4)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=critic_lr, eps=1e-5, weight_decay=5e-4)
        self.max_grad_norm = cfg.get("max_grad_norm", 2.0)
        self.critic_grad_clip = cfg.get("critic_grad_clip", self.max_grad_norm)

        # Return normalization — EMA of 5th/95th percentiles
        self._return_ema_low = None
        self._return_ema_high = None
        self._return_ema_decay = 0.99

    def _update_target_critic(self):
        tau = 1.0 - self.critic_ema_decay
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.lerp_(p.data, tau)

    def _normalize_returns(self, returns):
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

    def imagine_rollout(self, initial_emb):
        """Roll out in imagination using actor + predictor.

        Uses truncated history window (history_size) for the predictor,
        matching LeWM's rollout strategy for efficiency.

        Args:
            initial_emb: (B, D) starting embedding from real observations.

        Returns:
            emb_seq, rewards, values, log_probs, entropies, continuations
        """
        HS = self.history_size
        emb_buffer = [initial_emb]   # list of (B, D)
        act_buffer = []              # list of (B, act_dim)

        emb_list = []
        reward_list, value_list = [], []
        log_prob_list, entropy_list, cont_list = [], [], []

        for step_i in range(self.horizon):
            current_emb = emb_buffer[-1]

            # Bail early if embeddings went NaN (prevents corrupting actor)
            if torch.isnan(current_emb).any():
                break

            # Actor selects action (has grad for REINFORCE)
            action, log_prob, entropy = self.actor.get_action(current_emb)

            with torch.no_grad():
                act_buffer.append(action.detach())

                # Prepare truncated context: last HS (embedding, action) pairs
                ctx_embs = torch.stack(emb_buffer[-HS:], dim=1)  # (B, L, D)
                ctx_acts = torch.stack(act_buffer[-HS:], dim=1)   # (B, L, act_dim)

                # Predict next embedding — clamp to prevent drift over H steps
                pred = self.world_model.predictor(ctx_embs, ctx_acts)
                next_emb = pred[:, -1].clamp(-10, 10)  # (B, D)
                emb_buffer.append(next_emb)

                # Predict reward, value, continuation from next state
                # Keep reward and value in symlog space so lambda-returns
                # stay bounded — symexp here caused a self-reinforcing
                # value explosion (9740 when true value is ~-8).
                reward = self.world_model.reward_pred(next_emb).clamp(-5, 5)
                value = self.target_critic(next_emb).clamp(-10, 10)
                cont = torch.sigmoid(self.world_model.cont_pred(next_emb))

            emb_list.append(next_emb)
            reward_list.append(reward)
            value_list.append(value)
            log_prob_list.append(log_prob)
            entropy_list.append(entropy)
            cont_list.append(cont)

        if not emb_list:
            # Early break on first step — return zero-length tensors
            B = initial_emb.shape[0]
            D = initial_emb.shape[1]
            empty = lambda *s: torch.zeros(*s, device=self.device)
            return (
                empty(B, 0, D), empty(B, 0), empty(B, 0),
                empty(B, 0), empty(B, 0), empty(B, 0),
            )

        return (
            torch.stack(emb_list, dim=1),       # (B, H, D)
            torch.stack(reward_list, dim=1),     # (B, H)
            torch.stack(value_list, dim=1),      # (B, H)
            torch.stack(log_prob_list, dim=1),   # (B, H)
            torch.stack(entropy_list, dim=1),    # (B, H)
            torch.stack(cont_list, dim=1),       # (B, H)
        )

    def train_step(self, emb_seq, dones=None):
        """Train actor and critic from world model embeddings.

        Args:
            emb_seq: (B, T, D) embeddings from JEPA world model
            dones:   (B, T) done flags (optional)

        Returns:
            dict of losses
        """
        B, T, _ = emb_seq.shape

        # Pick random starting states, avoiding terminal and last step
        if dones is not None:
            valid = (dones < 0.5) & (
                torch.arange(T, device=self.device).unsqueeze(0) < T - 1
            )
            t_indices = torch.zeros(B, dtype=torch.long, device=self.device)
            for b in range(B):
                valid_t = valid[b].nonzero(as_tuple=False).squeeze(-1)
                if len(valid_t) == 0:
                    valid_t = torch.arange(T - 1, device=self.device)
                t_indices[b] = valid_t[torch.randint(len(valid_t), (1,))]
        else:
            t_indices = torch.randint(0, T - 1, (B,), device=self.device)

        start_emb = emb_seq[torch.arange(B), t_indices].detach()

        # Imagination rollout
        rollout = self.imagine_rollout(start_emb)
        emb_imag, rewards, values, log_probs, entropies, conts = rollout

        # Skip AC update if imagination produced empty or NaN results
        if emb_imag.shape[1] == 0 or torch.isnan(rewards).any():
            return {
                "actor_loss": 0.0, "critic_loss": 0.0,
                "imagined_reward": 0.0, "imagined_value": 0.0,
                "entropy": 0.0, "cont_mean": 0.0,
                "actor_grad_norm": 0.0, "critic_grad_norm": 0.0,
            }

        # Lambda-returns (computed in symlog space — rewards and values
        # are already in symlog space from imagine_rollout)
        lambda_returns = compute_lambda_returns(
            rewards, values, self.gamma, self.lambda_, continuations=conts
        )

        # ── Actor loss ── REINFORCE with return normalization
        normed_returns = self._normalize_returns(lambda_returns)
        actor_loss = -(normed_returns.detach() * log_probs).mean() \
                     - self.entropy_coeff * entropies.mean()

        if torch.isnan(actor_loss) or torch.isinf(actor_loss):
            return {
                "actor_loss": 0.0, "critic_loss": 0.0,
                "imagined_reward": rewards.mean().item(),
                "imagined_value": values.mean().item(),
                "entropy": entropies.mean().item(), "cont_mean": conts.mean().item(),
                "actor_grad_norm": 0.0, "critic_grad_norm": 0.0,
            }

        self.actor_opt.zero_grad()
        actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        # ── Critic loss ── MSE on lambda-returns (already in symlog space)
        critic_values = self.critic(emb_imag.detach())
        critic_target = lambda_returns.detach()
        critic_loss = F.mse_loss(critic_values, critic_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
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

    embed_dim = 256
    act_dim = 4

    cfg = {
        "obs_channels": 4,
        "embed_dim": embed_dim,
        "act_dim": act_dim,
        "cnn_depth": 48,
        "mlp_units": 400,
        "pred_depth": 4,
        "pred_heads": 4,
        "pred_mlp_dim": 512,
        "pred_dropout": 0.1,
        "batch_length": 30,
        "sigreg_weight": 0.1,
        "pred_weight": 1.0,
        "rollout_weight": 0.5,
        "reward_weight": 10.0,
        "use_symlog": True,
        "learning_rate": 1e-4,
        "actor_lr": 1e-4,
        "critic_lr": 1e-4,
        "mixed_precision": False,
        "max_grad_norm": 10.0,
        "horizon": 15,
        "gamma": 0.997,
        "lambda_": 0.95,
        "entropy_coeff": 3e-3,
        "critic_ema_decay": 0.98,
        "history_size": 3,
    }

    wm = JEPAWorldModel(cfg).to(device)
    actor = Actor(hidden_dim=embed_dim, stoch_dim=0, act_dim=act_dim,
                  units=400, discrete=True)
    critic = Critic(hidden_dim=embed_dim, stoch_dim=0, units=400)
    ac_trainer = JEPAActorCriticTrainer(actor, critic, wm, cfg, device)

    # Fake embeddings from world model
    B, T = 4, 20
    emb_seq = torch.randn(B, T, embed_dim, device=device)

    for step in range(3):
        losses = ac_trainer.train_step(emb_seq)
        print(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in losses.items()))

    # Verify target critic diverges
    for p, tp in zip(ac_trainer.critic.parameters(), ac_trainer.target_critic.parameters()):
        assert not torch.equal(p.data, tp.data), "Target should lag behind live critic"

    print("Smoke test passed!")
