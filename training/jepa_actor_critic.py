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
from training.jepa_world_model import JEPAWorldModel, symlog


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
        self.categorical_critic = critic.categorical

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

        # Real TD target weight (reality anchor for critic)
        self.real_td_weight = cfg.get("real_td_weight", 0.0)
        self.use_symlog = cfg.get("use_symlog", True)

        # Return normalization — EMA of 5th/95th percentiles (for actor only)
        self._return_ema_low = None
        self._return_ema_high = None
        self._return_ema_decay = 0.99

    def _update_target_critic(self):
        tau = 1.0 - self.critic_ema_decay
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.lerp_(p.data, tau)

    def _update_return_stats(self, returns):
        """Update EMA of return distribution percentiles (call once per train step)."""
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

    def _return_scale_offset(self):
        if self._return_ema_low is None:
            return 1.0, 0.0
        scale = (self._return_ema_high - self._return_ema_low).clamp(min=1.0)
        return scale, self._return_ema_low

    def _normalize_returns(self, returns):
        scale, offset = self._return_scale_offset()
        return (returns - offset) / scale

    def _denormalize_returns(self, normalized):
        """Convert critic output (normalized space) back to symlog space."""
        scale, offset = self._return_scale_offset()
        return normalized * scale + offset

    def _compute_real_td_loss(self, emb_seq, rewards, dones):
        """Compute critic loss on real 1-step TD targets (reality anchor).

        Args:
            emb_seq: (B, T, D) real embeddings from encoder
            rewards: (B, T) real rewards (reward[t] = reward after action at t)
            dones:   (B, T) done flags or None

        Returns:
            scalar loss
        """
        B, T, D = emb_seq.shape
        emb_t = emb_seq[:, :-1].detach()      # (B, T-1, D)
        emb_next = emb_seq[:, 1:].detach()     # (B, T-1, D)
        rew = rewards[:, :-1]                   # (B, T-1)

        if dones is not None:
            cont = 1.0 - dones[:, :-1]          # (B, T-1)
        else:
            cont = 1.0

        with torch.no_grad():
            rew_symlog = symlog(rew) if self.use_symlog else rew
            next_value = self.target_critic(emb_next)  # (B, T-1)
            td_target = rew_symlog + self.gamma * cont * next_value
            td_target = td_target.clamp(-10, 10)

        return self.critic.compute_loss(emb_t, td_target)

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
                # Target critic outputs scalar value directly:
                # - Categorical mode: expected value over bins (already in symlog space)
                # - Scalar mode: raw scalar output
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

    def train_step(self, emb_seq, dones=None, real_rewards=None):
        """Train actor and critic from world model embeddings.

        Args:
            emb_seq:      (B, T, D) embeddings from JEPA world model
            dones:        (B, T) done flags (optional)
            real_rewards: (B, T) real rewards for TD anchor (optional)

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

        # ── Return normalization for actor (percentile-based, DreamerV3-style) ──
        # Only used for actor REINFORCE signal, NOT for critic targets.
        self._update_return_stats(lambda_returns)
        normed_returns = self._normalize_returns(lambda_returns)

        # ── Actor loss ── REINFORCE with normalized returns
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

        # ── Critic loss ──
        # Categorical mode: cross-entropy against two-hot encoded lambda-returns
        #   (bounded gradients, handles regime shifts naturally)
        # Scalar mode: MSE on raw lambda-returns (v6-v9 fallback)
        imag_critic_loss = self.critic.compute_loss(
            emb_imag.detach(), lambda_returns.detach()
        )

        # ── Real TD anchor (v12+) ── breaks self-reinforcing feedback loop
        real_td_loss_val = 0.0
        if self.real_td_weight > 0 and real_rewards is not None:
            real_td_loss = self._compute_real_td_loss(emb_seq, real_rewards, dones)
            critic_loss = (1 - self.real_td_weight) * imag_critic_loss + self.real_td_weight * real_td_loss
            real_td_loss_val = real_td_loss.item()
        else:
            critic_loss = imag_critic_loss

        self.critic_opt.zero_grad()
        critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
        self.critic_opt.step()

        # ── EMA update target critic ──
        self._update_target_critic()

        # ── Rollout health diagnostics ──
        with torch.no_grad():
            imag_norm_start = emb_imag[:, 0].norm(dim=-1).mean().item()
            imag_norm_end = emb_imag[:, -1].norm(dim=-1).mean().item()
            return_scale = (
                (self._return_ema_high - self._return_ema_low).item()
                if self._return_ema_low is not None else 0.0
            )

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "imagined_reward": rewards.mean().item(),
            "imagined_value": values.mean().item(),
            "entropy": entropies.mean().item(),
            "cont_mean": conts.mean().item(),
            "actor_grad_norm": actor_grad.item(),
            "critic_grad_norm": critic_grad.item(),
            "imag_emb_norm_start": imag_norm_start,
            "imag_emb_norm_end": imag_norm_end,
            "return_scale": return_scale,
            "real_td_loss": real_td_loss_val,
        }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    embed_dim = 256
    act_dim = 4

    base_cfg = {
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

    wm = JEPAWorldModel(base_cfg).to(device)

    # Test scalar mode (backwards compatible)
    print("--- Scalar critic ---")
    actor_s = Actor(hidden_dim=embed_dim, stoch_dim=0, act_dim=act_dim,
                    units=400, discrete=True)
    critic_s = Critic(hidden_dim=embed_dim, stoch_dim=0, units=400)
    ac_s = JEPAActorCriticTrainer(actor_s, critic_s, wm, base_cfg, device)

    B, T = 4, 20
    emb_seq = torch.randn(B, T, embed_dim, device=device)
    for step in range(3):
        losses = ac_s.train_step(emb_seq)
        print(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in losses.items()))

    # Test categorical mode (v10)
    print("\n--- Categorical critic (128 bins) ---")
    actor_c = Actor(hidden_dim=embed_dim, stoch_dim=0, act_dim=act_dim,
                    units=400, discrete=True)
    critic_c = Critic(hidden_dim=embed_dim, stoch_dim=0, units=400,
                      num_bins=128, bin_low=-3.0, bin_high=3.0)
    ac_c = JEPAActorCriticTrainer(actor_c, critic_c, wm, base_cfg, device)

    for step in range(3):
        losses = ac_c.train_step(emb_seq)
        print(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in losses.items()))

    # Verify target critic diverges
    for p, tp in zip(ac_c.critic.parameters(), ac_c.target_critic.parameters()):
        assert not torch.equal(p.data, tp.data), "Target should lag behind live critic"

    print("\nBoth modes passed!")
