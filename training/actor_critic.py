"""
Actor-critic training via imagination rollouts through the world model.
- Imagines H=15 steps using learned RSSM dynamics
- Computes lambda-returns (lambda=0.95)
- Trains actor with backprop through rollout (straight-through gradients)
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


def compute_lambda_returns(rewards, values, gamma: float = 0.99, lambda_: float = 0.95):
    """Compute lambda-returns for a sequence.

    Args:
        rewards: (B, H) predicted rewards from imagination
        values:  (B, H) value estimates
        gamma:   discount factor
        lambda_: trace decay

    Returns:
        (B, H) lambda-return targets
    """
    B, H = rewards.shape
    returns = torch.zeros_like(rewards)
    last = values[:, -1]

    for t in reversed(range(H)):
        if t == H - 1:
            returns[:, t] = rewards[:, t] + gamma * last
        else:
            returns[:, t] = rewards[:, t] + gamma * (
                (1 - lambda_) * values[:, t + 1] + lambda_ * returns[:, t + 1]
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
        self.gamma = cfg.get("gamma", 0.99)
        self.lambda_ = cfg.get("lambda_", 0.95)
        self.use_symlog = cfg.get("use_symlog", True)

        actor_lr = cfg.get("actor_lr", cfg.get("learning_rate", 3e-4))
        critic_lr = cfg.get("critic_lr", cfg.get("learning_rate", 3e-4))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        self.max_grad_norm = cfg.get("max_grad_norm", 100.0)

    def imagine_rollout(self, initial_h, initial_z):
        """Roll out in imagination using actor policy through RSSM.

        Args:
            initial_h: (B, hidden_dim) - starting deterministic state
            initial_z: (B, stoch_dim) - starting stochastic state

        Returns:
            h_seq:   (B, H, hidden_dim)
            z_seq:   (B, H, stoch_dim)
            actions: (B, H, act_dim)
            rewards: (B, H)
            values:  (B, H)
        """
        h, z = initial_h, initial_z
        h_list, z_list, action_list, reward_list, value_list = [], [], [], [], []

        for _ in range(self.horizon):
            action, _ = self.actor.get_action(h, z)
            h, z, _ = self.world_model.rssm.imagine_step(h, z, action)

            reward = self.world_model.reward_pred(h, z)
            value = self.critic(h, z)

            h_list.append(h)
            z_list.append(z)
            action_list.append(action)
            reward_list.append(reward)
            value_list.append(value)

        return (
            torch.stack(h_list, dim=1),
            torch.stack(z_list, dim=1),
            torch.stack(action_list, dim=1),
            torch.stack(reward_list, dim=1),
            torch.stack(value_list, dim=1),
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
        start_h = h_seq[torch.arange(B), t_indices]  # (B, hidden_dim)
        start_z = z_seq[torch.arange(B), t_indices]  # (B, stoch_dim)

        # Imagine forward
        h_imag, z_imag, actions, rewards, values = self.imagine_rollout(
            start_h.detach(), start_z.detach()
        )

        # Undo symlog on rewards if needed
        if self.use_symlog:
            rewards = symexp(rewards)

        # Lambda-returns (detached for critic target)
        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards, values.detach(), self.gamma, self.lambda_
            )

        # ── Critic loss ──
        critic_loss = F.mse_loss(values, lambda_returns.detach())

        self.critic_opt.zero_grad()
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_opt.step()

        # ── Actor loss ──
        # Re-imagine with fresh graph; gradients flow through actor -> RSSM -> reward_pred
        h_imag2, z_imag2, actions2, rewards2, values2 = self.imagine_rollout(
            start_h.detach(), start_z.detach()
        )
        if self.use_symlog:
            rewards2 = symexp(rewards2)

        # Lambda-returns WITH gradient through rewards (actor can influence them)
        # but values are detached (critic is a fixed baseline here)
        lambda_returns2 = compute_lambda_returns(
            rewards2, values2.detach(), self.gamma, self.lambda_
        )

        # Actor maximizes lambda-returns (negate for gradient descent)
        actor_loss = -lambda_returns2.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "imagined_reward": rewards.mean().item(),
            "imagined_value": values.mean().item(),
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
        "kl_weight": 1.0,
        "kl_balance": 0.8,
        "free_bits": 1.0,
        "use_symlog": True,
        "learning_rate": 3e-4,
        "mixed_precision": False,
        "max_grad_norm": 100.0,
        "horizon": 15,
        "gamma": 0.99,
        "lambda_": 0.95,
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
