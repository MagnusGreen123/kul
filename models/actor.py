"""
Actor MLP: (h, z) -> action distribution.
Supports discrete (Categorical) and continuous (TanhNormal) actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


class Actor(nn.Module):
    def __init__(self, hidden_dim: int = 256, stoch_dim: int = 32,
                 act_dim: int = 4, units: int = 256, discrete: bool = True,
                 unimix: float = 0.01):
        super().__init__()
        self.discrete = discrete
        self.act_dim = act_dim
        self.unimix = unimix

        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim + stoch_dim, units),
            nn.ELU(),
            nn.Linear(units, units),
            nn.ELU(),
        )

        if discrete:
            self.head = nn.Linear(units, act_dim)
        else:
            self.mean_head = nn.Linear(units, act_dim)
            self.log_std_head = nn.Linear(units, act_dim)

    def forward(self, h: torch.Tensor, z: torch.Tensor = None):
        """
        Returns:
            dist: action distribution
        """
        x = self.trunk(torch.cat([h, z], dim=-1) if z is not None else h)

        if self.discrete:
            logits = self.head(x)
            # Unimix: blend with uniform distribution to prevent entropy collapse
            # without needing a large entropy bonus (DreamerV3 technique)
            if self.unimix > 0:
                probs = F.softmax(logits, dim=-1)
                probs = (1 - self.unimix) * probs + self.unimix / self.act_dim
                return Categorical(probs=probs)
            return Categorical(logits=logits)
        else:
            mean = self.mean_head(x)
            log_std = self.log_std_head(x).clamp(-5, 2)
            std = log_std.exp()
            base_dist = Normal(mean, std)
            return TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])

    def get_action(self, h: torch.Tensor, z: torch.Tensor = None):
        """Sample action and return (action, log_prob, entropy).

        For discrete: uses straight-through gradients so that the forward
        pass sees a hard one-hot but gradients flow through the softmax
        probabilities back to the actor parameters.
        """
        dist = self.forward(h, z)

        if self.discrete:
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            entropy = dist.entropy()
            # One-hot for feeding back into RSSM
            action_onehot = torch.zeros(*action_idx.shape, self.act_dim,
                                        device=h.device)
            action_onehot.scatter_(-1, action_idx.unsqueeze(-1), 1.0)
            return action_onehot, log_prob, entropy
        else:
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
            return action, log_prob, entropy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test discrete
    actor_d = Actor(hidden_dim=256, stoch_dim=32, act_dim=4, discrete=True).to(device)
    h = torch.randn(8, 256, device=device)
    z = torch.randn(8, 32, device=device)
    action, log_prob, entropy = actor_d.get_action(h, z)
    print(f"Discrete action: {action.shape}, log_prob: {log_prob.shape}, entropy: {entropy.shape}")

    # Test continuous
    actor_c = Actor(hidden_dim=256, stoch_dim=32, act_dim=2, discrete=False).to(device)
    action, log_prob, entropy = actor_c.get_action(h, z)
    print(f"Continuous action: {action.shape}, log_prob: {log_prob.shape}")

    print(f"Discrete params:   {sum(p.numel() for p in actor_d.parameters()):,}")
    print(f"Continuous params:  {sum(p.numel() for p in actor_c.parameters()):,}")
    print("Smoke test passed!")
