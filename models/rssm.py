"""
Recurrent State Space Model (RSSM) for DreamerV3.

State = (h, z) where:
    h: deterministic recurrent state (GRU hidden)
    z: stochastic latent state

Two modes:
    observe: uses posterior q(z|h, obs_embed) — for training with real observations
    imagine: uses prior p(z|h) — for planning/imagination rollouts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from einops import rearrange


class RSSM(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,      # encoder output dim
        stoch_dim: int = 32,       # stochastic z dimension
        hidden_dim: int = 256,     # GRU hidden h dimension
        act_dim: int = 4,         # action space size
    ):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim

        # h_{t} = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
        self.gru_input = nn.Linear(stoch_dim + act_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Prior p(z_t | h_t): predicts z from h alone (imagination)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2),  # mean + log_std
        )

        # Posterior q(z_t | h_t, embed_t): uses observation (training)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * 2),  # mean + log_std
        )

    def initial_state(self, batch_size: int, device: torch.device):
        """Return zero-initialized (h, z)."""
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.stoch_dim, device=device),
        )

    def _stats(self, raw: torch.Tensor):
        """Split raw params into mean and std."""
        mean, log_std = raw.chunk(2, dim=-1)
        std = F.softplus(log_std) + 0.1  # min std for stability
        return mean, std

    def observe_step(self, prev_h, prev_z, prev_action, obs_embed):
        """Single step with observation (training).

        Returns: h, z, prior_dist, posterior_dist
        """
        # Deterministic step
        gru_in = self.gru_input(torch.cat([prev_z, prev_action], dim=-1))
        gru_in = F.elu(gru_in)
        h = self.gru(gru_in, prev_h)

        # Prior
        prior_mean, prior_std = self._stats(self.prior_net(h))
        prior = Normal(prior_mean, prior_std)

        # Posterior (uses observation)
        post_mean, post_std = self._stats(self.posterior_net(torch.cat([h, obs_embed], dim=-1)))
        posterior = Normal(post_mean, post_std)

        # Sample z from posterior during training
        z = posterior.rsample()

        return h, z, prior, posterior

    def imagine_step(self, prev_h, prev_z, prev_action):
        """Single step without observation (imagination).

        Returns: h, z, prior_dist
        """
        gru_in = self.gru_input(torch.cat([prev_z, prev_action], dim=-1))
        gru_in = F.elu(gru_in)
        h = self.gru(gru_in, prev_h)

        prior_mean, prior_std = self._stats(self.prior_net(h))
        prior = Normal(prior_mean, prior_std)
        z = prior.rsample()

        return h, z, prior

    def observe_sequence(self, obs_embeds, actions, initial_state=None):
        """Process a sequence of observations (for world model training).

        Args:
            obs_embeds: (B, T, embed_dim)
            actions:    (B, T, act_dim) — one-hot or continuous
            initial_state: optional (h, z) tuple

        Returns:
            h_seq:      (B, T, hidden_dim)
            z_seq:      (B, T, stoch_dim)
            priors:     list of T Normal distributions
            posteriors: list of T Normal distributions
        """
        B, T, _ = obs_embeds.shape
        device = obs_embeds.device

        if initial_state is None:
            h, z = self.initial_state(B, device)
        else:
            h, z = initial_state

        h_list, z_list = [], []
        priors, posteriors = [], []

        for t in range(T):
            h, z, prior, posterior = self.observe_step(
                h, z, actions[:, t], obs_embeds[:, t]
            )
            h_list.append(h)
            z_list.append(z)
            priors.append(prior)
            posteriors.append(posterior)

        h_seq = torch.stack(h_list, dim=1)
        z_seq = torch.stack(z_list, dim=1)

        return h_seq, z_seq, priors, posteriors

    def imagine_sequence(self, actor, initial_h, initial_z, horizon: int):
        """Roll out in imagination using an actor policy.

        Args:
            actor:     callable (h, z) -> action
            initial_h: (B, hidden_dim)
            initial_z: (B, stoch_dim)
            horizon:   number of steps

        Returns:
            h_seq: (B, H, hidden_dim)
            z_seq: (B, H, stoch_dim)
        """
        h, z = initial_h, initial_z
        h_list, z_list = [], []

        for _ in range(horizon):
            action = actor(h, z)
            h, z, _ = self.imagine_step(h, z, action)
            h_list.append(h)
            z_list.append(z)

        return torch.stack(h_list, dim=1), torch.stack(z_list, dim=1)


def kl_loss(priors, posteriors, free_bits: float = 1.0, balance: float = 0.8):
    """KL divergence with free bits and KL balancing (DreamerV3).

    Args:
        priors:     list of T Normal distributions
        posteriors: list of T Normal distributions
        free_bits:  minimum KL per dimension
        balance:    weight on posterior (0.8 = mostly train prior toward posterior)

    Returns:
        scalar KL loss
    """
    kl_values = []
    for prior, post in zip(priors, posteriors):
        kl = kl_divergence(post, prior)  # (B, stoch_dim)
        kl = torch.clamp(kl, min=free_bits)
        kl_values.append(kl.sum(dim=-1))  # (B,)

    kl_tensor = torch.stack(kl_values, dim=1)  # (B, T)

    # KL balancing: mostly train prior toward posterior
    # balance=0.8 means 80% gradient to prior, 20% to posterior
    kl_balanced = (
        balance * kl_divergence(post.detach(), prior).clamp(min=free_bits).sum(-1).unsqueeze(1)
        + (1 - balance) * kl_divergence(post, prior.detach()).clamp(min=free_bits).sum(-1).unsqueeze(1)
    )

    return kl_tensor.mean()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, embed_dim, act_dim = 4, 20, 512, 4
    rssm = RSSM(embed_dim=embed_dim, stoch_dim=32, hidden_dim=256, act_dim=act_dim).to(device)

    # Simulate encoded observations and one-hot actions
    obs_embeds = torch.randn(B, T, embed_dim, device=device)
    actions = torch.zeros(B, T, act_dim, device=device)
    actions.scatter_(2, torch.randint(0, act_dim, (B, T, 1), device=device), 1.0)

    # Observe sequence
    h_seq, z_seq, priors, posteriors = rssm.observe_sequence(obs_embeds, actions)
    print(f"h_seq: {h_seq.shape}")  # (4, 20, 256)
    print(f"z_seq: {z_seq.shape}")  # (4, 20, 32)

    # KL loss
    kl = kl_loss(priors, posteriors)
    print(f"KL loss: {kl.item():.3f}")

    # Imagine sequence
    def dummy_actor(h, z):
        a = torch.zeros(h.shape[0], act_dim, device=h.device)
        a[:, 0] = 1.0
        return a

    h_imag, z_imag = rssm.imagine_sequence(dummy_actor, h_seq[:, -1], z_seq[:, -1], horizon=15)
    print(f"Imagined h: {h_imag.shape}")  # (4, 15, 256)
    print(f"Imagined z: {z_imag.shape}")  # (4, 15, 32)

    params = sum(p.numel() for p in rssm.parameters())
    print(f"RSSM params: {params:,}")
    print("Smoke test passed!")
