"""
Recurrent State Space Model (RSSM) with categorical latents (DreamerV3).

State = (h, z) where:
    h: deterministic recurrent state (GRU hidden), shape (B, hidden_dim)
    z: stochastic categorical latent (one-hot), shape (B, stoch_dim * n_classes)

Two modes:
    observe: uses posterior q(z|h, obs_embed) — for training with real observations
    imagine: uses prior p(z|h) — for planning/imagination rollouts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,      # encoder output dim
        stoch_dim: int = 32,       # number of categorical distributions
        n_classes: int = 32,       # classes per categorical
        hidden_dim: int = 256,     # GRU hidden h dimension
        act_dim: int = 4,          # action space size
        unimix: float = 0.01,     # uniform mixing ratio (prevents categorical collapse)
    ):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.unimix = unimix
        self.stoch_feat_dim = stoch_dim * n_classes  # flat z dimension (e.g. 32*32=1024)

        # h_{t} = LayerNorm(GRU(h_{t-1}, [z_{t-1}, a_{t-1}]))
        # LayerNorm after GRU is critical (DreamerV3): without it, h grows
        # to arbitrary magnitude and drowns out z in the decoder input.
        self.gru_input = nn.Linear(self.stoch_feat_dim + act_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.gru_norm = nn.LayerNorm(hidden_dim)

        # Prior p(z_t | h_t): predicts z from h alone (imagination)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * n_classes),
        )

        # Posterior q(z_t | h_t, embed_t): uses observation (training)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * n_classes),
        )

    def initial_state(self, batch_size: int, device: torch.device):
        """Return zero-initialized (h, z)."""
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.stoch_feat_dim, device=device),
        )

    def _logits_to_probs(self, logits):
        """Reshape logits to (*, stoch_dim, n_classes), softmax, apply unimix.

        Returns:
            probs: (*, stoch_dim, n_classes) probability distributions
        """
        logits = logits.unflatten(-1, (self.stoch_dim, self.n_classes))
        probs = F.softmax(logits, dim=-1)
        if self.unimix > 0:
            probs = (1 - self.unimix) * probs + self.unimix / self.n_classes
        return probs

    def _sample_straight_through(self, probs):
        """Sample from categorical with straight-through gradient.

        Forward: hard one-hot sample (discrete).
        Backward: gradient flows through soft probabilities.

        Args:
            probs: (*, stoch_dim, n_classes)
        Returns:
            z: (*, stoch_feat_dim) flattened one-hot
        """
        indices = torch.distributions.Categorical(probs=probs).sample()
        hard = F.one_hot(indices, self.n_classes).float()
        # Straight-through: hard in forward, gradient through probs in backward
        z = hard - probs.detach() + probs
        return z.flatten(-2)  # (*, stoch_dim * n_classes)

    def observe_step(self, prev_h, prev_z, prev_action, obs_embed):
        """Single step with observation (training).

        Returns: h, z, prior_logits, posterior_logits
            prior_logits:     (B, stoch_dim, n_classes)
            posterior_logits:  (B, stoch_dim, n_classes)
        """
        # Deterministic step
        gru_in = self.gru_input(torch.cat([prev_z, prev_action], dim=-1))
        gru_in = F.elu(gru_in)
        h = self.gru_norm(self.gru(gru_in, prev_h))

        # Prior p(z|h)
        prior_raw = self.prior_net(h)
        prior_logits = prior_raw.unflatten(-1, (self.stoch_dim, self.n_classes))

        # Posterior q(z|h, obs)
        post_raw = self.posterior_net(torch.cat([h, obs_embed], dim=-1))
        post_logits = post_raw.unflatten(-1, (self.stoch_dim, self.n_classes))

        # Sample z from posterior
        post_probs = self._logits_to_probs(post_raw)
        z = self._sample_straight_through(post_probs)

        return h, z, prior_logits, post_logits

    def imagine_step(self, prev_h, prev_z, prev_action):
        """Single step without observation (imagination).

        Returns: h, z, prior_logits
        """
        gru_in = self.gru_input(torch.cat([prev_z, prev_action], dim=-1))
        gru_in = F.elu(gru_in)
        h = self.gru_norm(self.gru(gru_in, prev_h))

        prior_raw = self.prior_net(h)
        prior_logits = prior_raw.unflatten(-1, (self.stoch_dim, self.n_classes))

        prior_probs = self._logits_to_probs(prior_raw)
        z = self._sample_straight_through(prior_probs)

        return h, z, prior_logits

    def observe_sequence(self, obs_embeds, actions, initial_state=None):
        """Process a sequence of observations (for world model training).

        Args:
            obs_embeds: (B, T, embed_dim)
            actions:    (B, T, act_dim) — one-hot or continuous
            initial_state: optional (h, z) tuple

        Returns:
            h_seq:          (B, T, hidden_dim)
            z_seq:          (B, T, stoch_feat_dim)
            prior_logits:   (B, T, stoch_dim, n_classes)
            post_logits:    (B, T, stoch_dim, n_classes)
        """
        B, T, _ = obs_embeds.shape
        device = obs_embeds.device

        if initial_state is None:
            h, z = self.initial_state(B, device)
        else:
            h, z = initial_state

        # Shift actions: at t=0 prev_action is zero (no action led to the
        # first observation), at t>0 prev_action is actions[:, t-1].
        zero_action = torch.zeros_like(actions[:, :1])
        prev_actions = torch.cat([zero_action, actions[:, :-1]], dim=1)

        h_list, z_list = [], []
        prior_logits_list, post_logits_list = [], []

        for t in range(T):
            h, z, prior_logits, post_logits = self.observe_step(
                h, z, prev_actions[:, t], obs_embeds[:, t]
            )
            h_list.append(h)
            z_list.append(z)
            prior_logits_list.append(prior_logits)
            post_logits_list.append(post_logits)

        h_seq = torch.stack(h_list, dim=1)
        z_seq = torch.stack(z_list, dim=1)
        prior_logits_seq = torch.stack(prior_logits_list, dim=1)
        post_logits_seq = torch.stack(post_logits_list, dim=1)

        return h_seq, z_seq, prior_logits_seq, post_logits_seq

    def imagine_sequence(self, actor, initial_h, initial_z, horizon: int):
        """Roll out in imagination using an actor policy.

        Args:
            actor:     callable (h, z) -> action
            initial_h: (B, hidden_dim)
            initial_z: (B, stoch_feat_dim)
            horizon:   number of steps

        Returns:
            h_seq: (B, H, hidden_dim)
            z_seq: (B, H, stoch_feat_dim)
        """
        h, z = initial_h, initial_z
        h_list, z_list = [], []

        for _ in range(horizon):
            action = actor(h, z)
            h, z, _ = self.imagine_step(h, z, action)
            h_list.append(h)
            z_list.append(z)

        return torch.stack(h_list, dim=1), torch.stack(z_list, dim=1)


def kl_loss(post_logits, prior_logits, free_bits: float = 1.0,
            balance: float = 0.8, unimix: float = 0.01):
    """Categorical KL divergence with free bits and KL balancing (DreamerV3).

    Args:
        post_logits:  (B, T, stoch_dim, n_classes)
        prior_logits: (B, T, stoch_dim, n_classes)
        free_bits:    minimum total KL (summed over categoricals)
        balance:      weight on prior term (0.8 = mostly train prior toward posterior)
        unimix:       uniform mixing ratio applied to probs

    Returns:
        (scalar KL loss, scalar raw KL before free_bits clipping)
    """
    def _probs(logits):
        probs = F.softmax(logits, dim=-1)
        if unimix > 0:
            probs = (1 - unimix) * probs + unimix / logits.shape[-1]
        return probs

    def _cat_kl(p, q):
        """KL(p || q), summed over classes and categoricals. Returns (B, T)."""
        return (p * (p.clamp(min=1e-8).log() - q.clamp(min=1e-8).log())).sum(-1).sum(-1)

    post_probs = _probs(post_logits)
    prior_probs = _probs(prior_logits)

    # Raw KL for logging (fully detached — no gradient)
    raw_kl = _cat_kl(post_probs.detach(), prior_probs.detach()).mean()

    # Train prior toward posterior (posterior is fixed target)
    kl_to_prior = _cat_kl(post_probs.detach(), prior_probs).clamp(min=free_bits).mean()

    # Train posterior toward prior (prior is fixed target)
    kl_to_post = _cat_kl(post_probs, prior_probs.detach()).clamp(min=free_bits).mean()

    return balance * kl_to_prior + (1 - balance) * kl_to_post, raw_kl


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, T, embed_dim, act_dim = 4, 20, 512, 6
    stoch_dim, n_classes = 32, 32
    rssm = RSSM(embed_dim=embed_dim, stoch_dim=stoch_dim, n_classes=n_classes,
                 hidden_dim=256, act_dim=act_dim).to(device)

    # Simulate encoded observations and one-hot actions
    obs_embeds = torch.randn(B, T, embed_dim, device=device)
    actions = torch.zeros(B, T, act_dim, device=device)
    actions.scatter_(2, torch.randint(0, act_dim, (B, T, 1), device=device), 1.0)

    # Observe sequence
    h_seq, z_seq, prior_logits, post_logits = rssm.observe_sequence(obs_embeds, actions)
    print(f"h_seq: {h_seq.shape}")   # (4, 20, 256)
    print(f"z_seq: {z_seq.shape}")   # (4, 20, 1024)
    print(f"prior_logits: {prior_logits.shape}")  # (4, 20, 32, 32)
    print(f"post_logits:  {post_logits.shape}")   # (4, 20, 32, 32)

    # KL loss
    kl, kl_raw = kl_loss(post_logits, prior_logits)
    print(f"KL loss: {kl.item():.3f}, raw KL: {kl_raw.item():.3f}")

    # Imagine sequence
    def dummy_actor(h, z):
        a = torch.zeros(h.shape[0], act_dim, device=h.device)
        a[:, 0] = 1.0
        return a

    h_imag, z_imag = rssm.imagine_sequence(dummy_actor, h_seq[:, -1], z_seq[:, -1], horizon=15)
    print(f"Imagined h: {h_imag.shape}")  # (4, 15, 256)
    print(f"Imagined z: {z_imag.shape}")  # (4, 15, 1024)

    params = sum(p.numel() for p in rssm.parameters())
    print(f"RSSM params: {params:,}")
    print("Smoke test passed!")
