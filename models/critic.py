"""
Critic MLP: (h, z) -> value estimate.

Supports two modes:
  - scalar: single output, trained with MSE (v6-v9)
  - categorical: softmax over discrete bins with two-hot encoding,
    trained with cross-entropy (v10+, DreamerV3-style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def twohot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Encode scalar values as two-hot vectors over discrete bins.

    Each value is represented as a weighted combination of the two nearest bins.

    Args:
        x:    (...,) scalar values
        bins: (N,) sorted bin centers

    Returns:
        (..., N) two-hot encoded vectors (sum to 1.0 per sample)
    """
    x = x.unsqueeze(-1)  # (..., 1)
    # Clamp to bin range
    x = x.clamp(bins[0], bins[-1])
    # Find the bin index just below x
    below = (x >= bins).sum(dim=-1) - 1  # (...,)
    below = below.clamp(0, len(bins) - 2)
    above = below + 1

    # Interpolation weight: how far x is between bins[below] and bins[above]
    below_val = bins[below]
    above_val = bins[above]
    weight_above = (x.squeeze(-1) - below_val) / (above_val - below_val).clamp(min=1e-8)
    weight_above = weight_above.clamp(0, 1)
    weight_below = 1.0 - weight_above

    # Build two-hot vector
    shape = x.shape[:-1]
    twohot = torch.zeros(*shape, len(bins), device=x.device)
    twohot.scatter_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
    twohot.scatter_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
    return twohot


def categorical_expected_value(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Compute expected value from categorical distribution over bins.

    Args:
        logits: (..., N) raw logits
        bins:   (N,) bin centers

    Returns:
        (...,) expected values
    """
    probs = F.softmax(logits, dim=-1)
    return (probs * bins).sum(dim=-1)


class Critic(nn.Module):
    def __init__(self, hidden_dim: int = 256, stoch_dim: int = 32, units: int = 256,
                 num_bins: int = 0, bin_low: float = -3.0, bin_high: float = 3.0):
        """
        Args:
            num_bins: If >0, use categorical mode with this many bins.
                      If 0, use scalar mode (backwards compatible).
        """
        super().__init__()
        self.num_bins = num_bins
        self.categorical = num_bins > 0

        out_dim = num_bins if self.categorical else 1

        self.net = nn.Sequential(
            nn.Linear(hidden_dim + stoch_dim, units),
            nn.ELU(),
            nn.Linear(units, units),
            nn.ELU(),
            nn.Linear(units, out_dim),
        )

        if self.categorical:
            bins = torch.linspace(bin_low, bin_high, num_bins)
            self.register_buffer("bins", bins)

    def forward(self, h: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            h: (..., hidden_dim)
            z: (..., stoch_dim) or None for JEPA mode
        Returns:
            Scalar mode:      (...,) value estimate
            Categorical mode: (...,) expected value (scalar)
        """
        x = self.net(torch.cat([h, z], dim=-1) if z is not None else h)
        if self.categorical:
            return categorical_expected_value(x, self.bins)
        return x.squeeze(-1)

    def forward_logits(self, h: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        """Return raw logits (categorical mode only, for loss computation).

        Args:
            h: (..., hidden_dim)
            z: (..., stoch_dim) or None
        Returns:
            (..., num_bins) raw logits
        """
        return self.net(torch.cat([h, z], dim=-1) if z is not None else h)

    def compute_loss(self, h: torch.Tensor, targets: torch.Tensor,
                     z: torch.Tensor = None) -> torch.Tensor:
        """Compute critic loss.

        Scalar mode:      MSE loss
        Categorical mode: Cross-entropy against two-hot encoded targets

        Args:
            h:       (..., hidden_dim) embeddings
            targets: (...,) scalar target values (e.g. lambda-returns)
            z:       (..., stoch_dim) or None
        Returns:
            scalar loss
        """
        if self.categorical:
            logits = self.forward_logits(h, z)  # (..., num_bins)
            twohot = twohot_encode(targets, self.bins)  # (..., num_bins)
            # Cross-entropy: -sum(target * log_softmax(logits))
            log_probs = F.log_softmax(logits, dim=-1)
            return -(twohot * log_probs).sum(dim=-1).mean()
        else:
            values = self.forward(h, z)
            return F.mse_loss(values, targets)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test scalar mode (backwards compatible)
    model_s = Critic(hidden_dim=256, stoch_dim=32).to(device)
    h = torch.randn(4, 15, 256, device=device)
    z = torch.randn(4, 15, 32, device=device)
    value = model_s(h, z)
    print(f"Scalar mode: {value.shape}")  # (4, 15)

    # Test categorical mode
    model_c = Critic(hidden_dim=256, stoch_dim=0, num_bins=128,
                     bin_low=-3.0, bin_high=3.0).to(device)
    h2 = torch.randn(4, 15, 256, device=device)
    value_c = model_c(h2)
    print(f"Categorical mode: {value_c.shape}")  # (4, 15)
    print(f"Value range: [{value_c.min().item():.2f}, {value_c.max().item():.2f}]")

    # Test two-hot encoding
    bins = model_c.bins
    targets = torch.tensor([-1.5, 0.0, 1.5, 2.9], device=device)
    twohot = twohot_encode(targets, bins)
    print(f"Two-hot shape: {twohot.shape}, sum: {twohot.sum(dim=-1)}")

    # Test loss computation
    targets_2d = torch.randn(4, 15, device=device).clamp(-3, 3)
    loss_c = model_c.compute_loss(h2, targets_2d)
    print(f"Categorical loss: {loss_c.item():.4f}")

    loss_s = model_s.compute_loss(h, targets_2d, z)
    print(f"Scalar loss: {loss_s.item():.4f}")

    # Verify gradient flow
    loss_c.backward()
    grad_sum = sum(p.grad.abs().sum().item() for p in model_c.parameters() if p.grad is not None)
    print(f"Categorical grad sum: {grad_sum:.4f}")

    print(f"Scalar params:      {sum(p.numel() for p in model_s.parameters()):,}")
    print(f"Categorical params: {sum(p.numel() for p in model_c.parameters()):,}")
    print("Smoke test passed!")
