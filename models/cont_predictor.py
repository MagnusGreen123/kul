"""
Continuation predictor MLP: (h, z) -> P(episode continues).
Used during imagination to discount through terminal states.
"""

import torch
import torch.nn as nn


class ContPredictor(nn.Module):
    def __init__(self, hidden_dim: int = 256, stoch_dim: int = 32, units: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + stoch_dim, units),
            nn.ELU(),
            nn.Linear(units, units),
            nn.ELU(),
            nn.Linear(units, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            h: (..., hidden_dim)
            z: (..., stoch_dim) or None for JEPA mode
        Returns:
            (...,) continuation logit (apply sigmoid for probability)
        """
        return self.net(torch.cat([h, z], dim=-1) if z is not None else h).squeeze(-1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContPredictor(hidden_dim=256, stoch_dim=32).to(device)

    h = torch.randn(4, 20, 256, device=device)
    z = torch.randn(4, 20, 32, device=device)
    logit = model(h, z)
    prob = torch.sigmoid(logit)
    print(f"Input h: {h.shape}, z: {z.shape}")
    print(f"Cont logit: {logit.shape}, prob range: [{prob.min():.3f}, {prob.max():.3f}]")
    print(f"Params:  {sum(p.numel() for p in model.parameters()):,}")
    print("Smoke test passed!")
