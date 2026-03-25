"""
CNN encoder: (B, 4, 64, 64) -> (B, 512) latent vector.
4 conv layers with ELU activation, following DreamerV3 style.
"""

import torch
import torch.nn as nn
from einops import rearrange


class ConvEncoder(nn.Module):
    """Encodes pixel observations to a flat latent vector.

    Architecture:
        (4, 64, 64) -> Conv2d(32) -> Conv2d(64) -> Conv2d(128) -> Conv2d(256) -> flatten
        Each conv: kernel=4, stride=2, padding=1 -> halves spatial dims.
        64 -> 32 -> 16 -> 8 -> 4, so output = 256 * 4 * 4 = 4096 -> linear -> latent_dim
    """

    def __init__(self, in_channels: int = 4, latent_dim: int = 512, depth: int = 32):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, depth, 4, stride=2, padding=1),  # -> (32, 32, 32)
            nn.ELU(),
            nn.Conv2d(depth, depth * 2, 4, stride=2, padding=1),    # -> (64, 16, 16)
            nn.ELU(),
            nn.Conv2d(depth * 2, depth * 4, 4, stride=2, padding=1),  # -> (128, 8, 8)
            nn.ELU(),
            nn.Conv2d(depth * 4, depth * 8, 4, stride=2, padding=1),  # -> (256, 4, 4)
            nn.ELU(),
        )
        self.fc = nn.Linear(depth * 8 * 4 * 4, latent_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, C, H, W) or (B, T, C, H, W) pixel observations

        Returns:
            (B, latent_dim) or (B, T, latent_dim) latent vectors
        """
        has_time = obs.dim() == 5
        if has_time:
            B, T = obs.shape[:2]
            obs = rearrange(obs, "b t c h w -> (b t) c h w")

        x = self.convs(obs)
        x = rearrange(x, "b c h w -> b (c h w)")
        x = self.fc(x)

        if has_time:
            x = rearrange(x, "(b t) d -> b t d", b=B, t=T)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = ConvEncoder(in_channels=4, latent_dim=512).to(device)

    # Test single batch
    x = torch.randn(8, 4, 64, 64, device=device)
    out = enc(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")  # (8, 512)

    # Test with time dimension (for sequence training)
    x_seq = torch.randn(4, 20, 4, 64, 64, device=device)
    out_seq = enc(x_seq)
    print(f"Seq input:  {x_seq.shape}")
    print(f"Seq output: {out_seq.shape}")  # (4, 20, 512)

    params = sum(p.numel() for p in enc.parameters())
    print(f"Parameters: {params:,}")
    print("Smoke test passed!")
