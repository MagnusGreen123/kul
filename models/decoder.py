"""
CNN decoder (symmetric to encoder): latent vector -> reconstructed observation.
Used to verify world model learns meaningful representations.
"""

import torch
import torch.nn as nn
from einops import rearrange


class ConvDecoder(nn.Module):
    """Decodes latent vector back to pixel observations.

    Architecture (symmetric to ConvEncoder):
        latent_dim -> linear -> (256, 4, 4) -> ConvT(128) -> ConvT(64) -> ConvT(32) -> ConvT(C)
        Each ConvTranspose2d: kernel=4, stride=2, padding=1 -> doubles spatial dims.
        4 -> 8 -> 16 -> 32 -> 64
    """

    def __init__(self, latent_dim: int = 512, out_channels: int = 4, depth: int = 32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, depth * 8 * 4 * 4, bias=False)
        self.depth = depth
        self.out_channels = out_channels

        # Shared trunk: 4x4 -> 32x32 feature map at `depth` channels.
        self.trunk = nn.Sequential(
            nn.ConvTranspose2d(depth * 8, depth * 4, 4, stride=2, padding=1),  # 4 -> 8
            nn.ELU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, stride=2, padding=1),  # 8 -> 16
            nn.ELU(),
            nn.ConvTranspose2d(depth * 2, depth, 4, stride=2, padding=1),      # 16 -> 32
            nn.ELU(),
        )

        # v23: two heads for per-pixel Gaussian NLL.
        # mean_head → reconstructed pixel mean. std_head → log_std per pixel,
        # so the decoder can be confident on static background (small σ → very
        # negative NLL) and less confident on moving ball/paddle pixels. This
        # gives ball reconstruction a real gradient signal: correctly predicting
        # a ball pixel at σ=0.01 saves ~-3.7 NLL vs ~0.7 for "give up" (σ=err).
        self.mean_head = nn.ConvTranspose2d(depth, out_channels, 4, stride=2, padding=1)
        self.std_head = nn.ConvTranspose2d(depth, out_channels, 4, stride=2, padding=1)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent: (B, latent_dim) or (B, T, latent_dim)

        Returns:
            (mean, log_std) — each of shape (B, C, H, W) or (B, T, C, H, W).
            log_std is clamped to [-5, 2].
        """
        has_time = latent.dim() == 3
        if has_time:
            B, T = latent.shape[:2]
            latent = rearrange(latent, "b t d -> (b t) d")

        x = self.fc(latent)
        x = rearrange(x, "b (c h w) -> b c h w", c=self.depth * 8, h=4, w=4)
        x = self.trunk(x)
        mean = self.mean_head(x)
        log_std = torch.clamp(self.std_head(x), min=-5.0, max=2.0)

        if has_time:
            mean = rearrange(mean, "(b t) c h w -> b t c h w", b=B, t=T)
            log_std = rearrange(log_std, "(b t) c h w -> b t c h w", b=B, t=T)

        return mean, log_std


if __name__ == "__main__":
    from encoder import ConvEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = ConvEncoder(in_channels=4, latent_dim=512).to(device)
    dec = ConvDecoder(latent_dim=512, out_channels=4).to(device)

    # Test roundtrip
    x = torch.randn(8, 4, 64, 64, device=device)
    z = enc(x)
    mean, log_std = dec(z)
    print(f"Input:   {x.shape}")
    print(f"Latent:  {z.shape}")
    print(f"Mean:    {mean.shape}")
    print(f"LogStd:  {log_std.shape}")
    assert x.shape == mean.shape == log_std.shape

    # Test with time dim
    x_seq = torch.randn(4, 20, 4, 64, 64, device=device)
    z_seq = enc(x_seq)
    mean_seq, log_std_seq = dec(z_seq)
    print(f"Seq input:   {x_seq.shape}")
    print(f"Seq mean:    {mean_seq.shape}")
    print(f"Seq log_std: {log_std_seq.shape}")
    assert x_seq.shape == mean_seq.shape == log_std_seq.shape

    params = sum(p.numel() for p in dec.parameters())
    print(f"Decoder params: {params:,}")
    print("Smoke test passed!")
