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
        # bias=False prevents the decoder from encoding a "mean image" in the
        # bias alone (6144-dim bias = 384ch × 4×4 spatial → upsampled to 64×64).
        # Without bias, the decoder MUST use its (h, z) input to produce any
        # spatially structured output. Deconv biases are per-channel only.
        self.fc = nn.Linear(latent_dim, depth * 8 * 4 * 4, bias=False)
        self.depth = depth

        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(depth * 8, depth * 4, 4, stride=2, padding=1),  # -> (128, 8, 8)
            nn.ELU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ELU(),
            nn.ConvTranspose2d(depth * 2, depth, 4, stride=2, padding=1),      # -> (32, 32, 32)
            nn.ELU(),
            nn.ConvTranspose2d(depth, out_channels, 4, stride=2, padding=1),   # -> (C, 64, 64)
            # v18: no output activation — tanh created gradient dead zones at
            # ±0.5 (black/white pixels), preventing decoder from modifying
            # committed pixel values. Raw output with L1 loss is stable.
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_dim) or (B, T, latent_dim)

        Returns:
            (B, C, H, W) or (B, T, C, H, W) reconstructed observations
        """
        has_time = latent.dim() == 3
        if has_time:
            B, T = latent.shape[:2]
            latent = rearrange(latent, "b t d -> (b t) d")

        x = self.fc(latent)
        x = rearrange(x, "b (c h w) -> b c h w", c=self.depth * 8, h=4, w=4)
        x = self.deconvs(x)

        if has_time:
            x = rearrange(x, "(b t) c h w -> b t c h w", b=B, t=T)

        return x


if __name__ == "__main__":
    from encoder import ConvEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = ConvEncoder(in_channels=4, latent_dim=512).to(device)
    dec = ConvDecoder(latent_dim=512, out_channels=4).to(device)

    # Test roundtrip
    x = torch.randn(8, 4, 64, 64, device=device)
    z = enc(x)
    recon = dec(z)
    print(f"Input:   {x.shape}")
    print(f"Latent:  {z.shape}")
    print(f"Recon:   {recon.shape}")
    assert x.shape == recon.shape, f"Shape mismatch: {x.shape} vs {recon.shape}"

    # Test with time dim
    x_seq = torch.randn(4, 20, 4, 64, 64, device=device)
    z_seq = enc(x_seq)
    recon_seq = dec(z_seq)
    print(f"Seq input:  {x_seq.shape}")
    print(f"Seq recon:  {recon_seq.shape}")
    assert x_seq.shape == recon_seq.shape

    params = sum(p.numel() for p in dec.parameters())
    print(f"Decoder params: {params:,}")
    print("Smoke test passed!")
