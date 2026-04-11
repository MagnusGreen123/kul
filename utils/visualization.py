"""
Visualization utilities for Dreamer world model.
Decode latent rollouts to pixels and save as gif/png.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def latent_to_images(decode_fn, h_seq, z_seq, denormalize: bool = True):
    """Decode latent states to pixel images.

    Args:
        decode_fn: callable (h, z) -> reconstructed obs, or a ConvDecoder
        h_seq:     (B, T, hidden_dim) or (T, hidden_dim)
        z_seq:     (B, T, stoch_dim) or (T, stoch_dim)
        denormalize: if True, map [-0.5, 0.5] -> [0, 255]

    Returns:
        numpy array (B, T, H, W) or (T, H, W) uint8 grayscale images
    """
    single = h_seq.dim() == 2
    if single:
        h_seq = h_seq.unsqueeze(0)
        z_seq = z_seq.unsqueeze(0)

    with torch.no_grad():
        if callable(getattr(decode_fn, 'forward', None)):
            # Raw decoder (legacy) — concat h and z directly
            features = torch.cat([h_seq, z_seq], dim=-1)
            recon = decode_fn(features)
        else:
            # decode_fn is a callable (h, z) -> recon
            recon = decode_fn(h_seq, z_seq)

    # Take first channel (grayscale) or mean across channels
    imgs = recon[:, :, 0].cpu().numpy()  # (B, T, H, W)

    if denormalize:
        imgs = np.clip((imgs + 0.5) * 255, 0, 255).astype(np.uint8)

    if single:
        imgs = imgs[0]

    return imgs


def save_reconstruction_grid(obs, recon, save_path: str, n_frames: int = 8,
                              start_t: int | None = None):
    """Save a grid comparing real observations with reconstructions.

    Args:
        obs:   (T, C, H, W) or (B, T, C, H, W) real observations
        recon: same shape, reconstructed observations
        save_path: output file path
        n_frames: number of frames to show
        start_t: first frame index to show. If None, centers the window so
            we skip the post-reset dead zone in Atari envs (Pong: ~15 steps
            with no ball on screen).
    """
    if obs.dim() == 5:
        obs = obs[0]
        recon = recon[0]

    T = obs.shape[0]
    if start_t is None:
        start_t = max(0, (T - n_frames) // 2 + (T - n_frames) // 4)
    end_t = min(T, start_t + n_frames)
    start_t = max(0, end_t - n_frames)

    obs = obs[start_t:end_t, 0].cpu().numpy()
    recon = recon[start_t:end_t, 0].cpu().numpy()
    n_frames = obs.shape[0]

    obs = np.clip((obs + 0.5) * 255, 0, 255).astype(np.uint8)
    recon = np.clip((recon + 0.5) * 255, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(2, n_frames, figsize=(2 * n_frames, 4))
    for i in range(n_frames):
        axes[0, i].imshow(obs[i], cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(f"t={start_t + i}")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i], cmap="gray", vmin=0, vmax=255)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Real", fontsize=12)
    axes[1, 0].set_ylabel("Recon", fontsize=12)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close()


def save_imagination_gif(decoder, h_seq, z_seq, save_path: str, fps: int = 10):
    """Save imagined rollout as gif.

    Args:
        decoder: ConvDecoder
        h_seq:   (T, hidden_dim) or (B, T, hidden_dim) - uses first batch element
        z_seq:   (T, stoch_dim) or (B, T, stoch_dim)
        save_path: output .gif path
        fps: frames per second
    """
    imgs = latent_to_images(decoder, h_seq, z_seq)  # (T, H, W) or (B, T, H, W)
    if imgs.ndim == 4:
        imgs = imgs[0]  # take first batch element

    try:
        from PIL import Image
        frames = [Image.fromarray(img, mode="L") for img in imgs]
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            save_path, save_all=True, append_images=frames[1:],
            duration=1000 // fps, loop=0,
        )
    except ImportError:
        # Fallback: save as png grid
        n = len(imgs)
        fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
        if n == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
        plt.tight_layout()
        png_path = save_path.replace(".gif", ".png")
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(png_path, dpi=100)
        plt.close()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from models.decoder import ConvDecoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dec = ConvDecoder(latent_dim=288, out_channels=4).to(device)

    h = torch.randn(1, 10, 256, device=device)
    z = torch.randn(1, 10, 32, device=device)

    imgs = latent_to_images(dec, h, z)
    print(f"Images shape: {imgs.shape}, dtype: {imgs.dtype}, range: [{imgs.min()}, {imgs.max()}]")

    save_imagination_gif(dec, h, z, "runs/test_imagination.gif")
    print("Saved test gif/png to runs/")

    # Test reconstruction grid
    fake_obs = torch.randn(1, 10, 4, 64, 64, device=device)
    fake_recon = torch.randn(1, 10, 4, 64, 64, device=device)
    save_reconstruction_grid(fake_obs, fake_recon, "runs/test_recon.png", n_frames=8)
    print("Saved test reconstruction grid to runs/")
    print("Smoke test passed!")
