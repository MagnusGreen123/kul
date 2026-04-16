"""
Decoder probe for JEPA embeddings.

Freezes the JEPA encoder and trains a lightweight ConvDecoder on top.
If the decoder can reconstruct ball/paddle/bricks, the JEPA embedding
captures the relevant visual features. If not, the representation is
missing critical game state.

Usage:
    python probe_jepa.py --checkpoint checkpoints/step_4008.pt \
                         --config configs/pong_jepa.yaml \
                         --steps 500 --save-dir runs/probe
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.wrappers import make_atari_env
from models.decoder import ConvDecoder
from training.jepa_world_model import JEPAWorldModel


def collect_observations(env_name, n_frames, n_obs=2048, seed=42):
    """Collect real observations by playing random actions."""
    env = make_atari_env(env_name, n_frames=n_frames, seed=seed)
    obs_list = []
    obs, _ = env.reset()
    for _ in range(n_obs):
        obs_list.append(obs)
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.stack(obs_list, axis=0)  # (N, C, H, W)


def save_comparison_grid(real, recon, save_path, n_frames=10, title=None):
    """Save top=real, bottom=recon comparison grid."""
    real = real[:n_frames, 0].cpu().numpy()
    recon = recon[:n_frames, 0].detach().cpu().numpy()
    real = np.clip((real + 0.5) * 255, 0, 255).astype(np.uint8)
    recon = np.clip((recon + 0.5) * 255, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(2, n_frames, figsize=(2 * n_frames, 4))
    for i in range(n_frames):
        axes[0, i].imshow(real[i], cmap="gray", vmin=0, vmax=255)
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i], cmap="gray", vmin=0, vmax=255)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Real", fontsize=12)
    axes[1, 0].set_ylabel("Probe", fontsize=12)
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Decoder probe for JEPA embeddings")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--steps", type=int, default=500,
                        help="Decoder training steps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-obs", type=int, default=4096,
                        help="Observations to collect for training")
    parser.add_argument("--save-dir", type=str, default="runs/probe")
    parser.add_argument("--decoder-depth", type=int, default=32,
                        help="ConvDecoder channel depth (smaller = lighter probe)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    embed_dim = cfg["embed_dim"]
    obs_channels = cfg.get("obs_channels", 4)
    cnn_depth = cfg.get("cnn_depth", 48)

    # ── Load JEPA world model (encoder only) ──
    wm = JEPAWorldModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    wm.load_state_dict(ckpt["world_model"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    print(f"Loaded JEPA encoder from {args.checkpoint}")
    print(f"  embed_dim={embed_dim}, cnn_depth={cnn_depth}")

    # ── Collect observations ──
    print(f"Collecting {args.n_obs} observations from {cfg['env']}...")
    obs_np = collect_observations(
        cfg["env"], n_frames=obs_channels, n_obs=args.n_obs
    )
    obs_all = torch.tensor(obs_np, dtype=torch.float32, device=device)
    print(f"  obs shape: {obs_all.shape}, range: [{obs_all.min():.2f}, {obs_all.max():.2f}]")

    # ── Check embedding stats ──
    with torch.no_grad():
        sample_emb = wm.encode(obs_all[:256])
        print(f"\nEmbedding stats (first 256 obs):")
        print(f"  mean: {sample_emb.mean():.4f}")
        print(f"  std:  {sample_emb.std():.4f}")
        print(f"  min:  {sample_emb.min():.4f}")
        print(f"  max:  {sample_emb.max():.4f}")
        nan_count = torch.isnan(sample_emb).sum().item()
        if nan_count > 0:
            print(f"  WARNING: {nan_count} NaN values in embeddings!")
            return

    # ── Decoder probe ──
    decoder = ConvDecoder(
        latent_dim=embed_dim, out_channels=obs_channels, depth=args.decoder_depth
    ).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nDecoder probe: {dec_params:,} params (depth={args.decoder_depth})")
    print(f"Training for {args.steps} steps...")

    # ── Train decoder ──
    N = obs_all.shape[0]
    losses = []
    for step in range(args.steps):
        idx = torch.randint(0, N, (args.batch_size,))
        obs_batch = obs_all[idx]

        with torch.no_grad():
            emb = wm.encode(obs_batch)

        recon = decoder(emb)
        loss = F.mse_loss(recon, obs_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 100 == 0:
            avg = np.mean(losses[-100:])
            print(f"  step {step+1:4d} | recon_loss: {avg:.6f}")

    # ── Save results ──
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE Recon Loss")
    ax.set_title("Decoder Probe Training")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(save_dir / "probe_loss.png", dpi=150)
    plt.close()
    print(f"Saved: {save_dir / 'probe_loss.png'}")

    # Reconstruction comparison — random frames
    with torch.no_grad():
        idx = torch.randint(0, N, (10,))
        obs_sample = obs_all[idx]
        emb_sample = wm.encode(obs_sample)
        recon_sample = decoder(emb_sample)

    save_comparison_grid(
        obs_sample, recon_sample,
        save_dir / "probe_random.png",
        n_frames=10,
        title=f"Decoder Probe (step {args.steps}, loss={np.mean(losses[-50:]):.6f})",
    )

    # Reconstruction comparison — find "interesting" frames (high reward moments
    # are hard to isolate with random play, so just pick evenly spaced)
    with torch.no_grad():
        idx = torch.linspace(0, N - 1, 10).long()
        obs_sample = obs_all[idx]
        emb_sample = wm.encode(obs_sample)
        recon_sample = decoder(emb_sample)

    save_comparison_grid(
        obs_sample, recon_sample,
        save_dir / "probe_evenly_spaced.png",
        n_frames=10,
        title="Decoder Probe — evenly spaced frames",
    )

    # ── Embedding similarity analysis ──
    # Check if visually similar frames have similar embeddings
    print("\nEmbedding similarity analysis:")
    with torch.no_grad():
        all_emb = []
        for i in range(0, N, 256):
            batch = obs_all[i:i+256]
            all_emb.append(wm.encode(batch))
        all_emb = torch.cat(all_emb, dim=0)  # (N, D)

        # Consecutive frames should be similar (small dt)
        consec_dist = (all_emb[1:] - all_emb[:-1]).norm(dim=-1)
        print(f"  Consecutive frame distance: {consec_dist.mean():.4f} +/- {consec_dist.std():.4f}")

        # Random pairs should be more distant
        perm = torch.randperm(N)
        random_dist = (all_emb - all_emb[perm]).norm(dim=-1)
        print(f"  Random pair distance:       {random_dist.mean():.4f} +/- {random_dist.std():.4f}")

        ratio = random_dist.mean() / consec_dist.mean()
        print(f"  Ratio (random/consecutive): {ratio:.2f}x")
        if ratio < 1.5:
            print("  WARNING: Embeddings barely distinguish nearby vs distant frames!")
            print("  This suggests the encoder may not capture temporal dynamics well.")
        else:
            print("  OK: Embeddings reflect temporal distance.")

    # Save decoder checkpoint for further analysis
    torch.save(decoder.state_dict(), save_dir / "probe_decoder.pt")
    print(f"\nDecoder saved to {save_dir / 'probe_decoder.pt'}")
    print("Done. Check the images in", save_dir)


if __name__ == "__main__":
    main()
