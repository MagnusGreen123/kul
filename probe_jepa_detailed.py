"""
Detailed JEPA embedding probe — reveals what the encoder actually captures.

Three complementary tests:
  1. FG-weighted decoder: reconstructs with fg_weight=50, same as training
  2. Linear ball locator: predicts ball (x,y) from embedding — cleanest test
  3. Per-region MSE breakdown: foreground vs background reconstruction quality

Usage:
    python probe_jepa_detailed.py --checkpoint checkpoints/step_200000.pt \
                                  --config configs/pong_jepa.yaml \
                                  --save-dir runs/probe_v6_detailed
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.wrappers import make_atari_env
from models.decoder import ConvDecoder
from training.jepa_world_model import JEPAWorldModel


def collect_observations(env_name, n_frames, n_obs=4096, seed=42):
    """Collect observations with random play."""
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
    return np.stack(obs_list, axis=0)


def compute_fg_mask(obs):
    """Foreground mask via temporal diff (same as training)."""
    # obs: (N, C, H, W)
    prev = torch.cat([obs[:1], obs[:-1]], dim=0)
    diff = (obs - prev).abs()
    fg = diff.mean(dim=1, keepdim=True)  # (N, 1, H, W)
    return fg


def find_ball_positions(obs):
    """Heuristic ball detector for Pong: small bright moving object.
    Returns (N, 2) with (row, col) normalized to [0,1], or NaN if no ball found.
    """
    N, C, H, W = obs.shape
    positions = torch.full((N, 2), float("nan"))

    for i in range(1, N):
        # temporal diff on last channel (most recent frame)
        diff = (obs[i, -1] - obs[i - 1, -1]).abs()

        # Ball is small and bright in diff — threshold
        mask = diff > 0.15
        # Exclude score area (top ~15 rows) and paddle columns (left/right edges)
        mask[:15, :] = False
        mask[:, :8] = False
        mask[:, -8:] = False

        coords = mask.nonzero(as_tuple=False)  # (K, 2) — row, col
        if len(coords) >= 1 and len(coords) <= 20:
            # Average position of the moving pixels = ball center
            center = coords.float().mean(dim=0)
            positions[i, 0] = center[0] / H  # row normalized
            positions[i, 1] = center[1] / W  # col normalized

    return positions


def save_grid(real, recon, save_path, title=None, n=10):
    """Save top=real, bottom=recon comparison grid."""
    real = real[:n, 0].cpu().numpy()
    recon = recon[:n, 0].detach().cpu().numpy()
    real = np.clip((real + 0.5) * 255, 0, 255).astype(np.uint8)
    recon = np.clip((recon + 0.5) * 255, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(real[i], cmap="gray", vmin=0, vmax=255)
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i], cmap="gray", vmin=0, vmax=255)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Real", fontsize=12)
    axes[1, 0].set_ylabel("Recon", fontsize=12)
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-obs", type=int, default=4096)
    parser.add_argument("--save-dir", type=str, default="runs/probe_detailed")
    parser.add_argument("--decoder-depth", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    embed_dim = cfg["embed_dim"]
    obs_channels = cfg.get("obs_channels", 4)
    fg_weight = cfg.get("fg_weight", 50.0)

    # Load encoder
    wm = JEPAWorldModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    wm.load_state_dict(ckpt["world_model"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    print(f"Loaded encoder from {args.checkpoint}")

    # Collect observations
    print(f"Collecting {args.n_obs} observations...")
    obs_np = collect_observations(cfg["env"], n_frames=obs_channels, n_obs=args.n_obs)
    obs_all = torch.tensor(obs_np, dtype=torch.float32, device=device)
    N = obs_all.shape[0]
    print(f"  obs: {obs_all.shape}, range [{obs_all.min():.2f}, {obs_all.max():.2f}]")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Precompute embeddings ──
    print("Computing embeddings...")
    all_emb = []
    with torch.no_grad():
        for i in range(0, N, 256):
            all_emb.append(wm.encode(obs_all[i:i+256]))
    all_emb = torch.cat(all_emb, dim=0)
    print(f"  emb: {all_emb.shape}, mean={all_emb.mean():.4f}, std={all_emb.std():.4f}")

    # ═══════════════════════════════════════════════
    # TEST 1: FG-weighted decoder probe
    # ═══════════════════════════════════════════════
    print(f"\n{'='*50}")
    print(f"TEST 1: FG-weighted decoder (fg_weight={fg_weight})")
    print(f"{'='*50}")

    fg_decoder = ConvDecoder(
        latent_dim=embed_dim, out_channels=obs_channels, depth=args.decoder_depth
    ).to(device)
    optimizer_fg = torch.optim.Adam(fg_decoder.parameters(), lr=args.lr)

    fg_mask_all = compute_fg_mask(obs_all)  # (N, 1, H, W)

    losses_fg = []
    for step in range(args.steps):
        idx = torch.randint(0, N, (args.batch_size,))
        emb = all_emb[idx]
        obs_batch = obs_all[idx]
        mask_batch = fg_mask_all[idx]

        recon = fg_decoder(emb)
        weight = 1.0 + fg_weight * mask_batch
        loss = ((obs_batch - recon) ** 2 * weight).mean()

        optimizer_fg.zero_grad()
        loss.backward()
        optimizer_fg.step()
        losses_fg.append(loss.item())

        if (step + 1) % 200 == 0:
            print(f"  step {step+1:4d} | fg_recon_loss: {np.mean(losses_fg[-200:]):.6f}")

    # Also train a plain decoder for comparison
    print(f"\n  Training plain decoder (no fg_weight) for comparison...")
    plain_decoder = ConvDecoder(
        latent_dim=embed_dim, out_channels=obs_channels, depth=args.decoder_depth
    ).to(device)
    optimizer_plain = torch.optim.Adam(plain_decoder.parameters(), lr=args.lr)

    losses_plain = []
    for step in range(args.steps):
        idx = torch.randint(0, N, (args.batch_size,))
        emb = all_emb[idx]
        obs_batch = obs_all[idx]

        recon = plain_decoder(emb)
        loss = F.mse_loss(recon, obs_batch)

        optimizer_plain.zero_grad()
        loss.backward()
        optimizer_plain.step()
        losses_plain.append(loss.item())

    # Save comparison images
    with torch.no_grad():
        # Pick frames where ball is visible (high fg_mask)
        fg_score = fg_mask_all.mean(dim=(1, 2, 3))
        _, ball_idx = fg_score.topk(20)
        # Take every other to avoid consecutive near-duplicates
        ball_idx = ball_idx[::2][:10]

        obs_sample = obs_all[ball_idx]
        emb_sample = all_emb[ball_idx]
        recon_fg = fg_decoder(emb_sample)
        recon_plain = plain_decoder(emb_sample)

    save_grid(obs_sample, recon_fg, save_dir / "fg_weighted_recon.png",
              title=f"FG-weighted decoder (fg_weight={fg_weight})")
    save_grid(obs_sample, recon_plain, save_dir / "plain_recon.png",
              title="Plain decoder (no fg_weight)")

    # 3-row comparison: real / fg-weighted / plain
    n = min(10, len(ball_idx))
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    for i in range(n):
        for row, (data, label) in enumerate([
            (obs_sample, "Real"),
            (recon_fg, "FG-weighted"),
            (recon_plain, "Plain"),
        ]):
            img = data[i, 0].detach().cpu().numpy()
            img = np.clip((img + 0.5) * 255, 0, 255).astype(np.uint8)
            axes[row, i].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[row, i].axis("off")
            if i == 0:
                axes[row, i].set_ylabel(label, fontsize=11)
    fig.suptitle("Frames with most foreground activity (ball movement)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / "fg_vs_plain_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: fg_vs_plain_comparison.png")

    # ═══════════════════════════════════════════════
    # TEST 2: Linear ball position probe
    # ═══════════════════════════════════════════════
    print(f"\n{'='*50}")
    print("TEST 2: Linear ball position probe")
    print(f"{'='*50}")

    ball_pos = find_ball_positions(obs_all.cpu())
    valid = ~torch.isnan(ball_pos[:, 0])
    n_valid = valid.sum().item()
    print(f"  Found ball in {n_valid}/{N} frames ({100*n_valid/N:.1f}%)")

    if n_valid > 100:
        # Train linear probe: embedding -> (row, col)
        ball_probe = nn.Linear(embed_dim, 2).to(device)
        optimizer_ball = torch.optim.Adam(ball_probe.parameters(), lr=1e-3)

        valid_idx = valid.nonzero(as_tuple=True)[0]
        valid_emb = all_emb[valid_idx]
        valid_pos = ball_pos[valid_idx].to(device)

        # Train/test split
        n_train = int(0.8 * len(valid_idx))
        train_emb, test_emb = valid_emb[:n_train], valid_emb[n_train:]
        train_pos, test_pos = valid_pos[:n_train], valid_pos[n_train:]

        losses_ball = []
        for step in range(2000):
            idx = torch.randint(0, n_train, (min(128, n_train),))
            pred = ball_probe(train_emb[idx])
            loss = F.mse_loss(pred, train_pos[idx])

            optimizer_ball.zero_grad()
            loss.backward()
            optimizer_ball.step()
            losses_ball.append(loss.item())

            if (step + 1) % 500 == 0:
                print(f"  step {step+1:4d} | ball_pos_mse: {np.mean(losses_ball[-500:]):.6f}")

        # Evaluate on test set
        with torch.no_grad():
            test_pred = ball_probe(test_emb)
            test_mse = F.mse_loss(test_pred, test_pos).item()
            # Error in pixels (64x64 grid)
            pixel_err = ((test_pred - test_pos) * 64).pow(2).sum(dim=1).sqrt()
            mean_px_err = pixel_err.mean().item()
            median_px_err = pixel_err.median().item()

        print(f"\n  Test MSE: {test_mse:.6f}")
        print(f"  Mean pixel error:   {mean_px_err:.1f} px (of 64)")
        print(f"  Median pixel error: {median_px_err:.1f} px (of 64)")

        if mean_px_err < 5:
            print("  GOOD: Embedding accurately encodes ball position")
        elif mean_px_err < 15:
            print("  PARTIAL: Embedding has rough ball position info")
        else:
            print("  POOR: Embedding does not reliably encode ball position")

        # Scatter plot: predicted vs actual ball position
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        tp = test_pos.cpu().numpy()
        pp = test_pred.cpu().numpy()

        for ax, dim, label in [(axes[0], 0, "Row (Y)"), (axes[1], 1, "Col (X)")]:
            ax.scatter(tp[:, dim], pp[:, dim], alpha=0.3, s=10)
            ax.plot([0, 1], [0, 1], "r--", linewidth=1)
            ax.set_xlabel(f"Actual {label}")
            ax.set_ylabel(f"Predicted {label}")
            ax.set_title(f"Ball {label}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")

        fig.suptitle(f"Linear Ball Position Probe (mean err: {mean_px_err:.1f}px)", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_dir / "ball_position_probe.png", dpi=150)
        plt.close()
        print(f"  Saved: ball_position_probe.png")
    else:
        print("  Not enough ball detections for position probe")

    # ═══════════════════════════════════════════════
    # TEST 3: Per-region MSE breakdown
    # ═══════════════════════════════════════════════
    print(f"\n{'='*50}")
    print("TEST 3: Per-region MSE breakdown")
    print(f"{'='*50}")

    with torch.no_grad():
        recon_all_fg = []
        for i in range(0, N, 256):
            recon_all_fg.append(fg_decoder(all_emb[i:i+256]))
        recon_all_fg = torch.cat(recon_all_fg, dim=0)

        sq_err = (obs_all - recon_all_fg) ** 2  # (N, C, H, W)

        # Region masks (for 64x64 Pong frames)
        H, W = obs_all.shape[2], obs_all.shape[3]
        score_mask = torch.zeros(1, 1, H, W, device=device)
        score_mask[:, :, :12, :] = 1  # top 12 rows = score

        left_paddle = torch.zeros(1, 1, H, W, device=device)
        left_paddle[:, :, 12:, :8] = 1

        right_paddle = torch.zeros(1, 1, H, W, device=device)
        right_paddle[:, :, 12:, -8:] = 1

        playfield = torch.zeros(1, 1, H, W, device=device)
        playfield[:, :, 12:, 8:-8] = 1  # middle area where ball moves

        bg_mask = 1 - (score_mask + left_paddle + right_paddle + playfield).clamp(0, 1)

        regions = {
            "Score area":    score_mask,
            "Left paddle":   left_paddle,
            "Right paddle":  right_paddle,
            "Playfield":     playfield,
        }

        print(f"\n  {'Region':<20s}  {'MSE':>10s}  {'% of total':>10s}")
        print(f"  {'-'*20}  {'-'*10}  {'-'*10}")
        total_mse = sq_err.mean().item()
        for name, mask in regions.items():
            region_err = (sq_err * mask).sum() / mask.sum() / N / obs_channels
            print(f"  {name:<20s}  {region_err.item():.6f}  {100*region_err.item()/total_mse:.1f}%")

        # Foreground vs background
        fg_err = (sq_err * (fg_mask_all > 0.05).float()).sum()
        fg_pixels = (fg_mask_all > 0.05).float().sum()
        bg_err = (sq_err * (fg_mask_all <= 0.05).float()).sum()
        bg_pixels = (fg_mask_all <= 0.05).float().sum()

        fg_mse = (fg_err / fg_pixels).item() if fg_pixels > 0 else 0
        bg_mse = (bg_err / bg_pixels).item() if bg_pixels > 0 else 0

        print(f"\n  {'Foreground (moving)':<20s}  {fg_mse:.6f}")
        print(f"  {'Background (static)':<20s}  {bg_mse:.6f}")
        print(f"  FG/BG ratio: {fg_mse/bg_mse:.1f}x" if bg_mse > 0 else "")

        if fg_mse / bg_mse > 3:
            print("  WARNING: Foreground much worse than background — ball likely not well encoded")
        elif fg_mse / bg_mse > 1.5:
            print("  PARTIAL: Foreground somewhat worse — ball partially captured")
        else:
            print("  GOOD: Foreground quality close to background — moving objects well encoded")

    # ═══════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Embedding: mean={all_emb.mean():.3f}, std={all_emb.std():.3f}")
    print(f"  FG-weighted recon loss: {np.mean(losses_fg[-100:]):.6f}")
    print(f"  Plain recon loss:       {np.mean(losses_plain[-100:]):.6f}")
    if n_valid > 100:
        print(f"  Ball position error:    {mean_px_err:.1f}px (linear probe)")
    print(f"  FG/BG MSE ratio:        {fg_mse/bg_mse:.1f}x")
    print(f"\n  Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
