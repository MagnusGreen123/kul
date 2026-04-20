"""
Detailed JEPA embedding probe — reveals what the encoder actually captures.

Three complementary tests:
  1. FG-weighted decoder vs plain: side-by-side with ball markers
  2. Linear ball locator: predicts ball (x,y) from embedding with validation overlay
  3. Per-region MSE breakdown: score / paddles / playfield / ball-area

Ball detection uses direct bright-pixel detection in the playfield area,
not temporal diff (which catches paddle movement and score changes too).

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
from matplotlib.patches import Circle

from envs.wrappers import make_atari_env
from models.decoder import ConvDecoder
from training.jepa_world_model import JEPAWorldModel


# ── Pong layout at 64x64 ──
# Original Pong: 210x160, resized to 64x64
# Score area:    rows 0-11
# Playfield:     rows 12-63
# Left paddle:   cols 4-7 (agent paddle, ~2px wide after resize)
# Right paddle:  cols 56-59 (opponent paddle)
# Center line:   col ~32 (dashed)
# Ball:          ~1-2px, anywhere in playfield

SCORE_ROWS = 12        # top rows reserved for score display
PADDLE_L_MAX = 10      # left paddle region ends here
PADDLE_R_MIN = 54      # right paddle region starts here
CENTER_COL = 32        # center dashed line


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


def find_ball_pixel(frame, debug=False):
    """Find ball position in a single 64x64 Pong frame.

    Args:
        frame: (H, W) tensor, normalized to [-0.5, 0.5]
        debug: print detailed detection info

    Returns:
        (row, col) tuple in pixels, or None if no ball found.

    Method: find all bright pixels below the score area, then separate the
    ball from paddles by shape — paddles are tall vertical bars (many bright
    pixels in the same column), the ball is small (1-2 pixels per column).
    Then cluster remaining pixels and pick the smallest compact cluster.
    """
    H, W = frame.shape

    # Step 1: find all bright pixels in playfield (below score)
    playfield = frame[SCORE_ROWS:, :]  # (H-12, W)
    pf_mean = playfield.mean()
    pf_std = playfield.std()
    thresh = pf_mean + 2.0 * pf_std
    bright = playfield > thresh

    # Step 1b: remove horizontal bright lines (score borders, dividers)
    # A row where >30% of pixels are bright is a border line, not an object
    PH, PW = playfield.shape
    row_bright_frac = bright.float().mean(dim=1)  # fraction bright per row
    border_rows = row_bright_frac > 0.3
    if border_rows.any():
        bright[border_rows] = False

    bright_coords = bright.nonzero(as_tuple=False)  # (K, 2) — row, col in playfield coords

    if debug:
        n_border = border_rows.sum().item()
        print(f"      playfield mean={pf_mean:.3f} std={pf_std:.3f} thresh={thresh:.3f}")
        print(f"      border rows removed: {n_border}")
        print(f"      bright pixels (after border removal): {len(bright_coords)}")

    if len(bright_coords) < 1:
        return None

    # Step 2: count bright pixels per column — paddle columns have many, ball has few
    col_counts = torch.zeros(PW, dtype=torch.long)
    for _, c in bright_coords:
        col_counts[c] += 1

    # Paddle columns: >= 3 bright pixels vertically
    paddle_cols = col_counts >= 3

    if debug:
        paddle_col_list = paddle_cols.nonzero(as_tuple=True)[0].tolist()
        print(f"      paddle columns (>=3 bright): {paddle_col_list}")

    # Step 3: filter out paddle pixels — keep only pixels in non-paddle columns
    ball_candidates = []
    for r, c in bright_coords:
        if not paddle_cols[c]:
            ball_candidates.append((r.item(), c.item()))

    if debug:
        print(f"      non-paddle candidates: {len(ball_candidates)}")
        if ball_candidates:
            print(f"      candidate positions: {ball_candidates[:20]}")

    if len(ball_candidates) == 0:
        return None

    # Step 4: cluster candidates by proximity and find the ball cluster
    coords = torch.tensor(ball_candidates, dtype=torch.float32)

    # Simple greedy clustering: group pixels within distance 3 of each other
    clusters = []
    used = set()
    for i in range(len(coords)):
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        for j in range(i + 1, len(coords)):
            if j in used:
                continue
            # Check if pixel j is close to any pixel already in cluster
            for k in cluster:
                dist = (coords[j] - coords[k]).pow(2).sum().sqrt().item()
                if dist <= 3:
                    cluster.append(j)
                    used.add(j)
                    break
        clusters.append(cluster)

    if debug:
        print(f"      clusters found: {len(clusters)}, sizes: {[len(c) for c in clusters]}")

    # Ball is the smallest compact cluster (1-6 pixels)
    best = None
    for cluster in clusters:
        if len(cluster) > 8:
            continue  # too big for a ball
        c_coords = coords[cluster]
        center = c_coords.mean(dim=0)
        # Check compactness
        if len(cluster) > 1:
            dists = (c_coords - center).pow(2).sum(dim=1).sqrt()
            if dists.max() > 4:
                continue
        if best is None or len(cluster) < len(best):
            best = cluster

    if best is None:
        return None

    center = coords[best].mean(dim=0)
    if debug:
        print(f"      ball cluster size={len(best)}, center=({center[0].item():.1f}, {center[1].item():.1f})")

    # Return in full-frame coordinates (add SCORE_ROWS offset to row)
    return (center[0].item() + SCORE_ROWS, center[1].item())


def detect_all_balls(obs, debug=False):
    """Detect ball in all frames. Uses the newest frame (last channel).

    Args:
        obs: (N, C, H, W) tensor
        debug: if True, print stats for first few frames

    Returns:
        positions: (N, 2) tensor with (row, col) in pixels, NaN if not found
        valid_mask: (N,) bool tensor
    """
    N = obs.shape[0]
    positions = torch.full((N, 2), float("nan"))

    for i in range(N):
        frame = obs[i, -1]  # newest frame in stack
        do_debug = debug and i < 8
        if do_debug:
            print(f"    frame {i}:")
        result = find_ball_pixel(frame, debug=do_debug)
        if result is not None:
            positions[i, 0] = result[0]
            positions[i, 1] = result[1]
        if do_debug:
            print(f"      result: {'found at ({:.1f}, {:.1f})'.format(*result) if result else 'none'}")

    valid = ~torch.isnan(positions[:, 0])
    return positions, valid


def frame_to_img(frame_tensor):
    """Convert [-0.5, 0.5] tensor to [0, 255] uint8 numpy for display."""
    img = frame_tensor.detach().cpu().numpy()
    return np.clip((img + 0.5) * 255, 0, 255).astype(np.uint8)


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
    N, C, H, W = obs_all.shape
    print(f"  obs: {obs_all.shape}, range [{obs_all.min():.2f}, {obs_all.max():.2f}]")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Precompute embeddings ──
    print("Computing embeddings...")
    all_emb = []
    with torch.no_grad():
        for i in range(0, N, 256):
            all_emb.append(wm.encode(obs_all[i:i + 256]))
    all_emb = torch.cat(all_emb, dim=0)
    print(f"  emb: {all_emb.shape}, mean={all_emb.mean():.4f}, std={all_emb.std():.4f}")

    # ── Detect ball in all frames ──
    print("\nDetecting ball positions (pixel-based, adaptive threshold)...")
    ball_pos, ball_valid = detect_all_balls(obs_all.cpu(), debug=True)
    n_valid = ball_valid.sum().item()
    print(f"  Found ball in {n_valid}/{N} frames ({100 * n_valid / N:.1f}%)")

    # Save raw frame samples so we can debug visually regardless of detection
    print("  Saving raw frame samples...")
    sample_every = max(1, N // 20)
    raw_idx = list(range(0, N, sample_every))[:20]
    n_raw = len(raw_idx)
    fig, axes = plt.subplots(2, n_raw, figsize=(2.5 * n_raw, 5))
    for i in range(n_raw):
        idx = raw_idx[i]
        # Newest frame (channel -1), full contrast stretch
        newest = obs_all[idx, -1].cpu().numpy()
        # Also show oldest frame for temporal diff
        oldest = obs_all[idx, 0].cpu().numpy()

        for row, (data, label) in enumerate([(newest, "ch3 (new)"), (oldest, "ch0 (old)")]):
            img = np.clip((data + 0.5) * 255, 0, 255).astype(np.uint8)
            axes[row, i].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[row, i].axis("off")
            if i == 0:
                axes[row, i].set_ylabel(label, fontsize=10)
            # Mark ball if found
            if ball_valid[idx] and row == 0:
                r, c = ball_pos[idx, 0].item(), ball_pos[idx, 1].item()
                circle = Circle((c, r), radius=3, fill=False, edgecolor="red", linewidth=2)
                axes[row, i].add_patch(circle)
        axes[0, i].set_title(f"#{idx}", fontsize=8)
    fig.suptitle(f"Raw frames — red circle = detected ball ({n_valid}/{N} found)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "raw_frames.png", dpi=150)
    plt.close()
    print(f"  Saved: raw_frames.png")

    # ── Save ball detection verification ──
    valid_indices = ball_valid.nonzero(as_tuple=True)[0]

    if n_valid > 0:
        print("  Saving ball detection verification...")
        if len(valid_indices) > 20:
            sample_idx = valid_indices[torch.linspace(0, len(valid_indices) - 1, 20).long()]
        else:
            sample_idx = valid_indices

        n_show = len(sample_idx)
        cols = min(10, n_show)
        rows = max(1, (n_show + cols - 1) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
        axes = np.array(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")
        for i in range(n_show):
            idx = sample_idx[i].item()
            img = frame_to_img(obs_all[idx, -1].cpu())
            axes[i].imshow(img, cmap="gray", vmin=0, vmax=255)
            r, c = ball_pos[idx, 0].item(), ball_pos[idx, 1].item()
            circle = Circle((c, r), radius=3, fill=False, edgecolor="red", linewidth=2)
            axes[i].add_patch(circle)
            axes[i].set_title(f"#{idx}", fontsize=8)
        fig.suptitle(f"Ball detection verification ({n_valid}/{N} found)", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_dir / "ball_detection_verify.png", dpi=150)
        plt.close()
        print(f"  Saved: ball_detection_verify.png")
    else:
        print("  WARNING: No balls detected! Check raw_frames.png to debug.")

    # ═══════════════════════════════════════════════
    # TEST 1: Decoder reconstruction with ball markers
    # ═══════════════════════════════════════════════
    print(f"\n{'=' * 50}")
    print(f"TEST 1: Decoder reconstruction (fg_weight={fg_weight} vs plain)")
    print(f"{'=' * 50}")

    # Train FG-weighted decoder
    fg_decoder = ConvDecoder(
        latent_dim=embed_dim, out_channels=obs_channels, depth=args.decoder_depth
    ).to(device)
    optimizer_fg = torch.optim.Adam(fg_decoder.parameters(), lr=args.lr)

    # FG mask via temporal diff (same method as training)
    prev_obs = torch.cat([obs_all[:1], obs_all[:-1]], dim=0)
    fg_mask_all = (obs_all - prev_obs).abs().mean(dim=1, keepdim=True)

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
            print(f"  step {step + 1:4d} | fg_recon_loss: {np.mean(losses_fg[-200:]):.6f}")

    # Train plain decoder
    print("  Training plain decoder for comparison...")
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

    # 3-row comparison: real / fg-weighted / plain
    print("  Saving decoder comparison...")
    with torch.no_grad():
        if n_valid >= 10:
            show_idx = valid_indices[torch.linspace(0, len(valid_indices) - 1, 10).long()]
        elif n_valid > 0:
            show_idx = valid_indices[:n_valid]
        else:
            # No ball detected — just show evenly spaced frames
            show_idx = torch.linspace(0, N - 1, 10).long()

        n = len(show_idx)
        obs_show = obs_all[show_idx]
        emb_show = all_emb[show_idx]
        recon_fg = fg_decoder(emb_show)
        recon_plain = plain_decoder(emb_show)

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    for i in range(n):
        fr_idx = show_idx[i].item()
        bp = ball_pos[fr_idx]

        for row, (data, label) in enumerate([
            (obs_show, "Real"),
            (recon_fg, "FG-weighted"),
            (recon_plain, "Plain"),
        ]):
            img = frame_to_img(data[i, -1])  # newest frame channel
            axes[row, i].imshow(img, cmap="gray", vmin=0, vmax=255)
            # Mark ball position on all rows
            if not torch.isnan(bp[0]):
                circle = Circle(
                    (bp[1].item(), bp[0].item()),
                    radius=3, fill=False, edgecolor="red", linewidth=1.5,
                )
                axes[row, i].add_patch(circle)
            axes[row, i].axis("off")
            if i == 0:
                axes[row, i].set_ylabel(label, fontsize=12)

    fig.suptitle("Decoder comparison — red circle = detected ball position", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_dir / "decoder_comparison.png", dpi=200)
    plt.close()
    print(f"  Saved: decoder_comparison.png")

    # Zoomed ball region comparison (only if balls detected)
    if n_valid > 0:
        print("  Saving zoomed ball region comparison...")
    zoom_r = 8  # zoom radius in pixels
    n_zoom = min(8, n) if n_valid > 0 else 0
    if n_zoom > 0:
        fig, axes = plt.subplots(3, n_zoom, figsize=(3 * n_zoom, 9))
    for i in range(n_zoom):
        fr_idx = show_idx[i].item()
        bp = ball_pos[fr_idx]
        if torch.isnan(bp[0]):
            continue
        r, c = int(bp[0].item()), int(bp[1].item())
        r0 = max(0, r - zoom_r)
        r1 = min(H, r + zoom_r)
        c0 = max(0, c - zoom_r)
        c1 = min(W, c + zoom_r)

        for row, (data, label) in enumerate([
            (obs_show, "Real"),
            (recon_fg, "FG-weighted"),
            (recon_plain, "Plain"),
        ]):
            patch = frame_to_img(data[i, -1])[r0:r1, c0:c1]
            axes[row, i].imshow(patch, cmap="gray", vmin=0, vmax=255,
                                interpolation="nearest")
            # Mark ball center
            axes[row, i].plot(c - c0, r - r0, "r+", markersize=10, markeredgewidth=2)
            axes[row, i].axis("off")
            if i == 0:
                axes[row, i].set_ylabel(label, fontsize=12)

    if n_zoom > 0:
        fig.suptitle(f"Zoomed {zoom_r * 2}x{zoom_r * 2} region around ball", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_dir / "ball_zoom_comparison.png", dpi=200)
        plt.close()
        print(f"  Saved: ball_zoom_comparison.png")

    # ═══════════════════════════════════════════════
    # TEST 2: Linear ball position probe
    # ═══════════════════════════════════════════════
    print(f"\n{'=' * 50}")
    print("TEST 2: Linear ball position probe")
    print(f"{'=' * 50}")

    if n_valid < 100:
        print("  Not enough ball detections for position probe")
        mean_px_err = float("nan")
    else:
        # Normalize positions to [0, 1]
        norm_pos = ball_pos.clone()
        norm_pos[:, 0] /= H
        norm_pos[:, 1] /= W

        valid_emb = all_emb[ball_valid]
        valid_pos = norm_pos[ball_valid].to(device)

        # Train/test split (shuffle to avoid temporal correlation)
        perm = torch.randperm(len(valid_emb))
        n_train = int(0.8 * len(valid_emb))
        train_emb = valid_emb[perm[:n_train]]
        train_pos = valid_pos[perm[:n_train]]
        test_emb = valid_emb[perm[n_train:]]
        test_pos = valid_pos[perm[n_train:]]

        ball_probe = nn.Linear(embed_dim, 2).to(device)
        optimizer_ball = torch.optim.Adam(ball_probe.parameters(), lr=1e-3)

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
                print(f"  step {step + 1:4d} | ball_pos_mse: {np.mean(losses_ball[-500:]):.6f}")

        # Evaluate
        with torch.no_grad():
            test_pred = ball_probe(test_emb)
            test_mse = F.mse_loss(test_pred, test_pos).item()
            pixel_err = ((test_pred - test_pos) * 64).pow(2).sum(dim=1).sqrt()
            mean_px_err = pixel_err.mean().item()
            median_px_err = pixel_err.median().item()

        print(f"\n  Test MSE:            {test_mse:.6f}")
        print(f"  Mean pixel error:    {mean_px_err:.1f} px (of 64)")
        print(f"  Median pixel error:  {median_px_err:.1f} px (of 64)")

        if mean_px_err < 5:
            print("  RESULT: Embedding accurately encodes ball position")
        elif mean_px_err < 15:
            print("  RESULT: Embedding has rough ball position info")
        else:
            print("  RESULT: Embedding does NOT reliably encode ball position")

        # Scatter plot
        tp = test_pos.cpu().numpy()
        pp = test_pred.cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, dim, label in [(axes[0], 0, "Row (Y)"), (axes[1], 1, "Col (X)")]:
            ax.scatter(tp[:, dim], pp[:, dim], alpha=0.3, s=10, c="steelblue")
            ax.plot([0, 1], [0, 1], "r--", linewidth=1)
            ax.set_xlabel(f"Actual {label}")
            ax.set_ylabel(f"Predicted {label}")
            ax.set_title(f"Ball {label}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
        fig.suptitle(f"Linear Ball Probe — mean err: {mean_px_err:.1f}px, median: {median_px_err:.1f}px",
                     fontsize=13)
        plt.tight_layout()
        plt.savefig(save_dir / "ball_position_probe.png", dpi=150)
        plt.close()
        print(f"  Saved: ball_position_probe.png")

        # Overlay: show predicted vs actual on real frames
        print("  Saving prediction overlay...")
        test_indices = perm[n_train:]
        orig_indices = ball_valid.nonzero(as_tuple=True)[0][test_indices]
        n_overlay = min(10, len(orig_indices))
        overlay_idx = torch.linspace(0, len(orig_indices) - 1, n_overlay).long()

        fig, axes = plt.subplots(1, n_overlay, figsize=(3 * n_overlay, 3))
        if n_overlay == 1:
            axes = [axes]
        for i in range(n_overlay):
            oi = overlay_idx[i]
            fr_idx = orig_indices[oi].item()
            img = frame_to_img(obs_all[fr_idx, -1].cpu())
            axes[i].imshow(img, cmap="gray", vmin=0, vmax=255)

            # Actual position (green)
            ar, ac = test_pos[oi, 0].item() * H, test_pos[oi, 1].item() * W
            # Predicted position (red)
            pr, pc = test_pred[oi, 0].item() * H, test_pred[oi, 1].item() * W

            axes[i].plot(ac, ar, "g+", markersize=12, markeredgewidth=2, label="Actual")
            axes[i].plot(pc, pr, "rx", markersize=10, markeredgewidth=2, label="Predicted")
            err = pixel_err[oi].item()
            axes[i].set_title(f"err={err:.1f}px", fontsize=9)
            axes[i].axis("off")
            if i == 0:
                axes[i].legend(fontsize=7, loc="lower left")

        fig.suptitle("Ball position: green+=actual, red x=predicted from embedding", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_dir / "ball_prediction_overlay.png", dpi=200)
        plt.close()
        print(f"  Saved: ball_prediction_overlay.png")

    # ═══════════════════════════════════════════════
    # TEST 3: Per-region MSE breakdown
    # ═══════════════════════════════════════════════
    print(f"\n{'=' * 50}")
    print("TEST 3: Per-region MSE breakdown")
    print(f"{'=' * 50}")

    with torch.no_grad():
        recon_all_fg = []
        for i in range(0, N, 256):
            recon_all_fg.append(fg_decoder(all_emb[i:i + 256]))
        recon_all_fg = torch.cat(recon_all_fg, dim=0)

        sq_err = (obs_all - recon_all_fg) ** 2

        # Region masks
        score_mask = torch.zeros(1, 1, H, W, device=device)
        score_mask[:, :, :SCORE_ROWS, :] = 1

        left_paddle = torch.zeros(1, 1, H, W, device=device)
        left_paddle[:, :, SCORE_ROWS:, :PADDLE_L_MAX] = 1

        right_paddle = torch.zeros(1, 1, H, W, device=device)
        right_paddle[:, :, SCORE_ROWS:, PADDLE_R_MIN:] = 1

        playfield = torch.zeros(1, 1, H, W, device=device)
        playfield[:, :, SCORE_ROWS:, PADDLE_L_MAX:PADDLE_R_MIN] = 1

        regions = {
            "Score area":    score_mask,
            "Left paddle":   left_paddle,
            "Right paddle":  right_paddle,
            "Playfield":     playfield,
        }

        print(f"\n  {'Region':<20s}  {'MSE':>12s}")
        print(f"  {'-' * 20}  {'-' * 12}")
        for name, mask in regions.items():
            region_err = (sq_err * mask).sum() / (mask.sum() * N * C)
            print(f"  {name:<20s}  {region_err.item():.6f}")

        # Ball-area MSE: compute MSE specifically at ball locations
        if n_valid >= 50:
            ball_region_errs = []
            for i in range(N):
                if not ball_valid[i]:
                    continue
                r, c = int(ball_pos[i, 0].item()), int(ball_pos[i, 1].item())
                r0, r1 = max(0, r - 2), min(H, r + 3)
                c0, c1 = max(0, c - 2), min(W, c + 3)
                ball_err = sq_err[i, :, r0:r1, c0:c1].mean().item()
                ball_region_errs.append(ball_err)

            ball_mse = np.mean(ball_region_errs)
            bg_region = sq_err[:, :, SCORE_ROWS:, PADDLE_L_MAX:PADDLE_R_MIN]
            bg_mse = bg_region.mean().item()

            print(f"\n  {'Ball region (5x5)':<20s}  {ball_mse:.6f}")
            print(f"  {'Playfield avg':<20s}  {bg_mse:.6f}")
            ratio = ball_mse / bg_mse if bg_mse > 0 else float("inf")
            print(f"  Ball/Playfield ratio: {ratio:.1f}x")

            if ratio < 2:
                print("  GOOD: Ball region reconstructed as well as playfield")
            elif ratio < 5:
                print("  PARTIAL: Ball region somewhat worse than playfield")
            else:
                print("  POOR: Ball region much worse — ball not well captured")
        else:
            ball_mse = float("nan")
            bg_mse = float("nan")
            ratio = float("nan")
            print("\n  (not enough ball detections for ball-region analysis)")

    # ═══════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Checkpoint:           {args.checkpoint}")
    print(f"  Embedding:            mean={all_emb.mean():.3f}, std={all_emb.std():.3f}")
    print(f"  Ball detected:        {n_valid}/{N} frames ({100 * n_valid / N:.1f}%)")
    print(f"  FG-weighted recon:    {np.mean(losses_fg[-100:]):.6f}")
    print(f"  Plain recon:          {np.mean(losses_plain[-100:]):.6f}")
    if not np.isnan(mean_px_err):
        print(f"  Ball position error:  {mean_px_err:.1f}px mean, {median_px_err:.1f}px median")
    if not np.isnan(ratio):
        print(f"  Ball/Playfield MSE:   {ratio:.1f}x")
    print(f"\n  All images saved to: {save_dir}/")
    print(f"  Key files:")
    print(f"    ball_detection_verify.png  — verify ball detector works")
    print(f"    decoder_comparison.png     — real vs fg-weighted vs plain")
    print(f"    ball_zoom_comparison.png   — zoomed view around ball")
    print(f"    ball_position_probe.png    — linear probe scatter plot")
    print(f"    ball_prediction_overlay.png — predicted vs actual on frames")


if __name__ == "__main__":
    main()
