"""
Rollout quality probe — tests if the predictor maintains meaningful
embeddings over multi-step imagination horizons.

Tests:
  1. Ball position accuracy at each rollout step (1, 3, 5, 10, 15)
  2. Embedding drift: predicted vs actual embedding distance over horizon
  3. Reward prediction accuracy on non-zero reward frames

Usage:
    python probe_rollout.py --checkpoint checkpoints/step_200000.pt \
                            --config configs/pong_jepa.yaml \
                            --save-dir runs/rollout_probe_v6
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
from training.jepa_world_model import JEPAWorldModel


def collect_sequences(env_name, n_frames, seq_len=30, n_seqs=256, seed=42):
    """Collect observation-action sequences with random play."""
    env = make_atari_env(env_name, n_frames=n_frames, seed=seed)
    sequences = []  # list of (obs_seq, act_seq, rew_seq)

    obs, _ = env.reset()
    obs_buf, act_buf, rew_buf = [obs], [], []

    for _ in range(n_seqs * seq_len + 1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        # One-hot action
        act_oh = np.zeros(env.action_space.n, dtype=np.float32)
        act_oh[action] = 1.0

        obs_buf.append(obs)
        act_buf.append(act_oh)
        rew_buf.append(reward)

        if terminated or truncated:
            obs, _ = env.reset()
            # Save completed sequences
            if len(obs_buf) >= seq_len + 1:
                for start in range(0, len(obs_buf) - seq_len, seq_len // 2):
                    if len(sequences) >= n_seqs:
                        break
                    sequences.append((
                        np.stack(obs_buf[start:start + seq_len]),
                        np.stack(act_buf[start:start + seq_len]),
                        np.array(rew_buf[start:start + seq_len]),
                    ))
            obs_buf, act_buf, rew_buf = [obs], [], []

        if len(sequences) >= n_seqs:
            break

    env.close()
    return sequences


def find_ball_in_frame(frame):
    """Find ball position in a single 64x64 Pong frame. Returns (row, col) or None."""
    SCORE_ROWS = 12
    playfield = frame[SCORE_ROWS:, :]
    pf_mean = playfield.mean()
    pf_std = playfield.std()
    thresh = pf_mean + 2.0 * pf_std
    bright = playfield > thresh

    # Remove horizontal border lines
    row_bright_frac = bright.float().mean(dim=1)
    bright[row_bright_frac > 0.3] = False

    bright_coords = bright.nonzero(as_tuple=False)
    if len(bright_coords) < 1:
        return None

    # Filter paddle columns (>=3 bright pixels vertically)
    W = playfield.shape[1]
    col_counts = torch.zeros(W, dtype=torch.long)
    for _, c in bright_coords:
        col_counts[c] += 1
    paddle_cols = col_counts >= 3

    ball_cands = []
    for r, c in bright_coords:
        if not paddle_cols[c]:
            ball_cands.append((r.item(), c.item()))

    if not ball_cands or len(ball_cands) > 12:
        return None

    coords = torch.tensor(ball_cands, dtype=torch.float32)
    center = coords.mean(dim=0)
    if len(ball_cands) > 1:
        dists = (coords - center).pow(2).sum(dim=1).sqrt()
        if dists.max() > 4:
            return None

    return (center[0].item() + SCORE_ROWS, center[1].item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-seqs", type=int, default=256)
    parser.add_argument("--save-dir", type=str, default="runs/rollout_probe")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    embed_dim = cfg["embed_dim"]
    history_size = cfg.get("history_size", 3)
    horizon = cfg.get("horizon", 15)
    act_dim = cfg.get("act_dim", 6)

    # Load world model
    wm = JEPAWorldModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    wm.load_state_dict(ckpt["world_model"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    print(f"Loaded from {args.checkpoint}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect sequences
    seq_len = history_size + horizon + 2
    print(f"Collecting {args.n_seqs} sequences of length {seq_len}...")
    sequences = collect_sequences(
        cfg["env"], n_frames=cfg.get("obs_channels", 4),
        seq_len=seq_len, n_seqs=args.n_seqs,
    )
    print(f"  Got {len(sequences)} sequences")

    # Stack into tensors
    obs_seqs = torch.tensor(np.stack([s[0] for s in sequences]), dtype=torch.float32, device=device)
    act_seqs = torch.tensor(np.stack([s[1] for s in sequences]), dtype=torch.float32, device=device)
    rew_seqs = torch.tensor(np.stack([s[2] for s in sequences]), dtype=torch.float32, device=device)
    N, T = obs_seqs.shape[:2]
    print(f"  obs: {obs_seqs.shape}, act: {act_seqs.shape}")

    # ═══════════════════════════════════════════
    # Encode all observations (ground truth embeddings)
    # ═══════════════════════════════════════════
    print("\nEncoding all observations...")
    with torch.no_grad():
        all_emb = []
        for i in range(0, N, 32):
            all_emb.append(wm.encode(obs_seqs[i:i + 32]))
        all_emb = torch.cat(all_emb, dim=0)  # (N, T, D)
    print(f"  embeddings: {all_emb.shape}")

    # ═══════════════════════════════════════════
    # TEST 1: Multi-step rollout — predicted vs real embeddings
    # ═══════════════════════════════════════════
    print(f"\n{'=' * 50}")
    print(f"TEST 1: Predictor rollout quality (horizon={horizon})")
    print(f"{'=' * 50}")

    # Start rollout from step `history_size` in each sequence
    start_t = history_size
    l2_per_step = []
    cosine_per_step = []

    with torch.no_grad():
        # Initialize with real context
        emb_buffer = [all_emb[:, start_t - i - 1 + history_size] for i in range(history_size)]
        emb_buffer = emb_buffer[::-1]  # oldest first
        act_buffer = [act_seqs[:, start_t - i - 1 + history_size] for i in range(history_size)]
        act_buffer = act_buffer[::-1]

        # Actually, let's be more careful. We want:
        # emb_buffer = [emb at t=start_t-HS+1, ..., emb at t=start_t]
        # act_buffer = [act at t=start_t-HS+1, ..., act at t=start_t]
        emb_buffer = [all_emb[:, start_t - history_size + 1 + i] for i in range(history_size)]
        act_buffer = [act_seqs[:, start_t - history_size + 1 + i] for i in range(history_size)]

        for step in range(horizon):
            real_t = start_t + step + 1
            if real_t >= T:
                break

            # Current action (from real sequence)
            action = act_seqs[:, start_t + step]
            act_buffer.append(action)

            # Prepare context
            ctx_embs = torch.stack(emb_buffer[-history_size:], dim=1)
            ctx_acts = torch.stack(act_buffer[-history_size:], dim=1)

            # Predict next embedding
            pred = wm.predictor(ctx_embs, ctx_acts)
            pred_emb = pred[:, -1].clamp(-10, 10)
            emb_buffer.append(pred_emb)

            # Compare to real embedding at this timestep
            real_emb = all_emb[:, real_t]
            l2 = (pred_emb - real_emb).pow(2).sum(dim=1).sqrt()  # (N,)
            cosine = F.cosine_similarity(pred_emb, real_emb, dim=1)  # (N,)

            l2_per_step.append(l2.cpu())
            cosine_per_step.append(cosine.cpu())

    l2_per_step = torch.stack(l2_per_step, dim=1)  # (N, steps)
    cosine_per_step = torch.stack(cosine_per_step, dim=1)

    print(f"\n  {'Step':>6s}  {'L2 dist':>10s}  {'Cosine sim':>12s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}")
    steps_to_show = [0, 2, 4, 9, 14]
    for s in steps_to_show:
        if s < l2_per_step.shape[1]:
            print(f"  {s+1:>6d}  {l2_per_step[:, s].mean():.4f}  {cosine_per_step[:, s].mean():.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    steps = list(range(1, l2_per_step.shape[1] + 1))

    ax1.plot(steps, l2_per_step.mean(dim=0).numpy(), "b-o", markersize=4)
    ax1.fill_between(steps,
                     l2_per_step.quantile(0.25, dim=0).numpy(),
                     l2_per_step.quantile(0.75, dim=0).numpy(),
                     alpha=0.2)
    ax1.set_xlabel("Rollout step")
    ax1.set_ylabel("L2 distance (pred vs real)")
    ax1.set_title("Embedding drift over rollout horizon")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, cosine_per_step.mean(dim=0).numpy(), "g-o", markersize=4)
    ax2.fill_between(steps,
                     cosine_per_step.quantile(0.25, dim=0).numpy(),
                     cosine_per_step.quantile(0.75, dim=0).numpy(),
                     alpha=0.2)
    ax2.set_xlabel("Rollout step")
    ax2.set_ylabel("Cosine similarity")
    ax2.set_title("Embedding direction preservation")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "rollout_drift.png", dpi=150)
    plt.close()
    print(f"  Saved: rollout_drift.png")

    # ═══════════════════════════════════════════
    # TEST 2: Ball position through rollout
    # ═══════════════════════════════════════════
    print(f"\n{'=' * 50}")
    print("TEST 2: Ball position accuracy through rollout")
    print(f"{'=' * 50}")

    # Detect ball in all real frames
    print("  Detecting ball positions in real frames...")
    ball_pos_all = torch.full((N, T, 2), float("nan"))
    for i in range(N):
        for t in range(T):
            bp = find_ball_in_frame(obs_seqs[i, t, -1].cpu())
            if bp is not None:
                ball_pos_all[i, t, 0] = bp[0] / 64.0
                ball_pos_all[i, t, 1] = bp[1] / 64.0

    n_with_ball = (~torch.isnan(ball_pos_all[:, :, 0])).sum().item()
    print(f"  Ball found in {n_with_ball}/{N*T} frames ({100*n_with_ball/(N*T):.1f}%)")

    # Train a single linear probe on real embeddings (all timesteps)
    print("  Training linear ball probe on real embeddings...")
    flat_emb = all_emb.reshape(-1, embed_dim)
    flat_pos = ball_pos_all.reshape(-1, 2).to(device)
    valid = ~torch.isnan(flat_pos[:, 0])

    if valid.sum() < 200:
        print("  Not enough ball detections for rollout ball probe")
    else:
        probe = nn.Linear(embed_dim, 2).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)

        valid_emb = flat_emb[valid]
        valid_pos = flat_pos[valid]
        perm = torch.randperm(len(valid_emb))
        n_train = int(0.8 * len(valid_emb))
        train_e, train_p = valid_emb[perm[:n_train]], valid_pos[perm[:n_train]]
        test_e, test_p = valid_emb[perm[n_train:]], valid_pos[perm[n_train:]]

        for step in range(2000):
            idx = torch.randint(0, n_train, (min(128, n_train),))
            loss = F.mse_loss(probe(train_e[idx]), train_p[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Test on real embeddings
        with torch.no_grad():
            real_pred = probe(test_e)
            real_err = ((real_pred - test_p) * 64).pow(2).sum(dim=1).sqrt()
            print(f"  Real embedding ball error: {real_err.mean():.1f}px (baseline)")

        # Now test on IMAGINED embeddings at each rollout step
        print("  Testing ball probe on imagined embeddings at each step...")
        imagined_embs = emb_buffer[history_size + 1:]  # skip initial context

        ball_err_per_step = []
        for step_i in range(min(len(imagined_embs), horizon)):
            real_t = start_t + step_i + 1
            if real_t >= T:
                break

            imag_emb = imagined_embs[step_i]  # (N, D)
            real_pos_t = ball_pos_all[:, real_t].to(device)  # (N, 2)
            valid_t = ~torch.isnan(real_pos_t[:, 0])

            if valid_t.sum() < 10:
                ball_err_per_step.append(float("nan"))
                continue

            with torch.no_grad():
                pred_pos = probe(imag_emb[valid_t])
                err = ((pred_pos - real_pos_t[valid_t]) * 64).pow(2).sum(dim=1).sqrt()
                ball_err_per_step.append(err.mean().item())

        print(f"\n  {'Step':>6s}  {'Ball error (px)':>15s}  {'vs baseline':>12s}")
        print(f"  {'-'*6}  {'-'*15}  {'-'*12}")
        baseline = real_err.mean().item()
        for s in steps_to_show:
            if s < len(ball_err_per_step) and not np.isnan(ball_err_per_step[s]):
                ratio = ball_err_per_step[s] / baseline
                print(f"  {s+1:>6d}  {ball_err_per_step[s]:>15.1f}  {ratio:>11.1f}x")

        # Plot
        valid_steps = [(i, e) for i, e in enumerate(ball_err_per_step) if not np.isnan(e)]
        if valid_steps:
            xs, ys = zip(*valid_steps)
            xs = [x + 1 for x in xs]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(xs, ys, "r-o", markersize=5, label="Imagined embedding")
            ax.axhline(baseline, color="g", linestyle="--", label=f"Real embedding ({baseline:.1f}px)")
            ax.set_xlabel("Rollout step")
            ax.set_ylabel("Ball position error (pixels)")
            ax.set_title("Ball tracking through imagination rollout")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / "ball_through_rollout.png", dpi=150)
            plt.close()
            print(f"  Saved: ball_through_rollout.png")

    # ═══════════════════════════════════════════
    # TEST 3: Reward prediction accuracy
    # ═══════════════════════════════════════════
    print(f"\n{'=' * 50}")
    print("TEST 3: Reward prediction accuracy")
    print(f"{'=' * 50}")

    with torch.no_grad():
        pred_rewards = []
        for i in range(0, N, 32):
            pred_rewards.append(wm.reward_pred(all_emb[i:i + 32]))
        pred_rewards = torch.cat(pred_rewards, dim=0)  # (N, T)

        # Shift rewards to match arrival convention
        actual = torch.cat([torch.zeros_like(rew_seqs[:, :1]), rew_seqs[:, :-1]], dim=1)
        from training.jepa_world_model import symlog
        actual_symlog = symlog(actual.to(device))

        # Overall MSE
        rew_mse = F.mse_loss(pred_rewards, actual_symlog).item()

        # Accuracy on non-zero rewards
        nonzero = actual.abs() > 0.5
        n_nonzero = nonzero.sum().item()
        n_total = actual.numel()

        if n_nonzero > 0:
            nz_pred = pred_rewards[nonzero.to(device)]
            nz_actual = actual_symlog[nonzero.to(device)]
            nz_mse = F.mse_loss(nz_pred, nz_actual).item()
            # Sign accuracy: does predicted reward have correct sign?
            sign_correct = ((nz_pred > 0) == (nz_actual > 0)).float().mean().item()
        else:
            nz_mse = float("nan")
            sign_correct = float("nan")

        # Imagined reward sparsity
        imag_nonzero_frac = (pred_rewards.abs() > 0.1).float().mean().item()

    print(f"  Total frames:           {n_total}")
    print(f"  Non-zero reward frames: {n_nonzero} ({100*n_nonzero/n_total:.2f}%)")
    print(f"  Overall reward MSE:     {rew_mse:.6f}")
    if n_nonzero > 0:
        print(f"  Non-zero reward MSE:    {nz_mse:.6f}")
        print(f"  Sign accuracy:          {100*sign_correct:.1f}%")
    print(f"  Predicted |r|>0.1:      {100*imag_nonzero_frac:.1f}%")

    if imag_nonzero_frac < 0.01:
        print("  WARNING: Reward predictor outputs near-zero for almost all frames")
        print("  This means the agent gets no reward signal during imagination")

    # ═══════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Rollout L2 drift:   step1={l2_per_step[:, 0].mean():.2f} → step{l2_per_step.shape[1]}={l2_per_step[:, -1].mean():.2f}")
    print(f"  Rollout cosine sim: step1={cosine_per_step[:, 0].mean():.3f} → step{cosine_per_step.shape[1]}={cosine_per_step[:, -1].mean():.3f}")
    if valid.sum() >= 200:
        print(f"  Ball error:         real={baseline:.1f}px → step15={ball_err_per_step[-1]:.1f}px" if not np.isnan(ball_err_per_step[-1]) else "")
    print(f"  Reward sign accuracy: {100*sign_correct:.1f}%" if n_nonzero > 0 else "  Reward: no non-zero samples")
    print(f"\n  Results saved to: {save_dir}/")


if __name__ == "__main__":
    main()
