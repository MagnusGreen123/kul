"""
Overfit test: can encoder + RSSM + decoder reconstruct a SINGLE fixed batch
of real Pong frames if we train only on that batch for N steps?

If YES — the vision pipeline is fine, the RL signal is the bottleneck.
If NO  — the decoder/encoder/loss is fundamentally broken.

Also runs ablations at the end:
    decoder(h, z)       — full
    decoder(h=0, z)     — is h needed?
    decoder(h, z=0)     — is z needed?  (if std of output doesn't change → z is ignored)

Usage:
    python scripts/overfit_test.py --config configs/pong.yaml --steps 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, ".")

from envs.wrappers import make_atari_env
from training.world_model import WorldModel, WorldModelTrainer
from utils.visualization import save_reconstruction_grid


def collect_batch(cfg, B, T, device, seed=42):
    """Collect B independent trajectories of length T using a random policy."""
    act_dim = cfg["act_dim"]
    obs_channels = cfg["obs_channels"]

    batch_obs, batch_act, batch_rew, batch_done = [], [], [], []
    rng = np.random.default_rng(seed)

    for i in range(B):
        env = make_atari_env(cfg["env"], n_frames=obs_channels, seed=seed + i)
        obs, _ = env.reset()
        obs_list, act_list, rew_list, done_list = [], [], [], []
        for _ in range(T):
            action = int(rng.integers(0, act_dim))
            obs_list.append(obs.copy())
            one_hot = np.zeros(act_dim, dtype=np.float32)
            one_hot[action] = 1.0
            act_list.append(one_hot)
            next_obs, reward, term, trunc, _ = env.step(action)
            rew_list.append(float(reward))
            done_list.append(1.0 if (term or trunc) else 0.0)
            if term or trunc:
                obs, _ = env.reset()
            else:
                obs = next_obs
        env.close()
        batch_obs.append(np.stack(obs_list))
        batch_act.append(np.stack(act_list))
        batch_rew.append(np.array(rew_list, dtype=np.float32))
        batch_done.append(np.array(done_list, dtype=np.float32))

    return {
        "obs":    torch.from_numpy(np.stack(batch_obs)).float().to(device),
        "action": torch.from_numpy(np.stack(batch_act)).float().to(device),
        "reward": torch.from_numpy(np.stack(batch_rew)).float().to(device),
        "done":   torch.from_numpy(np.stack(batch_done)).float().to(device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pong.yaml")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-length", type=int, default=30)
    parser.add_argument("--out-dir", default="runs/overfit-test")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--recon-only", action="store_true",
                        help="Disable KL and reward losses (pure vision test).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.recon_only:
        cfg["kl_weight"] = 0.0
        cfg["reward_weight"] = 0.0
        cfg["free_bits"] = 0.0
    cfg["mixed_precision"] = False  # fp32 for clean diagnostics
    cfg["learning_rate"] = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {'recon-only' if args.recon_only else 'full (recon+kl+reward+cont)'}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting batch B={args.batch_size} T={args.batch_length} from {cfg['env']}...")
    batch = collect_batch(cfg, args.batch_size, args.batch_length, device, seed=cfg.get("seed", 42))
    obs = batch["obs"]
    print(f"obs shape={tuple(obs.shape)} range=[{obs.min():.3f}, {obs.max():.3f}] "
          f"mean={obs.mean():.3f} std={obs.std():.3f}")
    n_nonzero_rewards = int((batch["reward"] != 0).sum().item())
    print(f"non-zero rewards in batch: {n_nonzero_rewards}")

    wm = WorldModel(cfg)
    trainer = WorldModelTrainer(wm, cfg, device)
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"WorldModel params: {n_params:,}")
    print()

    # Save the ground-truth obs once so we can compare against recon dumps
    save_reconstruction_grid(batch["obs"], batch["obs"],
                             str(out_dir / "ground_truth.png"), n_frames=8)

    print(f"{'step':>5} {'recon':>12} {'kl_raw':>8} {'reward':>10} "
          f"{'rec.std':>9} {'ls.min':>8} {'ls.mean':>8} {'ls.max':>8}")
    for step in range(args.steps + 1):
        losses, info = trainer.train_step(batch)

        if step % 10 == 0:
            recon = info["recon"]
            ls = info["recon_log_std"]
            print(f"{step:5d} {losses['recon']:12.1f} {losses.get('kl_raw', 0):8.3f} "
                  f"{losses['reward']:10.4f} {recon.std().item():9.4f} "
                  f"{ls.min().item():8.3f} {ls.mean().item():8.3f} {ls.max().item():8.3f}")

        if step % args.save_every == 0:
            save_reconstruction_grid(
                batch["obs"], info["recon"],
                str(out_dir / f"recon_step{step:04d}.png"),
                n_frames=8,
            )

    # ── Ablation ────────────────────────────────────────────────────────────
    print("\n=== Ablation: is the decoder actually using (h, z)? ===")
    wm.eval()
    with torch.no_grad():
        embeds = wm.encoder(batch["obs"])
        h_seq, z_seq, _, _ = wm.rssm.observe_sequence(embeds, batch["action"])

        recon_full  = wm.decode(h_seq, z_seq)
        recon_hzero = wm.decode(torch.zeros_like(h_seq), z_seq)
        recon_zzero = wm.decode(h_seq, torch.zeros_like(z_seq))
        recon_both  = wm.decode(torch.zeros_like(h_seq), torch.zeros_like(z_seq))

        def stats(name, t):
            print(f"  {name:12s} std={t.std().item():.4f}  "
                  f"mean={t.mean().item():.4f}  "
                  f"|delta_full|={(t - recon_full).abs().mean().item():.4f}")

        stats("full(h,z)",  recon_full)
        stats("h=0, z",     recon_hzero)
        stats("h, z=0",     recon_zzero)
        stats("h=0, z=0",   recon_both)

        save_reconstruction_grid(batch["obs"], recon_full,
                                 str(out_dir / "ablation_full.png"), n_frames=8)
        save_reconstruction_grid(batch["obs"], recon_hzero,
                                 str(out_dir / "ablation_h_zero.png"), n_frames=8)
        save_reconstruction_grid(batch["obs"], recon_zzero,
                                 str(out_dir / "ablation_z_zero.png"), n_frames=8)
        save_reconstruction_grid(batch["obs"], recon_both,
                                 str(out_dir / "ablation_both_zero.png"), n_frames=8)

    print(f"\nDone. All outputs in {out_dir}/")
    print("Interpretation:")
    print("  - recon_step{N}.png should show ball/paddle appearing by step 200-500.")
    print("  - If ablation_z_zero ≈ ablation_full → decoder ignores z (posterior collapse).")
    print("  - If ablation_both_zero is still a structured image → decoder is just bias.")


if __name__ == "__main__":
    main()
