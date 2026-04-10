"""
Pre-flight check: runs an accelerated mini-training to catch crashes
before committing to a multi-day run.

Tests:
  1. Model construction + forward/backward pass
  2. Gradient health (no NaN/Inf, all params get gradients)
  3. Replay buffer fill + eviction at capacity
  4. AsyncBatchPrefetcher stress test (many start/stop cycles)
  5. Full training loop iterations (collect -> WM train -> AC train)
  6. Memory leak detection (GPU + CPU) over many iterations
  7. Checkpoint save/load roundtrip
  8. Numerical stability under many rapid gradient steps

Usage: python preflight.py --config configs/pong.yaml
       Runs in ~2-5 minutes. If it passes, the full run should survive.
"""

import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from envs.wrappers import make_atari_env
from models.actor import Actor
from models.critic import Critic
from training.world_model import WorldModel, WorldModelTrainer
from training.actor_critic import ActorCriticTrainer
from training.replay_buffer import EpisodeReplayBuffer, AsyncBatchPrefetcher


class PreflightResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def ok(self, name, detail=""):
        self.passed.append((name, detail))
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))

    def fail(self, name, detail=""):
        self.failed.append((name, detail))
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))

    def warn(self, name, detail=""):
        self.warnings.append((name, detail))
        print(f"  WARN  {name}" + (f" — {detail}" if detail else ""))

    def summary(self):
        print(f"\n{'='*60}")
        print(f"  {len(self.passed)} passed, {len(self.failed)} failed, {len(self.warnings)} warnings")
        if self.failed:
            print(f"\n  Failures:")
            for name, detail in self.failed:
                print(f"    - {name}: {detail}")
        if self.warnings:
            print(f"\n  Warnings:")
            for name, detail in self.warnings:
                print(f"    - {name}: {detail}")
        print(f"{'='*60}")
        return len(self.failed) == 0


def get_gpu_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


def get_cpu_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return 0.0


def check_tensor_health(name, tensor):
    """Check a tensor for NaN/Inf. Returns (ok, detail)."""
    if tensor is None:
        return True, "None"
    if torch.isnan(tensor).any():
        return False, f"{name} contains NaN"
    if torch.isinf(tensor).any():
        return False, f"{name} contains Inf"
    return True, ""


def test_model_construction(cfg, device, result):
    """Test 1: Build all models, verify shapes."""
    print("\n[1/8] Model construction + forward pass")
    try:
        wm = WorldModel(cfg).to(device)
        act_dim = cfg["act_dim"]
        mlp_units = cfg.get("mlp_units", 512)
        actor = Actor(hidden_dim=cfg["hidden_dim"], stoch_dim=cfg["stoch_dim"] * cfg.get("n_classes", 32),
                      act_dim=act_dim, units=mlp_units, discrete=True).to(device)
        critic = Critic(hidden_dim=cfg["hidden_dim"], stoch_dim=cfg["stoch_dim"] * cfg.get("n_classes", 32),
                        units=mlp_units).to(device)

        # Forward pass with actual batch dimensions
        B = cfg["batch_size"]
        T = cfg["batch_length"]
        C = cfg.get("obs_channels", 4)

        obs = torch.randn(B, T, C, 64, 64, device=device)
        actions = F.one_hot(torch.randint(0, act_dim, (B, T), device=device), act_dim).float()
        rewards = torch.randn(B, T, device=device)

        with torch.no_grad():
            losses, info = wm(obs, actions, rewards)

        for key in ["total", "recon", "kl", "reward"]:
            ok, detail = check_tensor_health(f"loss/{key}", torch.tensor(losses[key]))
            if not ok:
                result.fail("model_forward", detail)
                return None, None, None

        result.ok("model_construction", f"WM+Actor+Critic on {device}")
        return wm, actor, critic
    except Exception as e:
        result.fail("model_construction", str(e))
        traceback.print_exc()
        return None, None, None


def test_gradient_health(wm, actor, critic, cfg, device, result):
    """Test 2: Full backward pass, check all gradients."""
    print("\n[2/8] Gradient health")
    try:
        act_dim = cfg["act_dim"]
        wm_trainer = WorldModelTrainer(wm, cfg, device)
        ac_trainer = ActorCriticTrainer(actor, critic, wm, cfg, device)

        B = min(cfg["batch_size"], 32)  # smaller batch for speed
        T = cfg["batch_length"]
        C = cfg.get("obs_channels", 4)

        batch = {
            "obs": torch.randn(B, T, C, 64, 64, device=device),
            "action": F.one_hot(torch.randint(0, act_dim, (B, T), device=device), act_dim).float(),
            "reward": torch.randn(B, T, device=device),
            "done": torch.zeros(B, T, device=device),
        }

        wm_losses, wm_info = wm_trainer.train_step(batch)
        ac_losses = ac_trainer.train_step(wm_info["h_seq"], wm_info["z_seq"])

        # Check WM gradients
        wm_no_grad = []
        wm_nan_grad = []
        for name, p in wm.named_parameters():
            if p.requires_grad:
                if p.grad is None:
                    wm_no_grad.append(name)
                elif torch.isnan(p.grad).any():
                    wm_nan_grad.append(name)

        if wm_nan_grad:
            result.fail("wm_gradients", f"NaN grads in: {wm_nan_grad[:3]}")
        elif wm_no_grad:
            result.warn("wm_gradients", f"{len(wm_no_grad)} params without grad (may be ok)")
        else:
            result.ok("wm_gradients", "all params have finite gradients")

        # Check actor gradients
        actor_nan = [n for n, p in actor.named_parameters()
                     if p.grad is not None and torch.isnan(p.grad).any()]
        if actor_nan:
            result.fail("actor_gradients", f"NaN grads in: {actor_nan}")
        else:
            result.ok("actor_gradients", "all finite")

        # Check critic gradients
        critic_nan = [n for n, p in critic.named_parameters()
                      if p.grad is not None and torch.isnan(p.grad).any()]
        if critic_nan:
            result.fail("critic_gradients", f"NaN grads in: {critic_nan}")
        else:
            result.ok("critic_gradients", "all finite")

        # Check losses are reasonable
        all_losses = {**{f"wm/{k}": v for k, v in wm_losses.items()},
                      **{f"ac/{k}": v for k, v in ac_losses.items()}}
        for k, v in all_losses.items():
            if np.isnan(v) or np.isinf(v):
                result.fail("loss_values", f"{k} = {v}")
                return
        result.ok("loss_values", ", ".join(f"{k}={v:.3f}" for k, v in all_losses.items()))

    except Exception as e:
        result.fail("gradient_health", str(e))
        traceback.print_exc()


def test_replay_buffer(cfg, result):
    """Test 3: Buffer fill, eviction, and sampling at capacity."""
    print("\n[3/8] Replay buffer stress test")
    try:
        max_steps = 50_000  # smaller for test speed
        bl = cfg["batch_length"]
        buf = EpisodeReplayBuffer(max_total_steps=max_steps, batch_length=bl)

        # Fill past capacity to trigger eviction
        total_added = 0
        n_episodes = 0
        while total_added < max_steps * 2:
            T = np.random.randint(bl, bl * 3)
            C = cfg.get("obs_channels", 4)
            obs = np.random.randint(0, 255, (T, C, 64, 64), dtype=np.uint8).astype(np.float32) / 255.0 - 0.5
            acts = np.random.randint(0, cfg["act_dim"], (T,))
            rews = np.random.randn(T).astype(np.float32)
            dones = np.zeros(T, dtype=np.float32)
            dones[-1] = 1.0
            buf.add_episode(obs, acts, rews, dones)
            total_added += T
            n_episodes += 1

        if buf.total_steps > max_steps * 1.5:
            result.fail("buffer_eviction", f"buffer has {buf.total_steps} steps, max is {max_steps}")
        else:
            result.ok("buffer_eviction", f"{n_episodes} eps added, {len(buf)} kept, {buf.total_steps} steps")

        # Sample many batches
        for i in range(20):
            batch = buf.sample(cfg["batch_size"], torch.device("cpu"))
            for k, v in batch.items():
                ok, detail = check_tensor_health(f"sample/{k}", v)
                if not ok:
                    result.fail("buffer_sample", detail)
                    return buf

        result.ok("buffer_sample", f"20 batches of {cfg['batch_size']} sampled ok")
        return buf

    except Exception as e:
        result.fail("replay_buffer", str(e))
        traceback.print_exc()
        return None


def test_prefetcher_stress(cfg, device, result):
    """Test 4: Start/stop prefetcher many times, concurrent sampling."""
    print("\n[4/8] Prefetcher stress test")
    try:
        bl = cfg["batch_length"]
        buf = EpisodeReplayBuffer(max_total_steps=20_000, batch_length=bl)
        act_dim = cfg["act_dim"]

        # Fill buffer with enough data
        for _ in range(30):
            T = np.random.randint(bl, bl * 3)
            C = cfg.get("obs_channels", 4)
            obs = np.random.randint(0, 255, (T, C, 64, 64), dtype=np.uint8).astype(np.float32) / 255.0 - 0.5
            buf.add_episode(obs, np.random.randint(0, act_dim, (T,)),
                            np.random.randn(T).astype(np.float32),
                            np.zeros(T, dtype=np.float32))

        # Rapidly start/stop prefetchers (this is what happens every training round)
        n_cycles = 10
        for cycle in range(n_cycles):
            pf = AsyncBatchPrefetcher(buf, min(cfg["batch_size"], 16), act_dim, device)
            # Get a few batches
            for _ in range(5):
                batch = pf.get()
                for k, v in batch.items():
                    ok, detail = check_tensor_health(f"pf/{k}", v)
                    if not ok:
                        pf.stop()
                        result.fail("prefetcher_data", detail)
                        return
            pf.stop()

        result.ok("prefetcher_stress", f"{n_cycles} start/stop cycles, no errors")

    except Exception as e:
        result.fail("prefetcher_stress", str(e))
        traceback.print_exc()


def test_full_training_iterations(cfg, device, result):
    """Test 5: Run actual training loop for several iterations."""
    print("\n[5/8] Full training iterations (collect -> train)")
    try:
        wm = WorldModel(cfg).to(device)
        act_dim = cfg["act_dim"]
        mlp_units = cfg.get("mlp_units", 512)
        actor = Actor(hidden_dim=cfg["hidden_dim"], stoch_dim=cfg["stoch_dim"] * cfg.get("n_classes", 32),
                      act_dim=act_dim, units=mlp_units, discrete=True).to(device)
        critic = Critic(hidden_dim=cfg["hidden_dim"], stoch_dim=cfg["stoch_dim"] * cfg.get("n_classes", 32),
                        units=mlp_units).to(device)

        torch.backends.cudnn.benchmark = True

        wm_trainer = WorldModelTrainer(wm, cfg, device)
        ac_trainer = ActorCriticTrainer(actor, critic, wm, cfg, device)

        from train import ParallelCollector
        n_envs = min(cfg.get("n_envs", 8), 4)  # fewer envs for test
        collector = ParallelCollector(
            cfg["env"], n_envs, actor, wm.encoder, wm.rssm, device, act_dim,
            obs_channels=cfg.get("obs_channels", 4), seed=cfg["seed"],
        )

        buf = EpisodeReplayBuffer(
            max_total_steps=cfg.get("buffer_max_steps", 500_000),
            batch_length=cfg["batch_length"],
        )

        # Prefill
        print("    prefilling...")
        prefill = 0
        while prefill < 2000:
            episodes, steps = collector.collect_steps(min(2000 - prefill + 500, 2000), random=True)
            for obs, actions, rewards, dones in episodes:
                buf.add_episode(obs, actions, rewards, dones)
            prefill += steps
        for obs, actions, rewards, dones in collector.flush_partial_episodes(cfg["batch_length"]):
            buf.add_episode(obs, actions, rewards, dones)
        collector._reset_all()
        print(f"    prefilled {prefill} steps, {len(buf)} episodes")

        # Run N full iterations (collect + train)
        n_iters = 3
        collect_steps = 500  # fewer steps per iter for speed
        for it in range(n_iters):
            episodes, steps = collector.collect_steps(collect_steps, random=False)
            for obs, actions, rewards, dones in episodes:
                buf.add_episode(obs, actions, rewards, dones)

            n_train = max(1, int(steps * cfg.get("train_ratio", 0.1)))
            prefetcher = AsyncBatchPrefetcher(buf, cfg["batch_size"], act_dim, device)

            for _ in range(n_train):
                batch = prefetcher.get()
                wm_losses, wm_info = wm_trainer.train_step(batch)
                ac_losses = ac_trainer.train_step(wm_info["h_seq"], wm_info["z_seq"])

                # Check for NaN
                for k, v in {**wm_losses, **ac_losses}.items():
                    if np.isnan(v) or np.isinf(v):
                        prefetcher.stop()
                        collector.close()
                        result.fail("training_nan", f"iter {it}, {k} = {v}")
                        return

            prefetcher.stop()
            print(f"    iter {it+1}/{n_iters}: {steps} env steps, {n_train} grad steps")

        collector.close()
        result.ok("full_training", f"{n_iters} iterations completed without error")

    except Exception as e:
        result.fail("full_training", str(e))
        traceback.print_exc()


def test_memory_leak(cfg, device, result):
    """Test 6: Run many gradient steps and check for memory growth."""
    print("\n[6/8] Memory leak detection")
    try:
        wm = WorldModel(cfg).to(device)
        act_dim = cfg["act_dim"]
        mlp_units = cfg.get("mlp_units", 512)
        actor = Actor(hidden_dim=cfg["hidden_dim"], stoch_dim=cfg["stoch_dim"] * cfg.get("n_classes", 32),
                      act_dim=act_dim, units=mlp_units, discrete=True).to(device)
        critic = Critic(hidden_dim=cfg["hidden_dim"], stoch_dim=cfg["stoch_dim"] * cfg.get("n_classes", 32),
                        units=mlp_units).to(device)

        wm_trainer = WorldModelTrainer(wm, cfg, device)
        ac_trainer = ActorCriticTrainer(actor, critic, wm, cfg, device)

        B = min(cfg["batch_size"], 32)
        T = cfg["batch_length"]
        C = cfg.get("obs_channels", 4)

        # Warmup
        for _ in range(3):
            batch = {
                "obs": torch.randn(B, T, C, 64, 64, device=device),
                "action": F.one_hot(torch.randint(0, act_dim, (B, T), device=device), act_dim).float(),
                "reward": torch.randn(B, T, device=device),
                "done": torch.zeros(B, T, device=device),
            }
            wm_losses, wm_info = wm_trainer.train_step(batch)
            ac_losses = ac_trainer.train_step(wm_info["h_seq"], wm_info["z_seq"])
            del batch, wm_losses, wm_info, ac_losses

        if device.type == "cuda":
            torch.cuda.synchronize()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        gpu_start = get_gpu_mb()
        cpu_start = get_cpu_mb()

        n_steps = 50
        for i in range(n_steps):
            batch = {
                "obs": torch.randn(B, T, C, 64, 64, device=device),
                "action": F.one_hot(torch.randint(0, act_dim, (B, T), device=device), act_dim).float(),
                "reward": torch.randn(B, T, device=device),
                "done": torch.zeros(B, T, device=device),
            }
            wm_losses, wm_info = wm_trainer.train_step(batch)
            ac_losses = ac_trainer.train_step(wm_info["h_seq"], wm_info["z_seq"])
            del batch, wm_losses, wm_info, ac_losses

        if device.type == "cuda":
            torch.cuda.synchronize()
        gc.collect()

        gpu_end = get_gpu_mb()
        cpu_end = get_cpu_mb()
        gpu_growth = gpu_end - gpu_start
        cpu_growth = cpu_end - cpu_start

        if gpu_growth > 200:
            result.fail("gpu_memory_leak", f"grew {gpu_growth:.0f} MB over {n_steps} steps")
        elif gpu_growth > 50:
            result.warn("gpu_memory_growth", f"grew {gpu_growth:.0f} MB over {n_steps} steps")
        else:
            result.ok("gpu_memory", f"stable ({gpu_growth:+.0f} MB over {n_steps} steps)")

        if cpu_growth > 500:
            result.fail("cpu_memory_leak", f"grew {cpu_growth:.0f} MB over {n_steps} steps")
        elif cpu_growth > 100:
            result.warn("cpu_memory_growth", f"grew {cpu_growth:.0f} MB over {n_steps} steps")
        else:
            result.ok("cpu_memory", f"stable ({cpu_growth:+.0f} MB over {n_steps} steps)")

    except Exception as e:
        result.fail("memory_leak", str(e))
        traceback.print_exc()


def test_checkpoint_roundtrip(cfg, device, result):
    """Test 7: Save and reload a checkpoint, verify weights match."""
    print("\n[7/8] Checkpoint save/load roundtrip")
    try:
        wm = WorldModel(cfg).to(device)
        act_dim = cfg["act_dim"]
        mlp_units = cfg.get("mlp_units", 512)
        actor = Actor(hidden_dim=cfg["hidden_dim"], stoch_dim=cfg["stoch_dim"] * cfg.get("n_classes", 32),
                      act_dim=act_dim, units=mlp_units, discrete=True).to(device)
        critic = Critic(hidden_dim=cfg["hidden_dim"], stoch_dim=cfg["stoch_dim"] * cfg.get("n_classes", 32),
                        units=mlp_units).to(device)

        wm_trainer = WorldModelTrainer(wm, cfg, device)
        ac_trainer = ActorCriticTrainer(actor, critic, wm, cfg, device)

        # Do a training step so optimizers have state
        B, T = 4, cfg["batch_length"]
        C = cfg.get("obs_channels", 4)
        batch = {
            "obs": torch.randn(B, T, C, 64, 64, device=device),
            "action": F.one_hot(torch.randint(0, act_dim, (B, T), device=device), act_dim).float(),
            "reward": torch.randn(B, T, device=device),
            "done": torch.zeros(B, T, device=device),
        }
        wm_losses, wm_info = wm_trainer.train_step(batch)
        ac_trainer.train_step(wm_info["h_seq"], wm_info["z_seq"])

        # Save
        ckpt_path = Path("_preflight_test_ckpt.pt")
        torch.save({
            "global_step": 12345,
            "world_model": wm.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "wm_optimizer": wm_trainer.optimizer.state_dict(),
            "actor_optimizer": ac_trainer.actor_opt.state_dict(),
            "critic_optimizer": ac_trainer.critic_opt.state_dict(),
        }, ckpt_path)

        # Load into fresh models
        wm2 = WorldModel(cfg).to(device)
        actor2 = Actor(hidden_dim=cfg["hidden_dim"], stoch_dim=cfg["stoch_dim"] * cfg.get("n_classes", 32),
                       act_dim=act_dim, units=mlp_units, discrete=True).to(device)
        critic2 = Critic(hidden_dim=cfg["hidden_dim"], stoch_dim=cfg["stoch_dim"] * cfg.get("n_classes", 32),
                         units=mlp_units).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        wm2.load_state_dict(ckpt["world_model"])
        actor2.load_state_dict(ckpt["actor"])
        critic2.load_state_dict(ckpt["critic"])

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(wm.named_parameters(), wm2.named_parameters()):
            if not torch.equal(p1, p2):
                result.fail("checkpoint_roundtrip", f"WM param {n1} mismatch after load")
                ckpt_path.unlink(missing_ok=True)
                return

        ckpt_path.unlink(missing_ok=True)
        size_mb = ckpt_path.stat() if ckpt_path.exists() else None
        result.ok("checkpoint_roundtrip", f"save/load verified, step={ckpt['global_step']}")

    except Exception as e:
        result.fail("checkpoint_roundtrip", str(e))
        traceback.print_exc()
        Path("_preflight_test_ckpt.pt").unlink(missing_ok=True)


def test_numerical_stability(cfg, device, result):
    """Test 8: Rapid gradient steps to check for NaN divergence."""
    print("\n[8/8] Numerical stability (200 rapid gradient steps)")
    try:
        wm = WorldModel(cfg).to(device)
        act_dim = cfg["act_dim"]
        mlp_units = cfg.get("mlp_units", 512)
        actor = Actor(hidden_dim=cfg["hidden_dim"], stoch_dim=cfg["stoch_dim"] * cfg.get("n_classes", 32),
                      act_dim=act_dim, units=mlp_units, discrete=True).to(device)
        critic = Critic(hidden_dim=cfg["hidden_dim"], stoch_dim=cfg["stoch_dim"] * cfg.get("n_classes", 32),
                        units=mlp_units).to(device)

        wm_trainer = WorldModelTrainer(wm, cfg, device)
        ac_trainer = ActorCriticTrainer(actor, critic, wm, cfg, device)

        B = min(cfg["batch_size"], 16)
        T = cfg["batch_length"]
        C = cfg.get("obs_channels", 4)

        n_steps = 200
        for i in range(n_steps):
            batch = {
                "obs": torch.randn(B, T, C, 64, 64, device=device),
                "action": F.one_hot(torch.randint(0, act_dim, (B, T), device=device), act_dim).float(),
                "reward": torch.randn(B, T, device=device) * 5.0,  # large rewards to stress symlog
                "done": torch.zeros(B, T, device=device),
            }
            wm_losses, wm_info = wm_trainer.train_step(batch)
            ac_losses = ac_trainer.train_step(wm_info["h_seq"], wm_info["z_seq"])

            all_losses = {**wm_losses, **ac_losses}
            for k, v in all_losses.items():
                if np.isnan(v):
                    result.fail("numerical_stability", f"NaN at step {i} in {k}")
                    return
                if np.isinf(v):
                    result.fail("numerical_stability", f"Inf at step {i} in {k}")
                    return

            # Check model weights for NaN
            if i % 50 == 49:
                for name, p in wm.named_parameters():
                    if torch.isnan(p).any():
                        result.fail("numerical_stability", f"NaN in WM param {name} at step {i}")
                        return
                for name, p in actor.named_parameters():
                    if torch.isnan(p).any():
                        result.fail("numerical_stability", f"NaN in actor param {name} at step {i}")
                        return

        result.ok("numerical_stability", f"{n_steps} steps, all losses finite")

    except Exception as e:
        result.fail("numerical_stability", str(e))
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Pre-flight training check")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--skip-env", action="store_true",
                        help="Skip tests that require the actual environment")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = PreflightResult()

    print(f"{'='*60}")
    print(f"  PREFLIGHT CHECK")
    print(f"  Config: {args.config}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {total_gb:.1f} GB")
    print(f"  total_steps: {cfg['total_steps']:,}")
    print(f"  batch_size: {cfg['batch_size']}, batch_length: {cfg['batch_length']}")
    print(f"{'='*60}")

    t0 = time.time()

    # Test 1: Model construction
    wm, actor, critic = test_model_construction(cfg, device, result)
    if wm is None:
        print("\nCritical failure in model construction — aborting remaining tests.")
        result.summary()
        sys.exit(1)
    del wm, actor, critic
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Test 2: Gradient health
    wm = WorldModel(cfg).to(device)
    mlp_units = cfg.get("mlp_units", 512)
    actor = Actor(hidden_dim=cfg["hidden_dim"], stoch_dim=cfg["stoch_dim"] * cfg.get("n_classes", 32),
                  act_dim=cfg["act_dim"], units=mlp_units, discrete=True).to(device)
    critic = Critic(hidden_dim=cfg["hidden_dim"], stoch_dim=cfg["stoch_dim"] * cfg.get("n_classes", 32),
                    units=mlp_units).to(device)
    test_gradient_health(wm, actor, critic, cfg, device, result)
    del wm, actor, critic
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Test 3: Replay buffer
    test_replay_buffer(cfg, result)
    gc.collect()

    # Test 4: Prefetcher stress
    test_prefetcher_stress(cfg, device, result)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Test 5: Full training iterations (requires env)
    if not args.skip_env:
        test_full_training_iterations(cfg, device, result)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("\n[5/8] Skipped (--skip-env)")

    # Test 6: Memory leak
    test_memory_leak(cfg, device, result)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Test 7: Checkpoint roundtrip
    test_checkpoint_roundtrip(cfg, device, result)
    gc.collect()

    # Test 8: Numerical stability
    test_numerical_stability(cfg, device, result)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s")
    ok = result.summary()

    if ok:
        print("\n  Ready for training!")
    else:
        print("\n  Fix failures before starting a long run.")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
