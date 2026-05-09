"""
JEPA world model training loop with parallel environment collection.
Usage: python train_jepa.py --config configs/pong_jepa.yaml
"""

import argparse
import concurrent.futures
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from envs.wrappers import make_atari_env
from models.actor import Actor
from models.critic import Critic
from training.jepa_world_model import JEPAWorldModel, JEPAWorldModelTrainer
from training.jepa_actor_critic import JEPAActorCriticTrainer
from training.replay_buffer import EpisodeReplayBuffer, AsyncBatchPrefetcher
from utils.logging import Logger


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JEPACollector:
    """Collect from N environments with batched GPU inference.

    Simpler than the RSSM collector: no running state to maintain.
    Just encode the current observation and let the actor choose an action.
    """

    def __init__(self, env_name, n_envs, actor, world_model, device, act_dim,
                 obs_channels=4, seed=42):
        self.envs = [make_atari_env(env_name, n_frames=obs_channels, seed=seed + i)
                     for i in range(n_envs)]
        self.n_envs = n_envs
        self.actor = actor
        self.world_model = world_model  # for encoder + encoder_bn
        self.device = device
        self.act_dim = act_dim
        self._step_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self._episode_data = [{"obs": [], "act": [], "rew": [], "done": []}
                              for _ in range(n_envs)]
        self._obs = []
        self._reset_all()

    def _reset_all(self):
        self._obs = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            self._obs.append(obs)
            self._episode_data[i] = {"obs": [], "act": [], "rew": [], "done": []}

    def _reset_env(self, i):
        obs, _ = self.envs[i].reset()
        self._obs[i] = obs
        self._episode_data[i] = {"obs": [], "act": [], "rew": [], "done": []}

    def collect_steps(self, n_steps, random=False):
        """Collect n_steps total across all envs.

        Returns list of completed episodes as (obs, actions, rewards, dones) tuples.
        """
        completed = []
        steps_collected = 0

        while steps_collected < n_steps:
            obs_batch = torch.tensor(
                np.stack(self._obs), dtype=torch.float32, device=self.device
            )  # (N, C, H, W)

            with torch.no_grad():
                # Use eval mode for BN during collection (stable running stats)
                self.world_model.encoder.eval()
                self.world_model.encoder_bn.eval()
                embeds = self.world_model.encode(obs_batch)  # (N, D)
                self.world_model.encoder.train()
                self.world_model.encoder_bn.train()

                if random or torch.isnan(embeds).any():
                    if not random and torch.isnan(embeds).any():
                        print("WARNING: NaN in embeddings, falling back to random actions")
                    action_indices = np.array([env.action_space.sample()
                                               for env in self.envs])
                    actions_onehot = np.zeros((self.n_envs, self.act_dim), dtype=np.float32)
                    for i, idx in enumerate(action_indices):
                        actions_onehot[i, idx] = 1.0
                else:
                    dist = self.actor(embeds)
                    action_indices_t = dist.sample()
                    # Guard against NaN from actor
                    if torch.isnan(action_indices_t).any():
                        print("WARNING: NaN in actor output, falling back to random actions")
                        action_indices = np.array([env.action_space.sample()
                                                   for env in self.envs])
                    else:
                        action_indices = action_indices_t.cpu().numpy()
                    actions_onehot = np.zeros((self.n_envs, self.act_dim), dtype=np.float32)
                    for i, idx in enumerate(action_indices):
                        actions_onehot[i, idx] = 1.0

            for i in range(self.n_envs):
                try:
                    future = self._step_executor.submit(
                        self.envs[i].step, int(action_indices[i])
                    )
                    obs, reward, terminated, truncated, _ = future.result(timeout=60.0)
                except (concurrent.futures.TimeoutError, TimeoutError):
                    print(f"Env {i} hung on step(), resetting...")
                    self._reset_env(i)
                    continue
                done = terminated or truncated

                self._episode_data[i]["obs"].append(self._obs[i])
                self._episode_data[i]["act"].append(actions_onehot[i])
                self._episode_data[i]["rew"].append(reward)
                self._episode_data[i]["done"].append(float(done))

                steps_collected += 1

                if done:
                    ep = self._episode_data[i]
                    completed.append((
                        np.array(ep["obs"], dtype=np.float32),
                        np.array(ep["act"], dtype=np.float32),
                        np.array(ep["rew"], dtype=np.float32),
                        np.array(ep["done"], dtype=np.float32),
                    ))
                    self._reset_env(i)
                else:
                    self._obs[i] = obs

        return completed, steps_collected

    def flush_partial_episodes(self, min_length=50):
        partial = []
        for i in range(self.n_envs):
            ep = self._episode_data[i]
            if len(ep["obs"]) >= min_length:
                partial.append((
                    np.array(ep["obs"], dtype=np.float32),
                    np.array(ep["act"], dtype=np.float32),
                    np.array(ep["rew"], dtype=np.float32),
                    np.array(ep["done"], dtype=np.float32),
                ))
        return partial

    def close(self):
        for env in self.envs:
            env.close()


def make_batch_actions_onehot(actions_tensor, act_dim):
    if actions_tensor.dim() == 2:
        return F.one_hot(actions_tensor.long(), act_dim).float()
    return actions_tensor


def find_latest_checkpoint(ckpt_dir: Path):
    ckpts = sorted(ckpt_dir.glob("step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    n_envs = cfg.get("n_envs", 8)
    act_dim = cfg["act_dim"]
    embed_dim = cfg["embed_dim"]

    print(f"Config: {args.config}")
    print(f"Env: {cfg['env']} x {n_envs} parallel")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
              if hasattr(torch.cuda.get_device_properties(0), "total_mem")
              else f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Models ──
    wm = JEPAWorldModel(cfg).to(device)
    mlp_units = cfg.get("mlp_units", 400)
    actor = Actor(
        hidden_dim=embed_dim,
        stoch_dim=0,          # JEPA: single embedding, no z
        act_dim=act_dim,
        units=mlp_units,
        discrete=True,
    ).to(device)
    critic_num_bins = cfg.get("critic_num_bins", 0)
    critic = Critic(
        hidden_dim=embed_dim,
        stoch_dim=0,
        units=mlp_units,
        num_bins=critic_num_bins,
        bin_low=cfg.get("critic_bin_low", -3.0),
        bin_high=cfg.get("critic_bin_high", 3.0),
    ).to(device)

    torch.backends.cudnn.benchmark = True

    total_params = sum(p.numel() for p in wm.parameters()) + \
                   sum(p.numel() for p in actor.parameters()) + \
                   sum(p.numel() for p in critic.parameters())
    print(f"Total parameters: {total_params:,}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU memory after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ── Trainers ──
    wm_trainer = JEPAWorldModelTrainer(wm, cfg, device)
    ac_trainer = JEPAActorCriticTrainer(actor, critic, wm, cfg, device)

    # ── Dirs ──
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    runs_dir = Path("runs") / cfg["run_name"]
    runs_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume ──
    global_step = 0
    episode_count = 0
    _resumed_best = float("-inf")
    _resumed_window = []
    if args.resume:
        ckpt_path = find_latest_checkpoint(ckpt_dir)
        if ckpt_path is not None:
            print(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            try:
                wm.load_state_dict(ckpt["world_model"])
                actor.load_state_dict(ckpt["actor"])
                critic.load_state_dict(ckpt["critic"])
                ac_trainer.target_critic.load_state_dict(ckpt["target_critic"])
                wm_trainer.optimizer.load_state_dict(ckpt["wm_optimizer"])
                if "wm_scheduler" in ckpt:
                    wm_trainer.scheduler.load_state_dict(ckpt["wm_scheduler"])
                ac_trainer.actor_opt.load_state_dict(ckpt["actor_optimizer"])
                ac_trainer.critic_opt.load_state_dict(ckpt["critic_optimizer"])
                if "critic_scheduler" in ckpt:
                    ac_trainer.critic_scheduler.load_state_dict(ckpt["critic_scheduler"])
                if "ac_step_count" in ckpt:
                    ac_trainer._ac_step_count = ckpt["ac_step_count"]
                _resumed_best = ckpt.get("best_window_avg", float("-inf"))
                _resumed_window = ckpt.get("reward_window", [])
                global_step = ckpt["global_step"]
                episode_count = ckpt.get("episode_count", 0)
                print(f"Resumed at step {global_step}, episode {episode_count}")
            except RuntimeError as e:
                print(f"Checkpoint incompatible, starting fresh: {e}")
                global_step = 0
                episode_count = 0
            del ckpt
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── Collector ──
    collector = JEPACollector(
        cfg["env"], n_envs, actor, wm, device, act_dim,
        obs_channels=cfg.get("obs_channels", 4), seed=cfg["seed"],
    )

    # ── Replay buffer ──
    buffer = EpisodeReplayBuffer(
        max_total_steps=cfg.get("buffer_max_steps", 2_000_000),
        batch_length=cfg["batch_length"],
    )

    # ── Logger ──
    logger = Logger(cfg, use_wandb=cfg.get("use_wandb", False))

    collect_per_step = cfg.get("collect_per_step", 1000)
    ac_warmup_steps = cfg.get("ac_warmup_steps", 0)
    ac_gate_threshold = cfg.get("ac_gate_threshold", None)  # metric-gated AC start
    if ac_gate_threshold is not None:
        print(f"AC metric gate: waiting for wm/pred < {ac_gate_threshold} (min {ac_warmup_steps} steps)")
        _pred_ema = None
        _ac_gate_passed = False
    elif ac_warmup_steps > 0:
        print(f"AC warmup: training only world model for first {ac_warmup_steps} steps")
        _ac_gate_passed = True
    else:
        _ac_gate_passed = True
    _ac_warmup_logged = False

    # ── Prefill ──
    if args.resume and global_step > 0:
        print(f"Resume mode: collecting {collect_per_step} warmup steps with learned policy...")
        episodes, steps = collector.collect_steps(collect_per_step, random=False)
        for obs, actions, rewards, dones in episodes:
            buffer.add_episode(obs, actions, rewards, dones)
        for obs, actions, rewards, dones in collector.flush_partial_episodes(cfg["batch_length"]):
            buffer.add_episode(obs, actions, rewards, dones)
        print(f"Warmup done: {steps} steps, {len(buffer)} episodes in buffer")
    else:
        print(f"Prefilling {cfg['prefill_steps']} steps with random actions ({n_envs} envs)...")
        prefill_steps = 0
        while prefill_steps < cfg["prefill_steps"]:
            episodes, steps = collector.collect_steps(
                min(cfg["prefill_steps"] - prefill_steps + 2000, cfg["prefill_steps"]),
                random=True,
            )
            for obs, actions, rewards, dones in episodes:
                buffer.add_episode(obs, actions, rewards, dones)
            prefill_steps += steps
        for obs, actions, rewards, dones in collector.flush_partial_episodes(cfg["batch_length"]):
            buffer.add_episode(obs, actions, rewards, dones)
        collector._reset_all()
        print(f"Prefilled {prefill_steps} steps in {len(buffer)} episodes")

    # ── Main loop ──
    total_steps = cfg["total_steps"]
    train_ratio = cfg.get("train_ratio", 1.0)

    # ── Reward window tracking (catch breakthroughs) ──
    reward_window_size = cfg.get("reward_window_size", 50)
    reward_window = deque(maxlen=reward_window_size)
    best_window_avg = float("-inf")
    best_window_min_eps = cfg.get("best_window_min_eps", 30)
    if args.resume and _resumed_best != float("-inf"):
        best_window_avg = _resumed_best
        for r in _resumed_window[-reward_window_size:]:
            reward_window.append(r)
        print(f"Restored: best_window_avg={best_window_avg:.2f}, window_size={len(reward_window)}")

    def save_checkpoint(tag=None, fixed_name=None):
        if fixed_name is not None:
            path = ckpt_dir / fixed_name
        else:
            name = f"step_{global_step}.pt" if tag is None else f"step_{global_step}_{tag}.pt"
            path = ckpt_dir / name
        torch.save({
            "global_step": global_step,
            "episode_count": episode_count,
            "world_model": wm.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "target_critic": ac_trainer.target_critic.state_dict(),
            "wm_optimizer": wm_trainer.optimizer.state_dict(),
            "wm_scheduler": wm_trainer.scheduler.state_dict(),
            "actor_optimizer": ac_trainer.actor_opt.state_dict(),
            "critic_optimizer": ac_trainer.critic_opt.state_dict(),
            "critic_scheduler": ac_trainer.critic_scheduler.state_dict(),
            "ac_step_count": ac_trainer._ac_step_count,
            "best_window_avg": best_window_avg,
            "reward_window": list(reward_window),
        }, path)
        print(f"Saved checkpoint: {path}")

    crash_count = 0
    max_crashes = 5

    while global_step < total_steps:
      try:
        # ── Collect phase ──
        episodes, steps = collector.collect_steps(collect_per_step, random=False)
        global_step += steps

        ep_rewards = []
        for obs, actions, rewards, dones in episodes:
            buffer.add_episode(obs, actions, rewards, dones)
            ep_rewards.append(float(rewards.sum()))
            episode_count += 1
            logger.log_episode(rewards.sum(), len(rewards), step=global_step)

        # Update reward window and check for new best
        for r in ep_rewards:
            reward_window.append(r)

        if len(reward_window) >= 10:
            window_avg = float(np.mean(reward_window))
            window_max = float(np.max(reward_window))
            window_min = float(np.min(reward_window))
            above_zero = sum(1 for r in reward_window if r > 0)
            above_5 = sum(1 for r in reward_window if r > 5)
            # Save best.pt only when window is full (avoids spam during early fill)
            # and improvement is meaningful (> 0.25 above previous best)
            new_best_flag = 0
            if (len(reward_window) >= best_window_min_eps
                    and window_avg > best_window_avg + 0.25):
                best_window_avg = window_avg
                save_checkpoint(fixed_name="best.pt")
                new_best_flag = 1
                print(f"NEW BEST: window_avg={window_avg:.2f} at step {global_step} "
                      f"(max={window_max:.0f}, above0={above_zero}/{len(reward_window)})")

            window_metrics = {
                "episode/window_avg": window_avg,
                "episode/window_max": window_max,
                "episode/window_min": window_min,
                "episode/above_zero_count": above_zero,
                "episode/above_5_count": above_5,
                "episode/window_size": len(reward_window),
                "episode/best_window_avg": (
                    best_window_avg if best_window_avg != float("-inf") else 0.0
                ),
                "episode/new_best_event": new_best_flag,
            }
            logger.log_step(window_metrics, step=global_step)

        # ── Train phase ──
        n_train = max(1, int(steps * train_ratio))
        prefetcher = AsyncBatchPrefetcher(buffer, cfg["batch_size"], act_dim, device)

        try:
            accumulated = {}
            for _ in range(n_train):
                batch = prefetcher.get()

                wm_losses, wm_info = wm_trainer.train_step(batch)

                # ── AC gating: metric-based or fixed warmup ──
                ac_ready = global_step >= ac_warmup_steps
                if ac_ready and ac_gate_threshold is not None and not _ac_gate_passed:
                    pred_val = wm_losses.get("pred", 1.0)
                    if _pred_ema is None:
                        _pred_ema = pred_val
                    else:
                        _pred_ema = 0.99 * _pred_ema + 0.01 * pred_val
                    if _pred_ema < ac_gate_threshold:
                        _ac_gate_passed = True
                        print(f"AC gate passed at step {global_step}: wm/pred EMA = {_pred_ema:.6f} < {ac_gate_threshold}")
                    else:
                        ac_ready = False

                if ac_ready and _ac_gate_passed:
                    if not _ac_warmup_logged:
                        print(f"AC training started at step {global_step}")
                        _ac_warmup_logged = True
                    ac_losses = ac_trainer.train_step(
                        wm_info["emb_seq"], dones=wm_info.get("dones"),
                        real_rewards=batch["reward"].to(device),
                        global_step=global_step,
                    )
                else:
                    ac_losses = {}

                # Extract inline diagnostics from world model info
                diag = wm_info.get("diagnostics", {})

                all_losses = {**{f"wm/{k}": v for k, v in wm_losses.items()},
                              **{f"ac/{k}": v for k, v in ac_losses.items()},
                              **{f"diag/{k}": v for k, v in diag.items()}}
                for k, v in all_losses.items():
                    accumulated.setdefault(k, []).append(v)

            avg_losses = {k: sum(v) / len(v) for k, v in accumulated.items()}
            logger.log_step(avg_losses, step=global_step)
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM at step {global_step}: {e}")
            traceback.print_exc()
            print("Clearing CUDA cache and continuing...")
            torch.cuda.empty_cache()
        finally:
            prefetcher.stop()

        # ── Log ──
        if ep_rewards:
            avg_reward = np.mean(ep_rewards)
            status = {
                "avg_reward": avg_reward,
                "episodes": episode_count,
                "n_eps": len(episodes),
                "buffer": buffer.total_steps,
                "train_steps": n_train,
            }
            if len(reward_window) >= 10:
                status["w_avg"] = float(np.mean(reward_window))
                status["w_best"] = best_window_avg if best_window_avg != float("-inf") else 0.0
            logger.print_status(global_step, status)

        # ── Checkpoint ──
        if global_step % cfg.get("checkpoint_every", 10000) < collect_per_step + 1000:
            save_checkpoint()

        crash_count = 0

      except KeyboardInterrupt:
        print("\nInterrupted by user.")
        save_checkpoint(tag="interrupted")
        break
      except Exception as e:
        crash_count += 1
        print(f"\n{'='*60}")
        print(f"CRASH #{crash_count} at step {global_step}: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        save_checkpoint(tag="crash")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if crash_count >= max_crashes:
            print(f"Hit {max_crashes} consecutive crashes, giving up.")
            break
        print("Recovering and continuing...")

    collector.close()
    logger.close()
    print(f"Training complete. {global_step} steps, {episode_count} episodes.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"FATAL ERROR: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        import sys
        sys.exit(1)
