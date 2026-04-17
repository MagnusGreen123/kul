"""
JEPA world model training loop with parallel environment collection.
Usage: python train_jepa.py --config configs/pong_jepa.yaml
"""

import argparse
import concurrent.futures
import traceback
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
    critic = Critic(
        hidden_dim=embed_dim,
        stoch_dim=0,
        units=mlp_units,
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
    if ac_warmup_steps > 0:
        print(f"AC warmup: training only world model for first {ac_warmup_steps} steps")
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

    def save_checkpoint(tag=None):
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
            ep_rewards.append(rewards.sum())
            episode_count += 1
            logger.log_episode(rewards.sum(), len(rewards), step=global_step)

        # ── Train phase ──
        n_train = max(1, int(steps * train_ratio))
        prefetcher = AsyncBatchPrefetcher(buffer, cfg["batch_size"], act_dim, device)

        try:
            accumulated = {}
            for _ in range(n_train):
                batch = prefetcher.get()

                wm_losses, wm_info = wm_trainer.train_step(batch)

                if global_step >= ac_warmup_steps:
                    if not _ac_warmup_logged and ac_warmup_steps > 0:
                        print(f"AC warmup complete at step {global_step} — starting actor-critic training")
                        _ac_warmup_logged = True
                    ac_losses = ac_trainer.train_step(
                        wm_info["emb_seq"], dones=wm_info.get("dones")
                    )
                else:
                    ac_losses = {}

                all_losses = {**{f"wm/{k}": v for k, v in wm_losses.items()},
                              **{f"ac/{k}": v for k, v in ac_losses.items()}}
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
            logger.print_status(global_step, {
                "avg_reward": avg_reward,
                "episodes": episode_count,
                "n_eps": len(episodes),
                "buffer": buffer.total_steps,
                "train_steps": n_train,
            })

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
