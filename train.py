"""
Dreamer main training loop with parallel environment collection.
Usage: python train.py --config configs/pong.yaml
"""

import argparse
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
from utils.logging import Logger
from utils.visualization import save_reconstruction_grid, save_imagination_gif


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ParallelCollector:
    """Collect from N environments in parallel with batched GPU inference."""

    def __init__(self, env_name, n_envs, actor, encoder, rssm, device, act_dim,
                 obs_channels=4, seed=42):
        self.envs = [make_atari_env(env_name, n_frames=obs_channels, seed=seed + i)
                     for i in range(n_envs)]
        self.n_envs = n_envs
        self.actor = actor
        self.encoder = encoder
        self.rssm = rssm
        self.device = device
        self.act_dim = act_dim

        # Per-env state
        self.h = None
        self.z = None
        self.prev_actions = None
        self._episode_data = [{"obs": [], "act": [], "rew": [], "done": []}
                              for _ in range(n_envs)]
        self._reset_all()

    def _reset_all(self):
        """Reset all envs and RSSM states."""
        self.h, self.z = self.rssm.initial_state(self.n_envs, self.device)
        self.prev_actions = torch.zeros(self.n_envs, self.act_dim, device=self.device)
        self._obs = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset()
            self._obs.append(obs)
            self._episode_data[i] = {"obs": [], "act": [], "rew": [], "done": []}

    def _reset_env(self, i):
        """Reset a single env after episode ends."""
        obs, _ = self.envs[i].reset()
        self._obs[i] = obs
        # Reset RSSM state for this env
        self.h[i] = 0.0
        self.z[i] = 0.0
        self.prev_actions[i] = 0.0
        self._episode_data[i] = {"obs": [], "act": [], "rew": [], "done": []}

    def collect_steps(self, n_steps, random=False):
        """Collect n_steps total across all envs.

        Returns list of completed episodes as (obs, actions, rewards, dones) tuples.
        """
        completed = []
        steps_collected = 0

        while steps_collected < n_steps:
            # Batch all obs -> GPU
            obs_batch = torch.tensor(
                np.stack(self._obs), dtype=torch.float32, device=self.device
            )  # (N, C, H, W)

            with torch.no_grad():
                # Batched encoder + RSSM observe
                embeds = self.encoder(obs_batch)  # (N, embed_dim)
                self.h, self.z, _, _ = self.rssm.observe_step(
                    self.h, self.z, self.prev_actions, embeds
                )

                if random:
                    action_indices = np.array([env.action_space.sample()
                                               for env in self.envs])
                    actions_onehot = np.zeros((self.n_envs, self.act_dim), dtype=np.float32)
                    for i, idx in enumerate(action_indices):
                        actions_onehot[i, idx] = 1.0
                else:
                    # Batched actor inference
                    dist = self.actor(self.h, self.z)
                    action_indices_t = dist.sample()  # (N,)
                    action_indices = action_indices_t.cpu().numpy()
                    actions_onehot = np.zeros((self.n_envs, self.act_dim), dtype=np.float32)
                    for i, idx in enumerate(action_indices):
                        actions_onehot[i, idx] = 1.0

            self.prev_actions = torch.tensor(actions_onehot, dtype=torch.float32,
                                             device=self.device)

            # Step all envs
            for i in range(self.n_envs):
                obs, reward, terminated, truncated, _ = self.envs[i].step(int(action_indices[i]))
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
        """Return any in-progress episodes that are long enough, then reset them."""
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
    """Convert integer actions to one-hot if needed."""
    if actions_tensor.dim() == 2:  # (B, T) integers
        return F.one_hot(actions_tensor.long(), act_dim).float()
    return actions_tensor  # already (B, T, act_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])

    n_envs = cfg.get("n_envs", 8)
    act_dim = cfg["act_dim"]

    print(f"Config: {args.config}")
    print(f"Env: {cfg['env']} x {n_envs} parallel")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Models ──
    wm = WorldModel(cfg).to(device)
    mlp_units = cfg.get("mlp_units", 512)
    actor = Actor(
        hidden_dim=cfg["hidden_dim"],
        stoch_dim=cfg["stoch_dim"],
        act_dim=act_dim,
        units=mlp_units,
        discrete=True,
    ).to(device)
    critic = Critic(
        hidden_dim=cfg["hidden_dim"],
        stoch_dim=cfg["stoch_dim"],
        units=mlp_units,
    ).to(device)

    # Enable cudnn benchmark for faster convolutions
    torch.backends.cudnn.benchmark = True

    total_params = sum(p.numel() for p in wm.parameters()) + \
                   sum(p.numel() for p in actor.parameters()) + \
                   sum(p.numel() for p in critic.parameters())
    print(f"Total parameters: {total_params:,}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU memory after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ── Trainers ──
    wm_trainer = WorldModelTrainer(wm, cfg, device)
    ac_trainer = ActorCriticTrainer(actor, critic, wm, cfg, device)

    # ── Parallel collector ──
    collector = ParallelCollector(
        cfg["env"], n_envs, actor, wm.encoder, wm.rssm, device, act_dim,
        obs_channels=cfg.get("obs_channels", 4), seed=cfg["seed"],
    )

    # ── Replay buffer ──
    buffer = EpisodeReplayBuffer(
        max_total_steps=cfg.get("buffer_max_steps", 2_000_000),
        batch_length=cfg["batch_length"],
    )

    # ── Logger ──
    logger = Logger(cfg, use_wandb=cfg.get("use_wandb", False))

    # ── Dirs ──
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    # ── Prefill ──
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
    # Also flush any in-progress episodes long enough to use
    for obs, actions, rewards, dones in collector.flush_partial_episodes(cfg["batch_length"]):
        buffer.add_episode(obs, actions, rewards, dones)
    collector._reset_all()
    print(f"Prefilled {prefill_steps} steps in {len(buffer)} episodes")

    # ── Main loop ──
    global_step = 0
    episode_count = 0
    total_steps = cfg["total_steps"]
    collect_per_step = cfg.get("collect_per_step", 1000)  # env steps between training
    train_ratio = cfg.get("train_ratio", 1.0)  # gradient steps per env step

    while global_step < total_steps:
        # ── Collect phase ──
        episodes, steps = collector.collect_steps(collect_per_step, random=False)
        global_step += steps

        ep_rewards = []
        for obs, actions, rewards, dones in episodes:
            buffer.add_episode(obs, actions, rewards, dones)
            ep_rewards.append(rewards.sum())
            episode_count += 1
            logger.log_episode(rewards.sum(), len(rewards), step=global_step)

        # ── Train phase with async prefetch ──
        n_train = max(1, int(steps * train_ratio))
        prefetcher = AsyncBatchPrefetcher(buffer, cfg["batch_size"], act_dim, device)

        for _ in range(n_train):
            batch = prefetcher.get()

            # train_step now returns detached states, no redundant forward pass
            wm_losses, wm_info = wm_trainer.train_step(batch)

            ac_losses = ac_trainer.train_step(
                wm_info["h_seq"], wm_info["z_seq"], dones=wm_info.get("dones")
            )

            all_losses = {**{f"wm/{k}": v for k, v in wm_losses.items()},
                          **{f"ac/{k}": v for k, v in ac_losses.items()}}
            logger.log_step(all_losses, step=global_step)

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
            ckpt_path = ckpt_dir / f"step_{global_step}.pt"
            torch.save({
                "global_step": global_step,
                "world_model": wm.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "target_critic": ac_trainer.target_critic.state_dict(),
                "wm_optimizer": wm_trainer.optimizer.state_dict(),
                "actor_optimizer": ac_trainer.actor_opt.state_dict(),
                "critic_optimizer": ac_trainer.critic_opt.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # ── Visualization ──
        if global_step % cfg.get("eval_every", 10000) < collect_per_step + 1000:
            try:
                vis_batch = buffer.sample(1, device)
                vis_batch["action"] = make_batch_actions_onehot(vis_batch["action"], act_dim)
                with torch.no_grad():
                    _, vis_info = wm(
                        vis_batch["obs"],
                        vis_batch["action"],
                        vis_batch["reward"],
                    )
                save_reconstruction_grid(
                    vis_batch["obs"], vis_info["recon"],
                    str(runs_dir / f"recon_step{global_step}.png"),
                )
                save_imagination_gif(
                    lambda h, z: wm.decode(h, z), vis_info["h_seq"], vis_info["z_seq"],
                    str(runs_dir / f"imagine_step{global_step}.gif"),
                )
                del vis_batch, vis_info
            except Exception as e:
                print(f"Visualization failed: {e}")

    collector.close()
    logger.close()
    print(f"Training complete. {global_step} steps, {episode_count} episodes.")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"FATAL ERROR: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        import sys
        sys.exit(1)
