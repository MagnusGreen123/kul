"""
Dreamer main training loop.
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
from training.replay_buffer import EpisodeReplayBuffer
from utils.logging import Logger
from utils.visualization import save_reconstruction_grid, save_imagination_gif


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_episode(env, actor, rssm, device, act_dim, random: bool = False):
    """Collect one episode.

    Args:
        env: gymnasium environment (pixel-based, wrapped)
        actor: Actor network (unused if random=True)
        rssm: RSSM for maintaining state during collection
        device: torch device
        act_dim: action dimension
        random: if True, take random actions (for prefill)

    Returns:
        obs_list, action_list, reward_list, done_list as numpy arrays
    """
    obs, _ = env.reset()
    h, z = rssm.initial_state(1, device)

    obs_list, action_list, reward_list, done_list = [], [], [], []

    while True:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            embed = None  # We don't observe during collection to keep it simple
            # Use prior for state tracking (no encoder needed for action selection)
            if random:
                action_idx = env.action_space.sample()
                action_onehot = np.zeros(act_dim, dtype=np.float32)
                action_onehot[action_idx] = 1.0
            else:
                # Encode current obs and update state
                from models.encoder import ConvEncoder
                action_onehot_t, _ = actor.get_action(h.squeeze(0), z.squeeze(0))
                action_onehot = action_onehot_t.cpu().numpy()
                action_idx = action_onehot.argmax()

        next_obs, reward, terminated, truncated, _ = env.step(action_idx)
        done = terminated or truncated

        obs_list.append(obs)
        action_list.append(action_onehot)
        reward_list.append(reward)
        done_list.append(float(done))

        # Update RSSM state for next step (using prior only, no encoder)
        with torch.no_grad():
            action_t = torch.tensor(action_onehot, dtype=torch.float32, device=device).unsqueeze(0)
            h, z, _ = rssm.imagine_step(h, z, action_t)

        obs = next_obs
        if done:
            break

    return (
        np.array(obs_list, dtype=np.float32),
        np.array(action_list, dtype=np.float32),
        np.array(reward_list, dtype=np.float32),
        np.array(done_list, dtype=np.float32),
    )


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

    print(f"Config: {args.config}")
    print(f"Env: {cfg['env']}")
    print(f"Device: {device}")

    # ── Environment ──
    env = make_atari_env(cfg["env"], n_frames=cfg.get("obs_channels", 4), seed=cfg["seed"])
    act_dim = cfg["act_dim"]

    # ── Models ──
    wm = WorldModel(cfg).to(device)
    actor = Actor(
        hidden_dim=cfg["hidden_dim"],
        stoch_dim=cfg["stoch_dim"],
        act_dim=act_dim,
        discrete=True,
    ).to(device)
    critic = Critic(
        hidden_dim=cfg["hidden_dim"],
        stoch_dim=cfg["stoch_dim"],
    ).to(device)

    # ── Trainers ──
    wm_trainer = WorldModelTrainer(wm, cfg, device)
    ac_trainer = ActorCriticTrainer(actor, critic, wm, cfg, device)

    # ── Replay buffer ──
    buffer = EpisodeReplayBuffer(
        capacity=1000,
        batch_length=cfg["batch_length"],
    )

    # ── Logger ──
    logger = Logger(cfg, use_wandb=cfg.get("use_wandb", False))

    # ── Checkpoint dir ──
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    # ── Prefill with random episodes ──
    print(f"Prefilling {cfg['prefill_steps']} steps with random actions...")
    prefill_steps = 0
    while prefill_steps < cfg["prefill_steps"]:
        obs, actions, rewards, dones = collect_episode(
            env, actor, wm.rssm, device, act_dim, random=True
        )
        buffer.add_episode(obs, actions, rewards, dones)
        prefill_steps += len(rewards)
    print(f"Prefilled {prefill_steps} steps in {len(buffer)} episodes")

    # ── Main training loop ──
    global_step = 0
    episode_count = 0
    total_steps = cfg["total_steps"]

    while global_step < total_steps:
        # Collect one episode
        obs, actions, rewards, dones = collect_episode(
            env, actor, wm.rssm, device, act_dim, random=False
        )
        buffer.add_episode(obs, actions, rewards, dones)
        ep_reward = rewards.sum()
        ep_length = len(rewards)
        global_step += ep_length
        episode_count += 1

        logger.log_episode(ep_reward, ep_length, step=global_step)

        # Train world model
        if global_step % cfg.get("train_every", 5) == 0:
            for _ in range(cfg.get("train_steps", 1)):
                batch = buffer.sample(cfg["batch_size"], device)

                # Ensure actions are one-hot
                batch["action"] = make_batch_actions_onehot(batch["action"], act_dim)

                # Ensure obs has channel dim (B, T, C, H, W)
                if batch["obs"].dim() == 4:
                    # (B, T, H, W) -> need to handle frame-stacked obs
                    pass  # already (B, T, C, H, W) from wrapper

                wm_losses = wm_trainer.train_step(batch)

                # Get states for actor-critic
                with torch.no_grad():
                    losses_info, info = wm(
                        batch["obs"].to(device),
                        batch["action"].to(device),
                        batch["reward"].to(device),
                    )
                h_seq = info["h_seq"]
                z_seq = info["z_seq"]

                # Train actor-critic
                ac_losses = ac_trainer.train_step(h_seq, z_seq)

                # Log
                all_losses = {**{f"wm/{k}": v for k, v in wm_losses.items()},
                              **{f"ac/{k}": v for k, v in ac_losses.items()}}
                logger.log_step(all_losses, step=global_step)

        # Print status
        if global_step % cfg.get("log_every", 500) < ep_length:
            logger.print_status(global_step, {
                "ep_reward": ep_reward,
                "ep_len": ep_length,
                "episodes": episode_count,
                "buffer": buffer.total_steps,
            })

        # Save checkpoint
        if global_step % cfg.get("checkpoint_every", 10000) < ep_length:
            ckpt_path = ckpt_dir / f"step_{global_step}.pt"
            torch.save({
                "global_step": global_step,
                "world_model": wm.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "wm_optimizer": wm_trainer.optimizer.state_dict(),
                "actor_optimizer": ac_trainer.actor_opt.state_dict(),
                "critic_optimizer": ac_trainer.critic_opt.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # Save visualization
        if global_step % cfg.get("eval_every", 5000) < ep_length:
            try:
                batch = buffer.sample(1, device)
                batch["action"] = make_batch_actions_onehot(batch["action"], act_dim)
                with torch.no_grad():
                    _, info = wm(
                        batch["obs"].to(device),
                        batch["action"].to(device),
                        batch["reward"].to(device),
                    )
                save_reconstruction_grid(
                    batch["obs"].to(device), info["recon"],
                    str(runs_dir / f"recon_step{global_step}.png"),
                )
                save_imagination_gif(
                    wm.decoder, info["h_seq"], info["z_seq"],
                    str(runs_dir / f"imagine_step{global_step}.gif"),
                )
            except Exception as e:
                print(f"Visualization failed: {e}")

    logger.close()
    env.close()
    print(f"Training complete. {global_step} steps, {episode_count} episodes.")


if __name__ == "__main__":
    main()
