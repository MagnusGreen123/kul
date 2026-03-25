"""
DQN agent for CartPole-v1.
Verifies training loop fundamentals before moving to Dreamer.
Target: ~200 reward within ~300 episodes.

Usage: python scripts/dqn_cartpole.py
"""

import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ── Hyperparameters ──────────────────────────────────────────────
SEED = 42
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 50_000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 500  # episodes over which epsilon decays linearly
TARGET_UPDATE = 10  # episodes between target net sync
MAX_EPISODES = 500
SOLVE_REWARD = 200.0  # average over 100 episodes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Q-network ────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, next_obs, done = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32, device=DEVICE),
            torch.tensor(act, dtype=torch.long, device=DEVICE),
            torch.tensor(rew, dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(next_obs), dtype=torch.float32, device=DEVICE),
            torch.tensor(done, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buf)


# ── Training ─────────────────────────────────────────────────────
def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    q_net = QNetwork(obs_dim, act_dim).to(DEVICE)
    target_net = QNetwork(obs_dim, act_dim).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)
    reward_history = deque(maxlen=100)

    for ep in range(1, MAX_EPISODES + 1):
        # Linear epsilon decay
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * ep / EPS_DECAY)

        obs, _ = env.reset(seed=SEED + ep)
        ep_reward = 0.0

        while True:
            # Epsilon-greedy action
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = q_net(torch.tensor(obs, dtype=torch.float32, device=DEVICE))
                    action = q_vals.argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            ep_reward += reward

            # Update Q-network
            if len(buffer) >= BATCH_SIZE:
                b_obs, b_act, b_rew, b_next, b_done = buffer.sample(BATCH_SIZE)

                q_values = q_net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    max_next_q = target_net(b_next).max(dim=1).values
                    target = b_rew + GAMMA * max_next_q * (1.0 - b_done)

                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        reward_history.append(ep_reward)
        avg = np.mean(reward_history)

        # Sync target network
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

        if ep % 10 == 0:
            print(f"Ep {ep:4d} | reward {ep_reward:6.1f} | avg100 {avg:6.1f} | eps {eps:.3f}")

        if avg >= SOLVE_REWARD and len(reward_history) == 100:
            print(f"\nSolved at episode {ep}! avg100 = {avg:.1f}")
            break

    env.close()
    print("Done.")


if __name__ == "__main__":
    train()
