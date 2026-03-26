"""
Sequence replay buffer for Dreamer-style training.
Stores episodes and samples sequential chunks of length `batch_length`.
"""

import numpy as np
import torch


class EpisodeReplayBuffer:
    """Ring buffer of episodes. Samples contiguous sequences for world model training.

    Each episode is stored as a dict of numpy arrays:
        obs:    (T, *obs_shape)
        action: (T,) or (T, act_dim)
        reward: (T,)
        done:   (T,)

    Sampling returns batches of shape (B, L, ...) where L = batch_length.
    """

    def __init__(self, capacity: int, batch_length: int):
        """
        Args:
            capacity: max number of episodes to store
            batch_length: length of sampled sequences
        """
        self.capacity = capacity
        self.batch_length = batch_length
        self.episodes: list[dict[str, np.ndarray]] = []
        self._total_steps = 0

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def __len__(self) -> int:
        return len(self.episodes)

    def add_episode(self, obs: np.ndarray, actions: np.ndarray,
                    rewards: np.ndarray, dones: np.ndarray):
        """Add a complete episode.

        Args:
            obs:     (T, *obs_shape)
            actions: (T,) or (T, act_dim)
            rewards: (T,)
            dones:   (T,)
        """
        ep_len = len(rewards)
        assert len(obs) == ep_len
        assert len(actions) == ep_len
        assert len(dones) == ep_len

        episode = {
            "obs": np.asarray(obs, dtype=np.float32),
            "action": np.asarray(actions),
            "reward": np.asarray(rewards, dtype=np.float32),
            "done": np.asarray(dones, dtype=np.float32),
        }
        self._total_steps += ep_len

        if len(self.episodes) >= self.capacity:
            removed = self.episodes.pop(0)
            self._total_steps -= len(removed["reward"])

        self.episodes.append(episode)

    def sample(self, batch_size: int, device: torch.device = torch.device("cpu")
               ) -> dict[str, torch.Tensor]:
        """Sample a batch of sequential chunks.

        Returns dict with tensors of shape (B, L, ...).
        Episodes shorter than batch_length are skipped.
        """
        # Filter episodes long enough to sample from
        valid = [ep for ep in self.episodes if len(ep["reward"]) >= self.batch_length]
        if not valid:
            raise ValueError(
                f"No episodes with length >= {self.batch_length}. "
                f"Have {len(self.episodes)} episodes, "
                f"longest: {max(len(e['reward']) for e in self.episodes) if self.episodes else 0}"
            )

        batch = {k: [] for k in ("obs", "action", "reward", "done")}
        rng = np.random.default_rng()

        for _ in range(batch_size):
            ep = valid[rng.integers(len(valid))]
            max_start = len(ep["reward"]) - self.batch_length
            start = rng.integers(max_start + 1)
            end = start + self.batch_length

            for key in batch:
                batch[key].append(ep[key][start:end])

        return {
            k: torch.tensor(np.stack(v), device=device, pin_memory=(device.type == "cuda"))
            if device.type != "cuda"
            else torch.tensor(np.stack(v)).pin_memory().to(device, non_blocking=True)
            for k, v in batch.items()
        }


class AsyncBatchPrefetcher:
    """Prefetches the next batch on a background thread while GPU trains.

    Usage:
        prefetcher = AsyncBatchPrefetcher(buffer, batch_size, device)
        for _ in range(n_train_steps):
            batch = prefetcher.get()   # returns pre-loaded batch
            # ... train on batch ...
        prefetcher.stop()
    """

    def __init__(self, buffer: EpisodeReplayBuffer, batch_size: int,
                 act_dim: int, device: torch.device):
        import threading
        import queue
        self.buffer = buffer
        self.batch_size = batch_size
        self.act_dim = act_dim
        self.device = device
        self.queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _make_onehot(self, actions):
        """Convert integer actions to one-hot if needed."""
        if actions.dim() == 2:
            return torch.nn.functional.one_hot(actions.long(), self.act_dim).float()
        return actions

    def _worker(self):
        while not self._stop.is_set():
            try:
                batch = self.buffer.sample(self.batch_size, self.device)
                batch["action"] = self._make_onehot(batch["action"])
                self.queue.put(batch, timeout=1.0)
            except Exception:
                if self._stop.is_set():
                    break

    def get(self):
        return self.queue.get()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5.0)


if __name__ == "__main__":
    # Quick smoke test
    buf = EpisodeReplayBuffer(capacity=100, batch_length=10)

    # Add some fake episodes
    for ep_i in range(5):
        T = np.random.randint(15, 50)
        buf.add_episode(
            obs=np.random.randn(T, 4),
            actions=np.random.randint(0, 2, size=(T,)),
            rewards=np.random.randn(T),
            dones=np.zeros(T),
        )

    batch = buf.sample(batch_size=8)
    for k, v in batch.items():
        print(f"{k:8s}: {v.shape} {v.dtype}")

    print(f"\nEpisodes: {len(buf)}, Total steps: {buf.total_steps}")
    print("Smoke test passed!")
