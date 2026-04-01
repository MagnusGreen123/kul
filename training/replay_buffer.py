"""
Sequence replay buffer for Dreamer-style training.
Stores episodes and samples sequential chunks of length `batch_length`.
Observations are stored as uint8 to save ~4x memory.
"""

import collections
import threading

import numpy as np
import torch


class EpisodeReplayBuffer:
    """Ring buffer of episodes capped by total transition count.

    Observations are compressed to uint8 [0, 255] on insert and
    converted back to float32 [-0.5, 0.5] during sampling (~4x RAM savings).

    Each episode is stored as a dict of numpy arrays:
        obs:    (T, *obs_shape) uint8
        action: (T,) or (T, act_dim) float32
        reward: (T,) float32
        done:   (T,) float32

    Sampling returns batches of shape (B, L, ...) where L = batch_length.
    """

    def __init__(self, max_total_steps: int, batch_length: int):
        """
        Args:
            max_total_steps: max total transitions across all episodes
            batch_length: length of sampled sequences
        """
        self.max_total_steps = max_total_steps
        self.batch_length = batch_length
        self.episodes: collections.deque[dict[str, np.ndarray]] = collections.deque()
        self._total_steps = 0
        self._lock = threading.Lock()

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def __len__(self) -> int:
        return len(self.episodes)

    def add_episode(self, obs: np.ndarray, actions: np.ndarray,
                    rewards: np.ndarray, dones: np.ndarray):
        """Add a complete episode.

        Args:
            obs:     (T, *obs_shape) — float32 [-0.5, 0.5] or uint8 [0, 255]
            actions: (T,) or (T, act_dim)
            rewards: (T,)
            dones:   (T,)
        """
        ep_len = len(rewards)
        assert len(obs) == ep_len
        assert len(actions) == ep_len
        assert len(dones) == ep_len

        # Compress float32 obs to uint8 for storage
        if obs.dtype in (np.float32, np.float64):
            obs_store = np.clip((obs + 0.5) * 255, 0, 255).astype(np.uint8)
        else:
            obs_store = np.asarray(obs, dtype=np.uint8)

        episode = {
            "obs": obs_store,
            "action": np.asarray(actions, dtype=np.float32),
            "reward": np.asarray(rewards, dtype=np.float32),
            "done": np.asarray(dones, dtype=np.float32),
        }
        with self._lock:
            self._total_steps += ep_len
            self.episodes.append(episode)

            # Evict oldest episodes until under step budget
            while self._total_steps > self.max_total_steps and len(self.episodes) > 1:
                removed = self.episodes.popleft()
                self._total_steps -= len(removed["reward"])

    def sample(self, batch_size: int, device: torch.device = torch.device("cpu")
               ) -> dict[str, torch.Tensor]:
        """Sample a batch of sequential chunks.

        Returns dict with tensors of shape (B, L, ...).
        Episodes shorter than batch_length are skipped.
        """
        with self._lock:
            valid = [ep for ep in self.episodes if len(ep["reward"]) >= self.batch_length]
            if not valid:
                raise ValueError(
                    f"No episodes with length >= {self.batch_length}. "
                    f"Have {len(self.episodes)} episodes, "
                    f"longest: {max(len(e['reward']) for e in self.episodes) if self.episodes else 0}"
                )

            obs_list, act_list, rew_list, done_list = [], [], [], []
            rng = np.random.default_rng()

            for _ in range(batch_size):
                ep = valid[rng.integers(len(valid))]
                max_start = len(ep["reward"]) - self.batch_length
                start = rng.integers(max_start + 1)
                end = start + self.batch_length

                obs_list.append(ep["obs"][start:end])
                act_list.append(ep["action"][start:end])
                rew_list.append(ep["reward"][start:end])
                done_list.append(ep["done"][start:end])

        # Decompress uint8 obs to float32 [-0.5, 0.5]
        obs_np = np.stack(obs_list).astype(np.float32) / 255.0 - 0.5

        obs_t = torch.from_numpy(obs_np).to(device)
        act_t = torch.from_numpy(np.stack(act_list)).to(device)
        rew_t = torch.from_numpy(np.stack(rew_list)).to(device)
        done_t = torch.from_numpy(np.stack(done_list)).to(device)

        return {"obs": obs_t, "action": act_t, "reward": rew_t, "done": done_t}


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
        import queue as queue_mod
        import threading
        self._queue_mod = queue_mod
        self.buffer = buffer
        self.batch_size = batch_size
        self.act_dim = act_dim
        self.device = device
        self.queue = queue_mod.Queue(maxsize=2)
        self._stop = threading.Event()
        self._error = None
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
            except self._queue_mod.Full:
                continue  # Queue full, just retry
            except Exception as e:
                if not self._stop.is_set():
                    self._error = e
                break  # Stop worker on any error — don't loop and mask it

    def get(self, retries: int = 5, timeout: float = 60.0):
        for attempt in range(retries):
            if self._error is not None:
                err = self._error
                self._error = None
                raise RuntimeError(f"Prefetcher worker failed: {err}") from err
            try:
                return self.queue.get(timeout=timeout)
            except self._queue_mod.Empty:
                # Worker may have died while we were waiting
                if self._error is not None:
                    err = self._error
                    self._error = None
                    raise RuntimeError(f"Prefetcher worker failed: {err}") from err
                if attempt < retries - 1:
                    print(f"Prefetcher slow (attempt {attempt + 1}/{retries}), retrying...")
                    continue
                raise

    def stop(self):
        self._stop.set()
        # Drain queue so worker can exit if blocked on put
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break
        self._thread.join(timeout=5.0)


if __name__ == "__main__":
    # Quick smoke test
    buf = EpisodeReplayBuffer(max_total_steps=10000, batch_length=10)

    # Add some fake episodes
    for ep_i in range(5):
        T = np.random.randint(15, 50)
        buf.add_episode(
            obs=np.random.randn(T, 4).astype(np.float32),
            actions=np.random.randint(0, 2, size=(T,)),
            rewards=np.random.randn(T).astype(np.float32),
            dones=np.zeros(T, dtype=np.float32),
        )

    batch = buf.sample(batch_size=8)
    for k, v in batch.items():
        print(f"{k:8s}: {v.shape} {v.dtype}")

    print(f"\nEpisodes: {len(buf)}, Total steps: {buf.total_steps}")
    print("Smoke test passed!")
