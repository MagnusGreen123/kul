"""
Gymnasium wrappers for Atari/pixel-based environments.
Resize to 64x64, grayscale, frame-stack (4), normalize to [-0.5, 0.5].
"""

import ale_py  # registers Atari envs with gymnasium
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MaxAndSkipWrapper(gym.Wrapper):
    """Repeat action for `skip` frames, return max of last 2 frames.
    Standard for NoFrameskip Atari envs."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self.skip - 2:
                self._obs_buffer[0] = obs
            if i == self.skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_obs = self._obs_buffer.max(axis=0)
        return max_obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer[0] = obs
        self._obs_buffer[1] = obs
        return obs, info


class GrayscaleWrapper(gym.ObservationWrapper):
    """Convert RGB observation to grayscale. Output shape: (H, W, 1)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs = env.observation_space
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(obs.shape[0], obs.shape[1], 1),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # ITU-R 601-2 luma transform
        gray = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140])
        return gray[:, :, np.newaxis].astype(np.uint8)


class ResizeWrapper(gym.ObservationWrapper):
    """Resize observation to (size, size) using area interpolation."""

    def __init__(self, env: gym.Env, size: int = 64):
        super().__init__(env)
        self.size = size
        channels = env.observation_space.shape[-1]
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(size, size, channels),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        import cv2
        h, w = obs.shape[:2]
        if obs.ndim == 3 and obs.shape[2] == 1:
            obs_2d = obs[:, :, 0]
            resized = cv2.resize(obs_2d, (self.size, self.size), interpolation=cv2.INTER_AREA)
            return resized[:, :, np.newaxis]
        return cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)


class NormalizeWrapper(gym.ObservationWrapper):
    """Normalize uint8 [0, 255] to float32 [-0.5, 0.5]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs = env.observation_space
        self.observation_space = spaces.Box(
            low=-0.5, high=0.5,
            shape=obs.shape,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return (obs.astype(np.float32) / 255.0) - 0.5


class FrameStackWrapper(gym.Wrapper):
    """Stack last `n_frames` observations along channel dim.

    Input obs shape:  (H, W, C)
    Output obs shape: (H, W, C * n_frames)

    For the model we transpose to channels-first (C*n, H, W) in TransposeWrapper.
    """

    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        obs = env.observation_space
        h, w, c = obs.shape
        self.observation_space = spaces.Box(
            low=np.repeat(obs.low, n_frames, axis=-1),
            high=np.repeat(obs.high, n_frames, axis=-1),
            shape=(h, w, c * n_frames),
            dtype=obs.dtype,
        )
        self.frames = np.zeros((h, w, c * n_frames), dtype=obs.dtype)
        self._c = c

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames[:] = 0
        for i in range(self.n_frames):
            self.frames[:, :, i * self._c:(i + 1) * self._c] = obs
        return self.frames.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Shift frames left, add new on the right
        self.frames[:, :, :-self._c] = self.frames[:, :, self._c:]
        self.frames[:, :, -self._c:] = obs
        return self.frames.copy(), reward, terminated, truncated, info


class TransposeWrapper(gym.ObservationWrapper):
    """Transpose (H, W, C) -> (C, H, W) for PyTorch conv layers."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs = env.observation_space
        h, w, c = obs.shape
        self.observation_space = spaces.Box(
            low=obs.low.transpose(2, 0, 1),
            high=obs.high.transpose(2, 0, 1),
            shape=(c, h, w),
            dtype=obs.dtype,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.transpose(obs, (2, 0, 1))


def make_atari_env(env_name: str, size: int = 64, n_frames: int = 4,
                   seed: int = 42) -> gym.Env:
    """Create a fully wrapped Atari environment.

    Pipeline: raw env -> max-and-skip -> grayscale -> resize -> normalize -> frame-stack -> transpose
    Output obs shape: (n_frames, size, size)  dtype: float32, range [-0.5, 0.5]
    """
    env = gym.make(env_name)
    env = MaxAndSkipWrapper(env, skip=4)
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env, size=size)
    env = NormalizeWrapper(env)
    env = FrameStackWrapper(env, n_frames=n_frames)
    env = TransposeWrapper(env)
    return env


if __name__ == "__main__":
    # Smoke test with a simple environment
    try:
        env = make_atari_env("PongNoFrameskip-v4")
        env_name = "PongNoFrameskip-v4"
    except Exception:
        # Fallback: generate random pixel obs to test the wrapper pipeline
        print("Atari not available, testing with dummy pixel obs")

        class DummyPixelEnv(gym.Env):
            """Minimal env that returns random RGB frames."""
            def __init__(self):
                self.observation_space = spaces.Box(0, 255, shape=(210, 160, 3), dtype=np.uint8)
                self.action_space = spaces.Discrete(2)

            def reset(self, **kwargs):
                return self.observation_space.sample(), {}

            def step(self, action):
                return self.observation_space.sample(), 1.0, False, False, {}

        env = DummyPixelEnv()
        env = GrayscaleWrapper(env)
        env = ResizeWrapper(env, size=64)
        env = NormalizeWrapper(env)
        env = FrameStackWrapper(env, n_frames=4)
        env = TransposeWrapper(env)
        env_name = "DummyPixelEnv"

    obs, info = env.reset(seed=42)
    print(f"Env: {env_name}")
    print(f"Obs shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Obs range: [{obs.min():.2f}, {obs.max():.2f}]")

    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    print(f"After step - shape: {obs.shape}, reward: {reward}")
    print("Smoke test passed!")
    env.close()
