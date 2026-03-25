"""
Logging wrapper for wandb with fallback to console-only logging.
Logs losses, rewards, and reconstruction images.
"""

import time
from pathlib import Path

import numpy as np


class Logger:
    """Unified logger supporting wandb and console output."""

    def __init__(self, cfg: dict, use_wandb: bool = True):
        self.use_wandb = use_wandb
        self.step = 0
        self.episode = 0
        self.start_time = time.time()
        self._wandb = None

        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=cfg.get("wandb_project", "dreamer"),
                    name=cfg.get("run_name", None),
                    config=cfg,
                )
                self._wandb = wandb
            except Exception as e:
                print(f"wandb init failed: {e}, falling back to console")
                self.use_wandb = False

    def log_step(self, metrics: dict, step: int = None):
        """Log scalar metrics.

        Args:
            metrics: dict of metric_name -> value
            step: global step (uses internal counter if None)
        """
        if step is not None:
            self.step = step

        if self.use_wandb and self._wandb:
            self._wandb.log(metrics, step=self.step)

    def log_episode(self, reward: float, length: int, step: int = None):
        """Log episode summary."""
        self.episode += 1
        elapsed = time.time() - self.start_time

        metrics = {
            "episode/reward": reward,
            "episode/length": length,
            "episode/number": self.episode,
            "episode/elapsed_min": elapsed / 60,
        }

        if self.use_wandb and self._wandb:
            self._wandb.log(metrics, step=step or self.step)

    def log_image(self, key: str, image_path: str, step: int = None):
        """Log an image file."""
        if self.use_wandb and self._wandb:
            import wandb
            self._wandb.log(
                {key: wandb.Image(image_path)},
                step=step or self.step,
            )

    def log_video(self, key: str, frames: np.ndarray, fps: int = 10, step: int = None):
        """Log video from numpy array (T, H, W) or (T, H, W, C).

        Args:
            frames: numpy array of video frames
            fps: frames per second
        """
        if self.use_wandb and self._wandb:
            import wandb
            if frames.ndim == 3:
                frames = frames[:, np.newaxis, :, :]  # (T, 1, H, W)
            self._wandb.log(
                {key: wandb.Video(frames, fps=fps, format="gif")},
                step=step or self.step,
            )

    def print_status(self, step: int, metrics: dict):
        """Print a status line to console."""
        elapsed = time.time() - self.start_time
        parts = [f"Step {step:7d}"]
        parts.append(f"({elapsed / 60:.1f}min)")
        for k, v in metrics.items():
            name = k.split("/")[-1]
            parts.append(f"{name}={v:.4f}")
        print(" | ".join(parts))

    def close(self):
        if self.use_wandb and self._wandb:
            self._wandb.finish()


if __name__ == "__main__":
    # Smoke test without wandb
    logger = Logger({"env": "test"}, use_wandb=False)

    logger.log_step({"loss/recon": 0.5, "loss/kl": 32.0}, step=100)
    logger.log_episode(reward=200.0, length=500, step=100)
    logger.print_status(100, {"loss/recon": 0.5, "loss/kl": 32.0, "reward": 200.0})

    print("Smoke test passed!")
