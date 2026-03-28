"""
World model training: encoder + RSSM + decoder + reward predictor.
Losses: reconstruction + KL divergence (with balancing) + reward prediction.
Supports mixed precision training via torch.cuda.amp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

import sys
sys.path.insert(0, ".")

from models.encoder import ConvEncoder
from models.decoder import ConvDecoder
from models.rssm import RSSM, kl_loss
from models.reward_predictor import RewardPredictor
from models.cont_predictor import ContPredictor


def symlog(x: torch.Tensor) -> torch.Tensor:
    """DreamerV3 symlog transform: sign(x) * ln(|x| + 1)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class WorldModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        obs_channels = cfg.get("obs_channels", 4)
        embed_dim = cfg.get("embed_dim", 512)
        hidden_dim = cfg.get("hidden_dim", 256)
        stoch_dim = cfg.get("stoch_dim", 32)
        act_dim = cfg.get("act_dim", 4)
        depth = cfg.get("cnn_depth", 32)

        self.encoder = ConvEncoder(in_channels=obs_channels, latent_dim=embed_dim, depth=depth)
        self.rssm = RSSM(embed_dim=embed_dim, stoch_dim=stoch_dim,
                         hidden_dim=hidden_dim, act_dim=act_dim)
        mlp_units = cfg.get("mlp_units", 256)
        self.reward_pred = RewardPredictor(hidden_dim=hidden_dim, stoch_dim=stoch_dim, units=mlp_units)
        self.cont_pred = ContPredictor(hidden_dim=hidden_dim, stoch_dim=stoch_dim, units=mlp_units)

        # Bottleneck h for decoder: project h down to stoch_dim so that
        # z (stoch_dim) and h_proj (stoch_dim) are equal-sized inputs.
        # This prevents the decoder from ignoring z in favor of h.
        # h keeps full hidden_dim for RSSM, reward predictor, and actor-critic.
        decoder_h_dim = cfg.get("decoder_h_dim", stoch_dim)
        self._decoder_h_proj = nn.Linear(hidden_dim, decoder_h_dim)
        self.decoder = ConvDecoder(latent_dim=decoder_h_dim + stoch_dim, out_channels=obs_channels, depth=depth)

        self._decoder_h_dim = decoder_h_dim
        self.kl_weight = cfg.get("kl_weight", 1.0)
        self.kl_balance = cfg.get("kl_balance", 0.8)
        self.free_bits = cfg.get("free_bits", 1.0)
        self.use_symlog = cfg.get("use_symlog", True)

    def decode(self, h, z):
        """Decode from (h, z) using h bottleneck projection."""
        h_proj = self._decoder_h_proj(h)
        return self.decoder(torch.cat([h_proj, z], dim=-1))

    def forward(self, obs, actions, rewards, dones=None):
        """Forward pass for training.

        Args:
            obs:     (B, T, C, H, W)
            actions: (B, T, act_dim) one-hot
            rewards: (B, T)
            dones:   (B, T) float32, 1.0 at terminal steps (optional)

        Returns:
            losses dict, extra info dict
        """
        B, T = obs.shape[:2]

        # Encode observations
        embeds = self.encoder(obs)  # (B, T, embed_dim)

        # Run RSSM
        h_seq, z_seq, priors, posteriors = self.rssm.observe_sequence(embeds, actions)

        # Decode from (h, z) via h bottleneck
        recon = self.decode(h_seq, z_seq)  # (B, T, C, H, W)

        # Predict rewards
        reward_pred = self.reward_pred(h_seq, z_seq)  # (B, T)

        # Predict continuation
        cont_logit = self.cont_pred(h_seq, z_seq)  # (B, T)

        # ── Losses ──
        # Reconstruction loss (MSE on pixels)
        recon_loss = F.mse_loss(recon, obs)

        # KL divergence with balancing and free bits
        kl = kl_loss(priors, posteriors, free_bits=self.free_bits, balance=self.kl_balance)

        # Reward loss (symlog targets for stability)
        if self.use_symlog:
            reward_loss = F.mse_loss(reward_pred, symlog(rewards))
        else:
            reward_loss = F.mse_loss(reward_pred, rewards)

        # Continuation loss (BCE on logits vs 1 - done)
        if dones is not None:
            cont_target = 1.0 - dones  # 1 = continue, 0 = terminal
            cont_loss = F.binary_cross_entropy_with_logits(cont_logit, cont_target)
        else:
            cont_loss = torch.zeros(1, device=obs.device)

        total_loss = recon_loss + self.kl_weight * kl + reward_loss + cont_loss

        losses = {
            "total": total_loss,
            "recon": recon_loss,
            "kl": kl,
            "reward": reward_loss,
            "cont": cont_loss,
        }
        info = {
            "h_seq": h_seq.detach(),
            "z_seq": z_seq.detach(),
            "recon": recon.detach(),
            "dones": dones,  # pass through for AC terminal filtering
        }
        return losses, info


class WorldModelTrainer:
    """Handles optimization loop with optional mixed precision."""

    def __init__(self, world_model: WorldModel, cfg: dict, device: torch.device):
        self.model = world_model.to(device)
        self.device = device
        self.lr = cfg.get("learning_rate", 3e-4)
        self.max_grad_norm = cfg.get("max_grad_norm", 100.0)
        self.use_amp = cfg.get("mixed_precision", False) and device.type == "cuda"

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

    def train_step(self, batch: dict) -> tuple[dict, dict]:
        """Single training step.

        Args:
            batch: dict with keys obs (B,T,C,H,W), action (B,T,act_dim),
                   reward (B,T), done (B,T)

        Returns:
            (losses_dict, info_dict) — info contains detached h_seq, z_seq
            for actor-critic training without a redundant forward pass.
        """
        obs = batch["obs"].to(self.device)
        actions = batch["action"].to(self.device)
        rewards = batch["reward"].to(self.device)
        dones = batch.get("done")
        if dones is not None:
            dones = dones.to(self.device)

        self.optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=self.use_amp):
            losses, info = self.model(obs, actions, rewards, dones=dones)

        self.scaler.scale(losses["total"]).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Ensure float32 for downstream actor-critic (AMP may produce float16)
        info = {k: v.float() if v.is_floating_point() else v for k, v in info.items()}

        return {k: v.item() for k, v in losses.items()}, info


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = {
        "obs_channels": 4,
        "embed_dim": 512,
        "hidden_dim": 256,
        "stoch_dim": 32,
        "act_dim": 4,
        "cnn_depth": 32,
        "kl_weight": 1.0,
        "kl_balance": 0.8,
        "free_bits": 1.0,
        "use_symlog": True,
        "learning_rate": 3e-4,
        "mixed_precision": device.type == "cuda",
        "max_grad_norm": 100.0,
    }

    wm = WorldModel(cfg)
    trainer = WorldModelTrainer(wm, cfg, device)

    # Fake batch
    B, T, C, H, W = 4, 20, 4, 64, 64
    act_dim = 4
    batch = {
        "obs": torch.randn(B, T, C, H, W),
        "action": F.one_hot(torch.randint(0, act_dim, (B, T)), act_dim).float(),
        "reward": torch.randn(B, T),
        "done": torch.zeros(B, T),
    }

    # Run a few training steps
    for step in range(5):
        losses, info = trainer.train_step(batch)
        print(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in losses.items()))

    print(f"\nTotal params: {sum(p.numel() for p in wm.parameters()):,}")
    print("Smoke test passed!")
