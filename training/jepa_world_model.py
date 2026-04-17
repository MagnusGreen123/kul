"""
JEPA world model: encoder + predictor + SIGReg + reward/continuation heads.

Replaces RSSM + reconstruction loss with:
  - Prediction loss: MSE between predicted and actual next-step embeddings
  - Rollout loss: 2-step prediction through predictor's own outputs
  - SIGReg: enforces isotropic Gaussian embeddings (no stop-gradient/EMA)
  - Reward/continuation prediction from embeddings

End-to-end training: gradients flow through encoder from all losses.
No decoder, no KL, no prior/posterior split.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from einops import rearrange

import sys
sys.path.insert(0, ".")

from models.encoder import ConvEncoder
from models.decoder import ConvDecoder
from models.jepa_predictor import ActionConditionedPredictor
from models.sigreg import SIGReg
from models.reward_predictor import RewardPredictor
from models.cont_predictor import ContPredictor


def symlog(x: torch.Tensor) -> torch.Tensor:
    """DreamerV3 symlog transform: sign(x) * ln(|x| + 1)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


class JEPAWorldModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        obs_channels = cfg.get("obs_channels", 4)
        embed_dim = cfg.get("embed_dim", 256)
        act_dim = cfg.get("act_dim", 6)
        cnn_depth = cfg.get("cnn_depth", 48)
        mlp_units = cfg.get("mlp_units", 400)

        # Encoder: CNN + BatchNorm projector
        self.encoder = ConvEncoder(
            in_channels=obs_channels, latent_dim=embed_dim, depth=cnn_depth
        )
        self.encoder_bn = nn.BatchNorm1d(embed_dim)

        # Predictor: transformer with AdaLN action conditioning
        self.predictor = ActionConditionedPredictor(
            embed_dim=embed_dim,
            act_dim=act_dim,
            depth=cfg.get("pred_depth", 4),
            heads=cfg.get("pred_heads", 4),
            mlp_dim=cfg.get("pred_mlp_dim", 512),
            max_seq_len=cfg.get("batch_length", 30) + 2,
            dropout=cfg.get("pred_dropout", 0.1),
        )

        # SIGReg regularizer
        self.sigreg = SIGReg(
            knots=cfg.get("sigreg_knots", 17),
            num_proj=cfg.get("sigreg_projections", 1024),
        )

        # Reward and continuation heads (stoch_dim=0: input is just embed_dim)
        self.reward_pred = RewardPredictor(
            hidden_dim=embed_dim, stoch_dim=0, units=mlp_units
        )
        self.cont_pred = ContPredictor(
            hidden_dim=embed_dim, stoch_dim=0, units=mlp_units
        )

        # Auxiliary decoder — forces encoder to preserve pixel-level detail
        # (especially foreground objects like ball/paddle via fg_weight)
        self.aux_recon_weight = cfg.get("aux_recon_weight", 0.0)
        if self.aux_recon_weight > 0:
            self.aux_decoder = ConvDecoder(
                latent_dim=embed_dim, out_channels=obs_channels,
                depth=cfg.get("aux_decoder_depth", 32),
            )
            self.fg_weight = cfg.get("fg_weight", 50.0)
        else:
            self.aux_decoder = None
            self.fg_weight = 0.0

        # Loss weights
        self.pred_weight = cfg.get("pred_weight", 1.0)
        self.rollout_weight = cfg.get("rollout_weight", 0.5)
        self.sigreg_weight = cfg.get("sigreg_weight", 0.1)
        self.reward_weight = cfg.get("reward_weight", 10.0)
        self.use_symlog = cfg.get("use_symlog", True)

        self.embed_dim = embed_dim

    def encode(self, obs):
        """Encode observations to JEPA embeddings.

        Args:
            obs: (B, C, H, W) or (B, T, C, H, W)

        Returns:
            (B, D) or (B, T, D) embeddings with BatchNorm applied.
        """
        x = self.encoder(obs)  # (B, D) or (B, T, D)
        has_time = x.dim() == 3
        if has_time:
            B, T, D = x.shape
            x = self.encoder_bn(x.reshape(B * T, D)).reshape(B, T, D)
        else:
            x = self.encoder_bn(x)
        # Clamp to prevent runaway magnitudes that lead to NaN downstream
        x = x.clamp(-10, 10)
        return x

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

        # ── Encode all observations ──
        embeddings = self.encode(obs)  # (B, T, D)

        # ── SIGReg: enforce N(0, I) on embeddings ──
        # Subsample to fixed size so the Epps-Pulley * N term doesn't scale
        # with batch_size * batch_length (which was 15360 and made SIGReg
        # dominate 87% of total loss, starving pred/reward gradients).
        flat_emb = embeddings.reshape(B * T, self.embed_dim)
        max_sigreg_samples = 1024
        if flat_emb.shape[0] > max_sigreg_samples:
            idx = torch.randperm(flat_emb.shape[0], device=flat_emb.device)[:max_sigreg_samples]
            flat_emb = flat_emb[idx]
        sigreg_loss = self.sigreg(flat_emb)

        # ── Teacher-forcing prediction ──
        # (e_t, a_t) -> predict e_{t+1}
        # No action shifting: a_t is the action taken AT state t.
        pred = self.predictor(embeddings[:, :-1], actions[:, :-1])  # (B, T-1, D)
        pred_target = embeddings[:, 1:]  # (B, T-1, D)
        # NO detach on target — end-to-end gradient through encoder
        pred_loss = F.mse_loss(pred, pred_target)

        # ── Multi-step rollout loss (2-step) ──
        # Feed teacher-forcing predictions back into predictor.
        # pred[:, i] = ê_{i+1}, paired with actions[:, i+1] = a_{i+1}
        if T > 2 and self.rollout_weight > 0:
            rollout_pred = self.predictor(
                pred[:, :-1].detach() if False else pred[:, :-1],  # keep grad flowing
                actions[:, 1:-1],
            )  # (B, T-2, D)
            rollout_target = embeddings[:, 2:]  # (B, T-2, D)
            rollout_loss = F.mse_loss(rollout_pred, rollout_target)
        else:
            rollout_loss = torch.zeros(1, device=obs.device)

        # ── Reward prediction ──
        # Arrival reward convention: reward at state t is rewards[t-1].
        reward_pred = self.reward_pred(embeddings)  # (B, T)
        shifted_rewards = torch.cat(
            [torch.zeros_like(rewards[:, :1]), rewards[:, :-1]], dim=1
        )
        target_reward = symlog(shifted_rewards) if self.use_symlog else shifted_rewards
        reward_loss = F.mse_loss(
            reward_pred, target_reward, reduction="none"
        ).sum(dim=1).mean()

        # ── Continuation prediction ──
        if dones is not None:
            cont_logit = self.cont_pred(embeddings)  # (B, T)
            cont_target = 1.0 - dones
            cont_loss = F.binary_cross_entropy_with_logits(cont_logit, cont_target)
        else:
            cont_loss = torch.zeros(1, device=obs.device)

        # ── Auxiliary reconstruction (fg-weighted, subsampled for memory) ──
        if self.aux_decoder is not None:
            # Subsample to avoid OOM: decode max 64 random frames from B*T
            BT = B * T
            max_aux_frames = min(BT, 64)
            idx = torch.randperm(BT, device=obs.device)[:max_aux_frames]
            emb_sub = embeddings.reshape(BT, -1)[idx]       # (S, D)
            obs_sub = obs.reshape(BT, *obs.shape[2:])[idx]  # (S, C, H, W)

            recon = self.aux_decoder(emb_sub)  # (S, C, H, W)

            # fg_weight: need consecutive frame pairs for temporal diff
            # Use obs_flat for diff computation on subsampled indices
            obs_flat = obs.reshape(BT, *obs.shape[2:])
            # For each sampled idx, compute |obs[idx] - obs[idx-1]| (wrap to 0 for idx=0)
            prev_idx = (idx - 1).clamp(min=0)
            diff = (obs_flat[idx] - obs_flat[prev_idx]).abs()
            fg_mask = diff.mean(dim=1, keepdim=True)  # (S, 1, H, W)
            weight = 1.0 + self.fg_weight * fg_mask
            sq_err = (obs_sub - recon) ** 2
            aux_recon_loss = (sq_err * weight).mean()
        else:
            aux_recon_loss = torch.zeros(1, device=obs.device)

        # ── Total loss ──
        total_loss = (
            self.pred_weight * pred_loss
            + self.rollout_weight * rollout_loss
            + self.sigreg_weight * sigreg_loss
            + self.reward_weight * reward_loss
            + cont_loss
            + self.aux_recon_weight * aux_recon_loss
        )

        losses = {
            "total": total_loss,
            "pred": pred_loss,
            "rollout": rollout_loss,
            "sigreg": sigreg_loss,
            "reward": reward_loss,
            "cont": cont_loss,
            "aux_recon": aux_recon_loss,
        }
        info = {
            "emb_seq": embeddings.detach(),
            "dones": dones,
        }
        return losses, info


class JEPAWorldModelTrainer:
    """Handles optimization with optional mixed precision."""

    def __init__(self, world_model: JEPAWorldModel, cfg: dict, device: torch.device):
        self.model = world_model.to(device)
        self.device = device
        self.lr = cfg.get("learning_rate", 1e-4)
        self.max_grad_norm = cfg.get("max_grad_norm", 2.0)
        self.use_amp = cfg.get("mixed_precision", False) and device.type == "cuda"

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, eps=1e-5, weight_decay=5e-4
        )
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # LR schedule: linear warmup + cosine decay
        warmup_steps = cfg.get("warmup_steps", 5000)
        total_steps = cfg.get("total_steps", 500000)
        self.warmup_steps = warmup_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: dict) -> tuple[dict, dict]:
        obs = batch["obs"].to(self.device)
        actions = batch["action"].to(self.device)
        rewards = batch["reward"].to(self.device)
        dones = batch.get("done")
        if dones is not None:
            dones = dones.to(self.device)

        self.optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=self.use_amp):
            losses, info = self.model(obs, actions, rewards, dones=dones)

        if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
            print(f"WARNING: NaN/Inf in total loss, skipping step. "
                  f"Losses: {{{', '.join(f'{k}={v.item():.4f}' for k, v in losses.items())}}}")
            return {k: v.item() for k, v in losses.items()}, info

        self.scaler.scale(losses["total"]).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        info = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v
                for k, v in info.items()}

        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict["grad_norm"] = grad_norm.item()
        loss_dict["lr"] = self.optimizer.param_groups[0]["lr"]
        return loss_dict, info


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = {
        "obs_channels": 4,
        "embed_dim": 256,
        "act_dim": 4,
        "cnn_depth": 48,
        "mlp_units": 400,
        "pred_depth": 4,
        "pred_heads": 4,
        "pred_mlp_dim": 512,
        "pred_dropout": 0.1,
        "batch_length": 30,
        "sigreg_weight": 0.1,
        "pred_weight": 1.0,
        "rollout_weight": 0.5,
        "reward_weight": 10.0,
        "use_symlog": True,
        "learning_rate": 1e-4,
        "mixed_precision": device.type == "cuda",
        "max_grad_norm": 10.0,
    }

    wm = JEPAWorldModel(cfg)
    trainer = JEPAWorldModelTrainer(wm, cfg, device)

    B, T, C, H, W = 4, 20, 4, 64, 64
    act_dim = 4
    batch = {
        "obs": torch.randn(B, T, C, H, W),
        "action": F.one_hot(torch.randint(0, act_dim, (B, T)), act_dim).float(),
        "reward": torch.randn(B, T),
        "done": torch.zeros(B, T),
    }

    for step in range(3):
        losses, info = trainer.train_step(batch)
        print(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in losses.items()))

    # Verify encoder gets gradients from both pred_loss and sigreg
    wm.zero_grad()
    with autocast(device_type="cuda", enabled=False):
        losses, _ = wm(
            batch["obs"].to(device), batch["action"].to(device),
            batch["reward"].to(device), batch["done"].to(device),
        )
    losses["total"].backward()
    enc_grad = sum(p.grad.abs().sum().item() for p in wm.encoder.parameters() if p.grad is not None)
    bn_grad = sum(p.grad.abs().sum().item() for p in wm.encoder_bn.parameters() if p.grad is not None)
    pred_grad = sum(p.grad.abs().sum().item() for p in wm.predictor.parameters() if p.grad is not None)
    print(f"\nEncoder grad sum: {enc_grad:.4f}")
    print(f"BN grad sum: {bn_grad:.4f}")
    print(f"Predictor grad sum: {pred_grad:.4f}")
    assert enc_grad > 0, "Encoder must receive gradients"
    assert pred_grad > 0, "Predictor must receive gradients"

    print(f"\nemb_seq shape: {info['emb_seq'].shape}")
    total_params = sum(p.numel() for p in wm.parameters())
    print(f"Total params: {total_params:,}")
    print("Smoke test passed!")
