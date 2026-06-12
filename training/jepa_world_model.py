"""
JEPA world model: encoder + predictor + SIGReg + reward/continuation heads.

Replaces RSSM + reconstruction loss with:
  - Prediction loss: MSE or BYOL-cosine on EMA target (v18)
  - Rollout loss: K-step prediction through predictor's own outputs
  - SIGReg: enforces isotropic Gaussian embeddings (no stop-gradient/EMA)
  - Reward/continuation prediction from embeddings
  - Inverse dynamics aux head (v18): encoder must encode action-distinguishing
    features (paddle position for Pong)

v18 additions (behind config flags):
  - use_ema_target: BYOL-style EMA target encoder + projection/prediction heads.
    Online encoder gets gradient only through input side; target side is
    detached EMA. Source: SPR (Schwarzer 2020), BYOL (Grill 2020).
  - inv_dyn_weight: inverse dynamics head predicts a_t from (emb_t, emb_{t+1}).
    Source: MuDreamer (2024).
"""

import copy

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
from models.cpc import ActionConditionedCPC


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

        # Predictor: transformer with AdaLN action conditioning.
        # Optional stochastic mode (v17): adds Gaussian (mean, log_var) head
        # so prediction loss switches to NLL and imagination samples noise.
        self.predictor = ActionConditionedPredictor(
            embed_dim=embed_dim,
            act_dim=act_dim,
            depth=cfg.get("pred_depth", 4),
            heads=cfg.get("pred_heads", 4),
            mlp_dim=cfg.get("pred_mlp_dim", 512),
            max_seq_len=cfg.get("batch_length", 30) + 2,
            dropout=cfg.get("pred_dropout", 0.1),
            stochastic=cfg.get("stochastic_predictor", False),
            logvar_init=cfg.get("logvar_init", -2.0),
            logvar_min=cfg.get("logvar_min", -5.0),
            logvar_max=cfg.get("logvar_max", 2.0),
        )
        self.stochastic_predictor = cfg.get("stochastic_predictor", False)
        # When true, the rollout chain feeds samples (with reparameterized
        # noise) back into itself. When false, it feeds the mean.
        self.rollout_sample = cfg.get("rollout_sample", False)

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

        # ── v18: EMA target encoder + BYOL projection/prediction heads ──
        # When use_ema_target is True, prediction loss switches to BYOL-style:
        #   online_pred = prediction_head(projection_head(predictor_output))
        #   target = stop_grad(target_projection_head(target_encoder(o_{t+1})))
        #   loss = 1 - cosine(online_pred, target)
        # The target_encoder and target_projection_head are EMA copies of the
        # online versions, updated in update_target_networks() after each step.
        self.use_ema_target = cfg.get("use_ema_target", False)
        self.target_ema_decay = cfg.get("target_ema_decay", 0.99)
        if self.use_ema_target:
            # BYOL projection head (online, with grad)
            self.online_projection_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            # BYOL prediction head (asymmetric, online side only)
            self.online_prediction_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            # EMA copies (target side, no grad)
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_encoder_bn = copy.deepcopy(self.encoder_bn)
            self.target_projection_head = copy.deepcopy(self.online_projection_head)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)
            for p in self.target_encoder_bn.parameters():
                p.requires_grad_(False)
            for p in self.target_projection_head.parameters():
                p.requires_grad_(False)
        else:
            self.online_projection_head = None
            self.online_prediction_head = None
            self.target_encoder = None
            self.target_encoder_bn = None
            self.target_projection_head = None

        # ── v19/v20: predictor loss mode ──
        # "byol" (v18): predictor trained via projection-space cosine. VERIFIED
        #   DEAD END — raw-space output degenerates (cosine 0.99 -> 0.68),
        #   imagination feeds OOD embeddings to reward/critic, value estimates
        #   flip sign. Kept only for v18 reproducibility.
        # "mse_target" (v19+): predictor trained with MSE against the EMA
        #   target embedding — stays anchored in the space imagination consumes,
        #   with a more stable target than the online encoder.
        # "mse" (v11/v15): MSE against online embeddings (use_ema_target=False).
        if self.use_ema_target:
            self.predictor_loss_mode = cfg.get("predictor_loss", "byol")
        else:
            self.predictor_loss_mode = "mse"
        if self.predictor_loss_mode == "mse_target" and not self.use_ema_target:
            raise ValueError("predictor_loss=mse_target requires use_ema_target=true")

        # ── v19/v20: BYOL as ENCODER-ONLY aux (same-timestep self-distillation)
        # online emb -> projection -> prediction vs stop-grad EMA target proj.
        # Keeps the SPR-style encoder pressure from v18 WITHOUT the predictor
        # in the loop. Reuses the existing projection/prediction heads.
        self.byol_encoder_weight = cfg.get("byol_encoder_weight", 0.0)
        if self.byol_encoder_weight > 0 and not self.use_ema_target:
            raise ValueError("byol_encoder_weight requires use_ema_target=true")

        # ── v20: action-conditioned CPC aux on the encoder ──
        # TWISTER-inspired InfoNCE over K horizons. Targets come from the EMA
        # target encoder (stop-grad). Weight ramps 0 -> cpc_weight over
        # cpc_ramp_steps env steps (review guarantee #4).
        self.cpc_weight = cfg.get("cpc_weight", 0.0)
        self.cpc_ramp_steps = cfg.get("cpc_ramp_steps", 0)
        if self.cpc_weight > 0:
            if not self.use_ema_target:
                raise ValueError("cpc_weight requires use_ema_target=true")
            self.cpc = ActionConditionedCPC(
                embed_dim=embed_dim,
                act_dim=act_dim,
                horizons=tuple(cfg.get("cpc_horizons", [1, 3, 5])),
                proj_dim=cfg.get("cpc_proj_dim", 128),
                hidden_dim=cfg.get("cpc_hidden_dim", 256),
                temperature=cfg.get("cpc_temperature", 0.1),
                var_weight=cfg.get("cpc_var_weight", 1.0),
                max_samples=cfg.get("cpc_max_samples", 1024),
            )
        else:
            self.cpc = None

        # ── v18: Inverse dynamics aux head ──
        # Predicts action a_t from (emb_t, emb_{t+1}). Forces encoder to
        # encode action-distinguishing features. Disabled when weight = 0.
        self.inv_dyn_weight = cfg.get("inv_dyn_weight", 0.0)
        if self.inv_dyn_weight > 0:
            self.inv_dyn_head = nn.Sequential(
                nn.Linear(2 * embed_dim, mlp_units),
                nn.ELU(),
                nn.Linear(mlp_units, mlp_units),
                nn.ELU(),
                nn.Linear(mlp_units, act_dim),
            )
        else:
            self.inv_dyn_head = None

        # Loss weights
        self.pred_weight = cfg.get("pred_weight", 1.0)
        self.rollout_weight = cfg.get("rollout_weight", 0.5)
        self.sigreg_weight = cfg.get("sigreg_weight", 0.1)
        self.reward_weight = cfg.get("reward_weight", 10.0)
        self.use_symlog = cfg.get("use_symlog", True)
        # v19+ reward loss: "sum" = legacy .sum(dim=1) (effective 300x/frame,
        # collapses to constant-0 on sparse rewards), "event" = weighted mean
        # with non-zero-reward frames upweighted by reward_event_weight.
        self.reward_loss_mode = cfg.get("reward_loss_mode", "sum")
        self.reward_event_weight = cfg.get("reward_event_weight", 30.0)

        self.embed_dim = embed_dim
        self.act_dim = act_dim
        self.history_size = cfg.get("history_size", 3)
        self.rollout_steps = cfg.get("rollout_steps", 2)

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

    @torch.no_grad()
    def encode_target(self, obs):
        """Encode using the EMA target encoder. No gradient.

        Returns the same shape as encode(). Caller must ensure use_ema_target.
        """
        x = self.target_encoder(obs)
        has_time = x.dim() == 3
        if has_time:
            B, T, D = x.shape
            x = self.target_encoder_bn(x.reshape(B * T, D)).reshape(B, T, D)
        else:
            x = self.target_encoder_bn(x)
        x = x.clamp(-10, 10)
        return x

    @torch.no_grad()
    def update_target_networks(self):
        """EMA update of target encoder, target BN, target projection head.
        Call once per WM training step. No-op if use_ema_target is False.
        """
        if not self.use_ema_target:
            return
        decay = self.target_ema_decay
        for online_p, target_p in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target_p.data.mul_(decay).add_(online_p.data, alpha=1.0 - decay)
        for online_p, target_p in zip(
            self.encoder_bn.parameters(), self.target_encoder_bn.parameters()
        ):
            target_p.data.mul_(decay).add_(online_p.data, alpha=1.0 - decay)
        # BatchNorm running stats also need to follow online
        self.target_encoder_bn.running_mean.data.mul_(decay).add_(
            self.encoder_bn.running_mean.data, alpha=1.0 - decay
        )
        self.target_encoder_bn.running_var.data.mul_(decay).add_(
            self.encoder_bn.running_var.data, alpha=1.0 - decay
        )
        for online_p, target_p in zip(
            self.online_projection_head.parameters(),
            self.target_projection_head.parameters(),
        ):
            target_p.data.mul_(decay).add_(online_p.data, alpha=1.0 - decay)

    def forward(self, obs, actions, rewards, dones=None, global_step: int = 0):
        """Forward pass for training.

        Args:
            obs:         (B, T, C, H, W)
            actions:     (B, T, act_dim) one-hot
            rewards:     (B, T)
            dones:       (B, T) float32, 1.0 at terminal steps (optional)
            global_step: env-step counter (used for the CPC weight ramp)

        Returns:
            losses dict, extra info dict
        """
        B, T = obs.shape[:2]

        # ── Encode all observations ──
        embeddings = self.encode(obs)  # (B, T, D)

        # ── v18: Compute EMA target embeddings if enabled ──
        # target_emb is the BYOL "target" — what the predictor should match.
        # Stop-gradient: target side never receives gradient.
        if self.use_ema_target:
            target_emb = self.encode_target(obs).detach()        # (B, T, D)
            # Project once for full sequence; reuse for pred + rollout
            # (apply BN equivalent through the LayerNorm in projection head)
            target_proj_seq = self.target_projection_head(
                target_emb.reshape(B * T, self.embed_dim)
            ).reshape(B, T, self.embed_dim).detach()             # (B, T, D)
        else:
            target_emb = None
            target_proj_seq = None

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
        # Three modes:
        #   (a) v18 BYOL: predictor → projection → prediction; target is
        #       EMA target_proj. Loss = 1 - cosine(online_pred, target_proj).
        #   (b) v17 stochastic: predictor returns (mean, log_var). NLL loss.
        #   (c) v11/v15 deterministic MSE: predictor → MSE against online emb.
        # wm/pred metric is always MSE-equivalent (against online or target
        # embedding) so AC gate threshold stays cross-version comparable.
        if self.use_ema_target:
            # Predictor still operates in embedding space — just like v15.
            pred = self.predictor(embeddings[:, :-1], actions[:, :-1])  # (B,T-1,D)
            # MSE-equivalent metric for gate (predictor output vs target emb)
            pred_mse = F.mse_loss(pred, target_emb[:, 1:])
            if self.predictor_loss_mode == "mse_target":
                # v19+: train the predictor IN embedding space. Projection
                # heads are fully decoupled from the predictor path.
                pred_loss = pred_mse
            else:
                # v18 "byol" (kept for reproducibility — verified dead end):
                # project + predict, cosine sim against EMA target
                pred_proj = self.online_projection_head(
                    pred.reshape(B * (T - 1), self.embed_dim)
                )
                pred_proj = self.online_prediction_head(pred_proj)
                pred_proj = pred_proj.reshape(B, T - 1, self.embed_dim)
                target_for_pred = target_proj_seq[:, 1:]  # already detached
                cos_sim = F.cosine_similarity(pred_proj, target_for_pred, dim=-1)
                pred_byol = (1.0 - cos_sim).mean()
                pred_loss = pred_byol
            pred_mean = pred  # for diagnostic cosine
            pred_logvar_mean = None
        elif self.stochastic_predictor:
            pred_target = embeddings[:, 1:]
            pred_mean, pred_logvar = self.predictor(
                embeddings[:, :-1], actions[:, :-1], return_dist=True
            )
            pred_mse = F.mse_loss(pred_mean, pred_target)
            # NLL: 0.5 * ((mean - target)^2 / sigma^2 + log sigma^2)
            pred_nll = 0.5 * (
                (pred_mean - pred_target) ** 2 * (-pred_logvar).exp() + pred_logvar
            ).mean()
            pred_loss = pred_nll
            pred_logvar_mean = pred_logvar.mean().detach()
        else:
            pred_target = embeddings[:, 1:]
            pred = self.predictor(embeddings[:, :-1], actions[:, :-1])
            pred_mse = F.mse_loss(pred, pred_target)
            pred_loss = pred_mse
            pred_mean = pred  # for diagnostic cosine
            pred_logvar_mean = None

        # ── K-step autoregressive rollout loss (single-start) ──
        # Roll out K steps autoregressively from t=0, feeding each
        # prediction back as next-step context. In stochastic mode the
        # chain optionally feeds reparameterized samples (rollout_sample),
        # otherwise feeds the mean. Step loss is NLL when stochastic, MSE
        # otherwise. Logged rollout_loss stays as MSE-equivalent for
        # cross-version comparability.
        HS = self.history_size
        K = self.rollout_steps
        if T >= HS + K and self.rollout_weight > 0 and K > 0:
            emb_buffer = [embeddings[:, t] for t in range(HS)]
            act_buffer = [actions[:, t] for t in range(HS)]

            rollout_terms_train = []
            rollout_terms_mse = []
            for k in range(K):
                ctx_e = torch.stack(emb_buffer[-HS:], dim=1)
                ctx_a = torch.stack(act_buffer[-HS:], dim=1)
                # Online emb target is used for MSE metric and (when not BYOL) for training
                online_target = embeddings[:, HS + k]

                if self.use_ema_target:
                    pred_seq = self.predictor(ctx_e, ctx_a)
                    next_pred_emb = pred_seq[:, -1].clamp(-10, 10)
                    if self.predictor_loss_mode == "mse_target":
                        # v19+: K-step rollout trained in embedding space
                        step_mse = F.mse_loss(next_pred_emb, target_emb[:, HS + k])
                        rollout_terms_train.append(step_mse)
                        rollout_terms_mse.append(step_mse.detach())
                    else:
                        # v18 BYOL rollout (dead end, kept for reproducibility):
                        # predictor output → online proj+pred vs EMA target proj
                        step_proj = self.online_projection_head(next_pred_emb)
                        step_proj = self.online_prediction_head(step_proj)
                        step_target = target_proj_seq[:, HS + k]  # detached
                        cos = F.cosine_similarity(step_proj, step_target, dim=-1)
                        rollout_terms_train.append((1.0 - cos).mean())
                        # MSE-equivalent for metric (vs target encoder embedding)
                        rollout_terms_mse.append(
                            F.mse_loss(next_pred_emb, target_emb[:, HS + k]).detach()
                        )
                    next_pred = next_pred_emb
                elif self.stochastic_predictor:
                    m_seq, lv_seq = self.predictor(ctx_e, ctx_a, return_dist=True)
                    next_mean = m_seq[:, -1].clamp(-10, 10)
                    next_logvar = lv_seq[:, -1]
                    nll = 0.5 * (
                        (next_mean - online_target) ** 2 * (-next_logvar).exp() + next_logvar
                    ).mean()
                    rollout_terms_train.append(nll)
                    rollout_terms_mse.append(
                        F.mse_loss(next_mean, online_target).detach()
                    )
                    if self.rollout_sample:
                        std = (0.5 * next_logvar).exp()
                        next_pred = (next_mean + std * torch.randn_like(next_mean)).clamp(-10, 10)
                    else:
                        next_pred = next_mean
                else:
                    pred_seq = self.predictor(ctx_e, ctx_a)
                    next_pred = pred_seq[:, -1].clamp(-10, 10)
                    step_mse = F.mse_loss(next_pred, online_target)
                    rollout_terms_train.append(step_mse)
                    rollout_terms_mse.append(step_mse.detach())

                emb_buffer.append(next_pred)
                act_buffer.append(actions[:, HS + k])

            rollout_loss = sum(rollout_terms_train) / K
            rollout_loss_mse = sum(rollout_terms_mse) / K
        else:
            rollout_loss = torch.zeros(1, device=obs.device)
            rollout_loss_mse = torch.zeros(1, device=obs.device)

        # ── Reward prediction ──
        # Arrival reward convention: reward at state t is rewards[t-1].
        reward_pred = self.reward_pred(embeddings)  # (B, T)
        shifted_rewards = torch.cat(
            [torch.zeros_like(rewards[:, :1]), rewards[:, :-1]], dim=1
        )
        target_reward = symlog(shifted_rewards) if self.use_symlog else shifted_rewards
        reward_se = F.mse_loss(reward_pred, target_reward, reduction="none")  # (B,T)
        event_mask = (target_reward.abs() > 1e-6).float()
        if self.reward_loss_mode == "event":
            # v19+ fix: the old `.sum(dim=1)` over T=30 with reward_weight=10
            # gave an effective 300x per-frame weight whose optimum on ~0.5%
            # non-zero-reward Pong is the constant 0 (verified: v18 ended with
            # reward_pred_sparsity=0.98, imagined_reward~0). Weighted MEAN with
            # event upweighting makes reward events dominate the loss instead.
            w = 1.0 + self.reward_event_weight * event_mask
            reward_loss = (reward_se * w).sum() / w.sum()
        else:
            # v18 and earlier behavior (kept for reproducibility)
            reward_loss = reward_se.sum(dim=1).mean()
        with torch.no_grad():
            n_events = event_mask.sum()
            reward_event_mse = (
                (reward_se * event_mask).sum() / n_events
                if n_events > 0 else torch.zeros((), device=obs.device)
            )

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

        # ── v18: Inverse dynamics aux head ──
        # Predict a_t from (emb_t, emb_{t+1}). Cross-entropy. Forces encoder
        # to encode action-distinguishing features (paddle position).
        if self.inv_dyn_head is not None and T >= 2:
            inv_in = torch.cat([embeddings[:, :-1], embeddings[:, 1:]], dim=-1)  # (B,T-1,2D)
            inv_logits = self.inv_dyn_head(inv_in.reshape(B * (T - 1), 2 * self.embed_dim))
            # actions are one-hot (B, T, act_dim) — convert to indices
            action_targets = actions[:, :-1].argmax(dim=-1).reshape(B * (T - 1))
            inv_dyn_loss = F.cross_entropy(inv_logits, action_targets)
            with torch.no_grad():
                inv_dyn_acc = (inv_logits.argmax(dim=-1) == action_targets).float().mean()
        else:
            inv_dyn_loss = torch.zeros(1, device=obs.device)
            inv_dyn_acc = torch.zeros(1, device=obs.device)

        # ── v19/v20: BYOL encoder-only aux (same-timestep self-distillation) ──
        # online emb_t -> proj -> pred vs stop-grad EMA target proj_t.
        # The dynamics predictor is NOT involved (v18 lesson).
        if self.byol_encoder_weight > 0:
            flat_online = embeddings.reshape(B * T, self.embed_dim)
            enc_proj = self.online_prediction_head(
                self.online_projection_head(flat_online)
            )
            enc_target = target_proj_seq.reshape(B * T, self.embed_dim)  # detached
            byol_enc_loss = (
                1.0 - F.cosine_similarity(enc_proj, enc_target, dim=-1)
            ).mean()
        else:
            byol_enc_loss = torch.zeros(1, device=obs.device)

        # ── v20: action-conditioned CPC aux on encoder ──
        # Weight ramps 0 -> cpc_weight over cpc_ramp_steps env steps.
        cpc_diag = {}
        if self.cpc is not None:
            if self.cpc_ramp_steps > 0:
                ramp = min(1.0, global_step / self.cpc_ramp_steps)
            else:
                ramp = 1.0
            cpc_scale = self.cpc_weight * ramp
            cpc_loss, cpc_diag = self.cpc(embeddings, target_emb, actions)
        else:
            cpc_scale = 0.0
            cpc_loss = torch.zeros(1, device=obs.device)

        # ── Total loss ──
        total_loss = (
            self.pred_weight * pred_loss
            + self.rollout_weight * rollout_loss
            + self.sigreg_weight * sigreg_loss
            + self.reward_weight * reward_loss
            + cont_loss
            + self.aux_recon_weight * aux_recon_loss
            + self.inv_dyn_weight * inv_dyn_loss
            + self.byol_encoder_weight * byol_enc_loss
            + cpc_scale * cpc_loss
        )

        # Logged "pred" / "rollout" stay MSE-equivalent so wm/pred is comparable
        # across deterministic, stochastic, and BYOL versions (gate threshold,
        # plots). Training uses pred_loss / rollout_loss which equal:
        #   - BYOL cosine loss (v18) when use_ema_target
        #   - NLL when stochastic_predictor
        #   - MSE otherwise
        if self.use_ema_target or self.stochastic_predictor:
            pred_metric = pred_mse
            rollout_metric = rollout_loss_mse
        else:
            pred_metric = pred_loss
            rollout_metric = rollout_loss
        losses = {
            "total": total_loss,
            "pred": pred_metric,
            "rollout": rollout_metric,
            "sigreg": sigreg_loss,
            "reward": reward_loss,
            "cont": cont_loss,
            "aux_recon": aux_recon_loss,
            "inv_dyn": inv_dyn_loss,
        }
        if self.stochastic_predictor:
            losses["pred_nll"] = pred_loss.detach()
            losses["rollout_nll"] = rollout_loss.detach() if torch.is_tensor(rollout_loss) else rollout_loss
        if self.use_ema_target and self.predictor_loss_mode == "byol":
            losses["pred_byol"] = pred_loss.detach()
            losses["rollout_byol"] = rollout_loss.detach() if torch.is_tensor(rollout_loss) else rollout_loss
        if self.byol_encoder_weight > 0:
            losses["byol_enc"] = byol_enc_loss.detach()
        if self.cpc is not None:
            losses["cpc"] = cpc_loss.detach()
        # ── Inline diagnostics (cheap stats on existing tensors) ──
        with torch.no_grad():
            # pred_cosine compares predictor output to actual next embedding
            # (online encoder for non-BYOL modes, target encoder for BYOL)
            cos_target = (
                target_emb[:, 1:] if self.use_ema_target else embeddings[:, 1:]
            )
            diag = {
                "emb_mean": embeddings.mean().item(),
                "emb_std": embeddings.std().item(),
                "pred_cosine": F.cosine_similarity(
                    pred_mean, cos_target, dim=-1
                ).mean().item(),
                "reward_pred_sparsity": (
                    reward_pred.abs() < 0.1
                ).float().mean().item(),
                "reward_event_mse": reward_event_mse.item(),
            }
            if pred_logvar_mean is not None:
                diag["pred_logvar_mean"] = pred_logvar_mean.item()
            if self.inv_dyn_head is not None:
                diag["inv_dyn_acc"] = inv_dyn_acc.item()
            if self.cpc is not None:
                diag["cpc_scale"] = cpc_scale
                diag.update(cpc_diag)

            # ── Long-horizon rollout diagnostic ──
            # Mimics imagination procedure: autoregressive predictor with
            # truncated history window (history_size). Logs cosine similarity
            # between predicted and real embeddings at 5/10/15 step horizons.
            # Critical to detect compounding rollout error during training,
            # which the 1-step pred_cosine masks completely.
            HS = self.history_size
            ROLLOUT_STEPS = [5, 10, 15]
            max_horizon = max(ROLLOUT_STEPS)
            # Need at least HS history + max_horizon real future = HS + max_horizon
            if T >= HS + max_horizon:
                emb_list = [embeddings[:, t] for t in range(HS)]  # initial history
                act_list = [actions[:, t] for t in range(HS)]      # paired actions
                rollout_cos = {}
                for step_i in range(max_horizon):
                    ctx_e = torch.stack(emb_list[-HS:], dim=1)   # (B, HS, D)
                    ctx_a = torch.stack(act_list[-HS:], dim=1)   # (B, HS, act_dim)
                    pred_seq = self.predictor(ctx_e, ctx_a)
                    next_pred = pred_seq[:, -1].clamp(-10, 10)   # (B, D)
                    emb_list.append(next_pred)
                    # Use real next action to feed the next rollout step
                    real_next_t = HS + step_i
                    if real_next_t < T:
                        act_list.append(actions[:, real_next_t])

                    # Compare to real embedding at this horizon
                    horizon = step_i + 1  # 1-indexed
                    if horizon in ROLLOUT_STEPS:
                        real_t = HS + step_i  # absolute index of e_{HS+step_i+1-1+1}
                        # next_pred is prediction for embedding at index HS+step_i
                        # (predictor predicts e_{i+1} from context ending at e_i;
                        # last context emb is at index HS+step_i-1, so prediction is for HS+step_i)
                        if real_t < T:
                            real_emb = embeddings[:, real_t]
                            cos = F.cosine_similarity(
                                next_pred, real_emb, dim=-1
                            ).mean().item()
                            rollout_cos[f"rollout_cosine_{horizon}step"] = cos
                diag.update(rollout_cos)

        info = {
            "emb_seq": embeddings.detach(),
            "dones": dones,
            "diagnostics": diag,
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

    @staticmethod
    def _module_grad_norm(module: nn.Module) -> float:
        """L2 norm of all gradients in a module (0.0 if no grads)."""
        sq = 0.0
        for p in module.parameters():
            if p.grad is not None:
                sq += p.grad.pow(2).sum().item()
        return sq ** 0.5

    def train_step(self, batch: dict, global_step: int = 0) -> tuple[dict, dict]:
        obs = batch["obs"].to(self.device)
        actions = batch["action"].to(self.device)
        rewards = batch["reward"].to(self.device)
        dones = batch.get("done")
        if dones is not None:
            dones = dones.to(self.device)

        self.optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=self.use_amp):
            losses, info = self.model(
                obs, actions, rewards, dones=dones, global_step=global_step
            )

        if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
            print(f"WARNING: NaN/Inf in total loss, skipping step. "
                  f"Losses: {{{', '.join(f'{k}={v.item():.4f}' for k, v in losses.items())}}}")
            return {k: v.item() for k, v in losses.items()}, info

        self.scaler.scale(losses["total"]).backward()
        self.scaler.unscale_(self.optimizer)
        # Per-head grad norms BEFORE clipping — task-harmonization telemetry
        # (detects one objective starving the others; cf. HarmonyDream)
        head_grads = {
            "grad_encoder": self._module_grad_norm(self.model.encoder),
            "grad_predictor": self._module_grad_norm(self.model.predictor),
            "grad_reward_head": self._module_grad_norm(self.model.reward_pred),
        }
        if self.model.aux_decoder is not None:
            head_grads["grad_aux_decoder"] = self._module_grad_norm(self.model.aux_decoder)
        if self.model.inv_dyn_head is not None:
            head_grads["grad_inv_dyn"] = self._module_grad_norm(self.model.inv_dyn_head)
        if self.model.cpc is not None:
            head_grads["grad_cpc"] = self._module_grad_norm(self.model.cpc)
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        # v18: EMA update of target encoder/projection (no-op when disabled)
        self.model.update_target_networks()

        info = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v
                for k, v in info.items()}

        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict["grad_norm"] = grad_norm.item()
        loss_dict["lr"] = self.optimizer.param_groups[0]["lr"]
        loss_dict.update(head_grads)
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
