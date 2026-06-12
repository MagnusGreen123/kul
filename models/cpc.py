"""
Action-conditioned Contrastive Predictive Coding (CPC) auxiliary loss.

TWISTER-inspired (ICLR 2025) InfoNCE objective adapted to continuous JEPA
embeddings: for each horizon k, a query head predicts the CPC-space projection
of the state k steps ahead from (emb_t, a_t..a_{t+k-1}); the model must pick
the true future among in-batch negatives.

Why this exists: MSE/reconstruction losses underweight small task-relevant
objects (probe data: ball region MSE 15-20x playfield). InfoNCE only rewards
encoding what *discriminates* futures — static/shared content contributes no
contrastive signal — so the pressure is task-relevant dynamics regardless of
environment (environment-agnostic by design; no Pong-specific constants).

Design decisions (per STRATEGIC_PLAN_V19 Phase 2B guarantees + review):
  - Target branch is the EMA target encoder (stop-grad), satisfying the
    stop-gradient guarantee. The key head q_k is trained online — its
    gradients cannot reach any encoder (input is detached), and InfoNCE's
    uniformity term penalizes key collapse directly.
  - Queries/keys are L2-normalized with fixed temperature. This replaces the
    reviewed norm-coupled tau (open question #1 flagged a positive feedback
    loop); normalization removes the norm degree of freedom entirely.
  - Variance hinge on PRE-normalization projections (both sides) implements
    the "VICReg-style var>=1 on h(z')" guarantee cheaply.
  - Weight ramping is handled by the caller (JEPAWorldModel) via cpc_ramp_steps.

Gradient flow: encoder <- query path only (emb_t context). Predictor is NOT
involved — CPC shapes the ENCODER; the dynamics predictor stays MSE-anchored
in embedding space (v18 lesson: never train the predictor in a space
imagination doesn't consume).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionConditionedCPC(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        act_dim: int = 6,
        horizons: tuple = (1, 3, 5),
        proj_dim: int = 128,
        hidden_dim: int = 256,
        temperature: float = 0.1,
        var_weight: float = 1.0,
        max_samples: int = 1024,
    ):
        super().__init__()
        self.horizons = list(horizons)
        self.temperature = temperature
        self.var_weight = var_weight
        self.max_samples = max_samples
        self.act_dim = act_dim

        # Per-horizon query heads: (emb_t, a_t..a_{t+k-1}) -> proj_dim
        self.query_heads = nn.ModuleDict({
            str(k): nn.Sequential(
                nn.Linear(embed_dim + k * act_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, proj_dim),
            )
            for k in self.horizons
        })
        # Per-horizon key heads: target_emb_{t+k} -> proj_dim
        self.key_heads = nn.ModuleDict({
            str(k): nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, proj_dim),
            )
            for k in self.horizons
        })

    def forward(self, emb: torch.Tensor, target_emb: torch.Tensor,
                actions: torch.Tensor):
        """Compute InfoNCE loss averaged over horizons.

        Args:
            emb:        (B, T, D) online encoder embeddings (grad -> encoder)
            target_emb: (B, T, D) EMA target encoder embeddings (detached)
            actions:    (B, T, act_dim) one-hot actions

        Returns:
            loss: scalar (InfoNCE + variance hinge)
            diag: dict of detached diagnostics (per-horizon accuracy, std)
        """
        B, T, D = emb.shape
        device = emb.device
        target_emb = target_emb.detach()

        nce_terms, var_terms = [], []
        diag = {}

        for k in self.horizons:
            if T - k < 2:
                continue
            L = T - k  # anchors at t in [0, T-k-1]

            # Action window a_t..a_{t+k-1}: (B, L, k*act_dim)
            act_win = torch.cat([actions[:, i:i + L] for i in range(k)], dim=-1)
            query_in = torch.cat([emb[:, :L], act_win], dim=-1)  # (B, L, D+k*A)
            key_in = target_emb[:, k:k + L]                       # (B, L, D)

            query_in = query_in.reshape(B * L, -1)
            key_in = key_in.reshape(B * L, D)

            # Subsample anchors so the logits matrix stays bounded
            N = B * L
            if N > self.max_samples:
                idx = torch.randperm(N, device=device)[: self.max_samples]
                query_in = query_in[idx]
                key_in = key_in[idx]
                N = self.max_samples

            q_raw = self.query_heads[str(k)](query_in)  # (N, P)
            k_raw = self.key_heads[str(k)](key_in)      # (N, P)

            # Variance hinge on pre-normalization projections (anti-collapse
            # guarantee). std computed over the sample dimension, per dim.
            if self.var_weight > 0:
                q_std = q_raw.std(dim=0)
                k_std = k_raw.std(dim=0)
                var_terms.append(
                    F.relu(1.0 - q_std).mean() + F.relu(1.0 - k_std).mean()
                )

            q = F.normalize(q_raw, dim=-1)
            kk = F.normalize(k_raw, dim=-1)

            logits = q @ kk.t() / self.temperature  # (N, N)
            labels = torch.arange(N, device=device)
            nce = F.cross_entropy(logits, labels)
            nce_terms.append(nce)

            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == labels).float().mean().item()
                diag[f"cpc_acc_{k}step"] = acc
                diag[f"cpc_nce_{k}step"] = nce.item()
                diag[f"cpc_std_{k}step"] = q_raw.std(dim=0).mean().item()

        if not nce_terms:
            zero = torch.zeros(1, device=device)
            return zero, diag

        loss = sum(nce_terms) / len(nce_terms)
        if var_terms:
            loss = loss + self.var_weight * (sum(var_terms) / len(var_terms))
        return loss, diag


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    B, T, D, A = 8, 30, 256, 6
    cpc = ActionConditionedCPC(embed_dim=D, act_dim=A).to(device)

    emb = torch.randn(B, T, D, device=device, requires_grad=True)
    target_emb = torch.randn(B, T, D, device=device)
    actions = F.one_hot(torch.randint(0, A, (B, T)), A).float().to(device)

    loss, diag = cpc(emb, target_emb, actions)
    print(f"Loss: {loss.item():.4f}")
    for kk, v in diag.items():
        print(f"  {kk}: {v:.4f}")

    # On random data, accuracy must be near chance (1/N) and loss near ln(N)
    N = min(B * (T - 1), cpc.max_samples)
    import math
    print(f"Expected chance NCE ~ ln({N}) = {math.log(N):.2f}")

    # Gradients must reach the online emb (encoder path), not the target
    loss.backward()
    assert emb.grad is not None and emb.grad.abs().sum() > 0, \
        "CPC must propagate gradient to online embeddings"
    print(f"emb grad sum: {emb.grad.abs().sum().item():.4f}")

    # Collapse detection: identical keys must give high loss (uniform logits)
    with torch.no_grad():
        const_target = torch.zeros_like(target_emb)
    loss_c, diag_c = cpc(emb.detach(), const_target, actions)
    print(f"Collapsed-target loss: {loss_c.item():.4f} (should stay ~ln(N))")

    params = sum(p.numel() for p in cpc.parameters())
    print(f"Parameters: {params:,}")
    print("Smoke test passed!")
