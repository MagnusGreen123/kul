"""
Action-conditioned transformer predictor for JEPA world model.

Predicts next-step embeddings from (embedding, action) pairs using
Adaptive Layer Normalization (AdaLN-zero) for action conditioning
and causal attention for autoregressive structure.

Architecture follows LeWorldModel (arXiv 2603.19312):
  - Action encoder: MLP maps actions to embedding dimension
  - Positional embeddings: learnable per-position
  - ConditionalBlocks: Transformer blocks with AdaLN-zero action conditioning
  - Causal masking: each position only attends to itself and earlier positions

During training: processes full sequences in parallel (teacher forcing).
During imagination: used autoregressively with a truncated history window.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(self.norm(x))


class CausalAttention(nn.Module):
    """Multi-head self-attention with causal masking."""

    def __init__(self, dim, heads=4, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning on actions.

    The action embedding produces 6 modulation parameters per block:
    (shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn).
    All initialized to zero so the block starts as identity.
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = CausalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # AdaLN modulation: action embedding -> 6 * dim params
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        # Zero-initialize for stable training start
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        """
        Args:
            x: (B, T, D) token sequence
            c: (B, T, D) action conditioning embedding
        """
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_a * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_f * self.ffn(modulate(self.norm2(x), shift_f, scale_f))
        return x


class ActionConditionedPredictor(nn.Module):
    """Autoregressive predictor for next-embedding prediction.

    At position t, given (e_t, a_t), predicts e_{t+1}.
    Causal masking ensures position t only sees positions 0..t.

    Args:
        embed_dim:   Dimension of state embeddings.
        act_dim:     Action space size.
        depth:       Number of transformer blocks.
        heads:       Number of attention heads.
        mlp_dim:     FFN hidden dimension.
        max_seq_len: Maximum sequence length (for positional embeddings).
        dropout:     Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        act_dim: int = 6,
        depth: int = 4,
        heads: int = 4,
        mlp_dim: int = 512,
        max_seq_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_head = embed_dim // heads

        # Action encoder: maps discrete/continuous actions to embedding dim
        self.action_encoder = nn.Sequential(
            nn.Linear(act_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)

        # Transformer blocks with AdaLN conditioning
        self.blocks = nn.ModuleList([
            ConditionalBlock(embed_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, embeddings, actions):
        """Predict next-step embeddings.

        Args:
            embeddings: (B, T, D) state embeddings.
            actions:    (B, T, act_dim) actions taken at each state.

        Returns:
            (B, T, D) predicted next-step embeddings.
            Output at position t is the prediction for e_{t+1}.
        """
        B, T, D = embeddings.shape

        # Encode actions for conditioning
        act_emb = self.action_encoder(actions)          # (B, T, D)

        # Add positional embeddings
        x = embeddings + self.pos_embedding[:, :T]

        # Transformer blocks with action conditioning via AdaLN
        for block in self.blocks:
            x = block(x, act_emb)

        return self.norm(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    embed_dim = 256
    act_dim = 6
    predictor = ActionConditionedPredictor(
        embed_dim=embed_dim, act_dim=act_dim, depth=4, heads=4,
        mlp_dim=512, max_seq_len=32, dropout=0.1,
    ).to(device)

    B, T = 4, 20
    emb = torch.randn(B, T, embed_dim, device=device)
    act = torch.zeros(B, T, act_dim, device=device)
    act[:, :, 0] = 1.0  # one-hot action 0

    # Test teacher-forcing forward
    pred = predictor(emb, act)
    print(f"Input:  emb {emb.shape}, act {act.shape}")
    print(f"Output: {pred.shape}")  # (4, 20, 256)

    # Test gradient flow
    target = torch.randn_like(pred)
    loss = F.mse_loss(pred, target)
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    print(f"Grad flows to embeddings: {emb.requires_grad}")  # False (not leaf requiring grad)

    # Test autoregressive mode (short context)
    ctx_emb = torch.randn(B, 3, embed_dim, device=device)
    ctx_act = torch.randn(B, 3, act_dim, device=device)
    pred_ar = predictor(ctx_emb, ctx_act)
    next_emb = pred_ar[:, -1]  # last position prediction
    print(f"AR input: {ctx_emb.shape}, output last: {next_emb.shape}")

    params = sum(p.numel() for p in predictor.parameters())
    print(f"Parameters: {params:,}")
    print("Smoke test passed!")
