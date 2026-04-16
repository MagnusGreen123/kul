"""
Sketched Isotropic Gaussian Regularizer (SIGReg).

Enforces embeddings to follow N(0, I) by testing univariate projections
against the standard Gaussian characteristic function (Epps-Pulley test).

Prevents representation collapse WITHOUT stop-gradient or EMA.
Only effective hyperparameter: the loss weight lambda (default 0.1).

Reference: LeJEPA (arXiv 2511.08544), LeWorldModel (arXiv 2603.19312)
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Projects embeddings onto random unit-norm directions and tests each
    1-D marginal against N(0,1) using the Epps-Pulley characteristic
    function test.  By the Cramer-Wold theorem, matching all 1-D
    marginals is equivalent to matching the full joint distribution.

    Args:
        knots:    Number of quadrature points for the integral (default 17).
        num_proj: Number of random projection directions (default 1024).
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

        # Quadrature grid over [0, 3] with trapezoidal weights
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # trapezoidal end-point correction

        # Gaussian window: phi(t) = exp(-t^2/2) is the N(0,1) char function
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            embeddings: (N, D) batch of embedding vectors.

        Returns:
            Scalar loss.  Approaches 0 as the embedding distribution
            converges to N(0, I_D).
        """
        N, D = embeddings.shape

        # Random unit-norm projection directions (resampled each call)
        A = torch.randn(D, self.num_proj, device=embeddings.device)
        A = A.div_(A.norm(p=2, dim=0))

        # Project: (N, num_proj), then outer product with quadrature grid
        proj = embeddings @ A                        # (N, num_proj)
        x_t = proj.unsqueeze(-1) * self.t            # (N, num_proj, knots)

        # Empirical characteristic function vs N(0,1)
        # ECF = (1/N) sum_n exp(i*t*x_n) = cos_mean + i*sin_mean
        cos_mean = x_t.cos().mean(dim=0)             # (num_proj, knots)
        sin_mean = x_t.sin().mean(dim=0)             # (num_proj, knots)

        # |ECF(t) - phi(t)|^2 = (cos_mean - phi)^2 + sin_mean^2
        err = (cos_mean - self.phi).square() + sin_mean.square()

        # Weighted integral (Epps-Pulley statistic), scale by N
        statistic = (err @ self.weights) * N          # (num_proj,)

        return statistic.mean()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)

    # Test 1: Gaussian embeddings should have low loss
    z_gaussian = torch.randn(512, 256, device=device)
    loss_gauss = sigreg(z_gaussian)
    print(f"Gaussian embeddings: SIGReg = {loss_gauss.item():.4f}")

    # Test 2: Collapsed embeddings (all same) should have high loss
    z_collapsed = torch.ones(512, 256, device=device) * 3.0
    loss_collapsed = sigreg(z_collapsed)
    print(f"Collapsed embeddings: SIGReg = {loss_collapsed.item():.4f}")

    # Test 3: Gradient flows
    z = torch.randn(512, 256, device=device, requires_grad=True)
    loss = sigreg(z)
    loss.backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    print(f"Gradient norm: {z.grad.norm():.4f}")

    print(f"Parameters: {sum(p.numel() for p in sigreg.parameters()):,} (should be 0 — all buffers)")
    print("Smoke test passed!")
