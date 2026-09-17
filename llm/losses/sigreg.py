"""Loss functions for LeJEPA self-supervised pretraining.

Includes SIGReg (Sketched Isotropic Gaussian Regularization) and the
invariance + SIGReg joint loss (Balestriero & LeCun, 2025).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (SIGReg).

    Projects feature representations onto random 1D unit slices and evaluates
    the empirical characteristic function against the standard normal N(0, 1) CF
    via numerical quadrature.

    Args:
        num_slices: Number of random 1D projections per forward pass.
        knots: Number of quadrature knots on [0, t_max]. Must be odd.
        t_max: Upper limit of integration.
    """

    t: Tensor
    phi: Tensor
    weights: Tensor

    def __init__(self, num_slices: int = 256, knots: int = 17, t_max: float = 3.0):
        super().__init__()
        if knots % 2 == 0:
            raise ValueError(f"knots must be odd for trapezoidal quadrature, got {knots}")
        self.num_slices = num_slices
        self.knots = knots
        self.t_max = t_max

        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # Endpoints have half weight in trapezoidal rule
        window = torch.exp(-t.square() / 2.0)  # Standard normal characteristic function

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: Tensor) -> Tensor:
        """Compute the SIGReg statistic.

        Args:
            proj: Projected embeddings of shape (..., N, D) where N is number of samples,
                and D is feature dimension.

        Returns:
            Scalar regularization loss.
        """
        device = proj.device
        dtype = proj.dtype

        # Random unit-norm projection directions A: (D, num_slices)
        A = torch.randn(proj.size(-1), self.num_slices, device=device, dtype=dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)

        t = self.t.to(device=device, dtype=dtype)
        phi = self.phi.to(device=device, dtype=dtype)
        weights = self.weights.to(device=device, dtype=dtype)

        # Project onto slices and evaluate empirical CF: (..., N, num_slices, knots)
        x_t = (proj @ A).unsqueeze(-1) * t
        # Mean over sample dimension (-3)
        err = (x_t.cos().mean(dim=-3) - phi).square() + x_t.sin().mean(dim=-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()

    def extra_repr(self) -> str:
        return f"num_slices={self.num_slices}, knots={self.knots}, t_max={self.t_max}"


def lejepa_loss(
    globals: Tensor,
    views: Tensor,
    sigreg: nn.Module,
    lamb: float = 0.02,
) -> dict[str, Tensor]:
    """Compute joint invariance + SIGReg loss.

    Args:
        globals: Global view embeddings of shape (n_globals, B, D).
        views: All view embeddings of shape (n_total_views, B, D).
        sigreg: SIGReg module instance.
        lamb: Weight for the SIGReg regularization term.

    Returns:
        Dictionary containing 'loss', 'inv', 'sigreg', 'weighted_sigreg'.
    """
    # Invariance loss: pull all views toward the mean of the global views
    global_mean = globals.mean(dim=0, keepdim=True)
    inv = (global_mean - views).square().mean()

    # SIGReg regularization
    sig = sigreg(views)
    weighted_sig = lamb * sig
    total_loss = inv + weighted_sig

    return {
        "loss": total_loss,
        "inv": inv,
        "sigreg": sig,
        "weighted_sigreg": weighted_sig,
    }
