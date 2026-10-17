"""Loss functions for LeJEPA self-supervised pretraining.

Includes SIGReg (Sketched Isotropic Gaussian Regularization) and the
invariance + SIGReg joint loss (Balestriero & LeCun, 2025).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class SIGRegConfig:
    """SIGReg hyperparameters."""
    num_slices: int = 256  # random 1D projections per forward pass
    knots: int = 17  # quadrature knots on [0, t_max]; odd
    t_max: float = 3.0  # upper limit of integration


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (SIGReg).

    Projects feature representations onto random 1D unit slices and evaluates
    the empirical characteristic function against the standard normal N(0, 1) CF
    via numerical quadrature.
    """

    t: Tensor
    phi: Tensor
    weights: Tensor

    def __init__(self, config: SIGRegConfig | None = None, **kwargs):
        super().__init__()
        self.config = c = replace(config or SIGRegConfig(), **kwargs)
        knots, t_max = c.knots, c.t_max
        assert knots % 2 == 1, f"knots must be odd for trapezoidal quadrature, got {knots}"

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
        A = torch.randn(proj.size(-1), self.config.num_slices, device=device, dtype=dtype)
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
        return repr(self.config)


class LejepaOutput(dict):
    """Output dictionary from LeJEPA supporting attribute and dict key access (e.g. out.loss, out['loss'])."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def lejepa_loss(
    pred: Tensor,
    target: Tensor,
    sig: Tensor,
    lamb: float = 0.02,
) -> LejepaOutput:
    """Compute the prediction (or invariance) + SIGReg loss, of either Lejepa objective.

    Args:
        pred: Predicted embeddings of the masked tokens, shape (B, M, D); "views": the mean of the globals' embeddings,
            shape (View, B, D), broadcast over the views.
        target: The same tokens' embeddings from the encoder seen the whole view, shape (B, M, D); "views": every view's.
        sig: The SIGReg statistic of the embeddings, over the samples LejepaConfig.sigreg_over picks (Lejepa.sigreg_term);
            "views": over each view's batch.
        lamb: Weight for the SIGReg regularization term.

    Returns:
        LejepaOutput containing 'loss', 'inv' (the prediction error, the old invariance term's slot), 'sigreg',
        'weighted_sigreg'.
    """
    inv = (pred - target).square().mean()
    weighted_sig = lamb * sig
    total_loss = inv + weighted_sig
    return LejepaOutput(
        loss=total_loss,
        inv=inv,
        sigreg=sig,
        weighted_sigreg=weighted_sig,
    )
