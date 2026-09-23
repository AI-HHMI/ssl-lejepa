"""Multi-crop view generation for self-supervised learning on 3D microscopy volumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import Tensor


@dataclass
class ViewConfig:
    """Configuration for multi-crop view creation."""
    views: str = "basic"
    n_global: int = 2
    n_local: int = 4
    global_scale: Tuple[float, float] = (0.5, 1.0)
    local_scale: Tuple[float, float] = (0.15, 0.5)
    flip: bool = True
    token_dropout: float = 0.0


class ViewMaker:
    """Generates global and local crops from 3D volumes.

    Supports 'basic' multi-crop strategy (random scales and random axis flips)
    and handles batched or unbatched inputs.
    """

    def __init__(
        self,
        n_global: int = 2,
        n_local: int = 4,
        global_scale: Tuple[float, float] = (0.5, 1.0),
        local_scale: Tuple[float, float] = (0.15, 0.5),
        flip: bool = True,
        generator: Optional[torch.Generator] = None,
        patch_size: Tuple[int, int, int] = (1, 1, 1),
    ):
        self.n_global = n_global
        self.n_local = n_local
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.flip = flip
        self.generator = generator
        self.patch_size = patch_size  # crop sizes are rounded to multiples of this

    def _rand_uniform(self, low: float, high: float) -> float:
        return low + (high - low) * torch.rand((), generator=self.generator).item()

    def _crop_shape(self, scale: float, spatial_shape: Tuple[int, int, int]) -> Tuple[int, ...]:
        """Crop with volume fraction `scale`, each side rounded to a whole number of patches."""
        linear_scale = scale ** (1.0 / 3.0)
        return tuple(
            p * max(1, min(s // p, round(s * linear_scale / p)))
            for s, p in zip(spatial_shape, self.patch_size)
        )

    def _view(self, x: Tensor, scale_range: Tuple[float, float]) -> Tensor:
        """One view: a random-origin, randomly flipped crop per sample, gathered in a single indexing op.

        x: Batch C Z Y X -> Batch C cZ cY cX, with the crop shape shared across the batch.
        """
        B, C = x.shape[:2]
        spatial_shape = tuple(x.shape[2:])
        assert len(spatial_shape)==3
        crop_shape = self._crop_shape(self._rand_uniform(*scale_range), spatial_shape)
        # Per-sample, per-axis indices; built on CPU (seeded by self.generator), then one small copy to device.
        idx = []
        for s, cs in zip(spatial_shape, crop_shape):
            origin = torch.randint(0, s - cs + 1, (B, 1), generator=self.generator)  # Batch 1
            offset = torch.arange(cs).expand(B, cs)  # Batch cs
            if self.flip:
                flip = torch.rand((B, 1), generator=self.generator) < 0.5
                offset = torch.where(flip, cs - 1 - offset, offset)
            idx.append((origin + offset).to(x.device, non_blocking=True))
        iz, iy, ix = idx
        b = torch.arange(B, device=x.device)
        c = torch.arange(C, device=x.device)
        # Adjacent advanced indices broadcast to Batch C cZ cY cX directly.
        return x[
            b[:, None, None, None, None],
            c[None, :, None, None, None],
            iz[:, None, :, None, None],
            iy[:, None, None, :, None],
            ix[:, None, None, None, :],
        ]

    def __call__(self, x: Tensor) -> Tuple[list[Tensor], list[Tensor]]:
        """Generate (globals, locals) from (B, C, Z, Y, X) or (C, Z, Y, X)."""
        if x.dim() == 4:
            x = x.unsqueeze(0)
        elif x.dim() != 5:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {tuple(x.shape)}")
        assert all(s % p == 0 for s, p in zip(x.shape[2:], self.patch_size)), (
            f"input spatial shape {tuple(x.shape[2:])} must be a multiple of patch size {self.patch_size}"
        )
        globals_ = [self._view(x, self.global_scale) for _ in range(self.n_global)]
        locals_ = [self._view(x, self.local_scale) for _ in range(self.n_local)]
        return globals_, locals_
