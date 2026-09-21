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
    ):
        self.n_global = n_global
        self.n_local = n_local
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.flip = flip
        self.generator = generator

    def _rand_uniform(self, low: float, high: float) -> float:
        return low + (high - low) * torch.rand((), generator=self.generator).item()

    def _maybe_flip(self, crop: Tensor) -> Tensor:
        """Random per-axis flip along spatial dimensions."""
        if not self.flip:
            return crop
        for dim in (-3, -2, -1):
            if self._rand_uniform(0.0, 1.0) < 0.5:
                crop = crop.flip(dim)
        return crop

    def __call__(self, x: Tensor) -> Tuple[list[Tensor], list[Tensor]]:
        """Generate (globals, locals) from (B, C, Z, Y, X) or (C, Z, Y, X)."""
        is_4d = x.dim() == 4
        if is_4d:
            x = x.unsqueeze(0)
        elif x.dim() != 5:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {tuple(x.shape)}")

        B, C, Z, Y, X_dim = x.shape
        spatial_shape = (Z, Y, X_dim)

        globals_: list[Tensor] = []
        for _ in range(self.n_global):
            scale = self._rand_uniform(*self.global_scale)
            linear_scale = scale ** (1.0 / 3.0)
            crop_shape = tuple(
                max(4, min(s, int(round(s * linear_scale))))
                for s in spatial_shape
            )
            view_batch: list[Tensor] = []
            for b in range(B):
                origins = [
                    int(torch.randint(0, max(1, s - cs + 1), (), generator=self.generator).item())
                    for s, cs in zip(spatial_shape, crop_shape)
                ]
                slices = (slice(None),) + tuple(
                    slice(o, o + cs) for o, cs in zip(origins, crop_shape)
                )
                crop = self._maybe_flip(x[b][slices])
                view_batch.append(crop)
            globals_.append(torch.stack(view_batch, dim=0))

        locals_: list[Tensor] = []
        if self.n_local > 0:
            for _ in range(self.n_local):
                scale = self._rand_uniform(*self.local_scale)
                linear_scale = scale ** (1.0 / 3.0)
                crop_shape = tuple(
                    max(4, min(s, int(round(s * linear_scale))))
                    for s in spatial_shape
                )
                view_batch = []
                for b in range(B):
                    origins = [
                        int(torch.randint(0, max(1, s - cs + 1), (), generator=self.generator).item())
                        for s, cs in zip(spatial_shape, crop_shape)
                    ]
                    slices = (slice(None),) + tuple(
                        slice(o, o + cs) for o, cs in zip(origins, crop_shape)
                    )
                    crop = self._maybe_flip(x[b][slices])
                    view_batch.append(crop)
                locals_.append(torch.stack(view_batch, dim=0))

        return globals_, locals_
