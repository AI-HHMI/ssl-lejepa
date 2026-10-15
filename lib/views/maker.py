"""Multi-crop view generation for self-supervised learning on 3D microscopy volumes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple
import torch
from torch import Tensor

from lib.types import Tup3Int, Views


@dataclass
class ViewConfig:
    """Configuration for multi-crop view creation (see ViewMaker)."""
    views: Views = "basic"  # 'none': Lejepa makes no views
    n_global: int = 1  # Lejepa trains on one view per sample (see its mask); other counts are for ViewMaker's own use
    n_local: int = 0
    global_scale: Tuple[float, float] = (0.5, 1.0)  # basic
    local_scale: Tuple[float, float] = (0.15, 0.5)  # basic
    global_size: Tup3Int = (40, 128, 128)  # displace
    local_size: Tup3Int = (32, 96, 96)  # displace
    flip: bool = True


class ViewMaker:
    """Generates global and local crops from 3D volumes, batched or unbatched.

    views='basic': each view gets a random volume fraction (global_scale / local_scale), independent origin.
    views='displace': fixed global_size / local_size crops; each local sits inside a randomly chosen
        global of the same sample, at a uniformly random displacement from that global's origin.
    Both modes flip each crop per sample and per axis with probability 0.5. No resizing.
    """

    def __init__(self, config: ViewConfig | None = None, patch_size: Tup3Int = (1, 1, 1),
                 generator: Optional[torch.Generator] = None, **kwargs):
        self.config = c = replace(config or ViewConfig(), **kwargs)
        assert c.views in ("basic", "displace"), f"unknown views {c.views!r}"
        self.generator = generator
        self.patch_size = patch_size  # crop sizes are rounded to multiples of this
        if c.views == "displace":
            for name, size in [("global_size", c.global_size), ("local_size", c.local_size)][:1 + (c.n_local > 0)]:  # no locals, no local_size
                assert all(s % p == 0 for s, p in zip(size, patch_size)), f"{name} {size} must be a multiple of patch size {patch_size}"
            assert c.n_local == 0 or all(l <= g for l, g in zip(c.local_size, c.global_size)), (
                f"local_size {c.local_size} must fit inside global_size {c.global_size}"
            )
            assert c.n_global > 0 or c.n_local == 0, "displace needs a global to anchor each local"

    def _rand_uniform(self, low: float, high: float) -> float:
        return low + (high - low) * torch.rand((), generator=self.generator).item()

    def _crop_shape(self, scale: float, spatial_shape: Tup3Int) -> Tuple[int, ...]:
        """Crop with volume fraction `scale`, each side rounded to a whole number of patches."""
        linear_scale = scale ** (1.0 / 3.0)
        return tuple(
            p * max(1, min(s // p, round(s * linear_scale / p)))
            for s, p in zip(spatial_shape, self.patch_size)
        )

    def _origins(self, outer: Tuple[int, ...], inner: Tuple[int, ...], B: int) -> Tensor:
        """Uniform per-sample origins (Batch 3, on CPU) placing an `inner` box inside an `outer` box."""
        return torch.stack([
            torch.randint(0, o - i + 1, (B,), generator=self.generator) for o, i in zip(outer, inner)
        ], dim=1)

    def _gather(self, x: Tensor, origin: Tensor, crop_shape: Tuple[int, ...]) -> Tensor:
        """Per-sample crops at `origin` (Batch 3), randomly flipped, gathered in a single indexing op.

        x: Batch C Z Y X -> Batch C cZ cY cX, with the crop shape shared across the batch.
        """
        B, C = x.shape[:2]
        assert len(crop_shape) == 3
        # Per-sample, per-axis indices; built on CPU (seeded by self.generator), then one small copy to device.
        idx = []
        for axis, cs in enumerate(crop_shape):
            offset = torch.arange(cs).expand(B, cs)  # Batch cs
            if self.config.flip:
                flip = torch.rand((B, 1), generator=self.generator) < 0.5
                offset = torch.where(flip, cs - 1 - offset, offset)
            idx.append((origin[:, axis:axis + 1] + offset).to(x.device, non_blocking=True))
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

    def _basic_view(self, x: Tensor, scale_range: Tuple[float, float]) -> Tensor:
        spatial_shape: Tup3Int = (x.shape[2], x.shape[3], x.shape[4])
        crop_shape = self._crop_shape(self._rand_uniform(*scale_range), spatial_shape)
        return self._gather(x, self._origins(spatial_shape, crop_shape, x.shape[0]), crop_shape)

    def _displace_views(self, x: Tensor) -> Tuple[list[Tensor], list[Tensor]]:
        c = self.config
        B = x.shape[0]
        spatial_shape = tuple(x.shape[2:])
        assert all(g <= s for g, s in zip(c.global_size, spatial_shape)), (
            f"global_size {c.global_size} larger than input {spatial_shape}"
        )
        g_origins = torch.stack([self._origins(spatial_shape, c.global_size, B) for _ in range(c.n_global)])  # Global Batch 3
        globals_ = [self._gather(x, o, c.global_size) for o in g_origins]
        locals_ = []
        for _ in range(c.n_local):
            anchor = torch.randint(0, c.n_global, (B,), generator=self.generator)  # Batch
            displacement = self._origins(c.global_size, c.local_size, B)  # Batch 3, keeps the local inside its global
            origin = g_origins[anchor, torch.arange(B)] + displacement
            locals_.append(self._gather(x, origin, c.local_size))
        return globals_, locals_

    def __call__(self, x: Tensor) -> Tuple[list[Tensor], list[Tensor]]:
        """Generate (globals, locals) from (B, C, Z, Y, X) or (C, Z, Y, X)."""
        if x.dim() == 4:
            x = x.unsqueeze(0)
        elif x.dim() != 5:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {tuple(x.shape)}")
        assert all(s % p == 0 for s, p in zip(x.shape[2:], self.patch_size)), (
            f"input spatial shape {tuple(x.shape[2:])} must be a multiple of patch size {self.patch_size}"
        )
        c = self.config
        if c.views == "displace":
            return self._displace_views(x)
        globals_ = [self._basic_view(x, c.global_scale) for _ in range(c.n_global)]
        locals_ = [self._basic_view(x, c.local_scale) for _ in range(c.n_local)]
        return globals_, locals_
