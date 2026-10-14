"""LeJEPA: Self-Supervised Learning with Invariance and SIGReg for 3D Microscopy."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from lib.encoders import EncoderConfig, ViT3DEncoder
from lib.losses import LejepaOutput, SIGReg, SIGRegConfig, lejepa_loss
from lib.views import ViewConfig, ViewMaker


def _norm1d(norm: str, dim: int) -> nn.Module:
    if norm == "bn":
        return SafeBatchNorm1d(dim)
    elif norm == "ln":
        return nn.LayerNorm(dim)
    elif norm == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm type: {norm!r}. Expected 'bn', 'ln', or 'none'.")


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that safely passes through single-sample inputs during training."""

    def forward(self, input: Tensor) -> Tensor:
        if self.training and input.shape[0] <= 1:
            return input
        return super().forward(input)


@dataclass
class ProjectorConfig:
    """ProjectorMLP hyperparameters (its input dim is the encoder's embed_dim)."""
    dim: int = 128  # output dim: the metric space SIGReg and invariance act in
    hidden: int = 2048
    n_hidden: int = 2
    norm: str = "bn"  # 'bn', 'ln' or 'none'


class ProjectorMLP(nn.Module):
    """Projection head mapping encoder embeddings to the metric space."""

    def __init__(self, in_dim: int, config: ProjectorConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(config.n_hidden):
            layers.extend([
                nn.Linear(d, config.hidden),
                _norm1d(config.norm, config.hidden),
                nn.GELU(),
            ])
            d = config.hidden
        layers.append(nn.Linear(d, config.dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class LejepaConfig:
    """Hyperparameters for LeJEPA: each part's own config, whole, plus how they combine."""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    projector: ProjectorConfig | None = field(default_factory=ProjectorConfig)  # None: views embed as the encoder's output
    view: ViewConfig = field(default_factory=ViewConfig)
    sigreg: SIGRegConfig = field(default_factory=SIGRegConfig)
    batch_views: bool = False  # one encoder call per group of same-shape views (projector BN then pools the group)
    lamb: float = 0.02  # loss = inv + lamb * sigreg


class Lejepa(nn.Module):
    """LeJEPA: Self-supervised representation learning for volumetric data.

    Combines a 3D Vision Transformer encoder with a multi-layer projection head,
    view augmentations, and the SIGReg + invariance objective.
    """

    def __init__(self, config: LejepaConfig | None = None, **kwargs: Any):
        super().__init__()
        self.config = c = replace(config or LejepaConfig(), **kwargs)
        self.encoder = ViT3DEncoder(c.encoder)
        self.projector = ProjectorMLP(c.encoder.embed_dim, c.projector) if c.projector else None
        self.sigreg = SIGReg(c.sigreg)
        self.view_maker = ViewMaker(c.view, patch_size=self.encoder.patch_embed.patch_size) if c.view.views != "none" else None

        # For FLOP accounting in forward.
        self._n_encoder_params = sum(p.numel() for p in self.encoder.parameters())
        self._n_projector_params = sum(p.numel() for p in self.projector.parameters()) if self.projector is not None else 0

    def encode_maybe_project(self, x: Tensor) -> Tensor:
        """Encode input volume and project to metric space."""
        emb = self.encoder(x)
        if self.projector:
            return self.projector(emb)
        return emb

    def encode_views(self, views: list[Tensor]) -> Tensor:
        """Encode and project each view (each Batch C Z Y X) -> View Batch D.

        With config.batch_views, same-shape views are concatenated along Batch into one encoder call.
        """
        if not self.config.batch_views:
            return torch.stack([self.encode_maybe_project(v) for v in views])
        groups: dict[tuple[int, ...], list[int]] = {}
        for i, v in enumerate(views):
            groups.setdefault(tuple(v.shape), []).append(i)
        out: list[Tensor] = [views[0]] * len(views)  # placeholders, all overwritten below
        for idx in groups.values():
            z = self.encode_maybe_project(torch.cat([views[i] for i in idx]))  # (len(idx)·Batch) D
            for i, zi in zip(idx, z.chunk(len(idx))):
                out[i] = zi
        return torch.stack(out)

    def forward(
        self,
        x: Union[Tensor, dict[str, Any]],
        views: Optional[Tuple[list[Tensor], list[Tensor]]] = None,
        return_loss: bool = True,
        **kwargs: Any,
    ) -> Union[Tensor, LejepaOutput]:
        # Handle dict input from Miao VolumeDataset (e.g. dl[0])
        if isinstance(x, dict):
            vol = x["img"]
        else:
            vol = x

        # Ensure 5D tensor: (B, C, Z, Y, X)
        if vol.dim() == 3:
            vol = vol.unsqueeze(0).unsqueeze(0)
        elif vol.dim() == 4:
            vol = vol.unsqueeze(0)
        elif vol.dim() != 5:
            raise ValueError(f"Expected 3D, 4D, or 5D input, got tensor with shape {tuple(vol.shape)}")

        # Ensure volume is on the same device as model parameters
        try:
            param_device = next(self.parameters()).device
            if vol.device != param_device:
                vol = vol.to(param_device)
        except StopIteration:
            pass

        # If views are disabled or inference mode requested, return embeddings directly
        if not return_loss or self.view_maker is None:
            return self.encode_maybe_project(vol)

        # Generate views if not pre-provided
        if views is not None:
            g_views, l_views = views
        else:
            g_views, l_views = self.view_maker(vol)

        # Encode and project each view group
        # Each view is (B, C, Z_v, Y_v, X_v) -> projected to (B, proj_dim)
        proj_globals = self.encode_views(g_views)  # (n_globals, B, proj_dim)

        if l_views:
            proj_locals = self.encode_views(l_views)  # (n_locals, B, proj_dim)
            proj_all = torch.cat([proj_globals, proj_locals], dim=0)
        else:
            proj_all = proj_globals

        # SIGReg's cos/sin statistics are precision sensitive, so keep the loss in fp32 under autocast.
        with torch.autocast(vol.device.type, enabled=False):
            out = lejepa_loss(
                globals=proj_globals.float(),
                views=proj_all.float(),
                sigreg=self.sigreg,
                lamb=self.config.lamb,
            )
        # Encoder tokens this step, summed over views and batch (python int, no device sync).
        patch_vol = prod(self.encoder.patch_embed.patch_size)
        out.n_tokens = sum(v.shape[0] * prod(v.shape[2:]) // patch_vol for v in [*g_views, *l_views])
        # Model training FLOPs this step (backward ~ 2x forward): per token, 6 x encoder params for the weights
        # plus 12 x depth x width x N for attention over its N-token view; per sample, 6 x projector params.
        # Ignores flash-attention recompute and elementwise ops, as MFU conventionally does.
        enc = self.config.encoder
        out.n_flops = sum(
            v.shape[0] * (n * (6 * self._n_encoder_params + 12 * enc.depth * enc.embed_dim * n) + 6 * self._n_projector_params)
            for v in [*g_views, *l_views]
            for n in [prod(v.shape[2:]) // patch_vol]
        )
        return out
