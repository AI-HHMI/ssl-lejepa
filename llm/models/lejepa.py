"""LeJEPA: Self-Supervised Learning with Invariance and SIGReg for 3D Microscopy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from llm.encoders import ViT3DEncoder
from llm.losses import SIGReg, lejepa_loss
from llm.views import ViewMaker


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

    def forward(self, x: Tensor) -> Tensor:
        if self.training and x.shape[0] <= 1:
            return x
        return super().forward(x)


class ProjectorMLP(nn.Module):
    """Projection head mapping encoder embeddings to the metric space."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_hidden: int = 2,
        norm: str = "bn",
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_hidden):
            layers.extend([
                nn.Linear(d, hidden_dim),
                _norm1d(norm, hidden_dim),
                nn.GELU(),
            ])
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class LejepaConfig:
    """Hyperparameters for LeJEPA architecture, views, and SIGReg objective.

    Supports convenient aliases such as `num_layers` / `n_layers`, `dim` / `width`,
    and gracefully accepts extra keyword arguments.
    """

    # Backbone Architecture
    n_layers: int = 12
    width: int = 512
    num_heads: int = 8
    mlp_ratio: float = 4.0
    patch_size: Union[int, Tuple[int, int, int]] = (8, 8, 8)
    in_channels: int = 1
    dropout: float = 0.0
    attention_dropout: float = 0.0
    qkv_bias: bool = True

    # Projector
    use_projector: bool = True
    proj_dim: int = 128
    proj_hidden: int = 2048
    proj_n_hidden: int = 2
    proj_norm: str = "bn"

    # Multi-Crop Views
    views: str = "basic"  # 'basic', 'dino', or 'none'
    n_global: int = 2
    n_local: int = 4
    global_scale: Tuple[float, float] = (0.5, 1.0)
    local_scale: Tuple[float, float] = (0.15, 0.5)
    flip: bool = True
    token_dropout: float = 0.0

    # Objective / Loss
    lamb: float = 0.02
    num_slices: int = 256
    sigreg_knots: int = 17
    sigreg_tmax: float = 3.0

    # Extra options bag
    extra: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        n_layers: int = 12,
        width: int = 512,
        num_heads: Optional[int] = None,
        mlp_ratio: float = 4.0,
        patch_size: Union[int, Tuple[int, int, int]] = (8, 8, 8),
        in_channels: int = 1,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        qkv_bias: bool = True,
        use_projector: bool = True,
        proj_dim: int = 128,
        proj_hidden: int = 2048,
        proj_n_hidden: int = 2,
        proj_norm: str = "bn",
        views: str = "basic",
        n_global: int = 2,
        n_local: int = 4,
        global_scale: Tuple[float, float] = (0.5, 1.0),
        local_scale: Tuple[float, float] = (0.15, 0.5),
        flip: bool = True,
        token_dropout: float = 0.0,
        lamb: float = 0.02,
        num_slices: int = 256,
        sigreg_knots: int = 17,
        sigreg_tmax: float = 3.0,
        **kwargs: Any,
    ):
        # Resolve aliases
        if "num_layers" in kwargs:
            n_layers = kwargs.pop("num_layers")
        if "depth" in kwargs:
            n_layers = kwargs.pop("depth")
        if "dim" in kwargs:
            width = kwargs.pop("dim")
        if "embed_dim" in kwargs:
            width = kwargs.pop("embed_dim")
        if "hidden_dim" in kwargs:
            width = kwargs.pop("hidden_dim")
        if "heads" in kwargs:
            num_heads = kwargs.pop("heads")
        if num_heads is None:
            num_heads = max(1, width // 64)
        if "n_global_views" in kwargs:
            n_global = kwargs.pop("n_global_views")
        if "n_local_views" in kwargs:
            n_local = kwargs.pop("n_local_views")
        if "lambda_sigreg" in kwargs:
            lamb = kwargs.pop("lambda_sigreg")
        if "weight" in kwargs:
            lamb = kwargs.pop("weight")

        self.n_layers = n_layers
        self.width = width
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.qkv_bias = qkv_bias
        self.use_projector = use_projector
        self.proj_dim = proj_dim
        self.proj_hidden = proj_hidden
        self.proj_n_hidden = proj_n_hidden
        self.proj_norm = proj_norm
        self.views = views
        self.n_global = n_global
        self.n_local = n_local
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.flip = flip
        self.token_dropout = token_dropout
        self.lamb = lamb
        self.num_slices = num_slices
        self.sigreg_knots = sigreg_knots
        self.sigreg_tmax = sigreg_tmax
        self.extra = kwargs

        # Sync kwargs onto self.__dict__ for direct access
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def num_layers(self) -> int:
        return self.n_layers

    @num_layers.setter
    def num_layers(self, val: int) -> None:
        self.n_layers = val

    @property
    def dim(self) -> int:
        return self.width

    @dim.setter
    def dim(self, val: int) -> None:
        self.width = val

    @property
    def embed_dim(self) -> int:
        return self.width

    @embed_dim.setter
    def embed_dim(self, val: int) -> None:
        self.width = val

    @property
    def heads(self) -> int:
        return self.num_heads

    @heads.setter
    def heads(self, val: int) -> None:
        self.num_heads = val

    @property
    def n_global_views(self) -> int:
        return self.n_global

    @n_global_views.setter
    def n_global_views(self, val: int) -> None:
        self.n_global = val

    @property
    def n_local_views(self) -> int:
        return self.n_local

    @n_local_views.setter
    def n_local_views(self, val: int) -> None:
        self.n_local = val

    def __repr__(self) -> str:
        parts = [
            f"n_layers={self.n_layers}",
            f"width={self.width}",
            f"num_heads={self.num_heads}",
            f"patch_size={self.patch_size}",
            f"views={self.views!r}",
            f"lamb={self.lamb}",
        ]
        if self.use_projector:
            parts.append(f"proj_dim={self.proj_dim}")
        return f"LejepaConfig({', '.join(parts)})"


class Lejepa(nn.Module):
    """LeJEPA: Self-supervised representation learning for volumetric data.

    Combines a 3D Vision Transformer encoder with a multi-layer projection head,
    view augmentations, and the SIGReg + invariance objective.
    """

    def __init__(self, config: Optional[LejepaConfig] = None, **kwargs: Any):
        super().__init__()
        if config is None:
            self.cfg = LejepaConfig(**kwargs)
        elif isinstance(config, LejepaConfig):
            self.cfg = config
        else:
            raise TypeError(f"config must be an instance of LejepaConfig or None, got {type(config)}")

        # 3D ViT Encoder
        self.encoder = ViT3DEncoder(
            in_channels=self.cfg.in_channels,
            patch_size=self.cfg.patch_size,
            embed_dim=self.cfg.width,
            depth=self.cfg.n_layers,
            num_heads=self.cfg.num_heads,
            mlp_ratio=self.cfg.mlp_ratio,
            dropout=self.cfg.dropout,
            attention_dropout=self.cfg.attention_dropout,
            qkv_bias=self.cfg.qkv_bias,
            token_dropout=self.cfg.token_dropout,
        )

        # Projection head
        if self.cfg.use_projector:
            self.projector = ProjectorMLP(
                in_dim=self.cfg.width,
                hidden_dim=self.cfg.proj_hidden,
                out_dim=self.cfg.proj_dim,
                n_hidden=self.cfg.proj_n_hidden,
                norm=self.cfg.proj_norm,
            )
        else:
            self.projector = None

        # SIGReg module
        self.sigreg = SIGReg(
            num_slices=self.cfg.num_slices,
            knots=self.cfg.sigreg_knots,
            t_max=self.cfg.sigreg_tmax,
        )

        # View maker
        if self.cfg.views != "none":
            self.view_maker = ViewMaker(
                n_global=self.cfg.n_global,
                n_local=self.cfg.n_local,
                global_scale=self.cfg.global_scale,
                local_scale=self.cfg.local_scale,
                flip=self.cfg.flip,
            )
        else:
            self.view_maker = None

    def encode(self, x: Tensor) -> Tensor:
        """Encode input volume directly to embedding representations (B, width)."""
        return self.encoder(x)

    def project(self, emb: Tensor) -> Tensor:
        """Project embedding to metric space (B, proj_dim)."""
        if self.projector is not None:
            return self.projector(emb)
        return emb

    def encode_and_project(self, x: Tensor) -> Tensor:
        """Encode input volume and project to metric space."""
        emb = self.encode(x)
        return self.project(emb)

    def forward(
        self,
        x: Union[Tensor, dict[str, Any]],
        views: Optional[Tuple[list[Tensor], list[Tensor]]] = None,
        return_loss: bool = True,
        **kwargs: Any,
    ) -> Union[Tensor, dict[str, Tensor]]:
        """Forward pass through LeJEPA.

        Args:
            x: Input volume tensor or batch dictionary from Miao VolumeDataset.
               Accepts 3D (Z, Y, X), 4D (C, Z, Y, X), or 5D (B, C, Z, Y, X).
            views: Optional pre-constructed tuple of (globals, locals).
            return_loss: If True and views are enabled, returns loss dictionary.
                         If False, returns projected features for input x.

        Returns:
            Loss dictionary containing 'loss', 'inv', 'sigreg', 'weighted_sigreg',
            or projected representation tensor.
        """
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

        # If views are disabled or inference mode requested, return embeddings directly
        if not return_loss or self.cfg.views == "none" or self.view_maker is None:
            emb = self.encode(vol)
            return self.project(emb)

        # Generate views if not pre-provided
        if views is not None:
            g_views, l_views = views
        else:
            g_views, l_views = self.view_maker(vol)

        # Encode and project each view group
        # Each view is (B, C, Z_v, Y_v, X_v) -> projected to (B, proj_dim)
        proj_globals = torch.stack([
            self.encode_and_project(v) for v in g_views
        ], dim=0)  # (n_globals, B, proj_dim)

        if l_views:
            proj_locals = torch.stack([
                self.encode_and_project(v) for v in l_views
            ], dim=0)  # (n_locals, B, proj_dim)
            proj_all = torch.cat([proj_globals, proj_locals], dim=0)
        else:
            proj_all = proj_globals

        return lejepa_loss(
            globals=proj_globals,
            views=proj_all,
            sigreg=self.sigreg,
            lamb=self.cfg.lamb,
        )
