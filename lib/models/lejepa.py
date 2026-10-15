"""LeJEPA: Self-Supervised Learning with masked-token prediction and SIGReg for 3D Microscopy."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod
from typing import Any, Literal, Union

import torch
import torch.nn as nn
from torch import Tensor

from lib.encoders import EncoderConfig, TransformerBlock, ViT3DEncoder, get_3d_sincos_pos_embed
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
class MaskConfig:
    """What the model predicts from what: one block of tokens is masked per sample."""
    block: float = 0.6  # the masked cube's side, as a fraction of the view's token grid side (~22% of the tokens)
    predictor_depth: int = 4  # transformer blocks of the predictor, at the encoder's width and heads


def mask_side(mask: MaskConfig, grid: tuple[int, ...]) -> tuple[int, ...]:
    """Side, in tokens, of the masked cube in a view with this token grid (the same for every sample: static shapes)."""
    side = tuple(max(1, round(mask.block * g)) for g in grid)
    assert all(s < g for s, g in zip(side, grid)), f"mask block {side} leaves no visible token in grid {grid}"
    return side


def train_flops(config: LejepaConfig, grid: tuple[int, ...], n_encoder: int, n_predictor: int, n_projector: int) -> int:
    """Training FLOPs of one sample whose view has this token grid (backward ~ 2x forward), given the parameter counts:
    per token, 6 x params for the weights plus 12 x depth x width x n for attention over the n tokens it sees. The
    encoder sees every token (the targets), then the visible ones; the predictor sees all N; the projector every token
    and every prediction. Ignores flash-attention recompute and elementwise ops, as MFU conventionally does."""
    n, m = prod(grid), prod(mask_side(config.mask, grid))
    attention = lambda n_tokens, depth: 12 * depth * config.encoder.embed_dim * n_tokens
    encoder = lambda n_tokens: n_tokens * (6 * n_encoder + attention(n_tokens, config.encoder.depth))
    predictor = n * (6 * n_predictor + attention(n, config.mask.predictor_depth))
    return encoder(n) + encoder(n - m) + predictor + 6 * n_projector * (n + m)


class Predictor(nn.Module):
    """Predicts the embeddings of masked tokens from the visible ones: a transformer over the visible tokens and one
    learned mask token per masked position, each carrying its position's sincos embedding."""

    def __init__(self, dim: int, num_heads: int, depth: int):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList(TransformerBlock(dim=dim, num_heads=num_heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def forward(self, context: Tensor, pos_context: Tensor, pos_masked: Tensor) -> Tensor:
        """context: Batch Nvis D, the encoder's tokens at pos_context (Batch Nvis D). -> Batch M D at pos_masked."""
        x = torch.cat([context + pos_context, self.mask_token + pos_masked], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x[:, context.shape[1]:])


@dataclass
class LejepaConfig:
    """Hyperparameters for LeJEPA: each part's own config, whole, plus how they combine."""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    projector: ProjectorConfig | None = field(default_factory=ProjectorConfig)  # None: tokens embed as the encoder's output
    view: ViewConfig = field(default_factory=ViewConfig)  # the one crop each sample trains on (n_global=1, n_local=0)
    mask: MaskConfig = field(default_factory=MaskConfig)
    sigreg: SIGRegConfig = field(default_factory=SIGRegConfig)
    lamb: float = 0.02  # loss = inv + lamb * sigreg
    # SIGReg's samples: "view", each view's N tokens (a code of position alone passes it: e02-maskloss-sweep collapsed so), or
    # "batch", at each position the batch's samples (averaged over positions), which a code shared across samples fails.
    sigreg_over: Literal["view", "batch"] = "view"
    stop_grad: bool = False  # detach the targets: the prediction loss trains only the context path, not what it predicts


class Lejepa(nn.Module):
    """LeJEPA: Self-supervised representation learning for volumetric data.

    Combines a 3D Vision Transformer encoder with a multi-layer projection head, a predictor, and the masked-token
    prediction + SIGReg objective: a block of the view's tokens is hidden from the encoder, and the predictor must
    produce those tokens' embeddings (as the encoder makes them from the whole view) from the visible tokens.
    """

    def __init__(self, config: LejepaConfig | None = None, **kwargs: Any):
        super().__init__()
        self.config = c = replace(config or LejepaConfig(), **kwargs)
        self.encoder = ViT3DEncoder(c.encoder)
        self.projector = ProjectorMLP(c.encoder.embed_dim, c.projector) if c.projector else None
        self.predictor = Predictor(c.encoder.embed_dim, c.encoder.num_heads, c.mask.predictor_depth)
        self.sigreg = SIGReg(c.sigreg)
        self.view_maker = ViewMaker(c.view, patch_size=self.encoder.patch_embed.patch_size) if c.view.views != "none" else None
        assert (c.view.n_global, c.view.n_local) == (1, 0), f"one view per sample, got {c.view.n_global} global and {c.view.n_local} local"

        # For FLOP accounting in forward.
        self._n_encoder_params = sum(p.numel() for p in self.encoder.parameters())
        self._n_predictor_params = sum(p.numel() for p in self.predictor.parameters())
        self._n_projector_params = sum(p.numel() for p in self.projector.parameters()) if self.projector is not None else 0

    def encode_maybe_project(self, x: Tensor) -> Tensor:
        """Encode input volume and project to metric space."""
        emb = self.encoder(x)
        if self.projector:
            return self.projector(emb)
        return emb

    def project(self, tokens: Tensor) -> Tensor:
        """Batch N D -> Batch N Dproj: the projector per token (its BatchNorm pools every token of the batch)."""
        return self.projector(tokens.flatten(0, 1)).unflatten(0, tokens.shape[:2]) if self.projector else tokens

    def mask_block(self, grid: tuple[int, ...], batch: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Token indices (visible: Batch Nvis, masked: Batch M) of one cube per sample, at a uniformly random origin."""
        side = mask_side(self.config.mask, grid)
        origin = torch.stack([torch.randint(0, g - s + 1, (batch,), device=device) for g, s in zip(grid, side)], dim=1)  # Batch 3
        coords = torch.stack(torch.meshgrid(*[torch.arange(g, device=device) for g in grid], indexing="ij"), dim=-1).reshape(-1, 3)  # N 3
        inside = ((coords >= origin[:, None]) & (coords < origin[:, None] + torch.tensor(side, device=device))).all(dim=-1)  # Batch N
        order = inside.to(torch.uint8).argsort(dim=1, stable=True)  # the visible tokens, then the masked cube's
        return order[:, :-prod(side)], order[:, -prod(side):]

    def forward(
        self,
        x: Union[Tensor, dict[str, Any]],
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

        view = self.view_maker(vol)[0][0]  # the one crop: Batch C Z Y X
        batch = view.shape[0]
        grid = tuple(s // p for s, p in zip(view.shape[2:], self.encoder.patch_embed.patch_size))
        gather = lambda t, i: torch.gather(t, 1, i.unsqueeze(-1).expand(-1, -1, t.shape[-1]))  # Batch N D, Batch K -> Batch K D

        tokens = self.encoder.forward_features(view)  # Batch N D, from the whole view: the targets
        visible, masked = self.mask_block(grid, batch, view.device)
        context = self.encoder.forward_features(view, keep=visible)  # Batch Nvis D, never having seen the masked cube
        pos = get_3d_sincos_pos_embed(tokens.shape[-1], grid, device=view.device, dtype=tokens.dtype).expand(batch, -1, -1)
        pred = self.project(self.predictor(context, gather(pos, visible), gather(pos, masked)))  # Batch M Dproj
        emb = self.project(tokens)  # Batch N Dproj
        target = gather(emb.detach() if self.config.stop_grad else emb, masked)  # Batch M Dproj
        cloud = emb if self.config.sigreg_over == "view" else emb.transpose(0, 1)  # Batch N Dproj, or N Batch Dproj

        # SIGReg's cos/sin statistics are precision sensitive, so keep the loss in fp32 under autocast.
        with torch.autocast(vol.device.type, enabled=False):
            out = lejepa_loss(pred=pred.float(), target=target.float(), tokens=cloud.float(),
                              sigreg=self.sigreg, lamb=self.config.lamb)
        # Encoder tokens this step: the whole view, then the visible part again (python ints, no device sync).
        out.n_tokens = batch * (prod(grid) + visible.shape[1])
        out.n_flops = batch * train_flops(self.config, grid, self._n_encoder_params, self._n_predictor_params, self._n_projector_params)
        return out
