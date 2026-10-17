"""LeJEPA: Self-Supervised Learning with masked-token prediction and SIGReg for 3D Microscopy."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod
from typing import Any, Literal, Union

import torch
import torch.nn as nn
from torch import Tensor

from lib.encoders import EncoderConfig, TransformerBlock, ViT3DEncoder, get_3d_sincos_pos_embed, sincos_at
from lib.losses import LejepaOutput, SIGReg, SIGRegConfig, lejepa_loss
from lib.views import ViewConfig, ViewMaker


def crop(x: Tensor, origin: Tensor, size: tuple[int, ...]) -> Tensor:
    """Batch C Z Y X -> Batch C *size: each sample's box at its own origin (Batch 3), in one indexing op."""
    iz, iy, ix = [origin[:, a, None] + torch.arange(s, device=x.device) for a, s in enumerate(size)]  # Batch s, per axis
    b, c = torch.arange(x.shape[0], device=x.device), torch.arange(x.shape[1], device=x.device)
    return x[b[:, None, None, None, None], c[None, :, None, None, None], iz[:, None, :, None, None], iy[:, None, None, :, None], ix[:, None, None, None, :]]


def as_volume(x: Union[Tensor, dict[str, Any]], model: nn.Module) -> Tensor:
    """A model's input as Batch C Z Y X on its device: a tensor of 3 to 5 dims, or a Miao VolumeDataset sample (e.g. dl[0])."""
    vol = x["img"] if isinstance(x, dict) else x
    assert 3 <= vol.dim() <= 5, f"Expected 3D, 4D, or 5D input, got tensor with shape {tuple(vol.shape)}"
    vol = vol.reshape((1,) * (5 - vol.dim()) + tuple(vol.shape))
    return vol.to(next(model.parameters()).device)


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
    # >0: the targets are the tokens of a second view of the sample, up to this many voxels (per axis, a multiple of the patch)
    # off the context's: both cut at random from the crop ViewMaker makes, each that much smaller (Lejepa.forward). Their token
    # grids are then offset by a random, fractional number of tokens, so only content lines up between them, not a code of
    # each view's own token positions (the shortcut e02-e04 took). 0: the targets are the context view's own tokens.
    shift: int = 0


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
    # "views" (LeJEPA's own, all runs before e02): each view's tokens pooled to one embedding, every view of a sample pulled
    # to the mean of its globals, SIGReg over the batch. "mask": masked-token prediction (Lejepa.mask_loss, from e02 on).
    objective: Literal["views", "mask"] = "mask"
    view: ViewConfig = field(default_factory=ViewConfig)  # "mask": the one crop each sample trains on (n_global=1, n_local=0)
    batch_views: bool = False  # "views": one encoder call per group of same-shape views (projector BN then pools the group)
    mask: MaskConfig = field(default_factory=MaskConfig)  # "mask" only
    sigreg: SIGRegConfig = field(default_factory=SIGRegConfig)
    lamb: float = 0.02  # loss = inv + lamb * sigreg
    # SIGReg's samples: "view", each view's N tokens (a code of position alone passes it: e02-maskloss-sweep collapsed so);
    # "batch", at each position the batch's samples (averaged over positions), which a code of the sample alone passes
    # (e03-anticollapse collapsed so: between-tile variance ~1); or "both" terms, which neither code passes (sigreg_term).
    sigreg_over: Literal["view", "batch", "both"] = "view"
    stop_grad: bool = False  # detach the targets: the prediction loss trains only the context path, not what it predicts


class Lejepa(nn.Module):
    """LeJEPA: Self-supervised representation learning for volumetric data.

    Combines a 3D Vision Transformer encoder with a multi-layer projection head and SIGReg, under one of two objectives
    (LejepaConfig.objective). "views": multi-view invariance of each view's pooled embedding. "mask": masked-token
    prediction, with a predictor: a block of the view's tokens is hidden from the encoder, and the predictor must
    produce those tokens' embeddings (as the encoder makes them from the whole view) from the visible tokens. With a
    shift (MaskConfig.shift) the targets are instead a second, offset view's tokens over the same voxels.
    """

    def __init__(self, config: LejepaConfig | None = None, **kwargs: Any):
        super().__init__()
        self.config = c = replace(config or LejepaConfig(), **kwargs)
        self.encoder = ViT3DEncoder(c.encoder)
        self.projector = ProjectorMLP(c.encoder.embed_dim, c.projector) if c.projector else None
        # Only "mask" has a predictor, so the two objectives' checkpoints have different keys (a strict load tells them apart).
        self.predictor = Predictor(c.encoder.embed_dim, c.encoder.num_heads, c.mask.predictor_depth) if c.objective == "mask" else None
        self.sigreg = SIGReg(c.sigreg)
        self.view_maker = ViewMaker(c.view, patch_size=self.encoder.patch_embed.patch_size) if c.view.views != "none" else None
        b1 = c.objective == "views"
        b2 = (c.view.n_global, c.view.n_local) == (1, 0)
        assert b1 or b2, f"masked tokens: one view per sample, got {c.view.n_global} global and {c.view.n_local} local"
        b3 = c.mask.shift == 0
        b4 = c.objective == "mask" and c.view.views == "displace" and all(c.mask.shift % p == 0 for p in self.encoder.patch_embed.patch_size)
        assert b3 or b4, f"a shift ({c.mask.shift}) needs masked tokens, displace views (got {c.view.views!r}) and a multiple of the patch {self.encoder.patch_embed.patch_size}"

        # For FLOP accounting in forward.
        self._n_encoder_params = sum(p.numel() for p in self.encoder.parameters())
        self._n_predictor_params = sum(p.numel() for p in self.predictor.parameters()) if self.predictor is not None else 0
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

    def project(self, tokens: Tensor) -> Tensor:
        """Batch N D -> Batch N Dproj: the projector per token (its BatchNorm pools every token of the batch)."""
        return self.projector(tokens.flatten(0, 1)).unflatten(0, tokens.shape[:2]) if self.projector else tokens

    def sigreg_term(self, emb: Tensor) -> Tensor:
        """SIGReg of the projected tokens emb (Batch N D) over config.sigreg_over's samples: each view's N tokens, each
        position's batch, or both. A term's pull is lamb x its sample count, so "both" scales the view term by Batch / N:
        each then pulls lamb x Batch, as "batch" alone does."""
        over = self.config.sigreg_over
        if over == "view":
            return self.sigreg(emb)
        across = self.sigreg(emb.transpose(0, 1))  # N Batch D: at each position, the batch's samples
        return across if over == "batch" else across + emb.shape[0] / emb.shape[1] * self.sigreg(emb)

    def mask_block(self, grid: tuple[int, ...], offset: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """One cube of the context view's tokens per sample, at a uniformly random origin among those that keep it inside the
        target view, offset voxels (Batch 3) from the context. (visible: Batch Nvis, the context's tokens outside the cube;
        masked: Batch M, the target view's tokens whose centres fall inside it, M the cube's; their coordinates in the
        context's token grid, Batch M 3, off the integers by offset / patch.) offset 0: the cube's own tokens, at its own."""
        device, s, patch = offset.device, mask_side(self.config.mask, grid), self.encoder.patch_embed.patch_size
        assert all(n - m >= -(-self.config.mask.shift // q) for n, m, q in zip(grid, s, patch)), (
            f"a mask cube {s} in grid {grid} does not fit both views at a shift of {self.config.mask.shift}")
        side, g, p = [torch.tensor(t, device=device) for t in (s, grid, patch)]
        lo = (-(-offset // p)).clamp(min=0)  # ceil(offset / p): the target's first whole token in the context's grid
        hi = g - side + (offset // p).clamp(max=0)
        origin = lo + (torch.rand(offset.shape, device=device) * (hi - lo + 1)).long()  # Batch 3
        grid_coords = lambda shape: torch.stack(torch.meshgrid(*[torch.arange(n, device=device) for n in shape], indexing="ij"), dim=-1).reshape(-1, 3)
        coords = grid_coords(grid)  # N 3
        inside = ((coords >= origin[:, None]) & (coords < origin[:, None] + side)).all(dim=-1)  # Batch N
        order = inside.to(torch.uint8).argsort(dim=1, stable=True)  # the visible tokens, then the masked cube's
        # The target token k's centre, offset + (k + 1/2) p in the context's voxels, is in the cube [origin p, (origin + side) p)
        # from k = ceil(origin - offset / p - 1/2) on.
        first = -((2 * offset + p - 2 * origin * p) // (2 * p))  # Batch 3
        k = first[:, None] + grid_coords(s)  # Batch M 3, in the cube's (Z-major) order, as order's
        return order[:, :-prod(s)], (k * torch.tensor([grid[1] * grid[2], grid[2], 1], device=device)).sum(-1), k + offset[:, None] / p

    def forward(
        self,
        x: Union[Tensor, dict[str, Any]],
        return_loss: bool = True,
        **kwargs: Any,
    ) -> Union[Tensor, LejepaOutput]:
        vol = as_volume(x, self)
        # If views are disabled or inference mode requested, return embeddings directly
        if not return_loss or self.view_maker is None:
            return self.encode_maybe_project(vol)
        return self.views_loss(vol) if self.config.objective == "views" else self.mask_loss(vol)

    def views_loss(self, vol: Tensor) -> LejepaOutput:
        """The "views" objective on vol (Batch C Z Y X): every view's pooled embedding pulled to the mean of its sample's
        globals (inv), SIGReg over the batch of each view."""
        assert self.view_maker is not None
        g_views, l_views = self.view_maker(vol)
        proj_globals = self.encode_views(g_views)  # Global Batch D
        proj_all = torch.cat([proj_globals, self.encode_views(l_views)]) if l_views else proj_globals  # View Batch D
        # SIGReg's cos/sin statistics are precision sensitive, so keep the loss in fp32 under autocast.
        with torch.autocast(vol.device.type, enabled=False):
            emb = proj_all.float()
            out = lejepa_loss(pred=proj_globals.float().mean(dim=0, keepdim=True).expand_as(emb), target=emb, sig=self.sigreg(emb), lamb=self.config.lamb)
        # Encoder tokens this step, summed over views and batch (python int, no device sync).
        patch_vol = prod(self.encoder.patch_embed.patch_size)
        out.n_tokens = sum(v.shape[0] * prod(v.shape[2:]) // patch_vol for v in [*g_views, *l_views])
        # Per token, 6 x encoder params for the weights plus 12 x depth x width x N for attention over its N-token view;
        # per sample, 6 x projector params (as train_flops).
        enc = self.config.encoder
        out.n_flops = sum(
            v.shape[0] * (n * (6 * self._n_encoder_params + 12 * enc.depth * enc.embed_dim * n) + 6 * self._n_projector_params)
            for v in [*g_views, *l_views]
            for n in [prod(v.shape[2:]) // patch_vol]
        )
        return out

    def mask_loss(self, vol: Tensor) -> LejepaOutput:
        """The "mask" objective on vol (Batch C Z Y X): the masked tokens' embeddings predicted from the visible ones (inv),
        SIGReg over config.sigreg_over's samples."""
        assert self.predictor is not None and self.view_maker is not None
        region = self.view_maker(vol)[0][0]  # the one crop, Batch C Z Y X: both views' region (each is it, without a shift)
        batch, shift = region.shape[0], self.config.mask.shift
        size = tuple(s - shift for s in region.shape[2:])
        origins = torch.randint(0, shift + 1, (2, batch, 3), device=region.device)  # context's, target's
        view, target_view = crop(region, origins[0], size), crop(region, origins[1], size)
        grid = tuple(s // p for s, p in zip(size, self.encoder.patch_embed.patch_size))
        gather = lambda t, i: torch.gather(t, 1, i.unsqueeze(-1).expand(-1, -1, t.shape[-1]))  # Batch N D, Batch K -> Batch K D

        tokens = self.encoder.forward_features(target_view)  # Batch N D, from the whole view: the targets
        visible, masked, masked_coords = self.mask_block(grid, origins[1] - origins[0])
        context = self.encoder.forward_features(view, keep=visible)  # Batch Nvis D, never having seen the masked cube
        pos = get_3d_sincos_pos_embed(tokens.shape[-1], grid, device=view.device, dtype=tokens.dtype).expand(batch, -1, -1)
        pred = self.project(self.predictor(context, gather(pos, visible), sincos_at(masked_coords, tokens.shape[-1]).to(tokens.dtype)))  # Batch M Dproj
        emb = self.project(tokens)  # Batch N Dproj
        target = gather(emb.detach() if self.config.stop_grad else emb, masked)  # Batch M Dproj

        # SIGReg's cos/sin statistics are precision sensitive, so keep the loss in fp32 under autocast.
        with torch.autocast(vol.device.type, enabled=False):
            out = lejepa_loss(pred=pred.float(), target=target.float(), sig=self.sigreg_term(emb.float()), lamb=self.config.lamb)
        # Encoder tokens this step: the whole view, then the visible part again (python ints, no device sync).
        out.n_tokens = batch * (prod(grid) + visible.shape[1])
        out.n_flops = batch * train_flops(self.config, grid, self._n_encoder_params, self._n_predictor_params, self._n_projector_params)
        return out
