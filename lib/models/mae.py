"""MAE: masked autoencoding (He et al. 2022) on the same ViT3D encoder as Lejepa, the pixel-space baseline it is compared to.
As mia-train's src/algorithms/mae.py (Gary's): random 75% of the tokens hidden, the encoder sees the rest, a small decoder
reconstructs every hidden patch's voxels, each patch normalized (norm_pix_loss); the decoder is discarded after training."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod
from typing import Any, Union

import torch
import torch.nn as nn
from torch import Tensor

from lib.encoders import EncoderConfig, TransformerBlock, ViT3DEncoder, get_3d_sincos_pos_embed
from lib.losses import LejepaOutput
from lib.models.lejepa import as_volume
from lib.views import ViewConfig, ViewMaker


@dataclass
class MaeConfig:
    """Hyperparameters for MAE: the encoder's and view's configs, whole, plus the masking and the decoder (mia-train's defaults)."""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    view: ViewConfig = field(default_factory=ViewConfig)  # the one crop each sample trains on (n_global=1, n_local=0)
    mask_ratio: float = 0.75  # the share of each view's tokens hidden from the encoder, at random
    decoder_dim: int = 192
    decoder_depth: int = 2
    decoder_heads: int = 6
    norm_pix: bool = True  # each target patch normalized to zero mean, unit variance (MAE's norm_pix_loss)


class Mae(nn.Module):
    """Masked autoencoder: the encoder embeds a random quarter of a view's tokens; the decoder, given them and one learned mask
    token per hidden position (each with its position's sincos embedding), predicts the hidden patches' voxels. The loss is
    the mean squared error over the hidden patches only (scoring the visible ones would reward copying the input)."""

    def __init__(self, config: MaeConfig | None = None, **kwargs: Any):
        super().__init__()
        self.config = c = replace(config or MaeConfig(), **kwargs)
        self.encoder = ViT3DEncoder(c.encoder)
        self.view_maker = ViewMaker(c.view, patch_size=self.encoder.patch_embed.patch_size) if c.view.views != "none" else None
        assert (c.view.n_global, c.view.n_local) == (1, 0), f"one view per sample, got {c.view.n_global} global and {c.view.n_local} local"
        assert 0 < c.mask_ratio < 1, f"mask_ratio must be in (0, 1), got {c.mask_ratio}"
        self.decoder_embed = nn.Linear(c.encoder.embed_dim, c.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, c.decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder_blocks = nn.ModuleList(TransformerBlock(dim=c.decoder_dim, num_heads=c.decoder_heads) for _ in range(c.decoder_depth))
        self.decoder_norm = nn.LayerNorm(c.decoder_dim)
        self.decoder_head = nn.Linear(c.decoder_dim, c.encoder.in_channels * prod(self.encoder.patch_embed.patch_size))

        # For FLOP accounting in forward.
        self._n_encoder_params = sum(p.numel() for p in self.encoder.parameters())
        self._n_decoder_params = sum(p.numel() for n, p in self.named_parameters() if not n.startswith("encoder."))

    def forward(self, x: Union[Tensor, dict[str, Any]], return_loss: bool = True, **kwargs: Any) -> Union[Tensor, LejepaOutput]:
        vol = as_volume(x, self)
        if not return_loss or self.view_maker is None:
            return self.encoder(vol)
        c = self.config
        view = self.view_maker(vol)[0][0]  # the one crop: Batch C Z Y X
        (pz, py, px), (Z, Y, X) = self.encoder.patch_embed.patch_size, view.shape[2:]
        batch, grid = view.shape[0], (Z // pz, Y // py, X // px)
        n = prod(grid)
        keep = max(1, round(n * (1 - c.mask_ratio)))
        gather = lambda t, i: torch.gather(t, 1, i.unsqueeze(-1).expand(-1, -1, t.shape[-1]))  # Batch N D, Batch K -> Batch K D

        order = torch.rand(batch, n, device=view.device).argsort(dim=1)  # Batch N: the visible tokens, then the hidden ones
        latent = self.encoder.forward_features(view, keep=order[:, :keep])  # Batch Nvis D
        dec = torch.cat([self.decoder_embed(latent), self.mask_token.expand(batch, n - keep, -1).to(latent.dtype)], dim=1)
        # Attention is order-free, so the decoder works in `order`'s order: each token carries its own position.
        dec = dec + gather(get_3d_sincos_pos_embed(c.decoder_dim, grid, device=view.device, dtype=dec.dtype).expand(batch, -1, -1), order)
        for block in self.decoder_blocks:
            dec = block(dec)
        pred = self.decoder_head(self.decoder_norm(dec[:, keep:]))  # Batch M (C pz py px), the hidden patches
        target = gather(self.encoder.patch_embed.patchify(view), order[:, keep:])
        with torch.autocast(vol.device.type, enabled=False):
            target = target.float()
            if c.norm_pix:
                target = (target - target.mean(dim=-1, keepdim=True)) / (target.var(dim=-1, keepdim=True) + 1e-6).sqrt()
            out = LejepaOutput(loss=(pred.float() - target).square().mean())
        # Encoder tokens this step: the visible ones (python ints, no device sync). FLOPs as lejepa.train_flops: per token,
        # 6 x params plus 12 x depth x width x the tokens it attends to; the encoder over the visible, the decoder over all.
        enc = c.encoder
        out.n_tokens = batch * keep
        out.n_flops = batch * (keep * (6 * self._n_encoder_params + 12 * enc.depth * enc.embed_dim * keep)
                               + n * (6 * self._n_decoder_params + 12 * c.decoder_depth * c.decoder_dim * n))
        return out
