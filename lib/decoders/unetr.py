"""UNETR: a dense decoder on a ViT3DEncoder. Its intermediate blocks' tokens (all on the same patch grid -- a ViT
has no spatial downsampling, unlike a CNN) are reshaped to that grid and upsampled back to voxel resolution in
log2(patch_size) transposed-conv stages, each with a skip connection into one of those intermediate blocks.
Reference: mia-muvit's muvit/unetr_decoder.py (MuViTUNETR), adapted to ViT3DEncoder's single-view-per-call
interface (no multi-level input, so no level_idx / per-level token offset bookkeeping is needed here).
fit_unetr trains one end to end to predict dense binary targets (e.g. lib.probe.affinities) from raw volumes."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lib.encoders import ViT3DEncoder
from lib.probe import average_precision
from lib.types import Tup3Int


def default_skip_layers(depth: int, num_stages: int) -> tuple[int, ...]:
    """num_stages encoder block indices (0-based), evenly spread over depth and ending at the last block, e.g.
    depth=12 num_stages=3 -> (3, 7, 11). One skip connection per upsample stage (Unetr)."""
    assert depth >= num_stages, f"depth={depth} < num_stages={num_stages}: not enough blocks for one skip each"
    return tuple(round((i + 1) * depth / num_stages) - 1 for i in range(num_stages))


class ConvBlock3d(nn.Module):
    """Two 3x3x3 convolutions, each followed by GroupNorm and GELU."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, 3, padding=1),
            nn.GroupNorm(min(32, out_dim), out_dim),
            nn.GELU(),
            nn.Conv3d(out_dim, out_dim, 3, padding=1),
            nn.GroupNorm(min(32, out_dim), out_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class UnetrConfig:
    out_dim: int = 1  # output channels per voxel
    hidden_dim: int = 64  # decoder channel width; dominates its cost, since every stage runs at full voxel resolution
    skip_layers: tuple[int, ...] | None = None  # one encoder block index per upsample stage; None: default_skip_layers


class Unetr(nn.Module):
    """UNETR dense decoder on encoder (a ViT3DEncoder). encoder.patch_embed.patch_size must be isotropic and a
    power of 2 p: forward() upsamples log2(p) stages from the patch grid back to the input's own voxel resolution.
    Freeze the encoder yourself (encoder.requires_grad_(False)) to train the decoder alone; fit_unetr backprops
    through whatever Unetr.parameters() includes."""

    def __init__(self, encoder: ViT3DEncoder, config: UnetrConfig | None = None, **kwargs):
        super().__init__()
        config = replace(config or UnetrConfig(), **kwargs)
        patch = encoder.patch_embed.patch_size
        assert len(set(patch)) == 1, f"patch_size must be isotropic, got {patch}"
        p = patch[0]
        num_stages = p.bit_length() - 1  # log2, for a power of 2
        assert 2 ** num_stages == p, f"patch_size must be a power of 2, got {p}"
        skip_layers = config.skip_layers or default_skip_layers(encoder.depth, num_stages)
        assert len(skip_layers) == num_stages, f"need {num_stages} skip_layers for patch_size={p}, got {len(skip_layers)}"
        assert list(skip_layers) == sorted(set(skip_layers)), f"skip_layers must be sorted and distinct: {skip_layers}"

        self.encoder = encoder
        self.cfg = config
        self.skip_layers = tuple(skip_layers)
        self.num_stages = num_stages

        dim = encoder.embed_dim
        self.bottleneck = ConvBlock3d(dim, config.hidden_dim)
        self.skip_projs = nn.ModuleList(ConvBlock3d(dim, config.hidden_dim) for _ in range(num_stages))
        self.upsample = nn.ModuleList(nn.ConvTranspose3d(config.hidden_dim, config.hidden_dim, kernel_size=2, stride=2)
                                      for _ in range(num_stages))
        self.decoder_blocks = nn.ModuleList(ConvBlock3d(config.hidden_dim * 2, config.hidden_dim) for _ in range(num_stages))
        self.final = nn.Conv3d(config.hidden_dim, config.out_dim, kernel_size=1)

    def _to_spatial(self, tokens: Tensor, grid: Tup3Int) -> Tensor:
        """Batch N D tokens -> Batch D Gz Gy Gx."""
        gz, gy, gx = grid
        return tokens.reshape(tokens.shape[0], gz, gy, gx, -1).permute(0, 4, 1, 2, 3)

    def forward(self, x: Tensor) -> Tensor:
        """x: Batch C Z Y X. Returns Batch OutDim Z Y X, at x's own resolution."""
        final, intermediates, grid = self.encoder.forward_intermediate(x, self.skip_layers)
        z = self.bottleneck(self._to_spatial(final, grid))
        # Deepest skip (closest to final) pairs with the stage closest to the bottleneck, standard UNETR order.
        for i in reversed(range(self.num_stages)):
            z = self.upsample[i](z)
            skip = self.skip_projs[i](self._to_spatial(intermediates[i], grid))
            # Every skip sits at the one native patch-grid resolution (no pyramid, unlike a CNN): interpolate it
            # up to each stage's growing resolution instead of the usual same-resolution U-Net concat.
            skip = F.interpolate(skip, size=z.shape[2:], mode="trilinear", align_corners=False)
            z = self.decoder_blocks[i](torch.cat([z, skip], dim=1))
        return self.final(z)


def random_crop(img: Tensor, targets: Tensor, valid: Tensor, crop: Tup3Int, g: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
    """One random crop-sized window, same offset into img, targets and valid (all C/OutDim Z Y X)."""
    Z, Y, X = img.shape[-3:]
    z0, y0, x0 = (int(torch.randint(0, s - c + 1, (1,), generator=g)) for s, c in zip((Z, Y, X), crop))
    sl = (slice(z0, z0 + crop[0]), slice(y0, y0 + crop[1]), slice(x0, x0 + crop[2]))
    return img[..., sl[0], sl[1], sl[2]], targets[..., sl[0], sl[1], sl[2]], valid[..., sl[0], sl[1], sl[2]]


def fit_unetr(model: Unetr, img: Tensor, targets: Tensor, valid: Tensor, crop: Tup3Int,
              held: tuple[Tensor, Tensor, Tensor] | None = None,
              steps: int = 3000, batch: int = 4, lr: float = 1e-4) -> list[dict]:
    """Train model end to end on random crop-sized windows of one training volume: img (C Z Y X, in [0, 1]),
    dense binary targets (OutDim Z Y X, e.g. lib.probe.affinities) and valid (OutDim Z Y X, BCE mask). Backprops
    through the decoder and, unless model.encoder was frozen first (encoder.requires_grad_(False)), the encoder
    too. Adam at a fixed lr; seeded, so a refit gives the same model.

    Also returns the fit curve, every 100 steps: the batch loss, and on held (a fixed (img, targets, valid) crop)
    the BCE and the AP of predicting a 0 (for affinities: boundary, different objects). Monitoring only.
    """
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    curve = []
    for step in range(steps):
        crops = [random_crop(img, targets, valid, crop, g) for _ in range(batch)]
        x = torch.stack([c[0] for c in crops]).to(device)
        y = torch.stack([c[1] for c in crops]).to(device).float()
        w = torch.stack([c[2] for c in crops]).to(device).float()
        loss = F.binary_cross_entropy_with_logits(model(x), y, weight=w, reduction="sum") / w.sum().clamp_min(1)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0 or step == steps - 1:
            row = {"step": step, "loss": loss.item()}
            if held is not None:
                with torch.no_grad():
                    hz = model(held[0].unsqueeze(0).to(device))[0]
                    ht, hw = held[1].to(device).float(), held[2].to(device).bool()
                    row["held_bce"] = float(F.binary_cross_entropy_with_logits(hz[hw], ht[hw]))
                    row["held_boundary_ap"] = average_precision(-hz[hw], 1 - ht[hw])
            curve.append(row)
    return curve
