"""3D Vision Transformer (ViT3D) encoder for volumetric microscopy data."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lib.types import Tup3Int


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: Tup3Int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Compute 3D sinusoidal position embeddings for a given (G_z, G_y, G_x) grid.

    Args:
        embed_dim: Embedding dimension.
        grid_size: Tuple of (G_z, G_y, G_x) patch grid dimensions.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape (1, G_z * G_y * G_x, embed_dim).
    """
    gz, gy, gx = grid_size
    dim_z = embed_dim // 3
    dim_y = embed_dim // 3
    dim_x = embed_dim - dim_z - dim_y

    # Ensure even dims for sin/cos pairs
    if dim_z % 2 != 0:
        dim_z -= 1
        dim_x += 1
    if dim_y % 2 != 0:
        dim_y -= 1
        dim_x += 1

    def get_1d_sincos(dim: int, length: int) -> Tensor:
        omega = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        pos = torch.arange(length, dtype=torch.float32)
        out = torch.einsum("m,d->md", pos, omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    emb_z = get_1d_sincos(dim_z, gz)
    emb_y = get_1d_sincos(dim_y, gy)
    emb_x = get_1d_sincos(dim_x, gx)

    # Broadcast to 3D grid: (gz, gy, gx, dim)
    emb_z = emb_z.view(gz, 1, 1, dim_z).expand(gz, gy, gx, dim_z)
    emb_y = emb_y.view(1, gy, 1, dim_y).expand(gz, gy, gx, dim_y)
    emb_x = emb_x.view(1, 1, gx, dim_x).expand(gz, gy, gx, dim_x)

    pos_embed = torch.cat([emb_z, emb_y, emb_x], dim=-1).reshape(1, gz * gy * gx, embed_dim)
    if device is not None:
        pos_embed = pos_embed.to(device=device, dtype=dtype)
    return pos_embed


class PatchEmbed3d(nn.Module):
    """3D volume -> patch tokens: cut into non-overlapping patches, then one Linear.

    Same math as a Conv3d with stride == kernel, but as one well-shaped matmul: on B300 the conv's
    implicit-GEMM kernels took ~15-17% of GPU time (e00/b300-*). Checkpoints from the Conv3d version load
    unchanged (their 5-D proj.weight is flattened in the same C pz py px order).
    """

    def __init__(
        self,
        patch_size: Union[int, Tup3Int] = (8, 8, 8),
        in_channels: int = 1,
        embed_dim: int = 512,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Linear(in_channels * patch_size[0] * patch_size[1] * patch_size[2], embed_dim)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        w = state_dict.get(prefix + "proj.weight")
        if w is not None and w.ndim == 5:  # Conv3d weight: embed_dim C pz py px
            state_dict[prefix + "proj.weight"] = w.reshape(w.shape[0], -1)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tup3Int]:
        """Extract patch tokens from 3D volume.

        Args:
            x: (B, C, Z, Y, X)

        Returns:
            tokens: (B, N, embed_dim) where N = G_z * G_y * G_x
            grid_size: (G_z, G_y, G_x)
        """
        assert all(s % p == 0 for s, p in zip(x.shape[2:], self.patch_size)), (
            f"input {tuple(x.shape)} must be a multiple of patch size {self.patch_size}"
        )
        B, C, Z, Y, X = x.shape
        pz, py, px = self.patch_size
        gz, gy, gx = Z // pz, Y // py, X // px
        # Batch C Z Y X -> Batch (Gz Gy Gx) (C pz py px), matching the Conv3d weight layout
        patches = x.reshape(B, C, gz, pz, gy, py, gx, px).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, gz * gy * gx, -1)
        return self.proj(patches), (gz, gy, gx)


class Attention(nn.Module):
    """Multi-head Self-Attention with scaled dot-product attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        dropout_p = self.attn_drop if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TransformerBlock(nn.Module):
    """Standard Pre-LayerNorm Transformer Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attention_dropout,
            proj_drop=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT3DEncoder(nn.Module):
    """3D Vision Transformer Encoder."""

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: Union[int, Tup3Int] = (8, 8, 8),
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        qkv_bias: bool = True,
        token_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.token_dropout = token_dropout

        self.patch_embed = PatchEmbed3d(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                qkv_bias=qkv_bias,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward_features(self, x: Tensor) -> Tensor:
        """Extract patch token representations.

        Args:
            x: (B, C, Z, Y, X)

        Returns:
            tokens: (B, N, embed_dim), after the final LayerNorm
        """
        return self.norm(self.forward_residual(x))

    def forward_residual(self, x: Tensor) -> Tensor:
        """Patch tokens from the residual stream after the last block, before the final LayerNorm.

        The LayerNorm fixes every token's norm to ~sqrt(embed_dim), so token-norm analyses
        (e.g. high-norm "register" tokens) need these. x: (B, C, Z, Y, X) -> (B, N, embed_dim).
        """
        tokens, grid_size = self.patch_embed(x)
        pos_embed = get_3d_sincos_pos_embed(
            self.embed_dim,
            grid_size,
            device=x.device,
            dtype=tokens.dtype,
        )
        x = tokens + pos_embed

        # Token dropout (random patch token drop during training)
        if self.training and self.token_dropout > 0.0:
            B, N, C = x.shape
            keep = max(1, int(N * (1.0 - self.token_dropout)))
            indices = torch.stack([
                torch.randperm(N, device=x.device)[:keep]
                for _ in range(B)
            ])
            x = torch.gather(x, 1, indices.unsqueeze(-1).expand(B, keep, C))

        for block in self.blocks:
            x = block(x)
        return x

    def forward_intermediate(self, x: Tensor, layer_idxs: Tuple[int, ...]) -> Tuple[Tensor, list[Tensor], Tup3Int]:
        """Same residual stream as forward_residual (final block, pre-LayerNorm), plus each requested block's own
        output (also pre-LayerNorm) -- for a decoder's skip connections (e.g. lib.decoders.unetr.Unetr), which need
        the complete, correctly shaped (Gz Gy Gx) token grid, so token dropout never applies here regardless of
        self.training.

        x: Batch C Z Y X. layer_idxs: 0-based block indices, each < depth.
        Returns (final Batch N D, [one Batch N D per layer_idxs, in that order], (Gz Gy Gx)).
        """
        assert all(0 <= i < self.depth for i in layer_idxs), f"layer_idxs {layer_idxs} out of range for depth {self.depth}"
        tokens, grid_size = self.patch_embed(x)
        pos_embed = get_3d_sincos_pos_embed(self.embed_dim, grid_size, device=x.device, dtype=tokens.dtype)
        x = tokens + pos_embed
        saved: dict[int, Tensor] = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in layer_idxs:
                saved[i] = x
        return x, [saved[i] for i in layer_idxs], grid_size

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with global average pooling (mean over tokens).

        Args:
            x: (B, C, Z, Y, X)

        Returns:
            embedding: (B, embed_dim)
        """
        tokens = self.forward_features(x)
        return tokens.mean(dim=1)
