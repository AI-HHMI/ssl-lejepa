"""3D Vision Transformer (ViT3D) encoder for volumetric microscopy data."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: Tuple[int, int, int],
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
    """3D Volume to Patch Embedding with dynamic padding."""

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]] = (8, 8, 8),
        in_channels: int = 1,
        embed_dim: int = 512,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int, int]]:
        """Extract patch tokens from 3D volume.

        Args:
            x: (B, C, Z, Y, X)

        Returns:
            tokens: (B, N, embed_dim) where N = G_z * G_y * G_x
            grid_size: (G_z, G_y, G_x)
        """
        pz, py, px = self.patch_size
        _, _, z, y, x_dim = x.shape

        # Dynamic padding if dimensions are not divisible by patch size
        pad_z = (pz - z % pz) % pz
        pad_y = (py - y % py) % py
        pad_x = (px - x_dim % px) % px

        if pad_z or pad_y or pad_x:
            x = F.pad(x, (0, pad_x, 0, pad_y, 0, pad_z), mode="replicate")

        feat = self.proj(x)  # (B, embed_dim, G_z, G_y, G_x)
        grid_size = (feat.shape[2], feat.shape[3], feat.shape[4])
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return tokens, grid_size


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
        patch_size: Union[int, Tuple[int, int, int]] = (8, 8, 8),
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
            tokens: (B, N, embed_dim)
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

        x = self.norm(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with global average pooling (mean over tokens).

        Args:
            x: (B, C, Z, Y, X)

        Returns:
            embedding: (B, embed_dim)
        """
        tokens = self.forward_features(x)
        return tokens.mean(dim=1)
