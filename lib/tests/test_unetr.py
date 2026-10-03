"""Tests for lib.decoders.unetr: skip-layer defaults, Unetr's shape/validation, and fit_unetr."""

import pytest
import torch

from lib.decoders import Unetr, UnetrConfig, default_skip_layers, fit_unetr
from lib.encoders import ViT3DEncoder


def test_default_skip_layers():
    assert default_skip_layers(12, 3) == (3, 7, 11)
    assert default_skip_layers(6, 3) == (1, 3, 5)
    assert default_skip_layers(4, 3) == (0, 2, 3)
    assert default_skip_layers(3, 3) == (0, 1, 2)
    with pytest.raises(AssertionError):
        default_skip_layers(2, 3)


def test_unetr_forward_shape():
    encoder = ViT3DEncoder(patch_size=(4, 4, 4), embed_dim=8, depth=6, num_heads=2)
    model = Unetr(encoder, UnetrConfig(out_dim=3, hidden_dim=16))
    assert model.skip_layers == (2, 5)  # default_skip_layers(6, 2): log2(4) = 2 stages
    x = torch.randn(2, 1, 8, 8, 8)  # grid 2 2 2
    out = model(x)
    assert out.shape == (2, 3, 8, 8, 8)  # back to x's own resolution, out_dim channels


def test_unetr_rejects_non_power_of_two_patch_size():
    encoder = ViT3DEncoder(patch_size=(6, 6, 6), embed_dim=8, depth=4, num_heads=2)
    with pytest.raises(AssertionError):
        Unetr(encoder)


def test_unetr_rejects_anisotropic_patch_size():
    encoder = ViT3DEncoder(patch_size=(4, 8, 8), embed_dim=8, depth=4, num_heads=2)
    with pytest.raises(AssertionError):
        Unetr(encoder)


def test_unetr_backprops_into_encoder_unless_frozen():
    encoder = ViT3DEncoder(patch_size=(2, 2, 2), embed_dim=8, depth=3, num_heads=2)
    model = Unetr(encoder, UnetrConfig(out_dim=1, hidden_dim=8))
    model(torch.randn(1, 1, 8, 8, 8)).sum().backward()
    assert any(p.grad is not None for p in encoder.parameters())

    encoder2 = ViT3DEncoder(patch_size=(2, 2, 2), embed_dim=8, depth=3, num_heads=2)
    encoder2.requires_grad_(False)
    model2 = Unetr(encoder2, UnetrConfig(out_dim=1, hidden_dim=8))
    model2(torch.randn(1, 1, 8, 8, 8)).sum().backward()
    assert all(p.grad is None for p in encoder2.parameters())


def test_fit_unetr_learns_a_planted_boundary():
    g = torch.Generator().manual_seed(0)
    Z, Y, X = 16, 16, 16
    img = torch.rand(1, Z, Y, X, generator=g)
    labels = torch.ones(Z, Y, X, dtype=torch.int64)
    labels[:, :, X // 2:] = 2  # split along x at the midpoint
    from lib.probe import affinities
    targets, valid = affinities(labels, [(0, 0, 1)])  # OutDim=1 Z Y X

    encoder = ViT3DEncoder(patch_size=(2, 2, 2), embed_dim=16, depth=3, num_heads=2)
    model = Unetr(encoder, UnetrConfig(out_dim=1, hidden_dim=16))
    held = (img, targets, valid)
    curve = fit_unetr(model, img, targets, valid, crop=(8, 8, 8), held=held, steps=200, batch=2, lr=1e-3)
    assert curve[0]["step"] == 0 and curve[-1]["step"] == 199
    assert curve[-1]["held_bce"] < curve[0]["held_bce"]
