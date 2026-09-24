"""Unit tests for lib package modules."""

import pytest
import torch

from lib.encoders import ViT3DEncoder, get_3d_sincos_pos_embed
from lib.losses import SIGReg, lejepa_loss
from lib.models import Lejepa, LejepaConfig
from lib.views import ViewMaker


def test_lejepa_config():
    cfg = LejepaConfig(n_layers=12, width=512, views="basic")
    assert cfg.n_layers == 12
    assert cfg.num_layers == 12
    assert cfg.width == 512
    assert cfg.dim == 512
    assert cfg.embed_dim == 512
    assert cfg.views == "basic"
    assert cfg.num_heads == 8


def test_pos_embed_3d():
    pos = get_3d_sincos_pos_embed(embed_dim=64, grid_size=(4, 8, 8))
    assert pos.shape == (1, 4 * 8 * 8, 64)


def test_vit3d_encoder_forward():
    encoder = ViT3DEncoder(
        in_channels=1,
        patch_size=(8, 8, 8),
        embed_dim=64,
        depth=2,
        num_heads=2,
    )
    x = torch.randn(2, 1, 16, 24, 24)
    out = encoder(x)
    assert out.shape == (2, 64)


def test_sigreg():
    sigreg = SIGReg(num_slices=32, knots=9, t_max=3.0)
    # (n_views, batch, features)
    proj = torch.randn(6, 4, 128)
    loss = sigreg(proj)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_lejepa_loss():
    sigreg = SIGReg(num_slices=32, knots=9, t_max=3.0)
    globals_ = torch.randn(2, 4, 64)
    locals_ = torch.randn(4, 4, 64)
    all_views = torch.cat([globals_, locals_], dim=0)

    res = lejepa_loss(globals_, all_views, sigreg, lamb=0.02)
    assert "loss" in res
    assert "inv" in res
    assert "sigreg" in res
    assert "weighted_sigreg" in res


def test_view_maker():
    maker = ViewMaker(n_global=2, n_local=4, global_scale=(0.5, 1.0), local_scale=(0.15, 0.5))
    x = torch.randn(2, 1, 32, 32, 32)
    globals_, locals_ = maker(x)
    assert len(globals_) == 2
    assert len(locals_) == 4
    for g in globals_:
        assert g.shape[0] == 2
        assert g.shape[1] == 1


def test_model_forward_and_backward():
    cfg = LejepaConfig(n_layers=2, width=64, num_heads=2, patch_size=(8, 8, 8), views="basic")
    model = Lejepa(cfg)

    x = torch.randn(2, 1, 32, 32, 32)
    res = model(x)
    assert "loss" in res
    loss = res["loss"]
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert len(grads) > 0
    assert all(g is not None for g in grads)


def test_view_maker_crops_are_flipped_subblocks():
    B, Z, Y, X = 3, 16, 24, 24
    x = torch.arange(B * Z * Y * X, dtype=torch.float32).view(B, 1, Z, Y, X)
    maker = ViewMaker(n_global=2, n_local=2, patch_size=(8, 8, 8), generator=torch.Generator().manual_seed(0))
    globals_, locals_ = maker(x)
    for v in globals_ + locals_:
        assert all(s % 8 == 0 for s in v.shape[2:])
        for b in range(B):
            crop = v[b, 0]
            # Undo flips: values increase along every axis of an unflipped crop of an arange volume.
            for d in range(3):
                if crop.select(d, 0).min() > crop.select(d, -1).min():
                    crop = crop.flip(d)
            z0, rem = divmod(int(crop.min()) - b * Z * Y * X, Y * X)
            y0, x0 = divmod(rem, X)
            cz, cy, cx = crop.shape
            assert torch.equal(crop, x[b, 0, z0:z0 + cz, y0:y0 + cy, x0:x0 + cx])


def test_view_maker_displace_locals_inside_globals():
    B, Z, Y, X = 4, 48, 64, 64
    x = torch.arange(B * Z * Y * X, dtype=torch.float32).view(B, 1, Z, Y, X)
    maker = ViewMaker(n_global=2, n_local=3, flip=False, patch_size=(8, 8, 8), views="displace",
                      global_size=(32, 48, 48), local_size=(16, 24, 24), generator=torch.Generator().manual_seed(0))
    globals_, locals_ = maker(x)
    assert all(g.shape == (B, 1, 32, 48, 48) for g in globals_)
    assert all(l.shape == (B, 1, 16, 24, 24) for l in locals_)
    for l in locals_:
        for b in range(B):
            # Unflipped: the local's values are a contiguous sub-block of one of this sample's globals.
            assert any(set(l[b].flatten().tolist()) <= set(g[b].flatten().tolist()) for g in globals_)
