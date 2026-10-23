"""Tests for lib.evals: the PCA maps of an encoder's tokens at each test tile size."""

import json

import torch

from lib import evals
from lib.encoders import EncoderConfig
from lib.models import Lejepa, LejepaConfig, ProjectorConfig
from lib.specs import Eval, Train


def test_pca_sizes_maps_one_crop_at_each_test_tile(tmp_path, monkeypatch):
    """Every tile side runs over the same crop, 2 x 2 tiles of the largest in y x and one deep; one png and json row per side."""
    from PIL import Image
    from lib.util import dump, load
    torch.manual_seed(0)
    model = Lejepa(LejepaConfig(encoder=EncoderConfig(depth=1, embed_dim=16, num_heads=2, patch_size=(4, 4, 4)), projector=ProjectorConfig(dim=8, hidden=16))).eval()
    monkeypatch.setattr(evals, "TEST_TILES", (8, 16))
    monkeypatch.setattr(evals, "eb_val_crop", lambda shape: torch.rand(1, *shape))
    e = Eval(savedir=f"{tmp_path}/", train=Train(), init_from="outdir/e00/x/d0", jobs=("pca_sizes",))
    assert load(str(dump(e)), Eval) == e  # the spec round-trips, jobs and all
    evals.pca_sizes(e, model, 7)
    rows = [json.loads(line) for line in (tmp_path / "pca_sizes.json").read_text().splitlines()]
    assert [(r["tile"], r["tiles"], r["step"]) for r in rows] == [(8, 2 * 4 * 4, 7), (16, 4, 7)]  # a 16 x 32 x 32 crop
    assert all(0 <= r["between_tile_variance"] <= 1 and r["effective_rank"] > 1 for r in rows)
    for side in (8, 16):  # EM | PCA | tile-centered PCA, at voxel size
        assert Image.open(tmp_path / f"pca_tile_{side:03d}.png").size == (3 * 32 + 2 * 8, 32)
