"""Tests for lib.evals: the PCA maps of an encoder's tokens at each test tile size."""

import json

import torch

from lib import evals
from lib.data import Dataset
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
    monkeypatch.setattr(evals, "crop", lambda volume, shape: torch.rand(1, *shape))
    e = Eval(savedir=f"{tmp_path}/", train=Train(), init_from="outdir/e00/x/a", data="hemibrain")
    assert load(str(dump(e)), Eval) == e  # the spec round-trips, jobs and all
    evals.pca_sizes(e, model, 7)
    rows = [json.loads(line) for line in (tmp_path / "pca_sizes.json").read_text().splitlines()]
    assert [(r["tile"], r["tiles"], r["step"]) for r in rows] == [(8, 2 * 4 * 4, 7), (16, 4, 7)]  # a 16 x 32 x 32 crop
    assert all(0 <= r["between_tile_variance"] <= 1 and r["effective_rank"] > 1 for r in rows)
    for side in (8, 16):  # EM | PCA | tile-centered PCA, at voxel size
        assert Image.open(tmp_path / f"pca_tile_{side:03d}.png").size == (3 * 32 + 2 * 8, 32)


def test_val_loss_measures_the_same_windows_every_time_on_held_out_and_training_volumes(tmp_path, monkeypatch):
    from lib.models import Lejepa
    torch.manual_seed(0)
    model = Lejepa(LejepaConfig(encoder=EncoderConfig(depth=1, embed_dim=16, num_heads=2, patch_size=(4, 4, 4)), projector=ProjectorConfig(dim=8, hidden=16)))
    seen = []
    monkeypatch.setattr(evals, "dataset", lambda name: Dataset(train=[("train", name)], val=[("val", name)]))
    monkeypatch.setattr(evals, "grid_configs", lambda volumes, patch: volumes)

    def grid_windows(configs, n):  # a different fixed image for each (kind, dataset)
        seen.append((configs, n))
        return torch.rand(6, 1, 16, 16, 16, generator=torch.Generator().manual_seed(hash(configs[0]) % 1000))

    monkeypatch.setattr(evals, "grid_windows", grid_windows)
    e = Eval(savedir=str(tmp_path), train=Train(data=("hemibrain", "fish2", "liconn_mouse", "microns"), patch_size=(16, 16, 16), batch_size=4, amp=False), init_from="random")
    evals.val_loss(e, model, 5)
    first = json.loads((tmp_path / "val.json").read_text())
    evals.val_loss(e, model, 5)
    assert json.loads((tmp_path / "val.json").read_text()) == first  # seeded views and masks, fixed windows: the same numbers
    assert first["step"] == 5 and set(first["val"]) == set(first["train"]) == {"hemibrain", "fish2", "liconn_mouse", "microns"}
    assert all({"loss", "inv", "sigreg"} <= set(v) and v["loss"] > 0 for d in ("val", "train") for v in first[d].values())
    assert all(n == evals.VAL_WINDOWS for _, n in seen)
