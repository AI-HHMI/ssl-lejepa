"""Tests for lib.probe and experiment.probe: affinity targets, token layout, fit, AP, and the probe end to end."""

import json
from types import SimpleNamespace

import numpy as np
import torch
import zarr

import experiment
from lib.models import Lejepa, LejepaConfig
from lib.probe import AFFINITY_OFFSETS_XYZ, affinities, average_precision, fit_probe, to_tokens, to_voxels


def test_affinities_mark_the_split_between_two_objects():
    labels = torch.ones(4, 4, 6, dtype=torch.int64)
    labels[..., 3:] = 2  # two objects split along x, between x = 2 and 3
    aff, valid = affinities(labels, [(0, 0, 1), (1, 0, 0), (0, 0, 3)])
    assert not aff[0, :, :, 2].any() and aff[0, :, :, [0, 1, 3, 4]].all()  # +x: only the split is a boundary
    assert not valid[0, :, :, 5].any()  # +x from the last layer leaves the block
    assert aff[1, :3].all() and not valid[1, 3].any()  # +z never crosses the split
    assert aff[2, :, :, :3].logical_not().all() and valid[2, :, :, :3].all() and not valid[2, :, :, 3:].any()


def test_tokens_voxels_round_trip():
    vox = torch.randn(6, 8, 12, 16)
    tok = to_tokens(vox, 4)
    assert tok.shape == (2 * 3 * 4, 6 * 64)
    assert torch.equal(tok[0].reshape(6, 4, 4, 4), vox[:, :4, :4, :4])  # first token = the corner patch
    assert torch.equal(to_voxels(tok, (2, 3, 4), 4), vox)


def test_fit_probe_learns_a_linear_target():
    g = torch.Generator().manual_seed(0)
    feats, w = torch.randn(2000, 16, generator=g), torch.randn(16, 5, generator=g)
    targets = feats @ w > 0
    valid = torch.ones_like(targets)
    valid[:, 0] = False  # an ignored column must not matter
    head = fit_probe(feats, targets, valid)
    with torch.no_grad():
        acc = ((head(feats) > 0) == targets)[:, 1:].float().mean()
    assert acc > 0.95


def test_average_precision():
    assert average_precision(torch.tensor([0.9, 0.8, 0.1]), torch.tensor([1, 1, 0])) == 1.0
    assert abs(average_precision(torch.tensor([0.9, 0.8, 0.1]), torch.tensor([0, 1, 1])) - (1 / 2 + 2 / 3) / 2) < 1e-6


def test_probe_end_to_end_on_a_synthetic_store(tmp_path, monkeypatch):
    # A 64^3 store in x y z, like the hemibrain crop: raw with a gradient, two objects split along x only.
    store = tmp_path / "crop.zarr"
    x, y, z = np.meshgrid(*[np.arange(64)] * 3, indexing="ij")
    zarr.open_array(str(store / "raw/s0"), mode="w", shape=(64, 64, 64), dtype="uint8")[:] = (x * 4).astype(np.uint8)
    lab = 1 + (x >= 40)
    zarr.open_array(str(store / "labels/cells/s0"), mode="w", shape=(64, 64, 64), dtype="uint64")[:] = lab
    boxes = {"train": [[24, 56], [24, 56], [24, 56]], "fit": [[20, 52], [24, 56], [24, 56]], "test": [[24, 56], [20, 52], [24, 56]]}
    par = experiment.Params(savedir=str(tmp_path / "run"), global_size=(24, 24, 24))  # 32^3 boxes -> 48^3 of context
    (tmp_path / "run").mkdir()
    monkeypatch.setattr(experiment, "allparams", lambda: [par])
    monkeypatch.setattr(experiment, "HEMIBRAIN_EB_PROBE_BOXES", boxes)
    monkeypatch.setattr(experiment, "HEMIBRAIN_EB_PROBE_ANNOTATED", {k: v for k, v in boxes.items() if k != "train"})
    monkeypatch.setattr(experiment, "HEMIBRAIN_EB_LABELS", "labels/cells")
    monkeypatch.setattr(experiment.lmd, "get", lambda name: SimpleNamespace(path=str(store)))
    monkeypatch.setattr(experiment, "code_provenance", lambda: {"commit_id": "test"})
    model = Lejepa(LejepaConfig(n_layers=1, width=16, num_heads=2, patch_size=(4, 4, 4), proj_hidden=16, proj_dim=8))
    monkeypatch.setattr(experiment, "load_checkpoint", lambda p: (model.eval(), 7))
    experiment.probe(0)

    stats = json.loads((tmp_path / "run/probe.json").read_text())
    assert stats["step"] == 7 and 0 <= stats["boundary_ap_short"] <= 1
    # Axes: only +x crosses the x = 40 split (1 of 31 valid layers); +y and +z never do.
    frac = [stats[f"boundary_frac_{o}"] for o in AFFINITY_OFFSETS_XYZ[:3]]
    assert abs(frac[0] - 1 / 31) < 0.003 and frac[1] == frac[2] == 0
    from artifact import open_artifact
    art = open_artifact(tmp_path / "run/probe/test/hemibrain_eb_test.zarr")
    assert art.spatial_shape == (32, 32, 32) and list(art.origin) == [24, 20, 24]
    arr = np.asarray(zarr.open_array(str(tmp_path / "run/probe/test/hemibrain_eb_test.zarr"), mode="r")[:])
    assert arr.shape == (6, 32, 32, 32) and 0 <= arr.min() and arr.max() <= 1
    assert (tmp_path / "run/probe.png").is_file() and (tmp_path / "run/probe/fit/hemibrain_eb_fit.zarr").is_dir()
