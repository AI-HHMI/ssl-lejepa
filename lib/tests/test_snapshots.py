"""Snapshot tests: each eval (lib.evals) run on a seeded synthetic store with a tiny model, its artifacts (images, json rows)
compared with the ones saved in lib/tests/snapshots/. A refactor of the model or inference code that keeps them keeps its
results. Images match to within a little numeric noise, numbers to a relative 1e-3. After a change that should alter them,
review the new artifacts and save them with `UPDATE_SNAPSHOTS=1 pytest lib/tests/test_snapshots.py`."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import lib.decoders
from lib import evals, specs
from lib.encoders import EncoderConfig
from lib.models import Lejepa
from test_probe import TINY, synthetic_store

SNAPSHOTS = Path(__file__).parent / "snapshots"
VOLATILE = {"seconds", "time", "run"}  # json keys that differ per run (run: the tmp dir's name)


def same_json(a, b, key=""):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_json(a[k], b[k], k) for k in a if k not in VOLATILE)
    if isinstance(a, list):
        return len(a) == len(b) and all(same_json(x, y, key) for x, y in zip(a, b))
    if isinstance(a, float):
        return a == pytest.approx(b, rel=1e-3, abs=1e-6)
    return a == b


def check(path: Path, name: str):
    """The artifact at path against SNAPSHOTS/name (saved over it when UPDATE_SNAPSHOTS is set)."""
    snap = SNAPSHOTS / name
    if os.environ.get("UPDATE_SNAPSHOTS"):
        snap.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, snap)
    assert snap.is_file(), f"no snapshot {snap}: review {path}, then save it with UPDATE_SNAPSHOTS=1"
    if path.suffix == ".png":
        a, b = np.asarray(Image.open(path), dtype=float), np.asarray(Image.open(snap), dtype=float)
        assert a.shape == b.shape, f"{name}: {a.shape} vs the snapshot's {b.shape}"
        assert np.abs(a - b).mean() < 1.0, f"{name}: differs from its snapshot by {np.abs(a - b).mean():.2f} grey levels on average (max {np.abs(a - b).max():.0f})"
    else:
        rows = lambda p: [json.loads(line) for line in p.read_text().splitlines()]
        assert same_json(rows(path), rows(snap)), f"{name}: differs from its snapshot ({path} vs {snap})"


def small_encoder(patch: int = 4) -> Lejepa:
    torch.manual_seed(0)
    return Lejepa(TINY, encoder=EncoderConfig(depth=2, embed_dim=16, num_heads=2, patch_size=(patch, patch, patch))).eval()


def test_probe_snapshot(tmp_path, monkeypatch):
    synthetic_store(tmp_path, monkeypatch)
    (tmp_path / "run").mkdir()
    model = small_encoder()
    evals.probe(specs.Eval(savedir=str(tmp_path / "run"), train=specs.Train(global_size=(24, 24, 24))), model, 7)
    for f in ("probe.png", "probe.json", "probe_fit.json"):
        check(tmp_path / "run" / f, f"probe/{f}")


@pytest.mark.parametrize("fn, files", [("finetune", ("finetune.png", "finetune_curve.json")), ("tiles", ("tiles.png",))])
def test_unetr_snapshot(tmp_path, monkeypatch, fn, files):
    synthetic_store(tmp_path, monkeypatch)
    config = lib.decoders.UnetrConfig
    monkeypatch.setattr(lib.decoders, "UnetrConfig", lambda out_dim: config(out_dim=out_dim, hidden_dim=8))
    model = small_encoder(patch=2)  # Unetr: one block per upsample stage
    par = specs.Finetune(savedir=str(tmp_path / "run"), train=specs.Train(global_size=(24, 24, 24)), fit_side=24, steps=100)
    (tmp_path / "run").mkdir()
    torch.manual_seed(0)  # the decoder's init
    getattr(evals, fn)(par, model, 7)
    for f in files:
        check(tmp_path / "run" / f, f"{fn}/{f}")


def test_pca_snapshot(tmp_path, monkeypatch):
    model = small_encoder()
    monkeypatch.setattr(evals, "eb_val_crop", lambda shape: torch.rand(1, *shape, generator=torch.Generator().manual_seed(1)))
    monkeypatch.setattr(evals, "TEST_TILES", (8, 16))
    e = specs.Eval(savedir=f"{tmp_path}/", train=specs.Train(global_size=(16, 16, 16)))
    evals.pca_maps(model.encoder, e.train, tmp_path, 7)
    evals.pca_sizes(e, model, 7)
    for f in ("pca2.png", "pca.json", "pca_tile_008.png", "pca_tile_016.png", "pca_sizes.json"):
        check(tmp_path / f, f"pca/{f}")
