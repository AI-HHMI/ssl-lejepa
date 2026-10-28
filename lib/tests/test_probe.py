"""Tests for lib.probe and lib.evals (probe, finetune), and task_utils.score: affinity targets, token layout, fit, AP, and the probe end to end."""

import json

import numpy as np
import pytest
import torch
import zarr

import experiment
import tasks
from lib import task_utils
import lib.data
import lib.probe
import lib.readers
from lib import evals, training
from lib.util import dump, load
from lib.encoders import EncoderConfig
from lib.models import Lejepa, LejepaConfig, ProjectorConfig
from lib.probe import AFFINITY_OFFSETS_XYZ, affinities, average_precision, fit_probe, to_tokens, to_voxels

TINY = LejepaConfig(encoder=EncoderConfig(depth=1, embed_dim=16, num_heads=2, patch_size=(4, 4, 4)),
                    projector=ProjectorConfig(dim=8, hidden=16))
BOXES = {"train": [[24, 56], [24, 56], [24, 56]], "fit": [[20, 52], [24, 56], [24, 56]], "test": [[24, 56], [20, 52], [24, 56]]}


def branch(tmp_path, data="hemibrain", view=24, **trunk) -> tasks.Branch:
    """A Branch of a trunk that evaluates on data, at view^3 crops, in tmp_path/run."""
    return tasks.Branch(savedir=str(tmp_path / "run"), trunk=tasks.Trunk(train=training.Train(data=training.Data(view=(view,) * 3)), eval_data=data, **trunk), init_from="")


def train_of(view: int = 24) -> training.Train:
    return training.Train(data=training.Data(view=(view,) * 3))


def probe(tmp_path, model, data="hemibrain", view=24):
    """evals.probe of model at step 7, on data at view^3 crops, in tmp_path/run."""
    evals.probe(model, train_of(view), data, str(tmp_path / "run"), 7)


def finetune(tmp_path, model, data="hemibrain", view=24):
    evals.finetune(model, train_of(view), data, str(tmp_path / "run"), 7)


def synthetic_store(tmp_path, monkeypatch, sets=None):
    """A 64^3 store in x y z at 8 nm, like the hemibrain crop: raw a gradient in x, two objects split at x = 40. The probe
    reads it through a stand-in for read_window (the real one reads OME-zarr through miao; checked on the cluster): the
    same z y x windows about a centre, edge-replicated (labels 0) outside. Datasets: hemibrain on BOXES (BOXES' train is its
    fit block, its fit the tune block), plus sets: {name: {"fit": volume args, "test": [volume args], "tune"?: volume args}}."""
    x, y, z = np.meshgrid(*[np.arange(64)] * 3, indexing="ij")
    raw, lab = torch.from_numpy((x * 4).astype(np.uint8)).permute(2, 1, 0).float() / 255, torch.from_numpy(1 + (x >= 40)).permute(2, 1, 0).long()

    def read_window(vol, center, size):
        lo = np.round(np.asarray(center) - np.array(size) / 2).astype(int)
        a, b = np.clip(lo, 0, 64), np.clip(lo + size, 0, 64)
        pad = [int(n) for l, s, aa, bb in zip(lo[::-1], size[::-1], a[::-1], b[::-1]) for n in (aa - l, l + s - bb)]
        img = torch.nn.functional.pad(raw[a[0]:b[0], a[1]:b[1], a[2]:b[2]][None, None], pad, mode="replicate")[0, 0]
        return img, torch.nn.functional.pad(lab[a[0]:b[0], a[1]:b[1], a[2]:b[2]], pad, value=0)

    vol = lambda name, box, annotated=None: lib.data.Volume(name, box, "cells", annotated=annotated, name=name)  # the crop is named for the volume
    hemibrain = {s: vol(f"hemibrain_eb_{s}", BOXES[s], BOXES[s] if s != "train" else None) for s in BOXES}
    dataset = lambda fit, test, tune=None: lib.data.Dataset(train=[], val=[fit, *test], fit=fit, test=test, tune=tune)
    monkeypatch.setattr(lib.data, "DATASETS", {"hemibrain": dataset(hemibrain["train"], [hemibrain["test"]], hemibrain["fit"]),
                                               **{name: dataset(vol(*roles["fit"]), [vol(*v) for v in roles["test"]], vol(*roles["tune"]) if "tune" in roles else None)
                                                  for name, roles in (sets or {}).items()}})
    monkeypatch.setattr(lib.readers, "volume_grid", lambda vol: (np.array([8.0, 8.0, 8.0]), np.array([64, 64, 64])))
    monkeypatch.setattr(lib.readers, "read_window", read_window)
    monkeypatch.setattr(lib.probe, "FIT_STEPS", 300)  # the linear probe's fit: plenty for the store's one split


def test_affinities_mark_the_split_between_two_objects():
    labels = torch.ones(4, 4, 6, dtype=torch.int64)
    labels[..., 3:] = 2  # two objects split along x, between x = 2 and 3
    aff, valid = affinities(labels, [(0, 0, 1), (1, 0, 0), (0, 0, 3)])
    assert not aff[0, :, :, 2].any() and aff[0, :, :, [0, 1, 3, 4]].all()  # +x: only the split is a boundary
    assert not valid[0, :, :, 5].any()  # +x from the last layer leaves the block
    assert aff[1, :3].all() and not valid[1, 3].any()  # +z never crosses the split
    assert aff[2, :, :, :3].logical_not().all() and valid[2, :, :, :3].all() and not valid[2, :, :, 3:].any()


def test_sparse_labels_score_a_labelled_cell_against_unlabelled_as_a_boundary():
    labels = torch.tensor([[[1, 1, 0, 2, 0, 0]]])  # x: cell 1, unlabelled, cell 2, unlabelled
    aff, dense = affinities(labels, [(0, 0, 1)])
    _, sparse = affinities(labels, [(0, 0, 1)], sparse=True)
    assert aff[0, 0, 0].tolist() == [True, False, False, False, False, False]  # same object only within cell 1
    assert dense[0, 0, 0].tolist() == [True, False, False, False, False, False]  # both labelled: cell 1's inside only
    assert sparse[0, 0, 0].tolist() == [True, True, True, True, False, False]  # one labelled: its edges too, not 0 vs 0
    blocks = lambda s: [s.fit, *s.test, *([s.tune] if s.tune else [])]
    assert all(v.sparse for n, s in lib.data.DATASETS.items() if n not in ("hemibrain", "cellmap") for v in blocks(s))
    assert not any(v.sparse for v in blocks(lib.data.DATASETS["hemibrain"]))  # dense proofread: unchanged


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
    held = (feats[:200], targets[:200], valid[:200])  # (not really held out here; just exercising the curve)
    head, curve = fit_probe(feats, targets, valid, held)
    with torch.no_grad():
        acc = ((head(feats) > 0) == targets)[:, 1:].float().mean()
    assert acc > 0.95
    assert curve[0]["step"] == 0 and curve[-1]["step"] == 2999 and curve[-1]["loss"] < curve[0]["loss"]  # goes down
    assert curve[-1]["held_bce"] < curve[0]["held_bce"] and curve[-1]["held_boundary_ap"] > 0.9


def test_standardized_fit_takes_raw_features_of_any_scale():
    g = torch.Generator().manual_seed(0)
    base, w = torch.randn(2000, 16, generator=g), torch.randn(16, 5, generator=g)
    targets, valid = base @ w > 0, torch.ones(2000, 5, dtype=torch.bool)
    feats = base * torch.logspace(-2, 2, 16) + 50  # channels from 0.01 to 100 wide, off zero
    for standardize, ok in ((False, False), (True, True)):
        head, _ = fit_probe(feats, targets, valid, standardize=standardize)
        with torch.no_grad():
            acc = ((head(feats) > 0) == targets).float().mean()  # the head takes the raw features either way
        assert (acc > 0.95) == ok, (standardize, acc)


def test_average_precision():
    assert average_precision(torch.tensor([0.9, 0.8, 0.1]), torch.tensor([1, 1, 0])) == 1.0
    assert abs(average_precision(torch.tensor([0.9, 0.8, 0.1]), torch.tensor([0, 1, 1])) - (1 / 2 + 2 / 3) / 2) < 1e-6


def test_probe_end_to_end_on_a_synthetic_store(tmp_path, monkeypatch):
    synthetic_store(tmp_path, monkeypatch)
    (tmp_path / "run").mkdir()  # 32^3 boxes -> 48^3 of context
    probe(tmp_path, Lejepa(TINY).eval())

    stats = json.loads((tmp_path / "run/probe.json").read_text())
    assert stats["step"] == 7 and 0 <= stats["boundary_ap_short"] <= 1
    # Axes: only +x crosses the x = 40 split (1 of 31 valid layers); +y and +z never do.
    frac = [stats[f"boundary_frac_{o}"] for o in AFFINITY_OFFSETS_XYZ[:3]]
    assert abs(frac[0] - 1 / 31) < 0.003 and frac[1] == frac[2] == 0
    assert (tmp_path / "run/probe.png").is_file()
    art = zarr.open_array(str(tmp_path / "run/probe/test/hemibrain_eb_test.zarr"), mode="r")
    assert dict(art.attrs)["kind"] == "affinity" and dict(art.attrs)["origin"] == [24, 20, 24] and dict(art.attrs)["scale"] == [1, 1, 1]
    arr = np.asarray(art[:])
    assert arr.shape == (6, 32, 32, 32) and 0 <= arr.min() and arr.max() <= 1
    assert (tmp_path / "run/probe/fit/hemibrain_eb_fit.zarr").is_dir()
    fit = [json.loads(l) for l in (tmp_path / "run/probe_fit.json").read_text().splitlines()]
    assert fit[0]["step"] == 0 and fit[-1]["step"] == 299 and all(f["tbl"] == "probe_fit" for f in fit)
    assert all(0 <= f["held_boundary_ap"] <= 1 and f["held_bce"] > 0 for f in fit)


def test_finetune_end_to_end_on_a_synthetic_store(tmp_path, monkeypatch):
    # The linear probe's synthetic store and boxes, a fit cube of the whole 32^3 train box (smaller than PROBE_SIDE: its
    # size comes from numpy, which json can't write unless cast) and view=24 crops. The tiny encoder has patch 2, since
    # Unetr needs one block per upsample stage (log2(patch_size)), and a narrow decoder.
    import lib.decoders
    synthetic_store(tmp_path, monkeypatch)
    config = lib.decoders.UnetrConfig
    monkeypatch.setattr(lib.decoders, "UnetrConfig", lambda out_dim: config(out_dim=out_dim, hidden_dim=8))
    monkeypatch.setattr(evals, "FINETUNE_STEPS", 150)
    model = Lejepa(TINY, encoder=EncoderConfig(depth=2, embed_dim=16, num_heads=2, patch_size=(2, 2, 2)))
    before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    (tmp_path / "run").mkdir(exist_ok=True)
    finetune(tmp_path, model.eval())

    data, *curve = [json.loads(l) for l in (tmp_path / "run/finetune_curve.json").read_text().splitlines()]
    assert data == {"tbl": "finetune_data", "step": 7, "fit_size": [32] * 3, "fit_labelled": 32 ** 3, "test_blocks": ["hemibrain_eb_test"]}
    assert [c["step"] for c in curve] == [100, 150] and all(c["tbl"] == "finetune" for c in curve)
    for c in curve:  # every one a doubling step or the last: the test block scored, as probe.json has it
        assert 0 <= c["boundary_ap_short"] <= 1 and abs(c["boundary_frac_(1, 0, 0)"] - 1 / 31) < 0.003 and c["boundary_frac_(0, 1, 0)"] == 0
    assert (tmp_path / "run/finetune.png").is_file()
    assert all(torch.equal(v, before[k]) for k, v in model.encoder.state_dict().items())  # the encoder is frozen: only the decoder trains
    art = zarr.open_array(str(tmp_path / "run/probe/test/hemibrain_eb_test.zarr"), mode="r")  # the scorer's artifacts, as the probe's
    assert dict(art.attrs)["kind"] == "affinity" and dict(art.attrs)["step"] == 7 and np.asarray(art[:]).shape == (6, 32, 32, 32)
    assert (tmp_path / "run/probe/fit/hemibrain_eb_fit.zarr").is_dir()


def test_probe_on_another_dataset_averages_its_test_blocks(tmp_path, monkeypatch):
    """A dataset with two test blocks: scored on each, averaged; one png each. Its boxes are cut to whole patches."""
    two = {"fit": ("two_train", [[24, 56], [24, 56], [24, 56]]),
           "test": [("two_a", [[22, 57], [24, 56], [24, 56]]), ("two_b", [[24, 56], [24, 56], [8, 40]])]}
    synthetic_store(tmp_path, monkeypatch, {"two": {**two, "tune": ("two_fit", [[24, 56]] * 3, [[20, 60]] * 3)}, "two-untuned": two})
    (tmp_path / "run").mkdir()
    model = Lejepa(TINY).eval()
    probe(tmp_path, model, "two")
    stats = json.loads((tmp_path / "run/probe.json").read_text())
    assert stats["data"] == "two" and stats["test_blocks"] == ["two_a", "two_b"]
    # In both, +x crosses the x = 40 split once in 32 voxels (two_a's 35 cut to whole patches, 32): 1 of 31 valid layers.
    frac = stats["boundary_frac_(1, 0, 0)"]
    assert abs(frac - 1 / 31) < 0.003 and stats["boundary_frac_(0, 1, 0)"] == 0
    assert (tmp_path / "run/probe.png").is_file() and (tmp_path / "run/probe_1.png").is_file()
    with pytest.raises(AssertionError, match="no mia-evals tune block for two-untuned"):  # nothing to score it on
        probe(tmp_path, model, "two-untuned")


def test_score_artifacts_of_any_set_with_a_tune_block_are_named_for_it(tmp_path, monkeypatch):
    box = [[24, 56], [24, 56], [24, 56]]
    synthetic_store(tmp_path, monkeypatch, {"three": {"fit": ("three_train", box), "tune": ("three_fit", box, [[20, 60]] * 3), "test": [("three_test", box, [[22, 58]] * 3)]}})
    (tmp_path / "run").mkdir()
    probe(tmp_path, Lejepa(TINY).eval(), "three")
    art = zarr.open_array(str(tmp_path / "run/probe/test/three_test.zarr"), mode="r")
    assert dict(art.attrs)["annotated_box"] == [[22, 58]] * 3 and dict(art.attrs)["source_label_key"] == "labels/cells"
    assert (tmp_path / "run/probe/fit/three_fit.zarr").is_dir()


def test_load_checkpoint_init_from(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "model_config", lambda par: TINY)
    small = training.Train(encoder=training.Encoder(n_layers=1, width=16))
    a, step = task_utils.load_checkpoint(small, "random")
    b, _ = task_utils.load_checkpoint(small, "random")
    assert step == 0 and all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))  # seeded
    # Another run's checkpoint: loaded into small's architecture even though its saved params differ from small's.
    trained = {k: v + 1 if v.is_floating_point() else v for k, v in a.state_dict().items()}
    torch.save({"step": 42, "params": {"encoder": {"width": 16}}, "model": trained}, tmp_path / "checkpoint.pt")
    c, step = task_utils.load_checkpoint(small, str(tmp_path / "checkpoint.pt"))
    assert step == 42 and all(torch.equal(x, trained[k]) for k, x in c.state_dict().items())
    with pytest.raises(AssertionError, match="no checkpoint"):
        task_utils.load_checkpoint(small, str(tmp_path))


def test_scoring_config_reads_instances_truth_from_its_own_data_configs():
    import tomllib
    from pathlib import Path
    path = task_utils.CODE / task_utils.BOARDS["neuron-instance"]
    c = tomllib.loads(path.read_text())
    assert c["task"]["truth_kind"] == "instances"  # GT read from the store, no .gt.zarr copies
    for split, box in [("test", "[[4000, 5000], [4000, 5000], [4000, 5000]]"), ("fit", "[[4000, 5000], [4000, 5000], [3000, 4000]]")]:
        data = (path.parent / c["data"][split]["config_path"]).read_text()  # mia-evals reads these YAMLs relative to the config
        assert box in data and lib.data.EB_LABEL in data


def test_score_calls_mia_evals_on_the_probe_artifacts(tmp_path, monkeypatch):
    par = tasks.Score(savedir=str(tmp_path / "run"), branch=branch(tmp_path))
    with pytest.raises(AssertionError, match="no probe artifacts"):
        task_utils.mia_eval(par)
    for split in ["fit", "test"]:
        (tmp_path / "run/probe" / split).mkdir(parents=True)
    monkeypatch.setattr(task_utils, "code_provenance", lambda: {"commit_id": "abc123"})
    calls = []
    def fake_score(args):  # mia-evals writes its record into the leaderboard dir
        calls.append(args)
        record = args.leaderboard / "task/records/run.step0.mws.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"scores": {"voxel_instance": {"pq": 0.25}}}))
    from mia_evals import evaluate
    monkeypatch.setattr(evaluate, "cmd_score", fake_score)
    assert task_utils.mia_eval(par) == 0.25  # the panoptic quality, which ranks
    (args,) = calls
    assert str(args.config).endswith(task_utils.BOARDS["neuron-instance"])
    assert args.test == tmp_path / "run/probe/test" and args.val == tmp_path / "run/probe/fit"
    assert args.leaderboard == tmp_path / "run/mia_evals" and args.no_scored
    assert args.run_dir == tmp_path / "run"
    assert (tmp_path / "run/git_commit.txt").read_text() == "abc123\n"  # copied into the record from run_dir
    assert json.loads((tmp_path / "run/resolved_config.json").read_text())["savedir"] == par.savedir
    assert not (tmp_path / "run/probe").exists()  # scored: its artifacts (GBs) are deleted, the record is in mia_evals/
