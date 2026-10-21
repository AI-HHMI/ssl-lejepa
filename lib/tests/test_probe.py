"""Tests for lib.probe and experiment.probe: affinity targets, token layout, fit, AP, and the probe end to end."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

import experiment
import lib.probe
from lib.encoders import EncoderConfig
from lib.models import Lejepa, LejepaConfig, ProjectorConfig
from lib.probe import AFFINITY_OFFSETS_XYZ, affinities, average_precision, fit_probe, to_tokens, to_voxels

TINY = LejepaConfig(encoder=EncoderConfig(depth=1, embed_dim=16, num_heads=2, patch_size=(4, 4, 4)),
                    projector=ProjectorConfig(dim=8, hidden=16))
BOXES = {"train": [[24, 56], [24, 56], [24, 56]], "fit": [[20, 52], [24, 56], [24, 56]], "test": [[24, 56], [20, 52], [24, 56]]}


def synthetic_store(tmp_path, monkeypatch, sets=None):
    """A 64^3 store in x y z at 8 nm, like the hemibrain crop: raw a gradient in x, two objects split at x = 40. The probe
    reads it through a stand-in for read_window (the real one reads OME-zarr through miao; checked on the cluster): the
    same z y x windows about a centre, edge-replicated (labels 0) outside. Probe sets: hemibrain_eb on BOXES, plus sets."""
    x, y, z = np.meshgrid(*[np.arange(64)] * 3, indexing="ij")
    raw, lab = torch.from_numpy((x * 4).astype(np.uint8)).permute(2, 1, 0).float() / 255, torch.from_numpy(1 + (x >= 40)).permute(2, 1, 0).long()

    def read_window(vol, center, size):
        lo = np.round(np.asarray(center) - np.array(size) / 2).astype(int)
        a, b = np.clip(lo, 0, 64), np.clip(lo + size, 0, 64)
        pad = [int(n) for l, s, aa, bb in zip(lo[::-1], size[::-1], a[::-1], b[::-1]) for n in (aa - l, l + s - bb)]
        img = torch.nn.functional.pad(raw[a[0]:b[0], a[1]:b[1], a[2]:b[2]][None, None], pad, mode="replicate")[0, 0]
        return img, torch.nn.functional.pad(lab[a[0]:b[0], a[1]:b[1], a[2]:b[2]], pad, value=0)

    vol = lambda name, box: experiment.ProbeVolume(name, str(tmp_path / "crop.zarr"), "cells", box)
    monkeypatch.setattr(experiment, "PROBE_SETS", {"hemibrain_eb": {s: [vol(f"hemibrain_eb_{s}", BOXES[s])] for s in ("train", "test")},
                                                   **{name: {s: [vol(*v) for v in vs] for s, vs in splits.items()} for name, splits in (sets or {}).items()}})
    monkeypatch.setattr(experiment, "HEMIBRAIN_EB_PROBE_BOXES", BOXES)
    monkeypatch.setattr(experiment, "HEMIBRAIN_EB_PROBE_ANNOTATED", {k: v for k, v in BOXES.items() if k != "train"})
    monkeypatch.setattr(experiment, "volume_grid", lambda vol: (np.array([8.0, 8.0, 8.0]), np.array([64, 64, 64])))
    monkeypatch.setattr(experiment, "read_window", read_window)
    monkeypatch.setattr(experiment, "code_provenance", lambda: {"commit_id": "test"})
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
    assert all(v.sparse for s in experiment.PROBE_SETS.values() for vs in s.values() for v in vs if not v.name.startswith("hemibrain"))
    assert not any(v.sparse for vs in experiment.PROBE_SETS["hemibrain_eb"].values() for v in vs)  # dense proofread: unchanged


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


def test_average_precision():
    assert average_precision(torch.tensor([0.9, 0.8, 0.1]), torch.tensor([1, 1, 0])) == 1.0
    assert abs(average_precision(torch.tensor([0.9, 0.8, 0.1]), torch.tensor([0, 1, 1])) - (1 / 2 + 2 / 3) / 2) < 1e-6


@pytest.mark.parametrize("score", [True, False])
def test_probe_end_to_end_on_a_synthetic_store(tmp_path, monkeypatch, score):
    synthetic_store(tmp_path, monkeypatch)
    par = experiment.Eval(savedir=str(tmp_path / "run"), train=experiment.Train(global_size=(24, 24, 24)), score=score)  # 32^3 boxes -> 48^3 of context
    (tmp_path / "run").mkdir()
    monkeypatch.setattr(experiment, "HEMIBRAIN_EB_LABELS", "labels/cells")
    model = Lejepa(TINY)
    monkeypatch.setattr(experiment, "load_checkpoint", lambda p: (model.eval(), 7))
    released = []
    monkeypatch.setattr(experiment, "release_milestone", released.append)
    experiment.probe(par)

    assert released == [par]  # its milestone, if it has one, is no longer needed
    stats = json.loads((tmp_path / "run/probe.json").read_text())
    assert stats["step"] == 7 and 0 <= stats["boundary_ap_short"] <= 1
    # Axes: only +x crosses the x = 40 split (1 of 31 valid layers); +y and +z never do.
    frac = [stats[f"boundary_frac_{o}"] for o in AFFINITY_OFFSETS_XYZ[:3]]
    assert abs(frac[0] - 1 / 31) < 0.003 and frac[1] == frac[2] == 0
    assert (tmp_path / "run/probe.png").is_file()
    if score:
        art = zarr.open_array(str(tmp_path / "run/probe/test/hemibrain_eb_test.zarr"), mode="r")
        assert dict(art.attrs)["kind"] == "affinity" and dict(art.attrs)["origin"] == [24, 20, 24] and dict(art.attrs)["scale"] == [1, 1, 1]
        arr = np.asarray(art[:])
        assert arr.shape == (6, 32, 32, 32) and 0 <= arr.min() and arr.max() <= 1
        assert (tmp_path / "run/probe/fit/hemibrain_eb_fit.zarr").is_dir()
    else:
        assert not (tmp_path / "run/probe").exists()  # nobody scores this run: GBs of artifacts nobody reads
    fit = [json.loads(l) for l in (tmp_path / "run/probe_fit.json").read_text().splitlines()]
    assert fit[0]["step"] == 0 and fit[-1]["step"] == 299 and all(f["tbl"] == "probe_fit" for f in fit)
    assert all(0 <= f["held_boundary_ap"] <= 1 and f["held_bce"] > 0 for f in fit)


@pytest.mark.parametrize("freeze, fit_side, size", [(True, 24, 24), (False, 0, 32)])
def test_finetune_end_to_end_on_a_synthetic_store(tmp_path, monkeypatch, freeze, fit_side, size):
    # The linear probe's synthetic store and boxes, a fit cube of 24 or the whole 32^3 train box (smaller than PROBE_SIDE:
    # its size comes from numpy, which json can't write unless cast) and global_size=24 crops. The tiny encoder has patch 2,
    # since Unetr needs one block per upsample stage (log2(patch_size)), and a narrow decoder.
    import lib.decoders
    synthetic_store(tmp_path, monkeypatch)
    config = lib.decoders.UnetrConfig
    monkeypatch.setattr(lib.decoders, "UnetrConfig", lambda out_dim: config(out_dim=out_dim, hidden_dim=8))
    par = experiment.Finetune(savedir=str(tmp_path / "run"), train=experiment.Train(global_size=(24, 24, 24)), fit_side=fit_side,
                              steps=150, freeze_encoder=freeze)
    spec = experiment.dump(par)  # as submit writes it
    model = Lejepa(TINY, encoder=EncoderConfig(depth=2, embed_dim=16, num_heads=2, patch_size=(2, 2, 2)))
    before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    monkeypatch.setattr(experiment, "load_checkpoint", lambda p: (model.eval(), 7))
    experiment.finetune(par)

    assert experiment.load(str(spec), experiment.Finetune) == par  # its results didn't overwrite its spec
    data, *curve = [json.loads(l) for l in (tmp_path / "run/finetune_curve.json").read_text().splitlines()]
    assert data == {"tbl": "finetune_data", "step": 7, "fit_size": [size] * 3, "fit_labelled": size ** 3, "test_blocks": ["hemibrain_eb_test"]}
    assert [c["step"] for c in curve] == [100, 150] and all(c["tbl"] == "finetune" for c in curve)
    for c in curve:  # every one a doubling step or the last: the test block scored, as probe.json has it
        assert 0 <= c["boundary_ap_short"] <= 1 and abs(c["boundary_frac_(1, 0, 0)"] - 1 / 31) < 0.003 and c["boundary_frac_(0, 1, 0)"] == 0
    assert (tmp_path / "run/finetune.png").is_file()
    moved = any(not torch.equal(v, before[k]) for k, v in model.encoder.state_dict().items())
    assert moved != freeze  # the encoder trained with the decoder only when not frozen
    with pytest.raises(AssertionError, match="whole patches"):
        experiment.finetune(experiment.Finetune(savedir=par.savedir, train=par.train, fit_side=25))
    with pytest.raises(AssertionError, match="smaller than a crop"):
        experiment.finetune(experiment.Finetune(savedir=par.savedir, train=par.train, fit_side=16))


def test_probe_on_another_set_averages_its_test_blocks(tmp_path, monkeypatch):
    """A probe set with two test blocks: scored on each, averaged; one png each. Its boxes are cut to whole patches."""
    synthetic_store(tmp_path, monkeypatch, {"two": {"train": [("two_train", [[24, 56], [24, 56], [24, 56]])],
                                                    "test": [("two_a", [[22, 57], [24, 56], [24, 56]]), ("two_b", [[24, 56], [24, 56], [8, 40]])]}})
    par = experiment.Eval(savedir=str(tmp_path / "run"), train=experiment.Train(global_size=(24, 24, 24)), probe_on="two")
    (tmp_path / "run").mkdir()
    monkeypatch.setattr(experiment, "load_checkpoint", lambda p: (Lejepa(TINY).eval(), 7))
    experiment.probe(par)
    stats = json.loads((tmp_path / "run/probe.json").read_text())
    assert stats["probe_on"] == "two" and stats["test_blocks"] == ["two_a", "two_b"]
    # In both, +x crosses the x = 40 split once in 32 voxels (two_a's 35 cut to whole patches, 32): 1 of 31 valid layers.
    frac = stats["boundary_frac_(1, 0, 0)"]
    assert abs(frac - 1 / 31) < 0.003 and stats["boundary_frac_(0, 1, 0)"] == 0
    assert (tmp_path / "run/probe.png").is_file() and (tmp_path / "run/probe_two_b.png").is_file()
    with pytest.raises(AssertionError, match="hemibrain_eb's"):
        experiment.probe(experiment.Eval(savedir=str(tmp_path / "run"), train=par.train, probe_on="two", score=True))


def test_load_checkpoint_init_from(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "lejepa_config", lambda par: TINY)
    small = experiment.Train(n_layers=1, width=16)
    a, step = experiment.load_checkpoint(experiment.Eval(savedir=str(tmp_path), train=small, init_from="random"))
    b, _ = experiment.load_checkpoint(experiment.Eval(savedir=str(tmp_path), train=small, init_from="random"))
    assert step == 0 and all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))  # seeded
    # Another run's checkpoint: loaded into par's architecture even though its saved params differ from par's.
    other = tmp_path / "old/d0"
    (other / "checkpoints").mkdir(parents=True)
    trained = {k: v + 1 if v.is_floating_point() else v for k, v in a.state_dict().items()}
    torch.save({"step": 42, "params": {"savedir": str(other), "width": 16}, "model": trained}, other / "checkpoints/step_0000042.pt")
    c, step = experiment.load_checkpoint(experiment.Eval(savedir=str(tmp_path / "new/d0"), train=small, init_from=str(other)))
    assert step == 42 and all(torch.equal(x, trained[k]) for k, x in c.state_dict().items())


def test_scoring_config_reads_instances_truth_from_its_own_data_configs():
    import tomllib
    from pathlib import Path
    path = Path(experiment.__file__).parent / experiment.SCORE_CONFIG
    c = tomllib.loads(path.read_text())
    assert c["task"]["truth_kind"] == "instances"  # GT read from the store, no .gt.zarr copies
    for split, box in [("test", "[[4000, 5000], [4000, 5000], [4000, 5000]]"), ("fit", "[[4000, 5000], [4000, 5000], [3000, 4000]]")]:
        data = (path.parent / c["data"][split]["config_path"]).read_text()  # mia-evals reads these YAMLs relative to the config
        assert box in data and experiment.HEMIBRAIN_EB_LABELS in data


def test_score_runs_the_sister_mia_evals_on_the_probe_artifacts(tmp_path, monkeypatch):
    par = experiment.Eval(savedir=str(tmp_path / "run"), train=experiment.Train(), score=True)
    for split in ["fit", "test"]:
        (tmp_path / "run/probe" / split).mkdir(parents=True)
    monkeypatch.setattr(experiment, "code_provenance", lambda: {"commit_id": "abc123"})
    monkeypatch.setattr(experiment, "MIA_EVALS", tmp_path / "mia-evals")
    calls = []
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda cmd, check, cwd: calls.append((cmd, cwd)))
    with pytest.raises(AssertionError, match="no mia-evals CLI"):  # not cloned and synced yet
        experiment.score(par)
    (tmp_path / "mia-evals/.venv/bin").mkdir(parents=True)
    (tmp_path / "mia-evals/.venv/bin/mia-evals").touch()
    experiment.score(par)
    ((cmd, cwd),) = calls
    assert cmd[0] == str(tmp_path / "mia-evals/.venv/bin/mia-evals") and cwd == tmp_path / "mia-evals"
    assert cmd[1] == "score" and cmd[2].endswith(experiment.SCORE_CONFIG)
    args = dict(zip(cmd[3::2], cmd[4::2]))
    assert args["--test"] == str(tmp_path / "run/probe/test") and args["--val"] == str(tmp_path / "run/probe/fit")
    assert args["--leaderboard"] == str(tmp_path / "run/mia_evals") and "--no-scored" in cmd
    assert (tmp_path / "run/git_commit.txt").read_text() == "abc123\n"  # copied into the record by --run-dir
    assert json.loads((tmp_path / "run/resolved_config.json").read_text())["savedir"] == par.savedir
    assert not (tmp_path / "run/probe").exists()  # scored: its artifacts (GBs) are deleted, the record is in mia_evals/


def test_lsf_score_submits_a_cpu_job_on_the_runs_own_eval(tmp_path, monkeypatch):
    run = str(tmp_path / "run")
    (tmp_path / "run/probe/test").mkdir(parents=True)
    monkeypatch.setattr(experiment, "roots", lambda: [experiment.Train(savedir=run)])
    monkeypatch.setattr(experiment, "assert_committed", lambda: None)
    monkeypatch.setattr(experiment, "snapshot", lambda paths, dest: tmp_path / "code")
    submitted = []
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: submitted.append(cmd))
    experiment.lsf("score", 0)
    (cmd,) = submitted
    assert f"-q {experiment.SCORE_QUEUE}" in cmd and f"-n {experiment.SCORE_SLOTS}" in cmd and "-gpu" not in cmd
    assert f"experiment.py score {run}/eval.json" in cmd and "job_score_%J.log" in cmd
    assert experiment.load(f"{run}/eval.json", experiment.Eval) == experiment.own_eval(experiment.Train(savedir=run))


def test_lejepa_n_params_builds_the_model_once_per_size(tmp_path, monkeypatch):
    built = []

    def fake_lejepa(config):
        e = config.encoder
        built.append((e.depth, e.embed_dim))
        return SimpleNamespace(_n_encoder_params=e.depth * e.embed_dim, _n_predictor_params=5, _n_projector_params=7)

    monkeypatch.setattr(experiment, "N_PARAMS_CACHE", tmp_path / ".cache/n_params.json")
    monkeypatch.setattr(experiment, "Lejepa", fake_lejepa)
    assert experiment.lejepa_n_params(4, 256) == (1024, 5, 7)
    assert experiment.lejepa_n_params(4, 256) == (1024, 5, 7)  # read back from the cache file
    assert experiment.lejepa_n_params(6, 384) == (2304, 5, 7)
    assert built == [(4, 256), (6, 384)]


def test_scaling_budgets_are_m_hours_at_the_anchor_speed(monkeypatch):
    monkeypatch.setattr(experiment, "lejepa_n_params", lambda n_layers, width: (n_layers * width ** 2, n_layers * width ** 2 // 3, 1000))
    assert experiment.step_seconds("m", 84) == pytest.approx(experiment.ANCHOR_STEP_S_M_B84)  # the anchor itself
    assert experiment.step_seconds("l", 84) > experiment.step_seconds("m", 84) > experiment.step_seconds("m", 42)
    budgets = experiment.scaling_budgets()
    m_steps_per_hour = 3600 / experiment.step_seconds("m", experiment.SCALING_BATCH)
    for b, hours in experiment.BUDGET_HOURS_AT_M.items():  # m trains BUDGET_HOURS_AT_M hours at each budget
        assert budgets[b] / (experiment.sample_flops("m") * experiment.SCALING_BATCH) == pytest.approx(hours * m_steps_per_hour)
