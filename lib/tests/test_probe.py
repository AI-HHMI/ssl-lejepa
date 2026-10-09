"""Tests for lib.probe and experiment.probe: affinity targets, token layout, fit, AP, and the probe end to end."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
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
    monkeypatch.setattr(experiment, "paramsall", lambda: [par])
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
    art = zarr.open_array(str(tmp_path / "run/probe/test/hemibrain_eb_test.zarr"), mode="r")
    assert dict(art.attrs)["kind"] == "affinity" and dict(art.attrs)["origin"] == [24, 20, 24] and dict(art.attrs)["scale"] == [1, 1, 1]
    arr = np.asarray(art[:])
    assert arr.shape == (6, 32, 32, 32) and 0 <= arr.min() and arr.max() <= 1
    assert (tmp_path / "run/probe.png").is_file() and (tmp_path / "run/probe/fit/hemibrain_eb_fit.zarr").is_dir()
    fit = [json.loads(l) for l in (tmp_path / "run/probe_fit.json").read_text().splitlines()]
    assert fit[0]["step"] == 0 and fit[-1]["step"] == 2999 and all(f["tbl"] == "probe_fit" for f in fit)
    assert all(0 <= f["held_boundary_ap"] <= 1 and f["held_bce"] > 0 for f in fit)


def test_unetr_probe_end_to_end_on_a_synthetic_store(tmp_path, monkeypatch):
    # Same synthetic store, boxes and context padding as the linear probe's end-to-end test above (32^3 boxes ->
    # 48^3 of context at global_size=24): only the decoder and the tiny encoder's patch_size/depth differ, since
    # Unetr needs at least one block per upsample stage (num_stages = log2(patch_size); patch_size=2 -> 1 stage).
    store = tmp_path / "crop.zarr"
    x, y, z = np.meshgrid(*[np.arange(64)] * 3, indexing="ij")
    zarr.open_array(str(store / "raw/s0"), mode="w", shape=(64, 64, 64), dtype="uint8")[:] = (x * 4).astype(np.uint8)
    lab = 1 + (x >= 40)
    zarr.open_array(str(store / "labels/cells/s0"), mode="w", shape=(64, 64, 64), dtype="uint64")[:] = lab
    boxes = {"train": [[24, 56], [24, 56], [24, 56]], "fit": [[20, 52], [24, 56], [24, 56]], "test": [[24, 56], [20, 52], [24, 56]]}
    par = experiment.Params(savedir=str(tmp_path / "run"), global_size=(24, 24, 24),
                            decoder="unetr", unetr_hidden_dim=8, unetr_steps=20)
    (tmp_path / "run").mkdir()
    monkeypatch.setattr(experiment, "paramsall", lambda: [par])
    monkeypatch.setattr(experiment, "HEMIBRAIN_EB_PROBE_BOXES", boxes)
    monkeypatch.setattr(experiment, "HEMIBRAIN_EB_PROBE_ANNOTATED", {k: v for k, v in boxes.items() if k != "train"})
    monkeypatch.setattr(experiment, "HEMIBRAIN_EB_LABELS", "labels/cells")
    monkeypatch.setattr(experiment.lmd, "get", lambda name: SimpleNamespace(path=str(store)))
    monkeypatch.setattr(experiment, "code_provenance", lambda: {"commit_id": "test"})
    model = Lejepa(LejepaConfig(n_layers=2, width=16, num_heads=2, patch_size=(2, 2, 2), proj_hidden=16, proj_dim=8))
    monkeypatch.setattr(experiment, "load_checkpoint", lambda p: (model.eval(), 7))
    experiment.probe(0)

    stats = json.loads((tmp_path / "run/probe.json").read_text())
    assert stats["step"] == 7 and 0 <= stats["boundary_ap_short"] <= 1
    frac = [stats[f"boundary_frac_{o}"] for o in AFFINITY_OFFSETS_XYZ[:3]]
    assert abs(frac[0] - 1 / 31) < 0.003 and frac[1] == frac[2] == 0
    art = zarr.open_array(str(tmp_path / "run/probe/test/hemibrain_eb_test.zarr"), mode="r")
    assert dict(art.attrs)["origin"] == [24, 20, 24] and dict(art.attrs)["convention"] == "sigmoid(logit), unetr decoder"
    arr = np.asarray(art[:])
    assert arr.shape == (6, 32, 32, 32) and 0 <= arr.min() and arr.max() <= 1  # tiled over the 48^3 padded region, then trimmed
    assert (tmp_path / "run/probe.png").is_file() and (tmp_path / "run/probe/fit/hemibrain_eb_fit.zarr").is_dir()
    fit = [json.loads(l) for l in (tmp_path / "run/probe_fit.json").read_text().splitlines()]
    assert fit[0]["step"] == 0 and fit[-1]["step"] == 19 and all(f["tbl"] == "probe_fit" for f in fit)
    assert all(0 <= f["held_boundary_ap"] <= 1 and f["held_bce"] > 0 for f in fit)


def test_load_checkpoint_init_from(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "lejepa_config", lambda par: LejepaConfig(n_layers=1, width=16, num_heads=2, patch_size=(4, 4, 4),
                                                                              proj_hidden=16, proj_dim=8))
    a, step = experiment.load_checkpoint(experiment.Params(init_from="random", n_layers=1, width=16))
    b, _ = experiment.load_checkpoint(experiment.Params(init_from="random", n_layers=1, width=16))
    assert step == 0 and all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))  # seeded
    # Another run's checkpoint: loaded into par's architecture even though its saved params differ from par's.
    other = tmp_path / "old/d0"
    (other / "checkpoints").mkdir(parents=True)
    trained = {k: v + 1 if v.is_floating_point() else v for k, v in a.state_dict().items()}
    torch.save({"step": 42, "params": {"savedir": str(other), "width": 16}, "model": trained}, other / "checkpoints/step_0000042.pt")
    c, step = experiment.load_checkpoint(experiment.Params(savedir=str(tmp_path / "new/d0"), init_from=str(other), n_layers=1, width=16))
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
    par = experiment.Params(savedir=str(tmp_path / "run"))
    for split in ["fit", "test"]:
        (tmp_path / "run/probe" / split).mkdir(parents=True)
    monkeypatch.setattr(experiment, "paramsall", lambda: [par])
    monkeypatch.setattr(experiment, "code_provenance", lambda: {"commit_id": "abc123"})
    monkeypatch.setattr(experiment, "MIA_EVALS", tmp_path / "mia-evals")
    calls = []
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda cmd, check, cwd: calls.append((cmd, cwd)))
    with pytest.raises(AssertionError, match="no mia-evals CLI"):  # not cloned and synced yet
        experiment.score(0)
    (tmp_path / "mia-evals/.venv/bin").mkdir(parents=True)
    (tmp_path / "mia-evals/.venv/bin/mia-evals").touch()
    experiment.score(0)
    ((cmd, cwd),) = calls
    assert cmd[0] == str(tmp_path / "mia-evals/.venv/bin/mia-evals") and cwd == tmp_path / "mia-evals"
    assert cmd[1] == "score" and cmd[2].endswith(experiment.SCORE_CONFIG)
    args = dict(zip(cmd[3::2], cmd[4::2]))
    assert args["--test"] == str(tmp_path / "run/probe/test") and args["--val"] == str(tmp_path / "run/probe/fit")
    assert args["--leaderboard"] == str(tmp_path / "run/mia_evals") and "--no-scored" in cmd
    assert (tmp_path / "run/git_commit.txt").read_text() == "abc123\n"  # copied into the record by --run-dir
    assert json.loads((tmp_path / "run/resolved_config.json").read_text())["savedir"] == par.savedir


def test_scorelsf_submits_a_cpu_job(tmp_path, monkeypatch):
    par = experiment.Params(savedir=str(tmp_path / "run"))
    (tmp_path / "run/probe/test").mkdir(parents=True)
    monkeypatch.setattr(experiment, "paramsall", lambda: [par])
    monkeypatch.setattr(experiment, "assert_committed", lambda: None)
    monkeypatch.setattr(experiment, "snapshot", lambda paths, dest: tmp_path / "code")
    submitted = []
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: submitted.append(cmd))
    experiment.scorelsf(0)
    (cmd,) = submitted
    assert f"-q {experiment.SCORE_QUEUE}" in cmd and f"-n {experiment.SCORE_SLOTS}" in cmd and "-gpu" not in cmd
    assert "experiment.py score 0" in cmd and "job_score_%J.log" in cmd


def test_lejepa_n_params_builds_the_model_once_per_size(tmp_path, monkeypatch):
    built = []

    def fake_lejepa(config):
        built.append((config.n_layers, config.width))
        return SimpleNamespace(_n_encoder_params=config.n_layers * config.width, _n_projector_params=7)

    monkeypatch.setattr(experiment, "N_PARAMS_CACHE", tmp_path / ".cache/n_params.json")
    monkeypatch.setattr(experiment, "Lejepa", fake_lejepa)
    assert experiment.lejepa_n_params(4, 256) == (1024, 7)
    assert experiment.lejepa_n_params(4, 256) == (1024, 7)  # read back from the cache file
    assert experiment.lejepa_n_params(6, 384) == (2304, 7)
    assert built == [(4, 256), (6, 384)]


def test_scaling_budgets_are_m_hours_at_the_anchor_speed(monkeypatch):
    monkeypatch.setattr(experiment, "lejepa_n_params", lambda n_layers, width: (n_layers * width ** 2, 1000))
    assert experiment.step_seconds("m", 84) == pytest.approx(experiment.ANCHOR_STEP_S_M_B84)  # the anchor itself
    assert experiment.step_seconds("l", 84) > experiment.step_seconds("m", 84) > experiment.step_seconds("m", 42)
    budgets = experiment.scaling_budgets()
    m_steps_per_hour = 3600 / experiment.step_seconds("m", experiment.SCALING_BATCH)
    for b, hours in experiment.BUDGET_HOURS_AT_M.items():  # m trains BUDGET_HOURS_AT_M hours at each budget
        assert budgets[b] / (experiment.sample_flops("m") * experiment.SCALING_BATCH) == pytest.approx(hours * m_steps_per_hour)
