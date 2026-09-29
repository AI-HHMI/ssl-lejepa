"""Tests for analysis: tables and figures are built from each run's own saved artifacts."""

import json

import plotly.express as px
import pytest

import analysis


def test_load_table_uses_saved_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for i, width in [(10, 768), (2, 512)]:
        d = tmp_path / f"outdir/e00/sweep/d{i}"
        d.mkdir(parents=True)
        (d / "performance.json").write_text(json.dumps({"tbl": "throughput", "params": {"width": width}}) + "\n")
        (d / "metrics.json").write_text("".join(json.dumps({"idx_step": s, "loss": 1.0 / (s + 1)}) + "\n" for s in [0, 10]))
    (tmp_path / "outdir/e00/sweep/notes").mkdir()  # not a run dir
    res = analysis.load_table("e00/sweep", "metrics.json")
    assert list(res.run) == ["d2", "d2", "d10", "d10"]  # numeric run order
    assert list(res.width) == [512, 512, 768, 768]  # saved params, not allparams()
    assert "n_layers" not in res  # only what runs saved; no current Params defaults (analysis doesn't import lib/)


def test_bench_reports_crashes_with_short_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shown = []
    monkeypatch.setattr(analysis, "show", lambda fig, name: shown.append(name))
    ok, crashed, legacy = (tmp_path / f"outdir/e00/b/d{i}" for i in range(3))
    base = {"queue": "gpu_b300", "width": 512, "compile": True, "cudagraphs": True, "eager_patch_embed": True, "n_gpus": 1}
    for d, bs in [(ok, 84), (crashed, 64)]:
        d.mkdir(parents=True)
        (d / "runs.json").write_text(json.dumps({"fn": "run", "params": {**base, "batch_size": bs, "savedir": str(d)}}) + "\n")
    legacy.mkdir(parents=True)  # crashed before runs.json had params
    (ok / "metrics.json").write_text("".join(json.dumps({"idx_step": s, "loss": l}) + "\n" for s, l in [(0, 2.0), (10, 1.5)]))
    (ok / "performance.json").write_text(json.dumps({"tbl": "throughput", "tokens_per_second": 2e6, "world_size": 2, "mfu": 0.2,
                                                     "tflops_per_second": 500.0, "seconds_per_step": 0.4}) + "\n")
    (ok / "job_run_1.log").write_text("...\nSuccessfully completed.\n")
    for d in [crashed, legacy]:
        (d / "job_run_2.log").write_text("CUDA error: an illegal memory access was encountered\nExited with exit code 1.\n")
    res = analysis.bench("e00/b")
    assert list(res.columns) == ["run", "config", "status", "steps", "EFLOP", "finite", "loss0", "loss_end", "ktok/s/gpu", "mfu %", "mem GB"]
    assert res.steps[0] == 11 and abs(res.EFLOP[0] - 500 * 0.4 * 11 / 1e6) < 1e-3  # last logged idx_step 10 -> 11 steps
    assert list(res.config) == ["B300 w512 b84 cudagraphs+eager-pe", "B300 w512 b64 cudagraphs+eager-pe", "? (no saved params)"]
    assert list(res.status) == ["ok", "exit 1: illegal memory access", "exit 1: illegal memory access"]
    assert res["ktok/s/gpu"][0] == 1000 and res.finite[0] and res.loss_end[0] == 1.75
    assert (tmp_path / "results/e00/b/bench.csv").is_file() and shown == ["e00/b/bench_loss", "e00/b/bench_speed"]


def test_ng_link_lists_zarr_label_layers(tmp_path, monkeypatch):
    from types import SimpleNamespace
    vol = tmp_path / "ds/crop-001.zarr"
    (vol / "labels").mkdir(parents=True)
    (vol / "labels/zarr.json").write_text(json.dumps({"attributes": {"ome": {"labels": ["seg-a", "mito-b"]}}}))
    monkeypatch.setattr(analysis.lmd, "get", lambda name: SimpleNamespace(path=str(vol), image_key="raw", zarr_version="zarr3",
                                                                          voxelsize=[8.0, 8.0, 8.0], axes=["x", "y", "z"]))
    url = analysis.ng_link("ds/crop-001")
    assert "seg-a" in url and "mito-b" in url and "groups_miaai_miaai" in url  # cluster path, not the local mount
    state = analysis.parse_neuroglancer_url(analysis.ng_link("ds/crop-001", {"test": [[40, 50], [0, 10], [4, 6]]}))
    box = state["layers"][-1]["annotations"][0]
    assert box["pointA"] == [40, 0, 4] and box["pointB"] == [50, 10, 6] and state["position"] == [45, 5, 5]
    assert state["dimensions"]["x"] == [8e-9, "m"]


def test_probe_table_and_curves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shown = []
    monkeypatch.setattr(analysis, "show", lambda fig, name: shown.append(name))
    for i, (steps, ap) in enumerate([(5000, 0.8), (100, 0.3)]):
        d = tmp_path / f"outdir/e00/p/d{i}"
        d.mkdir(parents=True)
        (d / "runs.json").write_text(json.dumps({"fn": "run", "params": {"queue": "gpu_h200", "width": 512, "batch_size": 64, "steps_per_epoch": steps}}) + "\n")
        st = {"tbl": "probe", "step": steps, "boundary_ap_short": ap, "boundary_ap_(1, 0, 0)": ap, "bce_(1, 0, 0)": 0.1, "bce_(0, 1, 0)": 0.3}
        (d / "probe.json").write_text(json.dumps(st) + "\n")
        (d / "probe_fit.json").write_text("".join(json.dumps({"tbl": "probe_fit", "step": s, "loss": 1 / (s + 1), "held_bce": 0.5,
                                                              "held_boundary_ap": ap}) + "\n" for s in [0, 100]))
    res = analysis.probe_table("e00/p")
    assert list(res["boundary AP short"]) == [0.8, 0.3] and list(res["BCE short"]) == [0.2, 0.2]
    assert res.config[1].endswith("steps=100") and "AP (1, 0, 0)" in res
    analysis.probe_curves("e00/p")
    assert shown == ["e00/p/probe_fit"] and (tmp_path / "results/e00/p/probe.csv").is_file()


def test_probe_vs_compute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    figs = []
    monkeypatch.setattr(analysis, "show", lambda fig, name: figs.append((fig, name)))
    ap = lambda v: {f"boundary_ap_{c}": v for c in ["(1, 0, 0)", "(0, 1, 0)", "(0, 0, 1)", "(10, 0, 0)", "(0, 10, 0)", "(0, 0, 10)"]}
    arch = {"patch_size": [128] * 3, "global_size": [96] * 3, "local_size": [64] * 3, "width": 512}
    trained, probed, rand = (tmp_path / f"outdir/e00/{s}/d0" for s in ["train", "probe", "rand"])
    for d, params, v in [(trained, arch, 0.6), (probed, {**arch, "init_from": str(trained.relative_to(tmp_path))}, 0.7),
                         (rand, {**arch, "init_from": "random"}, 0.4)]:
        d.mkdir(parents=True)
        (d / "runs.json").write_text(json.dumps({"fn": "run", "params": {"savedir": "old"}}) + "\n"
                                     + json.dumps({"fn": "probe", "params": params}) + "\n")  # the newest row counts
        (d / "probe.json").write_text(json.dumps(ap(v)) + "\n")
    (trained / "metrics.json").write_text(json.dumps({"idx_step": 99, "loss": 1.0}) + "\n")
    (trained / "performance.json").write_text(json.dumps({"tbl": "throughput", "tflops_per_second": 1e3, "seconds_per_step": 0.5}) + "\n")
    analysis.probe_vs_compute("e00/train", "e00/probe", "e00/rand")
    (fig, name), = figs
    pts = [t for t in fig.data if t.mode == "markers+text"]
    assert {round(x, 6) for t in pts for x in t.x} == {0.05}  # 100 steps x 500 TFLOP = 0.05 EFLOP, for both
    assert sorted({round(y, 6) for t in pts for y in t.y}) == [0.6, 0.7]  # random is a baseline, not a point
    assert [round(t.y[0], 6) for t in fig.data if t.mode == "lines"] == [0.4, 0.4] and name == "e00/train/probe_vs_compute"


def test_show_writes_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fig = px.line(x=[0, 1], y=[1, 0])
    monkeypatch.setattr(type(fig), "show", lambda self, *a, **k: None)
    analysis.show(fig, "e00/demo")
    assert (tmp_path / "results/e00/demo.html").is_file()


def test_profile_device_ms(tmp_path):
    sep = "  ".join(["-" * 20] * 11)
    row = lambda name, cuda: "  ".join([name, "0.00%", "0.000us", "0.00%", "0.000us", "0.000us", cuda, "1.00%", cuda, "1.000us", "10"])
    table = lambda *rows: "\n".join([sep, "Name  Self CPU %  x", sep, *rows, sep, "Self CPU time total: 1.000s", "Self CUDA time total: 2.000s"])
    (tmp_path / "profile.out").write_text(
        "Device: cuda:0; recorded steps (zero-based): 61..70\nOPERATORS SORTED BY SELF CPU TIME\n" + table(row("aten::mm", "5.000s"))
        + "\n\nOPERATORS SORTED BY SELF DEVICE TIME\n" + table(row("flash_bwd", "1.500s"), row("gelu", "300.000ms"), row("flash_bwd", "20us")))
    total, rows = analysis.profile_device_ms(tmp_path / "profile.out")
    assert total == pytest.approx(200.0)  # 2 s over 10 steps
    assert rows == pytest.approx({"flash_bwd": 150.002, "gelu": 30.0})  # device table only; truncated names merged


def test_analysis_imports_no_experiment_code():
    # Analysis reads outdir/ only and never builds a Lejepa: lib/ and experiment.py change between commits
    # (README: "How this repo works"). lib.util's entrypoint CLI is the one allowed dependency.
    import ast
    from pathlib import Path
    tree = ast.parse(Path(analysis.__file__).read_text())
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if (m.split(".")[0] == "lib" and m != "lib.util") or m == "experiment"], names
