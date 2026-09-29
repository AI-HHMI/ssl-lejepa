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
    (ok / "performance.json").write_text(json.dumps({"tbl": "throughput", "tokens_per_second": 2e6, "world_size": 2, "mfu": 0.2}) + "\n")
    (ok / "job_run_1.log").write_text("...\nSuccessfully completed.\n")
    for d in [crashed, legacy]:
        (d / "job_run_2.log").write_text("CUDA error: an illegal memory access was encountered\nExited with exit code 1.\n")
    res = analysis.bench("e00/b")
    assert list(res.columns) == ["run", "config", "status", "finite", "loss0", "loss_end", "ktok/s/gpu", "mfu %", "mem GB"]
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
