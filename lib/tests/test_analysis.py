"""Tests for analysis (readers) and analysis_plots (report cards): tables and charts are built from each run's own
saved artifacts."""

import json
import re

import pandas
import pytest

import analysis
import analysis_plots

AP_CHANNELS = ["(1, 0, 0)", "(0, 1, 0)", "(0, 0, 1)", "(10, 0, 0)", "(0, 10, 0)", "(0, 0, 10)"]


def ap(v):
    return {f"boundary_ap_{c}": v for c in AP_CHANNELS}


def write(d, name, *rows):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


def figures(cards) -> list[dict]:
    """The Plotly figure JSON in each card, in order (a table card has none)."""
    return [json.loads(s) for _, frag in cards for s in re.findall(r"<script type='application/json' id='iplot\d+-data'>(.*?)</script>", frag)]


def test_load_table_uses_saved_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for i, width in [(10, 768), (2, 512)]:
        d = tmp_path / f"outdir/e00/sweep/d{i}"
        write(d, "performance.json", {"tbl": "throughput", "params": {"width": width}})
        write(d, "metrics.json", *({"idx_step": s, "loss": 1.0 / (s + 1)} for s in [0, 10]))
    (tmp_path / "outdir/e00/sweep/notes").mkdir()  # not a run dir
    res = analysis.load_table("e00/sweep", "metrics.json")
    assert list(res.run) == ["d2", "d2", "d10", "d10"]  # numeric run order
    assert list(res.width) == [512, 512, 768, 768]  # saved params, not paramsall()
    assert "n_layers" not in res  # only what runs saved; no current Params defaults (analysis doesn't import lib/)


def test_bench_reports_crashes_with_short_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ok, crashed, legacy = (tmp_path / f"outdir/e00/b/d{i}" for i in range(3))
    base = {"queue": "gpu_b300", "width": 512, "compile": True, "cudagraphs": True, "eager_patch_embed": True, "n_gpus": 1}
    for d, bs in [(ok, 84), (crashed, 64)]:
        write(d, "runs.json", {"fn": "run", "params": {**base, "batch_size": bs, "savedir": str(d)}})
    legacy.mkdir(parents=True)  # crashed before runs.json had params
    write(ok, "metrics.json", *({"idx_step": s, "loss": l} for s, l in [(0, 2.0), (10, 1.5)]))
    write(ok, "performance.json", {"tbl": "throughput", "tokens_per_second": 2e6, "world_size": 2, "mfu": 0.2,
                                   "tflops_per_second": 500.0, "seconds_per_step": 0.4})
    (ok / "job_run_1.log").write_text("...\nSuccessfully completed.\n")
    for d in [crashed, legacy]:
        (d / "job_run_2.log").write_text("CUDA error: an illegal memory access was encountered\nExited with exit code 1.\n")
    res = pandas.DataFrame(analysis.bench_rows("e00/b")[0])
    assert list(res.columns) == ["run", "config", "status", "steps", "EFLOP", "walltime", "finite", "loss0", "loss_end", "ktok/s/gpu", "mfu %", "mem GB"]
    assert res.steps[0] == 11 and abs(res.EFLOP[0] - 500 * 0.4 * 11 / 1e6) < 1e-3  # last logged idx_step 10 -> 11 steps
    assert list(res.config) == ["B300 w512 b84 cudagraphs+eager-pe", "B300 w512 b64 cudagraphs+eager-pe", "? (no saved params)"]
    assert list(res.status) == ["ok", "exit 1: illegal memory access", "exit 1: illegal memory access"]
    assert res["ktok/s/gpu"][0] == 1000 and res.finite[0] and res.loss_end[0] == 1.75
    cards = analysis_plots.bench("e00/b")
    assert len(cards) == 3 and (tmp_path / "results/e00/b/bench.csv").is_file()
    loss, speed = figures(cards)
    assert [t["name"] for t in loss["data"]] == ["d0 B300 w512 b84 cudagraphs+eager-pe"]  # crashed runs have no curve
    assert [t["name"] for t in speed["data"]] == ["ok", "exit 1: illegal memory access"]  # bars coloured by status


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
    for i, (steps, v) in enumerate([(5000, 0.8), (100, 0.3)]):
        d = tmp_path / f"outdir/e00/p/d{i}"
        write(d, "runs.json", {"fn": "run", "params": {"queue": "gpu_h200", "width": 512, "batch_size": 64, "steps_per_epoch": steps}})
        write(d, "probe.json", {"tbl": "probe", "step": steps, "boundary_ap_short": v, "boundary_ap_(1, 0, 0)": v, "bce_(1, 0, 0)": 0.1, "bce_(0, 1, 0)": 0.3})
        write(d, "probe_fit.json", *({"tbl": "probe_fit", "step": s, "loss": 1 / (s + 1), "held_bce": 0.5, "held_boundary_ap": v} for s in [0, 100]))
    (title, table), = analysis_plots.probe_table("e00/p")
    res = pandas.read_csv(tmp_path / "results/e00/p/probe.csv")
    assert list(res["boundary AP short"]) == [0.8, 0.3] and list(res["BCE short"]) == [0.2, 0.2]
    assert res.config[1].endswith("steps=100") and "AP (1, 0, 0)" in res and "class='sortable'" in table
    curves = analysis_plots.probe_curves("e00/p")
    assert [t for t, _ in curves] == ["Probe fit: loss", "Probe fit: held_bce", "Probe fit: held_boundary_ap"]
    assert [t["name"] for t in figures(curves)[2]["data"]] == ["d0", "d1"]


def test_probe_vs_compute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    arch = {"patch_size": [128] * 3, "global_size": [96] * 3, "local_size": [64] * 3, "width": 512}
    trained, probed, rand = (tmp_path / f"outdir/e00/{s}/d0" for s in ["train", "probe", "rand"])
    for d, params, v in [(trained, arch, 0.6), (probed, {**arch, "init_from": str(trained.relative_to(tmp_path))}, 0.7),
                         (rand, {**arch, "init_from": "random"}, 0.4)]:
        write(d, "runs.json", {"fn": "run", "params": {"savedir": "old"}}, {"fn": "probe", "params": params})  # the newest row counts
        write(d, "probe.json", ap(v))
    write(trained, "metrics.json", {"idx_step": 99, "loss": 1.0})
    write(trained, "performance.json", {"tbl": "throughput", "tflops_per_second": 1e3, "seconds_per_step": 0.5})
    cards = analysis_plots.probe_vs_compute("e00/train", "e00/probe", "e00/rand")
    assert [t for t, _ in cards] == ["short (+1) boundary AP vs pretraining compute", "long (+10) boundary AP vs pretraining compute"]
    for fig in figures(cards):
        assert {round(x, 6) for t in fig["data"] for x in t["x"]} == {0.05}  # 100 steps x 500 TFLOP = 0.05 EFLOP, for both
        assert sorted(round(y, 6) for t in fig["data"] for y in t["y"]) == [0.6, 0.7]  # random is a baseline, not a point
        (baseline,) = fig["layout"]["shapes"]
        assert baseline["name"] == "random, linear" and baseline["y0"] == pytest.approx(0.4)


def test_scaling_law_tile_code_vs_probe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # source k = budget * 5 + size; tile code grows and AP falls with size and budget
    for i, (k, width, tile, v) in enumerate([(0, 256, 0.1, 0.6), (5, 256, 0.2, 0.5), (3, 768, 0.5, 0.4), (8, 768, 0.9, 0.3)]):
        src, pca = tmp_path / f"outdir/e00/scaling-law/d{k}", tmp_path / f"outdir/e00/scaling-law-pca/d{i}"
        write(src, "runs.json", {"fn": "run", "params": {"n_layers": width // 64, "width": width}})
        write(src, "probe.json", ap(v))
        write(pca, "runs.json", {"fn": "pca", "params": {"init_from": f"outdir/e00/scaling-law/d{k}/"}})
        write(pca, "pca.json", {"between_tile_variance": tile, "effective_rank": 100.0})
    rand = tmp_path / "outdir/e00/scaling-law-pca/d4"
    write(rand, "runs.json", {"fn": "pca", "params": {"init_from": "random", "n_layers": 12, "width": 512}})
    write(rand, "pca.json", {"between_tile_variance": 0.02, "effective_rank": 50.0})
    write(tmp_path / "outdir/e00/probe-test/d1", "probe.json", ap(0.2))
    cards = analysis_plots.scaling_law_tile_code_vs_probe()
    res = pandas.read_csv(tmp_path / "results/e00/scaling-law-pca/tile_code_vs_probe.csv")
    assert list(res.source) == ["d0", "d5", "d3", "d8"] and list(res.budget) == ["c1", "c2", "c1", "c2"]
    assert all("Spearman all -1.00; within size 4x256 -1.00, 12x768 -1.00" in t for t, _ in cards)
    for fig in figures(cards):
        (random,) = [t for t in fig["data"] if t["name"] == "random 12x512"]
        assert (random["x"][0], random["y"][0]) == pytest.approx((0.02, 0.2))  # one per range card


def test_mia_evals_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for i, pq in enumerate([0.12, 0.05]):
        d = tmp_path / f"outdir/e00/s/d{i}"
        write(d, "runs.json", {"fn": "run", "params": {"queue": "gpu_h200", "width": 512 * (i + 1), "batch_size": 64}})
        write(d / "mia_evals/gary_comparison_neuron_instance/records", "run.mws.json",
              {"route": "mws", "postprocess": {"describe": "mws(min_size=5000)"},
               "scores": {"voxel_instance": {"pq": pq, "voi_split": 1.0, "voi_merge": 2.0, "adapted_rand_error": 0.7,
                                             "instances_predicted": 700.0, "instances_truth": 2806.0}}})
    (_, table), = analysis_plots.mia_evals_table("e00/s")
    res = pandas.read_csv(tmp_path / "results/e00/s/mia_evals.csv")
    assert list(res.pq) == [0.12, 0.05] and list(res.instances) == ["700/2806"] * 2
    assert res.config[1].startswith("H200 w1024") and "700/2806" in table


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
    root = Path(analysis.__file__ or "").parent
    for path in [root / "analysis.py", root / "analysis_plots.py", *sorted(root.glob("reports/*.py"))]:
        tree = ast.parse(path.read_text())
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not [m for m in names if (m.split(".")[0] == "lib" and m != "lib.util") or m == "experiment"], (path.name, names)
