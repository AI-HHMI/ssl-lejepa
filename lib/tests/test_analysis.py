"""Tests for e00_analysis: tables and figures are built from each run's own saved artifacts."""

import json

import plotly.express as px
import pytest

import e00_analysis as analysis


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
    # Analysis reads outdir/ only and never builds a Lejepa: lib/ and eNN scripts change between commits
    # (README: "How this repo works"). lib.util's entrypoint CLI is the one allowed dependency.
    import ast
    from pathlib import Path
    tree = ast.parse(Path(analysis.__file__).read_text())
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if (m.split(".")[0] == "lib" and m != "lib.util") or m.startswith("e0")], names
