"""Tests for e00_analysis: tables and figures are built from each run's own saved artifacts."""

import json

import plotly.express as px

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
    assert (res.n_layers == analysis.Params().n_layers).all()  # fields not saved: current defaults


def test_show_writes_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fig = px.line(x=[0, 1], y=[1, 0])
    monkeypatch.setattr(type(fig), "show", lambda self, *a, **k: None)
    analysis.show(fig, "e00/demo")
    assert (tmp_path / "results/e00/demo.html").is_file()
