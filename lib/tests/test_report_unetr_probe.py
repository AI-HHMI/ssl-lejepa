"""Tests for report_unetr_probe.py: the unetr-probe page is built from each run's own saved artifacts."""

import json

import pytest

pytest.importorskip("matplotlib")  # the `analysis` extra

import report_unetr_probe as report  # noqa: E402

CHANNELS = ["(1, 0, 0)", "(0, 1, 0)", "(0, 0, 1)", "(10, 0, 0)", "(0, 10, 0)", "(0, 0, 10)"]


def write(d, name, *rows):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_run(d, params, ap, **files):
    write(d, "runs.json", {"fn": "probe", "params": params})
    write(d, "probe.json", {"tbl": "probe", **{f"boundary_ap_{c}": ap for c in CHANNELS}})
    for name, rows in files.items():
        write(d, name, *rows)


def test_report_compares_unetr_with_linear_on_the_same_encoder_and_notes_unscored_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    e00 = tmp_path / "outdir/e00"
    arch = {"n_layers": 4, "width": 256}
    make_run(e00 / "scaling-law/d15", arch, 0.2, **{  # the pretraining run, probed with the linear probe
        "metrics.json": [{"idx_step": 99}],
        "performance.json": [{"tbl": "throughput", "tflops_per_second": 100.0, "seconds_per_step": 0.5}]})
    make_run(e00 / "probe-test/d1", {"n_layers": 12, "width": 512, "init_from": "random"}, 0.1)  # linear, random encoder
    for d in ("d2", "d3"):  # earlier linear probes on another pretraining run (here the same one)
        make_run(e00 / "probe-test" / d, {**arch, "init_from": "outdir/e00/scaling-law/d15"}, 0.3)
    fit = [{"step": s, "held_boundary_ap": 0.3 + s / 1000} for s in (0, 100)]
    make_run(e00 / "unetr-probe/d0", {**arch, "init_from": "outdir/e00/scaling-law/d15/", "decoder": "unetr"}, 0.4, **{"probe_fit.json": fit})
    make_run(e00 / "unetr-probe/d1", {"n_layers": 12, "width": 512, "init_from": "random", "decoder": "unetr",
                                      "unetr_freeze_encoder": True}, 0.5, **{"probe_fit.json": fit})
    make_run(e00 / "unetr-probe/d2", {"n_layers": 24, "width": 1024}, 0.0)  # xl never trained; it has no probe.json
    (e00 / "unetr-probe/d2/probe.json").unlink()
    for d in ("d0", "d1"):
        (e00 / "unetr-probe" / d / "job_probe_1.log").write_text("Run time :   600 sec.\n")
        (e00 / "unetr-probe" / d / "probe.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    tiles, sections = report.build()
    assert dict((label, n) for n, label in tiles)["runs probed"] == "2/3"
    assert dict((label, n) for n, label in tiles)["pretrained beat random (short AP)"] == "0/1"  # 0.4 < 0.5
    assert dict((label, n) for n, label in tiles)["scored with mia-evals"] == "0/2"
    assert dict(sections)["Compute"]  # 100 TFLOP/s x 0.5 s x 100 steps = 5e-3 EFLOP, so log10 is finite
    report.main()
    page = (tmp_path / report.OUT).read_text()
    assert "xs (d15)" in page and "random, encoder frozen" in page
    assert "unetr, unetr-probe" in page and "linear, probe-test" in page and "random, linear" in page  # ap_vs_compute legend
    assert "plotly" in page and "id='viewer'" in page and "class='imggrid'" in page  # the interactive runtime is on the page
