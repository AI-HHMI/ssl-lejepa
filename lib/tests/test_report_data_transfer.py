"""Tests for reports/data_transfer.py: the e09 page puts every encoder's probe of every family in one matrix, beside random."""

import json

import pytest

pytest.importorskip("pandas")  # the `analysis` extra

from reports import data_transfer  # noqa: E402

CHANNELS = ["(1, 0, 0)", "(0, 1, 0)", "(0, 0, 1)", "(10, 0, 0)", "(0, 10, 0)", "(0, 0, 10)"]


def write(d, name, *rows):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


def probe(ap):
    return {"tbl": "probe", "step": 100, **{f"boundary_ap_{c}": ap for c in CHANNELS}, **{f"boundary_frac_{c}": 0.1 for c in CHANNELS}}


def test_report_puts_every_probe_in_one_matrix_beside_random(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sweep = tmp_path / "outdir" / data_transfer.SWEEP
    for i, data in enumerate(["hemibrain_wide", "lmd4"]):
        d = sweep / f"d{i}"
        write(d, "train.json", {"savedir": f"outdir/{data_transfer.SWEEP}/d{i}/", "data": data})
        write(d, "metrics.json", *[{"tbl": "metrics", "idx_step": k, "time": k, "loss": 1.0, "inv": 0.5, "sigreg": 5.0, "resid_norm": 100.0} for k in range(40)])
        for j, s in enumerate(data_transfer.SETS):
            write(d / f"probe_{s}", "probe.json", probe(0.3 + 0.1 * i + 0.01 * j))
        (d / "probe_fish2" / "probe.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # the others missing: a dash each
    for j, s in enumerate(data_transfer.SETS):
        d = sweep / f"d{2 + j}"
        write(d, "eval.json", {"savedir": f"outdir/{data_transfer.SWEEP}/d{2 + j}/", "train": {"data": "hemibrain_wide"}, "init_from": "random", "probe_on": s})
        write(d, "probe.json", probe(0.2))
    tiles, sections = data_transfer.build()
    tiles = {label: n for n, label in tiles}
    assert tiles["encoders trained"] == "2" and tiles["probes"] == "12"
    assert tiles["best long AP on Kasthuri15: joint (lmd4)"] == f"{0.4 + 0.03:.3f}"
    cards = dict(dict(sections)["All vs all"])
    assert any("minus the random encoder" in title for title in cards)
    data_transfer.main()
    assert "probe_fish2/probe.png" in (tmp_path / "results" / data_transfer.SWEEP / "report.html").read_text()
