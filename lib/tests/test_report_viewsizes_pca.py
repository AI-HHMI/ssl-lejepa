"""Tests for report_viewsizes_pca.py: the viewsizes-pca page lays the maps out by view-size axis, from saved artifacts."""

import json

import pytest

pytest.importorskip("matplotlib")  # the `analysis` extra

import report_viewsizes_pca as report  # noqa: E402

CHANNELS = ["(1, 0, 0)", "(0, 1, 0)", "(0, 0, 1)", "(10, 0, 0)", "(0, 10, 0)", "(0, 0, 10)"]
CONFIGS = [(128, 96, 64)] + [(p, 96, 64) for p in [104, 160, 192, 256]] + [(128, g, 64) for g in [64, 80, 112, 128]] \
    + [(128, 96, l) for l in [32, 48, 80, 96]] + [(96, 64, 32), (160, 128, 80), (192, 144, 96), (256, 192, 128)]


def write(d, name, row):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(row) + "\n")


def test_report_lays_maps_out_by_axis_and_links_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    e00 = tmp_path / "outdir/e00"
    for i, (inp, g, l) in enumerate(CONFIGS * 2):
        src, d = e00 / f"viewsizes-v2/d{i}", e00 / f"viewsizes-pca/d{i}"
        write(src, "runs.json", {"fn": "run", "params": {"batch_size": 84}})
        write(src, "metrics.json", {"idx_step": 999 + 10 * i})
        write(src, "performance.json", {"tbl": "throughput", "tflops_per_second": 100.0, "seconds_per_step": 0.5})
        if (inp, g, l) != (192, 144, 96):  # viewsizes-v2/d15 and d32 have no probe
            write(src, "probe.json", {f"boundary_ap_{c}": 0.5 - g / 1000 for c in CHANNELS})
        write(d, "runs.json", {"fn": "pca", "params": {"patch_size": [inp] * 3, "global_size": [g] * 3, "local_size": [l] * 3,
                                                       "init_from": f"outdir/e00/viewsizes-v2/d{i}/"}})
        write(d, "pca.json", {"between_tile_variance": g / 200, "effective_rank": 100.0, "centered_effective_rank": 90.0})
        (d / "pca2.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    write(e00 / "probe-test/d1", "probe.json", {f"boundary_ap_{c}": 0.1 for c in CHANNELS})

    tiles, sections = report.build()
    tiles = {label: n for n, label in tiles}
    assert tiles["runs with PCA maps"] == "34/34" and tiles["most tile code (0.96)"] == "256/192/128"
    assert tiles["Spearman between-tile vs short AP (32 probed)"] == "-1.00"  # AP falls as the global view (tile code) grows
    assert [t for t, _ in dict(sections)["PCA maps"]][0].startswith("input patch (global 96, local 64)")
    report.main()
    page = (tmp_path / report.OUT).read_text()
    assert page.count("class='imggrid'") == 4 and "src='../../../outdir/e00/viewsizes-pca/d0/pca2.png'" in page  # linked
    assert "data:image/png" not in page and "none for d15, d32" in page and "repeat 2" in page
