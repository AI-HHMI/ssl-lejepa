"""Tests for reports/tile_size.py: the e06-01 page plots pca_sizes.json against the tile side and links each map."""

import json

import pytest

pytest.importorskip("pandas")  # the `analysis` extra

from reports import tile_size  # noqa: E402


def test_report_plots_each_statistic_and_links_each_map(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "outdir" / tile_size.SWEEP / "d0"
    d.mkdir(parents=True)
    rows = [{"tbl": "pca_sizes", "step": 133600, "tile": t, "tiles": 4 * (512 // t) ** 3, "effective_rank": 300.0 - abs(t - 64),
             "explained_variance": [0.1] * 10, "between_tile_variance": 16 / t} for t in (16, 64, 512)]
    (d / "pca_sizes.json").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for t in (16, 64, 512):
        (d / f"pca_tile_{t:03d}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    tile_size.main()
    html = (tmp_path / "results" / tile_size.SWEEP / "report.html").read_text()
    assert "300 @ 64" in html and "between-tile variance vs test tile side" in html
    assert all(f"pca_tile_{t:03d}.png" in html for t in (16, 64, 512))
