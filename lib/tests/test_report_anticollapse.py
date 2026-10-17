"""Tests for reports/anticollapse.py: the e03 page labels each trunk by its fix and pull, and draws losses before any probe."""

import json

import pytest

pytest.importorskip("pandas")  # the `analysis` extra

from reports import anticollapse  # noqa: E402

CHANNELS = ["(1, 0, 0)", "(0, 1, 0)", "(0, 0, 1)", "(10, 0, 0)", "(0, 10, 0)", "(0, 0, 10)"]


def write(d, name, *rows):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_report_labels_trunks_and_plots_their_probes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    e03 = tmp_path / "outdir/e00/e03-anticollapse"
    base = {"init_from": "", "batch_size": 32, "global_size": [96] * 3, "mask_block": 0.3}
    write(e03 / "d0", "runs.json", {"params": {**base, "sigreg_over": "batch", "stop_grad": False, "lamb": 0.1}})
    write(e03 / "d1", "runs.json", {"params": {**base, "sigreg_over": "view", "stop_grad": True, "lamb": 10 / 1728}})
    for d in ["d0", "d1"]:
        write(e03 / d, "metrics.json", *[{"tbl": "metrics", "idx_step": 10 * i, "time": i, "loss": 1.0, "inv": 0.5, "sigreg": 5.0,
                                          "resid_norm": 100.0} for i in range(40)])
        benchmarked(e03 / d)
    write(tmp_path / "outdir/e00/probe-test/d1", "probe.json", {"tbl": "probe", **{f"boundary_ap_{c}": 0.1 for c in CHANNELS}})

    tiles, sections = anticollapse.build()  # trunks only, nothing probed yet: losses alone
    assert [s for s, _ in sections] == ["Loss trajectories"] and "d0 batch block 0.3 pull 3.2" in sections[0][1][0][1]
    assert "d1 stopgrad block 0.3 pull 10" in sections[0][1][0][1]

    write(e03 / "d2", "runs.json", {"params": {**base, "init_from": "outdir/e00/e03-anticollapse/d0/checkpoints/milestones/step_0005000.pt"}})
    probed(e03 / "d2", 5000, 0.3)
    tiles, sections = anticollapse.build()
    assert dict((label, n) for n, label in tiles)["best short AP: d0 batch block 0.3 pull 3.2 @ 5000"] == "0.300"
    assert [s for s, _ in sections] == ["Loss trajectories", "Probe vs trunk step", "PCA maps"]
    anticollapse.main()
    assert (tmp_path / "results/e00/e03-anticollapse/report.html").is_file()


def benchmarked(d, tflop_per_step=2.0):
    write(d, "performance.json", {"tbl": "throughput", "tflops_per_second": 4 * tflop_per_step, "seconds_per_step": 0.25})


def probed(d, step, ap):
    write(d, "probe.json", {"tbl": "probe", "step": step, **{f"boundary_ap_{c}": ap for c in CHANNELS}, **{f"boundary_frac_{c}": 0.02 for c in CHANNELS}})
    write(d, "pca.json", {"tbl": "pca", "step": step, "between_tile_variance": 0.01})


def test_report_reads_spec_files_and_milestone_evals_nested_in_their_trunk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    e04 = tmp_path / "outdir/e00/e04-sigreg-both"
    trunk = {"savedir": "outdir/e00/e04-sigreg-both/d0/", "batch_size": 32, "global_size": [96] * 3, "mask_block": 0.6,
             "sigreg_over": "both", "stop_grad": False, "lamb": 0.1}
    write(e04 / "d0", "train.json", trunk)
    write(e04 / "d0", "metrics.json", {"tbl": "metrics", "idx_step": 0, "time": 0, "loss": 1.0, "inv": 0.5, "sigreg": 5.0, "resid_norm": 100.0})
    benchmarked(e04 / "d0")
    write(e04 / "d0/step_0005000", "eval.json", {"savedir": "outdir/e00/e04-sigreg-both/d0/step_0005000/", "train": trunk,
                                                 "init_from": "outdir/e00/e04-sigreg-both/d0/checkpoints/milestones/step_0005000.pt"})
    probed(e04 / "d0/step_0005000", 5000, 0.2)
    probed(e04 / "d0", 7300, 0.4)  # the trunk's own end, off a milestone
    write(tmp_path / "outdir/e00/probe-test/d1", "probe.json", {"tbl": "probe", **{f"boundary_ap_{c}": 0.1 for c in CHANNELS}})
    tiles = dict((label, n) for n, label in anticollapse.build("e00/e04-sigreg-both")[0])
    assert tiles["trunks started"] == "1" and tiles["checkpoints probed"] == "2"
    assert tiles["best short AP: d0 both block 0.6 pull 3.2 @ 7300"] == "0.400"


def test_report_names_trunks_by_its_label_and_plots_probes_vs_compute(tmp_path, monkeypatch):
    """An e06-like sweep: an MAE trunk (no inv or sigreg logged) beside a masked one, named by the report's own label."""
    monkeypatch.chdir(tmp_path)
    sweep = tmp_path / "outdir/e00/e06-test"
    for d, objective, tflop in [("d0", "mask", 2.0), ("d1", "mae", 0.5)]:
        write(sweep / d, "train.json", {"savedir": f"outdir/e00/e06-test/{d}/", "objective": objective, "global_size": [96] * 3})
        terms = {"inv": 0.5, "sigreg": 5.0} if objective == "mask" else {}
        write(sweep / d, "metrics.json", *[{"tbl": "metrics", "idx_step": i, "time": i, "loss": 1.0, **terms, "resid_norm": 100.0} for i in range(40)])
        benchmarked(sweep / d, tflop)
        probed(sweep / d, 1000, 0.3 if objective == "mae" else 0.2)
    write(tmp_path / "outdir/e00/probe-test/d1", "probe.json", {"tbl": "probe", **{f"boundary_ap_{c}": 0.1 for c in CHANNELS}})
    tiles, sections = anticollapse.build("e00/e06-test", label=lambda p: p["objective"])
    assert dict((label, n) for n, label in tiles)["best short AP: d1 mae @ 1000"] == "0.300"
    cards = dict(sections)["Probe vs trunk step"]
    assert any("training compute" in title and "training PFLOP" in fragment for title, fragment in cards)


def test_cell_reads_both_as_the_batch_pull():
    p = {"sigreg_over": "both", "stop_grad": False, "batch_size": 32, "global_size": [96] * 3, "mask_block": 0.6, "lamb": 0.1}
    assert anticollapse.cell(p) == "both block 0.6 pull 3.2"
    assert anticollapse.cell({**p, "mask_shift": 32}) == "both block 0.6 pull 3.2 shift 32"
