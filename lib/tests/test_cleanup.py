"""Tests for cleanup.py: only probe artifacts whose source checkpoint survives are deleted."""

import json

import cleanup


def make_probed(d, init_from="", checkpoint=False):
    (d / "probe/test/hemibrain_eb_test.zarr").mkdir(parents=True)
    (d / "probe/test/hemibrain_eb_test.zarr/zarr.json").write_text("{}")
    (d / "probe.json").write_text("{}\n")
    (d / "runs.json").write_text(json.dumps({"fn": "probe", "params": {"init_from": init_from}}) + "\n")
    if checkpoint:
        (d / "checkpoints").mkdir()
        (d / "checkpoints/step_0000010.pt").touch()


def test_deletes_only_recreatable_probe_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    e00 = tmp_path / "outdir/e00"
    make_probed(e00 / "own/d0", checkpoint=True)  # probed its own checkpoint
    make_probed(e00 / "lost/d0")  # its checkpoint is gone: keep the artifact
    (e00 / "train/d3/checkpoints").mkdir(parents=True)
    (e00 / "train/d3/checkpoints/step_0000099.pt").touch()
    make_probed(e00 / "eval/d0", init_from="outdir/e00/train/d3/")  # probed another run's checkpoint
    make_probed(e00 / "eval/d1", init_from="random")
    make_probed(tmp_path / "outdir/.trash/e00/own/d0", checkpoint=True)  # trash is not ours to delete
    assert [str(z.parents[2]) for z in cleanup.recreatable()] == ["outdir/e00/eval/d0", "outdir/e00/eval/d1", "outdir/e00/own/d0"]
    cleanup.delete_zarrs()
    assert not list((e00 / "own").glob("d0/probe/*/*.zarr")) and (e00 / "own/d0/probe.json").is_file()  # stats stay
    assert list((e00 / "lost").glob("d0/probe/*/*.zarr")) and list((tmp_path / "outdir/.trash").glob("*/*/d0/probe/*/*.zarr"))
