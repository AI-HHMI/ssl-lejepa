"""Exercise the full training-loop profiler without the remote volume store."""

import json
from types import SimpleNamespace

import pytest
import torch

import e00_basic as experiment
from lib.models import LejepaConfig


@pytest.mark.parametrize(
    "profile_steps,n_steps,recorded_steps",
    [(2, 7, 2), (0, 4, 0), (3, 4, 0), (3, 5, 1)],
)
def test_training_profile(tmp_path, monkeypatch, profile_steps, n_steps, recorded_steps):
    params = experiment.Params(
        savedir=str(tmp_path), patch_size=[8, 8, 8], batch_size=2,
        n_epoch=n_steps, warmup_steps=1, benchmark_steps=2, profile_steps=profile_steps,
    )
    monkeypatch.setattr(experiment, "allparams", lambda: [params])
    volume = SimpleNamespace(
        name="exm-drosophila-flyliconn-matt-260601-60X-B4-2-045/crop-001",
        to_miao=lambda: None,
    )
    monkeypatch.setattr(experiment.lmd, "all", lambda: [volume])
    monkeypatch.setattr(experiment, "MiaoConfig", lambda **kwargs: None)
    reads = []

    class Dataset:
        def __getitem__(self, index):
            reads.append(index)
            return {"img": torch.randn(1, 8, 8, 8)}

    monkeypatch.setattr(experiment, "VolumeDataset", lambda config: Dataset())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def small_model_config(**kwargs):
        assert kwargs["profile"] is False  # No nested first-forward profiler.
        return LejepaConfig(
            n_layers=1, width=16, num_heads=2, patch_size=(4, 4, 4),
            proj_hidden=16, proj_dim=8, num_slices=4, sigreg_knots=3,
            n_global=2, n_local=1, profile=False,
        )

    monkeypatch.setattr(experiment, "LejepaConfig", small_model_config)
    experiment.run(0)

    assert reads == list(range(n_steps * params.batch_size))
    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["epoch"] == n_steps - 1  # Training continues beyond capture.
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in metrics)
    performance = json.loads((tmp_path / "performance.jsonl").read_text())
    assert performance["steps"] == 2
    assert performance["samples_per_second"] > 0
    assert performance["seconds_per_step"] == performance["seconds"] / 2

    trace_path = tmp_path / "profile.json"
    assert trace_path.exists() == bool(recorded_steps)
    if recorded_steps:
        trace = json.loads(trace_path.read_text())
        for phase in (
            "01_DATA_IO", "02_CPU_BATCH", "03_H2D_TRANSFER", "04_FORWARD_AND_LOSS",
            "05_BACKWARD", "06_OPTIMIZER", "07_LOGGING",
        ):
            assert sum(event.get("name") == phase for event in trace["traceEvents"]) == recorded_steps
        assert "SELF CPU TIME" in (tmp_path / "profile.out").read_text()
