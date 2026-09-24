"""Exercise the full training-loop profiler without the remote volume store."""

import json
from types import SimpleNamespace

import pytest
import torch

import e00_basic as experiment
from lib.models import Lejepa, LejepaConfig


class SyntheticDataset:
    """Picklable dataset for DataLoader workers, with an index marker per image."""

    def __init__(self, config):
        self.length = config.samples_per_epoch

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = torch.randn(1, 8, 8, 8)
        image[0, 0, 0, 0] = index
        return {"img": image}


@pytest.mark.parametrize(
    "profile_steps,n_steps,recorded_steps",
    [(2, 7, 2), (0, 4, 0), (3, 4, 0), (3, 5, 1)],
)
def test_training_profile(tmp_path, monkeypatch, profile_steps, n_steps, recorded_steps):
    params = experiment.Params(
        savedir=str(tmp_path), patch_size=(8, 8, 8), batch_size=2,
        steps_per_epoch=n_steps, warmup_steps=1, benchmark_steps=2, profile_steps=profile_steps,
        n_workers=1,
    )
    monkeypatch.setattr(experiment, "allparams", lambda: [params])
    monkeypatch.setattr(experiment.lmd, "get", lambda name: SimpleNamespace(to_miao=lambda **kw: None))
    monkeypatch.setattr(experiment, "MiaoConfig", SimpleNamespace)
    monkeypatch.setattr(experiment, "VolumeDataset", SyntheticDataset)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Keep provenance writes in the test directory and independent of git state.
    (tmp_path / "_diffs").mkdir()
    monkeypatch.setattr(experiment, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(experiment, "git_provenance", lambda: {
        "commit_id": "test-commit", "diff_hash": "test-diff", "diff": "",
    })
    consumed = []

    def record_batch(model, args):
        consumed.extend(args[0][:, 0, 0, 0, 0].tolist())

    def small_model(config):
        model = Lejepa(config)
        model.register_forward_pre_hook(record_batch)
        return model

    monkeypatch.setattr(experiment, "Lejepa", small_model)

    def small_model_config(**kwargs):
        return LejepaConfig(
            n_layers=1, width=16, num_heads=2, patch_size=(4, 4, 4),
            proj_hidden=16, proj_dim=8, num_slices=4, sigreg_knots=3,
            n_global=2, n_local=1,
        )

    monkeypatch.setattr(experiment, "LejepaConfig", small_model_config)
    experiment.run(0)

    assert consumed == list(range(n_steps * params.batch_size))
    metrics = [json.loads(line) for line in (tmp_path / "metrics.json").read_text().splitlines()]
    assert metrics[-1]["idx_step"] == n_steps - 1  # Training continues beyond capture.
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in metrics)
    performance = json.loads((tmp_path / "performance.json").read_text())
    assert performance["steps"] == 2
    assert performance["samples_per_second"] > 0
    assert performance["tflops_per_second"] > 0 and "mfu" not in performance  # CPU run: no peak to compare to
    # 8^3 input, 4^3 patches: 2 globals of 2^3 tokens, 1 local of 1^3 or 2^3 tokens.
    assert 2 * 8 + 1 <= performance["tokens_per_sample"] <= 3 * 8
    assert performance["seconds_per_step"] == performance["seconds"] / 2

    trace_path = tmp_path / "profile.json"
    assert trace_path.exists() == bool(recorded_steps)
    if recorded_steps:
        trace = json.loads(trace_path.read_text())
        for phase in (
            "01_DATA_IO", "03_H2D_TRANSFER", "04_FORWARD_AND_LOSS",
            "05_BACKWARD", "06_OPTIMIZER", "07_LOGGING",
        ):
            assert sum(event.get("name") == phase for event in trace["traceEvents"]) == recorded_steps
        assert "SELF CPU TIME" in (tmp_path / "profile.out").read_text()
        summary = json.loads((tmp_path / "trace_summary.json").read_text())
        assert summary["step_ms"] > 0 and summary["gpu_busy"] == 0  # CPU run: no GPU kernels.
        assert summary["04_FORWARD_AND_LOSS_ms"] > 0
