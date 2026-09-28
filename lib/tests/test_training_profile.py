"""Exercise the full training-loop profiler without the remote volume store."""

import json
from types import SimpleNamespace

import pytest
import torch

import experiment
from lib.losses import LejepaOutput
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


class Uint8Dataset(SyntheticDataset):
    """Images on the uint8 grid in [0, 1], like real (normalized uint8) EM, so bad_batch dumps store them exactly."""

    def __getitem__(self, index):
        return {"img": torch.randint(0, 256, (1, 8, 8, 8)).float() / 255}


class NanGrad(torch.autograd.Function):
    """Identity in the forward pass, NaN gradient in the backward: a finite loss whose backward goes non-finite."""

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs[0] * float("nan")


def patch_experiment(monkeypatch, tmp_path, params):
    """Point experiment at params, a synthetic dataset and a tiny model, with no cluster, git or GPU."""
    monkeypatch.setattr(experiment, "allparams", lambda: [params])
    monkeypatch.setattr(experiment.lmd, "get", lambda name: SimpleNamespace(to_miao=lambda **kw: None))
    monkeypatch.setattr(experiment, "MiaoConfig", SimpleNamespace)
    monkeypatch.setattr(experiment, "VolumeDataset", SyntheticDataset)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Provenance independent of git state.
    monkeypatch.setattr(experiment, "code_provenance", lambda: {
        "commit_id": "test-commit", "subject": "test", "dirty": False, "diff_hash": "test-diff",
    })

    def small_model_config(**kwargs):
        return LejepaConfig(
            n_layers=1, width=16, num_heads=2, patch_size=(4, 4, 4),
            proj_hidden=16, proj_dim=8, num_slices=4, sigreg_knots=3,
            n_global=2, n_local=1,
        )

    monkeypatch.setattr(experiment, "LejepaConfig", small_model_config)


@pytest.mark.parametrize(
    "profile_steps,n_steps,recorded_steps",
    [(2, 7, 2), (0, 4, 0), (3, 4, 0), (3, 5, 1)],
)
def test_training_profile(tmp_path, monkeypatch, profile_steps, n_steps, recorded_steps):
    params = experiment.Params(
        savedir=str(tmp_path), patch_size=(8, 8, 8), batch_size=2,
        steps_per_epoch=n_steps, warmup_steps=1, benchmark_steps=2, profile_steps=profile_steps,
        n_workers=1, defer_image_ops=False,  # synthetic dataset yields finished float images
    )
    patch_experiment(monkeypatch, tmp_path, params)
    consumed = []

    def record_batch(model, args):
        consumed.extend(args[0][:, 0, 0, 0, 0].tolist())

    def small_model(config):
        model = Lejepa(config)
        model.register_forward_pre_hook(record_batch)
        return model

    monkeypatch.setattr(experiment, "Lejepa", small_model)
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


def test_nonfinite_grad_step_is_skipped(tmp_path, monkeypatch):
    params = experiment.Params(
        savedir=str(tmp_path), patch_size=(8, 8, 8), batch_size=2, steps_per_epoch=6,
        warmup_steps=1, benchmark_steps=2, profile_steps=0, n_workers=1, defer_image_ops=False,
        compile=False,  # eager on CPU, so the replay below must reproduce the step's loss exactly
    )
    patch_experiment(monkeypatch, tmp_path, params)
    monkeypatch.setattr(experiment, "VolumeDataset", Uint8Dataset)
    losses = []

    class PoisonedStep(Lejepa):  # step 3's loss is finite but its gradient is NaN
        def forward(self, *args, **kwargs):
            out = super().forward(*args, **kwargs)
            assert isinstance(out, LejepaOutput)
            losses.append(out.loss.item())
            if len(losses) == 4:
                out.loss = NanGrad.apply(out.loss)
            return out

    monkeypatch.setattr(experiment, "Lejepa", PoisonedStep)
    experiment.run(0)

    metrics = [json.loads(line) for line in (tmp_path / "metrics.json").read_text().splitlines()]
    assert metrics[-1]["idx_step"] == 5 and metrics[-1]["skipped"] == 1  # training continued past the bad step
    dump = torch.load(tmp_path / "bad_batch_0000003.pt", weights_only=False)
    assert dump["step"] == 3 and dump["x_uint8"].dtype == torch.uint8 and dump["rng_cpu"] is not None
    ckpt = torch.load(sorted((tmp_path / "checkpoints").glob("step_*.pt"))[-1], weights_only=False)
    assert all(torch.isfinite(v).all() for v in ckpt["model"].values() if v.is_floating_point())

    # Replay: same weights, input and RNG give the same views and SIGReg slices, so the same loss
    # (the NaN came from the test's poisoned model, which the replay doesn't use).
    replay = experiment.replay_bad_batch(str(tmp_path / "bad_batch_0000003.pt"))
    runs = [json.loads(l) for l in (tmp_path / "runs.json").read_text().splitlines()]
    assert [r["fn"] for r in runs] == ["run", "replay_bad_batch"] and runs[0]["commit_id"] == "test-commit"
    assert replay["eager"][0] == pytest.approx(losses[3], rel=1e-5)
    assert replay["as trained"][0] == pytest.approx(losses[3], rel=1e-5)  # compile=False: both paths are eager
