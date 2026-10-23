"""Tests for lib.training: the step, its logs and profile, checkpoints and the callback at each milestone, the skipped
non-finite step and its replay, benchmark. The data is synthetic and the model tiny; no cluster, git or GPU."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lib import specs, training
from lib.encoders import EncoderConfig
from lib.losses import LejepaOutput, SIGRegConfig
from lib.models import Lejepa, LejepaConfig, MaeConfig, ProjectorConfig
from lib.specs import Train
from lib.views import ViewConfig

NO_MILESTONE_EVAL = lambda par, step: None


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


def patch_training(monkeypatch):
    """Point lib.training at a synthetic dataset and a tiny model, with no volumes or GPU."""
    monkeypatch.setattr(training.lmd, "get", lambda name: SimpleNamespace(to_miao=lambda **kw: None))
    monkeypatch.setattr(training, "MiaoConfig", SimpleNamespace)
    monkeypatch.setattr(training, "VolumeDataset", SyntheticDataset)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def small_model_config(par):  # par's objective, tiny
        encoder = EncoderConfig(depth=1, embed_dim=16, num_heads=2, patch_size=(4, 4, 4))
        if par.objective == "mae":
            return MaeConfig(encoder=encoder, decoder_dim=8, decoder_heads=2, decoder_depth=1)
        return LejepaConfig(
            encoder=encoder, objective=par.objective, view=ViewConfig(n_global=2, n_local=1) if par.objective == "views" else ViewConfig(),
            projector=ProjectorConfig(dim=8, hidden=16), sigreg=SIGRegConfig(num_slices=4, knots=3),
        )

    monkeypatch.setattr(specs, "model_config", small_model_config)


@pytest.mark.parametrize(
    "profile_steps,n_steps,recorded_steps",
    [(2, 7, 2), (0, 4, 0), (3, 4, 0), (3, 5, 1)],
)
def test_training_profile(tmp_path, monkeypatch, profile_steps, n_steps, recorded_steps):
    params = Train(
        savedir=str(tmp_path), patch_size=(8, 8, 8), batch_size=2,
        steps_per_epoch=n_steps, warmup_steps=1, benchmark_steps=2, profile_steps=profile_steps,
        n_workers=0, defer_image_ops=False,  # synthetic dataset yields finished float images; loaded in-process (no worker startup)
    )
    patch_training(monkeypatch)
    consumed = []

    def record_batch(model, args):
        consumed.extend(args[0][:, 0, 0, 0, 0].tolist())

    def small_model(config):
        model = Lejepa(config)
        model.register_forward_pre_hook(record_batch)
        return model

    monkeypatch.setattr(specs, "Lejepa", small_model)
    training.train(params, NO_MILESTONE_EVAL)

    assert consumed == list(range(n_steps * params.batch_size))
    metrics = [json.loads(line) for line in (tmp_path / "metrics.json").read_text().splitlines()]
    assert metrics[-1]["idx_step"] == n_steps - 1  # Training continues beyond capture.
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in metrics)
    lamb = LejepaConfig().lamb  # small_model_config keeps the default
    assert all(row["loss"] == pytest.approx(row["inv"] + lamb * row["sigreg"], rel=1e-5) for row in metrics)  # logged apart
    performance = json.loads((tmp_path / "performance.json").read_text())
    assert performance["steps"] == 2
    assert performance["samples_per_second"] > 0
    assert performance["tflops_per_second"] > 0 and "mfu" not in performance  # CPU run: no peak to compare to
    # 8^3 input, 4^3 patches: one 2^3-token view, encoded whole and again without its masked 1^3 token.
    assert performance["tokens_per_sample"] == 8 + 7
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


@pytest.mark.parametrize("objective", ["views", "mae"])
def test_each_objective_trains_logs_its_terms_and_reloads_for_eval(tmp_path, monkeypatch, objective):
    params = Train(savedir=str(tmp_path), objective=objective, patch_size=(8, 8, 8), batch_size=2, steps_per_epoch=3,
                              warmup_steps=1, benchmark_steps=1, profile_steps=0, n_workers=0, defer_image_ops=False)
    patch_training(monkeypatch)
    training.train(params, NO_MILESTONE_EVAL)
    metrics = [json.loads(line) for line in (tmp_path / "metrics.json").read_text().splitlines()]
    assert metrics[-1]["idx_step"] == 2 and all(torch.isfinite(torch.tensor(row["loss"])) for row in metrics)
    if objective == "views":  # LeJEPA's two terms, logged apart
        assert all(row["loss"] == pytest.approx(row["inv"] + LejepaConfig().lamb * row["sigreg"], rel=1e-5) for row in metrics)
    else:  # an MAE's reconstruction error alone
        assert not {"inv", "sigreg"} & set(metrics[-1])
    model, step = specs.load_checkpoint(specs.own_eval(params))  # strict: the saved weights fit the objective's model
    assert step == 3 and type(model).__name__ == {"views": "Lejepa", "mae": "Mae"}[objective]


def test_nonfinite_grad_step_is_skipped(tmp_path, monkeypatch):
    run = tmp_path / "d0"
    run.mkdir()
    params = Train(
        savedir=f"{run}/", patch_size=(8, 8, 8), batch_size=2, steps_per_epoch=6,
        warmup_steps=1, benchmark_steps=2, profile_steps=0, n_workers=0, defer_image_ops=False,
        compile=False,  # eager on CPU, so the replay below must reproduce the step's loss exactly
    )
    patch_training(monkeypatch)
    monkeypatch.setattr(training, "VolumeDataset", Uint8Dataset)
    losses = []

    class PoisonedStep(Lejepa):  # step 3's loss is finite but its gradient is NaN
        def forward(self, *args, **kwargs):
            out = super().forward(*args, **kwargs)
            assert isinstance(out, LejepaOutput)
            losses.append(out.loss.item())
            if len(losses) == 4:
                out.loss = NanGrad.apply(out.loss)
            return out

    monkeypatch.setattr(specs, "Lejepa", PoisonedStep)
    training.train(params, NO_MILESTONE_EVAL)

    metrics = [json.loads(line) for line in (run / "metrics.json").read_text().splitlines()]
    assert metrics[-1]["idx_step"] == 5 and metrics[-1]["skipped"] == 1  # training continued past the bad step
    dump = torch.load(run / "bad_batch_0000003.pt", weights_only=False)
    assert dump["step"] == 3 and dump["x_uint8"].dtype == torch.uint8 and dump["rng_cpu"] is not None
    ckpt = torch.load(sorted((run / "checkpoints").glob("step_*.pt"))[-1], weights_only=False)
    assert all(torch.isfinite(v).all() for v in ckpt["model"].values() if v.is_floating_point())

    # Replay: same weights, input and RNG give the same views and SIGReg slices, so the same loss
    # (the NaN came from the test's poisoned model, which the replay doesn't use).
    replay = training.replay_bad_batch(torch.load(run / "bad_batch_0000003.pt", weights_only=False))
    assert replay["eager"][0] == pytest.approx(losses[3], rel=1e-5)
    assert replay["as trained"][0] == pytest.approx(losses[3], rel=1e-5)  # compile=False: both paths are eager


def test_benchmark_times_and_profiles_the_training_step_only(tmp_path, monkeypatch):
    params = Train(
        savedir=str(tmp_path), patch_size=(8, 8, 8), batch_size=2, steps_per_epoch=1000, max_hours=1.0,
        warmup_steps=1, benchmark_steps=2, profile_steps=2, n_workers=0, defer_image_ops=False,
    )
    patch_training(monkeypatch)
    training.benchmark(params)
    performance = json.loads((tmp_path / "performance.json").read_text())
    assert performance["steps"] == 2 and performance["samples_per_second"] > 0
    assert (tmp_path / "profile.json").is_file() and (tmp_path / "trace_summary.json").is_file()
    assert not (tmp_path / "metrics.json").exists() and not (tmp_path / "checkpoints").exists()  # not a training run


def test_a_trunk_hands_each_milestone_to_its_callback_and_an_anneal_continues_one(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "LR_WARMUP", 100)  # a short warmup, so a short trunk gets past it
    src = Train(
        savedir=f"{tmp_path}/d0/", patch_size=(8, 8, 8), batch_size=2, steps_per_epoch=130, max_hours=1.0,  # past LR_WARMUP
        warmup_steps=1, benchmark_steps=2, profile_steps=0, n_workers=0, defer_image_ops=False, compile=False,
    )
    patch_training(monkeypatch)
    seen = []
    training.train(src, lambda par, step: seen.append((step, specs.milestone(par, step).is_file())))  # its milestone is on disk when called

    steps = range(13, 131, 13)
    assert seen == [(s, True) for s in steps]  # the last once only
    ckdir = Path(src.savedir) / "checkpoints"
    assert [p.name for p in ckdir.glob("step_*.pt")] == ["step_0000130.pt"]  # rolling: the newest only
    assert [p.name for p in sorted((ckdir / "milestones").glob("*.pt"))] == [f"step_{s:07d}.pt" for s in steps]  # the callback released none
    trunk = [json.loads(line) for line in (Path(src.savedir) / "metrics.json").read_text().splitlines()]
    assert all(r["lr"] == pytest.approx(src.lr * min(1, (r["idx_step"] + 2) / 100)) for r in trunk)  # warmup, then constant, no decay
    assert trunk[-1]["lr"] == src.lr

    par = replace(src, savedir=specs.branch_dir(src, 117), steps_per_epoch=12, anneal_from=str(specs.milestone(src, 117)))
    training.train(par, lambda par, step: pytest.fail("an anneal has no milestones"))
    start = torch.load(par.anneal_from, weights_only=False)
    metrics = [json.loads(line) for line in (Path(par.savedir) / "metrics.json").read_text().splitlines()]
    lr0 = start["opt"]["param_groups"][0]["lr"]
    assert lr0 == src.lr and metrics[0]["lr"] <= lr0 and metrics[-1]["lr"] == pytest.approx(0.01 * lr0)  # trunk's lr down to 1%
    done = torch.load(sorted((Path(par.savedir) / "checkpoints").glob("step_*.pt"))[-1], weights_only=False)
    assert not (Path(par.savedir) / "checkpoints/milestones").exists()
    assert done["step"] == 12 and done["opt"]["state"][0]["step"] == 117 + 12  # Adam's step count carried over
