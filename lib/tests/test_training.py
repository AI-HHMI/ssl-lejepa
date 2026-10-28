"""Tests for lib.training: the step, its logs and profile, the Trainer's lr and checkpoints, the skipped
non-finite step and its replay, benchmark. The data is synthetic and the model tiny; no cluster, git or GPU."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lib import readers, specs, training
from lib.data import Volume
from lib.encoders import EncoderConfig
from lib.losses import LejepaOutput, SIGRegConfig
from lib.models import Lejepa, LejepaConfig, MaeConfig, MaskConfig, ProjectorConfig
from lib.specs import Train
from lib.views import ViewConfig


def small(objective=None, profile_steps=0, compile=True, qk_norm=False) -> Train:
    """A Train on 8^3 windows and a synthetic loader (no workers)."""
    return Train(model=objective or specs.Masked(), encoder=specs.Encoder(qk_norm=qk_norm), data=specs.Data(window=(8, 8, 8)),
                 optim=specs.Optim(batch_size=2), hardware=specs.Hardware(n_workers=0), speed=specs.Speed(defer_image_ops=False, compile=compile),
                 profile=specs.Profile(warmup_steps=1, benchmark_steps=2, profile_steps=profile_steps))


def train(par: Train, savedir, steps: int, lr1: float | None = None):
    """A Trainer's one train() of steps steps, the lr par.optim.lr (to lr1, if given); the Trainer, closed."""
    with training.Trainer(par, str(savedir), steps) as t:
        t.train(steps, par.optim.lr, par.optim.lr if lr1 is None else lr1)
    return t


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
    monkeypatch.setattr(readers.lmd, "get", lambda name: SimpleNamespace(to_miao=lambda **kw: None))
    monkeypatch.setattr(training, "sampling", lambda names, power: [(Volume("synthetic", [[0, 1]] * 3), 1.0)])
    monkeypatch.setattr(training, "MiaoConfig", SimpleNamespace)
    monkeypatch.setattr(training, "VolumeDataset", SyntheticDataset)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def small_model_config(par):  # par's objective, tiny
        encoder = EncoderConfig(depth=1, embed_dim=16, num_heads=2, patch_size=(4, 4, 4))
        if par.model.kind == "mae":
            return MaeConfig(encoder=encoder, decoder_dim=8, decoder_heads=2, decoder_depth=1)
        return LejepaConfig(
            encoder=encoder, objective=par.model.kind, view=ViewConfig(n_global=2, n_local=1) if par.model.kind == "views" else ViewConfig(),
            mask=MaskConfig(cube=1), projector=ProjectorConfig(dim=8, hidden=16), sigreg=SIGRegConfig(num_slices=4, knots=3),
        )

    monkeypatch.setattr(specs, "model_config", small_model_config)
    monkeypatch.setattr(training, "model_config", small_model_config)


@pytest.mark.parametrize(
    "profile_steps,n_steps,recorded_steps",
    [(2, 7, 2), (0, 4, 0), (3, 4, 0), (3, 5, 1)],
)
def test_training_profile(tmp_path, monkeypatch, profile_steps, n_steps, recorded_steps):
    params = small(profile_steps=profile_steps)  # the synthetic dataset yields finished float images, loaded in-process (no worker startup)
    patch_training(monkeypatch)
    consumed = []

    def record_batch(model, args):
        consumed.extend(args[0][:, 0, 0, 0, 0].tolist())

    def small_model(config):
        model = Lejepa(config)
        model.register_forward_pre_hook(record_batch)
        return model

    monkeypatch.setattr(specs.models, "Lejepa", small_model)
    train(params, tmp_path, n_steps)

    assert consumed == list(range(n_steps * params.optim.batch_size))
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
    params = small({"views": specs.Views(), "mae": specs.Mae()}[objective])
    patch_training(monkeypatch)
    train(params, tmp_path, 3).save(tmp_path / "checkpoint.pt")
    metrics = [json.loads(line) for line in (tmp_path / "metrics.json").read_text().splitlines()]
    assert metrics[-1]["idx_step"] == 2 and all(torch.isfinite(torch.tensor(row["loss"])) for row in metrics)
    if objective == "views":  # LeJEPA's two terms, logged apart
        assert all(row["loss"] == pytest.approx(row["inv"] + LejepaConfig().lamb * row["sigreg"], rel=1e-5) for row in metrics)
    else:  # an MAE's reconstruction error alone
        assert not {"inv", "sigreg"} & set(metrics[-1])
    model, step = specs.load_checkpoint(params, str(tmp_path / "checkpoint.pt"))  # strict: the saved weights fit the objective's model
    assert step == 3 and type(model).__name__ == {"views": "Lejepa", "mae": "Mae"}[objective]


def test_nonfinite_grad_step_is_skipped(tmp_path, monkeypatch):
    run = tmp_path / "d0"
    run.mkdir()
    params = small(compile=False)  # eager on CPU, so the replay below must reproduce the step's loss exactly
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

    monkeypatch.setattr(specs.models, "Lejepa", PoisonedStep)
    train(params, run, 6).save(run / "checkpoint.pt")

    metrics = [json.loads(line) for line in (run / "metrics.json").read_text().splitlines()]
    assert metrics[-1]["idx_step"] == 5 and metrics[-1]["skipped"] == 1  # training continued past the bad step
    dump = torch.load(run / "bad_batch_0000003.pt", weights_only=False)
    assert dump["step"] == 3 and dump["x_uint8"].dtype == torch.uint8 and dump["rng_cpu"] is not None
    ckpt = torch.load(run / "checkpoint.pt", weights_only=False)
    assert all(torch.isfinite(v).all() for v in ckpt["model"].values() if v.is_floating_point())

    # Replay: same weights, input and RNG give the same views and SIGReg slices, so the same loss
    # (the NaN came from the test's poisoned model, which the replay doesn't use).
    replay = training.replay_bad_batch(torch.load(run / "bad_batch_0000003.pt", weights_only=False))
    assert replay["eager"][0] == pytest.approx(losses[3], rel=1e-5)
    assert replay["as trained"][0] == pytest.approx(losses[3], rel=1e-5)  # compile=False: both paths are eager


def test_benchmark_times_and_profiles_the_training_step_only(tmp_path, monkeypatch):
    params = small(profile_steps=2)
    patch_training(monkeypatch)
    training.benchmark(params, str(tmp_path))
    performance = json.loads((tmp_path / "performance.json").read_text())
    assert performance["steps"] == 2 and performance["samples_per_second"] > 0
    assert (tmp_path / "profile.json").is_file() and (tmp_path / "trace_summary.json").is_file()
    assert not (tmp_path / "metrics.json").exists()  # not a training run


def test_the_lr_goes_linearly_from_lr0_to_lr1_after_the_warmup(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "LR_WARMUP", 20)
    par = small(compile=False, qk_norm=True)
    patch_training(monkeypatch)
    train(par, tmp_path, 200, lr1=0.0)
    rows = [json.loads(line) for line in (tmp_path / "metrics.json").read_text().splitlines()]
    lr = par.optim.lr
    lerp = lambda i: lr * (1 - i / 200) * min(1, (i + 1) / 20)  # lr0 -> lr1 over the call, times the warmup
    assert all(r["lr"] == pytest.approx(lerp(r["idx_step"])) for r in rows)
    assert max(r["lr"] for r in rows) == pytest.approx(lr * 0.9, rel=0.1) and rows[-1]["lr"] < 0.02 * lr  # up, then down to ~0


def test_qk_norm_normalizes_each_heads_queries_and_keys():
    from lib.encoders.vit3d import Attention
    x = torch.randn(2, 5, 32) * 50
    plain, normed = Attention(32, num_heads=4), Attention(32, num_heads=4, qk_norm=True)
    assert isinstance(normed.q_norm, torch.nn.LayerNorm) and isinstance(plain.q_norm, torch.nn.Identity)
    assert normed(x).shape == plain(x).shape == (2, 5, 32)
    assert len(list(normed.parameters())) == len(list(plain.parameters())) + 4  # a weight and bias each, for q and k


def test_a_trainer_trains_in_calls_and_a_checkpoint_continues_it(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "LR_WARMUP", 100)  # a short warmup, so a short trunk gets past it
    par = small(compile=False)
    lr = par.optim.lr
    patch_training(monkeypatch)
    trunk_dir, branch_dir = tmp_path / "trunk", tmp_path / "trunk/step_0000130"
    trunk_dir.mkdir()
    branch_dir.mkdir()
    with training.Trainer(par, str(trunk_dir), 130) as t:
        t.train(65, lr, lr)
        t.train(65, lr, lr)  # the second call goes on where the first stopped: the step, the warmup, the loader
        assert t.step == 130
        t.save(branch_dir / "start.pt")
    trunk = [json.loads(line) for line in (trunk_dir / "metrics.json").read_text().splitlines()]
    assert [r["idx_step"] for r in trunk] == list(range(0, 70, 10)) + [64] + list(range(70, 130, 10)) + [129]  # one run of steps
    assert all(r["lr"] == pytest.approx(lr * min(1, (r["idx_step"] + 1) / 100)) for r in trunk)  # warmup, then constant
    assert trunk[-1]["lr"] == lr

    with training.Trainer(par, str(branch_dir), 12, torch.load(branch_dir / "start.pt", weights_only=False)) as t:
        assert t.step == 130  # the checkpoint's
        t.train(12, lr, 0.01 * lr)
    metrics = [json.loads(line) for line in (branch_dir / "metrics.json").read_text().splitlines()]
    assert metrics[0]["idx_step"] == 130 and metrics[0]["lr"] == pytest.approx(lr) and metrics[-1]["lr"] == pytest.approx(lr * (1 - 0.99 * 11 / 12))  # lr0 -> lr1 over the call
    t.save(branch_dir / "checkpoint.pt")
    done = torch.load(branch_dir / "checkpoint.pt", weights_only=False)
    assert done["step"] == 142 and done["opt"]["state"][0]["step"] == 142  # Adam's moments and step count carried over


def test_a_trainer_stops_on_a_non_finite_loss_and_trains_no_more(tmp_path, monkeypatch):
    par = small(compile=False)
    lr = par.optim.lr
    patch_training(monkeypatch)
    monkeypatch.setattr(training, "stop_reason", lambda *a: "diverged")
    with training.Trainer(par, str(tmp_path), 500) as t:
        t.train(200, lr, lr)
        assert t.stopped == "diverged" and t.step == 100  # agreed every 100 steps
        t.train(200, lr, lr)
        assert t.step == 100
