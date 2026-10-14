"""Exercise the full training-loop profiler without the remote volume store."""

import json
import re
from pathlib import Path
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
    monkeypatch.setattr(experiment, "paramsall", lambda: [params])
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
    lamb = LejepaConfig().lamb  # small_model_config keeps the default
    assert all(row["loss"] == pytest.approx(row["inv"] + lamb * row["sigreg"], rel=1e-5) for row in metrics)  # logged apart
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
    assert runs[0]["params"]["savedir"] == params.savedir  # every run dir describes itself, even if it crashes
    assert replay["eager"][0] == pytest.approx(losses[3], rel=1e-5)
    assert replay["as trained"][0] == pytest.approx(losses[3], rel=1e-5)  # compile=False: both paths are eager


def test_trunk_spawns_each_milestones_anneal_as_it_saves_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "EXPERIMENT", "e00/e01-test")
    src = experiment.Params(
        savedir="outdir/e00/e01-test/d0/", patch_size=(8, 8, 8), batch_size=2, steps_per_epoch=1300, max_hours=1.0,  # past LR_WARMUP
        warmup_steps=1, benchmark_steps=2, profile_steps=0, n_workers=1, defer_image_ops=False, compile=False,
    )
    params = experiment.with_branches([src])
    assert len(params) == 11 and [p.steps_per_epoch for p in params[1:]] == [s // 10 for s in range(130, 1301, 130)]
    assert all(p.savedir == f"outdir/e00/e01-test/d{j}/" for j, p in enumerate(params))
    assert all(p.benchmark_steps == p.profile_steps == 0 for p in params[1:])  # an anneal is not benchmarked or profiled
    patch_experiment(monkeypatch, tmp_path, src)
    monkeypatch.setattr(experiment, "paramsall", lambda: params)
    Path(src.savedir).mkdir(parents=True)
    submitted = []

    def bsub(par, job, minutes, n_gpus, cmd, **kw):
        submitted.append((par.savedir, cmd, Path(par.anneal_from).is_file()))
        if len(submitted) == 1:
            raise RuntimeError("bsub: queue is closed")  # a failed submit must not kill the trunk

    monkeypatch.setattr(experiment, "bsub", bsub)
    experiment.train(0)

    ckdir = Path(src.savedir) / "checkpoints"
    assert [p.name for p in ckdir.glob("step_*.pt")] == ["step_0001300.pt"]  # rolling: the newest only
    assert [p.name for p in sorted((ckdir / "milestones").glob("*.pt"))] == [f"step_{s:07d}.pt" for s in range(130, 1301, 130)]
    assert [(d, c) for d, c, _ in submitted] == [(f"outdir/e00/e01-test/d{j}/", f"torchrun --standalone --nproc_per_node=1 {{code}}/experiment.py run {j}")
                                                 for j in range(1, 11)]  # one anneal per milestone, in the order saved
    assert all(saved for _, _, saved in submitted)  # each submitted after its checkpoint was written
    trunk = [json.loads(line) for line in (Path(src.savedir) / "metrics.json").read_text().splitlines()]
    assert all(r["lr"] == pytest.approx(src.lr * min(1, (r["idx_step"] + 2) / 1000)) for r in trunk)  # warmup, then constant, no decay
    assert trunk[-1]["lr"] == src.lr

    par = params[9]  # the anneal of milestone 1170
    patch_experiment(monkeypatch, tmp_path, par)
    experiment.train(0)
    start = torch.load(par.anneal_from, weights_only=False)
    metrics = [json.loads(line) for line in (Path(par.savedir) / "metrics.json").read_text().splitlines()]
    lr0 = start["opt"]["param_groups"][0]["lr"]
    assert lr0 == src.lr and metrics[0]["lr"] <= lr0 and metrics[-1]["lr"] == pytest.approx(0.01 * lr0)  # trunk's lr down to 1%
    done = torch.load(sorted((Path(par.savedir) / "checkpoints").glob("step_*.pt"))[-1], weights_only=False)
    assert not (Path(par.savedir) / "checkpoints/milestones").exists()  # an anneal spawns nothing
    assert done["step"] == 117 and done["opt"]["state"][0]["step"] == 1170 + 117  # Adam's step count carried over


@pytest.fixture(autouse=True)
def submit_every_trunk(monkeypatch):
    monkeypatch.setattr(experiment, "RELAUNCH", None)


def test_submitall_relaunches_only_the_listed_trunks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    submitted = []
    monkeypatch.setattr(experiment, "RELAUNCH", [3, 9])
    monkeypatch.setattr(experiment, "submit", lambda n: submitted.append(n))
    experiment.submitall()
    assert submitted == [3, 9]
    p = experiment.paramsall()
    assert [(p[i].batch_size, p[i].max_hours > 2) for i in (3, 9)] == [(32, False), (16, True)]  # 192/64 gets the floor, 192/160 b16
    assert [round(p[i].max_hours, 2) for i in (4, 5, 6, 7, 8)] == [5.63, 5.63, 5.63, 19.89, 19.89]  # the finished and still-running trunks' caps


def test_paramsall_is_the_viewsizes_grid_and_each_trunks_ten_anneals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = experiment.paramsall()
    trunks, anneals = params[:10], params[10:]
    assert [p.anneal_from for p in trunks] == [""] * 10 and all(p.anneal_from for p in anneals) and len(anneals) == 100
    assert sorted((p.patch_size[0], p.global_size[0]) for p in trunks) == sorted((i, g) for g in [64, 96, 128, 160] for i in [96, 128, 160, 192] if i > g)
    assert all(p.local_size == p.global_size and p.batch_size == (16 if p.global_size[0] == 160 else 32) and p.steps_per_epoch == 40_000 for p in trunks)
    for t, branches in zip(trunks, [anneals[k:k + 10] for k in range(0, 100, 10)]):  # each trunk's anneals, one per milestone, the 10th too
        assert [Path(b.anneal_from).name for b in branches] == [f"step_{s:07d}.pt" for s in range(4000, 40_001, 4000)]
        assert all(Path(b.anneal_from).parent == Path(t.savedir) / "checkpoints/milestones" for b in branches)
        assert all((b.global_size, b.patch_size, b.decoder) == (t.global_size, t.patch_size, "linear") for b in branches)
    assert [p.savedir for p in params] == [f"outdir/{experiment.EXPERIMENT}/d{j}/" for j in range(110)]
    experiment.check_experiment()


def test_spawnall_submits_only_the_missed_anneals(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "EXPERIMENT", "e00/e01-test")
    params = experiment.with_branches([experiment.Params(savedir="outdir/e00/e01-test/d0/", steps_per_epoch=1000)])
    submitted = []
    monkeypatch.setattr(experiment, "paramsall", lambda: params)
    monkeypatch.setattr(experiment, "assert_committed", lambda: None)
    monkeypatch.setattr(experiment, "submit_run", lambda j: submitted.append(j))
    for j in (1, 2, 3):  # milestones saved
        Path(params[j].anneal_from).parent.mkdir(parents=True, exist_ok=True)
        Path(params[j].anneal_from).touch()
    Path(params[2].savedir).mkdir(parents=True)  # d2 was submitted
    experiment.spawnall()
    assert submitted == [1, 3]


def test_submitall_leaves_the_anneals_to_their_trunks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "EXPERIMENT", "e00/e01-test")
    params = experiment.with_branches([experiment.Params(savedir="outdir/e00/e01-test/d0/", steps_per_epoch=1000)])
    submitted = []
    monkeypatch.setattr(experiment, "paramsall", lambda: params)
    monkeypatch.setattr(experiment, "submit", lambda n: submitted.append(n))
    experiment.submitall()  # its anneals read their own sweep's trunk: not a dependency on another experiment
    assert submitted == [0]


@pytest.mark.parametrize("max_hours,evals", [(0.0, []), (1.0, ["pca", "probe"])])
def test_run_trains_then_evals_training_runs_only(monkeypatch, max_hours, evals):
    calls = []
    monkeypatch.setattr(experiment, "paramsall", lambda: [experiment.Params(max_hours=max_hours)])
    for fn in ["train", "pca", "probe"]:
        monkeypatch.setattr(experiment, fn, lambda n, fn=fn: calls.append(fn))
    monkeypatch.delenv("RANK", raising=False)
    experiment.run(0)
    assert calls == ["train", *evals]  # evals after train() has torn down the process group, benchmarks skip them


def test_runlsf_refuses_probe_only_runs(tmp_path, monkeypatch):
    # init_from runs are eval-only (pca, probe); training would silently ignore init_from and start from scratch.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "assert_committed", lambda: None)
    monkeypatch.setattr(experiment, "paramsall", lambda: [experiment.Params(savedir="outdir/e00/x/d0", init_from="random")])
    with pytest.raises(AssertionError, match="eval-only"):
        experiment.runlsf(0)
    assert not (tmp_path / "outdir").exists()  # refused before trash() or bsub


def sweep(monkeypatch, name, init_from=""):
    """Point experiment at EXPERIMENT = name with 3 runs (reading init_from + d<i>/ if set), submit() recording them."""
    submitted = []
    monkeypatch.setattr(experiment, "EXPERIMENT", name)
    monkeypatch.setattr(experiment, "paramsall", lambda: [experiment.Params(savedir=f"outdir/{name}/d{i}/",
                                                                            init_from=init_from and f"{init_from}/d{i}/") for i in range(3)])
    monkeypatch.setattr(experiment, "submit", lambda n: submitted.append(n))
    return submitted


def test_submitall_submits_every_run_with_the_experiments_entrypoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    submitted = sweep(monkeypatch, "e00/e01-pca", init_from="outdir/e00/viewsizes-v2")  # a root may read a pre-numbering sweep
    experiment.submitall()
    assert submitted == [0, 1, 2]


@pytest.mark.parametrize("name,init_from,existing,error", [
    ("e00/viewsizes-pca", "", [], "want e<series>"),  # no id
    ("e00/e01-pca", "", ["e01-train"], "id e01 is taken by e00/e01-train"),
    ("e00/e01-02-pca", "outdir/e00/e01-train", [], "builds on e00/e01, which doesn't exist"),
    ("e00/e01-02-pca", "outdir/e00/e03-train", ["e01-train", "e03-train"], "init_from read ['e03-train']"),
    ("e00/e04-pca", "outdir/e00/e01-train", ["e01-train"], "name it e01-<NN>-<slug>"),  # a root can't hide its parent
])
def test_submitall_refuses_an_experiment_misnamed(tmp_path, monkeypatch, name, init_from, existing, error):
    monkeypatch.chdir(tmp_path)
    for s in existing:
        (tmp_path / "outdir/e00" / s).mkdir(parents=True)
    submitted = sweep(monkeypatch, name, init_from)
    with pytest.raises(AssertionError, match=re.escape(error)):
        experiment.submitall()
    assert submitted == []  # refused before any job


def test_submitall_takes_a_child_of_its_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for s in ("e01-train", "e01-01-pca", "e01-02-probe"):  # e01-02-probe: this experiment's own dir, from a first submit
        (tmp_path / "outdir/e00" / s).mkdir(parents=True)
    submitted = sweep(monkeypatch, "e00/e01-02-probe", init_from="outdir/e00/e01-train")
    experiment.submitall()
    assert submitted == [0, 1, 2] and experiment.experiment_id("e01-02-probe") == "e01-02"
