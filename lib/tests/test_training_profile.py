"""Exercise the full training-loop profiler without the remote volume store."""

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import experiment
from lib.losses import LejepaOutput
from lib.encoders import EncoderConfig
from lib.losses import SIGRegConfig
from lib.models import Lejepa, LejepaConfig, MaeConfig, ProjectorConfig
from lib.views import ViewConfig


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
    monkeypatch.setattr(experiment, "roots", lambda: [params])
    monkeypatch.setattr(experiment.lmd, "get", lambda name: SimpleNamespace(to_miao=lambda **kw: None))
    monkeypatch.setattr(experiment, "MiaoConfig", SimpleNamespace)
    monkeypatch.setattr(experiment, "VolumeDataset", SyntheticDataset)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Provenance independent of git state.
    monkeypatch.setattr(experiment, "code_provenance", lambda: {
        "commit_id": "test-commit", "subject": "test", "dirty": False, "diff_hash": "test-diff",
    })

    def small_model_config(par):  # par's objective, tiny
        encoder = EncoderConfig(depth=1, embed_dim=16, num_heads=2, patch_size=(4, 4, 4))
        if par.objective == "mae":
            return MaeConfig(encoder=encoder, decoder_dim=8, decoder_heads=2, decoder_depth=1)
        return LejepaConfig(
            encoder=encoder, objective=par.objective, view=ViewConfig(n_global=2, n_local=1) if par.objective == "views" else ViewConfig(),
            projector=ProjectorConfig(dim=8, hidden=16), sigreg=SIGRegConfig(num_slices=4, knots=3),
        )

    monkeypatch.setattr(experiment, "model_config", small_model_config)
    for fn in ("pca", "probe"):  # a trunk's milestone evals (eval_milestone) need the real volumes
        monkeypatch.setattr(experiment, fn, lambda job: None)


@pytest.mark.parametrize(
    "profile_steps,n_steps,recorded_steps",
    [(2, 7, 2), (0, 4, 0), (3, 4, 0), (3, 5, 1)],
)
def test_training_profile(tmp_path, monkeypatch, profile_steps, n_steps, recorded_steps):
    params = experiment.Train(
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
    experiment.train(params)

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
    params = experiment.Train(savedir=str(tmp_path), objective=objective, patch_size=(8, 8, 8), batch_size=2, steps_per_epoch=3,
                              warmup_steps=1, benchmark_steps=1, profile_steps=0, n_workers=1, defer_image_ops=False)
    patch_experiment(monkeypatch, tmp_path, params)
    experiment.train(params)
    metrics = [json.loads(line) for line in (tmp_path / "metrics.json").read_text().splitlines()]
    assert metrics[-1]["idx_step"] == 2 and all(torch.isfinite(torch.tensor(row["loss"])) for row in metrics)
    if objective == "views":  # LeJEPA's two terms, logged apart
        assert all(row["loss"] == pytest.approx(row["inv"] + LejepaConfig().lamb * row["sigreg"], rel=1e-5) for row in metrics)
    else:  # an MAE's reconstruction error alone
        assert not {"inv", "sigreg"} & set(metrics[-1])
    model, step = experiment.load_checkpoint(experiment.own_eval(params))  # strict: the saved weights fit the objective's model
    assert step == 3 and type(model).__name__ == {"views": "Lejepa", "mae": "Mae"}[objective]


def test_mae_decoder_sets_the_mae_models_decoder():
    c = experiment.model_config(experiment.Train(objective="mae", mae_decoder=(512, 8)))
    assert isinstance(c, MaeConfig) and (c.decoder_dim, c.decoder_depth, c.decoder_heads) == (512, 8, 16)


def test_nonfinite_grad_step_is_skipped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "EXPERIMENT", "e00/e01-test")  # replay_bad_batch replays only this experiment's dumps
    run = tmp_path / "outdir/e00/e01-test/d0"
    run.mkdir(parents=True)
    params = experiment.Train(
        savedir="outdir/e00/e01-test/d0/", patch_size=(8, 8, 8), batch_size=2, steps_per_epoch=6,
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
    experiment.train(params)

    metrics = [json.loads(line) for line in (run / "metrics.json").read_text().splitlines()]
    assert metrics[-1]["idx_step"] == 5 and metrics[-1]["skipped"] == 1  # training continued past the bad step
    dump = torch.load(run / "bad_batch_0000003.pt", weights_only=False)
    assert dump["step"] == 3 and dump["x_uint8"].dtype == torch.uint8 and dump["rng_cpu"] is not None
    ckpt = torch.load(sorted((run / "checkpoints").glob("step_*.pt"))[-1], weights_only=False)
    assert all(torch.isfinite(v).all() for v in ckpt["model"].values() if v.is_floating_point())

    # Replay: same weights, input and RNG give the same views and SIGReg slices, so the same loss
    # (the NaN came from the test's poisoned model, which the replay doesn't use).
    replay = experiment.replay_bad_batch(str(run / "bad_batch_0000003.pt"))
    runs = [json.loads(l) for l in (run / "runs.json").read_text().splitlines()]
    assert [r["fn"] for r in runs] == ["run", "replay_bad_batch"] and runs[0]["commit_id"] == "test-commit"
    assert runs[0]["params"]["savedir"] == params.savedir  # every run dir describes itself, even if it crashes
    assert replay["eager"][0] == pytest.approx(losses[3], rel=1e-5)
    assert replay["as trained"][0] == pytest.approx(losses[3], rel=1e-5)  # compile=False: both paths are eager


def test_a_trunk_evaluates_each_milestone_and_an_anneal_continues_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "EXPERIMENT", "e00/e01-test")
    src = experiment.Train(
        savedir="outdir/e00/e01-test/d0/", patch_size=(8, 8, 8), batch_size=2, steps_per_epoch=1300, max_hours=1.0,  # past LR_WARMUP
        warmup_steps=1, benchmark_steps=2, profile_steps=0, n_workers=1, defer_image_ops=False, compile=False,
    )
    patch_experiment(monkeypatch, tmp_path, src)
    experiment.dump(src)  # as submit() does: the trunk was submitted
    monkeypatch.setattr(experiment, "bsub", lambda *a, **kw: pytest.fail("submitted"))  # a trunk spawns no jobs
    evals = []
    for fn in ("pca", "probe"):  # each in the trunk, once its milestone is on disk
        monkeypatch.setattr(experiment, fn, lambda job, fn=fn: evals.append((fn, job, Path(job.init_from).is_file())))
    experiment.train(src)

    steps = range(130, 1301, 130)
    assert evals == [(fn, experiment.eval_params(src, s), True) for s in steps for fn in ("pca", "probe")]  # the last once only
    assert all(experiment.load(f"{experiment.branch_dir(src, s)}eval.json", experiment.Eval) == experiment.eval_params(src, s) for s in steps)  # for `lsf probe`
    ckdir = Path(src.savedir) / "checkpoints"
    assert [p.name for p in ckdir.glob("step_*.pt")] == ["step_0001300.pt"]  # rolling: the newest only
    assert [p.name for p in sorted((ckdir / "milestones").glob("*.pt"))] == [f"step_{s:07d}.pt" for s in steps]  # the evals were stubs: none released
    trunk = [json.loads(line) for line in (Path(src.savedir) / "metrics.json").read_text().splitlines()]
    assert all(r["lr"] == pytest.approx(src.lr * min(1, (r["idx_step"] + 2) / 1000)) for r in trunk)  # warmup, then constant, no decay
    assert trunk[-1]["lr"] == src.lr

    par = replace(src, savedir=experiment.branch_dir(src, 1170), steps_per_epoch=117, anneal_from=str(experiment.milestone(src, 1170)))
    experiment.train(str(experiment.dump(par)))  # as its LSF job would: from its spec file
    start = torch.load(par.anneal_from, weights_only=False)
    metrics = [json.loads(line) for line in (Path(par.savedir) / "metrics.json").read_text().splitlines()]
    lr0 = start["opt"]["param_groups"][0]["lr"]
    assert lr0 == src.lr and metrics[0]["lr"] <= lr0 and metrics[-1]["lr"] == pytest.approx(0.01 * lr0)  # trunk's lr down to 1%
    done = torch.load(sorted((Path(par.savedir) / "checkpoints").glob("step_*.pt"))[-1], weights_only=False)
    assert not (Path(par.savedir) / "checkpoints/milestones").exists() and len(evals) == 20  # an anneal has no milestones
    assert done["step"] == 117 and done["opt"]["state"][0]["step"] == 1170 + 117  # Adam's step count carried over


def test_a_milestone_eval_neither_stops_nor_steers_training(tmp_path, monkeypatch):
    params = experiment.Train(
        patch_size=(8, 8, 8), batch_size=2, steps_per_epoch=21, milestone_every=10,
        warmup_steps=1, benchmark_steps=2, profile_steps=0, n_workers=1, defer_image_ops=False, compile=False,
    )
    patch_experiment(monkeypatch, tmp_path, params)
    losses, calls = {}, []
    for failing in (False, True):
        def pca(job):  # draws from the RNG training uses, then fails
            calls.append(failing)
            torch.rand(100)
            assert not failing, "eval failed"
        monkeypatch.setattr(experiment, "pca", pca)
        (tmp_path / str(failing)).mkdir()
        torch.manual_seed(0)
        experiment.train(replace(params, savedir=f"{tmp_path}/{failing}/"))
        losses[failing] = [json.loads(line)["loss"] for line in (tmp_path / f"{failing}/metrics.json").read_text().splitlines()]
    assert calls == [False, False, True, True]  # milestones 10 and 20
    assert len(losses[True]) == 3 and losses[True] == losses[False]  # trained to the end, drawing the same views


def test_a_milestone_is_deleted_once_its_eval_has_pca_and_probe_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "EXPERIMENT", "e00/e01-test")
    foreign = experiment.Eval(savedir="outdir/e00/e01-test/d2/", train=experiment.Train(),
                              init_from="outdir/e00/other/d0/checkpoints/step_0000100.pt")  # not our milestone
    trunk = experiment.Train(savedir="outdir/e00/e01-test/d0/", steps_per_epoch=1000)
    evaluated = experiment.eval_params(trunk, 100)
    milestone = experiment.milestone(trunk, 100)
    assert experiment.milestone_of(evaluated) == str(milestone)
    for path in (milestone, Path(foreign.init_from), Path(trunk.savedir) / "checkpoints/step_0001000.pt"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for p in (evaluated, foreign):
        Path(p.savedir).mkdir(parents=True, exist_ok=True)
        (Path(p.savedir) / "pca.json").write_text("{}\n")
    experiment.release_milestone(evaluated)
    assert milestone.exists()  # its probe is still to run
    for p in (evaluated, foreign):
        (Path(p.savedir) / "probe.json").write_text("{}\n")
    experiment.release_milestone(foreign)
    experiment.release_milestone(evaluated)
    experiment.release_milestone(evaluated)  # again: nothing left, nothing to fail
    assert not milestone.exists()
    assert Path(foreign.init_from).exists() and (Path(trunk.savedir) / "checkpoints/step_0001000.pt").exists()  # only our milestones go


def test_pca_releases_its_milestone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "EXPERIMENT", "e00/e01-test")
    par = experiment.eval_params(experiment.Train(savedir="outdir/e00/e01-test/d0/", steps_per_epoch=1000), 100)
    milestone = Path(par.init_from)
    milestone.parent.mkdir(parents=True)
    milestone.touch()
    Path(par.savedir).mkdir(parents=True)
    (Path(par.savedir) / "probe.json").write_text("{}\n")  # the probe finished first
    monkeypatch.setattr(experiment, "load_checkpoint", lambda p: (SimpleNamespace(encoder=None), 100))
    monkeypatch.setattr(experiment, "record", lambda *a: None)
    monkeypatch.setattr(experiment, "pca_maps", lambda encoder, par, savedir, step: (savedir / "pca.json").write_text("{}\n"))
    experiment.pca(par)
    assert not milestone.exists()


def test_pca_sizes_maps_one_crop_at_each_test_tile(tmp_path, monkeypatch):
    """Every tile side runs over the same crop, 2 x 2 tiles of the largest in y x and one deep; one png and json row per side."""
    from PIL import Image
    from lib.util import dump, load
    torch.manual_seed(0)
    model = Lejepa(LejepaConfig(encoder=EncoderConfig(depth=1, embed_dim=16, num_heads=2, patch_size=(4, 4, 4)), projector=ProjectorConfig(dim=8, hidden=16))).eval()
    monkeypatch.setattr(experiment, "TEST_TILES", (8, 16))
    monkeypatch.setattr(experiment, "eb_val_crop", lambda shape: torch.rand(1, *shape))
    monkeypatch.setattr(experiment, "load_checkpoint", lambda e: (model, 7))
    monkeypatch.setattr(experiment, "record", lambda *a: None)
    e = experiment.Eval(savedir=f"{tmp_path}/", train=experiment.Train(), init_from="outdir/e00/x/d0", jobs=("pca_sizes",))
    assert load(str(dump(e)), experiment.Eval) == e  # the spec round-trips, jobs and all
    experiment.pca_sizes(e)
    rows = [json.loads(line) for line in (tmp_path / "pca_sizes.json").read_text().splitlines()]
    assert [(r["tile"], r["tiles"], r["step"]) for r in rows] == [(8, 2 * 4 * 4, 7), (16, 4, 7)]  # a 16 x 32 x 32 crop
    assert all(0 <= r["between_tile_variance"] <= 1 and r["effective_rank"] > 1 for r in rows)
    for side in (8, 16):  # EM | PCA | tile-centered PCA, at voxel size
        assert Image.open(tmp_path / f"pca_tile_{side:03d}.png").size == (3 * 32 + 2 * 8, 32)


def test_start_submits_an_evals_own_jobs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    submitted = []
    monkeypatch.setattr(experiment, "submit", lambda fn, p: submitted.append(fn))
    monkeypatch.setattr(experiment, "trash", lambda savedir: None)
    experiment.start(experiment.Eval(savedir="outdir/e00/x/d0/", train=experiment.Train()))
    experiment.start(experiment.Eval(savedir="outdir/e00/x/d1/", train=experiment.Train(), jobs=("pca_sizes",)))
    assert submitted == ["pca", "probe", "pca_sizes"]


def test_load_checkpoint_reads_a_milestone_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    par = experiment.Train(patch_size=(8, 8, 8), views="displace", global_size=(8, 8, 8))
    monkeypatch.setattr(experiment, "lejepa_config", lambda par: LejepaConfig(
        encoder=EncoderConfig(depth=1, embed_dim=16, num_heads=2, patch_size=(4, 4, 4)), projector=ProjectorConfig(dim=8, hidden=16)))
    model = Lejepa(experiment.lejepa_config(par))
    path = tmp_path / "step_0000130.pt"
    torch.save({"step": 130, "model": model.state_dict()}, path)
    loaded, step = experiment.load_checkpoint(experiment.Eval(savedir=str(tmp_path), train=par, init_from=str(path)))
    assert step == 130 and torch.equal(loaded.encoder.blocks[0].attn.qkv.weight.cpu(), model.encoder.blocks[0].attn.qkv.weight)


def test_submitall_starts_each_root_not_submitted_before(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    started = []
    roots = [experiment.Train(savedir=f"outdir/e00/e01-test/d{i}/") for i in range(3)]
    monkeypatch.setattr(experiment, "EXPERIMENT", "e00/e01-test")
    monkeypatch.setattr(experiment, "roots", lambda: roots)
    monkeypatch.setattr(experiment, "assert_committed", lambda: None)
    monkeypatch.setattr(experiment, "start", started.append)
    experiment.dump(roots[1])  # submitted before
    experiment.submitall()
    assert started == [roots[0], roots[2]]


def test_lejepa_config_takes_the_mask_block_and_lamb_from_params():
    config = experiment.lejepa_config(experiment.Train(mask_block=0.45, lamb=0.02))
    assert config.mask.block == 0.45 and config.lamb == 0.02


def test_roots_are_the_trunks_and_their_milestone_evals_nest_in_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    roots = experiment.roots()
    assert [t.savedir for t in roots] == [f"outdir/{experiment.EXPERIMENT}/d{j}/" for j in range(len(roots))]
    for p in roots:  # the sweeps it builds on, which check_experiment looks for in outdir/
        if experiment.source(p).startswith("outdir/"):
            Path(*Path(experiment.source(p)).parts[:3]).mkdir(parents=True, exist_ok=True)
    experiment.check_experiment()
    for t in [p for p in roots if isinstance(p, experiment.Train)]:  # an experiment may also be evals of others' weights
        for s in experiment.milestone_steps(t):
            e = experiment.eval_params(t, s)
            assert e.savedir == f"{t.savedir}step_{s:07d}/" and experiment.milestone_of(e) == f"{t.savedir}checkpoints/milestones/step_{s:07d}.pt"
            assert e.train == t  # the trunk's architecture


@pytest.mark.parametrize("max_hours", [0.0, 1.0])
def test_run_trains_then_evals(tmp_path, monkeypatch, max_hours):
    monkeypatch.chdir(tmp_path)
    calls = []
    for fn in ["train", "pca", "probe"]:
        monkeypatch.setattr(experiment, fn, lambda job, fn=fn: calls.append(fn))
    monkeypatch.delenv("RANK", raising=False)
    experiment.run(experiment.Train(max_hours=max_hours))
    assert calls == ["train", "pca", "probe"]  # evals after train() has torn down the process group; a benchmark is its own entrypoint


def test_a_run_probed_on_several_sets_probes_each_in_its_own_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    probed = []
    monkeypatch.setattr(experiment, "train", lambda job: None)
    monkeypatch.setattr(experiment, "pca", lambda job: probed.append(("pca", job.savedir)))
    monkeypatch.setattr(experiment, "probe", lambda job: probed.append((job.probe_on, job.savedir)))
    monkeypatch.delenv("RANK", raising=False)
    experiment.run(experiment.Train(savedir="outdir/e00/x/d0/", probe_sets=("fish2", "kasthuri15")))
    assert probed == [("pca", "outdir/e00/x/d0/"), ("fish2", "outdir/e00/x/d0/probe_fish2/"), ("kasthuri15", "outdir/e00/x/d0/probe_kasthuri15/")]
    assert all(Path(d).is_dir() for _, d in probed)


def test_train_volumes_weigh_a_joint_sets_families_equally():
    from lib.data import JOINT_DATA, TRAIN_BOXES, train_volumes
    vols = train_volumes("lmd4")
    assert sum(w for _, _, w in vols) == pytest.approx(1)
    for f in JOINT_DATA["lmd4"]:  # each family a quarter, split equally among its volumes
        assert sum(w for n, _, w in vols if n in TRAIN_BOXES[f]) == pytest.approx(1 / 4)
    assert [w for _, _, w in train_volumes("fish2")] == [0.25] * 4 and train_volumes("hemibrain_eb")[0][2] == 1


def test_benchmark_times_and_profiles_the_training_step_only(tmp_path, monkeypatch):
    params = experiment.Train(
        savedir=str(tmp_path), patch_size=(8, 8, 8), batch_size=2, steps_per_epoch=1000, max_hours=1.0,
        warmup_steps=1, benchmark_steps=2, profile_steps=2, n_workers=1, defer_image_ops=False,
    )
    patch_experiment(monkeypatch, tmp_path, params)
    experiment.benchmark(params)
    performance = json.loads((tmp_path / "performance.json").read_text())
    assert performance["steps"] == 2 and performance["samples_per_second"] > 0
    assert (tmp_path / "profile.json").is_file() and (tmp_path / "trace_summary.json").is_file()
    assert not (tmp_path / "metrics.json").exists() and not (tmp_path / "checkpoints").exists()  # not a training run
    runs = [json.loads(l) for l in (tmp_path / "runs.json").read_text().splitlines()]
    assert [r["fn"] for r in runs] == ["benchmark"] and runs[0]["params"]["steps_per_epoch"] == 1 + 2 + 1 + 2  # its windows only


def test_milestones_are_every_n_steps_or_every_tenth():
    assert experiment.milestone_steps(experiment.Train(steps_per_epoch=1300)) == list(range(130, 1301, 130))
    assert experiment.milestone_steps(experiment.Train(steps_per_epoch=100, milestone_every=30)) == [30, 60, 90]  # the cap lists them
    doubling = experiment.Train(steps_per_epoch=100, milestone_every=10, milestone_doubling=True)
    assert experiment.milestone_steps(doubling) == [10, 20, 40, 80]
    assert experiment.milestone_steps(replace(doubling, steps_per_epoch=80)) == [10, 20, 40, 80]  # the cap itself, if on the doubling


@pytest.mark.parametrize("final_step,evals", [(1000, False), (1234, True)])  # a trunk stopped on a milestone, or off one (max_hours)
def test_a_trunk_evaluates_its_end_only_off_a_milestone(tmp_path, monkeypatch, final_step, evals):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "EXPERIMENT", "e00/e01-test")
    trunk = experiment.Train(savedir="outdir/e00/e01-test/d0/", steps_per_epoch=2000, milestone_every=1000, max_hours=1.0)
    monkeypatch.delenv("RANK", raising=False)
    calls = []
    for fn in ["train", "pca", "probe"]:
        monkeypatch.setattr(experiment, fn, lambda job, fn=fn: calls.append(fn))
    (tmp_path / "outdir/e00/e01-test/d0/checkpoints").mkdir(parents=True)
    (tmp_path / f"outdir/e00/e01-test/d0/checkpoints/step_{final_step:07d}.pt").touch()
    experiment.run(trunk)
    assert calls == ["train", *(["pca", "probe"] if evals else [])]  # on one, train() evaluated it
    anneal = replace(trunk, savedir="outdir/e00/e01-test/d0/step_0001000/", anneal_from=str(experiment.milestone(trunk, 1000)))
    calls.clear()
    experiment.run(anneal)  # an anneal has no milestones: it evaluates its end
    assert calls == ["train", "pca", "probe"]


def test_submit_refuses_a_job_its_spec_cant_run(tmp_path, monkeypatch):
    # An eval trains nothing: run would ignore its weights and train from scratch.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "bsub", lambda *a, **kw: pytest.fail("submitted"))
    with pytest.raises(AssertionError, match="no LSF job 'run' for a Eval"):
        experiment.submit("run", experiment.Eval(savedir="outdir/e00/x/d0/", train=experiment.Train(), init_from="random"))
    with pytest.raises(AssertionError, match="no LSF job 'probe' for a Train"):
        experiment.submit("probe", experiment.Train(savedir="outdir/e00/x/d0/"))
    assert not (tmp_path / "outdir").exists()  # refused before writing its spec


def test_submit_writes_the_spec_its_job_loads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    submitted = []
    monkeypatch.setattr(experiment, "bsub", lambda savedir, job, minutes, n_gpus, queue, cmd, **kw: submitted.append((job, n_gpus, queue, cmd)))
    trunk = experiment.Train(savedir="outdir/e00/x/d0/", queue="gpu_b300", global_size=(64, 64, 64), lamb=0.5, n_gpus=2, max_hours=1.0, n_workers=11)
    e = experiment.Eval(savedir="outdir/e00/x/d1/", train=trunk, init_from="random", probe_on="fish2")
    f = experiment.Finetune(savedir="outdir/e00/x/d2/", train=trunk, init_from="outdir/e00/y/d0/", fit_side=256, freeze_encoder=False)
    experiment.submit("run", trunk)
    experiment.submit("probe", e)
    experiment.submit("finetune", f)
    assert submitted == [("run", 2, "gpu_b300", "torchrun --standalone --nproc_per_node=2 {code}/experiment.py run outdir/e00/x/d0/train.json"),
                         ("probe", 1, "gpu_b300", "python {code}/experiment.py probe outdir/e00/x/d1/eval.json"),  # the trunk's queue
                         ("finetune", 1, "gpu_b300", "python {code}/experiment.py finetune outdir/e00/x/d2/finetune.json")]
    assert experiment.load("outdir/e00/x/d0/train.json", experiment.Train) == trunk  # what the job runs: the same spec
    assert experiment.load("outdir/e00/x/d1/eval.json", experiment.Eval) == e
    assert experiment.load("outdir/e00/x/d2/finetune.json", experiment.Finetune) == f
    with pytest.raises(AssertionError, match="no LSF job 'finetune' for a Eval"):
        experiment.submit("finetune", e)
    spec = json.loads(Path("outdir/e00/x/d1/eval.json").read_text())
    Path("bad.json").write_text(json.dumps({**spec, "score": "maybe"}))
    with pytest.raises(Exception, match="score"):  # validated: a typo fails at load, before any GPU work
        experiment.load("bad.json", experiment.Eval)
    Path("extra.json").write_text(json.dumps({**spec, "steps": 3}))
    with pytest.raises(AssertionError, match="doesn't round-trip"):  # so does a field the spec doesn't have
        experiment.load("extra.json", experiment.Eval)


def test_a_finetune_of_a_trunk_here_waits_in_lsf_for_its_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "EXPERIMENT", "e00/e01-test")
    held = []
    monkeypatch.setattr(experiment, "bsub", lambda savedir, job, minutes, n_gpus, queue, cmd, after="", **kw: held.append(after))
    trunk = experiment.Train(savedir="outdir/e00/e01-test/d0/")
    finetune = lambda init: experiment.Finetune(savedir="outdir/e00/e01-test/d1/", train=trunk, init_from=init)
    for init in [trunk.savedir, "random", "outdir/e00/e00-other/d0/"]:
        experiment.submit("finetune", finetune(init))
    Path(trunk.savedir).mkdir(parents=True)
    Path(trunk.savedir, "probe.json").write_text("{}")  # the trunk is done: nothing to wait for (LSF may have forgotten its job)
    experiment.submit("finetune", finetune(trunk.savedir))
    assert held == ["e00-e01-test-d0-run", "", "", ""]


def test_retryall_restarts_only_finetunes_that_ended_unfinished(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    footer = "Successfully completed.\n\nResource usage summary:\n"
    trunk = experiment.Train(savedir="outdir/e00/e01-test/d0/")
    runs = [trunk] + [experiment.Finetune(savedir=f"outdir/e00/e01-test/d{i}/", train=trunk) for i in range(1, 5)]
    for i, (log, png) in enumerate([(footer, True), (footer, False), ("still running", False), (None, False)], 1):  # done, failed, running, queued
        d = Path(runs[i].savedir)
        d.mkdir(parents=True)
        if log:
            (d / "job_finetune_1.log").write_text(log)
        if png:
            (d / "finetune.png").write_bytes(b"")
    started = []
    monkeypatch.setattr(experiment, "roots", lambda: runs)
    monkeypatch.setattr(experiment, "check_experiment", lambda: None)
    monkeypatch.setattr(experiment, "assert_committed", lambda: None)
    monkeypatch.setattr(experiment, "start", lambda p: started.append(p.savedir))
    experiment.retryall()
    assert started == ["outdir/e00/e01-test/d2/"]


def sweep(monkeypatch, name, init_from=""):
    """Point experiment at EXPERIMENT = name with 3 roots (evals of init_from + d<i>/ if set, else trainings), start() recording
    their indices, and no LSF."""
    submitted = []
    monkeypatch.setattr(experiment, "EXPERIMENT", name)
    run = lambda i: (experiment.Eval(savedir=f"outdir/{name}/d{i}/", train=experiment.Train(), init_from=f"{init_from}/d{i}/") if init_from
                     else experiment.Train(savedir=f"outdir/{name}/d{i}/"))
    monkeypatch.setattr(experiment, "roots", lambda: [run(i) for i in range(3)])
    monkeypatch.setattr(experiment, "assert_committed", lambda: None)
    monkeypatch.setattr(experiment, "start", lambda p: submitted.append(int(Path(p.savedir).name[1:])))
    monkeypatch.setattr(experiment, "bsub", lambda *a, **kw: None)
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
