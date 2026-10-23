"""Tests for experiment.py: the entrypoints' wiring (run, evals, milestone release), the LSF submission and the sweep's checks."""

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import experiment
from lib import evals, specs


@pytest.fixture(autouse=True)
def provenance(monkeypatch):  # run records independent of git state
    monkeypatch.setattr(specs, "code_provenance", lambda: {"commit_id": "test-commit", "subject": "test", "dirty": False, "diff_hash": "test-diff"})


def test_a_milestone_is_deleted_once_its_eval_has_pca_and_probe_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "EXPERIMENT", "e00/e01-test")
    foreign = experiment.Eval(savedir="outdir/e00/e01-test/d2/", train=experiment.Train(),
                              init_from="outdir/e00/other/d0/checkpoints/step_0000100.pt")  # not our milestone
    trunk = experiment.Train(savedir="outdir/e00/e01-test/d0/", steps_per_epoch=1000)
    evaluated = experiment.eval_params(trunk, 100)
    milestone = specs.milestone(trunk, 100)
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
    monkeypatch.setattr(evals, "pca_maps", lambda encoder, par, savedir, step, data: (savedir / "pca.json").write_text("{}\n"))
    experiment.pca(par)
    assert not milestone.exists()


def test_start_submits_an_evals_own_jobs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    submitted = []
    monkeypatch.setattr(experiment, "submit", lambda fn, p: submitted.append(fn))
    monkeypatch.setattr(experiment, "trash", lambda savedir: None)
    experiment.start(experiment.Eval(savedir="outdir/e00/x/d0/", train=experiment.Train()))
    experiment.start(experiment.Eval(savedir="outdir/e00/x/d1/", train=experiment.Train(), jobs=("pca_sizes",)))
    assert submitted == ["pca", "probe", "pca_sizes"]


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
    monkeypatch.setattr(experiment.training, "train", lambda par, on_milestone: calls.append("train"))
    for fn in ["pca", "probe"]:
        monkeypatch.setattr(experiment, fn, lambda job, fn=fn: calls.append(fn))
    monkeypatch.delenv("RANK", raising=False)
    experiment.run(experiment.Train(max_hours=max_hours))
    assert calls == ["train", "pca", "probe"]  # evals after train() has torn down the process group; a benchmark is its own entrypoint


def test_a_run_probed_on_several_sets_probes_each_in_its_own_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    probed = []
    monkeypatch.setattr(experiment.training, "train", lambda par, on_milestone: None)
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


@pytest.mark.parametrize("final_step,evals", [(1000, False), (1234, True)])  # a trunk stopped on a milestone, or off one (max_hours)
def test_a_trunk_evaluates_its_end_only_off_a_milestone(tmp_path, monkeypatch, final_step, evals):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "EXPERIMENT", "e00/e01-test")
    trunk = experiment.Train(savedir="outdir/e00/e01-test/d0/", steps_per_epoch=2000, milestone_every=1000, max_hours=1.0)
    monkeypatch.delenv("RANK", raising=False)
    calls = []
    monkeypatch.setattr(experiment.training, "train", lambda par, on_milestone: calls.append("train"))
    for fn in ["pca", "probe"]:
        monkeypatch.setattr(experiment, fn, lambda job, fn=fn: calls.append(fn))
    (tmp_path / "outdir/e00/e01-test/d0/checkpoints").mkdir(parents=True)
    (tmp_path / f"outdir/e00/e01-test/d0/checkpoints/step_{final_step:07d}.pt").touch()
    experiment.run(trunk)
    assert calls == ["train", *(["pca", "probe"] if evals else [])]  # on one, train() evaluated it
    anneal = replace(trunk, savedir="outdir/e00/e01-test/d0/step_0001000/", anneal_from=str(specs.milestone(trunk, 1000)))
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


def test_a_milestone_eval_runs_pca_then_probe_without_stopping_or_steering_training(tmp_path, monkeypatch):
    trunk = experiment.Train(savedir=f"{tmp_path}/d0/", steps_per_epoch=100)
    e = experiment.eval_params(trunk, 10)
    calls = []
    monkeypatch.setattr(experiment, "pca", lambda job: calls.append(("pca", job, torch.rand(100))))  # draws from the RNG training uses
    monkeypatch.setattr(experiment, "probe", lambda job: calls.append(("probe", job)))
    torch.manual_seed(0)
    experiment.eval_milestone(trunk, 10)
    after = torch.rand(3)
    torch.manual_seed(0)
    assert torch.equal(after, torch.rand(3))  # the RNG state is as if no eval had run
    assert [c[:2] for c in calls] == [("pca", e), ("probe", e)]
    assert experiment.load(f"{e.savedir}eval.json", experiment.Eval) == e  # written first, for `lsf probe`

    def failing(job):
        raise RuntimeError("eval failed")

    monkeypatch.setattr(experiment, "pca", failing)
    calls.clear()
    experiment.eval_milestone(trunk, 20)  # swallowed: the trunk trains on, and the probe is not tried
    assert calls == []
