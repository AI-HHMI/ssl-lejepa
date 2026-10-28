"""Tests for lib.jobs: the tasks' wiring (run, evals), the LSF submission, the commands on an Experiment, and the experiment file."""

import json
import math
import re
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import experiment
from lib import evals, jobs, specs
from lib.jobs import Experiment


@pytest.fixture(autouse=True)
def provenance(monkeypatch):  # run records independent of git state
    monkeypatch.setattr(specs, "code_provenance", lambda: {"commit_id": "test-commit", "subject": "test", "dirty": False, "diff_hash": "test-diff"})


INSIDE_RUN = [fn for fn, t in jobs.TASKS.items() if t.inside == "run"]  # what a run calls in its own process, in pipeline order


def test_pca_keeps_its_milestone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    par = jobs.eval_params(jobs.Train(savedir="outdir/e00/e01-test/mae/", total_steps=1000), 100)
    milestone = Path(par.init_from)
    milestone.parent.mkdir(parents=True)
    milestone.touch()
    Path(par.savedir).mkdir(parents=True)
    (Path(par.savedir) / "probe.json").write_text("{}\n")
    monkeypatch.setattr(jobs, "load_checkpoint", lambda p: (SimpleNamespace(encoder=None), 100))
    monkeypatch.setattr(jobs, "record", lambda *a: None)
    monkeypatch.setattr(evals, "pca_maps", lambda encoder, par, savedir, step, data: (savedir / "pca.json").write_text("{}\n"))
    jobs.pca(par)
    assert milestone.exists()  # kept: a later probe, or a branch, can reload it


def test_start_submits_an_evals_own_jobs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    submitted = []
    monkeypatch.setattr(jobs, "submit", lambda fn, p, after="": submitted.append(fn))
    monkeypatch.setattr(jobs, "trash", lambda savedir: None)
    jobs.start(jobs.Eval(savedir="outdir/e00/x/a/", train=jobs.Train()))
    jobs.start(jobs.Eval(savedir="outdir/e00/x/b/", train=jobs.Train(), jobs=("pca_sizes",)))
    assert submitted == ["pca", "probe", "pca_sizes"]


def test_every_task_is_an_entrypoint_and_its_where_is_known():
    assert set(jobs.TASKS) == set(jobs.ENTRYPOINTS), set(jobs.TASKS) ^ set(jobs.ENTRYPOINTS)
    assert {t.needs for t in jobs.TASKS.values()} <= {"gpus", "gpu", "cpu"}
    assert {t.inside for t in jobs.TASKS.values()} <= {"", "run"}  # no task is called inside another task's process but run's


def test_every_data_use_names_a_part_of_a_dataset():
    import lib.data
    for fn, task in jobs.TASKS.items():
        for use in task.data:
            assert use.role in lib.data.Dataset._fields, (fn, use)
            assert not use.when or all(use.when in spec.__dataclass_fields__ for spec in (jobs.Eval, jobs.Finetune)), (fn, use)
        assert len({u.role for u in task.data}) == len(task.data), (fn, "a role twice")


def test_submit_refuses_a_task_for_a_spec_it_does_not_take(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(AssertionError, match="no LSF job 'probe' for a Train"):
        jobs.submit("probe", jobs.Train(savedir="outdir/e00/x/a/"))


def test_the_experiment_file_names_runs_that_load_as_a_job_loads_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    roots = experiment.EXPERIMENT.roots()
    for name, r in roots.items():  # a job validates its json (pydantic's Literals), which building the dataclass doesn't
        assert r.savedir == f"outdir/{experiment.EXPERIMENT.id}/{name}/"
        assert jobs.parse(type(r), asdict(r)) == r, name
        specs.model_config(r)  # its model builds on its encoder's patch
    jobs.check_experiment(experiment.EXPERIMENT)
    for t in roots.values():
        for s in jobs.milestone_steps(t):
            e = jobs.eval_params(t, s)
            assert e.savedir == f"{t.savedir}step_{s:07d}/" and e.init_from == f"{t.savedir}checkpoints/milestones/step_{s:07d}.pt"
            assert e.train == t and e.data == t.eval_on  # the trunk's architecture, measured where it says


def test_submitall_starts_each_run_not_submitted_before(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    started = []
    exp = Experiment("e00/e01-test", {name: jobs.Train() for name in "abc"})
    monkeypatch.setattr(jobs, "assert_committed", lambda: None)
    monkeypatch.setattr(jobs, "start", started.append)
    jobs.dump(exp.roots()["b"])  # submitted before
    jobs.submitall(exp)
    assert [r.savedir for r in started] == ["outdir/e00/e01-test/a/", "outdir/e00/e01-test/c/"]


@pytest.mark.parametrize("max_hours", [0.0, 1.0])
def test_run_trains_then_evals(tmp_path, monkeypatch, max_hours):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(jobs.training, "train", lambda par, on_milestone: calls.append("train"))
    for fn in INSIDE_RUN:
        monkeypatch.setitem(jobs.ENTRYPOINTS, fn, lambda job, fn=fn: calls.append(fn))
    monkeypatch.delenv("RANK", raising=False)
    jobs.run(jobs.Train(max_hours=max_hours))
    assert calls == ["train", *INSIDE_RUN]  # evals after train() has torn down the process group; a benchmark is its own entrypoint


@pytest.mark.parametrize("final_step,evals", [(1000, False), (1234, True)])  # a trunk stopped on a milestone, or off one (max_hours)
def test_a_trunk_evaluates_its_end_only_off_a_milestone(tmp_path, monkeypatch, final_step, evals):
    monkeypatch.chdir(tmp_path)
    trunk = jobs.Train(savedir="outdir/e00/e01-test/a/", total_steps=2000, milestone_every=1000, max_hours=1.0)
    monkeypatch.delenv("RANK", raising=False)
    calls = []
    monkeypatch.setattr(jobs.training, "train", lambda par, on_milestone: calls.append("train"))
    for fn in INSIDE_RUN:
        monkeypatch.setitem(jobs.ENTRYPOINTS, fn, lambda job, fn=fn: calls.append(fn))
    (tmp_path / "outdir/e00/e01-test/a/checkpoints").mkdir(parents=True)
    (tmp_path / f"outdir/e00/e01-test/a/checkpoints/step_{final_step:07d}.pt").touch()
    jobs.run(trunk)
    assert calls == ["train", *(INSIDE_RUN if evals else [])]  # on one, train() evaluated it
    anneal = replace(trunk, savedir="outdir/e00/e01-test/a/step_0001000/", anneal_from=str(specs.milestone(trunk, 1000)))
    calls.clear()
    jobs.run(anneal)  # an anneal has no milestones: it evaluates its end
    assert calls == ["train", *INSIDE_RUN]


def test_submit_refuses_a_job_its_spec_cant_run(tmp_path, monkeypatch):
    # An eval trains nothing: run would ignore its weights and train from scratch.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jobs, "bsub", lambda *a, **kw: pytest.fail("submitted"))
    with pytest.raises(AssertionError, match="no LSF job 'run' for a Eval"):
        jobs.submit("run", jobs.Eval(savedir="outdir/e00/x/a/", train=jobs.Train(), init_from="random"))
    with pytest.raises(AssertionError, match="no LSF job 'probe' for a Train"):
        jobs.submit("probe", jobs.Train(savedir="outdir/e00/x/a/"))
    assert not (tmp_path / "outdir").exists()  # refused before writing its spec


def test_submit_writes_the_spec_its_job_loads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    submitted = []
    monkeypatch.setattr(jobs, "bsub", lambda savedir, job, minutes, n_gpus, queue, cmd, **kw: submitted.append((job, n_gpus, queue, cmd)))
    trunk = jobs.Train(savedir="outdir/e00/x/a/", queue="gpu_b300", global_size=(64, 64, 64), model=specs.Views(lamb=0.5), n_gpus=2, max_hours=1.0, n_workers=11)
    e = jobs.Eval(savedir="outdir/e00/x/b/", train=trunk, init_from="random", data="fish2")
    f = jobs.Finetune(savedir="outdir/e00/x/c/", train=trunk, init_from="outdir/e00/y/a/", fit_side=256, freeze_encoder=False)
    jobs.submit("run", trunk)
    jobs.submit("probe", e)
    jobs.submit("finetune", f)
    assert submitted == [("run", 2, "gpu_b300", "torchrun --standalone --nproc_per_node=2 {code}/experiment.py run outdir/e00/x/a/train.json"),
                         ("probe", 1, "gpu_b300", "python {code}/experiment.py probe outdir/e00/x/b/eval.json"),  # the trunk's queue
                         ("finetune", 1, "gpu_b300", "python {code}/experiment.py finetune outdir/e00/x/c/finetune.json")]
    assert jobs.load("outdir/e00/x/a/train.json", jobs.Train) == trunk  # what the job runs: the same spec
    assert jobs.load("outdir/e00/x/b/eval.json", jobs.Eval) == e
    assert jobs.load("outdir/e00/x/c/finetune.json", jobs.Finetune) == f
    with pytest.raises(AssertionError, match="no LSF job 'finetune' for a Eval"):
        jobs.submit("finetune", e)
    spec = json.loads(Path("outdir/e00/x/b/eval.json").read_text())
    Path("bad.json").write_text(json.dumps({**spec, "score": "maybe"}))
    with pytest.raises(Exception, match="score"):  # validated: a typo fails at load, before any GPU work
        jobs.load("bad.json", jobs.Eval)
    Path("extra.json").write_text(json.dumps({**spec, "steps": 3}))
    with pytest.raises(AssertionError, match="doesn't round-trip"):  # so does a field the spec doesn't have
        jobs.load("extra.json", jobs.Eval)


def test_a_scored_finetune_is_followed_by_its_score_job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    submitted = []
    monkeypatch.setattr(jobs, "bsub", lambda savedir, job, minutes, n_gpus, queue, cmd, **kw: submitted.append((job, n_gpus, queue, kw.get("after", ""))))
    trunk = jobs.Train(savedir="outdir/e00/x/a/", queue="gpu_b300")
    for score in (False, True):
        submitted.clear()
        jobs.start(jobs.Finetune(savedir=f"outdir/e00/x/{'ft-' * score}run/", train=trunk, init_from="random", score=score))
        assert [j for j, *_ in submitted] == ["finetune"] + ["score"] * score
    assert submitted[1] == ("score", 0, jobs.SCORE_QUEUE, "e00-x-ft-run-finetune")  # a CPU job, held for the finetune
    assert jobs.load("outdir/e00/x/ft-run/finetune.json", jobs.Finetune).score  # what score loads


def test_a_finetune_of_a_trunk_here_waits_in_lsf_for_its_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    held = []
    monkeypatch.setattr(jobs, "bsub", lambda savedir, job, minutes, n_gpus, queue, cmd, after="", **kw: held.append(after))
    trunk = jobs.Train(savedir="outdir/e00/e01-test/a/")
    finetune = lambda init: jobs.Finetune(savedir="outdir/e00/e01-test/b/", train=trunk, init_from=init)
    for init in [trunk.savedir, "random", "outdir/e00/e00-other/a/"]:
        jobs.submit("finetune", finetune(init))
    Path(trunk.savedir).mkdir(parents=True)
    Path(trunk.savedir, "probe.json").write_text("{}")  # the trunk is done: nothing to wait for (LSF may have forgotten its job)
    jobs.submit("finetune", finetune(trunk.savedir))
    assert held == ["e00-e01-test-a-run", "", "", ""]


def test_retryall_restarts_only_finetunes_that_ended_unfinished(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    footer = "Successfully completed.\n\nResource usage summary:\n"
    trunk = jobs.Train()
    exp = Experiment("e00/e01-test", {"trunk": trunk, **{n: jobs.Finetune(savedir="", train=trunk) for n in ("done", "failed", "running", "queued")}})
    for name, log, png in [("done", footer, True), ("failed", footer, False), ("running", "still running", False), ("queued", None, False)]:
        d = Path(f"outdir/e00/e01-test/{name}/")
        d.mkdir(parents=True)
        if log:
            (d / "job_finetune_1.log").write_text(log)
        if png:
            (d / "finetune.png").write_bytes(b"")
    started = []
    monkeypatch.setattr(jobs, "check_experiment", lambda exp: None)
    monkeypatch.setattr(jobs, "assert_committed", lambda: None)
    monkeypatch.setattr(jobs, "start", lambda p: started.append(p.savedir))
    jobs.retryall(exp)
    assert started == ["outdir/e00/e01-test/failed/"]


def sweep(monkeypatch, id, init_from=""):
    """The Experiment id of 3 runs a, b and c (evals of init_from/<run>/ if set, else trainings), start() recording their names, and no LSF."""
    submitted = []
    run = lambda n: jobs.Eval(savedir="", train=jobs.Train(), init_from=f"{init_from}/{n}/") if init_from else jobs.Train()
    monkeypatch.setattr(jobs, "assert_committed", lambda: None)
    monkeypatch.setattr(jobs, "start", lambda p: submitted.append(Path(p.savedir).name))
    monkeypatch.setattr(jobs, "bsub", lambda *a, **kw: None)
    return Experiment(id, {n: run(n) for n in "abc"}), submitted


def test_submitall_submits_every_run_of_the_experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp, submitted = sweep(monkeypatch, "e00/e01-pca", init_from="outdir/e00/viewsizes-v2")  # a run may read a pre-numbering sweep
    jobs.submitall(exp)
    assert submitted == ["a", "b", "c"]


@pytest.mark.parametrize("id,init_from,existing,error", [
    ("e00/viewsizes-pca", "", [], "want e<series>"),  # no id
    ("e00/e01-pca", "", ["e01-train"], "id e01 is taken by e00/e01-train"),
    ("e00/e01-02-pca", "outdir/e00/e01-train", [], "builds on e00/e01, which doesn't exist"),
    ("e00/e01-02-pca", "outdir/e00/e03-train", ["e01-train", "e03-train"], "init_from read ['e03-train']"),
    ("e00/e04-pca", "outdir/e00/e01-train", ["e01-train"], "name it e01-<NN>-<slug>"),  # a root can't hide its parent
])
def test_submitall_refuses_an_experiment_misnamed(tmp_path, monkeypatch, id, init_from, existing, error):
    monkeypatch.chdir(tmp_path)
    for s in existing:
        (tmp_path / "outdir/e00" / s).mkdir(parents=True)
    exp, submitted = sweep(monkeypatch, id, init_from)
    with pytest.raises(AssertionError, match=re.escape(error)):
        jobs.submitall(exp)
    assert submitted == []  # refused before any job


def test_submitall_takes_a_child_of_its_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for s in ("e01-train", "e01-01-pca", "e01-02-probe"):  # e01-02-probe: this experiment's own dir, from a first submit
        (tmp_path / "outdir/e00" / s).mkdir(parents=True)
    exp, submitted = sweep(monkeypatch, "e00/e01-02-probe", init_from="outdir/e00/e01-train")
    jobs.submitall(exp)
    assert submitted == ["a", "b", "c"] and jobs.experiment_id("e01-02-probe") == "e01-02"


def test_a_milestone_eval_runs_the_tasks_inside_run_without_stopping_or_steering_training(tmp_path, monkeypatch):
    trunk = jobs.Train(savedir=f"{tmp_path}/a/", total_steps=100)
    e = jobs.eval_params(trunk, 10)
    calls = []

    def stub(fn):
        return lambda job: calls.append((fn, job, torch.rand(100)))  # draws from the RNG training uses

    for fn in INSIDE_RUN:
        monkeypatch.setitem(jobs.ENTRYPOINTS, fn, stub(fn))
    torch.manual_seed(0)
    jobs.eval_milestone(trunk, 10)
    after = torch.rand(3)
    torch.manual_seed(0)
    assert torch.equal(after, torch.rand(3))  # the RNG state is as if no eval had run
    assert [c[:2] for c in calls] == [(fn, e) for fn in INSIDE_RUN]
    assert jobs.load(f"{e.savedir}eval.json", jobs.Eval) == e  # written first, for `lsf probe`

    def failing(job):
        raise RuntimeError("eval failed")

    monkeypatch.setitem(jobs.ENTRYPOINTS, INSIDE_RUN[0], failing)
    calls.clear()
    jobs.eval_milestone(trunk, 20)  # swallowed: the trunk trains on, and the later tasks are not tried
    assert calls == []


@pytest.mark.parametrize("fn,job_type", [("benchmark", "Train"), ("probe", "Eval")])
def test_lsf_submits_a_training_run_as_what_the_task_takes(tmp_path, monkeypatch, fn, job_type):
    monkeypatch.chdir(tmp_path)
    exp, _ = sweep(monkeypatch, "e00/e01-test")
    exp.runs["b"].n_workers = 4  # the default's 16 leave no core of CPUS_PER_GPU for the training process
    submitted = []
    monkeypatch.setattr(jobs, "dump", lambda job: "spec.json")
    monkeypatch.setattr(jobs, "bsub", lambda savedir, job, *a, **kw: submitted.append(job))
    jobs.lsf(exp, fn, "b")
    assert submitted == [fn], f"lsf {fn} on a training run should submit one {job_type}-taking job"
