"""Tests for lib.task_utils and tasks.py: the tasks' wiring (trunk, branch, score), the LSF submission, the commands on an Experiment,
and the experiment file."""

import json
import re
import typing
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiment
import tasks
from lib import evals, task_utils, training
from lib.task_utils import Experiment


@pytest.fixture(autouse=True)
def provenance(monkeypatch):  # run records independent of git state
    monkeypatch.setattr(task_utils, "code_provenance", lambda: {"commit_id": "test-commit", "subject": "test", "dirty": False, "diff_hash": "test-diff"})


class FakeTrainer:
    """tasks.Trainer without a model: records what the phase asks of it."""
    log = []
    stop_after = None  # the number of train() calls after which .stopped is set

    def __init__(self, par, savedir, steps, init=None):
        self.par, self.step, self.stopped, self.r, self.n = par, 0, "", SimpleNamespace(rank0=True), 0
        self.log.append(("init", savedir, steps, init))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def train(self, steps, lr0, lr1):
        self.step += steps
        self.n += 1
        self.stopped = "diverged" if self.n == self.stop_after else ""
        self.log.append(("train", steps, lr0, lr1))

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"step {self.step}")
        self.log.append(("save", str(path)))


@pytest.fixture
def gone(tmp_path, monkeypatch):
    """tasks.py's tasks in tmp_path with a FakeTrainer; the tasks they go() to are recorded (the fixture), not submitted."""
    monkeypatch.chdir(tmp_path)
    FakeTrainer.log, FakeTrainer.stop_after = [], None
    gone = []
    monkeypatch.setattr(tasks, "Trainer", FakeTrainer)
    monkeypatch.setattr(tasks, "go", lambda fn, savedir, hardware, minutes, **args: gone.append((fn.__name__, savedir, hardware, minutes, args)))
    monkeypatch.setattr(tasks, "assert_snapshot", lambda: None)
    monkeypatch.setattr(tasks, "load_checkpoint", lambda path: path)  # the loaded checkpoint stands for its path
    return gone


TRUNK = tasks.Trunk(savedir="outdir/e00/e01-test/a/", train=training.Train(optim=training.Optim(lr=1e-3)), segments=3, segment_steps=100,
                    anneal_steps=40, anneal_to=0.1, eval_data="hemibrain")


def test_a_trunk_branches_at_its_start_and_after_every_segment_at_a_constant_lr(gone):
    tasks.trunk(TRUNK)
    assert FakeTrainer.log[0] == ("init", TRUNK.savedir, 300, "")
    assert [e for e in FakeTrainer.log if e[0] == "train"] == [("train", 100, 1e-3, 1e-3)] * 3
    assert [Path(e[1]).parent.name for e in FakeTrainer.log if e[0] == "save"] == [f"step_{s:07d}" for s in [0, 100, 200, 300]]
    assert [g[0] for g in gone] == ["branch"] * 4
    # the branch of the first segment's end: its own dir, the trunk's hardware, the trunk and the fork point it takes
    assert gone[1] == ("branch", f"{TRUNK.savedir}step_0000100/", TRUNK.train.hardware, tasks.BRANCH_MINUTES,
                       {"t": TRUNK, "start": f"{TRUNK.savedir}step_0000100/start.pt"})


def test_a_trunk_that_diverges_branches_no_more(gone):
    FakeTrainer.stop_after = 2
    tasks.trunk(TRUNK)
    assert [Path(e[1]).parent.name for e in FakeTrainer.log if e[0] == "save"] == ["step_0000000", "step_0000100"]
    assert len(gone) == 2


def test_a_branch_anneals_then_evaluates_and_goes_to_score(gone, monkeypatch):
    calls = []
    start = Path(TRUNK.savedir) / "step_0000100/start.pt"
    start.parent.mkdir(parents=True)
    start.write_text("start")
    savedir = f"{start.parent}/"
    monkeypatch.setattr(tasks, "load_model", lambda train, init_from: (SimpleNamespace(encoder="enc"), 140))
    monkeypatch.setattr(evals, "pca_maps", lambda enc, par, savedir, step, data: calls.append(("pca", enc, step, data)))
    monkeypatch.setattr(evals, "probe", lambda model, train, data, savedir, step, standardize=False: calls.append(("probe", savedir, data, standardize)))
    monkeypatch.setattr(evals, "finetune", lambda model, train, data, savedir, step: calls.append(("finetune", savedir, data)))
    tasks.branch(TRUNK, str(start))
    end = f"{savedir}checkpoint.pt"
    assert FakeTrainer.log == [("init", savedir, 40, str(start)), ("train", 40, 1e-3, pytest.approx(1e-4)), ("save", end)]
    assert not start.exists()  # taken
    assert calls == [("pca", "enc", 140, "hemibrain"), ("probe", savedir, "hemibrain", False)]
    assert gone == [("score", savedir, None, task_utils.SCORE_MINUTES, {"t": TRUNK, "branch_dir": savedir})]  # a CPU job
    calls.clear(), gone.clear()
    start.write_text("start")
    tasks.branch(replace(TRUNK, eval_head="unetr"), str(start))  # the variant: a UNETR, scored all the same
    assert calls[1] == ("finetune", savedir, "hemibrain") and [g[0] for g in gone] == ["score"]


def test_score_keeps_a_branch_only_if_it_is_the_trunks_best(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(task_utils, "code_provenance", lambda: {"commit_id": "x", "subject": "", "dirty": False, "diff_hash": ""})
    trunk = Path("outdir/e00/e01-test/a")
    pqs = {"step_0000100": 0.2, "step_0000200": 0.5, "step_0000300": 0.3}
    for step, pq in pqs.items():
        d = trunk / step
        d.mkdir(parents=True)
        (d / "checkpoint.pt").write_text(step)
        monkeypatch.setattr(tasks, "mia_pq", lambda t, savedir, pq=pq: pq)
        tasks.score(TRUNK, f"{d}/")
    assert (trunk / "best.pt").read_text() == "step_0000200"
    assert json.loads((trunk / "best.json").read_text())["step"] == 200
    assert not list(trunk.glob("step_*/checkpoint.pt"))  # the losers' weights are deleted


def test_the_boards_of_a_trunk_are_the_leaderboards_task_utils_knows():
    assert set(task_utils.BOARD_CONFIGS) == set(typing.get_args(typing.get_type_hints(tasks.Trunk)["board"]))


def test_go_writes_the_job_file_that_load_args_reads_and_submit_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    submitted = []
    monkeypatch.setattr(task_utils, "bsub", lambda savedir, job, minutes, n_gpus, queue, cmd, **kw: submitted.append((savedir, job, minutes, n_gpus, queue, cmd, kw)))
    hardware = training.Hardware(queue="gpu_b300", n_gpus=2, n_workers=11)
    trunk = replace(TRUNK, savedir="outdir/e00/x/a/", hours=3, train=training.Train(hardware=hardware))
    start = "outdir/e00/x/a/step_0000100/start.pt"
    task_utils.go(tasks.branch, "outdir/e00/x/a/step_0000100/", hardware, 120, t=trunk, start=start)
    task_utils.go(tasks.score, "outdir/e00/x/a/step_0000100/", None, task_utils.SCORE_MINUTES, t=trunk, branch_dir="outdir/e00/x/a/step_0000100/")
    file = "outdir/e00/x/a/step_0000100/branch.json"
    assert submitted == [("outdir/e00/x/a/step_0000100", "branch", 120, 2, "gpu_b300", f"torchrun --standalone --nproc_per_node=2 {{code}}/experiment.py branch {file}", {}),
                         ("outdir/e00/x/a/step_0000100", "score", task_utils.SCORE_MINUTES, 0, task_utils.SCORE_QUEUE,
                          "python {code}/experiment.py score outdir/e00/x/a/step_0000100/score.json", {"slots": task_utils.SCORE_SLOTS})]
    assert task_utils.load_args(tasks.branch, file) == {"t": trunk, "start": start}  # what the job runs: the same arguments
    job = json.loads(Path(file).read_text())
    assert job["task"] == "branch" and job["minutes"] == 120 and job["hardware"]["n_gpus"] == 2
    bad = {**job, "args": {**job["args"], "start": 3}}
    Path("bad.json").write_text(json.dumps(bad))
    with pytest.raises(Exception, match="valid string"):  # validated: a mistyped argument fails at load, before any GPU work
        task_utils.load_args(tasks.branch, "bad.json")
    Path("extra.json").write_text(json.dumps({**job, "args": {**job["args"], "steps": 3}}))
    with pytest.raises(AssertionError, match="takes"):  # so does one the task doesn't take
        task_utils.load_args(tasks.branch, "extra.json")
    with pytest.raises(AssertionError, match="takes"):  # and go() refuses it before writing anything
        task_utils.go(tasks.branch, "outdir/e00/x/b/", hardware, 120, t=trunk)
    assert not Path("outdir/e00/x/b").exists()
    submitted.clear()
    task_utils.submit_file(file)  # a redo: the same job, from its file
    assert [s[1] for s in submitted] == ["branch"] and submitted[0][3] == 2


def test_the_experiment_file_names_runs_that_load_as_a_job_loads_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, r in experiment.EXPERIMENT.roots().items():  # a job validates its json (pydantic's Literals), which building the dataclass doesn't
        assert r.savedir == f"outdir/{experiment.EXPERIMENT.id}/{name}/"
        assert task_utils.parse(tasks.Trunk, asdict(r)) == r, name
        training.model_config(r.train)  # its model builds on its encoder's patch
        assert r.hours * 60 > 0 and r.segments * r.segment_steps > 0
        assert r.train.hardware.n_workers + 1 <= task_utils.CPUS_PER_GPU
    assert experiment.EXPERIMENT.task is tasks.trunk
    task_utils.assert_experiment(experiment.EXPERIMENT)


def sweep(monkeypatch, id, init_from=""):
    """The Experiment id of 3 trunks a, b and c (from init_from/<run>/checkpoint.pt if set), recording (savedir emptied | task started,
    run, its hardware, minutes) and with no LSF."""
    started = []
    run = lambda n: tasks.Trunk(init_from=f"{init_from}/{n}/checkpoint.pt" if init_from else "")
    monkeypatch.setattr(task_utils, "assert_committed", lambda: None)
    monkeypatch.setattr(task_utils, "trash", lambda savedir: started.append(("trash", Path(savedir).name)))
    monkeypatch.setattr(task_utils, "go", lambda fn, savedir, hardware, minutes, **args: started.append((fn.__name__, Path(savedir).name, hardware, minutes)))
    return Experiment(id, tasks.trunk, {n: run(n) for n in "abc"}), started


def test_submitall_starts_each_run_not_submitted_before_fresh(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp, started = sweep(monkeypatch, "e00/e01-test")
    Path("outdir/e00/e01-test/b").mkdir(parents=True)
    Path("outdir/e00/e01-test/b/trunk.json").write_text("{}")  # submitted before
    task_utils.submitall(exp)
    hours = int(tasks.Trunk().hours * 60)
    assert started == [("trash", "a"), ("trunk", "a", tasks.Trunk().train.hardware, hours), ("trash", "c"), ("trunk", "c", tasks.Trunk().train.hardware, hours)]


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
    exp, started = sweep(monkeypatch, id, init_from)
    with pytest.raises(AssertionError, match=re.escape(error)):
        task_utils.submitall(exp)
    assert started == []  # refused before any job


def test_submitall_takes_a_child_of_its_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for s in ("e01-train", "e01-01-pca", "e01-02-probe"):  # e01-02-probe: this experiment's own dir, from a first submit
        (tmp_path / "outdir/e00" / s).mkdir(parents=True)
    exp, started = sweep(monkeypatch, "e00/e01-02-probe", init_from="outdir/e00/e01-train")
    task_utils.submitall(exp)
    assert [s[1] for s in started if s[0] == "trunk"] == ["a", "b", "c"] and task_utils.experiment_id("e01-02-probe") == "e01-02"


def test_lsf_restarts_a_run_with_the_experiments_task_and_redoes_any_other_from_its_job_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp, started = sweep(monkeypatch, "e00/e01-test")
    submitted = []
    monkeypatch.setattr(task_utils, "submit_file", submitted.append)
    task_utils.resubmit_task(exp, "trunk", "b")
    assert [s[:2] for s in started] == [("trash", "b"), ("trunk", "b")]
    with pytest.raises(AssertionError, match="redo score from its job file"):
        task_utils.resubmit_task(exp, "score", "b")
    file = "outdir/e00/e01-test/b/step_0000100/score.json"
    task_utils.resubmit_task(exp, "score", file)
    assert submitted == [file]
    with pytest.raises(AssertionError, match="is not a run"):
        task_utils.resubmit_task(exp, "branch", file)  # the file is another task's
