"""Tests for lib.task_utils and experiment.py: the tasks' wiring (trunk, branch, score), the LSF submission, the commands on an
Experiment, and the experiment file."""

import json
import re
import typing
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiment
import tasks
from lib import evals, task_utils, specs
from lib.task_utils import Experiment


@pytest.fixture(autouse=True)
def provenance(monkeypatch):  # run records independent of git state
    monkeypatch.setattr(specs, "code_provenance", lambda: {"commit_id": "test-commit", "subject": "test", "dirty": False, "diff_hash": "test-diff"})


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
    monkeypatch.setattr(tasks, "go", lambda fn, job: gone.append((fn.__name__, job)))
    monkeypatch.setattr(tasks, "check_snapshot", lambda: None)
    monkeypatch.setattr(tasks, "load_checkpoint_file", lambda path: path)  # the loaded checkpoint stands for its path
    return gone


TRUNK = tasks.Trunk(savedir="outdir/e00/e01-test/a/", train=specs.Train(optim=specs.Optim(lr=1e-3)), segments=3, segment_steps=100,
                   anneal_steps=40, anneal_to=0.1, eval_data="hemibrain")


def test_a_trunk_branches_at_its_start_and_after_every_segment_at_a_constant_lr(gone):
    tasks.trunk(TRUNK)
    assert FakeTrainer.log[0] == ("init", TRUNK.savedir, 300, "")
    assert [e for e in FakeTrainer.log if e[0] == "train"] == [("train", 100, 1e-3, 1e-3)] * 3
    assert [Path(e[1]).parent.name for e in FakeTrainer.log if e[0] == "save"] == [f"step_{s:07d}" for s in [0, 100, 200, 300]]
    assert [fn for fn, _ in gone] == ["branch"] * 4
    first = gone[1][1]  # the branch of the first segment's end: its weights, in its own dir, of the trunk that made it
    assert first == tasks.Branch(savedir=f"{TRUNK.savedir}step_0000100/", trunk=TRUNK, init_from=f"{TRUNK.savedir}step_0000100/start.pt")


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
    b = tasks.Branch(savedir=f"{start.parent}/", trunk=TRUNK, init_from=str(start))
    monkeypatch.setattr(tasks, "load_checkpoint", lambda train, init_from: (SimpleNamespace(encoder="enc"), 140))
    monkeypatch.setattr(evals, "pca_maps", lambda enc, par, savedir, step, data: calls.append(("pca", enc, step, data)))
    monkeypatch.setattr(evals, "probe", lambda model, train, data, savedir, step, standardize=False: calls.append(("probe", savedir, data, standardize)))
    monkeypatch.setattr(evals, "finetune", lambda model, train, data, savedir, step: calls.append(("finetune", savedir, data)))
    tasks.branch(b)
    end = f"{start.parent}/checkpoint.pt"
    assert FakeTrainer.log == [("init", b.savedir, 40, str(start)), ("train", 40, 1e-3, pytest.approx(1e-4)), ("save", end)]
    assert not start.exists()  # taken
    assert calls == [("pca", "enc", 140, "hemibrain"), ("probe", b.savedir, "hemibrain", False)]
    assert gone == [("score", tasks.Score(savedir=b.savedir, branch=b))]
    calls.clear(), gone.clear()
    start.write_text("start")
    tasks.branch(replace(b, trunk=replace(TRUNK, eval_head="unetr")))  # the variant: a UNETR, scored all the same
    assert calls[1][0] == "finetune" and [fn for fn, _ in gone] == ["score"]


def test_score_keeps_a_branch_only_if_it_is_the_trunks_best(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trunk = Path("outdir/e00/e01-test/a")
    pqs = {"step_0000100": 0.2, "step_0000200": 0.5, "step_0000300": 0.3}
    for step, pq in pqs.items():
        d = trunk / step
        d.mkdir(parents=True)
        (d / "checkpoint.pt").write_text(step)
        b = tasks.Branch(savedir=f"{d}/", trunk=TRUNK, init_from="")
        monkeypatch.setattr(tasks, "mia_eval", lambda par, pq=pq: pq)
        tasks.score(tasks.Score(savedir=b.savedir, branch=b))
    assert (trunk / "best.pt").read_text() == "step_0000200"
    assert json.loads((trunk / "best.json").read_text())["step"] == 200
    assert not list(trunk.glob("step_*/checkpoint.pt"))  # the losers' weights are deleted


def test_every_task_declares_known_hardware():
    assert {"benchmark", "trunk", "branch", "score"} <= set(task_utils.TASKS)
    assert {t.needs for t in task_utils.TASKS.values()} <= {"gpus", "cpu"}
    assert set(task_utils.BOARDS) == set(typing.get_args(typing.get_type_hints(tasks.Trunk)["board"]))  # Trunk.board names a leaderboard


def test_go_refuses_a_task_for_a_spec_it_does_not_take(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(task_utils, "bsub", lambda *a, **kw: pytest.fail("submitted"))
    with pytest.raises(AssertionError, match="task 'branch' takes"):
        task_utils.go(tasks.branch, replace(TRUNK, savedir="outdir/e00/x/a/"))
    with pytest.raises(AssertionError, match="task 'trunk' takes"):
        task_utils.go(tasks.trunk, tasks.Branch(savedir="outdir/e00/x/a/", trunk=TRUNK, init_from=""))
    assert not (tmp_path / "outdir").exists()  # refused before writing its spec


def test_go_writes_the_spec_its_job_loads_and_asks_for_the_hardware_the_task_needs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    submitted = []
    monkeypatch.setattr(task_utils, "bsub", lambda savedir, job, minutes, n_gpus, queue, cmd, **kw: submitted.append((job, minutes, n_gpus, queue, cmd)))
    trunk = replace(TRUNK, savedir="outdir/e00/x/a/", hours=3, train=specs.Train(hardware=specs.Hardware(queue="gpu_b300", n_gpus=2, n_workers=11)))
    b = tasks.Branch(savedir="outdir/e00/x/a/step_0000100/", trunk=trunk, init_from="outdir/e00/x/a/step_0000100/start.pt")
    s = tasks.Score(savedir=b.savedir, branch=b)
    task_utils.go(tasks.trunk, trunk)
    task_utils.go(tasks.branch, b)  # the trunk's queue and GPUs
    task_utils.go(tasks.score, s)
    assert submitted == [("trunk", 180, 2, "gpu_b300", "torchrun --standalone --nproc_per_node=2 {code}/experiment.py trunk outdir/e00/x/a/trunk.json"),
                         ("branch", 120, 2, "gpu_b300", "torchrun --standalone --nproc_per_node=2 {code}/experiment.py branch outdir/e00/x/a/step_0000100/branch.json"),
                         ("score", task_utils.SCORE_MINUTES, 0, task_utils.SCORE_QUEUE, "python {code}/experiment.py score outdir/e00/x/a/step_0000100/score.json")]
    assert task_utils.load("outdir/e00/x/a/trunk.json", tasks.Trunk) == trunk  # what the job runs: the same spec
    assert task_utils.load("outdir/e00/x/a/step_0000100/branch.json", tasks.Branch) == b
    assert task_utils.load("outdir/e00/x/a/step_0000100/score.json", tasks.Score) == s
    spec = json.loads(Path("outdir/e00/x/a/step_0000100/branch.json").read_text())
    Path("bad.json").write_text(json.dumps({**spec, "init_from": 3}))
    with pytest.raises(Exception, match="init_from"):  # validated: a typo fails at load, before any GPU work
        task_utils.load("bad.json", tasks.Branch)
    Path("extra.json").write_text(json.dumps({**spec, "steps": 3}))
    with pytest.raises(AssertionError, match="doesn't round-trip"):  # so does a field the spec doesn't have
        task_utils.load("extra.json", tasks.Branch)


def test_the_experiment_file_names_runs_that_load_as_a_job_loads_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, r in experiment.EXPERIMENT.roots().items():  # a job validates its json (pydantic's Literals), which building the dataclass doesn't
        assert r.savedir == f"outdir/{experiment.EXPERIMENT.id}/{name}/"
        assert task_utils.parse(type(r), asdict(r)) == r, name
        specs.model_config(r.train)  # its model builds on its encoder's patch
        assert r.hours * 60 > 0 and r.segments * r.segment_steps > 0
        assert r.train.hardware.n_workers + 1 <= task_utils.CPUS_PER_GPU
    assert experiment.EXPERIMENT.task is tasks.trunk
    task_utils.check_experiment(experiment.EXPERIMENT)


def sweep(monkeypatch, id, init_from=""):
    """The Experiment id of 3 trunks a, b and c (from init_from/<run>/checkpoint.pt if set), start() recording their names, and no LSF."""
    started = []
    run = lambda n: tasks.Trunk(init_from=f"{init_from}/{n}/checkpoint.pt" if init_from else "")
    monkeypatch.setattr(task_utils, "assert_committed", lambda: None)
    monkeypatch.setattr(task_utils, "start", lambda exp, p: started.append(Path(p.savedir).name))
    monkeypatch.setattr(task_utils, "bsub", lambda *a, **kw: None)
    return Experiment(id, tasks.trunk, {n: run(n) for n in "abc"}), started


def test_start_empties_the_savedir_and_submits_the_first_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gone = []
    monkeypatch.setattr(task_utils, "trash", lambda savedir: gone.append(("trash", savedir)))
    monkeypatch.setattr(task_utils, "go", lambda fn, p: gone.append((fn.__name__, p.savedir)))
    exp = Experiment("e00/e01-test", tasks.trunk, {"a": tasks.Trunk()})
    task_utils.start(exp, exp.roots()["a"])
    assert gone == [("trash", "outdir/e00/e01-test/a/"), ("trunk", "outdir/e00/e01-test/a/")]


def test_submitall_starts_each_run_not_submitted_before(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp, started = sweep(monkeypatch, "e00/e01-test")
    task_utils.dump(exp.roots()["b"])  # submitted before
    task_utils.submitall(exp)
    assert started == ["a", "c"]


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
    assert started == ["a", "b", "c"] and task_utils.experiment_id("e01-02-probe") == "e01-02"


def test_lsf_restarts_a_run_at_its_first_task_and_redoes_any_other_in_place(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp, started = sweep(monkeypatch, "e00/e01-test")
    gone = []
    monkeypatch.setattr(task_utils, "go", lambda fn, job: gone.append((fn.__name__, job.savedir)))
    task_utils.lsf(exp, "trunk", "b")
    assert started == ["b"] and gone == []  # start() went
    task_utils.lsf(exp, "benchmark", "b")
    b = tasks.Branch(savedir="outdir/e00/e01-test/b/step_0000100/", trunk=exp.roots()["b"], init_from="random")
    s = tasks.Score(savedir=b.savedir, branch=b)
    task_utils.dump(s)
    task_utils.lsf(exp, "score", f"{s.savedir}score.json")
    assert gone == [("benchmark", "outdir/e00/e01-test/b/"), ("score", s.savedir)]
