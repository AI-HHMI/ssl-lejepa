"""Tests for lib.task_utils and tasks.py: the tasks' wiring (trunk, anneal, finetune, score), the LSF submission, the commands on an Experiment,
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
from lib import evals, scoring, task_entry, task_utils, training
from lib.task_utils import Experiment


@pytest.fixture(autouse=True)
def provenance(monkeypatch):  # run records independent of git state
    monkeypatch.setattr(task_utils, "code_provenance", lambda: {"commit_id": "test-commit", "subject": "test", "dirty": False, "diff_hash": "test-diff"})


class FakeTrainer:
    """tasks.Trainer without a model: records what the phase asks of it."""
    log = []
    stop_after = None  # the number of train() calls after which .stop_reason is set

    def __init__(self, config, savedir, steps, init=None):
        self.config, self.step, self.stop_reason, self.r, self.n = config, 0, "", SimpleNamespace(rank0=True), 0
        self.log.append(("init", savedir, steps, init))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def train(self, steps, lr0, lr1):
        self.step += steps
        self.n += 1
        self.stop_reason = "diverged" if self.n == self.stop_after else ""
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


TRUNK = tasks.Trunk(savedir="outdir/e00/e01-test/a", train=training.Train(optim=training.Optim(lr=1e-3)), segments=3, segment_steps=100,
                    anneal_steps=40, anneal_lr_frac=0.1, eval_data="hemibrain")


def test_a_trunk_branches_at_its_start_and_after_every_segment_at_a_constant_lr(gone):
    tasks.trunk(TRUNK)
    assert FakeTrainer.log[0] == ("init", TRUNK.savedir, 300, "")
    assert [e for e in FakeTrainer.log if e[0] == "train"] == [("train", 100, 1e-3, 1e-3)] * 3
    assert [Path(e[1]).parent.name for e in FakeTrainer.log if e[0] == "save"] == [f"step_{s:07d}" for s in [0, 100, 200, 300]]
    assert [g[0] for g in gone] == ["anneal"] * 4
    assert (Path(TRUNK.savedir) / "last.pt").read_text() == "step 300"  # the state to continue from
    # the branch of the first segment's end: its own dir, the trunk's hardware, the trunk and the fork point it takes
    assert gone[1] == ("anneal", f"{TRUNK.savedir}/step_0000100", tasks.ANNEAL_HARDWARE, tasks.ANNEAL_MINUTES,
                       {"t": TRUNK, "fork_point": f"{TRUNK.savedir}/step_0000100/fork.pt"})


def test_a_trunk_continuing_an_init_from_branches_only_after_its_segments(gone):
    tasks.trunk(replace(TRUNK, init_from="outdir/e00/e00-prev/a/last.pt"))
    assert FakeTrainer.log[0] == ("init", TRUNK.savedir, 300, "outdir/e00/e00-prev/a/last.pt")
    assert [Path(e[1]).parent.name for e in FakeTrainer.log if e[0] == "save"] == [f"step_{s:07d}" for s in [100, 200, 300]]


def test_a_trunk_that_diverges_branches_no_more(gone):
    FakeTrainer.stop_after = 2
    tasks.trunk(TRUNK)
    assert [Path(e[1]).parent.name for e in FakeTrainer.log if e[0] == "save"] == ["step_0000000", "step_0000100"]
    assert len(gone) == 2


def test_an_anneal_goes_to_finetune_on_one_small_gpu(gone):
    fork_point = Path(TRUNK.savedir) / "step_0000100/fork.pt"
    fork_point.parent.mkdir(parents=True)
    fork_point.write_text("fork point")
    branch_dir = str(fork_point.parent)
    tasks.anneal(TRUNK, str(fork_point))
    end = f"{branch_dir}/checkpoint.pt"
    assert FakeTrainer.log == [("init", branch_dir, 40, str(fork_point)), ("train", 40, 1e-3, pytest.approx(1e-4)), ("save", end)]
    assert not fork_point.exists()  # taken
    assert gone == [("finetune", branch_dir, tasks.FINETUNE_HARDWARE, tasks.FINETUNE_MINUTES, {"t": TRUNK, "branch_dir": branch_dir})]
    assert tasks.FINETUNE_HARDWARE.n_gpus == 1 and tasks.FINETUNE_HARDWARE.queue == "gpu_rtx6000"


def test_a_finetune_evaluates_the_annealed_weights_and_goes_to_score(gone, monkeypatch):
    calls = []
    branch_dir = f"{TRUNK.savedir}/step_0000100"
    monkeypatch.setattr(tasks, "load_model", lambda train, init_from: (SimpleNamespace(encoder="enc"), 140))
    monkeypatch.setattr(evals, "pca_maps", lambda enc, train, savedir, step, data: calls.append(("pca", enc, step, data)))
    monkeypatch.setattr(evals, "probe", lambda model, train, data, savedir, step, standardize=False: calls.append(("probe", savedir, data, standardize)))
    monkeypatch.setattr(evals, "finetune", lambda model, train, data, savedir, step: calls.append(("finetune", savedir, data)))
    tasks.finetune(TRUNK, branch_dir)
    assert calls == [("pca", "enc", 140, "hemibrain"), ("probe", branch_dir, "hemibrain", False)]
    assert gone == [("score", branch_dir, tasks.SCORE_HARDWARE, tasks.SCORE_MINUTES, {"t": TRUNK, "branch_dir": branch_dir})]  # a CPU job
    calls.clear(), gone.clear()
    tasks.finetune(replace(TRUNK, eval_head="finetune"), branch_dir)  # the variant: a UNETR, scored all the same
    assert calls[1] == ("finetune", branch_dir, "hemibrain") and [g[0] for g in gone] == ["score"]


def test_score_keeps_a_branch_only_if_it_is_the_trunks_best(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trunk = Path("outdir/e00/e01-test/a")
    pqs = {"step_0000100": 0.2, "step_0000200": 0.5, "step_0000300": 0.3}
    for step, pq in pqs.items():
        d = trunk / step
        d.mkdir(parents=True)
        (d / "checkpoint.pt").write_text(step)
        monkeypatch.setattr(tasks, "mia_pq", lambda t, savedir, pq=pq: pq)
        tasks.score(TRUNK, str(d))
    assert (trunk / "best.pt").read_text() == "step_0000200"
    assert json.loads((trunk / "best.json").read_text())["step"] == 200
    assert not list(trunk.glob("step_*/checkpoint.pt"))  # the losers' weights are deleted


def test_the_boards_of_a_trunk_are_the_leaderboards_scoring_knows():
    assert set(scoring.BOARD_CONFIGS) == set(typing.get_args(typing.get_type_hints(tasks.Trunk)["board"]))


def test_go_writes_the_job_file_that_load_args_reads_and_submit_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    submitted = []
    monkeypatch.setattr(task_utils, "bsub", lambda savedir, job, minutes, hardware, cmd: submitted.append((savedir, job, minutes, hardware, cmd)))
    hardware = training.Hardware(queue="gpu_b300", n_gpus=2, n_workers=11)
    trunk = replace(TRUNK, savedir="outdir/e00/x/a", train=training.Train(hardware=hardware))
    branch_dir, fork_point = "outdir/e00/x/a/step_0000100", "outdir/e00/x/a/step_0000100/fork.pt"
    task_utils.go(tasks.anneal, branch_dir, hardware, 120, t=trunk, fork_point=fork_point)
    task_utils.go(tasks.score, branch_dir, tasks.SCORE_HARDWARE, tasks.SCORE_MINUTES, t=trunk, branch_dir=branch_dir)
    file = f"{branch_dir}/anneal.json"
    assert submitted == [(branch_dir, "anneal", 120, hardware, f"torchrun --standalone --nproc_per_node=2 {{code}}/experiment.py anneal {file}"),
                         (branch_dir, "score", tasks.SCORE_MINUTES, tasks.SCORE_HARDWARE, f"python {{code}}/experiment.py score {branch_dir}/score.json")]  # a CPU job: no torchrun
    assert task_utils.load_args(tasks.anneal, file) == {"t": trunk, "fork_point": fork_point}  # what the job runs: the same arguments
    job = json.loads(Path(file).read_text())
    assert job["task"] == "anneal" and job["minutes"] == 120 and job["hardware"]["n_gpus"] == 2
    bad = {**job, "args": {**job["args"], "fork_point": 3}}
    Path("bad.json").write_text(json.dumps(bad))
    with pytest.raises(Exception, match="valid string"):  # validated: a mistyped argument fails at load, before any GPU work
        task_utils.load_args(tasks.anneal, "bad.json")
    Path("extra.json").write_text(json.dumps({**job, "args": {**job["args"], "steps": 3}}))
    with pytest.raises(AssertionError, match="takes"):  # so does one the task doesn't take
        task_utils.load_args(tasks.anneal, "extra.json")
    with pytest.raises(AssertionError, match="takes"):  # and go() refuses it before writing anything
        task_utils.go(tasks.anneal, "outdir/e00/x/b", hardware, 120, t=trunk)
    assert not Path("outdir/e00/x/b").exists()
    submitted.clear()
    task_utils.submit_file(file)  # a redo: the same job, from its file
    assert [s[1] for s in submitted] == ["anneal"] and submitted[0][3] == hardware


def test_the_experiment_file_names_runs_that_load_as_a_job_loads_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, r in experiment.EXPERIMENT.placed_runs().items():  # a job validates its json (pydantic's Literals), which building the dataclass doesn't
        assert r.savedir == f"outdir/{experiment.EXPERIMENT.id}/{name}"
        assert task_utils.parse(tasks.Trunk, asdict(r)) == r, name
        training.model_config(r.train)  # its model builds on its encoder's patch
        assert r.segments * r.segment_steps > 0
    assert experiment.EXPERIMENT.start is tasks.start
    for h in (tasks.TRUNK_HARDWARE, tasks.ANNEAL_HARDWARE, tasks.FINETUNE_HARDWARE):
        assert h.n_workers + 1 <= task_utils.CPUS_PER_GPU
    task_entry.assert_experiment(experiment.EXPERIMENT)


def sweep(monkeypatch, id, init_from=""):
    """The Experiment id of 3 trunks a, b and c (from init_from/<run>/checkpoint.pt if set), recording (savedir emptied | task started,
    run, its hardware, minutes) and with no LSF."""
    started = []
    run = lambda n: tasks.Trunk(init_from=f"{init_from}/{n}/checkpoint.pt" if init_from else "")
    monkeypatch.setattr(task_entry, "assert_committed", lambda: None)
    monkeypatch.setattr(task_entry, "trash", lambda savedir: started.append(("trash", Path(savedir).name)))
    monkeypatch.setattr(tasks, "go", lambda fn, savedir, hardware, minutes, **args: started.append((fn.__name__, Path(savedir).name, hardware, minutes)))
    return Experiment(id, tasks.start, {n: run(n) for n in "abc"}), started


def test_submitall_starts_each_run_not_submitted_before_fresh(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp, started = sweep(monkeypatch, "e00/e01-test")
    Path("outdir/e00/e01-test/b").mkdir(parents=True)
    Path("outdir/e00/e01-test/b/trunk.json").write_text("{}")  # submitted before
    task_entry.submitall(exp)
    a, c = [("trunk", n, tasks.TRUNK_HARDWARE, tasks.TRUNK_MINUTES) for n in "ac"]
    assert started == [("trash", "a"), a, ("trash", "c"), c]


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
        task_entry.submitall(exp)
    assert started == []  # refused before any job


def test_submitall_takes_a_child_of_its_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for s in ("e01-train", "e01-01-pca", "e01-02-probe"):  # e01-02-probe: this experiment's own dir, from a first submit
        (tmp_path / "outdir/e00" / s).mkdir(parents=True)
    exp, started = sweep(monkeypatch, "e00/e01-02-probe", init_from="outdir/e00/e01-train")
    task_entry.submitall(exp)
    assert [s[1] for s in started if s[0] == "trunk"] == ["a", "b", "c"] and task_entry.experiment_id("e01-02-probe") == "e01-02"


def test_lsf_restarts_a_run_with_the_experiments_task_and_redoes_any_other_from_its_job_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp, started = sweep(monkeypatch, "e00/e01-test")
    submitted = []
    monkeypatch.setattr(task_entry, "submit_file", submitted.append)
    task_entry.resubmit_task(exp, "start", "b")
    assert [s[:2] for s in started] == [("trash", "b"), ("trunk", "b")]
    with pytest.raises(AssertionError, match="redo score from its job file"):
        task_entry.resubmit_task(exp, "score", "b")
    file = "outdir/e00/e01-test/b/step_0000100/score.json"
    task_entry.resubmit_task(exp, "score", file)
    assert submitted == [file]
    with pytest.raises(AssertionError, match="is not a run"):
        task_entry.resubmit_task(exp, "anneal", file)  # the file is another task's


def test_the_tasks_compose_through_go_from_start_to_score(tmp_path, monkeypatch):
    """Every submission is a job file and an LSF command; running each command as a job does (main, on the file) walks start -> trunk -> anneal -> finetune -> score."""
    monkeypatch.chdir(tmp_path)  # the real go and submit_file, with only LSF and the heavy work faked below
    jobs, ran = [], []
    monkeypatch.setattr(task_utils, "bsub", lambda savedir, job, minutes, hardware, cmd: jobs.append(cmd.format(code="CODE").split()))
    monkeypatch.setattr(tasks, "Trainer", FakeTrainer)
    monkeypatch.setattr(tasks, "assert_snapshot", lambda: None)
    monkeypatch.setattr(tasks, "load_checkpoint", lambda path: path)
    monkeypatch.setattr(tasks, "load_model", lambda train, init_from: (SimpleNamespace(encoder="enc"), 140))
    monkeypatch.setattr(evals, "pca_maps", lambda *a: None)
    monkeypatch.setattr(evals, "probe", lambda *a: None)
    monkeypatch.setattr(tasks, "mia_pq", lambda t, savedir: 0.5)
    monkeypatch.setattr(tasks, "keep_if_best", lambda t, savedir, score: ran.append(("kept?", Path(savedir).name, score)))
    monkeypatch.setattr(task_entry, "log_command", lambda argv: None)
    monkeypatch.setattr(task_entry, "assert_committed", lambda: None)
    monkeypatch.setattr(task_entry, "trash", lambda savedir: None)
    FakeTrainer.log, FakeTrainer.stop_after = [], None
    exp = Experiment("e00/e01-test", tasks.start, {"a": replace(TRUNK, segments=1)})
    task_entry.submitall(exp)
    while jobs:
        _, *argv = jobs.pop(0)  # [torchrun --standalone --nproc_per_node=N | python] CODE/experiment.py <task> <file>
        i = next(i for i, a in enumerate(argv) if a.endswith("experiment.py"))
        monkeypatch.setattr("sys.argv", ["experiment.py", *argv[i + 1:]])
        ran.append(argv[i + 1])
        task_entry.main(exp, tasks.TASKS)
    # 1 segment: a trunk, a branch at its start and after the segment; each anneals, finetunes and scores
    assert ran[0] == "trunk" and sorted(r for r in ran[1:] if isinstance(r, str)) == ["anneal"] * 2 + ["finetune"] * 2 + ["score"] * 2
    assert [r[0] for r in ran if isinstance(r, tuple)] == ["kept?"] * 2
