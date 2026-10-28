"""Tests for lib.jobs and experiment.py: the phases' wiring (trunk, branch, scores), the LSF submission, the commands on an
Experiment, and the experiment file."""

import json
import re
import typing
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiment
from lib import evals, jobs, specs
from lib.jobs import Experiment


@pytest.fixture(autouse=True)
def provenance(monkeypatch):  # run records independent of git state
    monkeypatch.setattr(specs, "code_provenance", lambda: {"commit_id": "test-commit", "subject": "test", "dirty": False, "diff_hash": "test-diff"})


class FakeTrainer:
    """experiment.Trainer without a model: records what the phase asks of it."""
    log = []
    stop_after = None  # the number of train() calls after which .stopped is set

    def __init__(self, par, steps, init_from=""):
        self.par, self.step, self.stopped, self.r, self.n = par, 0, "", SimpleNamespace(rank0=True), 0
        self.log.append(("init", par.savedir, steps, init_from))

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
def phases(tmp_path, monkeypatch):
    """experiment's phases in tmp_path with a FakeTrainer; the phases they go() to are recorded, not submitted."""
    monkeypatch.chdir(tmp_path)
    FakeTrainer.log, FakeTrainer.stop_after = [], None
    gone = []
    monkeypatch.setattr(experiment, "Trainer", FakeTrainer)
    monkeypatch.setattr(experiment, "go", lambda fn, job: gone.append((fn.__name__, job)))
    monkeypatch.setattr(experiment, "check_snapshot", lambda: None)
    return gone


TRUNK = jobs.Train(savedir="outdir/e00/e01-test/a/", segments=3, segment_steps=100, lr=1e-3, anneal_steps=40, anneal_to=0.1, eval_on="hemibrain")


def test_a_trunk_branches_at_its_start_and_after_every_segment_at_a_constant_lr(phases):
    experiment.trunk(TRUNK)
    assert FakeTrainer.log[0] == ("init", TRUNK.savedir, 300, "")
    assert [e for e in FakeTrainer.log if e[0] == "train"] == [("train", 100, 1e-3, 1e-3)] * 3
    steps = [0, 100, 200, 300]
    assert [Path(e[1]).parent.name for e in FakeTrainer.log if e[0] == "save"] == [f"step_{s:07d}" for s in steps]
    assert [fn for fn, _ in phases] == ["branch"] * 4
    first = phases[1][1]  # the branch of the first segment's end: its weights, in its own dir, measured where the trunk says
    assert first == specs.Eval(savedir=f"{TRUNK.savedir}step_0000100/", train=TRUNK, init_from=f"{TRUNK.savedir}step_0000100/start.pt", data="hemibrain")


def test_a_trunk_that_diverges_branches_no_more(phases):
    FakeTrainer.stop_after = 2
    experiment.trunk(TRUNK)
    assert [Path(e[1]).parent.name for e in FakeTrainer.log if e[0] == "save"] == ["step_0000000", "step_0000100"]
    assert len(phases) == 2


def test_a_branch_anneals_then_evaluates_and_goes_to_scores(phases, monkeypatch):
    calls = []
    start = Path(TRUNK.savedir) / "step_0000100/start.pt"
    start.parent.mkdir(parents=True)
    start.write_text("start")
    e = specs.Eval(savedir=f"{start.parent}/", train=TRUNK, init_from=str(start), data="hemibrain")
    monkeypatch.setattr(experiment, "load_checkpoint", lambda e: (SimpleNamespace(encoder="enc"), 140))
    monkeypatch.setattr(evals, "pca_maps", lambda enc, par, savedir, step, data: calls.append(("pca", enc, step, data)))
    monkeypatch.setattr(evals, "probe", lambda e, model, step: calls.append(("probe", e.init_from, e.score)))
    monkeypatch.setattr(evals, "finetune", lambda f, model, step: calls.append(("finetune", f.init_from, f.score)))
    experiment.branch(e)
    end = f"{start.parent}/checkpoint.pt"
    assert FakeTrainer.log == [("init", e.savedir, 40, str(start)), ("train", 40, 1e-3, pytest.approx(1e-4)), ("save", end)]
    assert not start.exists()  # taken
    assert calls == [("pca", "enc", 140, "hemibrain"), ("probe", end, True)]
    assert phases == [("scores", replace(e, init_from=end, score=True))]
    calls.clear(), phases.clear()
    start.write_text("start")
    experiment.branch(replace(e, train=replace(TRUNK, branch_eval="unetr")))
    assert calls[1] == ("finetune", end, True) and isinstance(phases[0][1], specs.Finetune)  # the variant: a UNETR, scored all the same


def test_scores_keeps_a_branch_only_if_it_is_the_trunks_best(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trunk = Path("outdir/e00/e01-test/a")
    pqs = {"step_0000100": 0.2, "step_0000200": 0.5, "step_0000300": 0.3}
    monkeypatch.setattr(jobs, "code_provenance", lambda: {"commit_id": "x"})
    for step, pq in pqs.items():
        d = trunk / step
        d.mkdir(parents=True)
        (d / "checkpoint.pt").write_text(step)
        e = specs.Eval(savedir=f"{d}/", train=TRUNK, init_from=str(d / "checkpoint.pt"))
        monkeypatch.setattr(experiment, "mia_eval", lambda par, pq=pq: pq)
        experiment.scores(e)
    assert (trunk / "best.pt").read_text() == "step_0000200"
    assert json.loads((trunk / "best.json").read_text())["step"] == 200
    assert not list(trunk.glob("step_*/checkpoint.pt"))  # the losers' weights are deleted


def test_every_phase_declares_known_hardware():
    assert {"benchmark", "trunk", "branch", "scores"} <= set(jobs.TASKS)
    assert {t.needs for t in jobs.TASKS.values()} <= {"gpus", "cpu"}
    assert set(jobs.BOARDS) == set(typing.get_args(typing.get_type_hints(specs.Train)["board"]))  # Train.board names a leaderboard


def test_go_refuses_a_phase_for_a_spec_it_does_not_take(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jobs, "bsub", lambda *a, **kw: pytest.fail("submitted"))
    with pytest.raises(AssertionError, match="phase 'branch' takes"):
        jobs.go(experiment.branch, jobs.Train(savedir="outdir/e00/x/a/"))
    with pytest.raises(AssertionError, match="phase 'trunk' takes"):
        jobs.go(experiment.trunk, jobs.Eval(savedir="outdir/e00/x/a/", train=jobs.Train(), init_from="random"))
    assert not (tmp_path / "outdir").exists()  # refused before writing its spec


def test_go_writes_the_spec_its_job_loads_and_asks_for_the_hardware_the_phase_needs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    submitted = []
    monkeypatch.setattr(jobs, "bsub", lambda savedir, job, minutes, n_gpus, queue, cmd, **kw: submitted.append((job, minutes, n_gpus, queue, cmd)))
    trunk = jobs.Train(savedir="outdir/e00/x/a/", queue="gpu_b300", n_gpus=2, hours=3, n_workers=11)
    e = jobs.Eval(savedir="outdir/e00/x/a/step_0000100/", train=trunk, init_from="outdir/e00/x/a/step_0000100/checkpoint.pt")
    jobs.go(experiment.trunk, trunk)
    jobs.go(experiment.branch, e)  # the trunk's queue and GPUs
    jobs.go(experiment.scores, e)
    assert submitted == [("trunk", 180, 2, "gpu_b300", "torchrun --standalone --nproc_per_node=2 {code}/experiment.py trunk outdir/e00/x/a/train.json"),
                         ("branch", 120, 2, "gpu_b300", "torchrun --standalone --nproc_per_node=2 {code}/experiment.py branch outdir/e00/x/a/step_0000100/eval.json"),
                         ("scores", jobs.SCORE_MINUTES, 0, jobs.SCORE_QUEUE, "python {code}/experiment.py scores outdir/e00/x/a/step_0000100/eval.json")]
    assert jobs.load("outdir/e00/x/a/train.json", jobs.Train) == trunk  # what the job runs: the same spec
    assert jobs.load("outdir/e00/x/a/step_0000100/eval.json", jobs.Eval) == e
    spec = json.loads(Path("outdir/e00/x/a/step_0000100/eval.json").read_text())
    Path("bad.json").write_text(json.dumps({**spec, "score": "maybe"}))
    with pytest.raises(Exception, match="score"):  # validated: a typo fails at load, before any GPU work
        jobs.load("bad.json", jobs.Eval)
    Path("extra.json").write_text(json.dumps({**spec, "steps": 3}))
    with pytest.raises(AssertionError, match="doesn't round-trip"):  # so does a field the spec doesn't have
        jobs.load("extra.json", jobs.Eval)


def test_the_experiment_file_names_runs_that_load_as_a_job_loads_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, r in experiment.EXPERIMENT.roots().items():  # a job validates its json (pydantic's Literals), which building the dataclass doesn't
        assert r.savedir == f"outdir/{experiment.EXPERIMENT.id}/{name}/"
        assert jobs.parse(type(r), asdict(r)) == r, name
        specs.model_config(r)  # its model builds on its encoder's patch
        assert r.hours * 60 > 0 and r.segments * r.segment_steps > 0
    assert experiment.EXPERIMENT.root is experiment.trunk
    jobs.check_experiment(experiment.EXPERIMENT)


def sweep(monkeypatch, id, init_from=""):
    """The Experiment id of 3 trunks a, b and c (from init_from/<run>/checkpoint.pt if set), start() recording their names, and no LSF."""
    started = []
    run = lambda n: jobs.Train(init_from=f"{init_from}/{n}/checkpoint.pt" if init_from else "")
    monkeypatch.setattr(jobs, "assert_committed", lambda: None)
    monkeypatch.setattr(jobs, "start", lambda exp, p: started.append(Path(p.savedir).name))
    monkeypatch.setattr(jobs, "bsub", lambda *a, **kw: None)
    return Experiment(id, experiment.trunk, {n: run(n) for n in "abc"}), started


def test_start_empties_the_savedir_and_submits_the_first_phase(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gone = []
    monkeypatch.setattr(jobs, "trash", lambda savedir: gone.append(("trash", savedir)))
    monkeypatch.setattr(jobs, "go", lambda fn, p: gone.append((fn.__name__, p.savedir)))
    exp = Experiment("e00/e01-test", experiment.trunk, {"a": jobs.Train()})
    jobs.start(exp, exp.roots()["a"])
    assert gone == [("trash", "outdir/e00/e01-test/a/"), ("trunk", "outdir/e00/e01-test/a/")]


def test_submitall_starts_each_run_not_submitted_before(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp, started = sweep(monkeypatch, "e00/e01-test")
    jobs.dump(exp.roots()["b"])  # submitted before
    jobs.submitall(exp)
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
        jobs.submitall(exp)
    assert started == []  # refused before any job


def test_submitall_takes_a_child_of_its_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for s in ("e01-train", "e01-01-pca", "e01-02-probe"):  # e01-02-probe: this experiment's own dir, from a first submit
        (tmp_path / "outdir/e00" / s).mkdir(parents=True)
    exp, started = sweep(monkeypatch, "e00/e01-02-probe", init_from="outdir/e00/e01-train")
    jobs.submitall(exp)
    assert started == ["a", "b", "c"] and jobs.experiment_id("e01-02-probe") == "e01-02"


def test_lsf_restarts_a_run_at_its_first_phase_and_redoes_any_other_in_place(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp, started = sweep(monkeypatch, "e00/e01-test")
    exp.runs["b"].n_workers = 4  # the default's 16 leave no core of CPUS_PER_GPU for the training process
    gone = []
    monkeypatch.setattr(jobs, "go", lambda fn, job: gone.append((fn.__name__, job.savedir)))
    jobs.lsf(exp, "trunk", "b")
    assert started == ["b"] and gone == []  # start() went
    jobs.lsf(exp, "benchmark", "b")
    e = jobs.Eval(savedir="outdir/e00/e01-test/b/step_0000100/", train=exp.roots()["b"], init_from="random")
    jobs.dump(e)
    jobs.lsf(exp, "scores", f"{e.savedir}eval.json")
    assert gone == [("benchmark", "outdir/e00/e01-test/b/"), ("scores", e.savedir)]
