"""What an experiment.py is made of, and the LSF code that runs it. An experiment is a table of runs (Experiment) and the phases
that process each run: functions in experiment.py, each declared with @task (its spec, hardware, the data it uses and what it
writes) and each calling the next with go(), which submits it as an LSF job. A phase takes its spec, or the path of the spec's
json (`branch outdir/.../eval.json`), so one can be redone by hand: `lsf <phase> <run or spec file>`. Depends on lib/, so a
phase runs only from its experiment's commit. `python experiment.py` or `-h` lists the commands and phases (main)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import fcntl
from functools import partial
import inspect
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Callable, NamedTuple

from lib import training
from lib.specs import Eval, Finetune, Spec, Train, record
from lib.util import *

CODE = Path(__file__).resolve().parent.parent  # the code root: the shared checkout, or the .tmpcode/ snapshot a job runs
# Code copied into .tmpcode/<experiment>/ at submission (bsub); jobs import only from these. Add files here if the experiment
# starts depending on others. Dependencies (pyproject.toml / uv.lock) are not frozen.
SNAPSHOT_PATHS = ["experiment.py", "lib", "mia_evals"]
CPUS_PER_GPU = 12  # LSF slots per GPU: 8 GPUs -> all 96 cores; training processes need cores beyond the data workers
H200 = dict(queue="gpu_h200", n_gpus=7)  # whole nodes (Train.batch_size and n_workers are per GPU); an H200 node has one GPU taken by someone else's job
B300 = dict(queue="gpu_b300", n_gpus=8)
# mia-evals scoring (mia_eval): CPU only. mia-evals: ~1 h and ~200 GB peak for the mutex watershed at 896^3, "give it a
# whole node". The shortest CPU queue; check its run limit and memory per slot with `bqueues -l short`.
SCORE_QUEUE, SCORE_SLOTS, SCORE_MINUTES = "local", 16, 4*60
BOARDS = {"neuron-instance": "mia_evals/gary_comparison_neuron_instance/mws.toml"}  # Train.board: ours, truth_kind = "instances"
# mia_eval() calls mia-evals as a library (a path dependency on the sister checkout, see pyproject.toml); its commit is recorded
# in each score's provenance (mia_evals_commit).


@dataclass
class Experiment:
    """An experiment (README: experiment ids): the phase that starts each run, and its runs, each a Train whose savedir is
    outdir/<id>/<name>/."""
    id: str  # <series>/e<NN>[-<NN>...]-<slug>, e.g. e00/e15-02-pca
    root: Callable  # the first phase of every run (a task taking a Train)
    runs: dict[str, Train]  # by name: [a-z][a-z0-9_-]*

    def roots(self) -> dict[str, Train]:
        assert all(re.fullmatch(r"[a-z][a-z0-9_-]*", name) for name in self.runs), f"run names are lowercase words: {list(self.runs)}"
        return {name: replace(spec, savedir=f"outdir/{self.id}/{name}/") for name, spec in self.runs.items()}


def experiment_dir(savedir: str) -> str:
    """The experiment of a run dir, <series>/<experiment>: outdir/e00/e12-x/mae/ -> e00/e12-x."""
    return "/".join(Path(savedir).parts[1:3])


class Use(NamedTuple):
    """A dataset volume a phase reads, by its role (a field of lib.data.Dataset)."""
    role: str  # train, val, fit, test or tune
    labels: bool  # its labels are read (else its images only)

class Task(NamedTuple):
    """A phase: the function, the specs it takes, the hardware it needs, the data it uses and what it writes."""
    fn: Callable
    spec: type | tuple[type, ...]  # the spec(s) it takes
    needs: str  # "gpus": the Train's own, torchrun; "cpu": SCORE_QUEUE's slots
    minutes: int | Callable[[Spec], int]  # the LSF wall limit (of the spec)
    data: tuple[Use, ...]
    writes: tuple[str, ...]  # into its savedir

TRAIN, VAL, FIT, TEST, TUNE = Use("train", False), Use("val", False), Use("fit", True), Use("test", True), Use("tune", True)
TUNE_IMAGES = TUNE._replace(labels=False)  # a branch writes the tune volume's predictions; scores reads its labels

TASKS: dict[str, Task] = {}  # every phase, in the order declared: submit() reads it, `-h` prints it

def task(spec, needs: str, minutes, data: tuple[Use, ...], writes: tuple[str, ...]):
    """Declare a function a phase of the pipeline (TASKS) and an entrypoint of experiment.py."""
    def declare(fn):
        TASKS[fn.__name__] = Task(fn, spec, needs, minutes, data, writes)
        return fn
    return declare


def check_snapshot():
    """LSF jobs must run a code snapshot (bsub): the shared checkout's lib/ may be another sweep's by now."""
    b1 = "LSB_JOBID" in os.environ
    b2 = ".tmpcode" not in Path(__file__).resolve().parts
    assert not (b1 and b2), f"LSF job running {__file__} from the shared checkout; submit via go() (code snapshot)"


@task(Train, "gpus", 15, (TRAIN,), ("performance.json", "profile.out"))
def benchmark(job: Train | str):
    """Throughput and a profiler trace of the spec job's training step (or the train.json's it names), with no training run:
    lib.training.benchmark. Every DDP rank."""
    check_snapshot()
    par = load(job, Train)
    if int(os.environ.get("RANK", 0)) == 0:
        record(par, "benchmark")
    training.benchmark(par)

def replay_bad_batch(exp: Experiment, path: str):
    """Replay a skipped (non-finite gradient) training step from its bad_batch_*.pt dump: lib.training.replay_bad_batch. Like
    training, replay depends on lib/: only for dumps of this commit's own experiment (replaylsf submits it)."""
    import torch
    ours = Path("outdir", exp.id).resolve()
    assert Path(path).resolve().is_relative_to(ours), f"{path} isn't from this commit's experiment: replay it from its experiment's commit"
    dump = torch.load(path, map_location="cpu", weights_only=False)
    record(parse(Train, dump["params"]), "replay_bad_batch")
    return training.replay_bad_batch(dump)

def mia_eval(par: Eval | Finetune) -> float:
    """Score par's affinities (probe/{fit,test}, written by its probe or finetune) as mia-evals' neuron-instance task on par's board
    (mutex watershed + size filter, fitted on the fit block, reported on the test block); panoptic quality, which ranks, comes
    back (VOI and ARE are in the record).

    Writes the record into the run's own dir, savedir/mia_evals/<task>/records/*.json (pull.sh brings it down;
    analysis_plots.mia_evals_table reads it), not mia-evals' repo. resolved_config.json (params) and git_commit.txt (the
    commit that ran) go in the savedir first: mia-evals copies them into the record from run_dir.
    Calls mia_evals.evaluate.cmd_score (what `mia-evals score` runs) with this commit's board config. Deletes savedir/probe/
    (the artifacts it read) once the record is written.
    """
    from argparse import Namespace

    from mia_evals import evaluate
    savedir = Path(par.savedir)
    assert (savedir / "probe/test").is_dir() and (savedir / "probe/fit").is_dir(), f"no probe artifacts in {savedir}: probe or finetune first with score=True"
    (savedir / "resolved_config.json").write_text(json.dumps(asdict(par)) + "\n")
    (savedir / "git_commit.txt").write_text(code_provenance()["commit_id"] + "\n")
    config = CODE / BOARDS[par.train.board]  # the snapshot's copy when run as a job
    root = savedir.resolve()
    evaluate.cmd_score(Namespace(
        config=config, test=root / "probe/test", val=root / "probe/fit", leaderboard=root / "mia_evals", run_dir=root,
        no_scored=True,  # don't keep the 5.8 GB post-processed labelling per run
        scored_out=None, scratch=None))
    shutil.rmtree(savedir / "probe")  # the artifacts are scored (the record is in mia_evals/): GBs per run, not kept
    record_file = sorted(savedir.glob("mia_evals/*/records/*.json"))[-1]
    return json.loads(record_file.read_text())["scores"]["voxel_instance"]["pq"]

def keep_if_best(par: Eval | Finetune, score: float):
    """Keep par's weights (par.init_from, a branch's end) as its trunk's best.pt if score beats the best so far (best.json, in the
    trunk's dir); else delete them. The scores of one trunk's branches finish in any order, so the compare-and-move is under a lock
    held only for it (a rename, not a copy)."""
    trunk, weights = Path(par.savedir).parent, Path(par.init_from)
    step = int(Path(par.savedir).name.split("_")[1])
    with open(trunk / "best.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # released when the file closes
        best = json.loads((trunk / "best.json").read_text())["score"] if (trunk / "best.json").is_file() else float("-inf")
        if score > best:
            weights.replace(trunk / "best.pt")
            (trunk / "best.json").write_text(json.dumps({"score": score, "step": step, "board": par.train.board, "from": par.savedir}) + "\n")
        else:
            weights.unlink()
    print(f"{par.savedir}: score {score:.4f} {'beats' if score > best else 'does not beat'} the best so far, {best:.4f}", flush=True)

def job_name(savedir: str, job: str) -> str:
    """The LSF name of the run dir savedir's job: <series>-<experiment>-<run>-<job>, e.g. e00-e12-cellmap-mae-run, so bjobs shows which is which."""
    return "-".join(Path(savedir).parts[1:]) + f"-{job}"

def bsub(savedir: str, job: str, minutes: int, n_gpus: int, queue: str, cmd: str, slots: int | None = None):
    """Submit `uv run <cmd>` for the run dir savedir to LSF as job job_name(savedir, job), logging to savedir/job_<job>_%J.log,
    on queue with CPUS_PER_GPU slots per GPU (a CPU job, n_gpus = 0, names its slots).

    {code} in cmd is this commit's code snapshot (.tmpcode/<sweep>/), which the job runs instead of the shared
    checkout: that may have moved on (another sweep pushed) by the time the job starts. Python puts the script's
    dir first on sys.path, so `lib` comes from the snapshot too; cwd stays the repo root, so outdir/ and data paths
    resolve as before. Callers from the shared checkout assert_committed() first.
    """
    import subprocess
    name = job_name(savedir, job)
    # A job submitting a job hands on the snapshot it runs itself, not the shared checkout's current state.
    code = CODE if ".tmpcode" in CODE.parts else snapshot(SNAPSHOT_PATHS, Path(".tmpcode") / experiment_dir(savedir).replace("/", "-"))
    gpu = f'-gpu "num={n_gpus}:mode=exclusive_process"' if n_gpus else ""
    full = f""" bsub -J {name} \
        -W {minutes // 60}:{minutes % 60:02d} \
        -P miaai \
        -n {slots or n_gpus * CPUS_PER_GPU} \
        -R "span[hosts=1]" {gpu} \
        -q {queue} \
        -o {savedir}/job_{job}_%J.log \
        uv run {cmd.format(code=code)}
        """
    subprocess.run(full, shell=True, stdin=subprocess.DEVNULL, check=True)
    print(f"Submitted {name} (code {code}) to LSF.")

def go(fn: Callable, job: Spec):
    """Run the phase fn on the spec job as its own LSF job: job written to its savedir (dump: train.json, eval.json or
    finetune.json), and `experiment.py <fn> <that file>` submitted (bsub), which loads it. Hardware by the phase's needs: the
    Train's own GPUs (torchrun), or a CPU job on SCORE_QUEUE. No commit check: callers from the shared checkout (start, lsf)
    assert_committed() first; a job calling it is committed code."""
    name = fn.__name__
    t = TASKS[name]
    assert isinstance(job, t.spec), f"{job.savedir}: phase {name!r} takes a {t.spec}, not a {type(job).__name__}: {list(TASKS)}"
    train = job if isinstance(job, Train) else job.train
    assert t.needs != "gpus" or train.n_workers + 1 <= CPUS_PER_GPU, f"n_workers={train.n_workers} leaves no core for the training process"
    spec = dump(job)
    minutes = t.minutes(job) if callable(t.minutes) else t.minutes
    if t.needs == "gpus":
        bsub(job.savedir, name, minutes, train.n_gpus, train.queue, f"torchrun --standalone --nproc_per_node={train.n_gpus} {{code}}/experiment.py {name} {spec}")
    else:
        bsub(job.savedir, name, minutes, 0, SCORE_QUEUE, f"python {{code}}/experiment.py {name} {spec}", slots=SCORE_SLOTS)

def start(exp: Experiment, p: Train):
    """Submit p fresh: its savedir emptied (old contents -> outdir/.trash/), then the experiment's first phase."""
    trash(p.savedir)
    go(exp.root, p)


def lsf(exp: Experiment, fn: str, target: str):
    """Redo one phase on LSF, of the run named target or of the spec file target names (train.json, eval.json or
    finetune.json of any run or branch, e.g. outdir/.../mae/step_0010000/eval.json): `lsf scores <that>` after changing the
    board. The experiment's first phase on a run restarts it fresh (start); the others run on the spec in place."""
    assert_committed()
    roots = exp.roots()
    phase = TASKS[fn].fn if fn in TASKS else None
    assert phase, f"no phase {fn!r}: {list(TASKS)}"
    p = roots[target] if target in roots else load(target, Eval if target.endswith("eval.json") else Finetune if target.endswith("finetune.json") else Train)
    if phase is exp.root:
        start(exp, p)
    else:
        go(phase, p)

def replaylsf(exp: Experiment, name: str):
    """replay_bad_batch on LSF for the run name's first bad_batch dump. CUDA_LAUNCH_BLOCKING names a crashing kernel."""
    par = exp.roots()[name]
    assert isinstance(par, Train), f"{par.savedir}: an eval trains nothing, so it has no bad batches"
    dumps = sorted(Path(par.savedir).glob("bad_batch_*.pt"))
    assert dumps, f"no bad_batch_*.pt in {par.savedir}"
    assert_committed()
    bsub(par.savedir, "replay", 30, 1, par.queue, f"env CUDA_LAUNCH_BLOCKING=1 python {{code}}/experiment.py replay_bad_batch {dumps[0]}")

def experiment_id(name: str) -> str | None:
    """The id of a sweep dir name under outdir/<series>/: 'e15-02-pca' -> 'e15-02'. None for one from before
    numbering (e.g. 'viewsizes-v2')."""
    m = re.fullmatch(r"(e\d\d(?:-\d\d)*)-[a-z][a-z0-9-]*", name)
    return m[1] if m else None

def check_experiment(exp: Experiment):
    """The experiment's id rules (README: experiment ids), checked against outdir/, the cluster's source of truth: the id is
    well formed and no other sweep dir has it; a child (e15-02) builds on its parent (e15), i.e. every init_from points into
    outdir/<series>/e15-*/; a root (e15) builds on no numbered experiment. One that depends on two experiments can't be named,
    by design."""
    series, name = exp.id.split("/")
    eid = experiment_id(name)
    assert eid, f"{exp.id!r}: want e<series>/e<NN>[-<NN>...]-<slug>, e.g. e00/e15-02-pca"
    sweeps = [d.name for d in Path("outdir", series).glob("*")]
    taken = [s for s in sweeps if experiment_id(s) == eid and s != name]
    assert not taken, f"{exp.id}: id {eid} is taken by {series}/{taken[0]}"
    sources = [p.init_from for p in exp.roots().values()]
    deps = sorted({Path(s).parts[2] for s in sources if s not in ("", "random")} - {name})  # outdir/<series>/<experiment>/<run>; not our own
    parent = eid.rsplit("-", 1)[0] if "-" in eid else None
    if parent:
        parents = [s for s in sweeps if experiment_id(s) == parent]
        assert parents, f"{exp.id} builds on {series}/{parent}, which doesn't exist in outdir/"
        assert deps == parents, f"{exp.id} builds on {series}/{parents[0]}, but its init_from read {deps}"
    else:
        numbered = [d for d in deps if experiment_id(d)]
        assert not numbered, f"{exp.id} builds on {series}/{numbered[0]}: name it {experiment_id(numbered[0])}-<NN>-<slug>"

def submitall(exp: Experiment):
    """Submit the experiment (README: one experiment, one entrypoint): every run whose spec isn't in its savedir yet, i.e. not
    submitted before (whether it finished, is still going, or failed). Idempotent: run it again after adding runs; a run
    already submitted is left alone (`lsf <first phase> <name>` restarts one fresh)."""
    check_experiment(exp)
    assert_committed()
    for r in exp.roots().values():
        if not spec_path(r).is_file():
            start(exp, r)


COMMANDS = [replay_bad_batch, lsf, replaylsf, submitall]  # take the experiment first, bound by main


def main(exp: Experiment):
    """The command line of experiment.py: `<phase> <spec json>` (what an LSF job runs), or a command on the experiment's runs."""
    names = {**{name: t.fn for name, t in TASKS.items()}, **{f.__name__: partial(f, exp) for f in COMMANDS}}
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        for name, f in names.items():
            doc = (getattr(f, "func", f).__doc__ or "").strip().splitlines()[0]
            print(f"  {name}{str(inspect.signature(f)).replace('exp: Experiment, ', '')}  {doc}")
            if name in TASKS:
                t = TASKS[name]
                uses = [f"{u.role}, {'labels' if u.labels else 'images'}" for u in t.data]
                print(f"      needs: {t.needs}\n      data: {'; '.join(uses)}\n      writes: {', '.join(t.writes)}")
        sys.exit()
    assert sys.argv[1] in names, f"unknown command {sys.argv[1]!r}: {', '.join(names)}"
    log_command(sys.argv)  # commands issued outside jobs -> outdir/_log/commands.jsonl; jobs use runs.json
    names[sys.argv[1]](*sys.argv[2:])
