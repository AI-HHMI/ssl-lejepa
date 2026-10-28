"""What an experiment.py and its tasks.py are made of, and the LSF code that runs it. An experiment is a table of runs (Experiment), each a Trunk spec,
and the tasks that process each run: functions in tasks.py (trunk, branch, score), each declared with @task (its spec class,
defined beside it, and its hardware) and each calling the next with go(), which submits it as an LSF job: a job is a task
submitted. A spec has a savedir; a gpus task says which Hardware (queue, GPUs) its spec asks LSF for. A task takes its spec, or the path of the
spec's json (`branch outdir/.../branch.json`), so one can be redone by hand: `lsf <task> <run or spec file>`. Depends on lib/, so a
task runs only from its experiment's commit. `python experiment.py` or `-h` lists the commands and tasks (main)."""

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
from typing import Any, Callable, NamedTuple

from lib import training
from lib.training import Hardware, make_model
from lib.util import *

CODE = Path(__file__).resolve().parent.parent  # the code root: the shared checkout, or the .tmpcode/ snapshot a job runs
# Code copied into .tmpcode/<experiment>/ at submission (bsub); jobs import only from these. Add files here if the experiment
# starts depending on others. Dependencies (pyproject.toml / uv.lock) are not frozen.
SNAPSHOT_PATHS = ["experiment.py", "tasks.py", "lib", "mia_evals"]
CPUS_PER_GPU = 12  # LSF slots per GPU: 8 GPUs -> all 96 cores; training processes need cores beyond the data workers
# Whole nodes (Optim.batch_size and Hardware.n_workers are per GPU); an H200 node has one GPU taken by someone else's job.
H200 = Hardware(queue="gpu_h200", n_gpus=7, n_workers=CPUS_PER_GPU - 1)
B300 = Hardware(queue="gpu_b300", n_gpus=8, n_workers=CPUS_PER_GPU - 1)
# mia-evals scoring (mia_eval): CPU only. mia-evals: ~1 h and ~200 GB peak for the mutex watershed at 896^3, "give it a
# whole node". The shortest CPU queue; check its run limit and memory per slot with `bqueues -l short`.
SCORE_QUEUE, SCORE_SLOTS, SCORE_MINUTES = "local", 16, 4*60
BOARDS = {"neuron-instance": "mia_evals/gary_comparison_neuron_instance/mws.toml"}  # Trunk.board: ours, truth_kind = "instances"
# mia_eval() calls mia-evals as a library (a path dependency on the sister checkout, see pyproject.toml); its commit is recorded
# in each score's provenance (mia_evals_commit).


@dataclass
class Experiment:
    """An experiment (README: experiment ids): the task that starts each run, and its runs, each a Trunk whose savedir is
    outdir/<id>/<name>/."""
    id: str  # <series>/e<NN>[-<NN>...]-<slug>, e.g. e00/e15-02-pca
    task: Callable  # the first task of every run (one taking its Trunk)
    runs: dict[str, Any]  # its Trunks (experiment.py's), by name: [a-z][a-z0-9_-]*

    def roots(self) -> dict[str, Any]:
        assert all(re.fullmatch(r"[a-z][a-z0-9_-]*", name) for name in self.runs), f"run names are lowercase words: {list(self.runs)}"
        return {name: replace(spec, savedir=f"outdir/{self.id}/{name}/") for name, spec in self.runs.items()}


def experiment_dir(savedir: str) -> str:
    """The experiment of a run dir, <series>/<experiment>: outdir/e00/e12-x/mae/ -> e00/e12-x."""
    return "/".join(Path(savedir).parts[1:3])


class Task(NamedTuple):
    """A task: the function, the spec class it takes, and the hardware it needs."""
    fn: Callable
    spec: type  # the spec it takes (a dataclass with a savedir)
    needs: str  # "gpus": the Train's own, torchrun; "cpu": SCORE_QUEUE's slots
    minutes: int | Callable[[Any], int]  # the LSF wall limit (of the spec)
    hardware: Callable[[Any], Hardware] | None  # the Hardware (queue, GPUs) a "gpus" task's spec says; a "cpu" task has none

TASKS: dict[str, Task] = {}  # every task, in the order declared: submit() reads it, `-h` prints it

def task(spec: type, needs: str, minutes, hardware=None):
    """Declare a function a task of the pipeline (TASKS) and an entrypoint of experiment.py, taking a `spec`."""
    assert (needs == "gpus") == (hardware is not None), f"{needs!r} task {spec.__name__}: hardware is for (and only for) gpus tasks"
    def declare(fn):
        TASKS[fn.__name__] = Task(fn, spec, needs, minutes, hardware)
        return fn
    return declare


def check_snapshot():
    """LSF jobs must run a code snapshot (bsub): the shared checkout's lib/ may be another sweep's by now."""
    b1 = "LSB_JOBID" in os.environ
    b2 = ".tmpcode" not in Path(__file__).resolve().parts
    assert not (b1 and b2), f"LSF job running {__file__} from the shared checkout; submit via go() (code snapshot)"


def replay_bad_batch(exp: Experiment, path: str):
    """Replay a skipped (non-finite gradient) training step from its bad_batch_*.pt dump: lib.training.replay_bad_batch. Like
    training, replay depends on lib/: only for dumps of this commit's own experiment (replaylsf submits it)."""
    import torch
    ours = Path("outdir", exp.id).resolve()
    assert Path(path).resolve().is_relative_to(ours), f"{path} isn't from this commit's experiment: replay it from its experiment's commit"
    dump = torch.load(path, map_location="cpu", weights_only=False)
    return training.replay_bad_batch(dump)

def mia_eval(par) -> float:
    """Score par's branch's affinities (probe/{fit,test}, written by its probe or finetune) as mia-evals' neuron-instance task on its trunk's board
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
    assert (savedir / "probe/test").is_dir() and (savedir / "probe/fit").is_dir(), f"no probe artifacts in {savedir}: probe or finetune first"
    (savedir / "resolved_config.json").write_text(json.dumps(asdict(par)) + "\n")
    (savedir / "git_commit.txt").write_text(code_provenance()["commit_id"] + "\n")
    config = CODE / BOARDS[par.branch.trunk.board]  # the snapshot's copy when run as a job
    root = savedir.resolve()
    evaluate.cmd_score(Namespace(
        config=config, test=root / "probe/test", val=root / "probe/fit", leaderboard=root / "mia_evals", run_dir=root,
        no_scored=True,  # don't keep the 5.8 GB post-processed labelling per run
        scored_out=None, scratch=None))
    shutil.rmtree(savedir / "probe")  # the artifacts are scored (the record is in mia_evals/): GBs per run, not kept
    record_file = sorted(savedir.glob("mia_evals/*/records/*.json"))[-1]
    return json.loads(record_file.read_text())["scores"]["voxel_instance"]["pq"]

def keep_if_best(par, score: float):
    """Keep par's branch's weights (savedir/checkpoint.pt, its end) as its trunk's best.pt if score beats the best so far (best.json, in the
    trunk's dir); else delete them. The scores of one trunk's branches finish in any order, so the compare-and-move is under a lock
    held only for it (a rename, not a copy)."""
    trunk, weights = Path(par.savedir).parent, Path(par.savedir) / "checkpoint.pt"
    step = int(Path(par.savedir).name.split("_")[1])
    with open(trunk / "best.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # released when the file closes
        best = json.loads((trunk / "best.json").read_text())["score"] if (trunk / "best.json").is_file() else float("-inf")
        if score > best:
            weights.replace(trunk / "best.pt")
            (trunk / "best.json").write_text(json.dumps({"score": score, "step": step, "board": par.branch.trunk.board, "from": par.savedir}) + "\n")
        else:
            weights.unlink()
    print(f"{par.savedir}: score {score:.4f} {'beats' if score > best else 'does not beat'} the best so far, {best:.4f}", flush=True)

def load_checkpoint(train: Train, init_from: str):
    """(model on the GPU if any, step) for inference or finetuning: the weights init_from names, in train's architecture.

    A checkpoint .pt, loaded strictly into train's architecture so an incompatible one fails loudly (Conv3d-era patch embeds
    load, see PatchEmbed3d). Inference depends on lib/, so like training it only runs for this commit's own experiment: for an
    older sweep, go back to its commit.
    "random": untrained weights, seeded (the baseline every trained encoder must beat). Step 0.
    """
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if init_from == "random":
        torch.manual_seed(0)
        return make_model(train).to(device).eval(), 0
    assert Path(init_from).is_file(), f"no checkpoint at {init_from!r}: a checkpoint .pt or \"random\""
    ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
    model = make_model(train)
    model.load_state_dict(ckpt["model"])  # strict: architecture drift fails loudly
    return model.to(device).eval(), ckpt["step"]

def record(par, fn: str):
    """Append one row to par.savedir/runs.json: which task ran (fn), its spec par (a dataclass: params), code_provenance(),
    argv and LSF job. Written first thing, so even a run that crashes before any results describes itself.
    Repro for the dir's artifacts: that commit + argv. The command that submitted the job is in
    outdir/_log/commands.jsonl (log_command)."""
    row = {"fn": fn, "params": asdict(par), **code_provenance(), "argv": sys.argv, "lsf_job": os.environ.get("LSB_JOBID")}
    Path(par.savedir).mkdir(parents=True, exist_ok=True)
    with open(Path(par.savedir) / "runs.json", "a") as f:
        f.write(json.dumps(row) + "\n")

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

def go(fn: Callable, job):
    """Run the task fn on the spec job as its own LSF job: job written to its savedir (dump: trunk.json, branch.json or
    score.json), and `experiment.py <fn> <that file>` submitted (bsub), which loads it. Hardware by the task's needs: its
    Train's own GPUs (torchrun), or a CPU job on SCORE_QUEUE. No commit check: callers from the shared checkout (start, lsf)
    assert_committed() first; a job calling it is committed code."""
    name = fn.__name__
    t = TASKS[name]
    assert isinstance(job, t.spec), f"{job.savedir}: task {name!r} takes a {t.spec}, not a {type(job).__name__}: {list(TASKS)}"
    h = t.hardware(job) if t.hardware else None
    assert h is None or h.n_workers + 1 <= CPUS_PER_GPU, f"n_workers={h.n_workers} leaves no core for the training process"
    spec = dump(job)
    minutes = t.minutes(job) if callable(t.minutes) else t.minutes
    if h:
        bsub(job.savedir, name, minutes, h.n_gpus, h.queue, f"torchrun --standalone --nproc_per_node={h.n_gpus} {{code}}/experiment.py {name} {spec}")
    else:
        bsub(job.savedir, name, minutes, 0, SCORE_QUEUE, f"python {{code}}/experiment.py {name} {spec}", slots=SCORE_SLOTS)

def start(exp: Experiment, run):
    """Submit run fresh: its savedir emptied (old contents -> outdir/.trash/), then the experiment's first task."""
    trash(run.savedir)
    go(exp.task, run)


def lsf(exp: Experiment, fn: str, target: str):
    """Redo one task on LSF, of the run named target or of the spec file target names (trunk.json, branch.json or score.json
    of any run or branch, e.g. outdir/.../mae/step_0010000/score.json): `lsf score <that>` after changing the board. The
    experiment's first task on a run restarts it fresh (start); the others run on the spec in place."""
    assert_committed()
    roots = exp.roots()
    assert fn in TASKS, f"no task {fn!r}: {list(TASKS)}"
    p = roots[target] if target in roots else load(target, {t.spec.__name__.lower(): t.spec for t in TASKS.values()}[Path(target).stem])  # a spec file's name (spec_path) -> its class
    if TASKS[fn].fn is exp.task:
        start(exp, p)
    else:
        go(TASKS[fn].fn, p)

def replaylsf(exp: Experiment, name: str):
    """replay_bad_batch on LSF for the run name's first bad_batch dump. CUDA_LAUNCH_BLOCKING names a crashing kernel."""
    run = exp.roots()[name]
    dumps = sorted(Path(run.savedir).glob("bad_batch_*.pt"))
    assert dumps, f"no bad_batch_*.pt in {run.savedir}"
    assert_committed()
    bsub(run.savedir, "replay", 30, 1, run.train.hardware.queue, f"env CUDA_LAUNCH_BLOCKING=1 python {{code}}/experiment.py replay_bad_batch {dumps[0]}")

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
    already submitted is left alone (`lsf <first task> <name>` restarts one fresh)."""
    check_experiment(exp)
    assert_committed()
    for r in exp.roots().values():
        if not spec_path(r).is_file():
            start(exp, r)


COMMANDS = [replay_bad_batch, lsf, replaylsf, submitall]  # take the experiment first, bound by main


def main(exp: Experiment):
    """The command line of experiment.py: `<task> <spec json>` (what an LSF job runs), or a command on the experiment's runs."""
    names = {**{name: t.fn for name, t in TASKS.items()}, **{f.__name__: partial(f, exp) for f in COMMANDS}}
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        for name, f in names.items():
            doc = (getattr(f, "func", f).__doc__ or "").strip().splitlines()[0]
            print(f"  {name}{str(inspect.signature(f)).replace('exp: Experiment, ', '')}  {doc}")
            if name in TASKS:
                t = TASKS[name]
                print(f"      needs: {t.needs}")
        sys.exit()
    assert sys.argv[1] in names, f"unknown command {sys.argv[1]!r}: {', '.join(names)}"
    log_command(sys.argv)  # commands issued outside jobs -> outdir/_log/commands.jsonl; jobs use runs.json
    names[sys.argv[1]](*sys.argv[2:])
