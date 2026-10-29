"""What an experiment.py and its tasks.py are made of, and the LSF code that runs it. An experiment is a table of runs (Experiment),
each a Trunk, and the tasks that process each run: plain functions in tasks.py (trunk, anneal, finetune, score) with JSON-able arguments
(dataclasses, strings), each calling the next with go(), which submits it as an LSF job: a job is a task submitted. go() writes the
arguments and the job's resources to <savedir>/<task>.json and runs `experiment.py <task> <that file>`, which loads the arguments by
the task's type hints and calls it, so a task runs the same in-process, called with its arguments, and as a job; `resubmit_task <task> <run
or job file>` redoes one by hand. Depends on lib/, so a task runs only from its experiment's commit. `python experiment.py` or `-h`
lists the commands and tasks (main)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from functools import partial
import inspect
import json
import os
import re
import sys
from pathlib import Path
import typing
from typing import Any, Callable

from lib import training
from lib.training import Hardware
from lib.util import *

CODE_ROOT = Path(__file__).resolve().parent.parent  # the code root: the shared checkout, or the .tmpcode/ snapshot a job runs
# Code copied into .tmpcode/<experiment>/ at submission (bsub); jobs import only from these. Add files here if the experiment
# starts depending on others. Dependencies (pyproject.toml / uv.lock) are not frozen.
SNAPSHOT_PATHS = ["experiment.py", "tasks.py", "lib", "mia_evals"]
CPUS_PER_GPU = 12  # LSF slots per GPU: 8 GPUs -> all 96 cores; training processes need cores beyond the data workers
# Whole nodes (Optim.batch_size and Hardware.n_workers are per GPU); an H200 node has one GPU taken by someone else's job.
H200 = Hardware(queue="gpu_h200", n_gpus=7, n_workers=CPUS_PER_GPU - 1)
B300 = Hardware(queue="gpu_b300", n_gpus=8, n_workers=CPUS_PER_GPU - 1)
RTX6000 = Hardware(queue="gpu_rtx6000", n_gpus=1, n_workers=CPUS_PER_GPU - 1)  # one of a node's 8 GPUs


@dataclass
class Experiment:
    """An experiment (README: experiment ids): how a run starts, and its runs, each a Trunk whose savedir is
    outdir/<id>/<name>/."""
    id: str  # <series>/e<NN>[-<NN>...]-<slug>, e.g. e00/e15-02-pca
    start: Callable  # takes a run (its Trunk) and submits its first task (tasks.py's start): start_run
    runs: dict[str, Any]  # its Trunks (tasks.py's), by name: [a-z][a-z0-9_-]*

    def placed_runs(self) -> dict[str, Any]:
        assert all(re.fullmatch(r"[a-z][a-z0-9_-]*", name) for name in self.runs), f"run names are lowercase words: {list(self.runs)}"
        return {name: replace(trunk, savedir=f"outdir/{self.id}/{name}") for name, trunk in self.runs.items()}


def experiment_dir(savedir: str) -> str:
    """The experiment of a run dir, <series>/<experiment>: outdir/e00/e12-x/mae/ -> e00/e12-x."""
    return "/".join(Path(savedir).parts[1:3])


def assert_snapshot():
    """LSF jobs must run a code snapshot (bsub): the shared checkout's lib/ may be another sweep's by now."""
    b1 = "LSB_JOBID" in os.environ
    b2 = ".tmpcode" not in Path(__file__).resolve().parts
    assert not (b1 and b2), f"LSF job running {__file__} from the shared checkout; submit via go() (code snapshot)"


def replay_bad_batch(exp: Experiment, path: str):
    """Replay a skipped (non-finite gradient) training step from its bad_batch_*.pt dump: lib.training.replay_bad_batch. Like
    training, replay depends on lib/: only for dumps of this commit's own experiment (submit_replay submits it)."""
    import torch
    ours = Path("outdir", exp.id).resolve()
    assert Path(path).resolve().is_relative_to(ours), f"{path} isn't from this commit's experiment: replay it from its experiment's commit"
    dump = torch.load(path, map_location="cpu", weights_only=False)
    return training.replay_bad_batch(dump)

def record_job(savedir: str, task: str, **args):
    """Append one row to savedir/jobs.json: which task ran, its arguments (dataclasses as dicts), code_provenance(), argv and LSF
    job. Written first thing, so even a run that crashes before any results describes itself. Repro for the dir's artifacts: that
    commit + argv. The command that submitted the job is in outdir/_log/commands.jsonl (log_command)."""
    params = {k: asdict(v) if is_dataclass(v) else v for k, v in args.items()}
    row = {"fn": task, "params": params, **code_provenance(), "argv": sys.argv, "lsf_job": os.environ.get("LSB_JOBID")}
    Path(savedir).mkdir(parents=True, exist_ok=True)
    with open(Path(savedir) / "jobs.json", "a") as f:
        f.write(json.dumps(row) + "\n")

def job_name(savedir: str, job: str) -> str:
    """The LSF name of the run dir savedir's job: <series>-<experiment>-<run>-<job>, e.g. e00-e12-cellmap-mae-run, so bjobs shows which is which."""
    return "-".join(Path(savedir).parts[1:]) + f"-{job}"

def bsub(savedir: str, job: str, minutes: int, hardware: Hardware, cmd: str):
    """Submit `uv run <cmd>` for the run dir savedir to LSF as job job_name(savedir, job), logging to savedir/job_<job>_%J.log, on
    hardware's queue with its GPUs and slots (hardware.slots, or CPUS_PER_GPU per GPU).

    {code} in cmd is this commit's code snapshot (.tmpcode/<sweep>/), which the job runs instead of the shared
    checkout: that may have moved on (another sweep pushed) by the time the job starts. Python puts the script's
    dir first on sys.path, so `lib` comes from the snapshot too; cwd stays the repo root, so outdir/ and data paths
    resolve as before. Callers from the shared checkout assert_committed() first.
    """
    import subprocess
    name = job_name(savedir, job)
    # A job submitting a job hands on the snapshot it runs itself, not the shared checkout's current state.
    code = CODE_ROOT if ".tmpcode" in CODE_ROOT.parts else snapshot(SNAPSHOT_PATHS, Path(".tmpcode") / experiment_dir(savedir).replace("/", "-"))
    gpu = f'-gpu "num={hardware.n_gpus}:mode=exclusive_process"' if hardware.n_gpus else ""
    full = f""" bsub -J {name} \
        -W {minutes // 60}:{minutes % 60:02d} \
        -P miaai \
        -n {hardware.slots or hardware.n_gpus * CPUS_PER_GPU} \
        -R "span[hosts=1]" {gpu} \
        -q {hardware.queue} \
        -o {savedir}/job_{job}_%J.log \
        uv run {cmd.format(code=code)}
        """
    subprocess.run(full, shell=True, stdin=subprocess.DEVNULL, check=True)
    print(f"Submitted {name} (code {code}) to LSF.")

def submit_file(file: str):
    """Submit the job file (go's <savedir>/<task>.json) to LSF, with the resources it says: its Hardware (torchrun on its GPUs, or a
    plain python job on a CPU queue) and minutes. No commit check: callers from the shared checkout (start_run, resubmit_task)
    assert_committed() first; a job calling go() is committed code."""
    job = json.loads(Path(file).read_text())
    h, name, savedir = parse(Hardware, job["hardware"]), job["task"], str(Path(file).parent)
    launch = f"torchrun --standalone --nproc_per_node={h.n_gpus}" if h.n_gpus else "python"
    bsub(savedir, name, job["minutes"], h, f"{launch} {{code}}/experiment.py {name} {file}")

def go(fn: Callable, savedir: str, hardware: Hardware, minutes: int, **args):
    """Run the task fn(**args) as its own LSF job in savedir, on hardware (n_gpus=0: a CPU job), for at most minutes. The arguments
    must be the task's own parameters, JSON-able by its type hints: written, with the resources, to savedir/<fn>.json, which the
    job loads (load_args). The same file is what `resubmit_task <task> <file>` submits again."""
    name = fn.__name__
    hints = typing.get_type_hints(fn)
    assert set(args) == set(inspect.signature(fn).parameters), f"task {name} takes {list(inspect.signature(fn).parameters)}, not {list(args)}"
    assert hardware.n_gpus == 0 or hardware.n_workers + 1 <= CPUS_PER_GPU, f"n_workers={hardware.n_workers} leaves no core for the training process"
    dumped = {k: jsonable(hints[k], v) for k, v in args.items()}
    file = Path(savedir) / f"{name}.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps({"task": name, "hardware": asdict(hardware), "minutes": minutes, "args": dumped}, indent=1) + "\n")
    submit_file(str(file))

def load_args(fn: Callable, file: str) -> dict:
    """The arguments of the job file for the task fn, validated against fn's type hints (parse): a stale or hand-edited file fails
    here, before any GPU work."""
    hints, args = typing.get_type_hints(fn), json.loads(Path(file).read_text())["args"]
    assert set(args) == set(inspect.signature(fn).parameters), f"{file}: has arguments {list(args)}, but task {fn.__name__} takes {list(inspect.signature(fn).parameters)}"
    return {k: parse(hints[k], v) for k, v in args.items()}

def start_run(exp: Experiment, run):
    """Submit run fresh: its savedir emptied (old contents -> outdir/.trash/), then the experiment's start of it."""
    trash(run.savedir)
    exp.start(run)


def resubmit_task(exp: Experiment, task: str, run_or_file: str):
    """Redo one task on LSF: `resubmit_task <experiment's task> <run>` restarts the run named so fresh (start_run), `resubmit_task <task> <job file>` resubmits
    the job in that file (go's <savedir>/<task>.json, e.g. outdir/.../mae/step_0010000/score.json), as it was: after changing the
    board, say."""
    assert_committed()
    roots = exp.placed_runs()
    if run_or_file in roots:
        assert task == exp.start.__name__, f"a run is started by {exp.start.__name__!r}, not {task!r}: redo {task} from its job file"
        start_run(exp, roots[run_or_file])
    else:
        assert Path(run_or_file).name == f"{task}.json", f"{run_or_file} is not a run of {exp.id} ({list(roots)}) or the job file of task {task}"
        submit_file(run_or_file)

def submit_replay(exp: Experiment, name: str):
    """replay_bad_batch on LSF for the run name's first bad_batch dump. CUDA_LAUNCH_BLOCKING names a crashing kernel."""
    run = exp.placed_runs()[name]
    dumps = sorted(Path(run.savedir).glob("bad_batch_*.pt"))
    assert dumps, f"no bad_batch_*.pt in {run.savedir}"
    assert_committed()
    job = next(f for f in dumps[0].parent.glob("*.json") if f.name != "jobs.json")  # the job file of the task that dumped it: its hardware
    hardware = replace(parse(Hardware, json.loads(job.read_text())["hardware"]), n_gpus=1)
    bsub(run.savedir, "replay", 30, hardware, f"env CUDA_LAUNCH_BLOCKING=1 python {{code}}/experiment.py replay_bad_batch {dumps[0]}")

def experiment_id(name: str) -> str | None:
    """The id of a sweep dir name under outdir/<series>/: 'e15-02-pca' -> 'e15-02'. None for one from before
    numbering (e.g. 'viewsizes-v2')."""
    m = re.fullmatch(r"(e\d\d(?:-\d\d)*)-[a-z][a-z0-9-]*", name)
    return m[1] if m else None

def assert_experiment(exp: Experiment):
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
    sources = [p.init_from for p in exp.placed_runs().values()]
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
    """Submit the experiment (README: one experiment, one entrypoint): every run whose job file isn't in its savedir yet, i.e. not
    submitted before (whether it finished, is still going, or failed). Idempotent: run it again after adding runs; a run
    already submitted is left alone (`resubmit_task <the experiment's start> <name>` restarts one fresh)."""
    assert_experiment(exp)
    assert_committed()
    for r in exp.placed_runs().values():
        if not any(Path(r.savedir).glob("*.json")):  # a job file: a task was submitted
            start_run(exp, r)


COMMANDS = [replay_bad_batch, resubmit_task, submit_replay, submitall]  # take the experiment first, bound by main


def main(exp: Experiment, tasks: list[Callable]):
    """The command line of experiment.py: `<task> <job file>` (what an LSF job runs), or a command on the experiment's runs."""
    names = {**{f.__name__: f for f in tasks}, **{f.__name__: partial(f, exp) for f in COMMANDS}}
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        for name, f in names.items():
            doc = (getattr(f, "func", f).__doc__ or "").strip().splitlines()[0]
            print(f"  {name}{str(inspect.signature(f)).replace('exp: Experiment, ', '')}  {doc}")
        sys.exit()
    assert sys.argv[1] in names, f"unknown command {sys.argv[1]!r}: {', '.join(names)}"
    log_command(sys.argv)  # commands issued outside jobs -> outdir/_log/commands.jsonl; jobs use jobs.json
    if sys.argv[1] in {f.__name__ for f in tasks}:
        names[sys.argv[1]](**load_args(names[sys.argv[1]], sys.argv[2]))
    else:
        names[sys.argv[1]](*sys.argv[2:])
