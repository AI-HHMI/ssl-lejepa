"""What an experiment.py and its tasks.py are made of, and the LSF code that runs it. An experiment is a table of runs (Experiment),
each a Trunk, and the tasks that process each run: plain functions in tasks.py (trunk, anneal, finetune, score) with JSON-able arguments
(dataclasses, strings), each calling the next with go(), which submits it as an LSF job: a job is a task submitted. go() writes the
arguments and the job's resources to <savedir>/<task>.json and runs `experiment.py <task> <that file>`, which loads the arguments by
the task's type hints and calls it, so a task runs the same in-process, called with its arguments, and as a job; `resubmit_task <task> <run
or job file>` (lib.task_entry) redoes one by hand. Depends on lib/, so a task runs only from its experiment's commit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
import inspect
import json
import os
import re
import sys
from pathlib import Path
import typing
from typing import Any, Callable

from lib.training import Hardware
from lib.util import *

CODE_ROOT = Path(__file__).resolve().parent.parent  # the code root: the shared checkout, or the .tmpcode/ snapshot a job runs
# Code copied into .tmpcode/<experiment>/ at submission (bsub); jobs import only from these. Add files here if the experiment
# starts depending on others. Dependencies (pyproject.toml / uv.lock) are not frozen.
SNAPSHOT_PATHS = ["experiment.py", "tasks.py", "lib", "mia_evals"]
CPUS_PER_GPU = 12  # LSF slots per GPU: 8 GPUs -> all 96 cores; training processes need cores beyond the data workers
# Whole nodes (Optim.batch_size and Hardware.n_workers are per GPU).
H200 = Hardware(queue="gpu_h200", n_gpus=8, n_workers=CPUS_PER_GPU - 1)
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
