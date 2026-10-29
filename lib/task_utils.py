"""What an experiment.py and its tasks.py are made of, and the LSF code that runs it. An experiment is a table of runs (Experiment),
each a Trunk, and the tasks that process each run: plain functions in tasks.py (trunk, branch, score) with JSON-able arguments
(dataclasses, strings), each calling the next with go(), which submits it as an LSF job: a job is a task submitted. go() writes the
arguments and the job's resources to <savedir>/<task>.json and runs `experiment.py <task> <that file>`, which loads the arguments by
the task's type hints and calls it, so a task runs the same in-process, called with its arguments, and as a job; `lsf <task> <run
or job file>` redoes one by hand. Depends on lib/, so a task runs only from its experiment's commit. `python experiment.py` or `-h`
lists the commands and tasks (main)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
import fcntl
from functools import partial
import inspect
import json
import os
import re
import shutil
import sys
from pathlib import Path
import typing
from typing import Any, Callable

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
    """An experiment (README: experiment ids): the task every run starts with, and its runs, each a Trunk whose savedir is
    outdir/<id>/<name>/."""
    id: str  # <series>/e<NN>[-<NN>...]-<slug>, e.g. e00/e15-02-pca
    task: Callable  # takes a run (its `t`): submitted for each on the run's t.train.hardware for t.hours (start)
    runs: dict[str, Any]  # its Trunks (tasks.py's), by name: [a-z][a-z0-9_-]*

    def roots(self) -> dict[str, Any]:
        assert all(re.fullmatch(r"[a-z][a-z0-9_-]*", name) for name in self.runs), f"run names are lowercase words: {list(self.runs)}"
        return {name: replace(spec, savedir=f"outdir/{self.id}/{name}/") for name, spec in self.runs.items()}


def experiment_dir(savedir: str) -> str:
    """The experiment of a run dir, <series>/<experiment>: outdir/e00/e12-x/mae/ -> e00/e12-x."""
    return "/".join(Path(savedir).parts[1:3])


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

def mia_eval(t, savedir: str) -> float:
    """Score the affinities in savedir (probe/{fit,test}, written by a branch's probe or finetune) as mia-evals' neuron-instance task on trunk t's board
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
    savedir = Path(savedir)
    assert (savedir / "probe/test").is_dir() and (savedir / "probe/fit").is_dir(), f"no probe artifacts in {savedir}: probe or finetune first"
    (savedir / "resolved_config.json").write_text(json.dumps({"savedir": str(savedir), "trunk": asdict(t)}) + "\n")
    (savedir / "git_commit.txt").write_text(code_provenance()["commit_id"] + "\n")
    config = CODE / BOARDS[t.board]  # the snapshot's copy when run as a job
    root = savedir.resolve()
    evaluate.cmd_score(Namespace(
        config=config, test=root / "probe/test", val=root / "probe/fit", leaderboard=root / "mia_evals", run_dir=root,
        no_scored=True,  # don't keep the 5.8 GB post-processed labelling per run
        scored_out=None, scratch=None))
    shutil.rmtree(savedir / "probe")  # the artifacts are scored (the record is in mia_evals/): GBs per run, not kept
    record_file = sorted(savedir.glob("mia_evals/*/records/*.json"))[-1]
    return json.loads(record_file.read_text())["scores"]["voxel_instance"]["pq"]

def keep_if_best(t, savedir: str, score: float):
    """Keep the branch's weights (savedir/checkpoint.pt, its end) as its trunk's best.pt if score beats the best so far (best.json, in the
    trunk's dir); else delete them. The scores of one trunk's branches finish in any order, so the compare-and-move is under a lock
    held only for it (a rename, not a copy)."""
    trunk, weights = Path(savedir).parent, Path(savedir) / "checkpoint.pt"
    step = int(Path(savedir).name.split("_")[1])
    with open(trunk / "best.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # released when the file closes
        best = json.loads((trunk / "best.json").read_text())["score"] if (trunk / "best.json").is_file() else float("-inf")
        if score > best:
            weights.replace(trunk / "best.pt")
            (trunk / "best.json").write_text(json.dumps({"score": score, "step": step, "board": t.board, "from": str(savedir)}) + "\n")
        else:
            weights.unlink()
    print(f"{savedir}: score {score:.4f} {'beats' if score > best else 'does not beat'} the best so far, {best:.4f}", flush=True)

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

def record(savedir: str, fn: str, **args):
    """Append one row to savedir/runs.json: which task ran (fn), its arguments (dataclasses as dicts), code_provenance(), argv and LSF
    job. Written first thing, so even a run that crashes before any results describes itself. Repro for the dir's artifacts: that
    commit + argv. The command that submitted the job is in outdir/_log/commands.jsonl (log_command)."""
    params = {k: asdict(v) if is_dataclass(v) else v for k, v in args.items()}
    row = {"fn": fn, "params": params, **code_provenance(), "argv": sys.argv, "lsf_job": os.environ.get("LSB_JOBID")}
    Path(savedir).mkdir(parents=True, exist_ok=True)
    with open(Path(savedir) / "runs.json", "a") as f:
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

def submit(file: str):
    """Submit the job file (go's <savedir>/<task>.json) to LSF, with the resources it says: the task's GPUs (torchrun), or a CPU job on
    SCORE_QUEUE. No commit check: callers from the shared checkout (start, lsf) assert_committed() first; a job calling go() is
    committed code."""
    job = json.loads(Path(file).read_text())
    h, name, savedir = parse(Hardware | None, job["hardware"]), job["task"], str(Path(file).parent)
    if h:
        bsub(savedir, name, job["minutes"], h.n_gpus, h.queue, f"torchrun --standalone --nproc_per_node={h.n_gpus} {{code}}/experiment.py {name} {file}")
    else:
        bsub(savedir, name, job["minutes"], 0, SCORE_QUEUE, f"python {{code}}/experiment.py {name} {file}", slots=SCORE_SLOTS)

def go(fn: Callable, savedir: str, hardware: Hardware | None, minutes: int, **args):
    """Run the task fn(**args) as its own LSF job in savedir, on hardware (None: a CPU job), for at most minutes. The arguments must
    be the task's own parameters, JSON-able by its type hints: written, with the resources, to savedir/<fn>.json (job_file), which
    the job loads (load_args). The same file is what `lsf <task> <file>` resubmits."""
    name = fn.__name__
    hints = typing.get_type_hints(fn)
    assert set(args) == set(inspect.signature(fn).parameters), f"task {name} takes {list(inspect.signature(fn).parameters)}, not {list(args)}"
    assert hardware is None or hardware.n_workers + 1 <= CPUS_PER_GPU, f"n_workers={hardware.n_workers} leaves no core for the training process"
    dumped = {k: pydantic_dump(hints[k], v) for k, v in args.items()}
    file = Path(savedir) / f"{name}.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps({"task": name, "hardware": asdict(hardware) if hardware else None, "minutes": minutes, "args": dumped}, indent=1) + "\n")
    submit(str(file))

def pydantic_dump(tp, value):
    """value as JSON-able data of the type tp (parse's inverse)."""
    import pydantic
    return pydantic.TypeAdapter(tp).dump_python(value)

def load_args(fn: Callable, file: str) -> dict:
    """The arguments of the job file for the task fn, validated against fn's type hints (parse): a stale or hand-edited file fails
    here, before any GPU work."""
    hints, args = typing.get_type_hints(fn), json.loads(Path(file).read_text())["args"]
    assert set(args) == set(inspect.signature(fn).parameters), f"{file}: has arguments {list(args)}, but task {fn.__name__} takes {list(inspect.signature(fn).parameters)}"
    return {k: parse(hints[k], v) for k, v in args.items()}

def start(exp: Experiment, run):
    """Submit run fresh: its savedir emptied (old contents -> outdir/.trash/), then the experiment's task on its Train's hardware for its hours."""
    trash(run.savedir)
    go(exp.task, run.savedir, run.train.hardware, int(run.hours * 60), t=run)


def lsf(exp: Experiment, fn: str, target: str):
    """Redo one task on LSF: `lsf <experiment's task> <run>` restarts the run named so fresh (start), `lsf <task> <job file>` resubmits
    the job in that file (go's <savedir>/<task>.json, e.g. outdir/.../mae/step_0010000/score.json), as it was: after changing the
    board, say."""
    assert_committed()
    roots = exp.roots()
    if target in roots:
        assert fn == exp.task.__name__, f"a run is started by {exp.task.__name__!r}, not {fn!r}: redo {fn} from its job file"
        start(exp, roots[target])
    else:
        assert Path(target).name == f"{fn}.json", f"{target} is not a run of {exp.id} ({list(roots)}) or the job file of task {fn}"
        submit(target)

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
    already submitted is left alone (`lsf <the experiment's task> <name>` restarts one fresh)."""
    check_experiment(exp)
    assert_committed()
    for r in exp.roots().values():
        if not (Path(r.savedir) / f"{exp.task.__name__}.json").is_file():
            start(exp, r)


COMMANDS = [replay_bad_batch, lsf, replaylsf, submitall]  # take the experiment first, bound by main


def main(exp: Experiment, tasks: list[Callable]):
    """The command line of experiment.py: `<task> <job file>` (what an LSF job runs), or a command on the experiment's runs."""
    names = {**{f.__name__: f for f in tasks}, **{f.__name__: partial(f, exp) for f in COMMANDS}}
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        for name, f in names.items():
            doc = (getattr(f, "func", f).__doc__ or "").strip().splitlines()[0]
            print(f"  {name}{str(inspect.signature(f)).replace('exp: Experiment, ', '')}  {doc}")
        sys.exit()
    assert sys.argv[1] in names, f"unknown command {sys.argv[1]!r}: {', '.join(names)}"
    log_command(sys.argv)  # commands issued outside jobs -> outdir/_log/commands.jsonl; jobs use runs.json
    if sys.argv[1] in {f.__name__ for f in tasks}:
        names[sys.argv[1]](**load_args(names[sys.argv[1]], sys.argv[2]))
    else:
        names[sys.argv[1]](*sys.argv[2:])
