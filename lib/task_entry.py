"""The command line of experiment.py (main) and the commands it points to, on an Experiment's runs: submitall, resubmit_task,
submit_replay, replay_bad_batch; any other command is a task, run as an LSF job runs it (`<task> <job file>`, lib.task_utils.go)."""

import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

from lib import training
from lib.task_utils import Experiment, bsub, load_args, submit_file
from lib.util import *


def replay_bad_batch(exp: Experiment, path: str):
    """Replay a skipped (non-finite gradient) training step from its bad_batch_*.pt dump: lib.training.replay_bad_batch. Like
    training, replay depends on lib/: only for dumps of this commit's own experiment (submit_replay submits it)."""
    import torch
    ours = Path("outdir", exp.id).resolve()
    assert Path(path).resolve().is_relative_to(ours), f"{path} isn't from this commit's experiment: replay it from its experiment's commit"
    dump = torch.load(path, map_location="cpu", weights_only=False)
    return training.replay_bad_batch(dump)

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


USAGE = """usage: python experiment.py <command> [args]
  submitall                              submit every run not submitted yet
  resubmit_task <task> <run | job file>  restart a run fresh (task: the start), or redo one task from its job file
  submit_replay <run>                    replay the run's first bad batch on LSF
  replay_bad_batch <bad_batch_*.pt>      replay one bad batch here
  <task> <job file>                      run a task (what an LSF job does)"""


def main(exp: Experiment, tasks: list[Callable]):
    """The command line of experiment.py: `<task> <job file>` (what an LSF job runs), or a command on the experiment's runs."""
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("-h", [])
    if cmd in ("-h", "--help"):
        print(USAGE + f"\n  tasks: {', '.join(f.__name__ for f in tasks)}")
        return
    log_command(sys.argv)  # commands issued outside jobs -> outdir/_log/commands.jsonl; jobs use jobs.json
    if cmd == "submitall":
        submitall(exp)
    elif cmd == "resubmit_task":
        resubmit_task(exp, *args)
    elif cmd == "submit_replay":
        submit_replay(exp, *args)
    elif cmd == "replay_bad_batch":
        replay_bad_batch(exp, *args)
    else:
        task = {f.__name__: f for f in tasks}.get(cmd)
        assert task, f"unknown command {cmd!r}: submitall, resubmit_task, submit_replay, replay_bad_batch, or a task: {[f.__name__ for f in tasks]}"
        task(**load_args(task, args[0]))
