"""The pipeline of tasks and the LSF code that runs an experiment: experiment.py is the table of runs, this is everything else.
Each task is an entrypoint taking its spec or the path of the spec's json, e.g. `run outdir/.../train.json`: called in-process
with the spec, or as an LSF job with the file that submit() writes, so any task can submit another. TASKS has every task, with
the hardware it needs, the data it reads (by role, lib.data's Dataset) and what it writes. The pipeline, per run:
    run(spec):                                   # pretrain on the train volumes of the spec's datasets (images, no labels)
        train(spec)                              #   a checkpoint at each milestone, evaluated as it is saved (eval_milestone)
        for checkpoint in milestones + the end:  #   the tasks TASKS has inside "run":
            pca(checkpoint)                      # a val crop's token embeddings, as colours
            val(checkpoint)                      # pretraining loss: val = held-out volumes, train = the training volumes
            probe(checkpoint)                    # fit volume -> a Linear to the affinities; scored on the test volumes
    finetune(checkpoint)                         # fit volume -> a UNETR on the encoder; scored on the test volumes
    score(probe | finetune)                      # post-processing parameters chosen on the tune volume, applied to the test volumes
Depends on lib/, so it runs only from its experiment's commit. `python experiment.py` or `-h` lists the commands (main)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import partial
import inspect
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

from lib import evals, training
from lib.specs import Eval, Finetune, Spec, Train, eval_params, load_checkpoint, milestone_steps, own_eval, record, source
from lib.training import latest_step
from lib.util import *

CODE = Path(__file__).resolve().parent.parent  # the code root: the shared checkout, or the .tmpcode/ snapshot a job runs
# Code copied into .tmpcode/<experiment>/ at submission (bsub); jobs import only from these. Add files here if the experiment
# starts depending on others. Dependencies (pyproject.toml / uv.lock) are not frozen.
SNAPSHOT_PATHS = ["experiment.py", "lib", "mia_evals"]
CPUS_PER_GPU = 12  # LSF slots per GPU: 8 GPUs -> all 96 cores; training processes need cores beyond the data workers
H200 = dict(queue="gpu_h200", n_gpus=7)  # whole nodes (Train.batch_size and n_workers are per GPU); an H200 node has one GPU taken by someone else's job
B300 = dict(queue="gpu_b300", n_gpus=8)
# mia-evals scoring (score): CPU only. mia-evals: ~1 h and ~200 GB peak for the mutex watershed at 896^3, "give it a
# whole node". The shortest CPU queue; check its run limit and memory per slot with `bqueues -l short`.
SCORE_QUEUE, SCORE_SLOTS, SCORE_MINUTES = "local", 16, 4*60
SCORE_CONFIG = "mia_evals/gary_comparison_neuron_instance/mws.toml"  # ours: truth_kind = "instances"
# score() runs mia-evals from this sister checkout, with its own .venv (`uv sync` there) and its own commit, which
# mia-evals records in each score's provenance (mia_evals_commit). Not a dependency of this repo.
MIA_EVALS = Path.home() / "proj/mia-evals"


@dataclass
class Experiment:
    """An experiment (README: experiment ids): its runs, each a spec whose savedir is outdir/<id>/<name>/."""
    id: str  # <series>/e<NN>[-<NN>...]-<slug>, e.g. e00/e15-02-pca
    runs: dict[str, Spec]  # by name: [a-z][a-z0-9_-]*

    def roots(self) -> dict[str, Spec]:
        assert all(re.fullmatch(r"[a-z][a-z0-9_-]*", name) for name in self.runs), f"run names are lowercase words: {list(self.runs)}"
        return {name: replace(spec, savedir=f"outdir/{self.id}/{name}/") for name, spec in self.runs.items()}


def experiment_dir(savedir: str) -> str:
    """The experiment of a run dir, <series>/<experiment>: outdir/e00/e12-x/mae/ -> e00/e12-x."""
    return "/".join(Path(savedir).parts[1:3])


class Use(NamedTuple):
    """A dataset volume a task reads, by its role (a field of lib.data.Dataset)."""
    role: str  # train, val, fit, test or tune
    labels: bool  # its labels are read (else its images only)
    when: str = ""  # the spec flag it needs, e.g. "score" (the tune volume's predictions are written only for a score): "" always

class Task(NamedTuple):
    """One task of the pipeline: the specs it runs on, the hardware it needs, whether a task's process runs it itself or only a job
    does, the data it uses and what it writes."""
    spec: type | tuple[type, ...]  # the spec(s) its entrypoint takes
    needs: str  # the hardware: "gpus": the spec's own, torchrun; "gpu": one, on the trained run's queue; "cpu": SCORE_QUEUE's slots
    minutes: int  # the LSF wall limit if it is its own job (a run with max_hours: that + 60)
    inside: str  # "": only ever its own LSF job (submit); else the task whose process also calls it, in that process, without a new job
    data: tuple[Use, ...]
    writes: tuple[str, ...]  # into its savedir

TRAIN, VAL, FIT, TEST, TUNE = Use("train", False), Use("val", False), Use("fit", True), Use("test", True), Use("tune", True)
TUNE_IF_SCORE = TUNE._replace(labels=False, when="score")  # probe and finetune write the tune volume's predictions only for a score

# Every task, in pipeline order: submit() reads needs and minutes, evals_inside_run the tasks inside "run", `-h` prints the rest.
TASKS = {
    "run":       Task(Train,            "gpus", 15,            "",    (TRAIN,),                   ("checkpoints/", "metrics.json", "the evals of each milestone and of its end")),
    "benchmark": Task(Train,            "gpus", 15,            "",    (TRAIN,),                   ("performance.json", "profile.out")),
    "pca":       Task(Eval,             "gpu",  30,            "run", (VAL,),                     ("pca.json", "pca2.png")),
    "pca_sizes": Task(Eval,             "gpu",  60,            "",    (VAL,),                     ("pca_sizes.json", "pca_tile_<side>.png")),
    "val":       Task(Eval,             "gpu",  30,            "run", (VAL, TRAIN),               ("val.json",)),
    "probe":     Task(Eval,             "gpu",  60,            "run", (FIT, TEST, TUNE_IF_SCORE), ("probe.json", "probe_fit.json", "probe.png", "probe/{fit,test}/ (score)")),
    "finetune":  Task(Finetune,         "gpu",  120,           "",    (FIT, TEST, TUNE_IF_SCORE), ("finetune_curve.json", "finetune.png", "probe/{fit,test}/ (score)")),
    "tiles":     Task(Finetune,         "gpu",  120,           "",    (FIT, TEST),                ("tiles.png",)),
    "score":     Task((Eval, Finetune), "cpu",  SCORE_MINUTES, "",    (TUNE, TEST),               ("mia_evals/<task>/records/*.json",)),
}


def eval_milestone(trunk: Train, step: int):
    """The trunk's own eval of its milestone at step (eval_params: the tasks TASKS has inside "run", in branch_dir), in this process between two
    training steps: as LSF jobs, e02's waited hours behind the per-user GPU cap for minutes of work. Its eval.json is written
    first, so `lsf pca|probe <eval.json>` can redo one. A fresh eager model from the milestone file (load_checkpoint) leaves
    the compiled training model alone, and the RNG state is put back after, so training draws as if no eval had run. Its
    minutes count against max_hours; the other DDP ranks wait at their next collective (setup's timeout). A failed eval
    prints its traceback and keeps the milestone for a redo; the trunk trains on."""
    import torch
    import traceback
    e = eval_params(trunk, step)
    dump(e)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            evals_inside_run(e)
    except Exception:
        traceback.print_exc()
    torch.cuda.empty_cache()  # the eval's memory back for training (a no-op without CUDA)

def evals_inside_run(e: Eval):
    """The tasks a run calls in its own process, TASKS' inside == "run", on e, in pipeline order."""
    for fn, task in TASKS.items():
        if task.inside == "run":
            ENTRYPOINTS[fn](e)

def check_snapshot():
    """LSF jobs must run a code snapshot (bsub): the shared checkout's lib/ may be another sweep's by now."""
    b1 = "LSB_JOBID" in os.environ
    b2 = ".tmpcode" not in Path(__file__).resolve().parts
    assert not (b1 and b2), f"LSF job running {__file__} from the shared checkout; submit via submit() (code snapshot)"

def run(job: Train | str):
    """The LSF training job: train the spec job (or its train.json's), then rank 0 evaluates its last finite checkpoint (own_eval: the
    tasks TASKS has inside "run").

    Evals run after train() has destroyed the process group, so the other ranks exit instead of waiting at a
    collective for minutes; not for a trunk (a run with milestones: not an anneal) that stopped on a milestone: train()
    evaluated that checkpoint already (eval_milestone). One that max_hours stopped between two does. (Throughput
    benchmarks are their own entrypoint, benchmark, with no evals.)
    """
    check_snapshot()
    par = load(job, Train)
    b1 = int(os.environ.get("RANK", 0)) == 0
    if b1:  # first thing, so even a run that crashes early describes itself
        record(par, "run")
    training.train(par, eval_milestone)
    b2 = not par.anneal_from and latest_step(par) in milestone_steps(par)
    if b1 and not b2:
        evals_inside_run(own_eval(par))


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

def pca(job: Eval | str):
    """PCA maps (lib.evals.pca_maps) of the spec job's weights (or those of the eval.json it names). run() does it at the end of
    training; `lsf pca <run>` redoes it."""
    e = load(job, Eval)
    model, step = load_checkpoint(e)
    record(e, "pca")  # argv says whether it ran inside `run` or on its own
    evals.pca_maps(model.encoder, e.train, Path(e.savedir), step, e.data)

def pca_sizes(job: Eval | str):
    """The input size at test time (lib.evals.pca_sizes) of the spec job's weights (or those of the eval.json it names)."""
    e = load(job, Eval)
    model, step = load_checkpoint(e)
    record(e, "pca_sizes")
    evals.pca_sizes(e, model, step)

def val(job: Eval | str):
    """The pretraining loss of the spec job's weights (or those of the eval.json it names) on held-out EM and on the training
    volumes (lib.evals.val_loss): savedir/val.json. A trunk's milestone evals do it as they save a milestone."""
    par = load(job, Eval)
    model, step = load_checkpoint(par)
    record(par, "val")
    evals.val_loss(par, model, step)

def probe(job: Eval | str):
    """Linear affinity probe (lib.evals.probe) of the spec job's weights (or those of the eval.json it names)."""
    par = load(job, Eval)
    model, step = load_checkpoint(par)
    record(par, "probe")
    evals.probe(par, model, step)

def finetune(job: Finetune | str):
    """A UNETR on the spec job's weights (or those of the finetune.json it names), fitted and scored: lib.evals.finetune."""
    par = load(job, Finetune)
    model, step = load_checkpoint(par)
    record(par, "finetune")
    evals.finetune(par, model, step)

def tiles(job: Finetune | str):
    """finetune's fit without its scoring, and its test block's predictions with the current tiling: lib.evals.tiles."""
    par = load(job, Finetune)
    model, step = load_checkpoint(par)
    evals.tiles(par, model, step)

def score(job: Eval | Finetune | str):
    """Score probe(job)'s (or finetune(job)'s: a Finetune, or the finetune.json) affinities as mia-evals' neuron-instance task (SCORE_CONFIG: mutex watershed + size filter,
    fitted on the fit block, reported on the test block; panoptic quality ranks, VOI and ARE reported).

    Writes the record into the run's own dir, savedir/mia_evals/<task>/records/*.json (pull.sh brings it down;
    analysis_plots.mia_evals_table reads it), not mia-evals' repo. resolved_config.json (params) and git_commit.txt (the
    commit that ran) go in the savedir first: mia-evals copies them into the record from --run-dir.
    Runs the sister checkout MIA_EVALS's own CLI and .venv, from that checkout (its commit is in the record), with this
    commit's SCORE_CONFIG (`lsf score n`). Deletes savedir/probe/ (the artifacts it read) once the record is written.
    """
    import subprocess
    par = load(job, Finetune if isinstance(job, Finetune) or str(job).endswith("finetune.json") else Eval)
    savedir = Path(par.savedir)
    assert (savedir / "probe/test").is_dir() and (savedir / "probe/fit").is_dir(), f"no probe artifacts in {savedir}: probe or finetune first (a run with score=False writes none)"
    exe = MIA_EVALS / ".venv/bin/mia-evals"
    assert exe.is_file(), f"no mia-evals CLI at {exe}: clone github.com/AI-HHMI/mia-evals to {MIA_EVALS} and `uv sync` there"
    record(par, "score")
    (savedir / "resolved_config.json").write_text(json.dumps(asdict(par)) + "\n")
    (savedir / "git_commit.txt").write_text(code_provenance()["commit_id"] + "\n")
    config = CODE / SCORE_CONFIG  # the snapshot's copy when run as a job
    cmd = [str(exe), "score", str(config),
           "--test", str(savedir.resolve() / "probe/test"), "--val", str(savedir.resolve() / "probe/fit"),
           "--leaderboard", str(savedir.resolve() / "mia_evals"), "--run-dir", str(savedir.resolve()),
           "--no-scored"]  # don't keep the 5.8 GB post-processed labelling per run
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=MIA_EVALS)  # absolute paths: the cwd is mia-evals'
    shutil.rmtree(savedir / "probe")  # the artifacts are scored (the record is in mia_evals/): GBs per run, not kept

def job_name(savedir: str, job: str) -> str:
    """The LSF name of the run dir savedir's job: <series>-<experiment>-<run>-<job>, e.g. e00-e12-cellmap-mae-run, so bjobs shows which is which."""
    return "-".join(Path(savedir).parts[1:]) + f"-{job}"

def bsub(savedir: str, job: str, minutes: int, n_gpus: int, queue: str, cmd: str, slots: int | None = None, after: str = ""):
    """Submit `uv run <cmd>` for the run dir savedir to LSF as job job_name(savedir, job), logging to savedir/job_<job>_%J.log,
    on queue with CPUS_PER_GPU slots per GPU (a CPU job, n_gpus = 0, names its slots); held until the job named after is
    done (exited 0), if given.

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
    wait = f'-w "done({after})"' if after else ""
    full = f""" bsub -J {name} \
        -W {minutes // 60}:{minutes % 60:02d} \
        -P miaai \
        -n {slots or n_gpus * CPUS_PER_GPU} \
        -R "span[hosts=1]" {gpu} \
        -q {queue} {wait} \
        -o {savedir}/job_{job}_%J.log \
        uv run {cmd.format(code=code)}
        """
    subprocess.run(full, shell=True, stdin=subprocess.DEVNULL, check=True)  # waits: a later job may name this one in its after
    print(f"Submitted {name} (code {code}) to LSF.")

def submit(fn: str, job: Spec, after: str = ""):
    """Run the entrypoint fn on the spec job as an LSF job: job written to its savedir (dump: train.json, eval.json or
    finetune.json), and `experiment.py <fn> <that file>` submitted (bsub), which loads it. Resources by TASKS[fn].needs: a training
    job on job's queue and GPUs (torchrun), one GPU on the trained run's queue, or a CPU job on SCORE_QUEUE. (The tasks a run calls inside
    its own process, TASKS[fn].inside, are not submitted by it.) A finetune of one of this
    experiment's own trunks waits in LSF for the trunk's run job, unless the trunk is done (its own probe.json is written). No
    commit check: callers from the shared checkout (submitall, lsf) assert_committed() first; a job submitting one is committed
    code. after: the name of an LSF job a score waits for (start)."""
    task = TASKS[fn]
    assert isinstance(job, task.spec), f"{job.savedir}: no LSF job {fn!r} for a {type(job).__name__} (it takes {task.spec}): {list(TASKS)}"
    assert not isinstance(job, Train) or job.n_workers + 1 <= CPUS_PER_GPU, f"n_workers={job.n_workers} leaves no core for the training process"
    spec = dump(job)
    cmd = f"python {{code}}/experiment.py {fn} {spec}"
    if task.needs == "gpus":
        # Slack: startup, final checkpoint, pca + probe. A run without max_hours (a fixed step count) or a benchmark: task.minutes.
        minutes = int(job.max_hours * 60) + 60 if job.max_hours and fn == "run" else task.minutes
        bsub(job.savedir, fn, minutes, job.n_gpus, job.queue, f"torchrun --standalone --nproc_per_node={job.n_gpus} {{code}}/experiment.py {fn} {spec}")
    elif task.needs == "cpu":
        bsub(job.savedir, fn, task.minutes, 0, SCORE_QUEUE, cmd, slots=SCORE_SLOTS, after=after)
    else:
        b1 = isinstance(job, Finetune) and job.init_from.startswith(f"outdir/{experiment_dir(job.savedir)}/")
        b2 = not (Path(job.init_from) / "probe.json").is_file()
        bsub(job.savedir, fn, task.minutes, 1, job.train.queue, cmd, after=job_name(job.init_from, "run") if b1 and b2 else "")

def jobs(p: Spec) -> list[str]:
    """The tasks start() submits for p, in pipeline order: a training run (which evals itself after), an eval's (Eval.jobs: pca and
    probe, unless it says otherwise), or a finetune and, if its artifacts are to be scored, the score."""
    if isinstance(p, Train):
        return ["run"]
    return [*p.jobs] if isinstance(p, Eval) else ["finetune", "score"] if p.score else ["finetune"]

def start(p: Spec):
    """Submit p fresh: its savedir emptied (old contents -> outdir/.trash/), then its jobs. A score waits for the job that wrote its
    artifacts, the probe or the finetune."""
    trash(p.savedir)
    for fn in jobs(p):
        submit(fn, p, after=job_name(p.savedir, "probe" if isinstance(p, Eval) else "finetune") if fn == "score" else "")


def lsf(exp: Experiment, fn: str, target: str):
    """Redo one task on LSF, of the run named target or of the spec file target names (any run's, e.g. a milestone eval's
    outdir/.../mae/step_0005000/eval.json): `lsf probe mae` after changing probe. pca, probe, val or score of an eval, or of a training
    run's own checkpoint (own_eval), in place; finetune of a finetune, in place; benchmark of a training run, in place (its
    savedir gets a train.json, so submitall skips it: `lsf run` it after); run restarts a training run fresh (start), its
    milestone evals with it."""
    assert_committed()
    roots = exp.roots()
    p = roots[target] if target in roots else load(target, Eval if target.endswith("eval.json") else Finetune if target.endswith("finetune.json") else Train)
    if fn == "run":
        start(p)
    elif fn == "benchmark":
        submit(fn, p)
    else:
        submit(fn, own_eval(p) if isinstance(p, Train) else p)

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
    sources = [source(p) for p in exp.roots().values()]
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
    already submitted is left alone (`lsf run <name>` restarts one fresh)."""
    check_experiment(exp)
    assert_committed()
    for r in exp.roots().values():
        if not spec_path(r).is_file():
            start(r)

def retryall(exp: Experiment):
    """Resubmit the experiment's finetunes whose job ended without finishing (its LSF log has its footer, but there is no
    finetune.png, which finetune writes last), fresh (start: the failed dir's contents -> outdir/.trash/). Finished ones,
    and ones still queued or running, are left alone."""
    check_experiment(exp)
    assert_committed()
    for r in exp.roots().values():
        logs = sorted(Path(r.savedir).glob("job_finetune_*.log"))
        b1 = isinstance(r, Finetune) and not (Path(r.savedir) / "finetune.png").is_file()
        b2 = bool(logs) and "Resource usage summary" in logs[-1].read_text(errors="ignore")
        if b1 and b2:
            start(r)


ENTRYPOINTS = {f.__name__: f for f in [run, benchmark, pca, pca_sizes, val, probe, finetune, tiles, score]}  # take a spec or its json
COMMANDS = [replay_bad_batch, lsf, replaylsf, submitall, retryall]  # take the experiment first, bound by main


def main(exp: Experiment):
    """The command line of experiment.py: `<entrypoint> <spec json>` (what an LSF job runs), or a command on the experiment's runs."""
    names = {**ENTRYPOINTS, **{f.__name__: partial(f, exp) for f in COMMANDS}}
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        for name, f in names.items():
            doc = (f.__doc__ or f.func.__doc__ or "").strip().splitlines()[0]
            print(f"  {name}{str(inspect.signature(f)).replace('exp: Experiment, ', '')}  {doc}")
            if name in TASKS:
                t = TASKS[name]
                uses = [f"{u.role}, {'labels' if u.labels else 'images'}{f', if {u.when}' if u.when else ''}" for u in t.data]
                print(f"      needs: {t.needs}; as a job" + (f", and also inside {t.inside}'s own process" if t.inside else ""))
                print(f"      data: {'; '.join(uses)}\n      writes: {', '.join(t.writes)}")
        sys.exit()
    assert sys.argv[1] in names, f"unknown command {sys.argv[1]!r}: {', '.join(names)}"
    log_command(sys.argv)  # commands issued outside jobs -> outdir/_log/commands.jsonl; jobs use runs.json
    names[sys.argv[1]](*sys.argv[2:])
