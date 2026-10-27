"""This experiment's sweep and its entrypoints (remote): roots() lists its runs (savedirs under outdir/e00/), the LSF code submits them,
and each entrypoint wires one job: it loads its spec, records the run, calls the work in lib/ and does the experiment-specific
follow-up (a trunk's milestone evals, the code snapshot check). The work is in lib/: specs.py (Train, Eval,
Finetune, the model and weights they name), training.py, evals.py (pca, probe, val, finetune), readers.py (volumes) and inference.py
(tiled predictions, scores). Depends on lib/, so it runs only from its experiment's commit.
Each job is an entrypoint taking its spec or the path of the spec's json, e.g. `run outdir/.../train.json`: called in-process
with the spec, or as an LSF job with the file that submit() writes, so any job can submit another.
`python experiment.py` or `-h` lists the entrypoints (ENTRYPOINTS, at the bottom).
Analysis of the results lives in analysis_utils.py (readers), analysis_plots.py (report cards) and reports/ (one page per experiment)."""

from __future__ import annotations

from dataclasses import asdict
import inspect
import os
import sys
import json
import re
import shutil
from pathlib import Path


# local

from lib.specs import (Eval, Finetune, Spec, Train, eval_params, load_checkpoint, milestone_steps,
                       own_eval, probe_evals, record, source)
from lib import training
from lib import evals
from lib.training import latest_step
from lib.util import *
from lib.types import *

# external 



# Code copied into .tmpcode/<sweep>/ at submission (bsub); jobs import only from these. Add files here if
# the experiment starts depending on others. Dependencies (pyproject.toml / uv.lock) are not frozen.
SNAPSHOT_PATHS = ["experiment.py", "lib", "mia_evals"]
CPUS_PER_GPU = 12  # LSF slots per GPU: 8 GPUs -> all 96 cores; training processes need cores beyond the data workers
# mia-evals scoring (score): CPU only. mia-evals: ~1 h and ~200 GB peak for the mutex watershed at 896^3, "give it a
# whole node". The shortest CPU queue; check its run limit and memory per slot with `bqueues -l short`.
SCORE_QUEUE, SCORE_SLOTS, SCORE_MINUTES = "local", 16, 4*60
SCORE_CONFIG = "mia_evals/gary_comparison_neuron_instance/mws.toml"  # ours: truth_kind = "instances"
# score() runs mia-evals from this sister checkout, with its own .venv (`uv sync` there) and its own commit, which
# mia-evals records in each score's provenance (mia_evals_commit). Not a dependency of this repo.
MIA_EVALS = Path.home() / "proj/mia-evals"

# This experiment's id (README: experiment ids): its roots are outdir/<EXPERIMENT>/d<i>/, a trunk's branches in its own dir.
EXPERIMENT = "e00/e12-cellmap-qknorm-lrdecay-12h"

def roots() -> list[Spec]:
    """This experiment: the runs it starts from nothing (trunks, other training runs, evals of another experiment's weights).
    Each trunk evaluates its own milestones as it saves them (eval_milestone)."""
    # e11's recipes again, with a bigger encoder context for the masked tokens, more data, qk_norm (the encoder's queries and keys
    # normalized) and the lr falling linearly from 1e-4 to 0 over total_steps (lr_decay): e11's probes peaked by 160k steps and fell,
    # d3 diverged at 1.2M and d0 went unstable at 1.06M, all with a constant lr. Masked tokens now see the MAE's context: a view of 96
    # in a region of 128 (shift 32, so 12^3 tokens a view), not 64 in 88. Data: d0-d2 (MAE, masked, views) on lmd5_pow04, e11's
    # lmd4 families and CellMap's organelle EM (lib.data "cellmap", ~10 Tvox: 48% of the samples) weighted by voxel count ^ 0.4;
    # d3 masked on lmd4_pow04, without CellMap, and d4 masked on CellMap alone, to tell what it adds. total_steps is the pace we
    # expect over 12 h less ~5%, so the lr reaches 0 as the budget ends (a slower run stops before it): e11's for MAE and views
    # (s/step 0.127 and 0.41); the masked runs' are guessed from e11's (masked 64 on lmd4, 7 H200s: 0.083; on B300: 0.23) x ~4 for
    # 3.4x the tokens a view and a bigger share of attention: `lsf benchmark` d1 and d3 first. Milestones are kept, their evals add
    # a val loss (val.json). d1 and d4 take B300 nodes (8 GPUs, global batch 256), the others 7 GPUs of H200 nodes (global batch 224).
    BATCH, HOURS = 32, 12

    def trunk(objective: Literal["views", "mask", "mae"], data: TrainData, input_size: int, view: int) -> Train:
        p = Train()
        p.objective = objective
        p.data = data
        p.views = "displace"
        p.patch_size = (input_size, input_size, input_size)
        p.global_size = (view, view, view)
        p.batch_size = BATCH
        p.lr_decay = True
        p.qk_norm = True
        p.milestone_every = 5_000
        p.milestone_doubling = True
        p.max_hours = HOURS
        p.weight_decay = 0.05
        p.adam_beta2 = 0.95
        p.cudagraphs = True
        p.n_workers = 11  # 12 cores per GPU
        p.n_gpus = 7  # H200 nodes with one GPU taken by someone else's job
        p.queue = "gpu_h200"
        return p

    def masked(data: TrainData) -> Train:
        p = trunk("mask", data, 128, 96)
        p.mask_block = 0.6
        p.mask_shift = 32
        p.sigreg_over = "both"
        p.lamb = 10 / BATCH
        return p

    mae = trunk("mae", "lmd5_pow04", 128, 96)
    mae.mae_decoder = (512, 8)
    views = trunk("views", "lmd5_pow04", 128, 96)
    views.local_size = (96, 96, 96)
    views.batch_views = True
    views.lamb = 0.1
    runs: list[Spec] = [mae, masked("lmd5_pow04"), views, masked("lmd4_pow04"), masked("cellmap")]
    for i in (1, 4):  # the two B300 nodes
        runs[i].n_gpus = 8
        runs[i].queue = "gpu_b300"
    for i, (p, steps) in enumerate(zip(runs, [325_000, 62_500, 100_000, 125_000, 62_500])):
        p.total_steps = steps
        p.savedir = f"outdir/{EXPERIMENT}/d{i}/"
    return runs

def milestone_of(p: Spec) -> str:
    """The milestone checkpoint of this experiment's own trunk that a branch starts from, the one it anneals or evaluates;
    "" for a trunk, and for an eval of another experiment's weights (init_from outside outdir/<EXPERIMENT>/)."""
    return source(p) if source(p).startswith(f"outdir/{EXPERIMENT}/") else ""

def eval_milestone(trunk: Train, step: int):
    """The trunk's own eval of its milestone at step (eval_params: pca and probe, in branch_dir), in this process between two
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
            pca(e)
            probe(e)
            val(e)
    except Exception:
        traceback.print_exc()
    torch.cuda.empty_cache()  # the eval's memory back for training (a no-op without CUDA)

def check_snapshot():
    """LSF jobs must run a code snapshot (bsub): the shared checkout's lib/ may be another sweep's by now."""
    b1 = "LSB_JOBID" in os.environ
    b2 = ".tmpcode" not in Path(__file__).resolve().parts
    assert not (b1 and b2), f"LSF job running {__file__} from the shared checkout; submit via submit() (code snapshot)"

def run(job: Train | str):
    """The LSF training job: train the spec job (or its train.json's), then rank 0 evaluates its last finite checkpoint (pca
    of own_eval, its val loss, and probe_evals).

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
        pca(own_eval(par))
        val(own_eval(par))
        for e in probe_evals(par):
            Path(e.savedir).mkdir(parents=True, exist_ok=True)
            probe(e)


def benchmark(job: Train | str):
    """Throughput and a profiler trace of the spec job's training step (or the train.json's it names), with no training run:
    lib.training.benchmark. Every DDP rank."""
    check_snapshot()
    par = load(job, Train)
    if int(os.environ.get("RANK", 0)) == 0:
        record(par, "benchmark")
    training.benchmark(par)

def replay_bad_batch(path: str):
    """Replay a skipped (non-finite gradient) training step from its bad_batch_*.pt dump: lib.training.replay_bad_batch. Like
    training, replay depends on lib/: only for dumps of this commit's own experiment (replaylsf submits it)."""
    import torch
    ours = Path("outdir", EXPERIMENT).resolve()
    assert Path(path).resolve().is_relative_to(ours), f"{path} isn't from this commit's experiment: replay it from its experiment's commit"
    dump = torch.load(path, map_location="cpu", weights_only=False)
    record(parse(Train, dump["params"]), "replay_bad_batch")
    return training.replay_bad_batch(dump)

def pca(job: Eval | str):
    """PCA maps (lib.evals.pca_maps) of the spec job's weights (or those of the eval.json it names). run() does it at the end of
    training; `lsf pca n` redoes it."""
    e = load(job, Eval)
    model, step = load_checkpoint(e)
    record(e, "pca")  # argv says whether it ran inside `run` or on its own
    evals.pca_maps(model.encoder, e.train, Path(e.savedir), step, e.probe_on)

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

def score(job: Eval | str):
    """Score probe(job)'s affinities as mia-evals' neuron-instance task (SCORE_CONFIG: mutex watershed + size filter,
    fitted on the fit block, reported on the test block; panoptic quality ranks, VOI and ARE reported).

    Writes the record into the run's own dir, savedir/mia_evals/<task>/records/*.json (pull.sh brings it down;
    analysis_plots.mia_evals_table reads it), not mia-evals' repo. resolved_config.json (params) and git_commit.txt (the
    commit that ran) go in the savedir first: mia-evals copies them into the record from --run-dir.
    Runs the sister checkout MIA_EVALS's own CLI and .venv, from that checkout (its commit is in the record), with this
    commit's SCORE_CONFIG (`lsf score n`). Deletes savedir/probe/ (the artifacts it read) once the record is written.
    """
    import subprocess
    par = load(job, Eval)
    savedir = Path(par.savedir)
    assert (savedir / "probe/test").is_dir() and (savedir / "probe/fit").is_dir(), f"no probe artifacts in {savedir}: probe first (a run with score=False writes none)"
    exe = MIA_EVALS / ".venv/bin/mia-evals"
    assert exe.is_file(), f"no mia-evals CLI at {exe}: clone github.com/AI-HHMI/mia-evals to {MIA_EVALS} and `uv sync` there"
    record(par, "score")
    (savedir / "resolved_config.json").write_text(json.dumps(asdict(par)) + "\n")
    (savedir / "git_commit.txt").write_text(code_provenance()["commit_id"] + "\n")
    config = Path(__file__).resolve().parent / SCORE_CONFIG  # the snapshot's copy when run as a job
    cmd = [str(exe), "score", str(config),
           "--test", str(savedir.resolve() / "probe/test"), "--val", str(savedir.resolve() / "probe/fit"),
           "--leaderboard", str(savedir.resolve() / "mia_evals"), "--run-dir", str(savedir.resolve()),
           "--no-scored"]  # don't keep the 5.8 GB post-processed labelling per run
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=MIA_EVALS)  # absolute paths: the cwd is mia-evals'
    shutil.rmtree(savedir / "probe")  # the artifacts are scored (the record is in mia_evals/): GBs per run, not kept

def job_name(savedir: str, job: str) -> str:
    """The LSF name of the run dir savedir's job: <sweep>-<dN>-<job>, e.g. e00-nanhunt-d9-run, so bjobs shows which is which."""
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
    here = Path(__file__).resolve().parent
    # A job submitting a job hands on the snapshot it runs itself, not the shared checkout's current state.
    code = here if ".tmpcode" in here.parts else snapshot(SNAPSHOT_PATHS, Path(".tmpcode") / EXPERIMENT.replace("/", "-"))
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

def submit(fn: str, job: Spec):
    """Run the entrypoint fn on the spec job as an LSF job: job written to its savedir (dump: train.json, eval.json or
    finetune.json), and `experiment.py <fn> <that file>` submitted (bsub), which loads it. Resources by fn: run and benchmark,
    a training job on job's queue and GPUs (torchrun); pca, pca_sizes, probe, finetune and tiles, one GPU on the trained run's queue;
    score, a CPU job on SCORE_QUEUE. A finetune of one of this experiment's own trunks waits in LSF for the trunk's run job,
    unless the trunk is done (its own probe.json is written). No commit check: callers from the shared checkout (submitall,
    lsf) assert_committed() first; a job submitting one is committed code."""
    b1 = fn in ("run", "benchmark") and isinstance(job, Train)
    b2 = fn in ("pca", "pca_sizes", "probe", "val", "score") and isinstance(job, Eval)
    b3 = fn in ("finetune", "tiles") and isinstance(job, Finetune)
    assert b1 or b2 or b3, (f"{job.savedir}: no LSF job {fn!r} for a {type(job).__name__} (run and benchmark take a Train; pca, pca_sizes, "
                            f"probe, val and score an Eval; finetune a Finetune)")
    assert not isinstance(job, Train) or job.n_workers + 1 <= CPUS_PER_GPU, f"n_workers={job.n_workers} leaves no core for the training process"
    spec = dump(job)
    if isinstance(job, Train):
        # Slack: startup, final checkpoint, pca + probe. A run without max_hours (a fixed step count) or a benchmark: 15 min.
        minutes = int(job.max_hours * 60) + 60 if job.max_hours and fn == "run" else 15
        bsub(job.savedir, fn, minutes, job.n_gpus, job.queue, f"torchrun --standalone --nproc_per_node={job.n_gpus} {{code}}/experiment.py {fn} {spec}")
    elif fn == "score":
        bsub(job.savedir, fn, SCORE_MINUTES, 0, SCORE_QUEUE, f"python {{code}}/experiment.py score {spec}", slots=SCORE_SLOTS)
    else:
        b4 = job.init_from.startswith(f"outdir/{EXPERIMENT}/")
        b5 = not (Path(job.init_from) / "probe.json").is_file()
        after = job_name(job.init_from, "run") if isinstance(job, Finetune) and b4 and b5 else ""
        bsub(job.savedir, fn, {"pca": 30, "pca_sizes": 60, "probe": 60, "val": 30, "finetune": 120, "tiles": 120}[fn], 1, job.train.queue, f"python {{code}}/experiment.py {fn} {spec}",
             after=after)

def start(p: Spec):
    """Submit p fresh: its savedir emptied (old contents -> outdir/.trash/), then its jobs: a training run (which evals itself
    after), an eval's jobs (Eval.jobs: pca and probe, unless it says otherwise), or a finetune."""
    trash(p.savedir)
    for fn in ["run"] if isinstance(p, Train) else ["finetune"] if isinstance(p, Finetune) else p.jobs:
        submit(fn, p)

def lsf(fn: str, target: int | str):
    """Redo one job on LSF, of roots()[target] or of the spec file target names (any run's, e.g. a milestone eval's
    outdir/.../d3/step_0005000/eval.json): `lsf probe 3` after changing probe. pca, probe, val or score of an eval, or of a training
    run's own checkpoint (own_eval), in place; finetune of a finetune, in place; benchmark of a training run, in place (its
    savedir gets a train.json, so submitall skips it: `lsf run` it after); run restarts a training run fresh (start), its
    milestone evals with it."""
    assert_committed()
    if isinstance(target, int):
        p = roots()[target]
    else:
        p = load(target, Eval if target.endswith("eval.json") else Finetune if target.endswith("finetune.json") else Train)
    if fn == "run":
        start(p)
    elif fn == "benchmark":
        submit(fn, p)
    else:
        submit(fn, own_eval(p) if isinstance(p, Train) else p)

def replaylsf(n:int):
    """replay_bad_batch on LSF for roots()[n]'s first bad_batch dump. CUDA_LAUNCH_BLOCKING names a crashing kernel."""
    par = roots()[n]
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

def check_experiment():
    """EXPERIMENT's id rules (README: experiment ids), checked against outdir/, the cluster's source of truth: the id is
    well formed and no other sweep dir has it; every run is under outdir/<EXPERIMENT>/; a child (e15-02) builds on its
    parent (e15), i.e. every init_from points into outdir/<series>/e15-*/; a root (e15) builds on no numbered experiment.
    One that depends on two experiments can't be named, by design."""
    series, name = EXPERIMENT.split("/")
    eid = experiment_id(name)
    assert eid, f"EXPERIMENT={EXPERIMENT!r}: want e<series>/e<NN>[-<NN>...]-<slug>, e.g. e00/e15-02-pca"
    sweeps = [d.name for d in Path("outdir", series).glob("*")]
    taken = [s for s in sweeps if experiment_id(s) == eid and s != name]
    assert not taken, f"{EXPERIMENT}: id {eid} is taken by {series}/{taken[0]}"
    params = roots()
    assert all(p.savedir.startswith(f"outdir/{EXPERIMENT}/") for p in params), f"every savedir must be under outdir/{EXPERIMENT}/"
    sources = [source(p) for p in params]
    deps = sorted({Path(s).parts[2] for s in sources if s not in ("", "random")} - {name})  # outdir/<series>/<sweep>/dN; not our own trunks
    parent = eid.rsplit("-", 1)[0] if "-" in eid else None
    if parent:
        parents = [s for s in sweeps if experiment_id(s) == parent]
        assert parents, f"{EXPERIMENT} builds on {series}/{parent}, which doesn't exist in outdir/"
        assert deps == parents, f"{EXPERIMENT} builds on {series}/{parents[0]}, but its init_from read {deps}"
    else:
        numbered = [d for d in deps if experiment_id(d)]
        assert not numbered, f"{EXPERIMENT} builds on {series}/{numbered[0]}: name it {experiment_id(numbered[0])}-<NN>-<slug>"

def submitall():
    """Submit this experiment (README: one experiment, one entrypoint): every root whose spec isn't in its savedir yet, i.e. not
    submitted before (whether it finished, is still going, or failed). Idempotent: run it again after adding roots; a run
    already submitted is left alone (`lsf run n` restarts one fresh)."""
    check_experiment()
    assert_committed()
    for r in roots():
        if not spec_path(r).is_file():
            start(r)

def retryall():
    """Resubmit this experiment's finetunes whose job ended without finishing (its LSF log has its footer, but there is no
    finetune.png, which finetune writes last), fresh (start: the failed dir's contents -> outdir/.trash/). Finished ones,
    and ones still queued or running, are left alone."""
    check_experiment()
    assert_committed()
    for r in roots():
        logs = sorted(Path(r.savedir).glob("job_finetune_*.log"))
        b1 = isinstance(r, Finetune) and not (Path(r.savedir) / "finetune.png").is_file()
        b2 = bool(logs) and "Resource usage summary" in logs[-1].read_text(errors="ignore")
        if b1 and b2:
            start(r)


ENTRYPOINTS = [run, benchmark, replay_bad_batch, pca, pca_sizes, probe, val, score, finetune, tiles, lsf, replaylsf, submitall, retryall]

if __name__ == "__main__":
    print(sys.argv)
    names = {f.__name__: f for f in ENTRYPOINTS}
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        for name, f in names.items():
            print(f"  {name}{inspect.signature(f)}  {(f.__doc__ or '').strip().splitlines()[0]}")
        sys.exit()
    assert sys.argv[1] in names, f"unknown entrypoint {sys.argv[1]!r}: {', '.join(names)}"
    log_command(sys.argv)  # commands issued outside jobs -> outdir/_log/commands.jsonl; jobs use runs.json
    names[sys.argv[1]](*[int(a) if a.isdigit() else a for a in sys.argv[2:]])  # `lsf probe 3`: the target is a roots() index
