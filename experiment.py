"""e13-cellmap-12h: e12's six runs in this format (e12's are named d0-d5). Do CellMap's organelle EM, the masked cube's size (7^3 tokens,
e12's, or e11's 5^3) and the training objective change what 12 h of pretraining gives a linear probe on hemibrain?

Each run is a trunk, which branches at its start and after every segment; a branch is scored, and its trunk keeps the best:

    trunk(run):                                  # one job, the run's GPUs, up to run.hours
        fork(); repeat run.segments times: train(run.segment_steps, lr, lr); fork()
        fork() = go branch(a checkpoint of the weights and optimizer)       # on pretraining images (data: train)
    branch(checkpoint):                          # a job on the run's GPUs, as many as the trunk's
        train(run.anneal_steps, lr, lr x run.anneal_to); pca on a val crop; tiled inference, a Linear (or a UNETR) fitted on the
        fit volume and predicting the test and tune volumes; go scores(...)
    scores(branch):                              # a CPU job, ~1 h, so the GPUs aren't held
        mia-evals on the test volume, its post-processing fitted on the tune volume; best.pt = the best score of the trunk's branches

Every run shares BASE and differs by the fields its line names. Run it: `sh jrun.sh exp experiment.py submitall`, then
`lsf <phase> <run or spec file>`; `python experiment.py -h` lists the commands and phases with their hardware."""

from dataclasses import replace
from functools import partial
from pathlib import Path

from lib import evals
from lib.jobs import B300, CPUS_PER_GPU, H200, SCORE_MINUTES
from lib.jobs import Experiment, check_snapshot, go, keep_if_best, main, mia_eval, task
from lib.specs import Eval, Finetune, Mae, Masked, Train, Views, load_checkpoint, record
from lib.training import Trainer
from lib.util import load

DATA = ("hemibrain", "fish2", "liconn_mouse", "microns", "cellmap")
BATCH = 32  # per GPU
mask = partial(Masked, shift=32, lamb=10 / BATCH, sigreg_over="both")  # the targets are a second view, shifted up to 32 voxels from the context's

BASE = Train(
    data=DATA, data_power=0.4, eval_on="hemibrain",
    patch_size=(128, 128, 128), global_size=(96, 96, 96), batch_size=BATCH, n_workers=CPUS_PER_GPU - 1,
    qk_norm=True, weight_decay=0.05, adam_beta2=0.95, cudagraphs=True,
    segments=10, anneal_steps=10_000, anneal_to=0.01, hours=12,
)


@task(Train, "gpus", lambda p: int(p.hours * 60))
def trunk(job: Train | str):
    """Pretrain at a constant lr, with a branch at the start and after every segment (step_<N>/ each; metrics.json, best.pt)."""
    check_snapshot()
    par = load(job, Train)
    with Trainer(par, par.segments * par.segment_steps, par.init_from) as t:
        def fork():
            if t.r.rank0:  # the branch's start: weights and optimizer, deleted by the branch that took them
                start = Path(par.savedir) / f"step_{t.step:07d}/start.pt"
                t.save(start)
                go(branch, Eval(savedir=f"{start.parent}/", train=par, init_from=str(start), data=par.eval_on))
        if t.r.rank0:
            record(par, "trunk")
        fork()
        for _ in range(par.segments):
            t.train(par.segment_steps, par.lr, par.lr)
            if t.stopped:
                break  # diverged: no branch of its weights
            fork()


@task(Eval, "gpus", 120)
def branch(job: Eval | str):
    """Anneal a branch point's weights, then pca and tiled inference on them, and go scores (checkpoint.pt, pca2.png, probe.json or finetune_curve.json)."""
    check_snapshot()
    e = load(job, Eval)
    par = e.train
    end = Path(e.savedir) / "checkpoint.pt"
    with Trainer(replace(par, savedir=e.savedir), par.anneal_steps, e.init_from) as t:
        if t.r.rank0:
            record(e, "branch")
        t.train(par.anneal_steps, par.lr, par.lr * par.anneal_to)
        if t.r.rank0:
            t.save(end)
    if t.r.rank0:  # the other ranks are done
        Path(e.init_from).unlink()
        e = replace(e, init_from=str(end), score=True)
        model, step = load_checkpoint(e)  # a fresh eager model: the compiled one is gone with the trainer
        evals.pca_maps(model.encoder, par, Path(e.savedir), step, e.data)
        if par.branch_eval == "probe":
            evals.probe(e, model, step)
            go(scores, e)
        else:
            f = Finetune(savedir=e.savedir, train=par, init_from=e.init_from, data=e.data, score=True)
            evals.finetune(f, model, step)
            go(scores, f)


@task((Eval, Finetune), "cpu", SCORE_MINUTES)
def scores(job: Eval | Finetune | str):
    """mia-evals on a branch's predictions; its weights are kept if they score best in the trunk (mia_evals/<board>/records/*.json, ../best.pt)."""
    par = load(job, Finetune if str(job).endswith("finetune.json") or isinstance(job, Finetune) else Eval)
    record(par, "scores")
    keep_if_best(par, mia_eval(par))


EXPERIMENT = Experiment("e00/e13-cellmap-12h", trunk, {
    "mae":          BASE.but(model=Mae(decoder=(512, 8)),                       segment_steps=30_000, **H200),
    "views":        BASE.but(model=Views(local=96, batch_views=True, lamb=0.1), segment_steps=10_000, **H200),
    "mask7":        BASE.but(model=mask(cube=56),                               segment_steps=6_250,  **B300),
    "mask5":        BASE.but(model=mask(cube=40),                               segment_steps=6_250,  **B300),
    "mask7-nocell": BASE.but(model=mask(cube=56),                               segment_steps=12_500, **H200, data=DATA[:-1]),
    "mask7-cell":   BASE.but(model=mask(cube=56),                               segment_steps=6_250,  **B300, data=("cellmap",)),
})

if __name__ == "__main__":
    main(EXPERIMENT)
