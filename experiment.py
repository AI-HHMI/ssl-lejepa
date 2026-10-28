"""e13-cellmap-12h: e12's six runs in this format (e12's are named d0-d5). Do CellMap's organelle EM, the masked cube's size (7^3 tokens,
e12's, or e11's 5^3) and the training objective change what 12 h of pretraining gives a linear probe on hemibrain?

Each run is a Trunk, which branches at its start and after every segment; a branch is scored, and its trunk keeps the best. Three
tasks, each with a spec of its name (lib.specs): Trunk(train: Train), Branch(trunk), Score(branch).

    trunk(Trunk):                                # one job, the run's GPUs, up to run.hours
        fork(); repeat run.segments times: train(run.segment_steps, lr, lr); fork()
        fork() = go branch(a checkpoint of the weights and optimizer)       # data: the train volumes' images
    branch(Branch):                              # a job on the run's GPUs, as many as the trunk's
        train(run.anneal_steps, lr, lr x run.anneal_to); pca on a val crop; tiled inference, a Linear (or a UNETR) fitted on the
        fit volume and predicting the test and tune volumes; go score(...)
    score(Score):                                # a CPU job, ~1 h, so the GPUs aren't held
        mia-evals on the test volume, its post-processing fitted on the tune volume; best.pt = the best score of the trunk's branches

Every run shares BASE (its Train) and differs by the groups its line names. Run it: `sh jrun.sh exp experiment.py submitall`, then
`lsf <task> <run or spec file>`; `python experiment.py -h` lists the commands and tasks."""

from dataclasses import replace
from functools import partial
from pathlib import Path

from lib import evals
from lib.jobs import B300, H200, SCORE_MINUTES, Experiment, check_snapshot, go, keep_if_best, main, mia_eval, task
from lib.specs import Branch, Data, Encoder, Mae, Masked, Optim, Score, Speed, Train, Trunk, Views, load_checkpoint, record
from lib.training import Trainer
from lib.util import load

DATA = ("hemibrain", "fish2", "liconn_mouse", "microns", "cellmap")
BATCH = 32  # per GPU
mask = partial(Masked, shift=32, lamb=10 / BATCH, sigreg_over="both")  # the targets are a second view, shifted up to 32 voxels from the context's

BASE = Train(
    data=Data(sets=DATA, power=0.4, window=(128, 128, 128), view=(96, 96, 96)),
    encoder=Encoder(qk_norm=True),
    optim=Optim(batch_size=BATCH, weight_decay=0.05, adam_beta2=0.95),
    speed=Speed(cudagraphs=True),
)
run = partial(Trunk, eval_data="hemibrain", segments=10, anneal_steps=10_000, anneal_to=0.01, hours=12)


@task(Trunk, "gpus", lambda t: int(t.hours * 60))
def trunk(job: Trunk | str):
    """Pretrain at a constant lr, with a branch at the start and after every segment (step_<N>/ each; metrics.json, best.pt)."""
    check_snapshot()
    t = load(job, Trunk)
    lr = t.train.optim.lr
    with Trainer(t.train, t.savedir, t.segments * t.segment_steps, t.init_from) as trainer:
        def fork():
            if trainer.r.rank0:  # the branch's start: weights and optimizer, deleted by the branch that took them
                start = Path(t.savedir) / f"step_{trainer.step:07d}/start.pt"
                trainer.save(start)
                go(branch, Branch(savedir=f"{start.parent}/", trunk=t, init_from=str(start)))
        if trainer.r.rank0:
            record(t, "trunk")
        fork()
        for _ in range(t.segments):
            trainer.train(t.segment_steps, lr, lr)
            if trainer.stopped:
                break  # diverged: no branch of its weights
            fork()


@task(Branch, "gpus", 120)
def branch(job: Branch | str):
    """Anneal a fork point's weights, then pca and tiled inference on them, and go score (checkpoint.pt, pca2.png, probe.json or finetune_curve.json)."""
    check_snapshot()
    b = load(job, Branch)
    t = b.trunk
    lr = t.train.optim.lr
    end = Path(b.savedir) / "checkpoint.pt"
    with Trainer(t.train, b.savedir, t.anneal_steps, b.init_from) as trainer:
        if trainer.r.rank0:
            record(b, "branch")
        trainer.train(t.anneal_steps, lr, lr * t.anneal_to)
        if trainer.r.rank0:
            trainer.save(end)
    if trainer.r.rank0:  # the other ranks are done
        Path(b.init_from).unlink()
        model, step = load_checkpoint(t.train, str(end))  # a fresh eager model: the compiled one is gone with the trainer
        evals.pca_maps(model.encoder, t.train, Path(b.savedir), step, t.eval_data)
        (evals.probe if t.eval_head == "linear" else evals.finetune)(b, model, step)
        go(score, Score(savedir=b.savedir, branch=b))


@task(Score, "cpu", SCORE_MINUTES)
def score(job: Score | str):
    """mia-evals on a branch's predictions; its weights are kept if they score best in the trunk (mia_evals/<board>/records/*.json, ../best.pt)."""
    s = load(job, Score)
    record(s, "score")
    keep_if_best(s, mia_eval(s))


EXPERIMENT = Experiment("e00/e13-cellmap-12h", trunk, {
    "mae":          run(train=BASE.but(model=Mae(decoder=(512, 8)),                       hardware=H200), segment_steps=30_000),
    "views":        run(train=BASE.but(model=Views(local=96, batch_views=True, lamb=0.1), hardware=H200), segment_steps=10_000),
    "mask7":        run(train=BASE.but(model=mask(cube=56),                               hardware=B300), segment_steps=6_250),
    "mask5":        run(train=BASE.but(model=mask(cube=40),                               hardware=B300), segment_steps=6_250),
    "mask7-nocell": run(train=BASE.but(model=mask(cube=56),                               hardware=H200, data=replace(BASE.data, sets=DATA[:-1])), segment_steps=12_500),
    "mask7-cell":   run(train=BASE.but(model=mask(cube=56),                               hardware=B300, data=replace(BASE.data, sets=("cellmap",))), segment_steps=6_250),
})

if __name__ == "__main__":
    main(EXPERIMENT)
