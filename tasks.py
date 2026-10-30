"""The tasks of an experiment, and the Trunk they take: what experiment.py's runs are made of. Four tasks, plain functions with
JSON-able arguments (dataclasses, strings): trunk(t), anneal(t, fork_point), finetune(t, branch_dir) and score(t, branch_dir). Each is
a job's entrypoint (`python experiment.py <task> <job file>`, see lib.task_utils.go): it reads and writes the files and checkpoints,
submits the next task (go: its arguments, savedir, hardware and wall time, from the block below) and calls the library (lib/:
straight-line code on loaded models and a Train, no files named, no callbacks) in its own process.

    trunk(t):                                    # one job, up to TRUNK_MINUTES
        fork(); repeat t.segments times: train(t.segment_steps, lr, lr); fork()
        fork() = go anneal(t, a checkpoint of the weights and optimizer)    # data: the train volumes' images
    anneal(t, fork_point):                       # as many GPUs as the trunk
        train(t.anneal_steps, lr, lr x t.anneal_lr_frac); go finetune(t, the branch's dir)
    finetune(t, branch_dir):                     # one small GPU: the encoder is frozen or the head is small
        pca on a val crop; tiled inference, a Linear (or a UNETR) fitted on the fit volume and predicting the test and tune
        volumes; go score(t, the branch's dir)
    score(t, branch_dir):                        # a CPU job, ~1 h, so the GPUs aren't held
        mia-evals on the test volume, its post-processing fitted on the tune volume; best.pt = the best score of the trunk's branches
"""

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from lib import evals
from lib.scoring import keep_if_best, mia_pq
from lib.task_utils import B300, RTX6000, assert_snapshot, go, record_job
from lib.training import Hardware, Train, Trainer, make_model

# Where each task runs and its LSF wall limit (a trunk killed at it keeps the branches of its finished segments). The only place
# hardware is chosen: each task's Train gets its trunk's or anneal's Hardware (what its data workers and ranks are).
TRUNK_HARDWARE, TRUNK_MINUTES = B300, 12 * 60
ANNEAL_HARDWARE, ANNEAL_MINUTES = B300, 2 * 60
FINETUNE_HARDWARE, FINETUNE_MINUTES = RTX6000, 2 * 60  # pca, then a Linear or a UNETR
# A score is CPU only. mia-evals: ~1 h and ~200 GB peak for the mutex watershed at 896^3, "give it a whole node". The shortest CPU
# queue; check its run limit and memory per slot with `bqueues -l short`.
SCORE_HARDWARE, SCORE_MINUTES = Hardware(queue="local", n_gpus=0, slots=16), 4 * 60


def load_checkpoint(path: str) -> dict | None:
    """The checkpoint a Trainer starts from (what Trainer.save wrote): None for "", a fresh model."""
    import torch
    return torch.load(path, map_location="cpu", weights_only=False) if path else None


def load_model(train: Train, init_from: str):
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


@dataclass(slots=True)
class Trunk:
    """What a run is: pretrain train for segments x segment_steps steps, with a branch at the start and after each segment. You write
    these: an Experiment's runs. It also says what its branches do."""
    savedir: str = "outdir/e00/scratch"
    train: Train = field(default_factory=Train)
    init_from: str = ""  # the starting weights: "" random, or a checkpoint .pt (its weights and optimizer): to extend a run, the last.pt of its trunk
    segments: int = 10
    segment_steps: int = 10_000
    anneal_steps: int = 10_000  # a branch's steps, its lr falling linearly to train.optim.lr x anneal_lr_frac
    anneal_lr_frac: float = 0.01  # the branch's final lr, as a fraction of train.optim.lr
    eval_data: str = ""  # the lib.data.DATASETS a branch is evaluated on: its val images, and its fit, test and tune blocks
    eval_head: Literal["probe", "finetune"] = "probe"  # the branch's tiled inference: evals.probe (a Linear on the frozen encoder) or evals.finetune (a UNETR)
    standardize: bool = False  # linear head: fit on features standardized per channel (fit block's mean and std; folded back into the Linear)
    board: Literal["neuron-instance"] = "neuron-instance"  # the mia-evals leaderboard (lib.task_utils.BOARD_CONFIGS) a branch is scored on


def start(t: Trunk):
    """Submit a run: its trunk."""
    go(trunk, t.savedir, TRUNK_HARDWARE, TRUNK_MINUTES, t=t)


def trunk(t: Trunk):
    """Pretrain at a constant lr, with a branch at the start (unless it continues an init_from) and after every segment (step_<N>/ each;
    metrics.json, best.pt). last.pt is the latest fork point's weights and optimizer, a run's state to continue from."""
    assert_snapshot()
    train = replace(t.train, hardware=TRUNK_HARDWARE)
    with Trainer(train, t.savedir, t.segments * t.segment_steps, load_checkpoint(t.init_from)) as trainer:
        def fork():
            fork_point = Path(t.savedir) / f"step_{trainer.step:07d}/fork.pt"
            trainer.save(fork_point)
            last = Path(t.savedir) / "last.pt"  # a hard link, so it survives the anneal taking fork_point
            os.link(fork_point, last.with_suffix(".tmp"))
            os.replace(last.with_suffix(".tmp"), last)
            go(anneal, str(fork_point.parent), ANNEAL_HARDWARE, ANNEAL_MINUTES, t=t, fork_point=str(fork_point))
        if trainer.r.rank0:
            record_job(t.savedir, "trunk", t=t)
            if not t.init_from: fork()  # a continued trunk starts at a state its source already branched at
        lr = t.train.optim.lr
        for _ in range(t.segments):
            trainer.train(t.segment_steps, lr, lr)
            if trainer.stop_reason: break  # diverged: no branch of its weights
            if trainer.r.rank0: fork()


def anneal(t: Trunk, fork_point: str):
    """Anneal the weights saved at fork_point (in the branch's dir), and go finetune (checkpoint.pt)."""
    assert_snapshot()
    branch_dir, lr = str(Path(fork_point).parent), t.train.optim.lr
    end = Path(branch_dir) / "checkpoint.pt"
    train = replace(t.train, hardware=ANNEAL_HARDWARE)
    with Trainer(train, branch_dir, t.anneal_steps, load_checkpoint(fork_point)) as trainer:
        if trainer.r.rank0:
            record_job(branch_dir, "anneal", t=t, fork_point=fork_point)
        trainer.train(t.anneal_steps, lr, lr * t.anneal_lr_frac)
        if trainer.r.rank0:
            trainer.save(end)
    if not trainer.r.rank0: return  # the other ranks are done
    Path(fork_point).unlink()
    go(finetune, branch_dir, FINETUNE_HARDWARE, FINETUNE_MINUTES, t=t, branch_dir=branch_dir)


def finetune(t: Trunk, branch_dir: str):
    """pca and tiled inference on a branch's annealed weights, and go score (pca2.png, probe.json or finetune_curve.json)."""
    assert_snapshot()
    record_job(branch_dir, "finetune", t=t, branch_dir=branch_dir)
    train = replace(t.train, hardware=FINETUNE_HARDWARE)
    model, step = load_model(train, str(Path(branch_dir) / "checkpoint.pt"))
    evals.pca_maps(model.encoder, train, Path(branch_dir), step, t.eval_data)
    if t.eval_head == "probe":
        evals.probe(model, train, t.eval_data, branch_dir, step, t.standardize)
    else:
        evals.finetune(model, train, t.eval_data, branch_dir, step)
    go(score, branch_dir, SCORE_HARDWARE, SCORE_MINUTES, t=t, branch_dir=branch_dir)


def score(t: Trunk, branch_dir: str):
    """mia-evals on a branch's predictions; its weights are kept if they score best in the trunk (mia_evals/<board>/records/*.json, ../best.pt)."""
    record_job(branch_dir, "score", t=t, branch_dir=branch_dir)
    keep_if_best(t, branch_dir, mia_pq(t, branch_dir))


TASKS = [trunk, anneal, finetune, score]  # the entrypoints of experiment.py: `<task> <job file>`
