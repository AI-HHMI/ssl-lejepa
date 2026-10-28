"""The tasks of an experiment, and the specs they take: what experiment.py's runs are made of. Three tasks, each defined with a spec
of its name beside it, each spec holding its parent's: Trunk(train: Train), Branch(trunk), Score(branch). A spec is a JSON-able
dataclass; a task is the entrypoint of one job (`python experiment.py <task> <spec json>`) that loads it, reads and writes the
files and checkpoints, submits the next tasks (go) and calls the library (lib/: straight-line code on loaded models and specs,
no files named, no callbacks) in its own process.

    trunk(Trunk):                                # one job, the run's GPUs, up to run.hours
        fork(); repeat run.segments times: train(run.segment_steps, lr, lr); fork()
        fork() = go branch(a checkpoint of the weights and optimizer)       # data: the train volumes' images
    branch(Branch):                              # a job on the run's GPUs, as many as the trunk's
        train(run.anneal_steps, lr, lr x run.anneal_to); pca on a val crop; tiled inference, a Linear (or a UNETR) fitted on the
        fit volume and predicting the test and tune volumes; go score(...)
    score(Score):                                # a CPU job, ~1 h, so the GPUs aren't held
        mia-evals on the test volume, its post-processing fitted on the tune volume; best.pt = the best score of the trunk's branches
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lib import evals, training
from lib.task_utils import SCORE_MINUTES, check_snapshot, go, keep_if_best, mia_eval, task
from lib.specs import Train, load_checkpoint, record
from lib.training import Trainer
from lib.util import load


def load_checkpoint_file(path: str) -> dict | None:
    """The checkpoint a Trainer starts from (what Trainer.save wrote): None for "", a fresh model."""
    import torch
    return torch.load(path, map_location="cpu", weights_only=False) if path else None


@dataclass(slots=True)
class Trunk:
    """Spec of task trunk: pretrain train for segments x segment_steps steps, with a branch at the start and after each segment. You
    write these: an Experiment's runs. It also says what its branches do."""
    savedir: str = "outdir/e00/scratch/"
    train: Train = field(default_factory=Train)
    init_from: str = ""  # the starting weights: "" random, or a checkpoint .pt (its weights and optimizer)
    segments: int = 10
    segment_steps: int = 10_000
    hours: float = 12.0  # the LSF wall limit: a trunk killed at it keeps the branches of its finished segments
    anneal_steps: int = 10_000  # a branch's steps, its lr falling linearly to train.optim.lr x anneal_to
    anneal_to: float = 0.01
    eval_data: str = ""  # the lib.data.DATASETS a branch is evaluated on: its val images, and its fit, test and tune blocks
    eval_head: Literal["linear", "unetr"] = "linear"  # the branch's tiled inference: a Linear on the frozen encoder, or a UNETR
    standardize: bool = False  # linear head: fit on features standardized per channel (fit block's mean and std; folded back into the Linear)
    board: Literal["neuron-instance"] = "neuron-instance"  # the mia-evals leaderboard (lib.task_utils.BOARDS) a branch is scored on


@task(Trunk, "gpus", lambda t: int(t.hours * 60), lambda t: t.train.hardware)
def trunk(job: Trunk | str):
    """Pretrain at a constant lr, with a branch at the start and after every segment (step_<N>/ each; metrics.json, best.pt)."""
    check_snapshot()
    t = load(job, Trunk)
    lr = t.train.optim.lr
    with Trainer(t.train, t.savedir, t.segments * t.segment_steps, load_checkpoint_file(t.init_from)) as trainer:
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


@dataclass(slots=True)
class Branch:
    """Spec of task branch: anneal a fork point of trunk, then pca and tiled inference on the result. Made by the trunk's fork."""
    savedir: str
    trunk: Trunk
    init_from: str  # the fork point: a checkpoint .pt of weights and optimizer


@task(Branch, "gpus", 120, lambda b: b.trunk.train.hardware)
def branch(job: Branch | str):
    """Anneal a fork point's weights, then pca and tiled inference on them, and go score (checkpoint.pt, pca2.png, probe.json or finetune_curve.json)."""
    check_snapshot()
    b = load(job, Branch)
    t = b.trunk
    lr = t.train.optim.lr
    end = Path(b.savedir) / "checkpoint.pt"
    with Trainer(t.train, b.savedir, t.anneal_steps, load_checkpoint_file(b.init_from)) as trainer:
        if trainer.r.rank0:
            record(b, "branch")
        trainer.train(t.anneal_steps, lr, lr * t.anneal_to)
        if trainer.r.rank0:
            trainer.save(end)
    if trainer.r.rank0:  # the other ranks are done
        Path(b.init_from).unlink()
        model, step = load_checkpoint(t.train, str(end))  # a fresh eager model: the compiled one is gone with the trainer
        evals.pca_maps(model.encoder, t.train, Path(b.savedir), step, t.eval_data)
        if t.eval_head == "linear":
            evals.probe(model, t.train, t.eval_data, b.savedir, step, t.standardize)
        else:
            evals.finetune(model, t.train, t.eval_data, b.savedir, step)
        go(score, Score(savedir=b.savedir, branch=b))


@dataclass(slots=True)
class Score:
    """Spec of task score: mia-evals on the weights and affinities its branch wrote (savedir/checkpoint.pt, savedir/probe/). Made by the branch."""
    savedir: str  # the branch's
    branch: Branch


@task(Score, "cpu", SCORE_MINUTES)
def score(job: Score | str):
    """mia-evals on a branch's predictions; its weights are kept if they score best in the trunk (mia_evals/<board>/records/*.json, ../best.pt)."""
    s = load(job, Score)
    record(s, "score")
    keep_if_best(s, mia_eval(s))


@task(Trunk, "gpus", 15, lambda t: t.train.hardware)
def benchmark(job: Trunk | str):
    """Throughput and a profiler trace of the trunk's training step (or of the trunk.json it names), with no training run:
    lib.training.benchmark. Every DDP rank."""
    check_snapshot()
    t = load(job, Trunk)
    if int(os.environ.get("RANK", 0)) == 0:
        record(t, "benchmark")
    training.benchmark(t.train, t.savedir)
