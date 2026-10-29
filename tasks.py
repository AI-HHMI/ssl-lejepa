"""The tasks of an experiment, and the Trunk they take: what experiment.py's runs are made of. Three tasks, plain functions with
JSON-able arguments (dataclasses, strings): trunk(t), branch(t, start) and score(t, branch_dir). Each is a job's entrypoint
(`python experiment.py <task> <job file>`, see lib.task_utils.go): it reads and writes the files and checkpoints, submits the next
task (go: its arguments, savedir, hardware and wall time) and calls the library (lib/: straight-line code on loaded models and
specs, no files named, no callbacks) in its own process.

    trunk(t):                                    # one job, the run's GPUs, up to t.hours
        fork(); repeat t.segments times: train(t.segment_steps, lr, lr); fork()
        fork() = go branch(t, a checkpoint of the weights and optimizer)    # data: the train volumes' images
    branch(t, start):                            # a job on the run's GPUs, as many as the trunk's
        train(t.anneal_steps, lr, lr x t.anneal_to); pca on a val crop; tiled inference, a Linear (or a UNETR) fitted on the
        fit volume and predicting the test and tune volumes; go score(t, the branch's dir)
    score(t, branch_dir):                        # a CPU job, ~1 h, so the GPUs aren't held
        mia-evals on the test volume, its post-processing fitted on the tune volume; best.pt = the best score of the trunk's branches
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lib import evals, training
from lib.task_utils import SCORE_MINUTES, check_snapshot, go, keep_if_best, load_model, mia_eval, record
from lib.training import Train, Trainer

BRANCH_MINUTES = 120  # the LSF wall limit of a branch (anneal, pca, probe or UNETR)


def load_checkpoint(path: str) -> dict | None:
    """The checkpoint a Trainer starts from (what Trainer.save wrote): None for "", a fresh model."""
    import torch
    return torch.load(path, map_location="cpu", weights_only=False) if path else None


@dataclass(slots=True)
class Trunk:
    """What a run is: pretrain train for segments x segment_steps steps, with a branch at the start and after each segment. You write
    these: an Experiment's runs. It also says what its branches do."""
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
    board: Literal["neuron-instance"] = "neuron-instance"  # the mia-evals leaderboard (lib.task_utils.BOARD_CONFIGS) a branch is scored on


def trunk(t: Trunk):
    """Pretrain at a constant lr, with a branch at the start and after every segment (step_<N>/ each; metrics.json, best.pt)."""
    check_snapshot()
    with Trainer(t.train, t.savedir, t.segments * t.segment_steps, load_checkpoint(t.init_from)) as trainer:
        def fork():
            start = Path(t.savedir) / f"step_{trainer.step:07d}/start.pt"
            trainer.save(start)
            go(branch, f"{start.parent}/", t.train.hardware, BRANCH_MINUTES, t=t, start=str(start))
        if trainer.r.rank0:
            record(t.savedir, "trunk", t=t)
            fork()
        lr = t.train.optim.lr
        for _ in range(t.segments):
            trainer.train(t.segment_steps, lr, lr)
            if trainer.stopped: break  # diverged: no branch of its weights
            if trainer.r.rank0: fork()


def branch(t: Trunk, start: str):
    """Anneal a fork point's weights (start, in the branch's dir), then pca and tiled inference on them, and go score (checkpoint.pt, pca2.png, probe.json or finetune_curve.json)."""
    check_snapshot()
    savedir, lr = f"{Path(start).parent}/", t.train.optim.lr
    end = Path(savedir) / "checkpoint.pt"
    with Trainer(t.train, savedir, t.anneal_steps, load_checkpoint(start)) as trainer:
        if trainer.r.rank0:
            record(savedir, "branch", t=t, start=start)
        trainer.train(t.anneal_steps, lr, lr * t.anneal_to)
        if trainer.r.rank0:
            trainer.save(end)
    if not trainer.r.rank0: return  # the other ranks are done
    Path(start).unlink()
    model, step = load_model(t.train, str(end))  # a fresh eager model: the compiled one is gone with the trainer
    evals.pca_maps(model.encoder, t.train, Path(savedir), step, t.eval_data)
    if t.eval_head == "linear":
        evals.probe(model, t.train, t.eval_data, savedir, step, t.standardize)
    else:
        evals.finetune(model, t.train, t.eval_data, savedir, step)
    go(score, savedir, None, SCORE_MINUTES, t=t, branch_dir=savedir)


def score(t: Trunk, branch_dir: str):
    """mia-evals on a branch's predictions; its weights are kept if they score best in the trunk (mia_evals/<board>/records/*.json, ../best.pt)."""
    record(branch_dir, "score", t=t, branch_dir=branch_dir)
    keep_if_best(t, branch_dir, mia_eval(t, branch_dir))


def benchmark(t: Trunk):
    """Throughput and a profiler trace of the trunk's training step, with no training run: lib.training.benchmark. Every DDP rank."""
    check_snapshot()
    if int(os.environ.get("RANK", 0)) == 0:
        record(t.savedir, "benchmark", t=t)
    training.benchmark(t.train, t.savedir)

TASKS = [trunk, branch, score, benchmark]  # the entrypoints of experiment.py: `<task> <job file>`
