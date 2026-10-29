"""The tasks of an experiment, and the Trunk they take: what experiment.py's runs are made of. Three tasks, plain functions with
JSON-able arguments (dataclasses, strings): trunk(t), branch(t, fork_point) and score(t, branch_dir). Each is a job's entrypoint
(`python experiment.py <task> <job file>`, see lib.task_utils.go): it reads and writes the files and checkpoints, submits the next
task (go: its arguments, savedir, hardware and wall time) and calls the library (lib/: straight-line code on loaded models and
a Train, no files named, no callbacks) in its own process.

    trunk(t):                                    # one job, the run's GPUs, up to t.hours
        fork(); repeat t.segments times: train(t.segment_steps, lr, lr); fork()
        fork() = go branch(t, a checkpoint of the weights and optimizer)    # data: the train volumes' images
    branch(t, fork_point):                       # a job on the run's GPUs, as many as the trunk's
        train(t.anneal_steps, lr, lr x t.anneal_lr_frac); pca on a val crop; tiled inference, a Linear (or a UNETR) fitted on the
        fit volume and predicting the test and tune volumes; go score(t, the branch's dir)
    score(t, branch_dir):                        # a CPU job, ~1 h, so the GPUs aren't held
        mia-evals on the test volume, its post-processing fitted on the tune volume; best.pt = the best score of the trunk's branches
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lib import evals, training
from lib.scoring import keep_if_best, mia_pq
from lib.task_utils import assert_snapshot, go, record_job
from lib.training import Hardware, Train, Trainer, make_model

BRANCH_MINUTES = 120  # the LSF wall limit of a branch (anneal, pca, probe or UNETR)
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
    init_from: str = ""  # the starting weights: "" random, or a checkpoint .pt (its weights and optimizer)
    segments: int = 10
    segment_steps: int = 10_000
    hours: float = 12.0  # the LSF wall limit: a trunk killed at it keeps the branches of its finished segments
    anneal_steps: int = 10_000  # a branch's steps, its lr falling linearly to train.optim.lr x anneal_lr_frac
    anneal_lr_frac: float = 0.01  # the branch's final lr, as a fraction of train.optim.lr
    eval_data: str = ""  # the lib.data.DATASETS a branch is evaluated on: its val images, and its fit, test and tune blocks
    eval_head: Literal["probe", "finetune"] = "probe"  # the branch's tiled inference: evals.probe (a Linear on the frozen encoder) or evals.finetune (a UNETR)
    standardize: bool = False  # linear head: fit on features standardized per channel (fit block's mean and std; folded back into the Linear)
    board: Literal["neuron-instance"] = "neuron-instance"  # the mia-evals leaderboard (lib.task_utils.BOARD_CONFIGS) a branch is scored on


def trunk(t: Trunk):
    """Pretrain at a constant lr, with a branch at the start and after every segment (step_<N>/ each; metrics.json, best.pt)."""
    assert_snapshot()
    with Trainer(t.train, t.savedir, t.segments * t.segment_steps, load_checkpoint(t.init_from)) as trainer:
        def fork():
            fork_point = Path(t.savedir) / f"step_{trainer.step:07d}/fork.pt"
            trainer.save(fork_point)
            go(branch, str(fork_point.parent), t.train.hardware, BRANCH_MINUTES, t=t, fork_point=str(fork_point))
        if trainer.r.rank0:
            record_job(t.savedir, "trunk", t=t)
            fork()
        lr = t.train.optim.lr
        for _ in range(t.segments):
            trainer.train(t.segment_steps, lr, lr)
            if trainer.stop_reason: break  # diverged: no branch of its weights
            if trainer.r.rank0: fork()


def branch(t: Trunk, fork_point: str):
    """Anneal the weights saved at fork_point (in the branch's dir), then pca and tiled inference on them, and go score (checkpoint.pt, pca2.png, probe.json or finetune_curve.json)."""
    assert_snapshot()
    branch_dir, lr = str(Path(fork_point).parent), t.train.optim.lr
    end = Path(branch_dir) / "checkpoint.pt"
    with Trainer(t.train, branch_dir, t.anneal_steps, load_checkpoint(fork_point)) as trainer:
        if trainer.r.rank0:
            record_job(branch_dir, "branch", t=t, fork_point=fork_point)
        trainer.train(t.anneal_steps, lr, lr * t.anneal_lr_frac)
        if trainer.r.rank0:
            trainer.save(end)
    if not trainer.r.rank0: return  # the other ranks are done
    Path(fork_point).unlink()
    model, step = load_model(t.train, str(end))  # a fresh eager model: the compiled one is gone with the trainer
    evals.pca_maps(model.encoder, t.train, Path(branch_dir), step, t.eval_data)
    if t.eval_head == "probe":
        evals.probe(model, t.train, t.eval_data, branch_dir, step, t.standardize)
    else:
        evals.finetune(model, t.train, t.eval_data, branch_dir, step)
    go(score, branch_dir, SCORE_HARDWARE, SCORE_MINUTES, t=t, branch_dir=branch_dir)


def score(t: Trunk, branch_dir: str):
    """mia-evals on a branch's predictions; its weights are kept if they score best in the trunk (mia_evals/<board>/records/*.json, ../best.pt)."""
    record_job(branch_dir, "score", t=t, branch_dir=branch_dir)
    keep_if_best(t, branch_dir, mia_pq(t, branch_dir))


def benchmark(t: Trunk):
    """Throughput and a profiler trace of the trunk's training step, with no training run: lib.training.benchmark. Every DDP rank."""
    assert_snapshot()
    if int(os.environ.get("RANK", 0)) == 0:
        record_job(t.savedir, "benchmark", t=t)
    training.benchmark(t.train, t.savedir)

TASKS = [trunk, branch, score, benchmark]  # the entrypoints of experiment.py: `<task> <job file>`
