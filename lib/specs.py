"""The specs of a job (Train, Eval, Finetune: what an experiment's runs are and lib.jobs submits), what follows from one
(its model and the milestones it keeps), and the weights it loads."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Literal

from lib.encoders import EncoderConfig
from lib import models
from lib.views import ViewConfig
from lib.util import *
from lib.types import *


# The pretraining objectives: each its own knobs, whole. Train.model is one of them.
@dataclass(slots=True)
class Mae:
    """A masked autoencoder, the pixel-space baseline (lib.models.Mae), on the view Train.global_size."""
    decoder: tuple[int, int] = (192, 2)  # the decoder's (width, depth), heads width // 32
    kind: Literal["mae"] = "mae"


@dataclass(slots=True)
class Masked:
    """Masked-token prediction (lib.models.Lejepa): a cube of the view's tokens is hidden from the encoder and a predictor makes
    their embeddings, loss = prediction error + lamb x SIGReg. The view is Train.global_size, plus shift (both views' region)."""
    cube: int = 48  # the masked cube's side, voxels (a multiple of the patch)
    shift: int = 0  # >0: the targets are a second view's tokens, up to this many voxels off the context's (a multiple of the patch)
    aligned: bool = False  # shift only: whole-token view origins (lib.models.MaskConfig.aligned)
    expand: bool = False  # shift only: hide a token more of the context around the masked target tokens (MaskConfig.expand)
    # SIGReg's pull is lamb x its sample count (its statistic is N x an empirical-CF distance); it sees each view's N tokens.
    lamb: float = 0.002
    sigreg_over: Literal["view", "batch", "both"] = "view"  # SIGReg's samples: each view's tokens, each position's batch, or both
    stop_grad: bool = False  # detach the targets (lib.models.LejepaConfig.stop_grad)
    kind: Literal["mask"] = "mask"


@dataclass(slots=True)
class Views:
    """LeJEPA's multi-view invariance (lib.models.Lejepa): 2 global views of Train.global_size and 4 local ones, each pulled to the
    mean of the globals, loss = invariance + lamb x SIGReg."""
    local: int = 56  # the local views' side, voxels
    batch_views: bool = False  # one encoder call per group of same-shape views
    lamb: float = 0.002
    kind: Literal["views"] = "views"


@dataclass(slots=True)
class Train:
    savedir: str = "outdir/e00/scratch/"
    data: tuple[str, ...] = ()  # the lib.data.DATASETS whose train volumes pretraining samples
    data_power: float = 0.0  # their sampling: a dataset's share, and a volume's within it, go as voxel count ^ this (lib.data.sampling)
    eval_on: str = ""  # run(): the lib.data.DATASETS its end of training is evaluated on (its val images, and its fit and test blocks)
    model: Mae | Masked | Views = field(default_factory=Masked)
    patch_size: Tup3Int = (104, 104, 104)  # voxels are 8 nm isotropic (the dataloader reads every volume at 8 nm)
    batch_size: int = 84
    total_steps: int = 71  # warmup + benchmark + 1 profiler warmup + profile steps: stop right after profiling.
    # Training runs set it to the expected step count: milestone checkpoints are every 10% of it.
    max_hours: float = 0.0  # >0: stop after this many hours (total_steps is then an upper bound); sets LSF walltime
    milestone_every: int = 0  # steps between a trunk's milestone checkpoints (milestone_steps); 0: every 10% of total_steps
    milestone_doubling: bool = False  # milestones at milestone_every x 1, 2, 4, ...: log-spaced, a few evals however fast it steps
    n_layers: int = 12
    width: int = 512  # encoder width; heads = width // 64
    lr: float = 1e-4  # run(): linear warmup, then held at this lr (or lr_decay; see anneal_from)
    lr_decay: bool = False  # run(): after the warmup, lr falls linearly to 0 at total_steps (a run stopped by max_hours first ends mid-way)
    qk_norm: bool = False  # LayerNorm on the encoder's queries and keys (lib.encoders.EncoderConfig.qk_norm)
    global_size: Tup3Int = (88, 88, 88)  # the view each sample trains on, voxels (a multiple of the patch)

    # optimizations
    f32mode: F32Mode = "high"
    n_workers: int = 16
    prefetch_factor: int = 2
    amp: bool = True  # bf16 autocast for forward + loss
    compile: bool = True  # torch.compile(dynamic=True) the encoder
    cudagraphs: bool = False  # instead compile static with CUDA graphs (mode="reduce-overhead"); needs compile
    compile_blocks: bool = False  # compile each transformer block separately so DDP can overlap all-reduce with backward
    defer_image_ops: bool = True  # workers ship uint8 crops; cast + normalize on the GPU (miao.finish_images)
    eager_patch_embed: bool = True  # keep PatchEmbed3d out of torch.compile (see compile_model); False only to study the bug
    weight_decay: float = 0.0  # AdamW decay on weight matrices (biases/norms excluded); 0 = the original plain Adam
    adam_beta2: float = 0.999  # Adam second-moment decay; 0.999 = torch default (all runs so far), mia-muvit uses 0.95
    grad_compress: bool = False  # DDP bf16_compress_hook: all-reduce gradients in bf16 (half the bytes)
    queue: str = "gpu_h200"  # LSF queue
    n_gpus: int = 1  # DDP ranks on one node (launched via torchrun); batch_size and n_workers are per GPU

    anneal_from: str = ""  # a checkpoint .pt (e.g. a trunk's milestone): train() continues it for total_steps steps, Adam moments kept,
                           # lr cosine-decaying from the checkpoint's lr to 1% of it; run() then evals the result (pca, probe)

    # profiling params
    warmup_steps: int = 10
    benchmark_steps: int = 50
    profile_steps: int = 10  # Set to zero to disable trace collection.

    def but(self, **changes) -> Train:
        """This spec with these fields changed: how an experiment writes each run as a difference from its base."""
        return replace(self, **changes)

@dataclass(slots=True)
class Eval:
    """The spec of pca, probe and score: one set of weights, in the architecture of the run that made them, measured as they
    are (frozen; the probe fits only a per-token Linear). A run's own end-of-training eval (own_eval), a trunk milestone's
    (eval_params), or an older run's or untrained weights. Training anything bigger on labels is a Finetune."""
    savedir: str
    train: Train  # the run the weights are of: its architecture, its view size (the tiles pca and probe encode), its queue
    init_from: str = ""  # the weights: "" train's own latest checkpoint; a run dir, its latest; a checkpoint .pt; "random" untrained
    score: bool = False  # probe() also writes the affinity artifacts `score` reads (probe/{fit,test}, GBs each; score deletes them after)
    jobs: tuple[str, ...] = ("pca", "probe")  # the entrypoints start() submits for it
    data: str = ""  # the lib.data.DATASETS it is measured on: pca and pca_sizes on its first val volume, probe on its fit and test blocks
    standardize: bool = False  # probe(): fit on features standardized per channel (fit block's mean and std; folded back into the Linear)

@dataclass(slots=True)
class Finetune:
    """The spec of finetune (and of score, on its artifacts): a dense decoder (lib.decoders.unetr.Unetr) trained on a probe set's GT affinities on top of one
    set of weights, frozen or trained with it, and scored on the set's test blocks as it trains: AP against fit compute
    (steps) at one amount of GT (fit_side)."""
    savedir: str
    train: Train  # the weights' run, as Eval.train: its architecture, and its view size (the crops the decoder fits and predicts on)
    init_from: str = ""  # the weights, as Eval.init_from ("random": the from-scratch baseline)
    data: str = ""  # the lib.data.DATASETS whose fit block it fits and whose test blocks score it
    fit_side: int = 0  # >0: fit on a cube of this side (8 nm voxels) about the fit block's centre, not all of it (less GT)
    steps: int = 10_000  # fit steps (lib.decoders.unetr.fit_unetr, batch 4), scored at 100, 200, 400, ... and the last
    freeze_encoder: bool = True  # False: train the encoder with the decoder (end-to-end finetuning)
    score: bool = False  # finetune() also writes the affinity artifacts `score` reads (probe/{fit,test}, GBs each; score deletes them after)

Spec = Train | Eval | Finetune  # a job's spec: what an Experiment's runs are and start() submits

@dataclass(slots=True)
class Rank:
    """This process's place in a training job's DDP run (setup: torchrun's environment); a plain process is rank 0 of 1."""
    rank: int
    world: int
    local: int  # the GPU on this node

    @property
    def rank0(self) -> bool:
        return self.rank == 0

    def device(self):
        import torch
        return torch.device(f"cuda:{self.local}" if torch.cuda.is_available() else "cpu")

def encoder_config(par: Train) -> EncoderConfig:
    return EncoderConfig(depth=par.n_layers, embed_dim=par.width, num_heads=par.width // 64, qk_norm=par.qk_norm)

def model_config(par: Train) -> models.LejepaConfig | models.MaeConfig:
    """par's model's config: its objective on the encoder par says, and the view par.global_size (plus the shift of masked tokens)."""
    m, (z, y, x), patch = par.model, par.global_size, encoder_config(par).patch_size[0]  # the encoder's token side, voxels
    if isinstance(m, Mae):
        dim, depth = m.decoder
        return models.MaeConfig(encoder=encoder_config(par), view=ViewConfig(views="displace", global_size=par.global_size),
                         decoder_dim=dim, decoder_depth=depth, decoder_heads=dim // 32)
    if isinstance(m, Views):
        return models.LejepaConfig(encoder=encoder_config(par), objective="views", batch_views=m.batch_views, lamb=m.lamb,
                            view=ViewConfig(views="displace", n_global=2, n_local=4, global_size=par.global_size, local_size=(m.local,) * 3))
    assert m.cube % patch == 0, f"{par.savedir}: the masked cube {m.cube} must be a multiple of the patch {patch}"
    s = m.shift  # both views' region: the context's, and the targets' offset by up to the shift
    return models.LejepaConfig(encoder=encoder_config(par), objective="mask", lamb=m.lamb, sigreg_over=m.sigreg_over, stop_grad=m.stop_grad,
                        view=ViewConfig(views="displace", n_global=1, n_local=0, global_size=(z + s, y + s, x + s)),
                        mask=models.MaskConfig(cube=m.cube // patch, shift=s, aligned=m.aligned, expand=m.expand))

def make_model(par: Train) -> models.Lejepa | models.Mae:
    """par's model, untrained."""
    c = model_config(par)
    return models.Mae(c) if isinstance(c, models.MaeConfig) else models.Lejepa(c)

def load_checkpoint(e: Eval | Finetune):
    """(model on the GPU if any, step) for inference or finetuning (pca, probe, finetune): the weights e.init_from names, in
    e.train's architecture.

    "" (default): e.train's own latest checkpoint, whose saved params must match e.train. Inference depends on lib/, so
    like training it only runs for this commit's own experiment: for an older sweep, go back to its commit.
    A run dir (e.g. outdir/e00/viewsizes/d0): that run's latest checkpoint, loaded strictly into e.train's architecture,
    so an incompatible one fails loudly (Conv3d-era patch embeds load, see PatchEmbed3d). The run dir is an
    artifact in outdir/'s append-only log, so this is how a probe reuses a known-good older model.
    "random": untrained weights, seeded (the baseline every trained encoder must beat). Step 0.
    """
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if e.init_from == "random":
        torch.manual_seed(0)
        return make_model(e.train).to(device).eval(), 0
    src = Path(e.init_from or e.train.savedir)
    paths = [src] if src.suffix == ".pt" else sorted((src / "checkpoints").glob("step_*.pt")) or sorted(src.glob("checkpoint.pt"))
    assert paths, f"no checkpoints in {src}"
    ckpt = torch.load(paths[-1], map_location="cpu", weights_only=False)
    if not e.init_from:
        assert json_equal(ckpt["params"], asdict(e.train)), f"{paths[-1]} was made by other params: run from its experiment's commit"
    model = make_model(e.train)
    model.load_state_dict(ckpt["model"])  # strict: architecture drift fails loudly
    return model.to(device).eval(), ckpt["step"]

def milestone_steps(trunk: Train) -> list[int]:
    """The steps whose checkpoints train() keeps in checkpoints/milestones/: every milestone_every steps up to total_steps,
    or, if that is 0, every 10% of it (the last is the final checkpoint of a trunk that ran to completion). A trunk stopped
    early by max_hours never reaches the later ones: with a time budget, total_steps is just the cap that lists them.
    With milestone_doubling, they double from milestone_every instead."""
    if trunk.milestone_doubling:
        assert trunk.milestone_every > 0, f"{trunk.savedir}: doubling milestones start at milestone_every, which is 0"
        return [trunk.milestone_every * 2 ** k for k in range((trunk.total_steps // trunk.milestone_every).bit_length())]
    if trunk.milestone_every:
        return list(range(trunk.milestone_every, trunk.total_steps + 1, trunk.milestone_every))
    return sorted({round(k * trunk.total_steps / 10) for k in range(1, 11)})

def milestone(trunk: Train, step: int) -> Path:
    """The checkpoint train() keeps at this milestone step (0.6 GB for m: a milestone eval, or a branch, can reload it any time)."""
    assert step in milestone_steps(trunk), f"step {step} is not a milestone of {trunk.savedir}: {milestone_steps(trunk)}"
    return Path(trunk.savedir) / "checkpoints/milestones" / f"step_{step:07d}.pt"

def branch_dir(trunk: Train, step: int) -> str:
    """A milestone branch's savedir: inside its trunk's, <trunk>/step_<step>/."""
    return f"{Path(trunk.savedir) / f'step_{step:07d}'}/"

def eval_params(trunk: Train, step: int) -> Eval:
    """The eval of `trunk`'s milestone at `step`, as saved (not annealed), in branch_dir."""
    return Eval(savedir=branch_dir(trunk, step), train=trunk, init_from=str(milestone(trunk, step)), data=trunk.eval_on)

def own_eval(t: Train) -> Eval:
    """The eval of a training run's own latest checkpoint, in its own savedir: what run() does after training."""
    return Eval(savedir=t.savedir, train=t, data=t.eval_on)

def source(p: Spec) -> str:
    """The weights p starts from: an anneal's checkpoint (Train.anneal_from), or the ones an eval or finetune reads (init_from)."""
    return p.anneal_from if isinstance(p, Train) else p.init_from

def record(par: Spec, fn: str):
    """Append one row to par.savedir/runs.json: which experiment function ran (fn), its spec (params), code_provenance(),
    argv and LSF job. Written first thing, so even a run that crashes before any results describes itself.
    Repro for the dir's artifacts: that commit + argv. The command that submitted the job is in
    outdir/_log/commands.jsonl (log_command)."""
    row = {"fn": fn, "params": asdict(par), **code_provenance(), "argv": sys.argv, "lsf_job": os.environ.get("LSB_JOBID")}
    Path(par.savedir).mkdir(parents=True, exist_ok=True)
    with open(Path(par.savedir) / "runs.json", "a") as f:
        f.write(json.dumps(row) + "\n")

