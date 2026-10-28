"""The specs: Train (how a model is trained, in groups) and the three tasks' (Trunk, Branch, Score: what an experiment's runs are
and lib.jobs submits), what follows from one (its model), and the weights it loads."""

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
    """A masked autoencoder, the pixel-space baseline (lib.models.Mae), on the view Train.data.view."""
    decoder: tuple[int, int] = (192, 2)  # the decoder's (width, depth), heads width // 32
    kind: Literal["mae"] = "mae"


@dataclass(slots=True)
class Masked:
    """Masked-token prediction (lib.models.Lejepa): a cube of the view's tokens is hidden from the encoder and a predictor makes
    their embeddings, loss = prediction error + lamb x SIGReg. The view is Train.data.view, plus shift (both views' region)."""
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
    """LeJEPA's multi-view invariance (lib.models.Lejepa): 2 global views of Train.data.view and 4 local ones, each pulled to the
    mean of the globals, loss = invariance + lamb x SIGReg."""
    local: int = 56  # the local views' side, voxels
    batch_views: bool = False  # one encoder call per group of same-shape views
    lamb: float = 0.002
    kind: Literal["views"] = "views"


# How a model is trained, in groups: Train is what every task of a trunk shares (a trunk trains it, a branch anneals it, a score reads it).
@dataclass(slots=True)
class Encoder:
    n_layers: int = 12
    width: int = 512  # heads = width // 64
    qk_norm: bool = False  # LayerNorm on the encoder's queries and keys (lib.encoders.EncoderConfig.qk_norm)

@dataclass(slots=True)
class Data:
    sets: tuple[str, ...] = ()  # the lib.data.DATASETS whose train volumes pretraining samples
    power: float = 0.0  # their sampling: a dataset's share, and a volume's within it, go as voxel count ^ this (lib.data.sampling)
    window: Tup3Int = (104, 104, 104)  # what the loader reads per sample, voxels (8 nm isotropic: it reads every volume at 8 nm)
    view: Tup3Int = (88, 88, 88)  # the view each sample trains on, voxels (a multiple of the encoder's patch)

@dataclass(slots=True)
class Optim:
    batch_size: int = 84  # per GPU
    lr: float = 1e-4  # a trunk: linear warmup, then held; a branch: down to lr x Trunk.anneal_to
    weight_decay: float = 0.0  # AdamW decay on weight matrices (biases/norms excluded); 0 = the original plain Adam
    adam_beta2: float = 0.999  # Adam second-moment decay; 0.999 = torch default (all runs so far), mia-muvit uses 0.95
    grad_compress: bool = False  # DDP bf16_compress_hook: all-reduce gradients in bf16 (half the bytes)

@dataclass(slots=True)
class Hardware:
    queue: str = "gpu_h200"  # LSF queue
    n_gpus: int = 1  # DDP ranks on one node (launched via torchrun); Optim.batch_size and n_workers are per GPU
    n_workers: int = 16  # data loader workers per GPU (lib.jobs.CPUS_PER_GPU - 1 at most)
    prefetch_factor: int = 2

@dataclass(slots=True)
class Speed:
    f32mode: F32Mode = "high"
    amp: bool = True  # bf16 autocast for forward + loss
    compile: bool = True  # torch.compile(dynamic=True) the encoder
    cudagraphs: bool = False  # instead compile static with CUDA graphs (mode="reduce-overhead"); needs compile
    compile_blocks: bool = False  # compile each transformer block separately so DDP can overlap all-reduce with backward
    defer_image_ops: bool = True  # workers ship uint8 crops; cast + normalize on the GPU (miao.finish_images)
    eager_patch_embed: bool = True  # keep PatchEmbed3d out of torch.compile (see compile_model); False only to study the bug

@dataclass(slots=True)
class Profile:
    warmup_steps: int = 10
    benchmark_steps: int = 50
    profile_steps: int = 10  # Set to zero to disable trace collection.

@dataclass(slots=True)
class Train:
    """How a model is trained: its objective, encoder, data, optimizer, hardware, speed settings and profiling."""
    model: Mae | Masked | Views = field(default_factory=Masked)
    encoder: Encoder = field(default_factory=Encoder)
    data: Data = field(default_factory=Data)
    optim: Optim = field(default_factory=Optim)
    hardware: Hardware = field(default_factory=Hardware)
    speed: Speed = field(default_factory=Speed)
    profile: Profile = field(default_factory=Profile)

    def but(self, **changes) -> Train:
        """This spec with these groups replaced: how an experiment writes each run as a difference from its base."""
        return replace(self, **changes)


# The three tasks' specs, each holding its parent's (Score -> Branch -> Trunk -> Train), each in its own savedir.
@dataclass(slots=True)
class Trunk:
    """Task trunk: pretrain train for segments x segment_steps steps, with a branch at the start and after each segment. You write
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
    board: Literal["neuron-instance"] = "neuron-instance"  # the mia-evals leaderboard (lib.jobs.BOARDS) a branch is scored on

@dataclass(slots=True)
class Branch:
    """Task branch: anneal a fork point of trunk, then pca and tiled inference on the result. Made by the trunk's fork."""
    savedir: str
    trunk: Trunk
    init_from: str  # the fork point: a checkpoint .pt of weights and optimizer

@dataclass(slots=True)
class Score:
    """Task score: mia-evals on the weights and affinities branch wrote (savedir/checkpoint.pt, savedir/probe/). Made by the branch."""
    savedir: str  # the branch's
    branch: Branch

Spec = Trunk | Branch | Score  # a task's spec: what an Experiment's runs are (Trunk) and go() submits

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
    e = par.encoder
    return EncoderConfig(depth=e.n_layers, embed_dim=e.width, num_heads=e.width // 64, qk_norm=e.qk_norm)

def model_config(par: Train) -> models.LejepaConfig | models.MaeConfig:
    """par's model's config: its objective on the encoder par says, and the view par.global_size (plus the shift of masked tokens)."""
    m, (z, y, x), patch = par.model, par.data.view, encoder_config(par).patch_size[0]  # the encoder's token side, voxels
    if isinstance(m, Mae):
        dim, depth = m.decoder
        return models.MaeConfig(encoder=encoder_config(par), view=ViewConfig(views="displace", global_size=par.data.view),
                         decoder_dim=dim, decoder_depth=depth, decoder_heads=dim // 32)
    if isinstance(m, Views):
        return models.LejepaConfig(encoder=encoder_config(par), objective="views", batch_views=m.batch_views, lamb=m.lamb,
                            view=ViewConfig(views="displace", n_global=2, n_local=4, global_size=par.data.view, local_size=(m.local,) * 3))
    assert m.cube % patch == 0, f"the masked cube {m.cube} must be a multiple of the patch {patch}"
    s = m.shift  # both views' region: the context's, and the targets' offset by up to the shift
    return models.LejepaConfig(encoder=encoder_config(par), objective="mask", lamb=m.lamb, sigreg_over=m.sigreg_over, stop_grad=m.stop_grad,
                        view=ViewConfig(views="displace", n_global=1, n_local=0, global_size=(z + s, y + s, x + s)),
                        mask=models.MaskConfig(cube=m.cube // patch, shift=s, aligned=m.aligned, expand=m.expand))

def make_model(par: Train) -> models.Lejepa | models.Mae:
    """par's model, untrained."""
    c = model_config(par)
    return models.Mae(c) if isinstance(c, models.MaeConfig) else models.Lejepa(c)

def load_checkpoint(train: Train, init_from: str):
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

def record(par: Spec, fn: str):
    """Append one row to par.savedir/runs.json: which experiment function ran (fn), its spec (params), code_provenance(),
    argv and LSF job. Written first thing, so even a run that crashes before any results describes itself.
    Repro for the dir's artifacts: that commit + argv. The command that submitted the job is in
    outdir/_log/commands.jsonl (log_command)."""
    row = {"fn": fn, "params": asdict(par), **code_provenance(), "argv": sys.argv, "lsf_job": os.environ.get("LSB_JOBID")}
    Path(par.savedir).mkdir(parents=True, exist_ok=True)
    with open(Path(par.savedir) / "runs.json", "a") as f:
        f.write(json.dumps(row) + "\n")

