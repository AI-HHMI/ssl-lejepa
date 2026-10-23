"""Experiment code (remote): 3D ViT pretraining runs (LeJEPA, masked tokens, MAE), benchmarks and replays; evals of their weights
(pca maps, linear probe, mia-evals score) and UNETR finetunes; and their LSF sweep (roots, each trunk evaluating its own milestones;
savedirs under outdir/e00/). Depends on lib/, so it runs only from its experiment's commit.
Each job is an entrypoint taking its spec (Train, Eval or Finetune) or the path of the spec's json, e.g. `run outdir/.../train.json`:
called in-process with the spec, or as an LSF job with the file that submit() writes, so any job can submit another.
`python experiment.py` or `-h` lists the entrypoints (ENTRYPOINTS, at the bottom).
Analysis of the results lives in analysis_utils.py (readers), analysis_plots.py (report cards) and reports/ (one page per experiment)."""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from datetime import timedelta
import inspect
import os
import sys
import json
import re
import shutil
import time
from pathlib import Path

import math
from itertools import product

# local

from lib.data import HEMIBRAIN_EB_LABELS, HEMIBRAIN_EB_PROBE_ANNOTATED, HEMIBRAIN_EB_PROBE_BOXES, PROBE_SETS, ProbeVolume, train_volumes
from lib.benchmark import Benchmark
from lib.inference import boundary_png, mean_scores, pca_rgb, score_block, tile_features, tile_logits, write_affinities
from lib.readers import PROBE_SIDE, eb_val_crop, inner, read_block
from lib.losses import LejepaOutput
from lib.encoders import EncoderConfig
from lib.models import Lejepa, LejepaConfig, Mae, MaeConfig, MaskConfig
from lib.views import ViewConfig, ViewMaker
from lib.util import *
from lib.types import *

# external 

import lmd_catalog as lmd
from miao.config import MiaoConfig
from miao import VolumeDataset, collate_deferred, finish_images

from pprint import pprint
import numpy as np

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
# Training (train): the lr schedule (a trunk's linear warmup steps; an anneal's cosine ends at LR_FLOOR x the checkpoint's lr),
# the max global grad norm, checkpoints (every CHECKPOINT_EVERY steps, ~15 min at 8 GPUs; the newest CHECKPOINT_KEEP kept), and
# steps whose gradient is non-finite (skipped, not applied: the run stops after MAX_SKIPPED; the first BAD_BATCHES_KEPT dumped).
LR_WARMUP, LR_FLOOR, GRAD_CLIP = 1000, 0.01, 1.0
CHECKPOINT_EVERY, CHECKPOINT_KEEP = 2000, 1
MAX_SKIPPED, BAD_BATCHES_KEPT = 100, 2
TEST_TILES = (16, 32, 64, 128, 256, 512)  # pca_sizes: the tile sides an encoder is run at, test time (2^3 to 64^3 tokens)
@dataclass(slots=True)
class Train:
    savedir: str = "outdir/e00/main/basic/"
    data: TrainData = "hemibrain_eb"  # training volumes, see lib.data.TRAIN_BOXES
    probe_sets: tuple[str, ...] = ("hemibrain_eb",)  # run(): the lib.data.PROBE_SETS its end of training is probed on (probe_evals)
    patch_size: Tup3Int = (104, 104, 104)  # hemibrain EB is 8 nm isotropic
    batch_size: int = 84
    steps_per_epoch: int = 71  # warmup + benchmark + 1 profiler warmup + profile steps: stop right after profiling.
    # Training runs set it to the expected step count: milestone checkpoints are every 10% of it.
    max_hours: float = 0.0  # >0: stop after this many hours (steps_per_epoch is then an upper bound); sets LSF walltime
    milestone_every: int = 0  # steps between a trunk's milestone checkpoints (milestone_steps); 0: every 10% of steps_per_epoch
    milestone_doubling: bool = False  # milestones at milestone_every x 1, 2, 4, ...: log-spaced, a few evals however fast it steps
    n_layers: int = 12
    width: int = 512  # encoder width; heads = width // 64
    lr: float = 1e-4  # run(): linear warmup, then held at this lr (no decay; see anneal_from)
    # The pre-training objective: LeJEPA's multi-view invariance ("views", every run before e02), its masked-token prediction
    # ("mask", e02-e05), or a masked autoencoder ("mae", the pixel-space baseline: lib.models.Mae). See model_config.
    objective: Literal["views", "mask", "mae"] = "mask"
    views: Views = "basic"  # the crops: 'basic' (random scales) or 'displace' (global_size; "views": locals of local_size inside them)
    global_size: Tup3Int = (88, 88, 88)  # displace only
    local_size: Tup3Int = (56, 56, 56)  # displace, "views" only (analysis_utils reads it)
    batch_views: bool = False  # "views": one encoder call per group of same-shape views (2 per step with displace)
    mae_decoder: tuple[int, int] = (192, 2)  # "mae": the decoder's (width, depth), heads width // 32 (lib.models.MaeConfig)
    mask_block: float = 0.6  # the masked cube's side as a fraction of the view's token grid side (lib.models.MaskConfig.block)
    mask_aligned: bool = False  # mask_shift only: whole-token view origins (lib.models.MaskConfig.aligned)
    mask_expand: bool = False  # mask_shift only: hide a token more of the context around the masked target tokens (MaskConfig.expand)
    mask_shift: int = 0  # >0: targets from a second view up to this many voxels off the context's (lib.models.MaskConfig.shift); both global_size
    # loss = prediction error + lamb x SIGReg. SIGReg's pull is lamb x its sample count (its statistic is N x an empirical-CF
    # distance) and it sees each view's N tokens: the old lamb 0.1 over 32 batch samples (a pull of 3.2) is 0.002 at 12^3 tokens.
    lamb: float = 0.002
    sigreg_over: Literal["view", "batch", "both"] = "view"  # SIGReg's samples: each view's tokens, each position's batch, or both (lib.models.Lejepa.sigreg_term)
    stop_grad: bool = False  # detach the masked-token targets (lib.models.LejepaConfig.stop_grad)

    # optimizations
    f32mode: F32Mode = "high"
    n_workers: int = 16
    prefetch_factor: int = 2
    amp: bool = True  # bf16 autocast for forward + loss
    compile: bool = True  # torch.compile(dynamic=True) the encoder
    cudagraphs: bool = False  # instead compile static with CUDA graphs (mode="reduce-overhead"); needs compile + displace views
    compile_blocks: bool = False  # compile each transformer block separately so DDP can overlap all-reduce with backward
    defer_image_ops: bool = True  # workers ship uint8 crops; cast + normalize on the GPU (miao.finish_images)
    eager_patch_embed: bool = True  # keep PatchEmbed3d out of torch.compile (see compile_model); False only to study the bug
    weight_decay: float = 0.0  # AdamW decay on weight matrices (biases/norms excluded); 0 = the original plain Adam
    adam_beta2: float = 0.999  # Adam second-moment decay; 0.999 = torch default (all runs so far), mia-muvit uses 0.95
    grad_compress: bool = False  # DDP bf16_compress_hook: all-reduce gradients in bf16 (half the bytes)
    queue: str = "gpu_h200"  # LSF queue
    n_gpus: int = 1  # DDP ranks on one node (launched via torchrun); batch_size and n_workers are per GPU

    anneal_from: str = ""  # a checkpoint .pt (e.g. a trunk's milestone): train() continues it for steps_per_epoch steps, Adam moments kept,
                           # lr cosine-decaying from the checkpoint's lr to 1% of it; run() then evals the result (pca, probe)

    # profiling params
    warmup_steps: int = 10
    benchmark_steps: int = 50
    profile_steps: int = 10  # Set to zero to disable trace collection.

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
    probe_on: str = "hemibrain_eb"  # probe(): the lib.data.PROBE_SETS it fits on and is scored on

@dataclass(slots=True)
class Finetune:
    """The spec of finetune: a dense decoder (lib.decoders.unetr.Unetr) trained on a probe set's GT affinities on top of one
    set of weights, frozen or trained with it, and scored on the set's test blocks as it trains: AP against fit compute
    (steps) at one amount of GT (fit_side)."""
    savedir: str
    train: Train  # the weights' run, as Eval.train: its architecture, and its view size (the crops the decoder fits and predicts on)
    init_from: str = ""  # the weights, as Eval.init_from ("random": the from-scratch baseline)
    probe_on: str = "hemibrain_eb"  # the lib.data.PROBE_SETS whose train block it fits and whose test blocks score it
    fit_side: int = 0  # >0: fit on a cube of this side (8 nm voxels) about the train block's centre, not all of it (less GT)
    steps: int = 10_000  # fit steps (lib.decoders.unetr.fit_unetr, batch 4), scored at 100, 200, 400, ... and the last
    freeze_encoder: bool = True  # False: train the encoder with the decoder (end-to-end finetuning)

Spec = Train | Eval | Finetune  # a job's spec: what roots() lists and start() submits

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

# This experiment's id (README: experiment ids): its roots are outdir/<EXPERIMENT>/d<i>/, a trunk's branches in its own dir.
EXPERIMENT = "e00/e10-finetune-gap"

def roots() -> list[Spec]:
    """This experiment: the runs it starts from nothing (trunks, other training runs, evals of another experiment's weights).
    Each trunk evaluates its own milestones as it saves them (eval_milestone)."""
    # Is pretraining worth it once labels are used to train? unetr-probe's UNETR decoders on pretrained encoders led random ones
    # early in the fit, then random caught up by ~1-2k of 3k steps. Map the gap: test AP as a function of finetuning compute
    # (fit steps, scored at doubling steps to 10k: Finetune) and of GT (the fit block cut to a cube of side 128, 256, 512 or
    # whole: Finetune.fit_side), on all four lmd_ssl_v1 families (sparse ones scoring labelled-vs-unlabelled pairs as
    # boundaries), the encoder frozen or trained with the decoder. The encoders are trained here, on this commit, so every
    # load means what it did in training: the best of each objective so far on hemibrain_wide, at one budget (6 h of m,
    # 12x512, batch 32, one H200, ~0.7x a B300's speed: about e08's 4 h of B300 steps; linear-probe evals at doubling steps
    # from 5k): e08's MAE (decoder 512x8, view 96 of 128; 0.296 / 0.869 short / long AP at 4 h of B300), e06's masked tokens
    # at view 64 (shift 24, block 0.6, pull 10, SIGReg "both"; 0.205 / 0.847 at 8 h) and e06's multi-view invariance (input
    # 128, globals and locals 96; 0.113 / 0.544), and the masked recipe on lmd4 (all four families, as e09). Each finetunes
    # at its own view, beside a random encoder at views 96 and 64. B300s were full: the trunks on H200s, the finetunes (192
    # independent jobs) on the larger pool of RTX PRO 6000s (95 GB), their queue set on the Train each carries (submit uses
    # its queue; it doesn't change the weights). A trunk's finetunes wait in LSF for its training job (submit).
    BATCH, STEPS, HOURS = 32, 2_560_000, 6
    SETS, SIDES = ("hemibrain_eb", "fish2", "liconn_mouse", "kasthuri15"), (128, 256, 512, 0)

    def trunk(objective: Literal["views", "mask", "mae"], data: TrainData, input_size: int, view: int) -> Train:
        p = Train()
        p.objective = objective
        p.data = data
        p.views = "displace"
        p.patch_size = (input_size, input_size, input_size)
        p.global_size = (view, view, view)
        p.batch_size = BATCH
        p.steps_per_epoch = STEPS
        p.milestone_every = 5_000
        p.milestone_doubling = True
        p.max_hours = HOURS
        p.weight_decay = 0.05
        p.adam_beta2 = 0.95
        p.cudagraphs = True
        p.n_workers = 11  # 12 cores per GPU
        p.queue = "gpu_h200"
        return p

    def masked(data: TrainData) -> Train:
        p = trunk("mask", data, 88, 64)
        p.mask_block = 0.6
        p.mask_shift = 24
        p.sigreg_over = "both"
        p.lamb = 10 / BATCH
        return p

    mae = trunk("mae", "hemibrain_wide", 128, 96)
    mae.mae_decoder = (512, 8)
    views = trunk("views", "hemibrain_wide", 128, 96)
    views.local_size = (96, 96, 96)
    views.batch_views = True
    views.lamb = 0.1
    trunks = [mae, masked("hemibrain_wide"), views, masked("lmd4")]
    runs: list[Spec] = []
    for p in trunks:
        p.savedir = f"outdir/{EXPERIMENT}/d{len(runs)}/"
        runs.append(p)
    encoders = [(t, t.savedir) for t in trunks] + [(mae, "random"), (trunks[1], "random")]  # random at view 96 and 64
    for (t, init), s, freeze, side in product(encoders, SETS, (True, False), SIDES):
        runs.append(Finetune(savedir=f"outdir/{EXPERIMENT}/d{len(runs)}/", train=replace(t, queue="gpu_rtx6000"), init_from=init,
                             probe_on=s, fit_side=side, freeze_encoder=freeze))
    return runs

def milestone_steps(trunk: Train) -> list[int]:
    """The steps whose checkpoints train() keeps in checkpoints/milestones/: every milestone_every steps up to steps_per_epoch,
    or, if that is 0, every 10% of it (the last is the final checkpoint of a trunk that ran to completion). A trunk stopped
    early by max_hours never reaches the later ones: with a time budget, steps_per_epoch is just the cap that lists them.
    With milestone_doubling, they double from milestone_every instead."""
    if trunk.milestone_doubling:
        assert trunk.milestone_every > 0, f"{trunk.savedir}: doubling milestones start at milestone_every, which is 0"
        return [trunk.milestone_every * 2 ** k for k in range((trunk.steps_per_epoch // trunk.milestone_every).bit_length())]
    if trunk.milestone_every:
        return list(range(trunk.milestone_every, trunk.steps_per_epoch + 1, trunk.milestone_every))
    return sorted({round(k * trunk.steps_per_epoch / 10) for k in range(1, 11)})

def milestone(trunk: Train, step: int) -> Path:
    """The checkpoint train() keeps (until its eval is done with it, release_milestone) at this milestone step."""
    assert step in milestone_steps(trunk), f"step {step} is not a milestone of {trunk.savedir}: {milestone_steps(trunk)}"
    return Path(trunk.savedir) / "checkpoints/milestones" / f"step_{step:07d}.pt"

def branch_dir(trunk: Train, step: int) -> str:
    """A milestone branch's savedir: inside its trunk's, <trunk>/step_<step>/."""
    return f"{Path(trunk.savedir) / f'step_{step:07d}'}/"

def eval_params(trunk: Train, step: int) -> Eval:
    """The eval of `trunk`'s milestone at `step`: pca maps and the linear probe (pca, probe) of that checkpoint as saved, not
    annealed, in branch_dir."""
    return Eval(savedir=branch_dir(trunk, step), train=trunk, init_from=str(milestone(trunk, step)))

def own_eval(t: Train) -> Eval:
    """The eval of a training run's own latest checkpoint, in its own savedir: what run() does after training."""
    return Eval(savedir=t.savedir, train=t)

def probe_evals(t: Train) -> list[Eval]:
    """The probes run() makes of a training run's own latest checkpoint: own_eval, or, for a run probed on several sets
    (Train.probe_sets), one per set in its own savedir, <savedir>probe_<set>/."""
    if t.probe_sets == ("hemibrain_eb",):
        return [own_eval(t)]
    return [Eval(savedir=f"{t.savedir}probe_{s}/", train=t, probe_on=s) for s in t.probe_sets]

def source(p: Spec) -> str:
    """The weights p starts from: an anneal's checkpoint (Train.anneal_from), or the ones an eval or finetune reads (init_from)."""
    return p.anneal_from if isinstance(p, Train) else p.init_from

def milestone_of(p: Spec) -> str:
    """The milestone checkpoint of this experiment's own trunk that a branch starts from, the one it anneals or evaluates;
    "" for a trunk, and for an eval of another experiment's weights (init_from outside outdir/<EXPERIMENT>/)."""
    return source(p) if source(p).startswith(f"outdir/{EXPERIMENT}/") else ""

def record(par: Spec, fn: str):
    """Append one row to par.savedir/runs.json: which experiment function ran (fn), its spec (params), code_provenance(),
    argv and LSF job. Written first thing, so even a run that crashes before any results describes itself.
    Repro for the dir's artifacts: that commit + argv. The command that submitted the job is in
    outdir/_log/commands.jsonl (log_command)."""
    row = {"fn": fn, "params": asdict(par), **code_provenance(), "argv": sys.argv, "lsf_job": os.environ.get("LSB_JOBID")}
    with open(Path(par.savedir) / "runs.json", "a") as f:
        f.write(json.dumps(row) + "\n")

def collate_images(samples):
    import torch
    return torch.stack([s["img"] for s in samples])

def lejepa_config(par: Train) -> LejepaConfig:
    """par's Lejepa: "views" (all runs before e02, which kept lamb 0.1 in the model; set it in par) or "mask"."""
    assert par.objective != "mae", f"{par.savedir} trains an MAE: model_config"
    (z, y, x), s = par.global_size, par.mask_shift
    n_global, n_local = (2, 4) if par.objective == "views" else (1, 0)  # "views": ViewConfig's defaults before e02
    return LejepaConfig(
        encoder = EncoderConfig(depth=par.n_layers, embed_dim=par.width, num_heads=par.width // 64),
        objective = par.objective,
        view = ViewConfig(views=par.views, n_global=n_global, n_local=n_local, global_size=(z + s, y + s, x + s), local_size=par.local_size),  # "mask": both views' region
        batch_views = par.batch_views,
        mask = MaskConfig(block=par.mask_block, shift=par.mask_shift, aligned=par.mask_aligned, expand=par.mask_expand),
        lamb = par.lamb,
        sigreg_over = par.sigreg_over,
        stop_grad = par.stop_grad,
    )

def model_config(par: Train) -> LejepaConfig | MaeConfig:
    """par's model's config: an MAE on the same encoder and view, or lejepa_config."""
    if par.objective != "mae":
        return lejepa_config(par)
    (dim, depth) = par.mae_decoder
    return MaeConfig(encoder=EncoderConfig(depth=par.n_layers, embed_dim=par.width, num_heads=par.width // 64),
                     view=ViewConfig(views=par.views, global_size=par.global_size),
                     decoder_dim=dim, decoder_depth=depth, decoder_heads=dim // 32)

def make_model(par: Train) -> Lejepa | Mae:
    """par's model, untrained."""
    c = model_config(par)
    return Mae(c) if isinstance(c, MaeConfig) else Lejepa(c)

def compile_model(model: Lejepa | Mae, par: Train):
    """torch.compile the encoder in place, as par says (no-op unless par.compile)."""
    import torch
    assert par.compile or not (par.cudagraphs or par.compile_blocks), "cudagraphs / compile_blocks require compile"
    if not par.compile:
        return
    torch._logging.set_logs(recompiles=True)  # pyright: ignore[reportPrivateImportUsage]  # recompiles show up in job_*.log
    if par.cudagraphs:
        # CUDA graphs replay whole kernel sequences, removing per-kernel CPU launch cost.
        # They need static shapes: displace has exactly 2 encoder shapes (the whole view, and its visible tokens), basic has ~9.
        assert par.views == "displace", "cudagraphs needs views='displace' (few static shapes)"
        kwargs = dict(mode="reduce-overhead", dynamic=False)
    else:
        kwargs = dict(dynamic=True)  # Views change shape every step.
    # A whole-encoder compile releases every gradient at the end of one fused backward, so DDP can't
    # start all-reducing until backward is done. Per-block compile releases each block's grads as it finishes.
    for m in (model.encoder.blocks if par.compile_blocks else [model.encoder]):
        m.compile(**kwargs)
    if par.eager_patch_embed:
        # Inductor's fused patchify kernel (triton_poi_fused__to_copy__unsafe_view_clone_permute_view_0: the input's
        # reshape/permute + bf16 cast) hits an illegal memory access under static shapes at batch 64 on B300 and
        # H200 (e00/b300-compile, replay of b300-train8h/d0 with CUDA_LAUNCH_BLOCKING). Run it eagerly instead.
        pe = model.encoder.patch_embed
        setattr(pe, "forward", torch.compiler.disable(pe.forward))  # instance override; nn.Module.__call__ uses it

def save_view_pngs(par: Train, dataset, n_samples: int = 3):
    """Save what the model sees: for n_samples fresh samples, the input and each of its views, as
    center-z slices at native voxel size, left to right (input | globals | locals) -> savedir/views_{i}.png."""
    from PIL import Image
    cfg = model_config(par)
    p = cfg.encoder.patch_size
    maker = ViewMaker(cfg.view, patch_size=(p, p, p) if isinstance(p, int) else p)
    samples = [dataset[i] for i in range(n_samples)]
    x = finish_images(collate_deferred(samples))["img"] if par.defer_image_ops else collate_images(samples)  # Batch C Z Y X
    globals_, locals_ = maker(x)
    to_u8 = lambda v: (v[0, v.shape[1] // 2].clamp(0, 1) * 255).byte().numpy()  # C Z Y X -> Y X center slice
    savedir = Path(par.savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    for b in range(n_samples):
        panels = [to_u8(v[b]) for v in [x, *globals_, *locals_]]
        # Bottom-pad smaller views to the input height and separate panels with a white gap.
        panels = [np.pad(p, ((0, x.shape[-2] - p.shape[0]), (0, 8)), constant_values=255) for p in panels]
        Image.fromarray(np.concatenate(panels, axis=1)).save(savedir / f"views_{b}.png")
    print(f"Saved {n_samples} input + view slices to {savedir}/views_*.png", flush=True)

def dataloader(par: Train):
    import torch
    # Training volumes (lib/data.py), images only. Boxes are x y z; output is z y x.
    # miao samples each volume by its weight alone (size_weighting_exponent=0): equally within a family (lib.data.train_volumes),
    # and a joint set's families equally. Every volume is read at 8 nm (resolutions below), resampled where it is stored otherwise.
    volumes = [lmd.get(name).to_miao(bounding_box=box[::-1], weight=w) for name, box, w in train_volumes(par.data)]
    mcfg = MiaoConfig(
        volumes=volumes,
        patch_size=list(par.patch_size),
        resolutions=[[8.0, 8.0, 8.0]],
        # Extra batches keep workers busy (as in real training) through the last, profiled steps.
        samples_per_epoch=par.batch_size * (par.steps_per_epoch + par.n_workers * par.prefetch_factor),
        sampling="random",
        output_axes="lzyx",
        defer_image_ops=par.defer_image_ops,
    )
    dl = VolumeDataset(mcfg)
    loader = torch.utils.data.DataLoader(
      dl,
      batch_size=par.batch_size,
      num_workers=par.n_workers,
      collate_fn=collate_deferred if par.defer_image_ops else collate_images,
      prefetch_factor=par.prefetch_factor if par.n_workers else None,  # n_workers 0 (tests): loads in this process, no workers
      pin_memory=torch.cuda.is_available(),
      multiprocessing_context="spawn" if par.n_workers else None,
    )
    batches = iter(loader)
    if int(os.environ.get("RANK", 0)) == 0:
        save_view_pngs(par, dl)
    return batches

def setup(par: Train) -> Rank:
    """Torch settings, this rank's CUDA device and the DDP process group for training par; this process's Rank. Every rank."""
    import torch
    import torch.distributed as dist
    torch.set_float32_matmul_precision(par.f32mode)
    # cuDNN's fused attention backward returns NaN grads on some trained weights where flash and mem-efficient
    # don't (replay_bad_batch on e00/nanhunt_beta95/d9): the "divergences" of e00/viewsizes* and nanhunt*.
    # Runs before 2026-09-28 used it (torch's default pick on H200/B300).
    torch.backends.cuda.enable_cudnn_sdp(False)
    # Set by torchrun. Unset (plain `python`, tests) means a single process.
    # Don't seed torch identically across ranks: DataLoader workers derive miao's numpy seed from it.
    r = Rank(rank=int(os.environ.get("RANK", 0)), world=int(os.environ.get("WORLD_SIZE", 1)), local=int(os.environ.get("LOCAL_RANK", 0)))
    assert r.world == par.n_gpus, f"launched {r.world} ranks but par.n_gpus={par.n_gpus}"
    if torch.cuda.is_available():
        torch.cuda.set_device(r.local)
    if r.world > 1:
        # The other ranks wait at their next collective while rank 0 evaluates a milestone (eval_milestone): minutes, past NCCL's default 10.
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(hours=1))
    print(f"Rank {r.rank}/{r.world} is using torch device {r.device()} .")
    return r

def build(par: Train, r: Rank):
    """(model to step: compiled, and DDP-wrapped across ranks; the bare Lejepa, for checkpoints and logging; AdamW; its lr
    schedule) for training par: fresh, or from par.anneal_from's weights and Adam moments."""
    import torch
    model = make_model(par)
    ckpt = torch.load(par.anneal_from, map_location="cpu", weights_only=False) if par.anneal_from else None
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])  # strict: architecture drift fails loudly
    if r.rank0: pprint(model)
    model = model.to(r.device())
    compile_model(model, par)
    lejepa = model  # unwrapped, for checkpoints
    if r.world > 1:
        # SIGReg and projector BatchNorm statistics stay per-rank (local batch) for now.
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[r.local] if r.device().type == "cuda" else None)
        if par.grad_compress:
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            model.register_comm_hook(None, default_hooks.bf16_compress_hook)
    # AdamW with weight_decay=0 is exactly Adam. Decay only weight matrices, not biases or norm gains.
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": par.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=ckpt["opt"]["param_groups"][0]["lr"] if ckpt else par.lr,
                            betas=(0.9, par.adam_beta2))
    if ckpt is not None:  # the Adam moments and step count only: the fresh param groups keep the checkpoint's lr as the schedule's start
        ckpt["opt"]["param_groups"] = opt.state_dict()["param_groups"]
        opt.load_state_dict(ckpt["opt"])
    # By step count, so every rank uses the same lr. A trunk run warms up, then holds par.lr until it stops: it never
    # decays, so its milestone checkpoints are branch points (Train.anneal_from). An anneal has no warmup (the loaded moments
    # are warm) and cosine-decays from the checkpoint's lr to LR_FLOOR of it over steps_per_epoch.
    if ckpt is None:
        lr_scale = lambda s: min(1.0, (s + 1) / LR_WARMUP)
    else:
        lr_scale = lambda s: LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / par.steps_per_epoch)))
    return model, lejepa, opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

def train_step(model, opt, sched, batches, par: Train, r: Rank, bench: Benchmark):
    """One optimizer step on the next batch, each part in its profiler phase (bench). A step whose gradient is non-finite is
    skipped, not applied: a single one would write NaN into every weight via opt.step() (the sudden divergences in
    e00/viewsizes*); DDP ranks share the all-reduced norm, so they skip together, at one host sync per step.
    (batch on the device: Batch C Z Y X; LejepaOutput; grad norm; whether skipped; RNG state before the forward, (CPU, CUDA):
    views draw on the CPU generator and SIGReg slices on the GPU's, so a skipped step replays exactly from its dump.)"""
    import torch
    device = r.device()
    if par.cudagraphs:
        torch.compiler.cudagraph_mark_step_begin()  # previous step's graph outputs may be overwritten
    # No annotation hooks or profiler are active during the throughput baseline.
    with bench.phase("01_DATA_IO"):
        x = next(batches)
    with bench.phase("03_H2D_TRANSFER"):
        # Deferred batches are dicts of per-sample uint8 crops; finish_images copies and normalizes them.
        x = finish_images(x, device)["img"] if par.defer_image_ops else x.to(device, non_blocking=True)
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
    with bench.phase("04_FORWARD_AND_LOSS"), torch.autocast(device.type, dtype=torch.bfloat16, enabled=par.amp):
        out = model(x)
    bench.count(out)
    with bench.phase("05_BACKWARD"):
        out.loss.backward()
    with bench.phase("06_OPTIMIZER"):
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        skipped = not torch.isfinite(grad_norm).item()
        if not skipped:
            opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
    return x, out, grad_norm, skipped, rng

def save_bad_batch(par: Train, lejepa, x, rng, step: int):
    """A skipped step, replayable (replay_bad_batch): the weights, still the pre-step ones, the input (uint8: lossless) and
    the RNG state before its forward -> savedir/bad_batch_<step>.pt."""
    import torch
    torch.save({"step": step, "params": asdict(par), "model_config": asdict(lejepa.config), "model": lejepa.state_dict(),
                "x_uint8": (x.clamp(0, 1) * 255).round().byte().cpu(), "rng_cpu": rng[0], "rng_cuda": rng[1]},
               Path(par.savedir) / f"bad_batch_{step:07d}.pt")

def save_checkpoint(par: Train, lejepa, opt, sched, step: int):
    """checkpoints/step_<step>.pt, keeping the newest CHECKPOINT_KEEP; a milestone step's also kept in checkpoints/milestones/
    and evaluated there and then (eval_milestone, which deletes it once done: release_milestone). Never non-finite weights."""
    import torch
    state = lejepa.state_dict()
    bad = [k for k, v in state.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:  # never let a diverged model overwrite good checkpoints
        print(f"Not saving step {step}: {len(bad)} non-finite tensors, e.g. {bad[0]}", flush=True)
        return
    ckdir = Path(par.savedir) / "checkpoints"
    ckdir.mkdir(exist_ok=True)
    tmp, path = ckdir / "tmp.pt", ckdir / f"step_{step:07d}.pt"
    torch.save({"step": step, "params": asdict(par), "model_config": asdict(lejepa.config), "model": state,
                "opt": opt.state_dict(), "sched": sched.state_dict()}, tmp)
    tmp.replace(path)  # never leave a half-written checkpoint
    kept = ckdir / "milestones" / path.name
    b1 = not par.anneal_from and step in milestone_steps(par)  # an anneal has none: its own checkpoint is the one eval
    b2 = not kept.exists()
    for old in sorted(ckdir.glob("step_*.pt"))[:-CHECKPOINT_KEEP]:
        old.unlink()
    print(f"Saved {path}", flush=True)
    if b1 and b2:
        kept.parent.mkdir(exist_ok=True)
        os.link(path, kept)  # a hard link outlives the rolling checkpoints' prune
        eval_milestone(par, step)

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
    except Exception:
        traceback.print_exc()
    torch.cuda.empty_cache()  # the eval's memory back for training (a no-op without CUDA)

def log_metrics(f, par: Train, lejepa, x, out, grad_norm, sched, step: int, n_skipped: int, seconds: float):
    """One metrics.json row (and a progress line) for this step."""
    import torch
    loss = out.loss.detach().item()
    # Median pre-LayerNorm token norm on a global-view-sized corner of 4 samples: residual-stream
    # growth tends to precede divergence (see e00/viewsizes). Eager and grad-free, so cheap.
    g = par.global_size
    with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=par.amp):
        resid = lejepa.encoder.forward_residual(x[:4, :, :g[0], :g[1], :g[2]])
    # LeJEPA's loss = inv + lamb * sigreg, logged apart: sigreg is ~1.05 for exactly Gaussian projections
    # (its floor), and its pull scales with the per-GPU batch, so the total alone mixes invariance
    # with how Gaussian the projections are. An MAE's loss is its reconstruction error alone.
    f.write(json.dumps({"tbl": "metrics", "idx_step": step, "time": seconds, "loss": loss,
                        **{k: out[k].detach().item() for k in ("inv", "sigreg") if k in out},
                        "grad_norm": grad_norm.item(), "lr": sched.get_last_lr()[0],
                        "resid_norm": resid.float().norm(dim=-1).median().item(), "skipped": n_skipped}) + "\n")
    f.flush()
    print(f"finished step {step + 1}/{par.steps_per_epoch}, loss={loss:.4f}", flush=True)

def stop_reason(par: Train, r: Rank, out, n_skipped: int, seconds: float, step: int) -> str:
    """Why every rank stops after this step ("" to go on), agreed by one all-reduce: a non-finite loss (divergence), past
    max_hours, or too many skipped (non-finite gradient) steps."""
    import torch
    import torch.distributed as dist
    b1 = not torch.isfinite(out.loss.detach()).item()
    b2 = bool(par.max_hours) and seconds > par.max_hours * 3600
    b3 = n_skipped >= MAX_SKIPPED
    stop = torch.tensor([float(b1), float(b2), float(b3)], device=r.device())
    if r.world > 1:
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
    reasons = [f"Non-finite loss by step {step + 1}; stopping (checkpoints keep the last finite weights)",
               f"Reached max_hours={par.max_hours} at step {step + 1}",
               f"{n_skipped} steps skipped for non-finite gradients by step {step + 1}; stopping"]
    return next((m for m, s in zip(reasons, stop.tolist()) if s), "")

def record_oom(par: Train, r: Rank, step: int, err: Exception):
    """An out-of-memory error as a result row in performance.json (the table shows which configs don't fit)."""
    import torch
    row = {"tbl": "oom", "time": time.time(), "params": asdict(par), "idx_step": step, "error": str(err).splitlines()[0][:300]}
    if r.device().type == "cuda":
        row |= {"gpu_name": torch.cuda.get_device_name(r.device()), "max_mem_gb": torch.cuda.max_memory_allocated(r.device()) / 1e9}
    with open(Path(par.savedir) / "performance.json", "a") as f:
        f.write(json.dumps(row) + "\n")

def check_snapshot():
    """LSF jobs must run a code snapshot (bsub): the shared checkout's lib/ may be another sweep's by now."""
    b1 = "LSB_JOBID" in os.environ
    b2 = ".tmpcode" not in Path(__file__).resolve().parts
    assert not (b1 and b2), f"LSF job running {__file__} from the shared checkout; submit via submit() (code snapshot)"

def train(job: Train | str):
    """Train the spec job, or the one in the train.json it names (load), on every DDP rank; ends with no process group left.
    Rank 0 writes metrics.json, checkpoints (evaluating each milestone's: save_checkpoint), and the throughput window and
    profile of Benchmark. run() adds the eval of the end."""
    import torch
    import torch.distributed as dist
    start_time = time.time()
    check_snapshot()
    par = load(job, Train)
    if int(os.environ.get("RANK", 0)) == 0:  # first thing, so even a run that crashes early describes itself
        record(par, "run")
    r = setup(par)
    batches = dataloader(par)  # each rank its own, with its own miao seed (see setup)
    model, lejepa, opt, sched = build(par, r)
    n_skipped, step = 0, -1
    try:
        with open(Path(par.savedir) / "metrics.json" if r.rank0 else os.devnull, "a") as f, Benchmark(par, r.device(), r.world, r.rank0) as bench:
            for step in range(par.steps_per_epoch):
                bench.begin(step)
                x, out, grad_norm, skipped, rng = train_step(model, opt, sched, batches, par, r, bench)
                if skipped:
                    n_skipped += 1
                    print(f"Non-finite grad norm at step {step}; skipped ({n_skipped} so far)", flush=True)
                    if r.rank0 and n_skipped <= BAD_BATCHES_KEPT:
                        save_bad_batch(par, lejepa, x, rng, step)
                with bench.phase("07_LOGGING"):
                    if r.rank0 and (step % 10 == 0 or step + 1 == par.steps_per_epoch):
                        log_metrics(f, par, lejepa, x, out, grad_norm, sched, step, n_skipped, time.time() - start_time)
                bench.end(step)
                if step % 100 == 99 and (reason := stop_reason(par, r, out, n_skipped, time.time() - start_time, step)):
                    print(reason, flush=True)
                    break
                b1 = step % CHECKPOINT_EVERY == CHECKPOINT_EVERY - 1
                b2 = not par.anneal_from and step + 1 in milestone_steps(par)
                if r.rank0 and (b1 or b2):
                    save_checkpoint(par, lejepa, opt, sched, step + 1)
    except torch.OutOfMemoryError as err:
        if r.rank0:
            record_oom(par, r, step, err)
        raise  # then fail the job as usual
    if r.rank0 and latest_step(par) < step + 1:  # not saved in the loop already (that would evaluate a milestone twice)
        save_checkpoint(par, lejepa, opt, sched, step + 1)
    if r.world > 1:
        dist.destroy_process_group()

def benchmark(job: Train | str):
    """Throughput and a profiler trace of the spec job's training step (or the train.json's it names), with no training run:
    warmup_steps untimed, benchmark_steps timed (performance.json), then profile_steps traced (profile.json, profile.out,
    trace_summary.json); no metrics, checkpoints or evals. Same setup, data and step as train(). Every DDP rank."""
    import torch.distributed as dist
    check_snapshot()
    par = load(job, Train)
    par = replace(par, steps_per_epoch=par.warmup_steps + par.benchmark_steps + 1 + par.profile_steps)  # Benchmark's windows, no more
    if int(os.environ.get("RANK", 0)) == 0:
        record(par, "benchmark")
    r = setup(par)
    batches = dataloader(par)
    model, _, opt, sched = build(par, r)
    with Benchmark(par, r.device(), r.world, r.rank0) as bench:
        for step in range(par.steps_per_epoch):
            bench.begin(step)
            train_step(model, opt, sched, batches, par, r, bench)
            bench.end(step)
    if r.world > 1:
        dist.destroy_process_group()

def latest_step(par: Train) -> int:
    """The step of par's newest checkpoint (-1 if it has none)."""
    return max((int(p.stem.split("_")[1]) for p in (Path(par.savedir) / "checkpoints").glob("step_*.pt")), default=-1)

def run(job: Train | str):
    """The LSF training job: train the spec job (or its train.json's), then rank 0 evaluates its last finite checkpoint (pca
    of own_eval, and probe_evals).

    Evals run after train() has destroyed the process group, so the other ranks exit instead of waiting at a
    collective for minutes; not for a trunk (a run with milestones: not an anneal) that stopped on a milestone: train()
    evaluated that checkpoint already (eval_milestone). One that max_hours stopped between two does. (Throughput
    benchmarks are their own entrypoint, benchmark, with no evals.)
    """
    par = load(job, Train)
    train(par)
    b1 = int(os.environ.get("RANK", 0)) == 0
    b2 = not par.anneal_from and latest_step(par) in milestone_steps(par)
    if b1 and not b2:
        pca(own_eval(par))
        for e in probe_evals(par):
            Path(e.savedir).mkdir(parents=True, exist_ok=True)
            probe(e)


def replay_bad_batch(path: str):
    """Replay a skipped (non-finite gradient) training step from its bad_batch_*.pt dump.

    1. Eager under torch.autograd.detect_anomaly: raises at the first backward op that produced NaN/inf, with
       the traceback of the forward op that created it. Eager kernels can differ numerically from compiled
       ones, so 1 may not reproduce what 3 does.
    2. Eager, with SDPA forced to each fused attention backend in turn (cuDNN, flash, mem-efficient). If only
       cuDNN gives non-finite grads, its attention backward is the culprit. Eager because compiled replays
       ignored sdpa_kernel (identical losses for every backend). The math backend is left out: it materializes
       Batch Head N N attention and doesn't fit at training batch sizes. CUDA only (fused kernels).
    3. As trained (compiled per the run's Train): does the gradient come out non-finite again? Runs last:
       its CUDA graph pools stay allocated. Uses torch's default SDPA pick (cuDNN), as runs before run()
       disabled cuDNN attention did; 1 does too.
    Returns loss and non-finite param count per replay; also prints which params kept finite grads.
    """
    import torch
    from contextlib import nullcontext
    from torch.nn.attention import SDPBackend, sdpa_kernel
    # Like training, replay depends on lib/: only for dumps of this commit's own experiment (replaylsf submits it).
    ours = Path("outdir", EXPERIMENT).resolve()
    assert Path(path).resolve().is_relative_to(ours), f"{path} isn't from this commit's experiment: replay it from its experiment's commit"
    dump = torch.load(path, map_location="cpu", weights_only=False)
    par = parse(Train, dump["params"])
    record(par, "replay_bad_batch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision(par.f32mode)
    x = dump["x_uint8"].to(device).float() / 255  # Batch C Z Y X, exactly the training input

    def step(label: str, compiled: bool, backend: SDPBackend | None = None) -> tuple[float, int]:
        model = make_model(par)  # this commit's own experiment (asserted above), so its own architecture
        model.load_state_dict(dump["model"])  # the pre-step weights
        model.to(device).train()
        if compiled:
            compile_model(model, par)
            if par.cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
        # Same view crops (CPU generator) and SIGReg slices (GPU generator) as the original step.
        torch.set_rng_state(dump["rng_cpu"])
        if device.type == "cuda" and dump["rng_cuda"] is not None:
            torch.cuda.set_rng_state(dump["rng_cuda"], device)
        with sdpa_kernel(backend) if backend else nullcontext():
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=par.amp):
                out = model(x)
            assert isinstance(out, LejepaOutput)
            out.loss.backward()
        bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        # NaN flows backward from where it starts, so the params that stay finite locate the source.
        good = [n for n, p in model.named_parameters() if p.grad is not None and torch.isfinite(p.grad).all()]
        print(f"{label}: loss {out.loss.item():.6f}, {len(bad)} params with non-finite grads"
              + (f", e.g. {bad[:3]}; finite: {good}" if bad else ""), flush=True)
        return out.loss.item(), len(bad)

    print(f"Replaying step {dump['step']} of {par.savedir}", flush=True)
    try:
        with torch.autograd.detect_anomaly(check_nan=True):
            res = {"eager": step("eager, detect_anomaly", compiled=False)}
    except RuntimeError as err:  # the forward traceback of the culprit op was printed as a warning just before
        print(f"eager, detect_anomaly: {err}", flush=True)
        res = {"eager": (math.nan, -1)}
    for backend in [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] * (device.type == "cuda"):
        res[backend.name] = step(f"eager, {backend.name}", compiled=False, backend=backend)
    res["as trained"] = step("as trained", compiled=True)
    return res

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

def release_milestone(e: Eval):
    """Delete the trunk milestone e's weights came from (see milestone_of: an eval of it, or an anneal's own eval) once nothing
    reads it any more: e's pca and probe have both written their results (the second to finish deletes it; a failed job keeps
    it for a retry). A milestone is a full checkpoint, 0.6 GB for m."""
    kept = milestone_of(e) or milestone_of(e.train)
    if kept and all((Path(e.savedir) / f).is_file() for f in ("pca.json", "probe.json")):
        Path(kept).unlink(missing_ok=True)

def pca(job: Eval | str):
    """PCA maps (pca_maps) of the spec job's weights (or those of the eval.json it names). run() does it at the end of
    training; `lsf pca n` redoes it."""
    e = load(job, Eval)
    model, step = load_checkpoint(e)
    record(e, "pca")  # argv says whether it ran inside `run` or on its own
    pca_maps(model.encoder, e.train, Path(e.savedir), step)
    release_milestone(e)

def pca_maps(encoder, par: Train, savedir: Path, step: int):
    """PCA maps of patch-token embeddings on a held-out EB val crop -> savedir/pca2.png, pca.json.

    pca2.png rows: the middle z-slice of each of the crop's 3 tile layers (a token's tile is the 96^3 view the encoder
    saw at once, so rows from one layer would share one tile code). Columns: EM | PCA of tokens | PCA after subtracting
    each tile's mean token | PCA fitted within each tile (colors are not comparable across tiles) | pre-LayerNorm
    token L2 norm.
    """
    import torch
    from PIL import Image
    device = next(encoder.parameters()).device
    encoder.eval()

    # 3 tiles deep in z so each row of pca2.png is a different tile.
    shape = (3 * par.global_size[0], 4 * par.global_size[1], 4 * par.global_size[2])
    img = eb_val_crop(shape)

    # Tokens from non-overlapping tiles at the training global-view size, stitched into one grid.
    tile, patch = par.global_size, encoder.patch_embed.patch_size
    assert all(s % t == 0 for s, t in zip(shape, tile)), f"crop {shape} must tile by {tile}"
    grid = [s // p for s, p in zip(shape, patch)]
    feats = torch.empty(*grid, encoder.config.embed_dim)  # Z Y X D, tokens (after the final LayerNorm)
    norms = torch.empty(*grid)  # Z Y X, token norms before the final LayerNorm
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for z, y, x in product(*(range(0, s, t) for s, t in zip(shape, tile))):
            r = encoder.forward_residual(img[None, :, z:z + tile[0], y:y + tile[1], x:x + tile[2]].to(device))
            t = encoder.norm(r)
            tz, ty, tx = (o // p for o, p in zip((z, y, x), patch))
            gz, gy, gx = (t_ // p for t_, p in zip(tile, patch))
            feats[tz:tz + gz, ty:ty + gy, tx:tx + gx] = t[0].float().reshape(gz, gy, gx, -1).cpu()
            norms[tz:tz + gz, ty:ty + gy, tx:tx + gx] = r[0].float().norm(dim=-1).reshape(gz, gy, gx).cpu()

    # Tile-centered tokens: subtract each tile's mean token, removing the per-view code and keeping within-view structure.
    g = [t_ // p for t_, p in zip(tile, patch)]  # tokens per tile side
    nt = [n // gi for n, gi in zip(grid, g)]  # tiles per axis
    tiled = feats.reshape(nt[0], g[0], nt[1], g[1], nt[2], g[2], encoder.config.embed_dim)
    centered = (tiled - tiled.mean(dim=(1, 3, 5), keepdim=True)).reshape(*grid, encoder.config.embed_dim)
    rgb, erank, var = pca_rgb(feats.reshape(-1, encoder.config.embed_dim), grid)
    rgb_c, erank_c, var_c = pca_rgb(centered.reshape(-1, encoder.config.embed_dim), grid)
    rgb_t = np.empty((*grid, 3), np.uint8)  # PCA fitted separately within each tile
    for i, j, k in product(*map(range, nt)):
        rgb_t[i * g[0]:(i + 1) * g[0], j * g[1]:(j + 1) * g[1], k * g[2]:(k + 1) * g[2]] = pca_rgb(tiled[i, :, j, :, k, :].reshape(-1, encoder.config.embed_dim), g)[0]
    between_tile = 1 - float(centered.var(dim=(0, 1, 2)).sum() / feats.var(dim=(0, 1, 2)).sum())

    # Pre-LayerNorm token norms: "register"-like tokens that store global information show up as sparse
    # high-norm outliers. (After the final LayerNorm every token's norm is ~sqrt(width), so those can't show this.)
    med = float(norms.median())
    lo, hi = torch.quantile(norms.flatten(), torch.tensor([0.01, 0.999]))
    norm_u8 = ((norms - lo) / (hi - lo)).clamp(0, 1).mul(255).byte().numpy()

    # Rows: the middle z-slice of each tile layer; columns: EM | PCA | tile-centered PCA | per-tile PCA | token norm, upsampled to voxels.
    up = lambda a: a.repeat(patch[1], axis=0).repeat(patch[2], axis=1)
    vgap = np.full((shape[1], 8, 3), 255, np.uint8)
    rows = []
    for gz in [g[0] // 2 + g[0] * i for i in range(nt[0])]:
        zv = gz * patch[0] + patch[0] // 2
        em = np.repeat((img[0, zv].numpy() * 255).astype(np.uint8)[..., None], 3, axis=2)
        nm = np.repeat(norm_u8[gz][..., None], 3, axis=2)
        rows.append(np.concatenate([em, vgap, up(rgb[gz]), vgap, up(rgb_c[gz]), vgap, up(rgb_t[gz]), vgap, up(nm)], axis=1))
    gap = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
    Image.fromarray(np.concatenate([r for row in rows for r in (row, gap)][:-1], axis=0)).save(savedir / "pca2.png")
    stats = {"tbl": "pca", "step": step, "effective_rank": erank, "explained_variance": var,
             "centered_effective_rank": erank_c, "centered_explained_variance": var_c,
             "between_tile_variance": between_tile,  # fraction of token variance explained by tile means
             "norm_median": med, "norm_p99": float(torch.quantile(norms.flatten(), 0.99)), "norm_max": float(norms.max()),
             "norm_outliers": int((norms > 2 * med).sum()), "n_tokens": norms.numel()}
    (savedir / "pca.json").write_text(json.dumps(stats) + "\n")
    print(f"{savedir}: step {step}, effective rank {erank:.1f}, top-3 variance {sum(var[:3]):.2f}, "
          f"between-tile variance {between_tile:.2f}, norm outliers (>2x median) {stats['norm_outliers']}/{norms.numel()}; wrote pca2.png")

def pca_sizes(job: Eval | str):
    """The input size at test time: the spec job's encoder run on one held-out EB val crop in non-overlapping cubic tiles
    of each TEST_TILES side, the crop 2 x 2 tiles of the largest in y x and one deep (so the smallest are many), ->
    savedir/pca_tile_<side>.png and a pca_sizes.json row per side. Each png is one z-slice, the same voxels at every side:
    EM | PCA of its tokens | PCA after subtracting each tile's mean token, fitted on that slice. The row's effective rank
    and explained variance are of 65536 tokens drawn from the whole crop, its between-tile variance (the share of token
    variance that is each tile's mean) of all of them."""
    import torch
    from PIL import Image
    e = load(job, Eval)
    model, step = load_checkpoint(e)
    record(e, "pca_sizes")
    encoder, device = model.encoder, next(model.parameters()).device
    p = encoder.patch_embed.patch_size
    big = max(TEST_TILES)
    shape = (big, 2 * big, 2 * big)
    img = eb_val_crop(shape)[0].to(device)  # Z Y X
    zt = shape[0] // 2 // p[0]  # the slice shown: the middle of a token row near the crop's middle
    em = np.repeat((img[zt * p[0] + p[0] // 2].cpu().numpy() * 255).astype(np.uint8)[..., None], 3, axis=2)
    up = lambda a: a[0].repeat(p[1], axis=0).repeat(p[2], axis=1)  # 1 Gy Gx 3 tokens -> Y X 3 voxels
    vgap = np.full((shape[1], 8, 3), 255, np.uint8)
    (Path(e.savedir) / "pca_sizes.json").write_text("")  # a redo (lsf pca_sizes) starts over
    for side in TEST_TILES:
        t0 = time.time()
        feats = tile_features(encoder, img, (side, side, side)).float()  # Gz Gy Gx D
        grid, d = feats.shape[:3], feats.shape[-1]
        g = side // p[0]  # tokens per tile side
        tiled = feats.reshape(grid[0] // g, g, grid[1] // g, g, grid[2] // g, g, d)
        centered = (tiled - tiled.mean(dim=(1, 3, 5), keepdim=True)).reshape(*grid, d)
        between = 1 - float(centered.var(dim=(0, 1, 2)).sum() / feats.var(dim=(0, 1, 2)).sum())
        sample = feats.reshape(-1, d)[torch.randperm(feats[..., 0].numel(), device=device, generator=torch.Generator(device).manual_seed(0))[:65536]]
        _, erank, var = pca_rgb(sample, (len(sample),))
        rgb, rgb_c = pca_rgb(feats[zt].reshape(-1, d), (1, *grid[1:]))[0], pca_rgb(centered[zt].reshape(-1, d), (1, *grid[1:]))[0]
        Image.fromarray(np.concatenate([em, vgap, up(rgb), vgap, up(rgb_c)], axis=1)).save(Path(e.savedir) / f"pca_tile_{side:03d}.png")
        row = {"tbl": "pca_sizes", "step": step, "tile": side, "tiles": (shape[0] // side) * (shape[1] // side) * (shape[2] // side),
               "effective_rank": erank, "explained_variance": var, "between_tile_variance": between, "seconds": time.time() - t0}
        with open(Path(e.savedir) / "pca_sizes.json", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"tile {side}: {row['tiles']} tiles, effective rank {erank:.1f}, between-tile variance {between:.3f}, {row['seconds']:.0f} s", flush=True)
        del feats, tiled, centered
        torch.cuda.empty_cache()

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

def probe(job: Eval | str):
    """Linear affinity probe (lib.probe) of the spec job's weights (or those of the eval.json it names), frozen: one Linear per
    token to mia-evals' 6 affinity channels at every voxel of its patch, fitted on the train block of its probe set
    (par.probe_on, lib.data.PROBE_SETS) against its instance labels, scored on its test blocks (averaged) as mia-evals'
    neuron-instance task. Every block read at 8 nm, as training reads (read_block).
      savedir/probe.json          test-block BCE and boundary AP per channel (cheap, immediate)
      savedir/probe_fit.json      the fit curve every 100 steps: training loss, and BCE and boundary AP on a
                                  fixed held-out sample (monitoring only, nothing selects on it)
    Weights: par.init_from (load_checkpoint): this run's checkpoint, an older run's, or random (the baseline).
      savedir/probe/{fit,test}/hemibrain_eb_{split}.zarr   mia-evals affinity artifacts, (6, X, Y, Z) float16, for
                                  `mia-evals score` with truth_kind = "instances" (reads the GT from the store)
      savedir/probe.png           test block, middle z: EM | true boundaries | predicted boundaries (probe_<block>.png
                                  for any further test blocks)
    Reads get whole tiles of context (the training view, tile) around each block (at most PROBE_SIDE a side), as in
    training; where a block's context leaves its volume, the image is edge-replicated.
    """
    import torch
    from lib.probe import AFFINITY_OFFSETS_XYZ, affinities, fit_probe, to_tokens, to_voxels
    par = load(job, Eval)
    tile = par.train.global_size
    model, step = load_checkpoint(par)
    record(par, "probe")
    encoder, device = model.encoder, next(model.parameters()).device
    p = encoder.patch_embed.patch_size
    sets = PROBE_SETS[par.probe_on]
    assert not par.score or par.probe_on == "hemibrain_eb", f"{par.savedir}: score is hemibrain_eb's (its artifacts and boxes), not {par.probe_on}'s"
    assert len(sets["train"]) == 1, f"{par.probe_on}: the probe fits on one block"
    assert len(set(p)) == 1, f"patch {p} must be cubic"
    offsets = [o[::-1] for o in AFFINITY_OFFSETS_XYZ]  # x y z offsets; the model sees z y x

    def tokens(img, m, size):  # the box's token features, N D
        f = tile_features(encoder, img, tile)
        t0, n = [mi // q for mi, q in zip(m, p)], [s // q for s, q in zip(size, p)]
        return f[t0[0]:t0[0] + n[0], t0[1]:t0[1] + n[1], t0[2]:t0[2] + n[2]].reshape(-1, f.shape[-1])

    tests = [read_block(v, tile, p, device) for v in sets["test"]]  # (img, labels, m, size) each
    img, labels, m, size = tests[0]
    aff, valid = affinities(labels, offsets, sets["test"][0].sparse)
    feats = tokens(img, m, size)
    g = torch.Generator(device=device).manual_seed(0)
    i = torch.randperm(len(feats), device=device, generator=g)[:16384]  # held-out tokens, fit curve only
    held = (feats[i], to_tokens(aff, p[0])[i], to_tokens(valid, p[0])[i])
    img, labels, m, size = read_block(sets["train"][0], tile, p, device)
    aff, valid = affinities(labels, offsets, sets["train"][0].sparse)
    feats = tokens(img, m, size)
    del img, labels
    head, curve = fit_probe(feats, to_tokens(aff, p[0]), to_tokens(valid, p[0]), held)
    del feats, aff, valid, held

    def predict(img, m, size):  # C Z Y X sigmoid probabilities over the box
        with torch.no_grad():
            prob = torch.cat([torch.sigmoid(head(f.float())).half() for f in tokens(img, m, size).split(65536)])
        return to_voxels(prob, tuple(s // p[0] for s in size), p[0])

    run_name = "-".join(Path(par.savedir).parts[1:])  # mia-evals row name: letters, digits, _ and - only

    def artifact(split: str, vol: ProbeVolume, pred):  # for score
        (x0, _), (y0, _), (z0, _) = vol.box
        write_affinities(Path(par.savedir) / "probe" / split / f"hemibrain_eb_{split}.zarr", pred.permute(0, 3, 2, 1).cpu().numpy(),
                         origin=[x0, y0, z0], convention="sigmoid(logit), linear decoder",
                         run=run_name, step=step, axes="xyz", source_path=vol.path, source_label_key=HEMIBRAIN_EB_LABELS,
                         native_box=vol.box, scale=[1, 1, 1], annotated_box=HEMIBRAIN_EB_PROBE_ANNOTATED[split], covers_full_box=False)

    (Path(par.savedir) / "probe_fit.json").write_text("".join(json.dumps({"tbl": "probe_fit", **c}) + "\n" for c in curve))
    per_block = []
    for k, (vol, (img, labels, m, size)) in enumerate(zip(sets["test"], tests)):
        pred = predict(img, m, size)
        per_block.append(score_block(pred, labels, vol.sparse))
        boundary_png(Path(par.savedir) / ("probe.png" if k == 0 else f"probe_{vol.name}.png"), inner(img, m, size), labels, pred)
        if par.score:
            artifact("test", vol, pred)
    if par.score:  # also hemibrain_eb's mia-evals fit block, whose predictions only feed the artifacts
        vol = ProbeVolume("hemibrain_eb_fit", sets["test"][0].path, sets["test"][0].label_key, HEMIBRAIN_EB_PROBE_BOXES["fit"])
        img, _, m, size = read_block(vol, tile, p, device)
        artifact("fit", vol, predict(img, m, size))
    stats = {"tbl": "probe", "step": step, "run": run_name, "probe_on": par.probe_on, "test_blocks": [v.name for v in sets["test"]],
             **mean_scores(per_block)}
    (Path(par.savedir) / "probe.json").write_text(json.dumps(stats) + "\n")
    print(f"{par.savedir}: probe at step {step}: boundary AP (short-range mean) {stats['boundary_ap_short']:.3f}{'; wrote probe/' * par.score}", flush=True)
    release_milestone(par)

def finetune(job: Finetune | str):
    """A UNETR (lib.decoders.unetr: skip connections into the encoder's own blocks) on the spec job's weights (or those of the
    finetune.json it names), fitted to mia-evals' 6 affinity channels of its probe set's (par.probe_on) train block, or of the
    par.fit_side cube about its centre: on crops of the encoder's view size, with the encoder trained too unless
    par.freeze_encoder, and scored as it fits on the set's test blocks, tiled at that size. Blocks read as probe reads them.
      savedir/finetune_curve.json   a first row (tbl "finetune_data"): the weights' step, the fit cube's size and labelled voxels
                              (the GT the fit had), the test blocks. Then the fit curve (tbl "finetune", fit_unetr's): the batch
                              loss every 100 steps, and at 100, 200, 400, ... steps and the last, the test blocks' scores
                              averaged, as probe.json has them. (Not finetune.json: that is the job's spec, dump's.)
      savedir/finetune.png    test block after the fit, middle z: EM | true boundaries | predicted boundaries
                              (finetune_<block>.png for any further test blocks), written last: the job finished
    """
    import torch
    from lib.decoders import Unetr, UnetrConfig, fit_unetr
    from lib.probe import AFFINITY_OFFSETS_XYZ, affinities
    par = load(job, Finetune)
    torch.set_float32_matmul_precision(par.train.f32mode)  # as the encoder trained: TF32 matmuls ("high"), not full fp32
    tile = par.train.global_size
    model, step = load_checkpoint(par)
    record(par, "finetune")
    encoder, device = model.encoder, next(model.parameters()).device
    p = encoder.patch_embed.patch_size
    sets = PROBE_SETS[par.probe_on]
    side = par.fit_side or PROBE_SIDE
    assert len(sets["train"]) == 1, f"{par.probe_on}: the fit is on one block"
    assert side % p[0] == 0, f"{par.savedir}: fit_side {side} must be whole patches of {p[0]}"
    unetr = Unetr(encoder, UnetrConfig(out_dim=len(AFFINITY_OFFSETS_XYZ))).to(device)
    encoder.requires_grad_(not par.freeze_encoder)
    img, labels, m, size = read_block(sets["train"][0], tile, p, device, side)
    assert all(s >= t for s, t in zip(size, tile)), f"{par.savedir}: the fit cube {size} is smaller than a crop {tile}"
    aff, valid = affinities(labels, [o[::-1] for o in AFFINITY_OFFSETS_XYZ], sets["train"][0].sparse)  # x y z offsets; the model sees z y x
    data = {"tbl": "finetune_data", "step": step, "fit_size": size, "fit_labelled": int((labels != 0).sum()),
            "test_blocks": [v.name for v in sets["test"]]}
    del labels
    tests = [read_block(v, tile, p, device, context=True) for v in sets["test"]]  # (img, labels, m, size) each

    def predict(img, m, size):  # C Z Y X sigmoid probabilities over the box
        return torch.sigmoid(inner(tile_logits(unetr, img, tile), m, size)).half()

    def test_scores() -> dict:
        return mean_scores([score_block(predict(t_img, t_m, t_size), t_labels, v.sparse) for v, (t_img, t_labels, t_m, t_size) in zip(sets["test"], tests)])

    curve = fit_unetr(unetr, inner(img, m, size)[None], aff, valid, tile, test_scores, steps=par.steps)
    del img, aff, valid
    rows = [data] + [{"tbl": "finetune", **c} for c in curve]
    (Path(par.savedir) / "finetune_curve.json").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for k, (vol, (img, labels, m, size)) in enumerate(zip(sets["test"], tests)):
        boundary_png(Path(par.savedir) / ("finetune.png" if k == 0 else f"finetune_{vol.name}.png"), inner(img, m, size), labels, predict(img, m, size))
    print(f"{par.savedir}: finetune of step {step} on {size} voxels of {par.probe_on}: boundary AP (short-range mean) "
          f"{curve[-1]['boundary_ap_short']:.3f} after {par.steps} steps", flush=True)

def tiles(job: Finetune | str):
    """finetune's fit, without its scoring, and tile_logits' blended prediction on its first test block: savedir/tiles.png,
    finetune.png's panels (middle z: EM | true boundaries | predicted boundaries). The fit is seeded, so these are finetune's
    predictions, redone with the current tiling; nothing else is written."""
    import torch
    from lib.decoders import Unetr, UnetrConfig, fit_unetr
    from lib.probe import AFFINITY_OFFSETS_XYZ, affinities
    par = load(job, Finetune)
    torch.set_float32_matmul_precision(par.train.f32mode)
    tile = par.train.global_size
    model, step = load_checkpoint(par)
    encoder, device = model.encoder, next(model.parameters()).device
    p = encoder.patch_embed.patch_size
    sets = PROBE_SETS[par.probe_on]
    unetr = Unetr(encoder, UnetrConfig(out_dim=len(AFFINITY_OFFSETS_XYZ))).to(device)
    encoder.requires_grad_(not par.freeze_encoder)
    img, labels, m, size = read_block(sets["train"][0], tile, p, device, par.fit_side or PROBE_SIDE)
    aff, valid = affinities(labels, [o[::-1] for o in AFFINITY_OFFSETS_XYZ], sets["train"][0].sparse)  # x y z offsets; the model sees z y x
    fit_unetr(unetr, inner(img, m, size)[None], aff, valid, tile, steps=par.steps)
    img, labels, m, size = read_block(sets["test"][0], tile, p, device, context=True)
    pred = torch.sigmoid(inner(tile_logits(unetr, img, tile), m, size)).half()
    boundary_png(Path(par.savedir) / "tiles.png", inner(img, m, size), labels, pred)
    print(f"{par.savedir}: wrote tiles.png, finetune of step {step} for {par.steps} steps", flush=True)

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
    b2 = fn in ("pca", "pca_sizes", "probe", "score") and isinstance(job, Eval)
    b3 = fn in ("finetune", "tiles") and isinstance(job, Finetune)
    assert b1 or b2 or b3, (f"{job.savedir}: no LSF job {fn!r} for a {type(job).__name__} (run and benchmark take a Train; pca, pca_sizes, "
                            f"probe and score an Eval; finetune a Finetune)")
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
        bsub(job.savedir, fn, {"pca": 30, "pca_sizes": 60, "probe": 60, "finetune": 120, "tiles": 120}[fn], 1, job.train.queue, f"python {{code}}/experiment.py {fn} {spec}",
             after=after)

def start(p: Spec):
    """Submit p fresh: its savedir emptied (old contents -> outdir/.trash/), then its jobs: a training run (which evals itself
    after), an eval's jobs (Eval.jobs: pca and probe, unless it says otherwise), or a finetune."""
    trash(p.savedir)
    for fn in ["run"] if isinstance(p, Train) else ["finetune"] if isinstance(p, Finetune) else p.jobs:
        submit(fn, p)

def lsf(fn: str, target: int | str):
    """Redo one job on LSF, of roots()[target] or of the spec file target names (any run's, e.g. a milestone eval's
    outdir/.../d3/step_0005000/eval.json): `lsf probe 3` after changing probe. pca, probe or score of an eval, or of a training
    run's own checkpoint (own_eval), in place; finetune of a finetune, in place; run restarts a training run fresh (start), its
    milestone evals with it."""
    assert_committed()
    if isinstance(target, int):
        p = roots()[target]
    else:
        p = load(target, Eval if target.endswith("eval.json") else Finetune if target.endswith("finetune.json") else Train)
    if fn == "run":
        start(p)
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

def tilesall():
    """Submit tiles (in place: the run dirs are kept) for the finetunes the e10 report's Predictions panel shows (frozen encoder,
    256^3 of GT) that have no tiles.png yet. Idempotent."""
    check_experiment()
    assert_committed()
    for r in roots():
        if isinstance(r, Finetune) and r.freeze_encoder and r.fit_side == 256 and not (Path(r.savedir) / "tiles.png").is_file():
            submit("tiles", r)

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


ENTRYPOINTS = [run, benchmark, replay_bad_batch, pca, pca_sizes, probe, score, finetune, tiles, lsf, replaylsf, submitall, tilesall, retryall]

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
