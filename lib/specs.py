"""The specs of a job (Train, Eval, Finetune: what experiment.py's roots() lists and its LSF code submits), what follows from one
(its model and the milestones it keeps), and the weights it loads."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from lib.encoders import EncoderConfig
from lib.models import Lejepa, LejepaConfig, Mae, MaeConfig, MaskConfig
from lib.views import ViewConfig
from lib.util import *
from lib.types import *


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

def record(par: Spec, fn: str):
    """Append one row to par.savedir/runs.json: which experiment function ran (fn), its spec (params), code_provenance(),
    argv and LSF job. Written first thing, so even a run that crashes before any results describes itself.
    Repro for the dir's artifacts: that commit + argv. The command that submitted the job is in
    outdir/_log/commands.jsonl (log_command)."""
    row = {"fn": fn, "params": asdict(par), **code_provenance(), "argv": sys.argv, "lsf_job": os.environ.get("LSB_JOBID")}
    Path(par.savedir).mkdir(parents=True, exist_ok=True)
    with open(Path(par.savedir) / "runs.json", "a") as f:
        f.write(json.dumps(row) + "\n")

