"""Train (how a model is trained, in groups: the spec every task of an experiment shares), what follows from one (its model), and
training it: the data loader, DDP setup, model and optimizer, the step, the Trainer that takes a few at a time, checkpoints, the stop
rules, throughput benchmark and replay of a skipped step. Tasks and their bookkeeping (code snapshot check, run records, evals,
branches) are tasks.py's and lib.task_utils'."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from pprint import pprint
from typing import Literal

import numpy as np

from miao import VolumeDataset, collate_deferred, finish_images
from miao.config import MiaoConfig

from lib import models
from lib.benchmark import Benchmark
from lib.data import sampling
from lib.encoders import EncoderConfig
from lib.losses import LejepaOutput
from lib.readers import volume_config
from lib.util import *
from lib.views import ViewConfig, ViewMaker

# Training (Trainer): the lr warmup (steps from a fresh model), the max global grad norm, and steps whose gradient is
# non-finite (skipped, not applied: the run stops after MAX_SKIPPED; the first BAD_BATCHES_KEPT dumped).
LR_WARMUP, GRAD_CLIP = 1000, 1.0
MAX_SKIPPED, BAD_BATCHES_KEPT = 100, 2


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
    window: tuple[int, int, int] = (104, 104, 104)  # what the loader reads per sample, voxels (8 nm isotropic: it reads every volume at 8 nm)
    view: tuple[int, int, int] = (88, 88, 88)  # the view each sample trains on, voxels (a multiple of the encoder's patch)

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
    n_workers: int = 16  # data loader workers per GPU (lib.task_utils.CPUS_PER_GPU - 1 at most)
    prefetch_factor: int = 2

@dataclass(slots=True)
class Speed:
    f32mode: Literal["highest", "high", "medium"] = "high"
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



def collate_images(samples):
    import torch
    return torch.stack([s["img"] for s in samples])

def compile_model(model: models.Lejepa | models.Mae, par: Train):
    """torch.compile the encoder in place, as par says (no-op unless par.speed.compile)."""
    import torch
    assert par.speed.compile or not (par.speed.cudagraphs or par.speed.compile_blocks), "cudagraphs / compile_blocks require compile"
    if not par.speed.compile:
        return
    torch._logging.set_logs(recompiles=True)  # pyright: ignore[reportPrivateImportUsage]  # recompiles show up in job_*.log
    if par.speed.cudagraphs:
        # CUDA graphs replay whole kernel sequences, removing per-kernel CPU launch cost.
        # They need static shapes: displace has exactly 2 encoder shapes (the whole view, and its visible tokens), basic has ~9.
        kwargs = dict(mode="reduce-overhead", dynamic=False)
    else:
        kwargs = dict(dynamic=True)  # Views change shape every step.
    # A whole-encoder compile releases every gradient at the end of one fused backward, so DDP can't
    # start all-reducing until backward is done. Per-block compile releases each block's grads as it finishes.
    for m in (model.encoder.blocks if par.speed.compile_blocks else [model.encoder]):
        m.compile(**kwargs)
    if par.speed.eager_patch_embed:
        # Inductor's fused patchify kernel (triton_poi_fused__to_copy__unsafe_view_clone_permute_view_0: the input's
        # reshape/permute + bf16 cast) hits an illegal memory access under static shapes at batch 64 on B300 and
        # H200 (e00/b300-compile, replay of b300-train8h/d0 with CUDA_LAUNCH_BLOCKING). Run it eagerly instead.
        pe = model.encoder.patch_embed
        setattr(pe, "forward", torch.compiler.disable(pe.forward))  # instance override; nn.Module.__call__ uses it

def save_view_pngs(par: Train, savedir: str, dataset, n_samples: int = 3):
    """Save what the model sees: for n_samples fresh samples, the input and each of its views, as
    center-z slices at native voxel size, left to right (input | globals | locals) -> savedir/views_{i}.png."""
    from PIL import Image
    cfg = model_config(par)
    p = cfg.encoder.patch_size
    maker = ViewMaker(cfg.view, patch_size=(p, p, p) if isinstance(p, int) else p)
    samples = [dataset[i] for i in range(n_samples)]
    x = finish_images(collate_deferred(samples))["img"] if par.speed.defer_image_ops else collate_images(samples)  # Batch C Z Y X
    globals_, locals_ = maker(x)
    to_u8 = lambda v: (v[0, v.shape[1] // 2].clamp(0, 1) * 255).byte().numpy()  # C Z Y X -> Y X center slice
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    for b in range(n_samples):
        panels = [to_u8(v[b]) for v in [x, *globals_, *locals_]]
        # Bottom-pad smaller views to the input height and separate panels with a white gap.
        panels = [np.pad(p, ((0, x.shape[-2] - p.shape[0]), (0, 8)), constant_values=255) for p in panels]
        Image.fromarray(np.concatenate(panels, axis=1)).save(savedir / f"views_{b}.png")
    print(f"Saved {n_samples} input + view slices to {savedir}/views_*.png", flush=True)

def dataloader(par: Train, savedir: str, steps: int):
    import torch
    # The datasets' train volumes (lib/data.py), images only, each sampled by its weight alone (lib.data.sampling); output is z y x.
    volumes = [volume_config(v, v.zyx, weight=w) for v, w in sampling(par.data.sets, par.data.power)]
    mcfg = MiaoConfig(
        volumes=volumes,
        patch_size=list(par.data.window),
        resolutions=[[8.0, 8.0, 8.0]],
        # Extra batches keep workers busy (as in real training) through the last, profiled steps.
        samples_per_epoch=par.optim.batch_size * (steps + par.hardware.n_workers * par.hardware.prefetch_factor),
        sampling="random",
        output_axes="lzyx",
        defer_image_ops=par.speed.defer_image_ops,
    )
    dl = VolumeDataset(mcfg)
    loader = torch.utils.data.DataLoader(
      dl,
      batch_size=par.optim.batch_size,
      num_workers=par.hardware.n_workers,
      collate_fn=collate_deferred if par.speed.defer_image_ops else collate_images,
      prefetch_factor=par.hardware.prefetch_factor if par.hardware.n_workers else None,  # n_workers 0 (tests): loads in this process, no workers
      pin_memory=torch.cuda.is_available(),
      multiprocessing_context="spawn" if par.hardware.n_workers else None,
    )
    batches = iter(loader)
    if int(os.environ.get("RANK", 0)) == 0:
        save_view_pngs(par, savedir, dl)
    return batches

def setup(par: Train) -> Rank:
    """Torch settings, this rank's CUDA device and the DDP process group for training par; this process's Rank. Every rank."""
    import torch
    import torch.distributed as dist
    torch.set_float32_matmul_precision(par.speed.f32mode)
    # cuDNN's fused attention backward returns NaN grads on some trained weights where flash and mem-efficient
    # don't (replay_bad_batch on e00/nanhunt_beta95/d9): the "divergences" of e00/viewsizes* and nanhunt*.
    # Runs before 2026-09-28 used it (torch's default pick on H200/B300).
    torch.backends.cuda.enable_cudnn_sdp(False)
    # Set by torchrun. Unset (plain `python`, tests) means a single process.
    # Don't seed torch identically across ranks: DataLoader workers derive miao's numpy seed from it.
    r = Rank(rank=int(os.environ.get("RANK", 0)), world=int(os.environ.get("WORLD_SIZE", 1)), local=int(os.environ.get("LOCAL_RANK", 0)))
    assert r.world == par.hardware.n_gpus, f"launched {r.world} ranks but par.hardware.n_gpus={par.hardware.n_gpus}"
    if torch.cuda.is_available():
        torch.cuda.set_device(r.local)
    if r.world > 1:
        # The other ranks wait at their next collective while rank 0 saves a checkpoint: minutes for a big one, past NCCL's default 10.
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(hours=1))
    print(f"Rank {r.rank}/{r.world} is using torch device {r.device()} .")
    return r

def build(par: Train, r: Rank, ckpt: dict | None):
    """(model to step: compiled, and DDP-wrapped across ranks; the bare Lejepa, for checkpoints and logging; AdamW; the step it
    starts at) for training par: fresh, or from the loaded checkpoint ckpt's weights, Adam moments and step."""
    import torch
    model = make_model(par)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])  # strict: architecture drift fails loudly
    if r.rank0: pprint(model)
    model = model.to(r.device())
    compile_model(model, par)
    lejepa = model  # unwrapped, for checkpoints
    if r.world > 1:
        # SIGReg and projector BatchNorm statistics stay per-rank (local batch) for now.
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[r.local] if r.device().type == "cuda" else None)
        if par.optim.grad_compress:
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            model.register_comm_hook(None, default_hooks.bf16_compress_hook)
    # AdamW with weight_decay=0 is exactly Adam. Decay only weight matrices, not biases or norm gains.
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": par.optim.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=par.optim.lr, betas=(0.9, par.optim.adam_beta2))  # the lr is set at every step (Trainer.train)
    if ckpt is not None:  # the Adam moments and step count only: the fresh param groups keep their own lr
        ckpt["opt"]["param_groups"] = opt.state_dict()["param_groups"]
        opt.load_state_dict(ckpt["opt"])
    return model, lejepa, opt, ckpt["step"] if ckpt else 0

def train_step(model, opt, batches, par: Train, r: Rank, bench: Benchmark):
    """One optimizer step on the next batch, each part in its profiler phase (bench). A step whose gradient is non-finite is
    skipped, not applied: a single one would write NaN into every weight via opt.step() (the sudden divergences in
    e00/viewsizes*); DDP ranks share the all-reduced norm, so they skip together, at one host sync per step.
    (batch on the device: Batch C Z Y X; LejepaOutput; grad norm; whether skipped; RNG state before the forward, (CPU, CUDA):
    views draw on the CPU generator and SIGReg slices on the GPU's, so a skipped step replays exactly from its dump.)"""
    import torch
    device = r.device()
    if par.speed.cudagraphs:
        torch.compiler.cudagraph_mark_step_begin()  # previous step's graph outputs may be overwritten
    # No annotation hooks or profiler are active during the throughput baseline.
    with bench.phase("01_DATA_IO"):
        x = next(batches)
    with bench.phase("03_H2D_TRANSFER"):
        # Deferred batches are dicts of per-sample uint8 crops; finish_images copies and normalizes them.
        x = finish_images(x, device)["img"] if par.speed.defer_image_ops else x.to(device, non_blocking=True)
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
    with bench.phase("04_FORWARD_AND_LOSS"), torch.autocast(device.type, dtype=torch.bfloat16, enabled=par.speed.amp):
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
    return x, out, grad_norm, skipped, rng

def save_bad_batch(par: Train, savedir: str, lejepa, x, rng, step: int):
    """A skipped step, replayable (replay_bad_batch): the weights, still the pre-step ones, the input (uint8: lossless) and
    the RNG state before its forward -> savedir/bad_batch_<step>.pt."""
    import torch
    torch.save({"step": step, "params": asdict(par), "model_config": asdict(lejepa.config), "model": lejepa.state_dict(),
                "x_uint8": (x.clamp(0, 1) * 255).round().byte().cpu(), "rng_cpu": rng[0], "rng_cuda": rng[1]},
               Path(savedir) / f"bad_batch_{step:07d}.pt")

def log_metrics(f, par: Train, lejepa, x, out, grad_norm, lr: float, step: int, n_skipped: int, seconds: float):
    """One metrics.json row (and a progress line) for this step."""
    import torch
    loss = out.loss.detach().item()
    # Median pre-LayerNorm token norm on a global-view-sized corner of 4 samples: residual-stream
    # growth tends to precede divergence (see e00/viewsizes). Eager and grad-free, so cheap.
    g = par.data.view
    with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=par.speed.amp):
        resid = lejepa.encoder.forward_residual(x[:4, :, :g[0], :g[1], :g[2]])
    # LeJEPA's loss = inv + lamb * sigreg, logged apart: sigreg is ~1.05 for exactly Gaussian projections
    # (its floor), and its pull scales with the per-GPU batch, so the total alone mixes invariance
    # with how Gaussian the projections are. An MAE's loss is its reconstruction error alone.
    f.write(json.dumps({"tbl": "metrics", "idx_step": step, "time": seconds, "loss": loss,
                        **{k: out[k].detach().item() for k in ("inv", "sigreg") if k in out},
                        "grad_norm": grad_norm.item(), "lr": lr,
                        "resid_norm": resid.float().norm(dim=-1).median().item(), "skipped": n_skipped}) + "\n")
    f.flush()
    print(f"finished step {step + 1}, loss={loss:.4f}", flush=True)

def stop_reason(r: Rank, out, n_skipped: int, step: int) -> str:
    """Why every rank stops after this step ("" to go on), agreed by one all-reduce: a non-finite loss (divergence), or too many
    skipped (non-finite gradient) steps."""
    import torch
    import torch.distributed as dist
    b1 = not torch.isfinite(out.loss.detach()).item()
    b2 = n_skipped >= MAX_SKIPPED
    stop = torch.tensor([float(b1), float(b2)], device=r.device())
    if r.world > 1:
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
    reasons = [f"Non-finite loss by step {step + 1}; stopping (checkpoints keep the last finite weights)",
               f"{n_skipped} steps skipped for non-finite gradients by step {step + 1}; stopping"]
    return next((m for m, s in zip(reasons, stop.tolist()) if s), "")

def record_oom(par: Train, savedir: str, r: Rank, step: int, err: Exception):
    """An out-of-memory error as a result row in performance.json (the table shows which configs don't fit)."""
    import torch
    row = {"tbl": "oom", "time": time.time(), "params": asdict(par), "idx_step": step, "error": str(err).splitlines()[0][:300]}
    if r.device().type == "cuda":
        row |= {"gpu_name": torch.cuda.get_device_name(r.device()), "max_mem_gb": torch.cuda.max_memory_allocated(r.device()) / 1e9}
    with open(Path(savedir) / "performance.json", "a") as f:
        f.write(json.dumps(row) + "\n")

class Trainer:
    """A model, its optimizer and its data, trained a few steps at a time: `with Trainer(par, steps) as t:` then t.train(...) as
    often as you like, t.save(path) between. Every DDP rank makes one; leaving the block destroys the process group.
    Rank 0 writes metrics.json, and the throughput window and profile of Benchmark (at the first steps)."""

    def __init__(self, par: Train, savedir: str, steps: int, init: dict | None = None):
        """savedir: where it writes. steps: the most it will train in all (the loader's supply, the benchmark's windows).
        init: None fresh, or a loaded checkpoint (what save() wrote: torch.load of a .pt)."""
        self.par, self.savedir = par, savedir
        self.r = setup(par)
        self.batches = dataloader(par, savedir, steps)  # each rank its own, with its own miao seed (see setup)
        self.model, self.lejepa, self.opt, self.step = build(par, self.r, init)  # step: steps done, in all
        self.first, self.n_skipped, self.stopped, self.start_time, self.lr = self.step, 0, "", time.time(), par.optim.lr
        self.bench = Benchmark(par, savedir, steps, self.r.device(), self.r.world, self.r.rank0)

    def __enter__(self) -> Trainer:
        self.bench.__enter__()
        return self

    def __exit__(self, kind, err, tb):
        import torch
        import torch.distributed as dist
        self.bench.__exit__(kind, err, tb)
        if isinstance(err, torch.OutOfMemoryError) and self.r.rank0:
            record_oom(self.par, self.savedir, self.r, self.step, err)  # then the job fails as usual
        if self.r.world > 1:
            dist.destroy_process_group()

    def advance(self, lr: float):
        """One step at lr (times the warmup of a fresh model): train_step's results."""
        self.lr = lr * min(1.0, (self.step + 1) / LR_WARMUP)
        for g in self.opt.param_groups:
            g["lr"] = self.lr
        return train_step(self.model, self.opt, self.batches, self.par, self.r, self.bench)

    def train(self, steps: int, lr0: float, lr1: float):
        """steps more steps, the lr going linearly from lr0 to lr1. No validation, no checkpoints: those are the caller's. Stops
        early (self.stopped says why) on a non-finite loss or too many skipped steps; later calls then train nothing."""
        r, par = self.r, self.par
        with open(Path(self.savedir) / "metrics.json" if r.rank0 else os.devnull, "a") as f:
            for i in range(steps * (not self.stopped)):
                lr = lr0 + (lr1 - lr0) * i / steps
                self.bench.begin(self.step - self.first)
                x, out, grad_norm, skipped, rng = self.advance(lr)
                if skipped:
                    self.n_skipped += 1
                    print(f"Non-finite grad norm at step {self.step}; skipped ({self.n_skipped} so far)", flush=True)
                    if r.rank0 and self.n_skipped <= BAD_BATCHES_KEPT:
                        save_bad_batch(par, self.savedir, self.lejepa, x, rng, self.step)
                with self.bench.phase("07_LOGGING"):
                    if r.rank0 and (self.step % 10 == 0 or i + 1 == steps):
                        log_metrics(f, par, self.lejepa, x, out, grad_norm, self.lr, self.step, self.n_skipped, time.time() - self.start_time)
                self.bench.end(self.step - self.first)
                self.step += 1
                if self.step % 100 == 0 and (reason := stop_reason(r, out, self.n_skipped, self.step - 1)):
                    print(reason, flush=True)
                    self.stopped = reason
                    break

    def save(self, path: Path):
        """Rank 0: the weights and optimizer at this step -> path (written whole, then moved into place)."""
        import torch
        state = self.lejepa.state_dict()
        bad = [k for k, v in state.items() if v.is_floating_point() and not torch.isfinite(v).all()]
        assert not bad, f"not saving step {self.step}: {len(bad)} non-finite tensors, e.g. {bad[0]}"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"step": self.step, "params": asdict(self.par), "model_config": asdict(self.lejepa.config), "model": state,
                    "opt": self.opt.state_dict()}, tmp)
        tmp.replace(path)  # never leave a half-written checkpoint
        print(f"Saved {path}", flush=True)


def benchmark(par: Train, savedir: str):
    """Throughput and a profiler trace of par's training step, with no training run, in savedir:
    warmup_steps untimed, benchmark_steps timed (performance.json), then profile_steps traced (profile.json, profile.out,
    trace_summary.json); no metrics, checkpoints or evals. Same setup, data and step as Trainer.train. Every DDP rank."""
    steps = par.profile.warmup_steps + par.profile.benchmark_steps + 1 + par.profile.profile_steps  # Benchmark's windows, no more
    with Trainer(par, savedir, steps) as t:
        for i in range(steps):
            t.bench.begin(i)
            t.advance(par.optim.lr)
            t.bench.end(i)

def replay_bad_batch(dump: dict):
    """Replay a skipped (non-finite gradient) training step from its bad_batch_*.pt dump (save_bad_batch's dict, torch.load'ed).

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
    par = parse(Train, dump["params"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision(par.speed.f32mode)
    x = dump["x_uint8"].to(device).float() / 255  # Batch C Z Y X, exactly the training input

    def step(label: str, compiled: bool, backend: SDPBackend | None = None) -> tuple[float, int]:
        model = make_model(par)  # this commit's own experiment (experiment.replay_bad_batch asserts it), so its own architecture
        model.load_state_dict(dump["model"])  # the pre-step weights
        model.to(device).train()
        if compiled:
            compile_model(model, par)
            if par.speed.cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
        # Same view crops (CPU generator) and SIGReg slices (GPU generator) as the original step.
        torch.set_rng_state(dump["rng_cpu"])
        if device.type == "cuda" and dump["rng_cuda"] is not None:
            torch.cuda.set_rng_state(dump["rng_cuda"], device)
        with sdpa_kernel(backend) if backend else nullcontext():
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=par.speed.amp):
                out = model(x)
            assert isinstance(out, LejepaOutput)
            out.loss.backward()
        bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        # NaN flows backward from where it starts, so the params that stay finite locate the source.
        good = [n for n, p in model.named_parameters() if p.grad is not None and torch.isfinite(p.grad).all()]
        print(f"{label}: loss {out.loss.item():.6f}, {len(bad)} params with non-finite grads"
              + (f", e.g. {bad[:3]}; finite: {good}" if bad else ""), flush=True)
        return out.loss.item(), len(bad)

    print(f"Replaying step {dump['step']}", flush=True)
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

