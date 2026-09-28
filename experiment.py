"""Experiment code (remote): LeJEPA 3D ViT training runs, benchmarks, inference (pca) and replays, and their LSF
sweep (allparams, savedirs under outdir/e00/). Depends on lib/, so it runs only from its experiment's commit.
Analysis of the results lives in analysis.py."""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
import os, sys
import json
import time
from pathlib import Path

import math
from itertools import product

# local

from lib.data import HEMIBRAIN_EB, TRAIN_BOXES
from lib.benchmark import Benchmark
from lib.losses import LejepaOutput
from lib.models import Lejepa, LejepaConfig
from lib.views import ViewMaker
from lib.util import *
from lib.types import *

# external 

import lmd_catalog as lmd
from miao.config import MiaoConfig
from miao import VolumeDataset, collate_deferred, finish_images

from rich import print as pprint
import numpy as np

# Default (width, batch per GPU) per LSF GPU queue, for displace 128^3 / 96^3 / 64^3 views. Measured runs cited;
# "est." sizes extrapolate the measured ~1.7 GB/sample at width 768 and haven't been run. MFU with the Linear
# patch embedding (patchembed-linear) where measured; older Conv3d-era numbers are marked (conv), 1.1-1.6x lower.
QUEUE_ARCH = {
 "gpu_a100": (768, 32),     # patchembed-linear/d3: 54.9% MFU, 262 ktok/s, 55 of 80 GB
 "gpu_rtx6000": (768, 32),  # rtx6k-memory/d5 (conv): 46% MFU, 55 of 95 GB
 "gpu_h200": (768, 64),     # est. ~110 of 140 GB; width 512 x 84: patchembed-linear/d2 37.7% MFU (768 x 84 conv: 38%)
 "gpu_h100": (768, 32),     # est., same 80 GB as A100
 "gpu_b300": (1024, 64),    # patchembed-linear/d1: 49.3% MFU, 1010 ktok/s, 144 of 288 GB (512 x 84: 29.1%)
}
 
# Code copied into .tmpcode/<sweep>/ at submission (runlsf); jobs import only from these. Add files here if
# the experiment starts depending on others. Dependencies (pyproject.toml / uv.lock) are not frozen.
SNAPSHOT_PATHS = ["experiment.py", "lib"]

@dataclass(slots=True)
class Params:
    savedir: str = "outdir/e00/main/basic/"
    data: TrainData = "hemibrain_eb"  # training volumes, see lib.data.TRAIN_BOXES
    patch_size: Tup3Int = (104, 104, 104)  # hemibrain EB is 8 nm isotropic
    batch_size: int = 84
    steps_per_epoch: int = 71  # warmup + benchmark + 1 profiler warmup + profile steps: stop right after profiling
    max_hours: float = 0.0  # >0: stop after this many hours (steps_per_epoch is then an upper bound); sets LSF walltime
    n_layers: int = 12
    width: int = 512  # encoder width; heads = width // 64
    views: Views = "basic"  # 'basic' (random scales) or 'displace' (fixed sizes below, locals inside globals)
    global_size: Tup3Int = (88, 88, 88)  # displace only
    local_size: Tup3Int = (56, 56, 56)  # displace only

    # optimizations
    f32mode: F32Mode = "high"
    n_workers: int = 16
    prefetch_factor: int = 2
    amp: bool = True  # bf16 autocast for forward + loss
    compile: bool = True  # torch.compile(dynamic=True) the encoder
    cudagraphs: bool = False  # instead compile static with CUDA graphs (mode="reduce-overhead"); needs compile + displace views
    compile_blocks: bool = False  # compile each transformer block separately so DDP can overlap all-reduce with backward
    defer_image_ops: bool = True  # workers ship uint8 crops; cast + normalize on the GPU (miao.finish_images)
    batch_views: bool = False  # one encoder call per group of same-shape views (2 per step with displace)
    weight_decay: float = 0.0  # AdamW decay on weight matrices (biases/norms excluded); 0 = the original plain Adam
    adam_beta2: float = 0.999  # Adam second-moment decay; 0.999 = torch default (all runs so far), mia-muvit uses 0.95
    grad_compress: bool = False  # DDP bf16_compress_hook: all-reduce gradients in bf16 (half the bytes)
    queue: str = "gpu_h200"  # LSF queue
    n_gpus: int = 1  # DDP ranks on one node (launched via torchrun); batch_size and n_workers are per GPU

    # profiling params
    warmup_steps: int = 10
    benchmark_steps: int = 50
    profile_steps: int = 10  # Set to zero to disable trace collection.

def on_queue(p: Params, queue: str) -> Params:
    """Put p on an LSF GPU queue with that queue's default architecture (QUEUE_ARCH)."""
    assert queue in QUEUE_ARCH, f"no default architecture for {queue!r}; add it to QUEUE_ARCH"
    p.queue = queue
    p.width, p.batch_size = QUEUE_ARCH[queue]
    return p

def allparams():
    params = []
    # First long B300 training runs, with cuDNN attention off (the NaN-gradient fix, e00/nanhunt_flash) and the
    # Linear patch embed. One full 8xB300 node each, 8 h. Same everything except width: is the 2x-wider model
    # worth ~half the tokens at equal wall-clock? Batch held at 64 per GPU for both (loss depends on batch via
    # SIGReg), so compare PCA maps / later evals, and loss at equal steps, not final loss.
    # AdamW beta2 0.95 + weight decay 0.05, the recipe nanhunt_flash validates.
    # steps_per_epoch (cosine horizon) from 1-GPU B300 speed at 96% 8-GPU scaling: ~0.20 s/step for width 512
    # (patchembed-linear/d0; may be data-loader bound at ~2900 samples/s/node), ~0.37 s/step for 1024 (d1).
    for i, (w, step_s) in enumerate([(512, 0.20), (1024, 0.37)]):
        p = on_queue(Params(), "gpu_b300")
        p.savedir = f"outdir/e00/b300-train8h/d{i}/"
        p.width = w
        p.batch_size = 64
        p.data = "hemibrain_wide"
        p.views = "displace"
        p.patch_size = (128, 128, 128)
        p.global_size = (96, 96, 96)
        p.local_size = (64, 64, 64)
        p.max_hours = 8.0
        p.steps_per_epoch = int(8 * 3600 / step_s)
        p.weight_decay = 0.05
        p.adam_beta2 = 0.95
        p.cudagraphs = True
        p.batch_views = True
        p.n_workers = 11  # 12 cores per GPU
        p.n_gpus = 8
        params.append(p)
    return params

def record(savedir, fn: str):
    """Append one row to savedir/runs.json: which experiment function ran (fn), code_provenance(), argv, LSF job.
    Repro for the dir's artifacts: that commit + argv. Also logged per host in outdir/_log/ (log_command)."""
    with open(Path(savedir) / "runs.json", "a") as f:
        f.write(json.dumps({"fn": fn, **code_provenance(), "argv": sys.argv, "lsf_job": os.environ.get("LSB_JOBID")}) + "\n")

def collate_images(samples):
    import torch
    return torch.stack([s["img"] for s in samples])

def lejepa_config(par: Params):
    return LejepaConfig(
        n_layers = par.n_layers,
        width = par.width,
        views = par.views,
        global_size = par.global_size,
        local_size = par.local_size,
        batch_views = par.batch_views,
        lamb = 0.1,
    )

def compile_model(model: Lejepa, par: Params):
    """torch.compile the encoder in place, as par says (no-op unless par.compile)."""
    import torch
    assert par.compile or not (par.cudagraphs or par.compile_blocks), "cudagraphs / compile_blocks require compile"
    if not par.compile:
        return
    torch._logging.set_logs(recompiles=True)  # pyright: ignore[reportPrivateImportUsage]  # recompiles show up in job_*.log
    if par.cudagraphs:
        # CUDA graphs replay whole kernel sequences, removing per-kernel CPU launch cost.
        # They need static shapes: displace has exactly 2 view shapes, basic has ~9.
        assert par.views == "displace", "cudagraphs needs views='displace' (few static shapes)"
        kwargs = dict(mode="reduce-overhead", dynamic=False)
    else:
        kwargs = dict(dynamic=True)  # Views change shape every step.
    # A whole-encoder compile releases every gradient at the end of one fused backward, so DDP can't
    # start all-reducing until backward is done. Per-block compile releases each block's grads as it finishes.
    for m in (model.encoder.blocks if par.compile_blocks else [model.encoder]):
        m.compile(**kwargs)

def save_view_pngs(par: Params, dataset, n_samples: int = 3):
    """Save what the model sees: for n_samples fresh samples, the input and each of its views, as
    center-z slices at native voxel size, left to right (input | globals | locals) -> savedir/views_{i}.png."""
    from PIL import Image
    cfg = lejepa_config(par)
    p = cfg.patch_size
    patch = (p, p, p) if isinstance(p, int) else p
    maker = ViewMaker(n_global=cfg.n_global, n_local=cfg.n_local, global_scale=cfg.global_scale, local_scale=cfg.local_scale,
                      flip=cfg.flip, patch_size=patch, views=cfg.views, global_size=cfg.global_size, local_size=cfg.local_size)
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

def dataloader(n:int):
    import torch
    par: Params = allparams()[n]
    # volumes = [x.to_miao() for x in lmd.all() if x.name == "exm-drosophila-flyliconn-matt-260601-60X-B4-2-045/crop-001"]
    # Training volumes (lib/data.py), images only. Boxes are x y z; output is z y x.
    # With several volumes miao samples each equally (size_weighting_exponent=0); the wide crops are ~equal size.
    volumes = [lmd.get(name).to_miao(bounding_box=box[::-1]) for name, box in TRAIN_BOXES[par.data].items()]
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
      prefetch_factor=par.prefetch_factor,
      pin_memory=torch.cuda.is_available(),
      multiprocessing_context="spawn",
    )
    batches = iter(loader)
    # pprint(volumes)
    # pprint(mcfg)
    # pprint(dl[0]['img'].shape)
    if int(os.environ.get("RANK", 0)) == 0:
        save_view_pngs(par, dl)
    return batches

def run(n:int):
    start_time = time.time()
    # LSF jobs must run a runlsf snapshot: the shared checkout's allparams()[n] may be another sweep by now.
    b1 = "LSB_JOBID" in os.environ
    b2 = ".tmpcode" not in Path(__file__).resolve().parts
    assert not (b1 and b2), f"LSF job running {__file__} from the shared checkout; submit via runlsf (code snapshot)"
    par : Params = allparams()[n]
    import torch
    import torch.distributed as dist
    torch.set_float32_matmul_precision(par.f32mode)
    # cuDNN's fused attention backward returns NaN grads on some trained weights where flash and mem-efficient
    # don't (replay_bad_batch on e00/nanhunt_beta95/d9): the "divergences" of e00/viewsizes* and nanhunt*.
    # Runs before 2026-09-28 used it (torch's default pick on H200/B300).
    torch.backends.cuda.enable_cudnn_sdp(False)

    # Set by torchrun. Unset (plain `python`, tests) means a single process.
    # Don't seed torch identically across ranks: DataLoader workers derive miao's numpy seed from it.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    assert world_size == par.n_gpus, f"launched {world_size} ranks but par.n_gpus={par.n_gpus}"
    rank0 = rank == 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")

    # each rank/process gets it's own dataloader with a distinct np.random generator?
    batches = dataloader(n)

    savedir = Path(par.savedir)

    if rank0:  # repro: this commit + this command (runlsf only submits committed code)
        record(savedir, "run")

    model = Lejepa(lejepa_config(par))
    if rank0: pprint(model)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    print(f"Rank {rank}/{world_size} is using torch device {device} .")
    model = model.to(device)
    compile_model(model, par)
    lejepa = model  # unwrapped, for checkpoints
    if world_size > 1:
        # SIGReg and projector BatchNorm statistics stay per-rank (local batch) for now.
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)
        if par.grad_compress:
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            model.register_comm_hook(None, default_hooks.bf16_compress_hook)
    LR, LR_WARMUP, LR_FLOOR = 1e-4, 1000, 0.01  # peak lr; linear warmup steps; cosine ends at LR_FLOOR * LR
    GRAD_CLIP = 1.0  # max global grad norm
    CHECKPOINT_EVERY, CHECKPOINT_KEEP = 2000, 1  # steps (~15 min at 8 GPUs); newest checkpoints kept
    # AdamW with weight_decay=0 is exactly Adam. Decay only weight matrices, not biases or norm gains.
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": par.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=LR, betas=(0.9, par.adam_beta2))
    # Warmup then cosine decay over steps_per_epoch, by step count so every rank uses the same lr.
    # For max_hours runs set steps_per_epoch to the expected step count so the schedule completes.
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / LR_WARMUP) * (
        LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / par.steps_per_epoch)))))
    # Steps whose gradient is non-finite are skipped, not applied (see the optimizer phase below).
    MAX_SKIPPED, BAD_BATCHES_KEPT = 100, 2  # stop after this many skipped steps; replayable dumps written
    n_skipped = 0

    def save_checkpoint(step):
        if not rank0:
            return
        state = lejepa.state_dict()
        bad = [k for k, v in state.items() if v.is_floating_point() and not torch.isfinite(v).all()]
        if bad:  # never let a diverged model overwrite good checkpoints
            print(f"Not saving step {step}: {len(bad)} non-finite tensors, e.g. {bad[0]}", flush=True)
            return
        ckdir = savedir / "checkpoints"
        ckdir.mkdir(exist_ok=True)
        tmp, path = ckdir / "tmp.pt", ckdir / f"step_{step:07d}.pt"
        torch.save({"step": step, "params": asdict(par), "model_config": lejepa.cfg.to_kwargs(), "model": state,
                    "opt": opt.state_dict(), "sched": sched.state_dict()}, tmp)
        tmp.replace(path)  # never leave a half-written checkpoint
        for old in sorted(ckdir.glob("step_*.pt"))[:-CHECKPOINT_KEEP]:
            old.unlink()
        print(f"Saved {path}", flush=True)

    idx_step = -1
    try:
        # Only rank 0 writes results and profiles.
        with open(savedir / "metrics.json" if rank0 else os.devnull, "a") as metrics_file, Benchmark(par, device, world_size, rank0) as bench:
            for idx_step in range(par.steps_per_epoch):
                if par.cudagraphs:
                    torch.compiler.cudagraph_mark_step_begin()  # previous step's graph outputs may be overwritten
                bench.begin(idx_step)
                # No annotation hooks or profiler are active during the throughput baseline.
                with bench.phase("01_DATA_IO"):
                    x = next(batches)
                with bench.phase("03_H2D_TRANSFER"):
                    # Deferred batches are dicts of per-sample uint8 crops; finish_images copies and normalizes them.
                    x = finish_images(x, device)["img"] if par.defer_image_ops else x.to(device, non_blocking=True)
                # RNG state before the forward (views draw on the CPU generator, SIGReg slices on the GPU's),
                # so a skipped step's forward/backward can be replayed exactly from its bad_batch dump.
                rng_cpu = torch.get_rng_state()
                rng_cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                with bench.phase("04_FORWARD_AND_LOSS"), torch.autocast(device.type, dtype=torch.bfloat16, enabled=par.amp):
                    out = model(x)
                bench.count(out)
                with bench.phase("05_BACKWARD"):
                    out.loss.backward()
                with bench.phase("06_OPTIMIZER"):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    # A single non-finite gradient would write NaN into every weight via opt.step() (the sudden
                    # divergences in e00/viewsizes*). Skip such steps instead. DDP ranks share the all-reduced norm,
                    # so they all skip together. Costs one host sync per step.
                    if torch.isfinite(grad_norm).item():
                        opt.step()
                    else:
                        n_skipped += 1
                        print(f"Non-finite grad norm at step {idx_step}; skipped ({n_skipped} so far)", flush=True)
                        if rank0 and n_skipped <= BAD_BATCHES_KEPT:
                            # Weights are still the pre-step ones; inputs are uint8 on disk, so this is lossless.
                            torch.save({"step": idx_step, "params": asdict(par), "model_config": lejepa.cfg.to_kwargs(),
                                        "model": lejepa.state_dict(), "x_uint8": (x.clamp(0, 1) * 255).round().byte().cpu(),
                                        "rng_cpu": rng_cpu, "rng_cuda": rng_cuda}, savedir / f"bad_batch_{idx_step:07d}.pt")
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                with bench.phase("07_LOGGING"):
                    if rank0 and (idx_step % 10 == 0 or idx_step + 1 == par.steps_per_epoch):
                        loss = out.loss.detach().item()
                        # Median pre-LayerNorm token norm on a global-view-sized corner of 4 samples: residual-stream
                        # growth tends to precede divergence (see e00/viewsizes). Eager and grad-free, so cheap.
                        g = par.global_size
                        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=par.amp):
                            resid = lejepa.encoder.forward_residual(x[:4, :, :g[0], :g[1], :g[2]])
                        metrics_file.write(json.dumps({"tbl": "metrics", "idx_step": idx_step, "time": time.time() - start_time, "loss": loss,
                                                       "grad_norm": grad_norm.item(), "lr": sched.get_last_lr()[0],
                                                       "resid_norm": resid.float().norm(dim=-1).median().item(),
                                                       "skipped": n_skipped}) + "\n")
                        metrics_file.flush()
                        print(f"finished step {idx_step + 1}/{par.steps_per_epoch}, loss={loss:.4f}", flush=True)

                bench.end(idx_step)
                # Every 100 steps, one all-reduce so every rank stops at the same step: on a non-finite loss
                # (divergence), past max_hours, or after too many skipped (non-finite gradient) steps.
                if idx_step % 100 == 99:
                    b1 = not torch.isfinite(out.loss.detach()).item()
                    b2 = bool(par.max_hours) and time.time() - start_time > par.max_hours * 3600
                    b3 = n_skipped >= MAX_SKIPPED
                    stop = torch.tensor([float(b1), float(b2), float(b3)], device=device)
                    if world_size > 1:
                        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                    if stop[0].item():
                        print(f"Non-finite loss by step {idx_step + 1}; stopping (checkpoints keep the last finite weights)", flush=True)
                        break
                    if stop[1].item():
                        print(f"Reached max_hours={par.max_hours} at step {idx_step + 1}", flush=True)
                        break
                    if stop[2].item():
                        print(f"{n_skipped} steps skipped for non-finite gradients by step {idx_step + 1}; stopping", flush=True)
                        break
                if idx_step % CHECKPOINT_EVERY == CHECKPOINT_EVERY - 1:
                    save_checkpoint(idx_step + 1)
    except torch.OutOfMemoryError as err:
        # Record the OOM as a result (the table shows which configs don't fit), then fail the job as usual.
        if rank0:
            row = {"tbl": "oom", "time": time.time(), "params": asdict(par), "idx_step": idx_step,
                   "error": str(err).splitlines()[0][:300]}
            if device.type == "cuda":
                row |= {"gpu_name": torch.cuda.get_device_name(device), "max_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9}
            with open(savedir / "performance.json", "a") as f:
                f.write(json.dumps(row) + "\n")
        raise
    save_checkpoint(idx_step + 1)
    if rank0 and par.max_hours:  # training runs (not benchmarks): PCA maps of the last finite checkpoint
        pca(n)
    if world_size > 1:
        dist.destroy_process_group()


def replay_bad_batch(path: str):
    """Replay a skipped (non-finite gradient) training step from its bad_batch_*.pt dump.

    1. Eager under torch.autograd.detect_anomaly: raises at the first backward op that produced NaN/inf, with
       the traceback of the forward op that created it. Eager kernels can differ numerically from compiled
       ones, so 1 may not reproduce what 3 does.
    2. Eager, with SDPA forced to each fused attention backend in turn (cuDNN, flash, mem-efficient). If only
       cuDNN gives non-finite grads, its attention backward is the culprit. Eager because compiled replays
       ignored sdpa_kernel (identical losses for every backend). The math backend is left out: it materializes
       Batch Head N N attention and doesn't fit at training batch sizes. CUDA only (fused kernels).
    3. As trained (compiled per the run's Params): does the gradient come out non-finite again? Runs last:
       its CUDA graph pools stay allocated. Uses torch's default SDPA pick (cuDNN), as runs before run()
       disabled cuDNN attention did; 1 does too.
    Returns loss and non-finite param count per replay; also prints which params kept finite grads.
    """
    import torch
    from contextlib import nullcontext
    from torch.nn.attention import SDPBackend, sdpa_kernel
    # Like training, replay depends on lib/: only for dumps of this commit's own experiment (replaylsf submits it).
    ours = {Path(p.savedir).resolve() for p in allparams()}
    assert Path(path).resolve().parent in ours, f"{path} isn't from this commit's allparams(): replay it from its experiment's commit"
    record(Path(path).parent, "replay_bad_batch")
    dump = torch.load(path, map_location="cpu", weights_only=False)
    par = Params(**dump["params"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision(par.f32mode)
    x = dump["x_uint8"].to(device).float() / 255  # Batch C Z Y X, exactly the training input

    def step(label: str, compiled: bool, backend: SDPBackend | None = None) -> tuple[float, int]:
        model = Lejepa(LejepaConfig(**dump["model_config"]))
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

def pca(n:int):
    """PCA maps (pca_maps) of allparams()[n]'s latest checkpoint. Runs at the end of training; pcalsf(n) redoes it.

    Inference depends on lib/, so like training it only runs for this commit's own experiment: for an older
    sweep, go back to its commit (jj new <commit>). The checkpoint's saved params must match allparams()[n].
    """
    import torch
    par = allparams()[n]
    paths = sorted((Path(par.savedir) / "checkpoints").glob("step_*.pt"))
    assert paths, f"no checkpoints in {par.savedir}"
    ckpt = torch.load(paths[-1], map_location="cpu", weights_only=False)
    assert json_equal(ckpt["params"], asdict(par)), f"{paths[-1]} was made by other params: run pca from its experiment's commit"
    model = Lejepa(lejepa_config(par))
    model.load_state_dict(ckpt["model"])  # strict: architecture drift fails loudly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    record(par.savedir, "pca")  # argv says whether it ran inside `run n` or on its own (pcalsf)
    pca_maps(model.encoder.to(device), par, Path(par.savedir), ckpt["step"])

def pca_maps(encoder, par: Params, savedir: Path, step: int):
    """PCA maps of patch-token embeddings on a held-out EB val crop -> savedir/pca.png, pca.json.

    pca.png columns: EM | PCA of tokens | PCA after subtracting each tile's mean token | pre-LayerNorm token L2 norm.
    """
    import torch
    from PIL import Image
    device = next(encoder.parameters()).device
    encoder.eval()

    # One (g, 4g, 4g) window (z y x, g = global view size, so it tiles exactly) inside EB's val slab
    # (EB z 3000-4000), held out from training.
    # miao needs the box strictly larger than the patch: one extra voxel per axis leaves exactly one window.
    shape, z0, y0, x0 = (par.global_size[0], 4 * par.global_size[1], 4 * par.global_size[2]), 3400, 2000, 2000
    vol = lmd.get(HEMIBRAIN_EB).to_miao(bounding_box=[[o, o + s + 1] for o, s in zip((z0, y0, x0), shape)])
    mcfg = MiaoConfig(volumes=[vol], patch_size=list(shape), resolutions=[[8.0, 8.0, 8.0]],
                      samples_per_epoch=1, sampling="random", output_axes="lzyx")
    img = VolumeDataset(mcfg)[0]["img"]  # 1 Z Y X in [0, 1]

    # Tokens from non-overlapping tiles at the training global-view size, stitched into one grid.
    tile, patch = par.global_size, encoder.patch_embed.patch_size
    assert all(s % t == 0 for s, t in zip(shape, tile)), f"crop {shape} must tile by {tile}"
    grid = [s // p for s, p in zip(shape, patch)]
    feats = torch.empty(*grid, encoder.embed_dim)  # Z Y X D, tokens (after the final LayerNorm)
    norms = torch.empty(*grid)  # Z Y X, token norms before the final LayerNorm
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for z, y, x in product(*(range(0, s, t) for s, t in zip(shape, tile))):
            r = encoder.forward_residual(img[None, :, z:z + tile[0], y:y + tile[1], x:x + tile[2]].to(device))
            t = encoder.norm(r)
            tz, ty, tx = (o // p for o, p in zip((z, y, x), patch))
            gz, gy, gx = (t_ // p for t_, p in zip(tile, patch))
            feats[tz:tz + gz, ty:ty + gy, tx:tx + gx] = t[0].float().reshape(gz, gy, gx, -1).cpu()
            norms[tz:tz + gz, ty:ty + gy, tx:tx + gx] = r[0].float().norm(dim=-1).reshape(gz, gy, gx).cpu()

    def pca_rgb(f):  # N D tokens -> (Z Y X 3 uint8 top-3 PCs, effective rank, top-10 explained variance)
        f = f - f.mean(0)
        _, sv, vh = torch.linalg.svd(f, full_matrices=False)
        p = sv / sv.sum()
        erank = float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))  # effective rank; ~1 means collapse
        pcs = (f @ vh[:3].T).reshape(*grid, 3)
        lo, hi = torch.quantile(pcs.reshape(-1, 3), torch.tensor([0.01, 0.99]), dim=0)
        rgb = ((pcs - lo) / (hi - lo)).clamp(0, 1).mul(255).byte().numpy()
        return rgb, erank, (sv ** 2 / (sv ** 2).sum())[:10].tolist()

    # Tile-centered tokens: subtract each tile's mean token, removing the per-view code and keeping within-view structure.
    g = [t_ // p for t_, p in zip(tile, patch)]  # tokens per tile side
    nt = [n // gi for n, gi in zip(grid, g)]  # tiles per axis
    tiled = feats.reshape(nt[0], g[0], nt[1], g[1], nt[2], g[2], encoder.embed_dim)
    centered = (tiled - tiled.mean(dim=(1, 3, 5), keepdim=True)).reshape(*grid, encoder.embed_dim)
    rgb, erank, var = pca_rgb(feats.reshape(-1, encoder.embed_dim))
    rgb_c, erank_c, var_c = pca_rgb(centered.reshape(-1, encoder.embed_dim))
    between_tile = 1 - float(centered.var(dim=(0, 1, 2)).sum() / feats.var(dim=(0, 1, 2)).sum())

    # Pre-LayerNorm token norms: "register"-like tokens that store global information show up as sparse
    # high-norm outliers. (After the final LayerNorm every token's norm is ~sqrt(width), so those can't show this.)
    med = float(norms.median())
    lo, hi = torch.quantile(norms.flatten(), torch.tensor([0.01, 0.999]))
    norm_u8 = ((norms - lo) / (hi - lo)).clamp(0, 1).mul(255).byte().numpy()

    # Rows: 3 z-slices; columns: EM | PCA | tile-centered PCA | token norm, upsampled to voxels.
    up = lambda a: a.repeat(patch[1], axis=0).repeat(patch[2], axis=1)
    vgap = np.full((shape[1], 8, 3), 255, np.uint8)
    rows = []
    for gz in [grid[0] // 6, grid[0] // 2, grid[0] * 5 // 6]:
        zv = gz * patch[0] + patch[0] // 2
        em = np.repeat((img[0, zv].numpy() * 255).astype(np.uint8)[..., None], 3, axis=2)
        nm = np.repeat(norm_u8[gz][..., None], 3, axis=2)
        rows.append(np.concatenate([em, vgap, up(rgb[gz]), vgap, up(rgb_c[gz]), vgap, up(nm)], axis=1))
    gap = np.full((8, rows[0].shape[1], 3), 255, np.uint8)
    Image.fromarray(np.concatenate([r for row in rows for r in (row, gap)][:-1], axis=0)).save(savedir / "pca.png")
    stats = {"tbl": "pca", "step": step, "effective_rank": erank, "explained_variance": var,
             "centered_effective_rank": erank_c, "centered_explained_variance": var_c,
             "between_tile_variance": between_tile,  # fraction of token variance explained by tile means
             "norm_median": med, "norm_p99": float(torch.quantile(norms.flatten(), 0.99)), "norm_max": float(norms.max()),
             "norm_outliers": int((norms > 2 * med).sum()), "n_tokens": norms.numel()}
    (savedir / "pca.json").write_text(json.dumps(stats) + "\n")
    print(f"{savedir}: step {step}, effective rank {erank:.1f}, top-3 variance {sum(var[:3]):.2f}, "
          f"between-tile variance {between_tile:.2f}, norm outliers (>2x median) {stats['norm_outliers']}/{norms.numel()}; wrote pca.png")

def bsub(par: Params, job: str, minutes: int, n_gpus: int, cmd: str):
    """Submit `uv run <cmd>` for par's run dir to LSF as job <sweep>-<dN>-<job>, logging to savedir/job_<job>_%J.log.

    {code} in cmd is this commit's code snapshot (.tmpcode/<sweep>/), which the job runs instead of the shared
    checkout: that may have moved on (another sweep pushed) by the time the job starts. Python puts the script's
    dir first on sys.path, so `lib` comes from the snapshot too; cwd stays the repo root, so outdir/ and data paths
    resolve as before. Callers assert_committed() first.
    """
    import subprocess
    name = "-".join(Path(par.savedir).parts[1:]) + f"-{job}"  # e.g. e00-nanhunt-d9-run, so bjobs shows which is which
    code = snapshot(SNAPSHOT_PATHS, Path(".tmpcode") / "-".join(Path(par.savedir).parts[1:-1]))
    CPUS_PER_GPU = 12  # 8 GPUs -> all 96 cores; training processes need cores beyond the data workers
    assert par.n_workers + 1 <= CPUS_PER_GPU, f"n_workers={par.n_workers} leaves no core for the training process"
    full = f""" bsub -J {name} \
        -W {minutes // 60}:{minutes % 60:02d} \
        -P miaai \
        -n {n_gpus * CPUS_PER_GPU} \
        -R "span[hosts=1]" \
        -gpu "num={n_gpus}:mode=exclusive_process" \
        -q {par.queue} \
        -o {par.savedir}/job_{job}_%J.log \
        uv run {cmd.format(code=code)}
        """
    subprocess.Popen(full, shell=True, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"Submitted {name} (code {code}) to LSF.")

def runlsf(n:int):
    """Train allparams()[n] on LSF, in a fresh savedir (old contents -> outdir/.trash/)."""
    par:Params = allparams()[n]
    assert_committed()
    trash(par.savedir)
    minutes = int(par.max_hours * 60) + 30 if par.max_hours else 15  # 30 min slack for startup + final checkpoint
    bsub(par, "run", minutes, par.n_gpus, f"torchrun --standalone --nproc_per_node={par.n_gpus} {{code}}/experiment.py run {n}")

def pcalsf(n:int):
    """Redo pca(n) on LSF, e.g. after changing pca_maps in this experiment's change (run() already does it once)."""
    assert_committed()
    bsub(allparams()[n], "pca", 30, 1, f"python {{code}}/experiment.py pca {n}")

def pcamany():
    for i in range(len(allparams())):
        pcalsf(i)

def replaylsf(n:int):
    """replay_bad_batch on LSF for allparams()[n]'s first bad_batch dump. CUDA_LAUNCH_BLOCKING names a crashing kernel."""
    par = allparams()[n]
    dumps = sorted(Path(par.savedir).glob("bad_batch_*.pt"))
    assert dumps, f"no bad_batch_*.pt in {par.savedir}"
    assert_committed()
    bsub(par, "replay", 30, 1, f"env CUDA_LAUNCH_BLOCKING=1 python {{code}}/experiment.py replay_bad_batch {dumps[0]}")

def runmany():
    for i in range(len(allparams())):
        runlsf(i)

def runmany_sequential():
    for i in range(len(allparams())):
        run(i)

def test():
    x = lmd.all()
    for xi in x:
        if xi.name.startswith("em-"):
            pprint(xi)

def size():
  import zarr
  from typing import Any, cast
  v = lmd.get("exm-drosophila-flyliconn-matt-260601-60X-B4-2-045/crop-001")
  g = zarr.open_group(v.path, mode="r")[v.image_key]
  assert isinstance(g, zarr.Group)
  meta = cast(dict[str, Any], g.attrs.get("ome", g.attrs))
  a = g[meta["multiscales"][0]["datasets"][0]["path"]]
  assert isinstance(a, zarr.Array)
  print(a.size)
  print(a.shape)

if __name__ == "__main__":
    import sys
    print(sys.argv)
    if len(sys.argv) == 1:
        pick_entrypoint()
    else:
        log_command(sys.argv)  # experiment code: every CLI call -> outdir/_log/commands-<host>.jsonl
        call_entrypoint(sys.argv[1], *sys.argv[2:])
