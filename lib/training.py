"""Training a Train spec: the data loader, DDP setup, model and optimizer, the step, checkpoints (with a callback for each
milestone's eval), the stop rules, throughput benchmark and replay of a skipped step. Entrypoints and their bookkeeping
(code snapshot check, run records, evals) are experiment.py's."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from pprint import pprint

import numpy as np

import lmd_catalog as lmd
from miao import VolumeDataset, collate_deferred, finish_images
from miao.config import MiaoConfig

from lib.benchmark import Benchmark
from lib.data import train_volumes
from lib.losses import LejepaOutput
from lib.models import Lejepa, Mae
from lib.specs import Rank, Train, make_model, milestone_steps, model_config
from lib.util import *
from lib.views import ViewMaker

# Training (train): the lr schedule (a trunk's linear warmup steps; an anneal's cosine ends at LR_FLOOR x the checkpoint's lr),
# the max global grad norm, checkpoints (every CHECKPOINT_EVERY steps, ~15 min at 8 GPUs; the newest CHECKPOINT_KEEP kept), and
# steps whose gradient is non-finite (skipped, not applied: the run stops after MAX_SKIPPED; the first BAD_BATCHES_KEPT dumped).
LR_WARMUP, LR_FLOOR, GRAD_CLIP = 1000, 0.01, 1.0
CHECKPOINT_EVERY, CHECKPOINT_KEEP = 2000, 1
MAX_SKIPPED, BAD_BATCHES_KEPT = 100, 2


def collate_images(samples):
    import torch
    return torch.stack([s["img"] for s in samples])

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

def save_checkpoint(par: Train, lejepa, opt, sched, step: int, on_milestone):
    """checkpoints/step_<step>.pt, keeping the newest CHECKPOINT_KEEP; a milestone step's also kept in checkpoints/milestones/
    and handed to on_milestone(par, step) there and then (experiment.eval_milestone: it evaluates it, then deletes it). Never non-finite weights."""
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
        on_milestone(par, step)

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

def train(par: Train, on_milestone):
    """Train par on every DDP rank; ends with no process group left. Rank 0 writes metrics.json, checkpoints (calling
    on_milestone(par, step) at each milestone: save_checkpoint), and the throughput window and profile of Benchmark."""
    import torch
    import torch.distributed as dist
    start_time = time.time()
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
                    save_checkpoint(par, lejepa, opt, sched, step + 1, on_milestone)
    except torch.OutOfMemoryError as err:
        if r.rank0:
            record_oom(par, r, step, err)
        raise  # then fail the job as usual
    if r.rank0 and latest_step(par) < step + 1:  # not saved in the loop already (that would evaluate a milestone twice)
        save_checkpoint(par, lejepa, opt, sched, step + 1, on_milestone)
    if r.world > 1:
        dist.destroy_process_group()

def benchmark(par: Train):
    """Throughput and a profiler trace of par's training step, with no training run:
    warmup_steps untimed, benchmark_steps timed (performance.json), then profile_steps traced (profile.json, profile.out,
    trace_summary.json); no metrics, checkpoints or evals. Same setup, data and step as train(). Every DDP rank."""
    import torch.distributed as dist
    par = replace(par, steps_per_epoch=par.warmup_steps + par.benchmark_steps + 1 + par.profile_steps)  # Benchmark's windows, no more
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
    torch.set_float32_matmul_precision(par.f32mode)
    x = dump["x_uint8"].to(device).float() / 255  # Batch C Z Y X, exactly the training input

    def step(label: str, compiled: bool, backend: SDPBackend | None = None) -> tuple[float, int]:
        model = make_model(par)  # this commit's own experiment (experiment.replay_bad_batch asserts it), so its own architecture
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

