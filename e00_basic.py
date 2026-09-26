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
import pandas
import plotly.express as px
import plotly.graph_objects as go

# Default (width, batch per GPU) per LSF GPU queue, for displace 128^3 / 96^3 / 64^3 views. Measured runs cited;
# "est." sizes extrapolate the measured ~1.7 GB/sample at width 768 and haven't been run.
QUEUE_ARCH = {
 "gpu_a100": (768, 32),     # a100/d9: 50% MFU, 55 of 80 GB
 "gpu_rtx6000": (768, 32),  # rtx6k-memory/d5: 46% MFU, 55 of 95 GB
 "gpu_h200": (768, 64),     # est. ~110 of 140 GB (b84 peaked at 142); width-defer/d1: 38% MFU at b84
 "gpu_h100": (768, 32),     # est., same 80 GB as A100
 "gpu_b300": (768, 128),    # est. ~220 of 288 GB; untested with displace views
}
 
# Code copied into .tmpcode/<sweep>/ at submission (runlsf); jobs import only from these. Add files here if
# the experiment starts depending on others. Dependencies (pyproject.toml / uv.lock) are not frozen.
SNAPSHOT_PATHS = ["e00_basic.py", "lib"]

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
    grad_compress: bool = False  # DDP bf16_compress_hook: all-reduce gradients in bf16 (half the bytes)
    queue: str = "gpu_h200"  # LSF queue
    n_gpus: int = 1  # DDP ranks on one node (launched via torchrun); batch_size and n_workers are per GPU

    # profiling params
    warmup_steps: int = 10
    benchmark_steps: int = 50
    profile_steps: int = 10  # Set to zero to disable trace collection.

def allparams():
    params = []
    # View-size study: 17 one-GPU H200 runs, 8 h each, on the wide hemibrain data. Baseline (input, global, local)
    # = (128, 96, 64); vary input, global and local one at a time, plus 4 jointly scaled configs.
    base = (128, 96, 64)
    runs = [base]
    runs += [(p, 96, 64) for p in [104, 160, 192, 256]]  # input patch: room for globals to move
    runs += [(128, g, 64) for g in [64, 80, 112, 128]]  # global view size
    runs += [(128, 96, l) for l in [32, 48, 80, 96]]  # local view size
    runs += [(96, 64, 32), (160, 128, 80), (192, 144, 96), (256, 192, 128)]  # all scaled together
    tokens = lambda g, l: 2 * (g // 8) ** 3 + 4 * (l // 8) ** 3  # per sample: 2 globals + 4 locals, 8^3 patches
    # Model FLOPs per sample, relative units: 6 x ~38M encoder params per token + 12 x depth x width x N attention.
    flops = lambda g, l: sum(k * (n * (6 * 38.1e6 + 12 * 12 * 512 * n)) for k, n in [(2, (g // 8) ** 3), (4, (l // 8) ** 3)])
    for i, (inp, g, l) in enumerate(runs):
        p = Params()
        p.savedir = f"outdir/e00/viewsizes/d{i}/"
        p.data = "hemibrain_wide"
        p.views = "displace"
        p.patch_size = (inp, inp, inp)
        p.global_size = (g, g, g)
        p.local_size = (l, l, l)
        # Hold tokens per step (so GPU memory, ~95 GB at the baseline) about constant: the baseline's
        # 84 x 5504 tokens. Multiple of 4, capped at 2x the baseline batch.
        p.batch_size = min(168, max(8, 4 * round(84 * tokens(*base[1:]) / tokens(g, l) / 4)))
        # Cosine horizon = expected steps in 8 h: the baseline's ~0.45 s/step scaled by FLOPs per step.
        # Data-loading-bound configs (large inputs, big batches) will be slower; max_hours stops them regardless.
        step_s = 0.45 * (p.batch_size * flops(g, l)) / (84 * flops(*base[1:]))
        p.steps_per_epoch = int(8 * 3600 / step_s)
        p.max_hours = 8.0
        p.cudagraphs = True
        p.batch_views = True
        p.n_workers = 11  # 12 cores per GPU
        p.n_gpus = 1
        p.queue = "gpu_h200"
        params.append(p)
    # pprint(params)

    return params

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

    if rank0:
        with open(savedir / "runs.json", 'a') as rfile, open(repo_root() / "_diffs/diffs.json", "a") as difflog:
            gp = git_provenance()
            rfile.write(json.dumps({k:gp[k] for k in ['commit_id', 'diff_hash']}) + "\n")
            difflog.write(json.dumps({gp['diff_hash']:gp['diff']}) + "\n")

    model = Lejepa(lejepa_config(par))
    if rank0: pprint(model)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    print(f"Rank {rank}/{world_size} is using torch device {device} .")
    model = model.to(device)
    assert par.compile or not (par.cudagraphs or par.compile_blocks), "cudagraphs / compile_blocks require compile"
    if par.compile:
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
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # Warmup then cosine decay over steps_per_epoch, by step count so every rank uses the same lr.
    # For max_hours runs set steps_per_epoch to the expected step count so the schedule completes.
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / LR_WARMUP) * (
        LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / par.steps_per_epoch)))))

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
                with bench.phase("04_FORWARD_AND_LOSS"), torch.autocast(device.type, dtype=torch.bfloat16, enabled=par.amp):
                    out = model(x)
                bench.count(out)
                with bench.phase("05_BACKWARD"):
                    out.loss.backward()
                with bench.phase("06_OPTIMIZER"):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # no host sync
                    opt.step()
                    opt.zero_grad()
                    sched.step()
                with bench.phase("07_LOGGING"):
                    if rank0 and (idx_step % 10 == 0 or idx_step + 1 == par.steps_per_epoch):
                        loss = out.loss.detach().item()
                        metrics_file.write(json.dumps({"tbl": "metrics", "idx_step": idx_step, "time": time.time() - start_time, "loss": loss,
                                                       "grad_norm": grad_norm.item(), "lr": sched.get_last_lr()[0]}) + "\n")
                        metrics_file.flush()
                        print(f"finished step {idx_step + 1}/{par.steps_per_epoch}, loss={loss:.4f}", flush=True)

                bench.end(idx_step)
                # Every 100 steps, one all-reduce so every rank stops at the same step:
                # on a non-finite loss (divergence) or past max_hours.
                if idx_step % 100 == 99:
                    b1 = not torch.isfinite(out.loss.detach()).item()
                    b2 = bool(par.max_hours) and time.time() - start_time > par.max_hours * 3600
                    stop = torch.tensor([float(b1), float(b2)], device=device)
                    if world_size > 1:
                        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                    if stop[0].item():
                        print(f"Non-finite loss by step {idx_step + 1}; stopping (checkpoints keep the last finite weights)", flush=True)
                        break
                    if stop[1].item():
                        print(f"Reached max_hours={par.max_hours} at step {idx_step + 1}", flush=True)
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
        pca(str(savedir))
    if world_size > 1:
        dist.destroy_process_group()


def pca(run: str):
    """PCA maps (pca_maps) of a run's latest checkpoint. Runs automatically at the end of training runs.

    `run` is an index into allparams() or any run's savedir. The model is rebuilt from the checkpoint's own
    LejepaConfig (strict state_dict load, so architecture drift fails loudly) and the eval settings from its
    saved Params, so old sweeps work without being in allparams().
    """
    import torch
    savedir = Path(allparams()[int(run)].savedir if run.isdigit() else run)
    # Newest numbered checkpoint; runs from before numbered checkpoints saved a single checkpoint.pt.
    paths = sorted((savedir / "checkpoints").glob("step_*.pt")) or [savedir / "checkpoint.pt"]
    assert paths[-1].is_file(), f"no checkpoint under {savedir}"
    ckpt = torch.load(paths[-1], map_location="cpu", weights_only=False)
    saved = ckpt["params"]
    unknown = saved.keys() - {f.name for f in fields(Params)}
    assert not unknown, f"{paths[-1]} has params no longer in Params: {sorted(unknown)}"
    par = Params(**saved)  # fields added since the run take their defaults
    # Checkpoints from before model_config was saved rebuild it from Params with the current lejepa_config.
    cfg = LejepaConfig(**ckpt["model_config"]) if "model_config" in ckpt else lejepa_config(par)
    model = Lejepa(cfg)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pca_maps(model.encoder.to(device), par, savedir, ckpt["step"])

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
    Image.fromarray(np.concatenate([r for row in rows for r in (row, gap)][:-1], axis=0)).save(savedir / "pca2.png")
    stats = {"tbl": "pca", "step": step, "effective_rank": erank, "explained_variance": var,
             "centered_effective_rank": erank_c, "centered_explained_variance": var_c,
             "between_tile_variance": between_tile,  # fraction of token variance explained by tile means
             "norm_median": med, "norm_p99": float(torch.quantile(norms.flatten(), 0.99)), "norm_max": float(norms.max()),
             "norm_outliers": int((norms > 2 * med).sum()), "n_tokens": norms.numel()}
    (savedir / "pca.json").write_text(json.dumps(stats) + "\n")
    print(f"{savedir}: step {step}, effective rank {erank:.1f}, top-3 variance {sum(var[:3]):.2f}, "
          f"between-tile variance {between_tile:.2f}, norm outliers (>2x median) {stats['norm_outliers']}/{norms.numel()}; wrote pca.png")

def run_pcasweep(sweepdir: str):
    cmd = f"""bsub -P miaai -q gpu_a100 -n 12 -gpu "num=1" -W 1:00 \
          -o {sweepdir}/pca_%J.log uv run python e00_basic.py pca_sweep {sweepdir}
          """
    subprocess.Popen(cmd, shell=True)
    print(f"Submitted pca_sweep {sweepdir} to LSF.")

def pca_sweep(sweepdir: str):
    """pca() for every run dir (d0, d1, ...) under sweepdir that has a checkpoint."""
    for d in sorted(Path(sweepdir).glob("d*"), key=lambda d: int(d.name[1:])):
        if (d / "checkpoints").is_dir() or (d / "checkpoint.pt").is_file():
            pca(str(d))

def runlsf(n:int):
    import subprocess
    par:Params = allparams()[n]
    wipedir(par.savedir)
    # Job name from the savedir, e.g. outdir/e00/nanhunt/d9/ -> e00-nanhunt-d9, so bjobs shows which run is which.
    RUN_NAME = "-".join(Path(par.savedir).parts[1:])
    # The job runs a per-sweep snapshot of this code, not the shared checkout, which may have moved on (another
    # sweep pushed) by the time the job starts. Python puts the script's dir first on sys.path, so `lib` comes
    # from the snapshot too; cwd stays the repo root, so outdir/ and data paths resolve as before.
    code = snapshot(SNAPSHOT_PATHS, Path(".tmpcode") / "-".join(Path(par.savedir).parts[1:-1]))
    minutes = int(par.max_hours * 60) + 30 if par.max_hours else 15  # 30 min slack for startup + final checkpoint
    CPUS_PER_GPU = 12  # 8 GPUs -> all 96 cores; training processes need cores beyond the data workers
    assert par.n_workers + 1 <= CPUS_PER_GPU, f"n_workers={par.n_workers} leaves no core for the training process"
    cmd = f""" bsub -J {RUN_NAME} \
        -W {minutes // 60}:{minutes % 60:02d} \
        -P miaai \
        -n {par.n_gpus * CPUS_PER_GPU} \
        -R "span[hosts=1]" \
        -gpu "num={par.n_gpus}:mode=exclusive_process" \
        -q {par.queue} \
        -o {par.savedir}/job_%J.log \
        uv run torchrun --standalone --nproc_per_node={par.n_gpus} {code}/e00_basic.py run {n}
        """
    subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"Submitted {RUN_NAME} (allparams()[{n}], code {code}) to LSF.")

def runmany():
    for i in range(len(allparams())):
        runlsf(i)

def runmany_sequential():
    for i in range(len(allparams())):
        run(i)

def check_runs(root: str = "outdir/e00"):
    """Flag run dirs a job wrote under the wrong name (e.g. the shared-checkout race), or that lack results.

    Conflicts: a row whose params.savedir isn't the dir it sits in, a job log whose run wrote elsewhere, or
    more than one runs.json row (two jobs wrote here). Several job logs alone are just resubmissions.
    Missing: every run dir needs at least a job log and metrics.json (pending or still-running jobs show up too).
    Different: within a sweep, a run lacking file names (top level, excluding IGNORED) that other runs have.
    """
    import re
    from collections import defaultdict
    flagged = 0
    run_dirs = sorted(p for p in Path(root).glob("**/d*/") if re.fullmatch(r"d\d+", p.name))
    from fnmatch import fnmatch
    IGNORED = ["job_*.log", ".DS_Store"]  # expected to differ between runs, or not ours
    names = {d: {f.name for f in d.iterdir() if not any(fnmatch(f.name, g) for g in IGNORED)} for d in run_dirs}
    sweep_names = defaultdict(set)
    for d in run_dirs:
        sweep_names[d.parent] |= names[d]
    for d in run_dirs:
        issues = []
        missing = sweep_names[d.parent] - names[d]
        if missing:
            issues.append(f"lacks {', '.join(sorted(missing))}. ")
        if not any(d.glob("job_*.log")):
            issues.append("no job log")
        if not (d / "metrics.json").is_file():
            issues.append("no metrics.json")
        perf = d / "performance.json"
        for line in (perf.read_text().splitlines() if perf.is_file() else []):
            saved = json.loads(line).get("params", {}).get("savedir")
            if saved and Path(saved) != d:
                issues.append(f"row for {saved}")
        runs = d / "runs.json"
        n = len(runs.read_text().splitlines()) if runs.is_file() else 0
        if n > 1:
            issues.append(f"{n} runs.json rows")
        for log in d.glob("job_*.log"):
            m = re.search(r"input \+ view slices to (\S+)/views", log.read_text(errors="ignore"))
            if m and Path(m.group(1)) != d:
                issues.append(f"{log.name} ran as {m.group(1)}")
        if issues:
            flagged += 1
            print(f"{d}: {'; '.join(sorted(set(issues)))}")
    print(f"{flagged} run dirs flagged under {root}")

def loadJsonTable(filename):
    rows = []
    for par in allparams():
        params = asdict(par)
        path = Path(par.savedir) / filename
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if "params" in record:
                    # Params added after a run was saved must still be at their default, which that run used.
                    saved = record["params"]
                    defaults = asdict(Params())
                    diffs = [f"  {k}: saved {saved[k]!r}, no longer a Params field" for k in saved if k not in params]
                    diffs += [f"  {k}: current {params[k]!r}, saved {saved[k]!r}"
                              for k in saved if k in params and not json_equal(params[k], saved[k])]
                    diffs += [f"  {k}: current {params[k]!r}, not saved (run used the default {defaults[k]!r})"
                              for k in params if k not in saved and not json_equal(params[k], defaults[k])]
                    assert not diffs, f"Parameter mismatch in {path}:\n" + "\n".join(diffs)
                row = {**params, **record}
                rows.append(row)
    res = pandas.DataFrame(rows)
    # pprint(res)
    return res

def short_runs(savedirs):
    """savedir relative to the sweep's common prefix, e.g. 'outdir/e00/x/d3/' -> 'd3'."""
    prefix = os.path.commonpath(list(savedirs))
    return savedirs.str[len(prefix):].str.strip("/")

def plot1():
    """Loss curves, one line per run (and per repeat of a run), faceted by n_gpus."""
    res = loadJsonTable("metrics.json")
    assert len(res), "no metrics.json rows for allparams(); run ./pull.sh?"
    # Repeats append to the same metrics.json; each restarts at idx_step 0.
    repeat = (res.idx_step == 0).groupby(res.savedir).cumsum() - 1
    res["run"] = short_runs(res.savedir) + repeat.map(lambda k: f".{k}" if k else "")
    res["sizes"] = res.patch_size.astype(str) + " " + res.global_size.astype(str) + " " + res.local_size.astype(str)
    px.line(res, x="idx_step", y="loss", color="run", line_dash="views", facet_col="n_gpus",
            hover_data=["sizes", "n_workers"], markers=True, log_y=True).show()

def plot2():
    """ktok/s per GPU: one bar per result row, bars grouped by n_gpus with gaps between groups, colored by width + defer_image_ops."""
    res = loadJsonTable("performance.json")
    res["ktok_s_per_gpu"] = res.tokens_per_second / res.n_gpus / 1e3
    assert len(res), "no performance.json rows for allparams(); run ./pull.sh?"
    # Bar label: short run name, plus a suffix for repeated rows in one run.
    repeat = res.groupby("savedir").cumcount()
    res["run"] = short_runs(res.savedir) + repeat.map(lambda k: f".{k}" if k else "")
    res["color"] = "width=" + res.width.astype(str) + ", defer=" + res.defer_image_ops.astype(str)
    res = res.sort_values(["n_gpus", "color", "run"]).reset_index(drop=True)
    # x positions: consecutive within a group, GROUP_GAP extra slots between groups.
    GROUP_GAP = 0.8
    group_idx = res.n_gpus.rank(method="dense").astype(int) - 1
    res["x"] = res.index + GROUP_GAP * group_idx
    fig = go.Figure()
    for color, r in res.groupby("color", sort=False):
        fig.add_bar(x=r.x, y=r.ktok_s_per_gpu, name=str(color), width=0.9)
    for g, r in res.groupby("n_gpus"):
        fig.add_annotation(x=r.x.mean(), y=-0.12, yref="paper", text=f"<b>{g} gpu</b>", showarrow=False)
    fig.update_xaxes(tickvals=res.x, ticktext=res.run)
    fig.update_layout(yaxis_title="ktok/s per GPU", legend_title="", margin=dict(b=80))
    fig.show()

def table():
    res = loadJsonTable("performance.json")
    trace = loadJsonTable("trace_summary.json")
    # Host ms per profiled step in each phase (see lib.util.trace_summary).
    phases = {"01_DATA_IO_ms": "io ms", "04_FORWARD_AND_LOSS_ms": "fwd ms", "05_BACKWARD_ms": "bwd ms", "06_OPTIMIZER_ms": "opt ms"}
    for k in ["gpu_busy", "step_ms", *phases]:
        res[k] = res.savedir.map(dict(zip(trace.savedir, trace[k]))) if k in trace else float("nan")
    cols = {
        "savedir": "run",
        "tbl": "result",  # "throughput", or "oom" for runs that ran out of GPU memory
        # "views": "views",
        # "patch_size": "input",
        # "global_size": "global",
        # "local_size": "local",
        # "compile": "compile",
        # "cudagraphs": "cudagraphs",
        # "width": "width",
        # "defer_image_ops": "defer",
        # "compile_blocks": "blocks",
        # "grad_compress": "compress",
        # "batch_views": "batch views",
        "n_gpus": "gpus",
        # "batch_size": "batch",
        # "n_workers": "workers",
        "gpu_busy": "gpu busy %",
        "samples_per_second": "samples/s",
        "tokens_per_second": "tok/s",
        "tflops_per_second": "TFLOP/s",
        "mfu": "mfu %",
        "max_mem_gb": "mem GB",
        "input_mvox_per_second": "Mvox/s",
        "step_ms": "prof step ms",
        **phases,
    }
    for k in cols:  # older runs predate some columns
        if k not in res:
            res[k] = float("nan")
    res = res[list(cols)].rename(columns=cols) # type: ignore
    res["gpu busy %"] *= 100
    res["mfu %"] *= 100
    res["TFLOP/s"] /= res["gpus"]
    res = res.rename(columns={"TFLOP/s": "TFLOP/s/gpu"})
    res["tok/s"] /= 1e3
    res.insert(list(res.columns).index("tok/s") + 1, "ktok/s/gpu", res["tok/s"] / res["gpus"])
    res = res.rename(columns={"tok/s": "ktok/s"}).round(1)
    print(res.to_string(index=False))
    return res

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
        call_entrypoint(sys.argv[1], *sys.argv[2:])
