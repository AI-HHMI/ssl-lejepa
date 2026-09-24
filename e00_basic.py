from __future__ import annotations

from dataclasses import dataclass, asdict
import os, sys
import json
import time
from pathlib import Path
from contextlib import ExitStack, nullcontext

from math import prod
from itertools import product

# local

from lib.data import HEMIBRAIN_EB, TRAIN_BOXES
from lib.models import Lejepa, LejepaConfig
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

# Dense bf16 tensor-core peak per GPU, matched by substring of torch.cuda.get_device_name().
# B300/B200 from NVIDIA's HGX spec (36 PFLOPS sparse per 8 GPUs); check the datasheet if it matters.
PEAK_BF16_TFLOPS = {"H100": 989, "H200": 989, "B200": 2250, "B300": 2250}


@dataclass(slots=True)
class Params:
    savedir: str = "outdir/e00/main/basic/"
    # patch_size: list[int] = field(default_factory=lambda: [104, 232, 232])
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
    n_gpus: int = 1  # DDP ranks on one node (launched via torchrun); batch_size and n_workers are per GPU

    # profiling params
    warmup_steps: int = 10
    benchmark_steps: int = 50
    profile_steps: int = 10  # Set to zero to disable trace collection.

def allparams():
    params = []
    # Training runs: best throughput config on a full 8xH200 node for 6 h (~7.8M tok/s, ~45k steps each).
    # d0 trains on the EB train split (~850x coverage per voxel); d1 adds hemibrain crops 002 + 003 (~30x).
    for i, data in enumerate(["hemibrain_eb", "hemibrain_wide"]):
        p = Params()
        p.savedir = f"outdir/e00/train6h_hemi/d{i}/"
        p.data = data
        p.views = "displace"
        p.patch_size = (128, 128, 128)
        p.global_size = (96, 96, 96)
        p.local_size = (64, 64, 64)
        p.cudagraphs = True
        p.batch_views = True
        p.n_workers = 8
        p.n_gpus = 8
        p.max_hours = 6.0
        p.steps_per_epoch = 1_000_000  # upper bound; max_hours stops it
        params.append(p)
    # pprint(params)
    return params

def write_trace_summary(savedir):
    summary = trace_summary(Path(savedir) / "profile.json")
    (Path(savedir) / "trace_summary.json").write_text(json.dumps({"tbl": "trace_summary", **summary}) + "\n")
    print(f"{savedir}: GPU busy during profiled steps {100 * summary['gpu_busy']:.0f}%")

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

def run(n:int):
    start_time = time.time()
    par : Params = allparams()[n]
    if min(par.warmup_steps, par.benchmark_steps, par.profile_steps) < 0:
        raise ValueError("Profiling and benchmark step counts must be nonnegative")
    # lmd.set_data_root("/Volumes/miaai/lmd-v0.0.1/data")
    # volumes = [x.to_miao() for x in lmd.all() if "flyliconn" in x.name]

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
        torch._logging.set_logs(recompiles=True)  # recompiles show up in job_*.log
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
    opt = torch.optim.Adam(model.parameters(), lr = 1e-4)
    CHECKPOINT_EVERY = 2000  # steps; ~15 min at 8 GPUs

    def save_checkpoint(step):
        if not rank0:
            return
        tmp, path = savedir / "checkpoint.pt.tmp", savedir / "checkpoint.pt"
        torch.save({"step": step, "params": asdict(par), "model": lejepa.state_dict(), "opt": opt.state_dict()}, tmp)
        tmp.replace(path)  # never leave a half-written checkpoint.pt
        print(f"Saved {path} at step {step}", flush=True)
    use_cuda = device.type == "cuda"
    activities = [torch.profiler.ProfilerActivity.CPU]
    if use_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profile_start = par.warmup_steps + par.benchmark_steps
    profile_stop = min(profile_start + 1 + par.profile_steps, par.steps_per_epoch)
    benchmark_stop = min(profile_start, par.steps_per_epoch)

    def synchronize():
        if use_cuda:
            torch.cuda.synchronize(device)

    def save_trace(prof):
        prof.export_chrome_trace(str(savedir / "profile.json"))
        write_trace_summary(savedir)
        averages = prof.key_averages()
        report = (
            f"Device: {device}; recorded steps (zero-based): "
            f"{profile_start + 1}..{profile_stop - 1}\n"
            "Open profile.json in https://ui.perfetto.dev to inspect CPU and CUDA tracks.\n"
            "01_DATA_IO includes storage/network waits, decoding, and preprocessing;\n"
            "02_CPU_BATCH is host tensor assembly. Neither is a pure disk counter.\n"
            "03_H2D_TRANSFER is host-side transfer time; inspect Memcpy HtoD on CUDA tracks.\n"
            "GPU gaps aligned with loading suggest input starvation; gaps amid CPU dispatch\n"
            "suggest host overhead. Busy GPU tracks alone do not prove compute saturation.\n"
            "CPU and device times overlap: do not add them into bottleneck percentages.\n\n"
            "OPERATORS SORTED BY SELF CPU TIME\n"
            + averages.table(sort_by="self_cpu_time_total", row_limit=50)
        )
        if use_cuda:
            report += "\n\nOPERATORS SORTED BY SELF DEVICE TIME\n" + averages.table(
                sort_by="self_device_time_total", row_limit=50,
            )
        (savedir / "profile.out").write_text(report)
        print(f"Saved {savedir / 'profile.json'} and {savedir / 'profile.out'}")

    prof = None
    benchmark_started = None
    n_tokens = 0  # encoder tokens since benchmark start
    n_flops = 0  # model training FLOPs since benchmark start (see Lejepa.forward)
    # Only rank 0 writes results and profiles.
    with open(savedir / "metrics.json" if rank0 else os.devnull, "a") as metrics_file, ExitStack() as profile_scope:
        for idx_step in range(par.steps_per_epoch):
            if par.cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()  # previous step's graph outputs may be overwritten
            if idx_step == par.warmup_steps and idx_step < benchmark_stop:
                synchronize()
                benchmark_started = time.perf_counter()
            if rank0 and par.profile_steps and idx_step == profile_start and idx_step + 1 < profile_stop:
                synchronize()
                prof = profile_scope.enter_context(torch.profiler.profile(
                    activities=activities,
                    schedule=torch.profiler.schedule(wait=0, warmup=1, active=par.profile_steps, repeat=1),
                    on_trace_ready=save_trace,
                    record_shapes=False, profile_memory=False, with_stack=False,
                ))
                prof.add_metadata_json("run_config", json.dumps(asdict(par)))

            # No annotation hooks or profiler are active during the throughput baseline.
            phase = torch.profiler.record_function if prof is not None else lambda name: nullcontext()
            with phase("01_DATA_IO"):
                x = next(batches)
            with phase("03_H2D_TRANSFER"):
                # Deferred batches are dicts of per-sample uint8 crops; finish_images copies and normalizes them.
                x = finish_images(x, device)["img"] if par.defer_image_ops else x.to(device, non_blocking=True)
            with phase("04_FORWARD_AND_LOSS"), torch.autocast(device.type, dtype=torch.bfloat16, enabled=par.amp):
                out = model(x)
            if benchmark_started is not None:
                n_tokens += out.n_tokens
                n_flops += out.n_flops
            with phase("05_BACKWARD"):
                out.loss.backward()
            with phase("06_OPTIMIZER"):
                opt.step()
                opt.zero_grad()
            with phase("07_LOGGING"):
                if rank0 and (idx_step % 10 == 0 or idx_step + 1 == par.steps_per_epoch):
                    loss = out.loss.detach().item()
                    metrics_file.write(json.dumps({"tbl": "metrics", "idx_step": idx_step, "time": time.time() - start_time, "loss": loss}) + "\n")
                    metrics_file.flush()
                    print(f"finished step {idx_step + 1}/{par.steps_per_epoch}, loss={loss:.4f}", flush=True)

            if benchmark_started is not None and idx_step + 1 == benchmark_stop:
                if world_size > 1:  # Tokens and FLOPs summed over ranks; every rank reaches this step.
                    total = torch.tensor([n_tokens, n_flops], dtype=torch.float64, device=device)
                    dist.all_reduce(total)
                    n_tokens, n_flops = int(total[0].item()), float(total[1].item())
                synchronize()  # Only window boundaries synchronize; phases remain asynchronous.
                seconds = time.perf_counter() - benchmark_started
                steps = benchmark_stop - par.warmup_steps
                samples_per_second = steps * par.batch_size * world_size / seconds
                result = {
                    "tbl": "throughput", "time": time.time(), "device": str(device),
                    "torch_version": torch.__version__, "params": asdict(par),
                    "steps": steps, "seconds": seconds, "seconds_per_step": seconds / steps,
                    "samples_per_second": samples_per_second,
                    "input_mvox_per_second": samples_per_second * prod(par.patch_size) / 1e6,
                    "tokens_per_second": n_tokens / seconds,
                    "tokens_per_sample": n_tokens / (steps * par.batch_size * world_size),
                    "world_size": world_size,
                    "tflops_per_second": n_flops / seconds / 1e12,  # node total
                }
                if use_cuda:
                    gpu = torch.cuda.get_device_name(device)
                    peaks = [v for k, v in PEAK_BF16_TFLOPS.items() if k in gpu]
                    assert len(peaks) == 1, f"add {gpu!r} to PEAK_BF16_TFLOPS"
                    result["gpu_name"] = gpu
                    result["mfu"] = result["tflops_per_second"] / world_size / peaks[0]
                    result["max_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                if rank0:
                    with open(savedir / "performance.json", "a") as f:
                        f.write(json.dumps(result) + "\n")
                print(f"Unprofiled: {seconds / steps:.3f} s/step, {samples_per_second:.2f} samples/s, {n_tokens / seconds:.0f} tokens/s, "
                      f"{result['tflops_per_second'] / world_size:.0f} TFLOP/s/GPU" + (f", MFU {100 * result['mfu']:.1f}%" if use_cuda else ""))
            if prof is not None:
                prof.step()
                if idx_step + 1 == profile_stop:
                    profile_scope.close()
                    prof = None
            if idx_step % CHECKPOINT_EVERY == CHECKPOINT_EVERY - 1:
                save_checkpoint(idx_step + 1)
            # Time limit: checked every 100 steps with one all-reduce, so every rank stops at the same step.
            if par.max_hours and idx_step % 100 == 99:
                stop = torch.tensor(float(time.time() - start_time > par.max_hours * 3600), device=device)
                if world_size > 1:
                    dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                if stop.item():
                    print(f"Reached max_hours={par.max_hours} at step {idx_step + 1}", flush=True)
                    break
    save_checkpoint(idx_step + 1)
    if world_size > 1:
        dist.destroy_process_group()


def pca(n: int):
    """PCA maps of patch-token embeddings from allparams()[n]'s latest checkpoint, on a held-out EB val crop.

    pca.png columns: EM | PCA of tokens | PCA after subtracting each tile's mean token | token L2 norm.
    """
    import torch
    from PIL import Image
    par: Params = allparams()[n]
    savedir = Path(par.savedir)
    ckpt = torch.load(savedir / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = Lejepa(lejepa_config(par))
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = model.encoder.to(device).eval()

    # One 96x384x384 (z y x) window inside EB's val slab (EB z 3000-4000), held out from training.
    # miao needs the box strictly larger than the patch: one extra voxel per axis leaves exactly one window.
    shape, z0, y0, x0 = (96, 384, 384), 3400, 2000, 2000
    vol = lmd.get(HEMIBRAIN_EB).to_miao(bounding_box=[[o, o + s + 1] for o, s in zip((z0, y0, x0), shape)])
    mcfg = MiaoConfig(volumes=[vol], patch_size=list(shape), resolutions=[[8.0, 8.0, 8.0]],
                      samples_per_epoch=1, sampling="random", output_axes="lzyx")
    img = VolumeDataset(mcfg)[0]["img"]  # 1 Z Y X in [0, 1]

    # Tokens from non-overlapping tiles at the training global-view size, stitched into one grid.
    tile, patch = par.global_size, encoder.patch_embed.patch_size
    assert all(s % t == 0 for s, t in zip(shape, tile)), f"crop {shape} must tile by {tile}"
    grid = [s // p for s, p in zip(shape, patch)]
    feats = torch.empty(*grid, par.width)  # Z Y X D, tokens
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for z, y, x in product(*(range(0, s, t) for s, t in zip(shape, tile))):
            t = encoder.forward_features(img[None, :, z:z + tile[0], y:y + tile[1], x:x + tile[2]].to(device))
            tz, ty, tx = (o // p for o, p in zip((z, y, x), patch))
            gz, gy, gx = (t_ // p for t_, p in zip(tile, patch))
            feats[tz:tz + gz, ty:ty + gy, tx:tx + gx] = t[0].float().reshape(gz, gy, gx, -1).cpu()

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
    tiled = feats.reshape(nt[0], g[0], nt[1], g[1], nt[2], g[2], par.width)
    centered = (tiled - tiled.mean(dim=(1, 3, 5), keepdim=True)).reshape(*grid, par.width)
    rgb, erank, var = pca_rgb(feats.reshape(-1, par.width))
    rgb_c, erank_c, var_c = pca_rgb(centered.reshape(-1, par.width))
    between_tile = 1 - float(centered.var(dim=(0, 1, 2)).sum() / feats.var(dim=(0, 1, 2)).sum())

    # Token norms: "register"-like tokens that store global information show up as sparse high-norm outliers.
    norms = feats.norm(dim=-1)  # Z Y X
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
    stats = {"tbl": "pca", "step": ckpt["step"], "effective_rank": erank, "explained_variance": var,
             "centered_effective_rank": erank_c, "centered_explained_variance": var_c,
             "between_tile_variance": between_tile,  # fraction of token variance explained by tile means
             "norm_median": med, "norm_p99": float(torch.quantile(norms.flatten(), 0.99)), "norm_max": float(norms.max()),
             "norm_outliers": int((norms > 2 * med).sum()), "n_tokens": norms.numel()}
    (savedir / "pca.json").write_text(json.dumps(stats) + "\n")
    print(f"{savedir}: step {ckpt['step']}, effective rank {erank:.1f}, top-3 variance {sum(var[:3]):.2f}, "
          f"between-tile variance {between_tile:.2f}, norm outliers (>2x median) {stats['norm_outliers']}/{norms.numel()}; wrote pca.png")

def runlsf(n:int):
    import subprocess
    par:Params = allparams()[n]
    wipedir(par.savedir)
    RUN_NAME = "e00_basic"
    minutes = int(par.max_hours * 60) + 30 if par.max_hours else 15  # 30 min slack for startup + final checkpoint
    CPUS_PER_GPU = 12  # 8 GPUs -> all 96 cores; training processes need cores beyond the data workers
    assert par.n_workers + 1 <= CPUS_PER_GPU, f"n_workers={par.n_workers} leaves no core for the training process"
    cmd = f""" bsub -J {RUN_NAME} \
        -W {minutes // 60}:{minutes % 60:02d} \
        -P miaai \
        -n {par.n_gpus * CPUS_PER_GPU} \
        -R "span[hosts=1]" \
        -gpu "num={par.n_gpus}:mode=exclusive_process" \
        -q gpu_h200 \
        -o {par.savedir}/job_%J.log \
        uv run torchrun --standalone --nproc_per_node={par.n_gpus} e00_basic.py run {n}
        """
    subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"Submitted {RUN_NAME} {n} to LSF.")

def runmany():
    for i in range(len(allparams())):
        runlsf(i)

def runmany_sequential():
    for i in range(len(allparams())):
        run(i)

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
                    b1 = saved.keys() <= params.keys()
                    b2 = all(json_equal(params[k], saved[k]) for k in saved if k in params)
                    b3 = all(json_equal(params[k], defaults[k]) for k in params.keys() - saved.keys())
                    assert b1 and b2 and b3, (
                        f"Parameter mismatch in {path}:\n"
                        f"current: {params}\nsaved: {saved}"
                    )
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
        # "views": "views",
        # "patch_size": "input",
        # "global_size": "global",
        # "local_size": "local",
        # "compile": "compile",
        # "cudagraphs": "cudagraphs",
        "width": "width",
        "defer_image_ops": "defer",
        # "compile_blocks": "blocks",
        # "grad_compress": "compress",
        # "batch_views": "batch views",
        "n_gpus": "gpus",
        "batch_size": "batch",
        "n_workers": "workers",
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
    res.insert(res.columns.get_loc("tok/s") + 1, "ktok/s/gpu", res["tok/s"] / res["gpus"])
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
