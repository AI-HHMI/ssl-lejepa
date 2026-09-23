from __future__ import annotations

from typing import Literal
from dataclasses import dataclass, field, asdict
import os, sys
import json
import time
from pathlib import Path
from contextlib import ExitStack, nullcontext

from math import prod
from itertools import product

# local

from lib.models import Lejepa, LejepaConfig
from lib.util import *

# external 

import lmd_catalog as lmd
from miao.config import MiaoConfig
from miao import VolumeDataset

from rich import print as pprint
import pandas
import plotly.express as px


F32Mode = Literal["highest", "high", "medium"]

@dataclass(slots=True)
class Params:
    savedir: str = "outdir/e00/main/basic/"
    # patch_size: list[int] = [104, 232, 232]
    # patch_size: list[int] = field(default_factory=lambda: [104, 232, 232])
    patch_size: list[int] = field(default_factory=lambda: [48, 144, 144])
    batch_size: int = 42
    steps_per_epoch: int = 71  # warmup + benchmark + 1 profiler warmup + profile steps: stop right after profiling
    n_layers: int = 12
    f32mode: F32Mode = "high"
    n_workers: int = 2
    prefetch_factor: int = 2
    amp: bool = False  # bf16 autocast for forward + loss
    compile: bool = False  # torch.compile(dynamic=True) the encoder

    # profiling params
    warmup_steps: int = 10
    benchmark_steps: int = 50
    profile_steps: int = 10  # Set to zero to disable trace collection.

def allparams():
    params = []
    # patchsize = logish_samples([4, 12, 12], [2,3], 2, 7)[1:]
    n_workers = [4, 8, 16]
    for i, nw in enumerate(n_workers):
        p = Params()
        p.savedir = f"outdir/e00/workers/d{i}/"
        p.n_workers = nw
        p.amp = True
        p.compile = True
        p.batch_size = 84
        print(i, nw)
        params.append(p)
    # pprint(params)
    return params

def write_trace_summary(savedir):
    summary = trace_summary(Path(savedir) / "profile.json")
    (Path(savedir) / "trace_summary.json").write_text(json.dumps({"tbl": "trace_summary", **summary}) + "\n")
    print(f"{savedir}: GPU busy during profiled steps {100 * summary['gpu_busy']:.0f}%")

# def backfill_gpu_busy():
#     """Write gpu_busy.json from profile.json for runs that predate it. Run on the cluster (traces aren't pulled)."""
#     for par in allparams():
#         b1 = (Path(par.savedir) / "profile.json").is_file()
#         b2 = (Path(par.savedir) / "trace_summary.json").is_file()
#         if b1 and not b2:
#             write_trace_summary(par.savedir)

def collate_images(samples):
    import torch
    return torch.stack([s["img"] for s in samples])

def run(n:int):
    start_time = time.time()
    par : Params = allparams()[n]
    if min(par.warmup_steps, par.benchmark_steps, par.profile_steps) < 0:
        raise ValueError("Profiling and benchmark step counts must be nonnegative")
    # lmd.set_data_root("/Volumes/miaai/lmd-v0.0.1/data")
    # volumes = [x.to_miao() for x in lmd.all() if "flyliconn" in x.name]

    import torch
    torch.set_float32_matmul_precision(par.f32mode)

    volumes = [x.to_miao() for x in lmd.all() if x.name == "exm-drosophila-flyliconn-matt-260601-60X-B4-2-045/crop-001"]
    mcfg = MiaoConfig(
        volumes=volumes,
        patch_size=par.patch_size,
        resolutions=[[25.0, 10.0, 10.0]],
        samples_per_epoch=par.batch_size * par.steps_per_epoch,
        sampling="random",
        output_axes="lzyx",
    )
    dl = VolumeDataset(mcfg)
    loader = torch.utils.data.DataLoader(
      dl,
      batch_size=par.batch_size,
      num_workers=par.n_workers,
      collate_fn=collate_images,
      prefetch_factor=par.prefetch_factor,
      pin_memory=torch.cuda.is_available(),
      multiprocessing_context="spawn",
    )
    batches = iter(loader)

    # pprint(volumes)
    # pprint(mcfg)
    # pprint(dl[0]['img'].shape)

    savedir = Path(par.savedir)

    with open(savedir / "runs.json", 'a') as rfile, open(repo_root() / "_diffs/diffs.json", "a") as difflog:
        gp = git_provenance()
        rfile.write(json.dumps({k:gp[k] for k in ['commit_id', 'diff_hash']}) + "\n")
        difflog.write(json.dumps({gp['diff_hash']:gp['diff']}) + "\n")

    cfg = LejepaConfig(
        n_layers = par.n_layers,
        width = 512,
        views = 'basic',
        lamb = 0.1,
    )
    model = Lejepa(cfg)
    pprint(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"We're using torch device {device} .")
    model = model.to(device)
    if par.compile:
        torch._logging.set_logs(recompiles=True)  # recompiles show up in job_*.log
        model.encoder.compile(dynamic=True)  # Views change shape every step.
    opt = torch.optim.Adam(model.parameters(), lr = 1e-4)
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
    with open(savedir / "metrics.json", "a") as metrics_file, ExitStack() as profile_scope:
        for idx_step in range(par.steps_per_epoch):
            if idx_step == par.warmup_steps and idx_step < benchmark_stop:
                synchronize()
                benchmark_started = time.perf_counter()
            if par.profile_steps and idx_step == profile_start and idx_step + 1 < profile_stop:
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
                x = x.to(device, non_blocking=True)
            with phase("04_FORWARD_AND_LOSS"), torch.autocast(device.type, dtype=torch.bfloat16, enabled=par.amp):
                out = model(x)
            if benchmark_started is not None:
                n_tokens += out.n_tokens
            with phase("05_BACKWARD"):
                out.loss.backward()
            with phase("06_OPTIMIZER"):
                opt.step()
                opt.zero_grad()
            with phase("07_LOGGING"):
                if idx_step % 10 == 0 or idx_step + 1 == par.steps_per_epoch:
                    loss = out.loss.detach().item()
                    metrics_file.write(json.dumps({"tbl": "metrics", "idx_step": idx_step, "time": time.time() - start_time, "loss": loss}) + "\n")
                    metrics_file.flush()
                    print(f"finished step {idx_step + 1}/{par.steps_per_epoch}, loss={loss:.4f}", flush=True)

            if benchmark_started is not None and idx_step + 1 == benchmark_stop:
                synchronize()  # Only window boundaries synchronize; phases remain asynchronous.
                seconds = time.perf_counter() - benchmark_started
                steps = benchmark_stop - par.warmup_steps
                samples_per_second = steps * par.batch_size / seconds
                result = {
                    "tbl": "throughput", "time": time.time(), "device": str(device),
                    "torch_version": torch.__version__, "params": asdict(par),
                    "steps": steps, "seconds": seconds, "seconds_per_step": seconds / steps,
                    "samples_per_second": samples_per_second,
                    "input_mvox_per_second": samples_per_second * prod(par.patch_size) / 1e6,
                    "tokens_per_second": n_tokens / seconds,
                    "tokens_per_sample": n_tokens / (steps * par.batch_size),
                }
                with open(savedir / "performance.json", "a") as f:
                    f.write(json.dumps(result) + "\n")
                print(f"Unprofiled: {seconds / steps:.3f} s/step, {samples_per_second:.2f} samples/s, {n_tokens / seconds:.0f} tokens/s")
            if prof is not None:
                prof.step()
                if idx_step + 1 == profile_stop:
                    profile_scope.close()
                    prof = None


def runlsf(n:int):
    import subprocess
    par:Params = allparams()[n]
    wipedir(par.savedir)
    RUN_NAME = "e00_basic"
    NUM_GPUS = 1
    cmd = f""" bsub -J {RUN_NAME} \
        -W 4:00 \
        -P miaai \
        -n {par.n_workers + 1} \
        -R "span[hosts=1]" \
        -gpu "num={NUM_GPUS}:mode=exclusive_process" \
        -q gpu_b300 \
        -o {par.savedir}/job_%J.log \
        uv run python e00_basic.py run {n}
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
                    assert record["params"] == params, (
                        f"Parameter mismatch in {path}:\n"
                        f"current: {params}\nsaved: {record['params']}"
                    )
                row = {**params, **record}
                row["patch_size"] = tuple(row["patch_size"])
                rows.append(row)
    res = pandas.DataFrame(rows)
    pprint(res)
    return res

def plot1():
    res = loadJsonTable("metrics.json")
    px.line(res, x="idx_step", y="loss", color="n_workers", markers=True).show()

def plot2():
    res = loadJsonTable("performance.json")
    res['vox'] = res.patch_size.apply(prod)
    pprint(res.columns)
    px.bar(res, x="n_workers", y="tokens_per_second", color="n_workers", barmode="group").show()
    px.bar(res, x="compile", y="input_mvox_per_second", color="compile", facet_col="batch_size", facet_row="amp", barmode="group").show()

def table():
    res = loadJsonTable("performance.json")
    trace = loadJsonTable("trace_summary.json")
    # Host ms per profiled step in each phase (see lib.util.trace_summary).
    phases = {"01_DATA_IO_ms": "io ms", "04_FORWARD_AND_LOSS_ms": "fwd ms", "05_BACKWARD_ms": "bwd ms", "06_OPTIMIZER_ms": "opt ms"}
    for k in ["gpu_busy", "step_ms", *phases]:
        res[k] = res.savedir.map(dict(zip(trace.savedir, trace[k]))) if k in trace else float("nan")
    cols = {
        "savedir": "run", "compile": "compile", "batch_size": "batch", "n_workers": "workers",
        "gpu_busy": "gpu busy %", "samples_per_second": "samples/s",
        "tokens_per_second": "tok/s", "input_mvox_per_second": "Mvox/s",
        "step_ms": "prof step ms", **phases,
    }
    res = res[list(cols)].rename(columns=cols) # type: ignore
    res["gpu busy %"] *= 100
    res["tok/s"] /= 1e3
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
