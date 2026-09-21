from __future__ import annotations

from dataclasses import dataclass, field, asdict
from contextlib import ExitStack, nullcontext
from functools import reduce
from math import prod
from pathlib import Path

from lib.models import Lejepa, LejepaConfig
from lib.util import pick_entrypoint
import lmd_catalog as lmd
from miao.config import MiaoConfig
from miao import VolumeDataset
from rich import print as pprint
import os, sys
import json
import time


@dataclass(slots=True)
class Params:
    savedir: str = "outdir/e00/main/basic/"
    # patch_size: list[int] = [104, 232, 232]
    patch_size: list[int] = field(default_factory=lambda: [104, 232, 232])
    batch_size: int = 42
    n_epoch: int = 1000

    # profiling params
    warmup_steps: int = 10
    benchmark_steps: int = 50
    profile_steps: int = 3  # Set to zero to disable trace collection.
    
def allparams():
    params = []
    ps = [
        [2**3  , 2**3*3, 2**3*3],
        [2**2*3, 2**2*9, 2**2*9],
        [2**4  , 2**4*3, 2**4*3],
        [2**3*3, 2**3*9, 2**3*9],
        [2**5  , 2**5*3, 2**5*3],
        [2**4*3, 2**4*9, 2**4*9],
        [2**6  , 2**6*3, 2**6*3],
        [2**5*3, 2**5*9, 2**5*9],
    ]
    for i, _p in enumerate(ps):
        p = Params()
        p.patch_size = _p
        p.savedir = f"outdir/e00/main/basic/d{i}/"
        params.append(p)
    return params

def run(n:int):
    import torch

    par : Params = allparams()[n]
    if min(par.warmup_steps, par.benchmark_steps, par.profile_steps) < 0:
        raise ValueError("Profiling and benchmark step counts must be nonnegative")
    # lmd.set_data_root("/Volumes/miaai/lmd-v0.0.1/data")
    # volumes = [x.to_miao() for x in lmd.all() if "flyliconn" in x.name]
    volumes = [x.to_miao() for x in lmd.all() if x.name == "exm-drosophila-flyliconn-matt-260601-60X-B4-2-045/crop-001"]
    pprint(volumes)

    mcfg = MiaoConfig(
        volumes=volumes,
        patch_size=par.patch_size,
        resolutions=[[25.0, 10.0, 10.0]],
        samples_per_epoch=1000,
        sampling="random",
        output_axes="lzyx",
    )
    dl = VolumeDataset(mcfg)
    # pprint(mcfg)
    # pprint(dl[0]['img'].shape)

    os.makedirs(par.savedir, exist_ok=True)

    cfg = LejepaConfig(
        n_layers = 12,
        width = 512,
        views = 'basic',
        profile=False,  # The training loop owns profiling, including backward and I/O.
        lamb = 0.1,
    )
    model = Lejepa(cfg)
    pprint(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"We're using torch device {device} .")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr = 1e-4)
    savedir = Path(par.savedir)
    use_cuda = device.type == "cuda"
    activities = [torch.profiler.ProfilerActivity.CPU]
    if use_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profile_start = par.warmup_steps + par.benchmark_steps
    profile_stop = min(profile_start + 1 + par.profile_steps, par.n_epoch)
    benchmark_stop = min(profile_start, par.n_epoch)

    def synchronize():
        if use_cuda:
            torch.cuda.synchronize(device)

    def save_trace(prof):
        prof.export_chrome_trace(str(savedir / "profile.json"))
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
    with open(savedir / "metrics.jsonl", "a") as metrics_file, ExitStack() as profile_scope:
        for ep in range(par.n_epoch):
            if ep == par.warmup_steps and ep < benchmark_stop:
                synchronize()
                benchmark_started = time.perf_counter()
            if par.profile_steps and ep == profile_start and ep + 1 < profile_stop:
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
                samples = [dl[par.batch_size * ep + i] for i in range(par.batch_size)]
            with phase("02_CPU_BATCH"):
                x = torch.stack([sample["img"] for sample in samples])
            with phase("03_H2D_TRANSFER"):
                x = x.to(device)
            with phase("04_FORWARD_AND_LOSS"):
                out = model(x)
            with phase("05_BACKWARD"):
                out.loss.backward()
            with phase("06_OPTIMIZER"):
                opt.step()
                opt.zero_grad()
            with phase("07_LOGGING"):
                if ep % 10 == 0 or ep + 1 == par.n_epoch:
                    loss = out.loss.detach().item()
                    metrics_file.write(json.dumps({"tbl": "metrics", "epoch": ep, "time": time.time(), "loss": loss}) + "\n")
                    metrics_file.flush()
                    print(f"finished step {ep + 1}/{par.n_epoch}, loss={loss:.4f}", flush=True)

            if benchmark_started is not None and ep + 1 == benchmark_stop:
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
                }
                with open(savedir / "performance.jsonl", "a") as f:
                    f.write(json.dumps(result) + "\n")
                print(f"Unprofiled: {seconds / steps:.3f} s/step, {samples_per_second:.2f} samples/s")
            if prof is not None:
                prof.step()
                if ep + 1 == profile_stop:
                    profile_scope.close()
                    prof = None


def runlsf(n:int):
    import subprocess
    par = allparams()[n]
    RUN_NAME = "e00_basic"
    NUM_GPUS = 1
    cmd = f""" bsub -J {RUN_NAME} \
        -W 46:00 \
        -P miaai \
        -n {NUM_GPUS} \
        -R "span[hosts=1]" \
        -gpu "num={NUM_GPUS}:mode=exclusive_process" \
        -q gpu_a100 \
        -o {par.savedir}/job_%J.log \
        uv run python e00_basic.py {n}
        """
    subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"Submitted {RUN_NAME} {n} to LSF.")

def runmany():
    for i in range(len(allparams())):
        runlsf(i)

def runmany_sequential():
    for i in range(len(allparams())):
        run(i)

def analysis():
    import pandas
    import plotly.express as px
    def loadAndFuse(par:Params):
        # metr = json.load(open(par.savedir + "metrics.jsonl", "r"))
        try:
            with open(par.savedir + "metrics.jsonl") as f:
                metr = [json.loads(line) for line in f if line.strip()]
            tabl = [{**m, **asdict(par)} for m in metr]
            return tabl
        except:
            return []
    params = [loadAndFuse(p) for p in allparams()]
    res = list(reduce(lambda a,b: a+b, params))

    for r in res:
        r['patch_size'] = tuple(r['patch_size'])
    res = pandas.DataFrame(res)
    pl = px.scatter(res, x="epoch", y="loss", color="patch_size")
    pl.show()
    print(res)

def test():
    x = lmd.all()
    for xi in x:
        pprint(xi)

if __name__ == "__main__":
    import sys
    print(sys.argv)
    if len(sys.argv) == 1:
        pick_entrypoint()
    elif sys.argv[1] == 'many':
        runmany()
    elif sys.argv[1] == 'lsf':
        runlsf(int(sys.argv[2]))
    elif sys.argv[1] == 'test':
        test()
    elif sys.argv[1] == 'anl':
        analysis()
    elif sys.argv[1] == 'seq':
        runmany_sequential()
    else:
        run(int(sys.argv[1]))
