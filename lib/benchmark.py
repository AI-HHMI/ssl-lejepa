"""Throughput benchmark window and profiler window for a training loop."""

from __future__ import annotations

import json
import time
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from math import prod
from pathlib import Path

import torch
import torch.distributed as dist

from lib.util import trace_summary

# Dense bf16 tensor-core peak per GPU, matched by substring of torch.cuda.get_device_name().
# B300/B200 from NVIDIA's HGX spec (36 PFLOPS sparse per 8 GPUs); check the datasheet if it matters.
# RTX PRO 6000 Blackwell Server Edition: ~500 dense (1 PFLOPS sparse), from memory; verify against NVIDIA's datasheet.
PEAK_BF16_TFLOPS = {"A100": 312, "H100": 989, "H200": 989, "B200": 2250, "B300": 2250, "RTX PRO 6000": 500}

PROFILE_README = (
    "Open profile.json in https://ui.perfetto.dev to inspect CPU and CUDA tracks.\n"
    "01_DATA_IO includes storage/network waits, decoding, and preprocessing;\n"
    "02_CPU_BATCH is host tensor assembly. Neither is a pure disk counter.\n"
    "03_H2D_TRANSFER is host-side transfer time; inspect Memcpy HtoD on CUDA tracks.\n"
    "GPU gaps aligned with loading suggest input starvation; gaps amid CPU dispatch\n"
    "suggest host overhead. Busy GPU tracks alone do not prove compute saturation.\n"
    "CPU and device times overlap: do not add them into bottleneck percentages.\n\n"
)


def write_trace_summary(savedir):
    summary = trace_summary(Path(savedir) / "profile.json")
    (Path(savedir) / "trace_summary.json").write_text(json.dumps({"tbl": "trace_summary", **summary}) + "\n")
    print(f"{savedir}: GPU busy during profiled steps {100 * summary['gpu_busy']:.0f}%")


class Benchmark:
    """Owns a training loop's timing and profiling state.

    Steps [warmup_steps, warmup_steps + benchmark_steps) are timed with no profiler or annotations
    active and appended to savedir/performance.json. The next step warms the profiler up and the
    following profile_steps are traced (rank 0 only) to profile.json, profile.out and
    trace_summary.json. Use as a context manager around the loop, and each step call
    begin(i), phase(name) around each part, count(out) after the forward, and end(i).
    `par` needs savedir, warmup_steps, benchmark_steps, profile_steps, steps_per_epoch,
    batch_size and patch_size.
    """

    def __init__(self, par, device: torch.device, world_size: int, rank0: bool):
        self.par = par
        self.device = device
        self.world_size = world_size
        self.rank0 = rank0
        self.savedir = Path(par.savedir)
        self.use_cuda = device.type == "cuda"
        self.profile_start = par.warmup_steps + par.benchmark_steps
        self.profile_stop = min(self.profile_start + 1 + par.profile_steps, par.steps_per_epoch)
        self.benchmark_stop = min(self.profile_start, par.steps_per_epoch)
        self.started: float | None = None  # perf_counter at the start of the timed window
        self.n_tokens = 0  # encoder tokens since the window started
        self.n_flops = 0  # model training FLOPs since the window started (see Lejepa.forward)
        self.prof: torch.profiler.profile | None = None
        self.scope = ExitStack()

    def __enter__(self) -> Benchmark:
        return self

    def __exit__(self, *exc) -> None:
        self.scope.close()

    def synchronize(self) -> None:
        if self.use_cuda:
            torch.cuda.synchronize(self.device)

    def phase(self, name: str):
        """Named record_function while profiling; a no-op otherwise, so the timed window has no hooks."""
        return torch.profiler.record_function(name) if self.prof is not None else nullcontext()

    def begin(self, idx_step: int) -> None:
        par = self.par
        if idx_step == par.warmup_steps and idx_step < self.benchmark_stop:
            self.synchronize()
            self.started = time.perf_counter()
        b1 = self.rank0 and par.profile_steps > 0
        b2 = idx_step == self.profile_start and idx_step + 1 < self.profile_stop
        if b1 and b2:
            self.synchronize()
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.use_cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.prof = self.scope.enter_context(torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=par.profile_steps, repeat=1),
                on_trace_ready=self._save_trace,
                record_shapes=False, profile_memory=False, with_stack=False,
            ))
            self.prof.add_metadata_json("run_config", json.dumps(asdict(par)))

    def count(self, out) -> None:
        if self.started is not None:
            self.n_tokens += out.n_tokens
            self.n_flops += out.n_flops

    def end(self, idx_step: int) -> None:
        if self.started is not None and idx_step + 1 == self.benchmark_stop:
            self._write_throughput()
        if self.prof is not None:
            self.prof.step()
            if idx_step + 1 == self.profile_stop:
                self.scope.close()
                self.prof = None

    def _write_throughput(self) -> None:
        assert self.started is not None, "timed window never started"
        par, world_size = self.par, self.world_size
        n_tokens, n_flops = self.n_tokens, self.n_flops
        if world_size > 1:  # Tokens and FLOPs summed over ranks; every rank reaches this step.
            total = torch.tensor([n_tokens, n_flops], dtype=torch.float64, device=self.device)
            dist.all_reduce(total)
            n_tokens, n_flops = int(total[0].item()), float(total[1].item())
        self.synchronize()  # Only window boundaries synchronize; phases remain asynchronous.
        seconds = time.perf_counter() - self.started
        steps = self.benchmark_stop - par.warmup_steps
        samples_per_second = steps * par.batch_size * world_size / seconds
        result = {
            "tbl": "throughput", "time": time.time(), "device": str(self.device),
            "torch_version": torch.__version__, "params": asdict(par),
            "steps": steps, "seconds": seconds, "seconds_per_step": seconds / steps,
            "samples_per_second": samples_per_second,
            "input_mvox_per_second": samples_per_second * prod(par.patch_size) / 1e6,
            "tokens_per_second": n_tokens / seconds,
            "tokens_per_sample": n_tokens / (steps * par.batch_size * world_size),
            "world_size": world_size,
            "tflops_per_second": n_flops / seconds / 1e12,  # node total
        }
        if self.use_cuda:
            gpu = torch.cuda.get_device_name(self.device)
            peaks = [v for k, v in PEAK_BF16_TFLOPS.items() if k in gpu]
            assert len(peaks) == 1, f"add {gpu!r} to PEAK_BF16_TFLOPS"
            result["gpu_name"] = gpu
            result["mfu"] = result["tflops_per_second"] / world_size / peaks[0]
            result["max_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
        if self.rank0:
            with open(self.savedir / "performance.json", "a") as f:
                f.write(json.dumps(result) + "\n")
        print(f"Unprofiled: {seconds / steps:.3f} s/step, {samples_per_second:.2f} samples/s, {n_tokens / seconds:.0f} tokens/s, "
              f"{result['tflops_per_second'] / world_size:.0f} TFLOP/s/GPU" + (f", MFU {100 * result['mfu']:.1f}%" if self.use_cuda else ""))

    def _save_trace(self, prof) -> None:
        savedir = self.savedir
        prof.export_chrome_trace(str(savedir / "profile.json"))
        write_trace_summary(savedir)
        averages = prof.key_averages()
        report = (
            f"Device: {self.device}; recorded steps (zero-based): "
            f"{self.profile_start + 1}..{self.profile_stop - 1}\n"
            + PROFILE_README
            + "OPERATORS SORTED BY SELF CPU TIME\n"
            + averages.table(sort_by="self_cpu_time_total", row_limit=50)
        )
        if self.use_cuda:
            report += "\n\nOPERATORS SORTED BY SELF DEVICE TIME\n" + averages.table(
                sort_by="self_device_time_total", row_limit=50,
            )
        (savedir / "profile.out").write_text(report)
        print(f"Saved {savedir / 'profile.json'} and {savedir / 'profile.out'}")
