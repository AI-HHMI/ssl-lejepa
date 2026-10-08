"""Readers for e00 results (local): run dirs, their saved params, compute, walltimes and probe scores, from pulled
artifacts in outdir/ only, a mirror of the cluster's append-only run dirs. Imports no experiment code (experiment,
lib/ models: nothing that builds a Lejepa): artifacts describe themselves (saved params), so every function works
on any past sweep at HEAD. Shared by report_*.py and analysis_plots.py (each experiment's figures and tables).
"""

import json
import re
from pathlib import Path

BOUNDARY_CHANNELS = {"short (+1)": ["(1, 0, 0)", "(0, 1, 0)", "(0, 0, 1)"], "long (+10)": ["(10, 0, 0)", "(0, 10, 0)", "(0, 0, 10)"]}


def read_jsonl(path) -> list[dict]:
    """JSON-lines rows of `path`, or [] if it doesn't exist (a run dir missing a file it predates, or hasn't
    written yet)."""
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []

def run_dirs(sweep: str) -> list[Path]:
    """outdir/<sweep>/dN/ run dirs in numeric order. A sweep that is itself a run dir (e00/probe-test/d1) gives just it."""
    if re.fullmatch(r"d\d+", Path(sweep).name) and Path("outdir", sweep).is_dir():
        return [Path("outdir", sweep)]
    dirs = [d for d in Path("outdir", sweep).glob("d*/") if re.fullmatch(r"d\d+", d.name)]
    assert dirs, f"no run dirs in outdir/{sweep}/; run ./pull.sh?"
    return sorted(dirs, key=lambda d: int(d.name[1:]))

def saved_params(d: Path) -> dict:
    """A run's own params: runs.json (written first thing, so crashed runs have it too), else performance.json
    (runs from before runs.json had params), else {}. The newest row: a dir's rows agree on params except where an
    older job predates a field (e.g. a probe row from before init_from)."""
    for name in ["runs.json", "performance.json"]:
        found = next((r["params"] for r in reversed(read_jsonl(d / name)) if "params" in r), None)
        if found:
            return found
    return {}

def tflop_per_step(d: Path) -> float | None:
    """TFLOP for one training step of this run dir (all GPUs), from its measured benchmark-window throughput row
    (performance.json). None before the run has logged one (still warming up, or crashed/OOM'd first)."""
    tp = [r for r in read_jsonl(d / "performance.json") if r.get("tbl") == "throughput"]
    t = tp[-1] if tp else {}
    return t["tflops_per_second"] * t["seconds_per_step"] if t.get("tflops_per_second") else None

def run_compute(d: Path) -> tuple[int | None, float | None]:
    """(training steps, total EFLOP) of a run dir: the last logged step in metrics.json + 1, times FLOPs per step from
    the benchmark window (TFLOP/s x s/step, all GPUs; fixed with displace views). None where a file is missing."""
    m = read_jsonl(d / "metrics.json")
    steps = m[-1]["idx_step"] + 1 if m else None
    tps = tflop_per_step(d)
    return steps, tps * steps / 1e6 if tps and steps else None

def job_walltime_s(text: str) -> float | None:
    """LSF wall-clock run time in seconds, from a job log's "Resource usage summary" footer. None if the job
    hasn't finished (no footer yet)."""
    m = re.search(r"Run time\s*:\s*([\d.]+) sec", text)
    return float(m[1]) if m else None

def probe_walltime_s(d: Path) -> float | None:
    """Wall-clock time spent probing, not pretraining: a dedicated probe job's LSF run time when probe ran as its
    own job (job_probe_*.log, e.g. probe-test/probeall), else, when probe ran inline at the end of the training
    job (e.g. viewsizes-v2, no separate job log to read), the time from metrics.json's last write (training done)
    to probe.json's (probe done) -- both pulled with rsync -a, so mtimes are the cluster's. None if neither applies."""
    probe_logs = sorted(d.glob("job_probe_*.log"))
    if probe_logs:
        return job_walltime_s(probe_logs[-1].read_text(errors="ignore"))
    metrics, probe = d / "metrics.json", d / "probe.json"
    return probe.stat().st_mtime - metrics.stat().st_mtime if metrics.is_file() and probe.is_file() else None

def sweep_name(sweep: str) -> str:
    """The sweep's short name, e.g. 'e00/viewsizes-v2' -> 'viewsizes-v2'; for a single run dir, its sweep's
    ('e00/probe-test/d2' -> 'probe-test')."""
    parts = sweep.split("/")
    return parts[-2] if re.fullmatch(r"d\d+", parts[-1]) else parts[-1]

def probed_dirs(sweep: str) -> list[Path]:
    """run_dirs(sweep) that have a probe.json."""
    return [d for d in run_dirs(sweep) if (d / "probe.json").is_file()]

def probe_stats(d: Path) -> dict:
    """The (last, only) row of a run's probe.json."""
    return read_jsonl(d / "probe.json")[-1]

def boundary_ap(st: dict) -> dict[str, float]:
    """Mean boundary AP per range (BOUNDARY_CHANNELS) from one probe.json row."""
    return {r: sum(st[f"boundary_ap_{c}"] for c in chans) / 3 for r, chans in BOUNDARY_CHANNELS.items()}

def probe_source(d: Path) -> tuple[Path, dict, str]:
    """(source run dir, its saved params, init_from) for a probe.json row: d itself, unless init_from points at
    another run's checkpoint (a probe-only job, e.g. probeall) or "random" (no source run, the baseline)."""
    init = saved_params(d).get("init_from", "")
    src = Path(init) if init and init != "random" else d
    return src, saved_params(src), init

def probe_runs(sweeps: tuple[str, ...]) -> tuple[list[tuple], dict[str, dict[str, float]]]:
    """((sweep, run dir, source run dir, source params, init_from) per probed run, boundary AP per random-baseline name)
    over sweeps: the runs analysis_plots' probe_vs_compute and probe_short_vs_long plot, with the init_from random
    runs split out as baselines named by their decoder, e.g. 'random, linear' / 'random, unetr' / 'random, unetr
    (encoder trained)'."""
    runs, random = [], {}
    for sw in sweeps:
        for d in probed_dirs(sw):
            src, p, init = probe_source(d)
            if init == "random":
                dec = p.get("decoder", "linear")
                random[f"random, {dec}" + ("" if dec == "linear" or p.get("unetr_freeze_encoder", True) else " (encoder trained)")] = boundary_ap(probe_stats(d))
            else:
                runs.append((sw, d, src, p, init))
    assert runs, f"no probe.json in {sweeps}; run ./pull.sh?"
    return runs, random
