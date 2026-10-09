"""Readers for e00 results (local): run dirs, their saved params, compute, walltimes, probe scores and benchmark rows,
from pulled artifacts in outdir/ only, a mirror of the cluster's append-only run dirs, plus a few tools (ng_link,
check_runs). Imports no experiment code (experiment, lib/ models: nothing that builds a Lejepa), only lib.util's
entrypoint CLI: artifacts describe themselves (saved params), so every function works on any past sweep at HEAD.
Shared by analysis_plots.py (report cards) and reports/<sweep>.py (one page per experiment).
CLI for the tools: uv run --extra analysis python analysis.py check_runs
"""

import json
import os
import re
import sys
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path
from urllib.parse import quote

import lmd_catalog as lmd
import pandas
from lmd_catalog.catalog import DEFAULT_DATA_ROOT
from lmd_catalog.viewers import make_neuroglancer_url, parse_neuroglancer_url, to_fileglancer_content_url

from lib.util import call_entrypoint, pick_entrypoint

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

def load_table(sweep: str, filename: str) -> pandas.DataFrame:
    """JSON-lines rows of `filename` from every run dir outdir/<sweep>/dN/, each joined with that run's own saved
    params (saved_params), savedir and run name. Params a run predates are missing (NaN in the table)."""
    rows = []
    for d in run_dirs(sweep):
        rows += [{**saved_params(d), **r, "savedir": str(d), "run": d.name} for r in read_jsonl(d / filename)]
    assert rows, f"no {filename} in outdir/{sweep}/d*/; run ./pull.sh?"
    return pandas.DataFrame(rows)

def save_table(df: pandas.DataFrame, name: str):
    """Keep a copy of a report's table at results/<name>.csv."""
    path = Path("results") / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def short_runs(savedirs):
    """savedir relative to the sweep's common prefix, e.g. 'outdir/e00/x/d3/' -> 'd3'."""
    prefix = os.path.commonpath(list(savedirs))
    return savedirs.str[len(prefix):].str.strip("/")

def sweep_name(sweep: str) -> str:
    """The sweep's short name, e.g. 'e00/viewsizes-v2' -> 'viewsizes-v2'; for a single run dir, its sweep's
    ('e00/probe-test/d2' -> 'probe-test')."""
    parts = sweep.split("/")
    return parts[-2] if re.fullmatch(r"d\d+", parts[-1]) else parts[-1]

def job_text(d: Path, *prefixes: str) -> str:
    """Latest job log's text for a run dir: the first prefix with a job_<prefix>_*.log (e.g. "run", "probe"),
    else the older job_<jobid>.log name. "" if the run dir has no job log yet (still queued)."""
    for prefix in prefixes:
        logs = sorted(d.glob(f"job_{prefix}_*.log"))
        if logs:
            return logs[-1].read_text(errors="ignore")
    logs = sorted(d.glob("job_[0-9]*.log"))
    return logs[-1].read_text(errors="ignore") if logs else ""

def job_walltime_s(text: str) -> float | None:
    """LSF wall-clock run time in seconds, from a job log's "Resource usage summary" footer. None if the job
    hasn't finished (no footer yet)."""
    m = re.search(r"Run time\s*:\s*([\d.]+) sec", text)
    return float(m[1]) if m else None

def fmt_duration(seconds: float | None) -> str | None:
    """Wall-clock duration as e.g. "8h 13m" (minutes only past 1h; seconds only under 1m). None passes through,
    so a missing walltime stays NaN in a table rather than becoming the string "None"."""
    if seconds is None:
        return None
    m, s = divmod(round(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m" if h else f"{m}m {s}s" if m else f"{s}s"

def varying_params(params: list[dict]) -> set:
    """Param names whose values differ between runs (runs without saved params are left out)."""
    known = [p for p in params if p]
    return {k for k in dict.fromkeys(k for p in known for k in p) if len({json.dumps(p.get(k)) for p in known}) > 1}

def profile_device_ms(path: str | Path) -> tuple[float, dict[str, float]]:
    """GPU ms per profiled step from a Benchmark profile.out: the total, and each row of its self-device-time table.

    Rows overlap (a CompiledFxGraph call's time includes its kernels'), so only sum disjoint rows, e.g. kernels
    picked by name. Row names are truncated by the table.
    """
    text = Path(path).read_text()
    steps = re.search(r"recorded steps \(zero-based\): (\d+)\.\.(\d+)", text)
    total = re.search(r"Self CUDA time total: ([\d.]+)(us|ms|s)", text)
    assert steps and total and "SORTED BY SELF DEVICE TIME" in text, f"{path} is not a CUDA Benchmark profile.out"
    n = int(steps[2]) - int(steps[1]) + 1
    scale = {"us": 1e-3, "ms": 1.0, "s": 1e3}
    rows: dict[str, float] = {}
    for line in text.split("SORTED BY SELF DEVICE TIME")[1].split("Self CPU time total")[0].splitlines():
        f = re.split(r"\s{2,}", line.strip())  # Name, Self CPU %, Self CPU, ..., Self CUDA (7th), ..., # of Calls
        t = re.fullmatch(r"([\d.]+)(us|ms|s)", f[6]) if len(f) == 11 else None
        if t:
            rows[f[0]] = rows.get(f[0], 0.0) + float(t[1]) * scale[t[2]] / n
    return float(total[1]) * scale[total[2]] / n, rows

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

def view_sizes(p: dict) -> str:
    """'<patch>/<global>/<local>' view-size label from saved params, e.g. '16/128/96'."""
    return f'{p["patch_size"][0]}/{p["global_size"][0]}/{p["local_size"][0]}'

def config_label(p: dict, varying: set) -> str:
    """Short run label from saved params, e.g. 'B300 w512 b64 cudagraphs+eager-pe'; view sizes and GPU count only
    if they vary. '?' for runs that crashed before params were saved (runs.json had none until 2026-09-28)."""
    if not p:
        return "? (no saved params)"
    mode = "eager" if not p.get("compile", True) else "cudagraphs" if p.get("cudagraphs") else "dynamic"
    parts = [p["queue"].removeprefix("gpu_").upper(), f'w{p["width"]}', f'b{p["batch_size"]}',
             mode + ("+eager-pe" if p.get("eager_patch_embed") else "")]
    if varying & {"patch_size", "global_size", "local_size"}:
        parts.append(view_sizes(p))
    if "n_gpus" in varying or p.get("n_gpus", 1) > 1:
        parts.append(f'x{p.get("n_gpus", 1)}')
    if "lr" in varying:
        parts.append(f'lr{p["lr"]:.1e}')
    return " ".join(parts)

def size_label(p: dict, varying: set) -> str:
    """Label from saved params with only what differs between the plotted runs (varying_params): width, view sizes,
    GPU count. '?' if params are missing."""
    if not p:
        return "?"
    parts = [f'w{p["width"]}'] if "width" in varying else []
    if varying & {"patch_size", "global_size", "local_size"}:
        parts.append(view_sizes(p))
    if "n_gpus" in varying:
        parts.append(f'x{p.get("n_gpus", 1)}')
    return " ".join(parts)

def bench_rows(sweep: str, *compare: str) -> tuple[list[dict], list[dict]]:
    """Per-run bench_table rows (config, LSF outcome, training steps, walltime, total compute, loss sanity, speed)
    and loss curve points (one per logged metrics.json step), for a benchmark sweep next to any reference sweeps
    it's compared with. walltime is the training job's LSF wall-clock run time (its job log's resource-usage
    footer), not the sum of GPU-seconds across n_gpus. EFLOP is the benchmark window's TFLOP/s x s/step (FLOPs per
    step, all GPUs; fixed with displace views) x steps trained. A fast run with a NaN loss is broken, not fast
    (patchembed-linear/d1): read status and loss first."""
    sweeps = (sweep, *compare)
    runs = [(sweep, d) for sweep in sweeps for d in run_dirs(sweep)]
    params = {d: saved_params(d) for _, d in runs}
    varying = varying_params(list(params.values()))
    rows, curves = [], []
    for s, d in runs:
        name = f"{sweep_name(s)}/{d.name}" if compare else d.name
        m = read_jsonl(d / "metrics.json")
        tp = [r for r in read_jsonl(d / "performance.json") if r.get("tbl") == "throughput"]
        text = job_text(d, "run")
        exit_code = re.search(r"Exited with exit code (\d+)", text)
        b1 = "illegal memory access" in text
        b2 = "OutOfMemoryError" in text
        status = ("ok" if "Successfully completed" in text else f"exit {exit_code[1]}" if exit_code else "running?") + \
                 (": illegal memory access" if b1 else ": OOM" if b2 else "")
        losses = [r["loss"] for r in m]
        if losses and not all(l == l for l in losses):
            status += ": NaN loss"  # finished fast on garbage (miscompile), not a working run
        t = tp[-1] if tp else {}
        n_gpus = t.get("world_size") or params[d].get("n_gpus", 1)
        config = config_label(params[d], varying)
        steps, eflop = run_compute(d)
        walltime = job_walltime_s(text)
        rows.append({"run": name, "config": config, "status": status, "steps": steps, "EFLOP": eflop,
                     "walltime": fmt_duration(walltime),
                     "finite": all(l == l for l in losses) if losses else None,
                     "loss0": losses[0] if losses else None, "loss_end": sum(losses[-3:]) / len(losses[-3:]) if losses else None,
                     "ktok/s/gpu": t["tokens_per_second"] / n_gpus / 1e3 if t else None,
                     "mfu %": 100 * t["mfu"] if t.get("mfu") else None, "mem GB": t.get("max_mem_gb")})
        curves += [{"run": f"{name} {config}", "step": r["idx_step"], "loss": r["loss"]} for r in m]
    return rows, curves

def scaling_law_budget(d: Path) -> str:
    """This run's nominal compute budget label (c1..c4), from its position in e00/scaling-law's grid: 4 budgets x 5
    sizes, ordered d{budget_index*5 + size_index} (paramsall()'s BUDGETS x SIZES loop order). Sweep-specific, not
    a saved param."""
    return f"c{int(d.name[1:]) // 5 + 1}"

def ng_link(name: str, boxes: dict[str, list[list[int]]] | None = None) -> str:
    """Neuroglancer (Fileglancer) link for an lmd_catalog volume, e.g. em-drosophila-flyem-cns-mito-gt-v6/crop-001_box000:
    its raw image plus every OME-zarr label layer in <volume>.zarr/labels/. Finds labels even when
    VolumeEntry.tracked_by is empty (public GT ingested straight into the zarr, so has_ground_truth says False).
    Reads the label list through the local data root (LMD_DATA_ROOT, mounted); the link uses the cluster path.
    Log in to fileglancer.int.janelia.org in that browser first, or every layer gets a 401 and shows black.
    boxes: {label: [[x0, x1], [y0, y1], [z0, z1]]} in level-0 voxels, drawn as a bounding-box annotation layer;
    the view centres on the first."""
    v = lmd.get(name)
    assert Path(v.path).is_dir(), f"{v.path} not found: mount the data root (LMD_DATA_ROOT)"
    meta = Path(v.path) / "labels" / "zarr.json"
    labels = json.loads(meta.read_text())["attributes"]["ome"]["labels"] if meta.is_file() else []
    path = f"{DEFAULT_DATA_ROOT.rstrip('/')}/{name}.zarr"  # Fileglancer URLs are built from the cluster path
    layers = [{"type": "segmentation", "name": l, "source": to_fileglancer_content_url(path, key=f"labels/{l}", zarr_version=v.zarr_version)}
              for l in labels]
    assert v.voxelsize and v.axes, f"{name} has no voxelsize/axes in the catalog"
    vox = {a: [v.voxelsize[i] * 1e-9, "m"] for i, a in enumerate(v.axes)}  # one voxel, in neuroglancer's units
    if boxes:
        layers.append({"type": "annotation", "name": "boxes",
                       "source": {"url": "local://annotations", "transform": {"outputDimensions": vox}},
                       "annotations": [{"type": "axis_aligned_bounding_box", "id": str(i), "description": label,
                                        "pointA": [b[0] for b in box], "pointB": [b[1] for b in box]}
                                       for i, (label, box) in enumerate(boxes.items())]})
    url = make_neuroglancer_url(path, raw_key=v.image_key, raw_name=name.split("/")[-1], raw_zarr_version=v.zarr_version,
                                additional_layers=layers)
    if boxes:  # centre the view on the first box
        state = parse_neuroglancer_url(url)
        first = next(iter(boxes.values()))
        state |= {"dimensions": vox, "position": [(lo + hi) / 2 for lo, hi in first]}
        url = url.split("#!", 1)[0] + "#!" + quote(json.dumps(state, separators=(",", ":")))
    print(f"{name}: raw + {len(labels)} label layers {labels}" + (f" + boxes {list(boxes)}" if boxes else "") + f"\n{url}")
    return url

def ng_mia_evals_hemibrain() -> str:
    """Hemibrain EB crop-001 with mia-evals' neuron-instance GT boxes (labels/proofread-cell-hemibrain-v1.2) and our
    own EB train/val/test split, to see what each eval scores and whether it overlaps our training data.
    Boxes copied from ~/proj/mia-evals configs/*/data/*.yaml (2026-09-29) and lib/data.py HEMIBRAIN_EB_BOXES."""
    return ng_link("em-drosophila-flyem-hemibrain/crop-001_EllipsoidBody_x24000_y23000_z17000", {
        "mia-evals gary_comparison test": [[4000, 5000], [4000, 5000], [4000, 5000]],  # = our test box
        "mia-evals gary_comparison fit": [[4000, 5000], [4000, 5000], [3000, 4000]],  # inside our val slab
        "mia-evals lmd_ssl_v1 fit": [[1988, 3012], [1988, 3012], [1988, 3012]],  # mostly in our train region
        "ssl-lejepa EB train": [[0, 5000], [0, 5000], [0, 3000]],
        "ssl-lejepa EB val": [[0, 5000], [0, 5000], [3000, 4000]],
    })

def check_runs(root: str = "outdir/e00"):
    """Flag run dirs a job wrote under the wrong name (e.g. the shared-checkout race), or that lack results.

    Conflicts: a row whose params.savedir isn't the dir it sits in, a job log whose run wrote elsewhere, or
    more than one training row in runs.json (two training jobs wrote here; pca/replay rows are fine). Rows from
    before runs.json had "fn" count as training. Several job logs alone are just resubmissions.
    Missing: every run dir needs at least a job log and metrics.json (pending or still-running jobs show up too).
    Different: within a sweep, a run lacking file names (top level, excluding IGNORED) that other runs have.
    """
    flagged = 0
    dirs = sorted(p for p in Path(root).glob("**/d*/") if re.fullmatch(r"d\d+", p.name))
    IGNORED = ["job_*.log", ".DS_Store"]  # expected to differ between runs, or not ours
    names = {d: {f.name for f in d.iterdir() if not any(fnmatch(f.name, g) for g in IGNORED)} for d in dirs}
    sweep_names = defaultdict(set)
    for d in dirs:
        sweep_names[d.parent] |= names[d]
    for d in dirs:
        issues = []
        missing = sweep_names[d.parent] - names[d]
        if missing:
            issues.append(f"lacks {', '.join(sorted(missing))}. ")
        if not any(d.glob("job_*.log")):
            issues.append("no job log")
        if not (d / "metrics.json").is_file():
            issues.append("no metrics.json")
        for r in read_jsonl(d / "performance.json"):
            saved = r.get("params", {}).get("savedir")
            if saved and Path(saved) != d:
                issues.append(f"row for {saved}")
        n = sum(r.get("fn", "run") == "run" for r in read_jsonl(d / "runs.json"))
        if n > 1:
            issues.append(f"{n} training rows in runs.json")
        for log in d.glob("job_*.log"):
            m = re.search(r"input \+ view slices to (\S+)/views", log.read_text(errors="ignore"))
            if m and Path(m.group(1)) != d:
                issues.append(f"{log.name} ran as {m.group(1)}")
        if issues:
            flagged += 1
            print(f"{d}: {'; '.join(sorted(set(issues)))}")
    print(f"{flagged} run dirs flagged under {root}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        pick_entrypoint()
    else:
        call_entrypoint(sys.argv[1], *sys.argv[2:])  # no log_command: local, and outdir/ is the cluster's mirror
