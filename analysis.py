"""Analysis of e00 results (local): figures, tables and summaries built only from pulled artifacts in outdir/, a
mirror of the cluster's append-only run dirs. Imports no experiment code (experiment, lib/ models: nothing that
builds a Lejepa), only lib.util's entrypoint CLI: artifacts describe themselves (saved params), so every function
works on any past sweep at HEAD.
Figures and tables also go to results/ (local, not committed).
Each experiment has one entrypoint, named after its sweep (e00/b300-compile -> e00_b300_compile), that makes all of
its figures and tables: uv run python analysis.py e00_b300_compile. Cross-experiment pages: perf_journey.
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
import plotly.express as px
import plotly.graph_objects as go
from lmd_catalog.catalog import DEFAULT_DATA_ROOT
from lmd_catalog.viewers import make_neuroglancer_url, parse_neuroglancer_url, to_fileglancer_content_url

from lib.util import call_entrypoint, pick_entrypoint

BOUNDARY_CHANNELS = {"short (+1)": ["(1, 0, 0)", "(0, 1, 0)", "(0, 0, 1)"], "long (+10)": ["(10, 0, 0)", "(0, 10, 0)", "(0, 0, 10)"]}


def read_jsonl(path) -> list[dict]:
    """JSON-lines rows of `path`, or [] if it doesn't exist (a run dir missing a file it predates, or hasn't
    written yet)."""
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []

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

def run_dirs(sweep: str) -> list[Path]:
    """outdir/<sweep>/dN/ run dirs in numeric order."""
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

def save_table(df: pandas.DataFrame, name: str):
    """Print a table and keep a copy at results/<name>.csv."""
    path = Path("results") / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"{name}:\n{df.to_string(index=False)}\n")

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

def show(fig, name: str):
    """Display a plotly figure and keep a copy at results/<name>.html, e.g. name = e00/nanhunt_plot."""
    path = Path("results") / f"{name}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path)
    fig.show()

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

def load_table(sweep: str, filename: str) -> pandas.DataFrame:
    """JSON-lines rows of `filename` from every run dir outdir/<sweep>/dN/, each joined with that run's own saved
    params (saved_params), savedir and run name. Params a run predates are missing (NaN in the table)."""
    rows = []
    for d in run_dirs(sweep):
        rows += [{**saved_params(d), **r, "savedir": str(d), "run": d.name} for r in read_jsonl(d / filename)]
    assert rows, f"no {filename} in outdir/{sweep}/d*/; run ./pull.sh?"
    return pandas.DataFrame(rows)

def short_runs(savedirs):
    """savedir relative to the sweep's common prefix, e.g. 'outdir/e00/x/d3/' -> 'd3'."""
    prefix = os.path.commonpath(list(savedirs))
    return savedirs.str[len(prefix):].str.strip("/")

def loss_curves(sweep: str):
    """Loss curves of one sweep, e.g. e00/nanhunt_flash: one line per run (and per repeat of a run)."""
    res = load_table(sweep, "metrics.json")
    # Repeats append to the same metrics.json; each restarts at idx_step 0.
    repeat = (res.idx_step == 0).groupby(res.savedir).cumsum() - 1
    res["run"] = short_runs(res.savedir) + repeat.map(lambda k: f".{k}" if k else "")
    res["sizes"] = res.patch_size.astype(str) + " " + res.global_size.astype(str) + " " + res.local_size.astype(str)
    show(px.line(res, x="idx_step", y="loss", color="width", line_dash="batch_size",
                 hover_data=["width", "batch_size", "queue"], markers=True, log_y=True,
                 category_orders={"batch_size": sorted(res.batch_size.unique())}), f"{sweep}/loss_curves")

def run_compute(d: Path) -> tuple[int | None, float | None]:
    """(training steps, total EFLOP) of a run dir: the last logged step in metrics.json + 1, times FLOPs per step from
    the benchmark window (TFLOP/s x s/step, all GPUs; fixed with displace views). None where a file is missing."""
    m = read_jsonl(d / "metrics.json")
    tp = [r for r in read_jsonl(d / "performance.json") if r.get("tbl") == "throughput"]
    steps = m[-1]["idx_step"] + 1 if m else None
    t = tp[-1] if tp else {}
    tflop_per_step = t["tflops_per_second"] * t["seconds_per_step"] if t.get("tflops_per_second") else None
    return steps, tflop_per_step * steps / 1e6 if tflop_per_step and steps else None

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

def varying_params(params: list[dict]) -> set:
    """Param names whose values differ between runs (runs without saved params are left out)."""
    known = [p for p in params if p]
    return {k for k in dict.fromkeys(k for p in known for k in p) if len({json.dumps(p.get(k)) for p in known}) > 1}

def sweep_name(sweep: str) -> str:
    """The sweep's short name, e.g. 'e00/viewsizes-v2' -> 'viewsizes-v2'."""
    return sweep.split("/")[-1]

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

def bench_table(sweep: str, *compare: str) -> pandas.DataFrame:
    """A benchmark sweep's per-run table (bench_rows), next to any reference sweeps it's compared with.
    Output: results/<sweep>/bench.csv."""
    rows, _ = bench_rows(sweep, *compare)
    res = pandas.DataFrame(rows).round(3)
    save_table(res, f"{sweep}/bench")
    return res

def bench_loss(sweep: str, *compare: str):
    """Loss vs step for a benchmark sweep, next to any reference sweeps it's compared with: one line per run. A
    broken run is missing (crashed before metrics.json) or diverges from its reference.
    Output: results/<sweep>/bench_loss.html."""
    sweeps = (sweep, *compare)
    _, curves = bench_rows(sweep, *compare)
    assert curves, f"no metrics.json in {sweeps}; run ./pull.sh?"
    show(px.line(pandas.DataFrame(curves), x="step", y="loss", color="run", log_y=True,
                 title=f"Loss per run: {', '.join(sweeps)} (a broken run is missing or diverges from its reference)"), f"{sweep}/bench_loss")

def bench_speed(sweep: str, *compare: str):
    """Benchmark throughput per GPU for a sweep, next to any reference sweeps it's compared with: one bar per run.
    Crashed runs draw as zero-length bars, so their status still shows (text outside the bar end).
    Output: results/<sweep>/bench_speed.html."""
    sweeps = (sweep, *compare)
    rows, _ = bench_rows(sweep, *compare)
    res = pandas.DataFrame(rows).round(3)
    speed = res.assign(label=res["run"] + " " + res["config"], x=res["ktok/s/gpu"].fillna(0),
                       text=res["mfu %"].map(lambda v: f"{v:.0f}% MFU" if v == v else "").where(res.status == "ok", res.status))
    fig = px.bar(speed, x="x", y="label", orientation="h", text="text", color="status",
                 title=f"Benchmark throughput per GPU: {', '.join(sweeps)}")
    fig.update_traces(textposition="outside", cliponaxis=False).update_yaxes(title="", autorange="reversed")
    fig.update_xaxes(title="ktok/s per GPU", range=[0, 1.15 * max(speed.x.max(), 1)])  # room for the labels
    show(fig, f"{sweep}/bench_speed")

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

def size_label(p: dict) -> str:
    """'<patch>/<global>/<local> w<width>' view-size + width label from saved params, or '?' if missing."""
    return f'{view_sizes(p)} w{p["width"]}' if p else "?"

def probe_table(sweep: str) -> pandas.DataFrame:
    """One row per run with a probe.json: config, the pretraining checkpoint step it probed, how many steps the
    linear probe itself trained for (probe_fit.json, fixed at fit_probe's STEPS unless a run crashed mid-fit) and
    its own walltime (probe_walltime_s: excludes pretraining), test-block boundary AP (mean of the 3 short-range
    channels, then each short and long channel) and mean short-range BCE. Higher AP is better; compare against a
    near-random encoder's run (e.g. probe-test/d1)."""
    dirs = probed_dirs(sweep)
    assert dirs, f"no probe.json in outdir/{sweep}/d*/; run ./pull.sh?"
    params = {d: saved_params(d) for d in dirs}
    varying = varying_params(list(params.values()))
    rows = []
    for d in dirs:
        st = probe_stats(d)
        fit = read_jsonl(d / "probe_fit.json")
        aps = {k.removeprefix("boundary_ap_"): v for k, v in st.items() if k.startswith("boundary_ap_(")}
        bce = [v for k, v in st.items() if k.startswith("bce_(")][:3]
        walltime = probe_walltime_s(d)
        rows.append({"run": d.name, "config": config_label(params[d], varying) + f' steps={params[d].get("steps_per_epoch")}',
                     "probe steps": fit[-1]["step"] + 1 if fit else None,
                     "walltime": fmt_duration(walltime), "boundary AP short": st["boundary_ap_short"],
                     **{f"AP {k}": v for k, v in aps.items()}, "BCE short": sum(bce) / len(bce)})
    res = pandas.DataFrame(rows).round(3)
    save_table(res, f"{sweep}/probe")
    return res

def probe_vs_compute(sweep: str, *compare: str):
    """Probe boundary AP vs total training compute, one point per probed run of sweep and any compare sweeps, for
    short-range (+1 voxel: membranes) and long-range (+10: same neuron?) affinities. A probe-only run (init_from a run
    dir) is placed at its source run's compute; init_from random is the dashed baseline in each panel."""
    points, random = [], {}
    for sw in (sweep, *compare):
        for d in probed_dirs(sw):
            ap = boundary_ap(probe_stats(d))
            src, p, init = probe_source(d)
            if init == "random":
                random = ap
                continue
            steps, eflop = run_compute(src)
            label = size_label(p) + (f' x{p["n_gpus"]}' if p.get("n_gpus", 1) > 1 else "")
            points += [{"range": r, "AP": v, "EFLOP": eflop, "steps": steps, "run": f"{sweep_name(sw)}/{d.name}", "config": label,
                        "source": "/".join(src.parts[-2:]), "sweep": sweep_name(sw) if not init else "/".join(src.parts[-2:-1])} for r, v in ap.items()]
    assert points, f"no probe.json in {(sweep, *compare)}; run ./pull.sh?"
    res = pandas.DataFrame(points)
    res["label"] = res.config.where(res.sweep != sweep_name(sweep), "")  # name only the reference models; hover the rest
    fig = px.scatter(res, x="EFLOP", y="AP", color="sweep", facet_col="range", text="label", log_x=True,
                     hover_data=["config", "run", "source", "steps"], category_orders={"range": list(BOUNDARY_CHANNELS)},
                     title=f"Linear-probe boundary AP vs training compute: {', '.join((sweep, *compare))}")
    fig.update_traces(textposition="top center", textfont_size=10).update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    lo, hi = res.EFLOP.min() * 0.8, res.EFLOP.max() * 1.25
    for col, r in enumerate(BOUNDARY_CHANNELS, start=1):
        if r in random:  # dashed line across the panel
            fig.add_scatter(x=[lo, hi], y=[random[r]] * 2, mode="lines", line=dict(dash="dash", color="gray"),
                            name="random encoder", showlegend=col == 1, row=1, col=col)
    show(fig, f"{sweep}/probe_vs_compute")

def probe_short_vs_long(sweep: str, *compare: str):
    """Scatter of linear-probe boundary AP, short-range (+1 voxel: membranes) vs long-range (+10: same neuron?), one
    point per probed run of sweep and any compare sweeps: do configs that separate membranes also separate neurons?
    init_from random is the dashed crosshair baseline."""
    points, random = [], {}
    for sw in (sweep, *compare):
        for d in probed_dirs(sw):
            ap = boundary_ap(probe_stats(d))
            _, p, init = probe_source(d)
            row = {**ap, "run": f"{sweep_name(sw)}/{d.name}", "config": size_label(p), "sweep": sweep_name(sw)}
            if init == "random":
                random = row
            else:
                points.append(row)
    assert points, f"no probe.json in {(sweep, *compare)}; run ./pull.sh?"
    res = pandas.DataFrame(points)
    fig = px.scatter(res, x="short (+1)", y="long (+10)", color="sweep", text="config", hover_data=["run"],
                     title=f"Linear-probe boundary AP, short vs long range: {', '.join((sweep, *compare))}")
    fig.update_traces(textposition="top center", textfont_size=10)
    if random:
        fig.add_scatter(x=[random["short (+1)"]], y=[random["long (+10)"]], mode="markers",
                        marker=dict(symbol="x", size=12, color="black"), name="random encoder")
    show(fig, f"{sweep}/probe_short_vs_long")

def mia_evals_table(sweep: str) -> pandas.DataFrame:
    """One row per mia-evals record of a sweep (experiment.score: savedir/mia_evals/<task>/records/*.json): the neuron
    segmentation scores of the probe's affinities on the test block. pq (panoptic quality) ranks; VOI split/merge
    (lower is better) and adapted Rand error (ARE) are reported; the size filter was fitted on the fit block. Next to
    the probe's long-range boundary AP. Reference: gary's supervised dinov3 model, pq ~0.11 at 100k steps (cc_threshold)."""
    rows, params = [], {}
    for d in run_dirs(sweep):
        pj = read_jsonl(d / "probe.json")
        st = pj[-1] if pj else {}
        for f in sorted(d.glob("mia_evals/*/records/*.json")):
            r = json.loads(f.read_text())
            v = r["scores"]["voxel_instance"]
            params[d] = saved_params(d)
            rows.append({"run": d.name, "route": r["route"], "pq": v["pq"], "voi_split": v["voi_split"], "voi_merge": v["voi_merge"],
                         "ARE": v["adapted_rand_error"], "instances": f'{int(v["instances_predicted"])}/{int(v["instances_truth"])}',
                         "postprocess": r["postprocess"]["describe"],
                         "probe AP long": sum(st.get(f"boundary_ap_{c}", float("nan")) for c in BOUNDARY_CHANNELS["long (+10)"]) / 3})
    assert rows, f"no mia-evals records in outdir/{sweep}/d*/mia_evals/; score first (experiment.scorelsf), then ./pull.sh"
    varying = varying_params(list(params.values()))
    res = pandas.DataFrame(rows)
    res.insert(1, "config", [config_label(params[Path("outdir", sweep, r)], varying) for r in res.run])
    save_table(res.round(3), f"{sweep}/mia_evals")
    return res

def probe_curves(sweep: str):
    """The probe's fit curves per run, every 100 steps: training BCE, and BCE and boundary AP on held-out test-block
    tokens. Flat by the end means the fit converged; AP is the binary boundary-vs-same-object metric."""
    res = load_table(sweep, "probe_fit.json")
    metrics = [m for m in ["loss", "held_bce", "held_boundary_ap"] if m in res]  # held_* since the held-out eval (probe-test)
    long = res.melt(id_vars=["step", "run"], value_vars=metrics, var_name="metric")
    fig = px.line(long, x="step", y="value", color="run", facet_row="metric", height=250 * len(metrics),
                  category_orders={"metric": metrics}, title=f"Linear probe fit: {sweep}")
    fig.update_yaxes(matches=None, title="").for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    show(fig, f"{sweep}/probe_fit")

# def plot2(sweep: str):
#     """ktok/s per GPU: one bar per result row, bars grouped by n_gpus with gaps between groups, colored by width + defer_image_ops."""
#     res = load_table(sweep, "performance.json")
#     res["ktok_s_per_gpu"] = res.tokens_per_second / res.n_gpus / 1e3
#     assert len(res), "no performance.json rows for paramsall(); run ./pull.sh?"
#     # Bar label: short run name, plus a suffix for repeated rows in one run.
#     repeat = res.groupby("savedir").cumcount()
#     res["run"] = short_runs(res.savedir) + repeat.map(lambda k: f".{k}" if k else "")
#     res["color"] = "width=" + res.width.astype(str) + ", defer=" + res.defer_image_ops.astype(str)
#     res = res.sort_values(["n_gpus", "color", "run"]).reset_index(drop=True)
#     # x positions: consecutive within a group, GROUP_GAP extra slots between groups.
#     GROUP_GAP = 0.8
#     group_idx = res.n_gpus.rank(method="dense").astype(int) - 1
#     res["x"] = res.index + GROUP_GAP * group_idx
#     fig = go.Figure()
#     for color, r in res.groupby("color", sort=False):
#         fig.add_bar(x=r.x, y=r.ktok_s_per_gpu, name=str(color), width=0.9)
#     for g, r in res.groupby("n_gpus"):
#         fig.add_annotation(x=r.x.mean(), y=-0.12, yref="paper", text=f"<b>{g} gpu</b>", showarrow=False)
#     fig.update_xaxes(tickvals=res.x, ticktext=res.run)
#     fig.update_layout(yaxis_title="ktok/s per GPU", legend_title="", margin=dict(b=80))
#     fig.show()

# def table(sweep: str):
#     res = load_table(sweep, "performance.json")
#     trace = load_table(sweep, "trace_summary.json")
#     # Host ms per profiled step in each phase (see lib.util.trace_summary).
#     phases = {"01_DATA_IO_ms": "io ms", "04_FORWARD_AND_LOSS_ms": "fwd ms", "05_BACKWARD_ms": "bwd ms", "06_OPTIMIZER_ms": "opt ms"}
#     for k in ["gpu_busy", "step_ms", *phases]:
#         res[k] = res.savedir.map(dict(zip(trace.savedir, trace[k]))) if k in trace else float("nan")
#     cols = {
#         "savedir": "run",
#         "tbl": "result",  # "throughput", or "oom" for runs that ran out of GPU memory
#         # "views": "views",
#         # "patch_size": "input",
#         # "global_size": "global",
#         # "local_size": "local",
#         # "compile": "compile",
#         # "cudagraphs": "cudagraphs",
#         # "width": "width",
#         # "defer_image_ops": "defer",
#         # "compile_blocks": "blocks",
#         # "grad_compress": "compress",
#         # "batch_views": "batch views",
#         "n_gpus": "gpus",
#         # "batch_size": "batch",
#         # "n_workers": "workers",
#         "gpu_busy": "gpu busy %",
#         "samples_per_second": "samples/s",
#         "tokens_per_second": "tok/s",
#         "tflops_per_second": "TFLOP/s",
#         "mfu": "mfu %",
#         "max_mem_gb": "mem GB",
#         "input_mvox_per_second": "Mvox/s",
#         "step_ms": "prof step ms",
#         **phases,
#     }
#     for k in cols:  # older runs predate some columns
#         if k not in res:
#             res[k] = float("nan")
#     res = res[list(cols)].rename(columns=cols) # type: ignore
#     res["gpu busy %"] *= 100
#     res["mfu %"] *= 100
#     res["TFLOP/s"] /= res["gpus"]
#     res = res.rename(columns={"TFLOP/s": "TFLOP/s/gpu"})
#     res["tok/s"] /= 1e3
#     res.insert(list(res.columns).index("tok/s") + 1, "ktok/s/gpu", res["tok/s"] / res["gpus"])
#     res = res.rename(columns={"tok/s": "ktok/s"}).round(1)
#     print(res.to_string(index=False))
#     return res

def nanhunt_plot():
    """NaN hunt: residual norm, grad norm and loss vs step for Adam beta2 0.999 (e00/nanhunt), 0.95
    (e00/nanhunt_beta95), and 0.95 with cuDNN attention off (e00/nanhunt_flash).

    Faster residual growth under 0.95 means earlier failure, but failures hit at no fixed norm, and nothing in the
    curves warns: an x marks each run's first non-finite step, after which grad norm is NaN (the line ends).
    The cause was cuDNN's attention backward (replay_bad_batch); nanhunt_flash should have no x.
    Color = sweep; columns = run (view-size config), rows = metric. Loss and grad norm are smoothed.
    """
    rows = []
    for sweep in ["nanhunt", "nanhunt_beta95", "nanhunt_flash"]:
        for f in sorted(Path(f"outdir/e00/{sweep}").glob("d*/metrics.json")):
            rows += [{**r, "run": f.parent.name, "sweep": sweep} for r in read_jsonl(f)]
    assert rows, "no e00/nanhunt* metrics.json; run ./pull.sh?"
    res = pandas.DataFrame(rows).query("tbl == 'metrics'")
    metrics = ["resid_norm", "grad_norm", "loss"]
    # Rolling mean over 20 logged points (200 steps) per run; kept NaN where the raw value is, so lines still end
    # at the first non-finite step.
    for m in ["grad_norm", "loss"]:
        smooth = res.groupby(["sweep", "run"])[m].transform(lambda v: v.rolling(20, min_periods=1).mean())
        res[m] = smooth.where(res[m].notna())
    long = res.melt(id_vars=["idx_step", "run", "sweep"], value_vars=metrics, var_name="metric")
    # print(res)
    # return
    # x=idx_step, y="value", color="sweep", yfacet="run"
    # fig = px.line(long, x="idx_step", y="value", color="run", line_dash="sweep", facet_row="metric", log_y=True, height=900)
    runs = sorted(res.run.unique(), key=lambda r: int(r[1:]))  # d2 d9 d10 d15
    fig = px.line(long, x="idx_step", y="value", color="sweep", facet_col="run", facet_row="metric", log_y=True, height=900,
                  category_orders={"run": runs, "metric": metrics})
    # One y range per metric row, shared across the run columns. px numbers facet rows from the bottom.
    fig.update_yaxes(title="")
    fig.update_xaxes(title="step", row=1)
    for row in range(1, len(metrics) + 1):
        first = []  # this row's col-1 y axis, e.g. layout name "yaxis3" -> trace ref "y3"
        fig.for_each_yaxis(lambda a: first.append(a.plotly_name.replace("axis", "")), row=row, col=1)
        fig.update_yaxes(matches=first[0], row=row)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fails = res.query("skipped > 0").groupby(["run", "sweep"]).first().reset_index()
    for i, metric in enumerate(metrics):
        if metric == "grad_norm":
            continue  # NaN at the first non-finite step
        for col, run in enumerate(runs, start=1):
            f = fails[fails.run == run]
            # Scattergl like px's (WebGL) lines: WebGL draws above all SVG traces, so an SVG marker would hide.
            fig.add_trace(go.Scattergl(x=f.idx_step, y=f[metric], mode="markers", name="first non-finite step",
                                       marker=dict(symbol="x", size=11, color="black"), text=f.sweep,
                                       showlegend=i == 0 and col == 1), row=len(metrics) - i, col=col)
    fig.update_layout(title="NaN hunt: Adam beta2 0.999 (nanhunt) vs 0.95 (nanhunt_beta95) vs 0.95 + flash attention (nanhunt_flash)")
    show(fig, "e00/nanhunt_plot")

def flash_perf():
    """Flash vs cuDNN attention on one H200: e00/nanhunt_flash vs nanhunt and nanhunt_beta95 (cuDNN), same 4 configs.

    Fig 1: ktok/s per GPU, labelled with the change vs nanhunt_beta95 (identical but for the kernel).
    Fig 2: GPU ms per profiled step, split into attention forward, attention backward and everything else,
    from each run's profile.out.
    """
    ATTN, BWD = r"sdpa|flash|fmha|dot_do_o|convert_dq", r"bprop|bwd|dot_do_o|convert_dq"  # kernel names
    rows = []
    for sweep, attention in [("nanhunt", "cuDNN β2=.999"), ("nanhunt_beta95", "cuDNN"), ("nanhunt_flash", "flash")]:
        for d in sorted(Path(f"outdir/e00/{sweep}").glob("d*/")):
            r = [x for x in read_jsonl(d / "performance.json") if x["tbl"] == "throughput"][-1]
            total, kernels = profile_device_ms(d / "profile.out")
            fwd = sum(v for k, v in kernels.items() if re.search(ATTN, k) and not re.search(BWD, k))
            bwd = sum(v for k, v in kernels.items() if re.search(ATTN, k) and re.search(BWD, k))
            p = r["params"]
            rows.append({"attention": attention, "sweep": sweep, "mfu %": 100 * r["mfu"],
                         "config": f'{d.name} {view_sizes(p)} b{p["batch_size"]}',
                         "ktok/s per GPU": r["tokens_per_second"] / 1e3 / r["world_size"],
                         "attention fwd": fwd, "attention bwd": bwd, "other": total - fwd - bwd})
    assert rows, "no e00/nanhunt* results; run ./pull.sh?"
    res = pandas.DataFrame(rows)
    res = res.iloc[res.config.map(lambda c: int(c.split()[0][1:])).argsort(kind="stable")]  # d2 d9 d10 d15
    base = res[res.sweep == "nanhunt_beta95"].set_index("config")["ktok/s per GPU"]
    res["vs cuDNN"] = (res["ktok/s per GPU"] / res.config.map(base) - 1).map(lambda x: f"{x:+.0%}")
    show(px.bar(res, x="config", y="ktok/s per GPU", color="attention", barmode="group", text="vs cuDNN", hover_data=["mfu %"],
                title="Throughput: flash vs cuDNN attention, one H200 (config = input/global/local, batch)"), "e00/flash_perf_throughput")
    long = res.melt(id_vars=["config", "attention"], value_vars=["attention fwd", "attention bwd", "other"],
                    var_name="kernels", value_name="GPU ms per step")
    fig = px.bar(long, x="attention", y="GPU ms per step", color="kernels", facet_col="config",
                 title="GPU time per step by kernel group (profile.out)")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(title="")
    show(fig, "e00/flash_perf_kernels")

def perf_journey():
    """Write the run-derived numbers in results/perf_journey.html: its steps / phases / scaling arrays, between the
    BEGIN/END perf_journey() markers in its <script>. Tiles and prose there are still hand-written.

    Bars are listed here by hand, per phase (a phase = one GPU type and setup). Each bar's ktok/s per GPU, total
    tok/s and MFU come from its run's last throughput row in performance.json (run ./pull.sh first). Its change
    label compares it with the `vs` run: same GPU count -> "+14%" (or "x3.1" at 2x+); 8 GPUs vs 1 -> per-GPU
    scaling efficiency, "94%".
    """
    B3, H2 = "B300", "H200"
    phases = [  # (label, [(bar label with | line breaks, run, GPU type, vs run, notes)])
        ("B300 · flyliconn, basic views", [
            ("fp32", "compile-amp-tok_s/d0", B3, None, "fp32, batch 42, 4 workers, eager. Attention on fp32 mem-efficient kernel."),
            ("+ bf16", "compile-amp-tok_s/d5", B3, "compile-amp-tok_s/d0", "bf16 autocast (SIGReg fp32), batch 84. cuDNN flash attention."),
            ("+ compile", "compile-amp-tok_s/d7", B3, "compile-amp-tok_s/d5", "torch.compile(dynamic=True) on the encoder. Now data-loader bound (~170 samples/s)."),
            ("+ 8|workers", "ddp/d0", B3, "compile-amp-tok_s/d7", "8 DataLoader workers: data wait gone, 88% GPU busy (≈ workers/d1)."),
            ("DDP", "ddp/d3", B3, "ddp/d0", "8×B300 with DDP."),
        ]),
        ("H200 · hemibrain EB 128³, displace views", [
            ("new|setup", "cudagraphs/d0", H2, None, "H200, hemibrain EB 128³, displace 96³/64³, compile(dynamic), workers active through profile."),
            ("+ CUDA|graphs", "cudagraphs/d4", H2, "cudagraphs/d0", "compile(mode='reduce-overhead'): replay recorded kernel sequences. 99% GPU busy."),
            ("+ batch|views", "batchviews/d1", H2, "cudagraphs/d4", "2 encoder calls per step (all globals, all locals) instead of 6."),
            ("+ uint8", "width-defer/d3", H2, "batchviews/d1", "defer_image_ops: workers ship uint8, GPU normalizes."),
            ("+ Linear|embed", "patchembed-linear/d2", H2, "width-defer/d3", "Patch embedding as reshape + one Linear instead of Conv3d."),
            ("DDP", "cudagraphs/d7", H2, "cudagraphs/d4", "8×H200 with CUDA graphs, 72 cores."),
            ("+ full|node", "allreduce/d2", H2, "allreduce/d0", "12 cores/GPU → all 96 cores. bf16 grad compression: no further gain."),
            ("+ batch|views", "batchviews/d3", H2, "batchviews/d1", "Batched encoder calls at 8 GPUs."),
            ("+ uint8", "width-defer/d5", H2, "width-defer/d3", "uint8 transfer at 8 GPUs."),
        ]),
        ("B300 · H200 setup", [
            ("H200|setup", "b300-revisit/d0", B3, None, "Everything from the H200 phase, on one B300. Conv3d patch embed ~15–17% of GPU time."),
            ("DDP", "b300-revisit/d1", B3, "b300-revisit/d0", "8×B300."),
            ("+ Linear|embed", "patchembed-linear/d0", B3, "b300-revisit/d0", "Patch embedding as reshape + one Linear (cuDNN attention)."),
            ("safe|stack", "cudagraph-fix/d3", B3, "patchembed-linear/d0", "What training runs now: flash attention (cuDNN's "
             "backward returned NaN grads, nanhunt_flash) and the patch embed outside the compiled graph (its Triton kernel "
             "miscompiled under cudagraphs, b300-compile). Nearly all of the cost is flash."),
        ]),
    ]
    scaling = [  # (label, run, GPU type, 1-GPU run whose 8x is ideal)
        ("B300 old · 1 GPU", "ddp/d0", B3, None), ("B300 old · 8 GPUs", "ddp/d3", B3, "ddp/d0"),
        ("H200 · 1 GPU", "width-defer/d3", H2, None), ("H200 · 8 GPUs", "width-defer/d5", H2, "width-defer/d3"),
        ("B300 · 1 GPU", "b300-revisit/d0", B3, None), ("B300 · 8 GPUs", "b300-revisit/d1", B3, "b300-revisit/d0"),
    ]

    def perf(run):  # (total tok/s, n_gpus, mfu or None) from the run's last throughput row
        f = Path("outdir/e00") / run / "performance.json"
        tp = [r for r in read_jsonl(f) if r["tbl"] == "throughput"]
        assert tp, f"no throughput row in {f}; run ./pull.sh?"
        r = tp[-1]
        return r["tokens_per_second"], r.get("world_size") or r["params"].get("n_gpus", 1), r.get("mfu")  # pre-DDP: 1 GPU

    steps = []
    for _, bars in phases:
        for k, run, hw, vs, d in bars:
            tok, g, mfu = perf(run)
            step = {"k": k, "g": g, "v": round(tok / g / 1e3), "hw": hw, "run": run,
                    "d": d + (f" {tok / 1e6:.2f}M tok/s total." if g > 1 else "") + (f" {100 * mfu:.1f}% MFU." if mfu else "")}
            if vs:
                vs_tok, vs_g, _ = perf(vs)
                r = (tok / g) / (vs_tok / vs_g)
                step["x"] = f"{r:.0%}" if g != vs_g else f"×{r:.1f}" if r >= 2 else f"{r - 1:+.0%}"
                if g != vs_g:  # the 1-GPU run this bar's scaling % is against, drawn as an outline behind it
                    step["ref"] = round(vs_tok / vs_g / 1e3)
            steps.append(step)
    nodes = []
    for k, run, hw, vs in scaling:
        tok, g, _ = perf(run)
        nodes.append({"k": k, "v": round(tok / 1e6, 3), "hw": hw, "run": run} | ({"ideal": round(g * perf(vs)[0] / 1e6, 3)} if vs else {}))

    path = Path("results/perf_journey.html")
    html = path.read_text()
    BEGIN, END = "// BEGIN perf_journey() data", "// END perf_journey() data"
    assert html.count(BEGIN) == 1 and html.count(END) == 1, f"{path} needs one {BEGIN!r} ... {END!r} block"
    rows = lambda xs: "[\n" + ",\n".join("  " + json.dumps(x, ensure_ascii=False) for x in xs) + ",\n]"
    data = (f"{BEGIN}: written by analysis.py perf_journey(); edit the lists there, not here.\n"
            f"const steps = {rows(steps)};\n"
            f"// One label per phase; a new phase starts wherever the GPU type changes.\n"
            f"const phases = {json.dumps([label for label, _ in phases], ensure_ascii=False)};\n"
            f"const scaling = {rows(nodes)};\n")
    path.write_text(html[:html.index(BEGIN)] + data + html[html.index(END):])
    print(f"Wrote {len(steps)} bars and {len(nodes)} node results to {path}")

# One entrypoint per experiment (sweep e00/<name> -> e00_<name>, - -> _): all of its figures and tables.

def e00_nanhunt_flash():
    """cuDNN attention's NaN gradients vs flash (nanhunt, nanhunt_beta95, nanhunt_flash), and flash's throughput cost."""
    nanhunt_plot()
    flash_perf()
    loss_curves("e00/nanhunt_flash")

def e00_b300_compile():
    """Is torch.compile broken on B300 for the Linear patch embed at batch 64? (cudagraphs crash or miscompile)"""
    bench_table("e00/b300-compile")
    bench_loss("e00/b300-compile")
    bench_speed("e00/b300-compile")

def e00_cudagraph_fix():
    """Do cudagraphs work with PatchEmbed3d out of the compiled graph (eager_patch_embed)? Compare with b300-compile."""
    bench_table("e00/cudagraph-fix", "e00/b300-compile")
    bench_loss("e00/cudagraph-fix", "e00/b300-compile")
    bench_speed("e00/cudagraph-fix", "e00/b300-compile")

def e00_b300_train8h_dynamic():
    """First long 8xB300 runs, width 512 vs 1024, dynamic compile."""
    bench_table("e00/b300-train8h-dynamic")
    bench_loss("e00/b300-train8h-dynamic")
    bench_speed("e00/b300-train8h-dynamic")
    loss_curves("e00/b300-train8h-dynamic")

def e00_probe_test():
    """Linear affinity probe on a known-good trained encoder (d0: viewsizes/d0, 8 h) vs a random one (d1), probe only:
    fit curves (training BCE, held-out BCE and boundary AP) and test-block boundary AP per channel.
    Images: outdir/e00/probe-test/d*/probe.png (EM | true boundaries | predicted boundaries)."""
    probe_table("e00/probe-test")
    probe_curves("e00/probe-test")

def e00_viewsizes_v2():
    """View-size study with the fixed stack: training loss, probe scores per config (d0-d16 and repeats d17-d33)."""
    # bench_table("e00/viewsizes-v2")
    bench_loss("e00/viewsizes-v2")
    # bench_speed("e00/viewsizes-v2")
    # loss_curves("e00/viewsizes-v2") ## TOO SLOW
    # probe_table("e00/viewsizes-v2") ## good
    # probe_curves("e00/viewsizes-v2") ## info sparse
    # probe_vs_compute("e00/viewsizes-v2", "e00/probe-test") ## only useful to show that FLOPS does not explain performance.
    # probe_short_vs_long("e00/viewsizes-v2", "e00/probe-test") ## good

if __name__ == "__main__":
    if len(sys.argv) == 1:
        pick_entrypoint()
    else:
        call_entrypoint(sys.argv[1], *sys.argv[2:])  # no log_command: local, and outdir/ is the cluster's mirror
