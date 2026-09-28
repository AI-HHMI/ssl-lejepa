"""Analysis of e00 results (local): figures, tables and summaries built only from pulled artifacts in outdir/, a
mirror of the cluster's append-only run dirs. Imports no experiment code (experiment, lib/ models: nothing that
builds a Lejepa), only lib.util's entrypoint CLI: artifacts describe themselves (saved params), so every function
works on any past sweep at HEAD.
Figures also go to results/ (local, not committed). Run: uv run python analysis.py <function> [args].
"""

import json
import os
import re
import sys
from pathlib import Path

import pandas
import plotly.express as px
import plotly.graph_objects as go

from lib.util import call_entrypoint, pick_entrypoint


def read_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

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
        n = sum(r.get("fn", "run") == "run" for r in read_jsonl(runs)) if runs.is_file() else 0
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
    params (the first performance.json row with params), savedir and run name. Params a run predates are
    missing (NaN in the table)."""
    rows = []
    run_dirs = [d for d in Path("outdir", sweep).glob("d*/") if re.fullmatch(r"d\d+", d.name)]
    for d in sorted(run_dirs, key=lambda d: int(d.name[1:])):
        if not (d / filename).is_file():
            continue
        perf = d / "performance.json"
        saved = next((r["params"] for r in read_jsonl(perf) if "params" in r), {}) if perf.is_file() else {}
        rows += [{**saved, **r, "savedir": str(d), "run": d.name} for r in read_jsonl(d / filename)]
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

# def plot2(sweep: str):
#     """ktok/s per GPU: one bar per result row, bars grouped by n_gpus with gaps between groups, colored by width + defer_image_ops."""
#     res = load_table(sweep, "performance.json")
#     res["ktok_s_per_gpu"] = res.tokens_per_second / res.n_gpus / 1e3
#     assert len(res), "no performance.json rows for allparams(); run ./pull.sh?"
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
            rows += [{**json.loads(l), "run": f.parent.name, "sweep": sweep} for l in f.read_text().splitlines() if l.strip()]
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
            perf = [json.loads(l) for l in (d / "performance.json").read_text().splitlines() if l.strip()]
            r = [x for x in perf if x["tbl"] == "throughput"][-1]
            total, kernels = profile_device_ms(d / "profile.out")
            fwd = sum(v for k, v in kernels.items() if re.search(ATTN, k) and not re.search(BWD, k))
            bwd = sum(v for k, v in kernels.items() if re.search(ATTN, k) and re.search(BWD, k))
            p = r["params"]
            rows.append({"attention": attention, "sweep": sweep, "mfu %": 100 * r["mfu"],
                         "config": f'{d.name} {p["patch_size"][0]}/{p["global_size"][0]}/{p["local_size"][0]} b{p["batch_size"]}',
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
        ]),
    ]
    scaling = [  # (label, run, GPU type, 1-GPU run whose 8x is ideal)
        ("B300 old · 1 GPU", "ddp/d0", B3, None), ("B300 old · 8 GPUs", "ddp/d3", B3, "ddp/d0"),
        ("H200 · 1 GPU", "width-defer/d3", H2, None), ("H200 · 8 GPUs", "width-defer/d5", H2, "width-defer/d3"),
        ("B300 · 1 GPU", "b300-revisit/d0", B3, None), ("B300 · 8 GPUs", "b300-revisit/d1", B3, "b300-revisit/d0"),
    ]

    def perf(run):  # (total tok/s, n_gpus, mfu or None) from the run's last throughput row
        f = Path("outdir/e00") / run / "performance.json"
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.is_file() else []
        tp = [r for r in rows if r["tbl"] == "throughput"]
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

if __name__ == "__main__":
    if len(sys.argv) == 1:
        pick_entrypoint()
    else:
        call_entrypoint(sys.argv[1], *sys.argv[2:])  # no log_command: local, and outdir/ is the cluster's mirror
