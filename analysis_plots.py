"""Report cards for e00 results (local): each builder reads pulled run dirs (through analysis.py) and returns a list of
(card title, html fragment) cards, plots_interactive charts and tables, for a reports/<sweep>.py page. Tables also go to
results/<sweep>/*.csv. Imports no experiment code, so every builder works on any past sweep at HEAD. A faceted figure
is one card per facet.
"""

import json
import re
from pathlib import Path

import pandas

import plots_interactive as pi
from analysis import (BOUNDARY_CHANNELS, bench_rows, boundary_ap, config_label, fmt_duration, load_table, probe_runs,
                      probe_stats, probe_walltime_s, probed_dirs, profile_device_ms, read_jsonl, run_compute, run_dirs,
                      saved_params, save_table, scaling_law_budget, short_runs, size_label, sweep_name, tflop_per_step,
                      varying_params, view_sizes)

LINEAR_RANDOM = Path("outdir/e00/probe-test/d1")  # the linear probe on a random 12x512 encoder, mia-evals' probe blocks


def frame(df: pandas.DataFrame) -> str:
    """A DataFrame as a sortable table."""
    return pi.table(list(df.columns), df.values.tolist())

def series_by(df: pandas.DataFrame, key: str, x: str, y: str) -> dict:
    """label -> (xs, ys) for pi.lines: one line per value of df[key], in first-seen order."""
    return {str(k): (g[x].tolist(), g[y].tolist()) for k, g in df.groupby(key, sort=False)}

def loss_curves(sweep: str) -> list:
    """Loss vs step of one sweep: one line per run (and per repeat of a run: repeats append to the same metrics.json,
    each restarting at idx_step 0), labelled with width and batch size."""
    res = load_table(sweep, "metrics.json")
    repeat = (res.idx_step == 0).groupby(res.savedir).cumsum() - 1
    res["label"] = (short_runs(res.savedir) + repeat.map(lambda k: f".{k}" if k else "")
                    + " w" + res.width.astype(str) + " b" + res.batch_size.astype(str))
    return [(f"Loss per run: {sweep}", pi.lines(series_by(res, "label", "idx_step", "loss"), "step", "loss", logy=True))]

def bench_table(sweep: str, *compare: str) -> list:
    """A benchmark sweep's per-run table (analysis.bench_rows), next to any reference sweeps it's compared with.
    Also results/<sweep>/bench.csv."""
    rows, _ = bench_rows(sweep, *compare)
    res = pandas.DataFrame(rows).round(3)
    save_table(res, f"{sweep}/bench")
    return [(f"Runs: {', '.join((sweep, *compare))} (read status and loss first: a fast run with a NaN loss is broken)", frame(res))]

def bench_loss(sweep: str, *compare: str) -> list:
    """Loss vs step for a benchmark sweep, next to any reference sweeps it's compared with: one line per run. A
    broken run is missing (crashed before metrics.json) or diverges from its reference."""
    _, curves = bench_rows(sweep, *compare)
    assert curves, f"no metrics.json in {(sweep, *compare)}; run ./pull.sh?"
    return [("Loss per run (a broken run is missing or diverges from its reference)",
             pi.lines(series_by(pandas.DataFrame(curves), "run", "step", "loss"), "step", "loss", logy=True))]

def bench_speed(sweep: str, *compare: str) -> list:
    """Benchmark throughput per GPU, one bar per run coloured by its LSF status; a crashed run is a zero-length bar."""
    rows, _ = bench_rows(sweep, *compare)
    bars = [(f"{r['run']} {r['config']}" + (f" ({r['mfu %']:.0f}% MFU)" if r["mfu %"] else ""), r["ktok/s/gpu"] or 0.0, r["status"])
            for r in rows]
    return [("Benchmark throughput per GPU, coloured by status", pi.bar(bars, "ktok/s per GPU"))]

def bench(sweep: str, *compare: str) -> list:
    """A benchmark sweep's cards: bench_table, bench_loss and bench_speed."""
    return bench_table(sweep, *compare) + bench_loss(sweep, *compare) + bench_speed(sweep, *compare)

def probe_table(sweep: str) -> list:
    """One row per run with a probe.json: config, the pretraining checkpoint step it probed, how many steps the
    linear probe itself trained for (probe_fit.json, fixed at fit_probe's STEPS unless a run crashed mid-fit) and
    its own walltime (probe_walltime_s: excludes pretraining), test-block boundary AP (mean of the 3 short-range
    channels, then each short and long channel) and mean short-range BCE. Higher AP is better; compare against a
    near-random encoder's run (e.g. probe-test/d1). Also results/<sweep>/probe.csv."""
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
    return [(f"Linear probe per run: {sweep}", frame(res))]

def probe_curves(sweep: str) -> list:
    """The probe's fit curves per run, every 100 steps, one card per metric: training BCE, and BCE and boundary AP on
    held-out test-block tokens (since probe-test). Flat by the end means the fit converged."""
    res = load_table(sweep, "probe_fit.json")
    return [(f"Probe fit: {m}", pi.lines(series_by(res, "run", "step", m), "probe step", m))
            for m in ["loss", "held_bce", "held_boundary_ap"] if m in res]

def probe_vs_compute(sweep: str, *compare: str) -> list:
    """Probe boundary AP vs the probed encoder's pretraining compute, short range (+1 voxel: membranes) and long range
    (+10: same neuron?), one point per probed run of sweep and any compare sweeps, coloured by decoder and sweep. A
    probe-only run (init_from a run dir) sits at its source run's compute; each init_from random baseline is a gray
    line across the plot."""
    runs, random = probe_runs((sweep, *compare))
    varying = varying_params([r[3] for r in runs])
    cards = []
    for r in BOUNDARY_CHANNELS:
        groups = {}
        for sw, d, src, p, _ in runs:
            steps, eflop = run_compute(src)
            groups.setdefault(f"{saved_params(d).get('decoder', 'linear')}, {sweep_name(sw)}", []).append(
                {"EFLOP": eflop, "AP": boundary_ap(probe_stats(d))[r], "run": f"{sweep_name(sw)}/{d.name}",
                 "source": "/".join(src.parts[-2:]), "config": size_label(p, varying), "steps": steps})
        cards.append((f"{r} boundary AP vs pretraining compute", pi.scatter(groups, "EFLOP", "AP", logx=True,
                                                                            hlines={name: ap[r] for name, ap in random.items()})))
    return cards

def probe_short_vs_long(sweep: str, *compare: str) -> list:
    """Linear-probe boundary AP, short range (+1 voxel: membranes) vs long range (+10: same neuron?), one point per
    probed run of sweep and any compare sweeps: do configs that separate membranes also separate neurons? Random-
    encoder baselines are their own points."""
    runs, random = probe_runs((sweep, *compare))
    varying = varying_params([r[3] for r in runs])
    groups = {}
    for sw, d, _, p, _ in runs:
        groups.setdefault(sweep_name(sw), []).append({**boundary_ap(probe_stats(d)), "run": f"{sweep_name(sw)}/{d.name}",
                                                      "config": size_label(p, varying)})
    groups |= {name: [ap] for name, ap in random.items()}
    return [("Boundary AP, short vs long range", pi.scatter(groups, "short (+1)", "long (+10)"))]

def mia_evals_table(sweep: str) -> list:
    """One row per mia-evals record of a sweep (savedir/mia_evals/<task>/records/*.json): the neuron segmentation
    scores of the probe's affinities on the test block. pq (panoptic quality) ranks; VOI split/merge (lower is
    better) and adapted Rand error (ARE) are reported; the size filter was fitted on the fit block. Next to the
    probe's long-range boundary AP. Reference: gary's supervised dinov3 model, pq ~0.11 at 100k steps (cc_threshold).
    Also results/<sweep>/mia_evals.csv."""
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
    assert rows, f"no mia-evals records in outdir/{sweep}/d*/mia_evals/; score first, then ./pull.sh"
    varying = varying_params(list(params.values()))
    res = pandas.DataFrame(rows)
    res.insert(1, "config", [config_label(params[Path("outdir", sweep, r)], varying) for r in res.run])
    save_table(res.round(3), f"{sweep}/mia_evals")
    return [(f"mia-evals neuron instance segmentation: {sweep}", frame(res.round(3)))]

def nanhunt_plot() -> list:
    """NaN hunt: residual norm, grad norm and loss vs step for Adam beta2 0.999 (e00/nanhunt), 0.95
    (e00/nanhunt_beta95), and 0.95 with cuDNN attention off (e00/nanhunt_flash), one card per run (view-size config)
    and metric, one line per sweep. Loss and grad norm are smoothed (rolling mean over 200 steps, kept NaN where the
    raw value is, so lines still end at the first non-finite step). Faster residual growth under 0.95 means earlier
    failure, but failures hit at no fixed norm, and nothing in the curves warns: a marker is each run's first
    non-finite step. The cause was cuDNN's attention backward (replay_bad_batch); nanhunt_flash should have none."""
    rows = []
    for sweep in ["nanhunt", "nanhunt_beta95", "nanhunt_flash"]:
        for f in sorted(Path(f"outdir/e00/{sweep}").glob("d*/metrics.json")):
            rows += [{**r, "run": f.parent.name, "sweep": sweep} for r in read_jsonl(f)]
    assert rows, "no e00/nanhunt* metrics.json; run ./pull.sh?"
    res = pandas.DataFrame(rows).query("tbl == 'metrics'")
    for m in ["grad_norm", "loss"]:
        smooth = res.groupby(["sweep", "run"])[m].transform(lambda v: v.rolling(20, min_periods=1).mean())
        res[m] = smooth.where(res[m].notna())
    fails = res.query("skipped > 0").groupby(["run", "sweep"]).first().reset_index()
    cards = []
    for run in sorted(res.run.unique(), key=lambda r: int(r[1:])):  # d2 d9 d10 d15
        for m in ["resid_norm", "grad_norm", "loss"]:
            series = series_by(res.query("run == @run"), "sweep", "idx_step", m)
            if m != "grad_norm":  # grad norm is NaN at the first non-finite step
                series |= {f"{f['sweep']}: first non-finite": ([f["idx_step"]], [f[m]]) for f in fails.query("run == @run").to_dict("records")}
            cards.append((f"{run}: {m}", pi.lines(series, "step", m, logy=True)))
    return cards

def flash_perf() -> list:
    """Flash vs cuDNN attention on one H200: e00/nanhunt_flash vs nanhunt and nanhunt_beta95 (cuDNN), same 4 configs:
    ktok/s per GPU, labelled with the change vs nanhunt_beta95 (identical but for the kernel), and GPU ms per profiled
    step split into attention forward, attention backward and everything else, from each run's profile.out."""
    ATTN, BWD = r"sdpa|flash|fmha|dot_do_o|convert_dq", r"bprop|bwd|dot_do_o|convert_dq"  # kernel names
    rows = []
    for sweep, attention in [("nanhunt", "cuDNN β2=.999"), ("nanhunt_beta95", "cuDNN"), ("nanhunt_flash", "flash")]:
        for d in sorted(Path(f"outdir/e00/{sweep}").glob("d*/")):
            r = [x for x in read_jsonl(d / "performance.json") if x["tbl"] == "throughput"][-1]
            total, kernels = profile_device_ms(d / "profile.out")
            fwd = sum(v for k, v in kernels.items() if re.search(ATTN, k) and not re.search(BWD, k))
            bwd = sum(v for k, v in kernels.items() if re.search(ATTN, k) and re.search(BWD, k))
            p = r["params"]
            rows.append({"attention": attention, "sweep": sweep, "config": f'{d.name} {view_sizes(p)} b{p["batch_size"]}',
                         "ktok/s": r["tokens_per_second"] / 1e3 / r["world_size"],
                         "attention fwd": fwd, "attention bwd": bwd, "other": total - fwd - bwd})
    assert rows, "no e00/nanhunt* results; run ./pull.sh?"
    rows.sort(key=lambda r: int(r["config"].split()[0][1:]))  # d2 d9 d10 d15
    base = {r["config"]: r["ktok/s"] for r in rows if r["sweep"] == "nanhunt_beta95"}
    speed = [(f'{r["config"]} · {r["attention"]} ({r["ktok/s"] / base[r["config"]] - 1:+.0%})', r["ktok/s"], r["attention"]) for r in rows]
    kernels = [(f'{r["config"]} · {r["attention"]}', r[k], k) for r in rows for k in ["attention fwd", "attention bwd", "other"]]
    return [("Throughput per GPU, change vs cuDNN (config: run input/global/local, batch)", pi.bar(speed, "ktok/s per GPU")),
            ("GPU time per step by kernel group (profile.out)", pi.bar(kernels, "GPU ms per step", stack=True))]

def scaling_law_calib_lr_scan() -> list:
    """Final loss vs lr, one line per model size (e00/scaling-law-calib: 5 sizes x 0.5/1/2x lr, 15 min each): is the
    curve monotonic, and which direction, before trusting main's per-size lr? A run missing from a line OOM'd before
    logging any step (see bench_table's status column)."""
    res = load_table("e00/scaling-law-calib", "metrics.json")
    last = res.sort_values("idx_step").groupby("savedir").last().reset_index().sort_values(["width", "lr"])
    return [("Final loss vs lr, per width", pi.lines(series_by(last, "width", "lr", "loss"), "lr", "final loss", logx=True))]

def scaling_law_calib_loss_vs_flops() -> list:
    """Final loss vs total training compute (EFLOP), one point per run coloured by width (e00/scaling-law-calib). Not an
    IsoFLOP comparison (each run trains 15 min wall-clock, not a matched compute budget): shows where each size
    naturally lands, not yet which size is compute-optimal at a fixed budget (that's the main grid)."""
    groups = {}
    for d in run_dirs("e00/scaling-law-calib"):
        m = read_jsonl(d / "metrics.json")
        if not m:
            continue  # OOM'd before logging any step (xl)
        p = saved_params(d)
        groups.setdefault(f'w{p.get("width")}', []).append({"EFLOP": run_compute(d)[1], "loss": m[-1]["loss"], "run": d.name, "lr": p.get("lr")})
    return [("Final loss vs training compute, per width", pi.scatter(dict(sorted(groups.items(), key=lambda kv: int(kv[0][1:]))),
                                                                       "EFLOP", "loss", logx=True, logy=True))]

def scaling_law_loss_curves() -> list:
    """Loss vs step for e00/scaling-law (the main IsoFLOP grid), one card per nominal compute budget
    (analysis.scaling_law_budget), one line per run: any divergence, and are all 5 sizes in a budget progressing
    normally (similar step counts so far)?"""
    res = load_table("e00/scaling-law", "metrics.json")
    res["budget"] = res.savedir.map(lambda s: scaling_law_budget(Path(s)))
    res["label"] = res.run + " " + res.n_layers.astype(str) + "x" + res.width.astype(str)
    return [(f"Loss vs step, budget {b}", pi.lines(series_by(res.query("budget == @b").sort_values(["n_layers", "idx_step"]), "label", "idx_step", "loss"),
                                                  "step", "loss", logy=True)) for b in [f"c{i}" for i in range(1, 5)]]

def scaling_law_loss_vs_flops() -> list:
    """Loss vs cumulative training compute (EFLOP) for e00/scaling-law, one line per run: the classic loss-vs-compute
    plot, whose lower envelope at any compute value is the compute-optimal size there. Unlike a final-loss comparison
    this stays valid while runs are in progress: read it by slicing vertically at a compute value, not by comparing
    where curves end (a bigger model's curve ends earlier in EFLOP per wall-clock hour, not because it's worse)."""
    res = load_table("e00/scaling-law", "metrics.json")
    res["tflop_per_step"] = res.run.map({d.name: tflop_per_step(d) for d in run_dirs("e00/scaling-law")})
    res = res.dropna(subset=["tflop_per_step"])
    res["EFLOP"] = res.tflop_per_step * (res.idx_step + 1) / 1e6
    res["label"] = res.run + " " + res.n_layers.astype(str) + "x" + res.width.astype(str)
    return [("Loss vs cumulative training compute", pi.lines(series_by(res.sort_values(["n_layers", "EFLOP"]), "label", "EFLOP", "loss"),
                                                            "EFLOP", "loss", logx=True, logy=True))]

def scaling_law_lr_vs_flops() -> list:
    """lr vs training compute so far (EFLOP), one point per run in e00/scaling-law coloured by depth: does the main
    grid's per-size lr (calibrated in e00/scaling-law-calib) track depth sensibly? Each size's lr is fixed across its
    4 budgets, so expect 4 points per colour at different EFLOP (more compute reached at looser budgets)."""
    groups = {}
    for d in run_dirs("e00/scaling-law"):
        eflop = run_compute(d)[1]
        if eflop is None:
            continue  # OOM'd before logging a throughput row (xl)
        p = saved_params(d)
        groups.setdefault(f'{p.get("n_layers")} layers', []).append({"EFLOP": eflop, "lr": p.get("lr"), "run": d.name,
                                                                    "width": p.get("width"), "budget": scaling_law_budget(d)})
    return [("lr vs training compute, per depth", pi.scatter(groups, "EFLOP", "lr", logx=True, logy=True))]

def scaling_law_isoflop() -> list:
    """Current loss vs model width, one line per nominal compute budget: the classic IsoFLOP curve, whose U-shaped
    minimum is the compute-optimal size at that budget. Only a fair comparison once every size in a budget has reached
    similar progress (bench_table's steps/status); while runs are in progress, prefer scaling_law_loss_vs_flops."""
    rows = []
    for d in run_dirs("e00/scaling-law"):
        m = read_jsonl(d / "metrics.json")
        if not m:
            continue  # not logged a step yet (still starting up, or crashed/OOM'd first)
        rows.append({"width": saved_params(d).get("width"), "budget": scaling_law_budget(d), "loss": m[-1]["loss"]})
    res = pandas.DataFrame(rows).sort_values(["budget", "width"])
    return [("Current loss vs width, per compute budget (IsoFLOP)", pi.lines(series_by(res, "budget", "width", "loss"), "width", "loss",
                                                                            logx=True, logy=True))]

def scaling_law_tile_code_vs_probe() -> list:
    """Between-tile variance (scaling-law-pca/d*/pca.json: the share of token variance that is just each 96^3 view's
    own code) vs the linear probe's boundary AP on the same checkpoint (its scaling-law source run's probe.json), one
    card per range, coloured by model size. The loss sees only the token mean, so nothing keeps tokens local; do
    checkpoints whose tokens collapse onto their view's code (bigger, longer trained) separate boundaries worse? The
    random 12x512 encoder pairs scaling-law-pca's init_from random run with probe-test/d1's linear probe. Card titles
    give Spearman rank correlations, overall and within size. Also results/e00/scaling-law-pca/tile_code_vs_probe.csv."""
    rows, random_tile = [], None
    for d in run_dirs("e00/scaling-law-pca"):
        pca = read_jsonl(d / "pca.json")
        assert pca, f"{d}: no pca.json; run ./pull.sh?"
        src = Path(saved_params(d).get("init_from", ""))
        if src.name == "random":
            random_tile = pca[-1]["between_tile_variance"]
            continue
        assert (src / "probe.json").is_file(), f"{src}: no probe.json, the linear probe of {d}'s checkpoint"
        p = saved_params(src)
        rows.append({"run": d.name, "source": src.name, "size": f'{p["n_layers"]}x{p["width"]}', "width": p["width"],
                     "budget": scaling_law_budget(src), "EFLOP": run_compute(src)[1], "between_tile": pca[-1]["between_tile_variance"],
                     "effective_rank": pca[-1]["effective_rank"], **boundary_ap(probe_stats(src))})
    assert random_tile is not None, "no init_from random run in e00/scaling-law-pca"
    res = pandas.DataFrame(rows).sort_values(["width", "budget"])
    save_table(res.round(3), "e00/scaling-law-pca/tile_code_vs_probe")
    random_ap = boundary_ap(probe_stats(LINEAR_RANDOM))
    cards = []
    for r in BOUNDARY_CHANNELS:
        within = ", ".join(f"{s} {g.between_tile.corr(g[r], method='spearman'):+.2f}" for s, g in res.groupby("size", sort=False))
        groups = {s: [{"between_tile": x["between_tile"], "AP": x[r], "run": x["run"], "source": x["source"], "budget": x["budget"]}
                      for x in g.to_dict("records")] for s, g in res.groupby("size", sort=False)}
        groups["random 12x512"] = [{"between_tile": random_tile, "AP": random_ap[r]}]
        title = f"Tile code vs {r} AP: Spearman all {res.between_tile.corr(res[r], method='spearman'):+.2f}; within size {within}"
        cards.append((title, pi.scatter(groups, "between_tile", "AP")))
    return cards
