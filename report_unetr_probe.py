"""One-page HTML report of e00/unetr-probe (a UNETR decoder vs the linear probe on the scaling-law encoders and random
encoders), in lmd-catalog's report style (report_page.py). From pulled artifacts in outdir/ only.
Usage: uv sync --extra analysis && uv run python report_unetr_probe.py  ->  results/e00/unetr-probe/report.html
"""

import json
import math
import statistics
from pathlib import Path

import plots
import matplotlib.pyplot as plt  # after plots, which picks the Agg backend
import report_page
from analysis import (boundary_ap, probe_runs, probe_source, probe_stats, probe_walltime_s, probed_dirs, read_jsonl, run_compute,
                      run_dirs, saved_params, sweep_name)

SWEEP = "e00/unetr-probe"
LINEAR_RANDOM = Path("outdir/e00/probe-test/d1")  # the linear probe on a random encoder, same probe blocks
# The earlier linear probes, as in analysis.e00_unetr_probe's probe_vs_compute: every probed scaling-law run, and
# probe-test d1 (random encoder), d2/d3 (b300-train8h-dynamic d0/d1: 12x512 / 12x1024, 8 h).
COMPARE = ("e00/scaling-law", "e00/probe-test/d1", "e00/probe-test/d2", "e00/probe-test/d3")
OUT = Path("results") / SWEEP / "report.html"
SIZES = {(4, 256): "xs", (6, 384): "s", (12, 512): "m", (16, 768): "l"}  # (n_layers, width)
INTRO = """UNETR decoder (skip connections into the encoder's own blocks) vs the linear probe, on the encoders of
scaling-law's budget-c4 runs and on random encoders. AP is the average precision of ranking different-object voxel pairs
first, per affinity channel: short range (+1 voxel, membranes; 2% of pairs are boundaries) and long range (+10 voxels;
17-22%), each the mean of its three axes, on the held-out EB test block. Higher is better."""


def run_info(d: Path) -> dict:
    """A probed unetr-probe run beside the linear probe on the same encoder (its source scaling-law run, or the random
    baseline), and its mia-evals record if it has been scored."""
    p = saved_params(d)
    src, sp, init = probe_source(d)
    random = init == "random"
    name = SIZES[(sp["n_layers"], sp["width"])]
    kind = "pretrained" if not random else "random, encoder " + ("frozen" if p.get("unetr_freeze_encoder", True) else "trained")
    steps, eflop = (None, None) if random else run_compute(src)
    walltime = probe_walltime_s(d)
    assert walltime, f"{d}: no probe job log; run ./pull.sh?"
    records = sorted(d.glob("mia_evals/*/records/*.json"))
    return dict(
        run=d.name, name=name, kind=kind, label=f"{name} ({src.name})" if not random else f"{name}, {kind}",
        source="-" if random else f"{src.parent.name}/{src.name}", steps=steps, eflop=eflop, minutes=walltime / 60,
        unetr=boundary_ap(probe_stats(d)), linear=boundary_ap(probe_stats(LINEAR_RANDOM if random else src)),
        fit=read_jsonl(d / "probe_fit.json"), score=json.loads(records[-1].read_text())["scores"]["voxel_instance"] if records else None,
    )


def ap_bars(runs, linear_random, key: str) -> str:
    """UNETR bar for every run, a linear bar beside each pretrained one, and one linear bar for the random encoder."""
    rows = []
    for r in runs:
        rows.append((f"{r['label']}: UNETR", r["unetr"][key], "UNETR"))
        if r["kind"] == "pretrained":
            rows.append((f"{r['label']}: linear", r["linear"][key], "Linear"))
    rows.append(("random encoder: linear", linear_random[key], "Linear"))
    return plots.group_bar(rows, "boundary AP")


def ap_vs_compute(key: str) -> str:
    """analysis.probe_vs_compute as an SVG: boundary AP vs the encoder's pretraining compute, one point per probed run of
    SWEEP and COMPARE, coloured by decoder and sweep; each random-encoder baseline is a gray line across the plot."""
    runs, random = probe_runs((SWEEP, *COMPARE))
    groups = {}
    for sw, d, src, _, _ in runs:
        eflop = run_compute(src)[1]
        assert eflop, f"{src}: no compute (metrics.json, performance.json); run ./pull.sh?"
        xs, ys = groups.setdefault(f"{saved_params(d).get('decoder', 'linear')}, {sweep_name(sw)}", ([], []))
        xs.append(math.log10(eflop))
        ys.append(boundary_ap(probe_stats(d))[key])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for (label, (xs, ys)), c in zip(groups.items(), plots.COLORS):
        ax.scatter(xs, ys, s=14, alpha=0.6, color=c, label=label)
    for (name, ap), ls in zip(random.items(), ["--", ":", "-."]):
        ax.axhline(ap[key], color=plots.GREY, ls=ls, lw=1, label=name)
    ax.set_xlabel("log10 pretraining EFLOP")
    ax.set_ylabel(f"{key} boundary AP")
    ax.legend(**plots.LEGEND_RIGHT)
    return plots.to_svg(fig)


def build() -> tuple:
    """Return (tiles, sections): tiles are (number, label); sections are (title, [(card title, svg or table html)])."""
    runs = [run_info(d) for d in probed_dirs(SWEEP)]
    linear_random = boundary_ap(probe_stats(LINEAR_RANDOM))
    frozen = next(r for r in runs if r["kind"] == "random, encoder frozen")
    pre = [r for r in runs if r["kind"] == "pretrained"]
    best = max(pre, key=lambda r: r["unetr"]["short (+1)"])
    scored = [r for r in runs if r["score"]]
    tiles = [
        (f"{len(runs)}/{len(run_dirs(SWEEP))}", "runs probed"),
        (f"{best['unetr']['short (+1)']:.3f}", f"best pretrained short AP ({best['name']})"),
        (f"{frozen['unetr']['short (+1)']:.3f}", "random frozen encoder, short AP"),
        (f"{sum(r['unetr']['short (+1)'] > frozen['unetr']['short (+1)'] for r in pre)}/{len(pre)}", "pretrained beat random (short AP)"),
        (f"{statistics.median(r['minutes'] for r in runs):.0f}", "min per probe (median)"),
        (f"{len(scored)}/{len(runs)}", "scored with mia-evals"),
    ]
    dash = lambda v, f: "—" if v is None else f.format(v)
    table = plots.table(
        ["Run", "Model", "Source", "Pretrain steps", "EFLOP", "UNETR short AP", "UNETR long AP", "Linear short AP", "Linear long AP", "Probe min"],
        [[r["run"], r["label"] if r["kind"] != "pretrained" else r["name"], r["source"], r["steps"] or "—", r["eflop"] or "—",
          f"{r['unetr']['short (+1)']:.3f}", f"{r['unetr']['long (+10)']:.3f}", f"{r['linear']['short (+1)']:.3f}",
          f"{r['linear']['long (+10)']:.3f}", f"{r['minutes']:.1f}"] for r in runs],
    )
    score_table = plots.table(
        ["Run", "Model", "PQ", "VOI split", "VOI merge", "Adapted Rand error"],
        [[r["run"], r["label"], *(dash(r["score"] and r["score"][k], "{:.3f}") for k in ["pq", "voi_split", "voi_merge", "adapted_rand_error"])] for r in runs],
    )
    fits = {r["label"]: ([row["step"] for row in r["fit"]], [row["held_boundary_ap"] for row in r["fit"]]) for r in runs}
    sections = [
        ("Results", [
            ("Runs (linear = linear probe on the same encoder; random rows compare with the random linear baseline)", table),
            ("Short-range boundary AP (+1 voxel: membranes)", ap_bars(runs, linear_random, "short (+1)")),
            ("Long-range boundary AP (+10 voxels: same neuron?)", ap_bars(runs, linear_random, "long (+10)")),
        ]),
        ("Compute", [
            ("Short-range boundary AP vs pretraining compute, beside the earlier linear probes", ap_vs_compute("short (+1)")),
            ("Long-range boundary AP vs pretraining compute, beside the earlier linear probes", ap_vs_compute("long (+10)")),
        ]),
        ("Probe fit", [("Held-out boundary AP vs probe step (every 100 steps; monitoring only)", plots.lines(fits, "held-out boundary AP"))]),
        ("Cost", [("Probe job walltime (minutes, whole job: startup, fit, test inference, artifacts)", plots.bar({r["label"]: r["minutes"] for r in runs}, "minutes"))]),
        ("Scoring", [("mia-evals neuron instance segmentation of the probe's affinities (test block; — = not scored)", score_table)]),
    ]
    return tiles, sections


def main():
    report_page.write(OUT, f"{SWEEP} report", INTRO, *build())


if __name__ == "__main__":
    main()
