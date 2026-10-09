"""e00/unetr-probe: a UNETR decoder (skip connections into the encoder's own blocks) vs the linear probe, on the encoders
of scaling-law's budget-c4 runs and on random encoders. AP is the average precision of ranking different-object voxel
pairs first, per affinity channel: short range (+1 voxel, membranes; 2% of pairs are boundaries) and long range (+10
voxels; 17-22%), each the mean of its three axes, on the held-out EB test block. Higher is better."""

import statistics
from pathlib import Path

import plots_interactive as pi
from analysis import boundary_ap, probe_source, probe_stats, probe_walltime_s, probed_dirs, run_compute, run_dirs
from analysis_plots import LINEAR_RANDOM, mia_evals_table, probe_curves, probe_vs_compute
from reports import page

SWEEP = "e00/unetr-probe"
# The earlier linear probes: every probed scaling-law run, and probe-test d1 (random encoder), d2/d3
# (b300-train8h-dynamic d0/d1: 12x512 / 12x1024, 8 h).
COMPARE = ("e00/scaling-law", "e00/probe-test/d1", "e00/probe-test/d2", "e00/probe-test/d3")
SIZES = {(4, 256): "xs", (6, 384): "s", (12, 512): "m", (16, 768): "l"}  # (n_layers, width)


def run_info(d: Path) -> dict:
    """A probed unetr-probe run beside the linear probe on the same encoder (its source scaling-law run, or the random
    baseline)."""
    src, sp, init = probe_source(d)
    random = init == "random"
    name = SIZES[(sp["n_layers"], sp["width"])]
    kind = "pretrained" if not random else "random, encoder " + ("frozen" if sp.get("unetr_freeze_encoder", True) else "trained")
    steps, eflop = (None, None) if random else run_compute(src)
    walltime = probe_walltime_s(d)
    assert walltime, f"{d}: no probe job log; run ./pull.sh?"
    return dict(dir=d, run=d.name, name=name, kind=kind, label=f"{name} ({src.name})" if not random else f"{name}, {kind}",
                source="-" if random else f"{src.parent.name}/{src.name}", steps=steps, eflop=eflop, minutes=walltime / 60,
                unetr=boundary_ap(probe_stats(d)), linear=boundary_ap(probe_stats(LINEAR_RANDOM if random else src)))


def ap_bars(runs, key: str) -> tuple:
    """UNETR bar for every run, a linear bar beside each pretrained one, and one linear bar for the random encoder."""
    rows = [bar for r in runs for bar in [(f"{r['label']}: UNETR", r["unetr"][key], "UNETR")]
            + ([(f"{r['label']}: linear", r["linear"][key], "Linear")] if r["kind"] == "pretrained" else [])]
    rows.append(("random encoder: linear", boundary_ap(probe_stats(LINEAR_RANDOM))[key], "Linear"))
    return (f"{key} boundary AP", pi.bar(rows, "boundary AP"))


def main():
    runs = [run_info(d) for d in probed_dirs(SWEEP)]
    frozen = next(r for r in runs if r["kind"] == "random, encoder frozen")
    pre = [r for r in runs if r["kind"] == "pretrained"]
    best = max(pre, key=lambda r: r["unetr"]["short (+1)"])
    tiles = [
        (f"{len(runs)}/{len(run_dirs(SWEEP))}", "runs probed"),
        (f"{best['unetr']['short (+1)']:.3f}", f"best pretrained short AP ({best['name']})"),
        (f"{frozen['unetr']['short (+1)']:.3f}", "random frozen encoder, short AP"),
        (f"{sum(r['unetr']['short (+1)'] > frozen['unetr']['short (+1)'] for r in pre)}/{len(pre)}", "pretrained beat random (short AP)"),
        (f"{statistics.median(r['minutes'] for r in runs):.0f}", "min per probe (median)"),
    ]
    table = pi.table(
        ["Run", "Model", "Source", "Pretrain steps", "EFLOP", "UNETR short AP", "UNETR long AP", "Linear short AP", "Linear long AP", "Probe min"],
        [[r["run"], r["label"] if r["kind"] != "pretrained" else r["name"], r["source"], r["steps"], r["eflop"],
          r["unetr"]["short (+1)"], r["unetr"]["long (+10)"], r["linear"]["short (+1)"], r["linear"]["long (+10)"], r["minutes"]] for r in runs])
    images = pi.images({r["label"]: {"probe.png": [r["dir"] / "probe.png"]} for r in runs}, base=page.out_path(SWEEP).parent)
    page.write(SWEEP, __doc__, tiles, [
        ("Results", [("Runs (linear = the linear probe on the same encoder; random rows: the random linear baseline)", table),
                     ap_bars(runs, "short (+1)"), ap_bars(runs, "long (+10)")]),
        ("Compute", probe_vs_compute(SWEEP, *COMPARE)),
        ("Scoring", mia_evals_table(SWEEP)),
        ("Probe fit", probe_curves(SWEEP)),
        ("Predictions", [("Each run's probe.png (test block)", images)]),
        ("Cost", [("Probe job walltime, minutes (startup, fit, test inference, artifacts)", pi.bar([(r["label"], r["minutes"], "probe") for r in runs], "minutes"))]),
    ])


if __name__ == "__main__":
    main()
