"""e00/e09-data-transfer: does pretraining on one kind of data transfer to others, and does pretraining on all of them at
once cost anything? mia-evals' lmd_ssl_v1 families (one per modality pair): fly EM (hemibrain), zebrafish EM (fish2), mouse
ExM (LICONN) and mouse cortex EM (Kasthuri15, whose data are all eval volumes: its encoder pretrained on MICrONS). One
encoder per family and one on all four (lmd4, families weighted equally), all at the same compute: e07-mask-side's cube of
side 3 (view 96, shift 32), 4 h of m (12x512) at batch 32 on one B300. Each probed on every family (the linear boundary
probe, fitted on the family's fit block, scored on its test blocks), beside a random encoder; every volume read at 8 nm."""

import json

import plots_interactive as pi
from analysis_plots import trunk_loss_curves
from analysis_utils import BOUNDARY_CHANNELS, boundary_ap, probe_stats, run_dirs, run_params, trunk_metrics
from reports import page

SWEEP = "e00/e09-data-transfer"
ENCODERS = {"hemibrain_wide": "hemibrain", "fish2": "fish2", "liconn_mouse": "LICONN", "microns": "MICrONS", "lmd4": "joint (lmd4)"}
SETS = {"hemibrain_eb": "hemibrain EB", "fish2": "fish2", "liconn_mouse": "LICONN", "kasthuri15": "Kasthuri15"}
RANDOM = "random"

# Read from the runs' probe.json (each probe_<set>/), metrics.json and probe.png (2026-10-20).
FINDINGS = """<p><b>Read the long-range AP, not the short.</b> The probe scores voxel pairs only where both voxels are labelled,
which suits hemibrain's dense labels but not the others, where only some cells are labelled: almost every real boundary
there is a labelled cell against an unlabelled neighbour, and is left out. Short-range (+1 voxel) boundaries are then 0.1%
of the scored pairs on fish2 and LICONN and 0.5-1% on Kasthuri (2% on hemibrain), every encoder's short AP sits at that
base rate, and the probe learns to predict no boundary at all (its probe.png maps are black). The long range (+10 voxels,
80 nm) still has 7-18% boundaries on fish2 and Kasthuri and 1.2% on LICONN.</p>
<ul>
<li><b>The joint encoder costs almost nothing at the same compute.</b> It is best or tied on fish2 (0.268), LICONN (0.118)
and Kasthuri15 (0.356), and second on hemibrain (0.656 against 0.691): spreading the same 4 h over four families did not
dilute it.</li>
<li><b>Pretraining on the family itself helped only for hemibrain and LICONN.</b> The LICONN encoder beats the other
single-family ones on LICONN (0.112 against 0.074-0.078); but on fish2 the fish2 encoder is beaten by the hemibrain one
(0.242 against 0.268), and on Kasthuri the MICrONS encoder is the worst but for random.</li>
<li><b>EM transfers between EM families.</b> The hemibrain and fish2 encoders lie within ~0.05 of each other on all three
EM sets, both well above random on hemibrain (0.64-0.69 against 0.48) and Kasthuri (0.34 against 0.24).</li>
<li><b>The MICrONS encoder learned close to nothing:</b> at random on hemibrain (0.476 against 0.481) and LICONN, below it
on fish2. Its input is very low contrast (a sample's std ~0.03 after /255), and it had the one large gradient spike (229).</li>
<li>Training was clean: 59-67k steps in 4 h, no skipped steps, no errors.</li>
</ul>
<p><b>Caveats.</b> Side 3 is the setting that probed near random in e07, so these are weak encoders; one seed each. fish2's
labels are stored at 16 x 16 x 30 nm and upsampled to 8 nm.</p>
<p><b>Next.</b> Score labelled-against-unlabelled pairs as boundaries on the sparsely labelled sets (their labels are whole
cells: proofread, or manual "final"), skipping only pairs unlabelled on both sides, and re-probe these checkpoints (e09-01:
24 probe jobs, no training). Then repeat with side 7-8, and with MICrONS clipped to its own intensity range.</p>"""


def existing(path) -> list:
    """An images cell: path, or nothing (a dash) where a job wrote none."""
    return [path] if path.is_file() else []


def probes() -> dict:
    """(encoder, set) -> probe.json row: each trunk's probe_<set>/ and each random encoder's own dir (an Eval: its set in
    its eval.json)."""
    out = {}
    for d in run_dirs(SWEEP):
        p = run_params(d)
        if p["init_from"] == "random":
            out[(RANDOM, SETS[json.loads((d / "eval.json").read_text())["probe_on"]])] = probe_stats(d)
        else:
            for s, name in SETS.items():
                if (d / f"probe_{s}" / "probe.json").is_file():
                    out[(ENCODERS[p["data"]], name)] = probe_stats(d / f"probe_{s}")
    return out


def build() -> tuple:
    """Return (tiles, sections) for page.write."""
    rows = probes()
    trunks = [d for d in run_dirs(SWEEP) if run_params(d)["init_from"] == ""]
    encoders, sets = [*ENCODERS.values(), RANDOM], list(SETS.values())
    ap = {k: boundary_ap(q) for k, q in rows.items()}
    base = {k: {r: sum(q[f"boundary_frac_{c}"] for c in cs) / 3 for r, cs in BOUNDARY_CHANNELS.items()} for k, q in rows.items()}
    long, short = "long (+10)", "short (+1)"
    gain = {(e, s): ap[(e, s)][long] - ap[(RANDOM, s)][long] for e, s in ap if e != RANDOM and (RANDOM, s) in ap}
    best = {s: max((e for e in ENCODERS.values() if (e, s) in ap), key=lambda e: ap[(e, s)][long]) for s in sets if any((e, s) in ap for e in ENCODERS.values())}
    tiles = [(f"{len(trunks)}", "encoders trained"), (f"{len(rows)}", "probes"),
             *((f"{ap[(best[s], s)][long]:.3f}", f"best long AP on {s}: {best[s]}") for s in best)]
    heat = lambda z, title, zr: (title, pi.heatmap({(s, e): v for (e, s), v in z.items()}, sets, encoders[::-1], "probed on", "trained on", zr))
    sections = [
        ("Findings", [("What e09 shows", FINDINGS)]),
        ("All vs all", [
            heat({k: v[long] for k, v in ap.items()}, "Long-range (+10) boundary AP: rows the encoder's pretraining data, columns the family probed", (0, 0.7)),
            heat(gain, "Long-range AP minus the random encoder's on the same family", (-0.1, 0.25)),
            heat({k: v[short] for k, v in ap.items()}, "Short-range (+1) boundary AP (meaningful on hemibrain EB only: see Findings)", (0, 0.2)),
            ("Every probe: AP and the boundary base rate (a probe that finds nothing) per range",
             pi.table(["trained on", "probed on", "short AP", "short base", "long AP", "long base", "long AP - random"],
                      [[e, s, ap[(e, s)][short], base[(e, s)][short], ap[(e, s)][long], base[(e, s)][long], gain.get((e, s))]
                       for e in encoders for s in sets if (e, s) in ap])),
        ]),
        ("Loss trajectories", trunk_loss_curves({ENCODERS[run_params(d)["data"]]: trunk_metrics(d) for d in trunks}, ["loss", "inv", "sigreg", "resid_norm"])),
        ("Probe maps", [("Rows the encoder, columns the family: EM | true boundaries (white: boundary or unlabelled) | predicted, test block's middle z",
                         pi.images({ENCODERS[run_params(d)["data"]]: {name: existing(d / f"probe_{s}" / "probe.png") for s, name in SETS.items()} for d in trunks},
                                   base=page.out_path(SWEEP).parent))]),
        ("PCA maps", [("Each encoder on the held-out hemibrain EB crop (pca2.png: EM | PCA | minus tile mean | per tile | token norm)",
                       pi.images({ENCODERS[run_params(d)["data"]]: {"EB crop": existing(d / "pca2.png")} for d in trunks}, base=page.out_path(SWEEP).parent))]),
    ]
    return tiles, sections


def main():
    page.write(SWEEP, __doc__, *build())


if __name__ == "__main__":
    main()
