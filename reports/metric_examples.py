"""e00/metric-examples: instance-segmentation metrics on a real crop with one kind of mistake added at a time. The crop, its
eight corrupted labellings and their scores (PQ, SQ, RQ, VOI split and merge, adapted Rand error, and boundary AP of the hard
labelling) come from the top-level metric_examples.py, run on the cluster; this builds the page. Which mistake costs which metric?"""

import json
from pathlib import Path

import plots_interactive as pi
from reports import page

SWEEP = "e00/metric-examples"
OUT = Path("outdir") / SWEEP
METRICS = ["pq", "sq", "rq", "voi_split", "voi_merge", "adapted_rand_error", "ap_short", "ap_long"]

# Read from metrics.json (2026-10-27): one 128^3 crop, 15 objects, one mistake at a time.
FINDINGS = """<ul>
<li><b>PQ and RQ count objects; VOI and Rand error count voxels.</b> 300 specks cost PQ 0.999 to 0.091 (RQ 0.091: each speck is a false
object) and VOI 0.02, Rand error 0.001. Deleting the smaller half of the objects costs PQ 0.636 (RQ) and Rand error 0.001. This is
why the size filter decides PQ in mws.toml while VOI barely moves. SQ is 0.999 and 1.0 in both: the matched objects are exact.</li>
<li><b>One big merge or split is a small mistake to PQ and a large one to VOI.</b> Cutting the largest object in two: PQ 0.955 (one object of
15 is wrong), VOI split 0.27, Rand error 0.095. Merging it with its neighbour: PQ 0.935, VOI merge 0.66, Rand error 0.299. VOI names the
direction (split or merge, never both here); PQ and Rand error do not. Rand error and VOI weight a mistake by the voxels it touches.</li>
<li><b>Shattering is the worst for everything:</b> PQ 0.082 (RQ 0.096: no piece overlaps its object by IoU above 0.5; SQ 0.849 for the few that
do), VOI split 4.37, Rand error 0.917.</li>
<li><b>Boundary placement shows in SQ and in short-range AP, little elsewhere.</b> Every object shifted 2 voxels: Rand error 0.056 and
VOI 0.34 / 0.25, but PQ 0.692 (SQ 0.798, and RQ 0.867 as small objects drop under IoU 0.5) and short-range AP 0.046 (the +1 pairs
are all at the boundaries) against long-range AP 0.655. A 1-voxel gap between objects: PQ 0.709, short AP 0.213, long AP 0.734.
Long AP is the more tolerant: its +10 pairs straddle boundaries by a margin.</li>
<li><b>AP here is of hard labellings</b> (the probe's soft predictions would score differently). Specks barely move it (short 0.933, long
0.986); deleting objects costs short AP 0.466 against long 0.934 (an erased voxel has no same-object pairs at +1, and few at +10).</li>
</ul>
<p><b>Caveats.</b> One crop, a few large objects: a 1 µm³ crop has 15 objects, so one mistake is a large share of its PQ; the
eight mistakes are synthetic, a real segmentation makes several at once; the crop is of the hemibrain EB test block, which the
probe is scored on.</p>"""


def main():
    rows = json.loads((OUT / "metrics.json").read_text())
    table = pi.table(["mistake", *METRICS], [[r["case"], *[f"{r[k]:.3f}" for k in METRICS]] for r in rows])
    images = [(f"{r['case']}: {r['what']} Columns: EM | truth | labelling | boundaries (green both, red missed, blue extra); z {r['z']}",
               pi.img(OUT / f"{r['case'].replace(' ', '_')}.png", page.out_path(SWEEP).parent, style="max-width:100%")) for r in rows]
    page.write(SWEEP, __doc__, [(str(rows[0]["objects"]), "objects in the crop")], [("Findings", [("What each metric penalizes, in this crop", FINDINGS)]), ("Metrics by mistake", [("Higher is better for pq, sq, rq and AP; lower for VOI and adapted Rand error", table)]), ("Examples", images)])


if __name__ == "__main__":
    main()
