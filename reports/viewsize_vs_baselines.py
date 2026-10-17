"""e00/e06-viewsize-vs-baselines: e05-shifted-target's best masked-token cell (targets from a second view shifted 32 voxels,
a cube of 0.6 of the grid side, pull 10, SIGReg "both") at views of 64, 96, 128 and 160 voxels (8^3 to 20^3 tokens), beside
two references on the same 8 h of one B300 each: an MAE (view 96 of input 128) and LeJEPA's multi-view invariance at its
best view sizes so far (input 128, 2 globals and 4 locals of 96). How does the masked objective depend on how much each
token sees, and how does it compare with the two? Training FLOPs per sample span ~100x across the six runs, so equal hours
are not equal steps: read the probe against compute as well as against step."""

from reports import anticollapse, page

SWEEP = "e00/e06-viewsize-vs-baselines"

# Read from the runs' pca.json, probe.json, metrics.json, performance.json and pca2.png (2026-10-17).
FINDINGS = """<ul>
<li><b>The smallest view is best, and the best encoder of any objective so far.</b> Masked tokens on a 64-voxel view end at
short / long AP 0.205 / 0.847 (random encoder 0.124 / 0.481). The long AP is the highest yet; the short is second only to the
pre-e02 invariance run at 12x1024 (0.216 / 0.764). It is also the cheapest masked run (0.30 TFLOP a sample) and still rising
at 8 h (133k steps).</li>
<li><b>Smaller views win at equal steps and at equal compute.</b> At 40k steps short AP is 0.197 / 0.177 / 0.165 for views
64 / 96 / 128; at ~1.3 EFLOP it is 0.205 for 64, between 0.141 (0.86 EFLOP) and 0.177 (1.7 EFLOP) for 96, and 0.120
for 128. Final short AP: 0.205, 0.189, 0.176 and 0.127
for views 64 to 160 (the larger ones at fewer steps: 57k and 20k). The PCA maps of view 64 follow the neurites, with
membranes a band of their own.</li>
<li><b>The MAE gets there with far less compute, then stalls.</b> 0.173 short at 5k steps (22 PFLOP) and 0.19-0.20 from 10k
steps on (44 PFLOP); masked view 96 reached 0.187 only at 80k steps (3.4 EFLOP). The MAE ends at 0.195 / 0.773, level with masked view
96 on short AP and below it on long (0.823), and below masked view 64 on both. Its AP dips mid-run (0.156 at 160k) and its
effective rank falls from 220 to 75; its top three components are a code of token position alone, the same gradient in
every tile, so the content the probe reads sits further down the spectrum.</li>
<li><b>The invariance objective collapsed onto the tile code.</b> It peaked at 0.146 / 0.570 at 40k steps, then between-tile
variance rose from 0.09 to 0.92 and AP fell below a random encoder's (0.113 short at the end); the residual stream grew to
~2800 (others 130-460). Its PCA maps are one flat colour per tile with the membranes drawn over it. viewsizes-v2's run of
this config reached 0.18 at batch 44 (a SIGReg pull of 4.4 against 3.2 here): the weaker pull is the likeliest difference.</li>
<li>Training was clean: no skipped steps; peak memory 6-165 GB (view 160).</li>
</ul>
<p><b>Next.</b> Masked tokens at view 64 are the objective to take forward. Before a scaling grid: views below 64 (48, 32) to
find where it turns over, and a longer run of view 64 to see where it plateaus. Against the MAE the comparison should be at
matched compute, where the MAE leads early: run both to the same EFLOP rather than the same hours.</p>"""


def label(p: dict) -> str:
    """A trunk's name: its objective and view sizes (input/view, and the locals of "views")."""
    sizes = f"{p['patch_size'][0]}/{p['global_size'][0]}"
    return {"mask": f"masked view {p['global_size'][0]} shift {p['mask_shift']}", "mae": f"MAE {sizes}",
            "views": f"invariance {sizes}/{p['local_size'][0]}"}[p["objective"]]


def main():
    tiles, sections = anticollapse.build(SWEEP, label=label)
    page.write(SWEEP, __doc__, tiles, [("Findings", [("What e06 shows", FINDINGS)]), *sections])


if __name__ == "__main__":
    main()
