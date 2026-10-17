"""e00/e05-shifted-target: the masked-token objective with its targets from a second view of the same voxels, up to a shift
(16 or 32 voxels: 2 or 4 tokens) off the context's, so the two token grids sit a random, fractional number of tokens apart
and only content lines up between them. SIGReg over both marginals, as e04 (e04-sigreg-both), whose tokens became a code of
the sample and position together. Eight 2 h trunks: shift (16, 32) x mask block (0.3, 0.6) x pull (3.2, 10), every 5k-step
milestone evaluated as saved. Does a linear probe now find content, above the boundary base rate and a random encoder?"""

from reports import anticollapse, page

SWEEP = "e00/e05-shifted-target"

# Read from the runs' pca.json, probe.json, metrics.json and pca2.png (2026-10-16).
FINDINGS = """<ul>
<li><b>Yes: the first masked-token sweep to beat a random encoder.</b> At the end of training six of eight trunks are above a
random encoder's short-range AP (0.124) and all eight above its long-range AP (0.481). The best, block 0.6 / pull 10 / shift
32, reaches 0.146 short and 0.679 long; e04's best was 0.079 / 0.42, and e04 ended at 0.042-0.051 short.</li>
<li><b>AP rises with training in every trunk</b>, the reverse of e02-e04, and is still rising at 2 h: the best gained 0.022
short and 0.12 long from 5k to 20k steps. Effective rank grows with it (to 134-201, e04's 52-116).</li>
<li><b>The PCA maps follow the cells.</b> In the best trunks each neurite's profile is one smooth colour, continuous across
tile seams, with membranes a colour of their own; e04's maps were a lattice inside each tile that ignored the membranes.
Block 0.3 with shift 16 still shows some of each tile's own gradient.</li>
<li><b>More context hidden and a larger shift help; the pull matters less.</b> Final short AP: block 0.6 0.135-0.146, block 0.3
0.107-0.132; shift 32 above shift 16 in all four pairs (by 0.004-0.016). Pull 10 keeps between-tile variance lower (0.03-0.09
against 0.07-0.19 at pull 3.2) but gives no consistent AP gain.</li>
<li><b>Prediction is harder, as it should be.</b> The prediction error stays higher than e04's at the same pull (pull 10:
inv 0.14-0.25, e04 0.08-0.16): the targets can no longer be read off position.</li>
<li><b>Not yet the invariance objective's level.</b> The pre-e02 multi-view runs probed 0.17-0.19 short / 0.64-0.71 long, after
8 h at a larger batch; these are 2 h trunks at batch 32, still improving. Training was clean: no skipped steps, no failed evals.</li>
</ul>
<p><b>Next.</b> Run the best cell (block 0.6, shift 32) for as long as the invariance runs, beside "views" on the same budget,
before choosing the objective for the scaling grid.</p>"""


def main():
    tiles, sections = anticollapse.build(SWEEP)
    page.write(SWEEP, __doc__, tiles, [("Findings", [("What e05 shows", FINDINGS)]), *sections])


if __name__ == "__main__":
    main()
