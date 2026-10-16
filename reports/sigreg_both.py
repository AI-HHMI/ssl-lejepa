"""e00/e04-sigreg-both: the masked-token objective with SIGReg over each view's tokens and over each position's batch at once.
Alone, the first let a code of position pass (e02-maskloss-sweep) and the second a code of the sample (e03-anticollapse);
together neither passes. Eight 2 h trunks: mask block (0.3, 0.6) x pull (lamb x the batch, each term's: 1, 3.2, 10, 32),
every 5k-step milestone evaluated as saved. Does a linear probe now find content, above the boundary base rate and a random
encoder? In the PCA maps a position code is a lattice identical on every crop, and a sample code one flat colour per crop."""

from reports import anticollapse, page

SWEEP = "e00/e04-sigreg-both"

# Read from the runs' pca.json, probe.json, metrics.json and job logs (2026-10-16). The first sweep whose milestones the trunk
# evaluated itself (experiment.eval_milestone).
FINDINGS = """<ul>
<li><b>No.</b> No run's probe comes near a random encoder (short / long AP 0.124 / 0.481). The best is 0.079 / 0.42
(block 0.6, pull 3.2, step 5k), against a base rate of 0.020 / 0.186. That beats e03's best (0.063, stop-grad) and e02's
(0.033), but the encoder still knows less about boundaries than random weights.</li>
<li><b>AP falls as training goes on, in all eight trunks.</b> Each peaks at its first milestone (5k steps: 0.053-0.079) and ends
at 0.042-0.051. Whatever the objective rewards, it moves the tokens away from local content, not toward it.</li>
<li><b>The two shortcuts went, as designed, but a third took their place: a code of the sample and the position together.</b>
The batch term removes each crop's own mean token once its pull is 10 or more (between-tile variance 0.003-0.024, against
0.1-0.6 at pull 1 and 3.2, where the sample code partly survives). The tokens do not collapse (effective rank 52-116). But the
PCA maps are a lattice inside every tile whose colours and phase change from tile to tile, and that lattice ignores the
membranes under it. Tokens like f(sample, position) pass both terms: within a view they spread over positions, and at a
position they spread over samples. The masked tokens are then predictable from the visible ones, which carry the same
sample code, without looking at the image. SIGReg constrains marginals, and this code has the right ones.</li>
<li><b>Pull trades prediction for Gaussianity, with no effect on AP.</b> At pull 10 and 32 SIGReg sits near its floor
(1.14-1.17) and the prediction error stays higher (inv 0.08-0.15); at pull 1 inv reaches 0.004-0.011 and SIGReg 1.4-2.3. Final
AP spans only 0.042-0.051 across all eight, with no order by pull or block.</li>
<li><b>Training was stable but drifting.</b> No skipped steps. The residual-stream norm grew 2-5x over every run (60-100 at 1k
steps to 190-310 at 20k). Three of the four block-0.6 runs had gradient-norm spikes near 10, and in all four the prediction
error rose again after 5-10k steps.</li>
<li><b>In-trunk evals worked.</b> Every milestone was probed with no queue wait and no failed eval. Each took 1.0-3.6 min of the
trunk's 2 h (about 3-9% over three milestones). The two trunks that stopped off a milestone (block 0.3, pull 10 and 32)
evaluated their end in run().</li>
</ul>
<p><b>Next.</b> A regularizer on token marginals cannot rule out a code that depends on the sample and position but not the
content. The target has to carry something the context cannot supply from those two alone: e.g. targets from a separate
(EMA) encoder as in I-JEPA, a pixel-reconstruction term, or a second view of the same voxels at a shifted token grid, so
that only content lines up between them.</p>"""


def main():
    tiles, sections = anticollapse.build(SWEEP)
    page.write(SWEEP, __doc__, tiles, [("Findings", [("What e04 shows", FINDINGS)]), *sections])


if __name__ == "__main__":
    main()
