"""e00/e04-sigreg-both: the masked-token objective with SIGReg over each view's tokens and over each position's batch at once.
Alone, the first let a code of position pass (e02-maskloss-sweep) and the second a code of the sample (e03-anticollapse);
together neither passes. Eight 2 h trunks: mask block (0.3, 0.6) x pull (lamb x the batch, each term's: 1, 3.2, 10, 32),
every 5k-step milestone evaluated as saved. Does a linear probe now find content, above the boundary base rate and a random
encoder? In the PCA maps a position code is a lattice identical on every crop, and a sample code one flat colour per crop."""

from reports import anticollapse, page

SWEEP = "e00/e04-sigreg-both"


def main():
    page.write(SWEEP, __doc__, *anticollapse.build(SWEEP))


if __name__ == "__main__":
    main()
