"""e00/probe-test: the linear affinity probe on a known-good trained encoder (d0: viewsizes/d0, 8 h) vs a random one
(d1), probe only; d2/d3 probe b300-train8h-dynamic d0/d1. Fit curves (training BCE, held-out BCE and boundary AP) and
test-block boundary AP per channel. Images: outdir/e00/probe-test/d*/probe.png (EM | true | predicted boundaries)."""

from analysis_plots import probe_curves, probe_table
from reports import page

SWEEP = "e00/probe-test"


def main():
    page.write(SWEEP, __doc__, [], [("Probe", probe_table(SWEEP)), ("Probe fit", probe_curves(SWEEP))])


if __name__ == "__main__":
    main()
