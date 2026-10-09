"""e00/viewsizes-v2: the view-size study with the fixed stack: training loss per config (d0-d16, repeats d17-d33)."""

from analysis_plots import bench_loss
from reports import page

SWEEP = "e00/viewsizes-v2"


def main():
    page.write(SWEEP, __doc__, [], [("Loss", bench_loss(SWEEP))])
    # bench_table, bench_speed; loss_curves (too slow); probe_table (good); probe_curves (info sparse);
    # probe_vs_compute(SWEEP, "e00/probe-test") (only shows that FLOPs don't explain performance);
    # probe_short_vs_long(SWEEP, "e00/probe-test") (good)


if __name__ == "__main__":
    main()
