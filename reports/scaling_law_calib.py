"""e00/scaling-law-calib: per-size lr sensitivity before trusting the main grid's lr (lr ~ 1/width): does loss end lower
with higher or lower lr, and is any size non-monotonic? xl OOMs at batch 64 on one B300 (see the status column)."""

from analysis_plots import bench_table, scaling_law_calib_loss_vs_flops, scaling_law_calib_lr_scan
from reports import page

SWEEP = "e00/scaling-law-calib"


def main():
    page.write(SWEEP, __doc__, [], [("Runs", bench_table(SWEEP)),
                                    ("lr scan", scaling_law_calib_lr_scan() + scaling_law_calib_loss_vs_flops())])


if __name__ == "__main__":
    main()
