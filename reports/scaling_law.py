"""e00/scaling-law: the IsoFLOP grid (4 budgets x 5 sizes). Status and progress, loss curves per budget, loss vs compute
(valid mid-run), lr vs compute per size, and the IsoFLOP U-curve per budget (only trustworthy once sizes are similarly
progressed: check the table's steps and status first)."""

from reports import page
from analysis_plots import bench_table, scaling_law_isoflop, scaling_law_loss_curves, scaling_law_loss_vs_flops, scaling_law_lr_vs_flops

SWEEP = "e00/scaling-law"


def main():
    page.write(SWEEP, __doc__, [], [("Runs", bench_table(SWEEP)), ("Loss", scaling_law_loss_curves()),
                                    ("Compute", scaling_law_loss_vs_flops() + scaling_law_lr_vs_flops() + scaling_law_isoflop())])


if __name__ == "__main__":
    main()
