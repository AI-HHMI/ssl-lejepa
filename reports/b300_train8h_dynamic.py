"""e00/b300-train8h-dynamic: the first long 8xB300 runs, width 512 vs 1024, dynamic compile."""

from analysis_plots import bench, loss_curves
from reports import page

SWEEP = "e00/b300-train8h-dynamic"


def main():
    page.write(SWEEP, __doc__, [], [("Benchmark", bench(SWEEP)), ("Loss", loss_curves(SWEEP))])


if __name__ == "__main__":
    main()
