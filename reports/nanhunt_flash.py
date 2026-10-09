"""e00/nanhunt_flash: cuDNN attention's NaN gradients vs flash (nanhunt, nanhunt_beta95, nanhunt_flash), and flash's
throughput cost."""

from analysis_plots import flash_perf, loss_curves, nanhunt_plot
from reports import page

SWEEP = "e00/nanhunt_flash"


def main():
    page.write(SWEEP, __doc__, [], [("NaN hunt", nanhunt_plot()), ("Flash vs cuDNN", flash_perf()), ("Loss", loss_curves(SWEEP))])


if __name__ == "__main__":
    main()
