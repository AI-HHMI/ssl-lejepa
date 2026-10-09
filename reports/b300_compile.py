"""e00/b300-compile: is torch.compile broken on B300 for the Linear patch embed at batch 64 (cudagraphs crash or
miscompile)?"""

from analysis_plots import bench
from reports import page

SWEEP = "e00/b300-compile"


def main():
    page.write(SWEEP, __doc__, [], [("Benchmark", bench(SWEEP))])


if __name__ == "__main__":
    main()
