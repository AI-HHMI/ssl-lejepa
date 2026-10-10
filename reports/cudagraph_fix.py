"""e00/cudagraph-fix: do cudagraphs work with PatchEmbed3d out of the compiled graph (eager_patch_embed)? Compared with
e00/b300-compile."""

from analysis_plots import bench
from reports import page

SWEEP = "e00/cudagraph-fix"


def main():
    # Full width: runs from two sweeps make long labels (sweep/dN + config) and legends.
    page.write(SWEEP, __doc__, [], [("Benchmark", page.wide(bench(SWEEP, "e00/b300-compile")))])


if __name__ == "__main__":
    main()
