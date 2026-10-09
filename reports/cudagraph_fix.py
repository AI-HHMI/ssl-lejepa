"""e00/cudagraph-fix: do cudagraphs work with PatchEmbed3d out of the compiled graph (eager_patch_embed)? Compared with
e00/b300-compile."""

from analysis_plots import bench
from reports import page

SWEEP = "e00/cudagraph-fix"


def main():
    page.write(SWEEP, __doc__, [], [("Benchmark", bench(SWEEP, "e00/b300-compile"))])


if __name__ == "__main__":
    main()
