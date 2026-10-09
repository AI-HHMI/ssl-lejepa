"""e00/scaling-law-pca: why do the scaling-law PCA maps look the same in every row? Tokens collapse onto their 96^3
view's code as models grow and train longer; does that cost the linear probe? Maps: one per checkpoint, rows = model
size, columns = compute budget; each pca2.png is EM | PCA | PCA minus tile mean | per-tile PCA | token norm, over the
3 tile layers of a held-out crop."""

from pathlib import Path

import plots_interactive as pi
from analysis import run_dirs, saved_params, scaling_law_budget
from analysis_plots import scaling_law_tile_code_vs_probe
from reports import page

SWEEP = "e00/scaling-law-pca"


def maps() -> list:
    """pca2.png per checkpoint: rows = size (n_layers x width), columns = its source run's budget; random init last."""
    grid = {}
    for d in run_dirs(SWEEP):
        p, src = saved_params(d), Path(saved_params(d)["init_from"])
        row, col = (f'{p["n_layers"]}x{p["width"]}', scaling_law_budget(src)) if src.name != "random" else ("random init", "")
        grid.setdefault(row, {})[col] = [d / "pca2.png"]
    return [("PCA maps: size x budget", pi.images(grid, base=page.out_path(SWEEP).parent))]


def main():
    page.write(SWEEP, __doc__, [], [("PCA maps", maps()), ("Tile code vs probe", scaling_law_tile_code_vs_probe())])


if __name__ == "__main__":
    main()
