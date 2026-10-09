"""e00/viewsizes-pca: PCA maps of the 34 viewsizes-v2 encoders (17 input/global/local view-size configs x 2 repeats; all
12x512, 8 h on one B300), loaded through init_from, laid out by the view-size axes that sweep explored. Each pca2.png
shows a 3 x 4 x 4 crop of the run's own global view, held out in EB's val slab: columns EM | PCA of tokens | PCA after
subtracting each tile's mean | PCA fitted within each tile | pre-LayerNorm token norm; rows the middle slice of each of
the 3 tile layers. Between-tile variance is the share of token variance that is just each global view's own code: in
e00/scaling-law-pca it grew with model size and training. Does view geometry move it, does it track the linear probe's
boundary AP (viewsizes-v2's own probe), and is either beyond the repeat-to-repeat noise? Runs trained a fixed 8 h, so
step counts differ ~4x between configs: see the training-length section before reading a view-size effect."""

import statistics
from pathlib import Path

import pandas

import plots_interactive as pi
from analysis import boundary_ap, probe_stats, read_jsonl, run_compute, run_dirs, saved_params
from analysis_plots import LINEAR_RANDOM
from reports import page

SWEEP = "e00/viewsizes-pca"
N_CONFIGS = 17  # viewsizes-v2's configs; d0-d16 are repeat 1, d17-d33 repeat 2
SCALED = [(96, 64, 32), (128, 96, 64), (160, 128, 80), (192, 144, 96), (256, 192, 128)]  # (input, global, local)
# viewsizes-v2's axes: name -> (x axis label, which (input, global, local) configs lie on it, their x). The base
# (128, 96, 64) lies on all four.
AXES = {
    "input patch": ("input patch (global 96, local 64)", lambda c: c[1:] == (96, 64), lambda c: c[0]),
    "global view": ("global view (input 128, local 64)", lambda c: (c[0], c[2]) == (128, 64), lambda c: c[1]),
    "local view": ("local view (input 128, global 96)", lambda c: c[:2] == (128, 96), lambda c: c[2]),
    "all scaled": ("input patch, views scaled with it", lambda c: c in SCALED, lambda c: c[0]),
}
def label(c) -> str:
    return "/".join(map(str, c))


def run_info(d: Path) -> dict:
    """One viewsizes-pca run: its view sizes, PCA stats (pca.json), and its source viewsizes-v2 run's training length
    and linear-probe boundary AP (None where that run has no probe.json)."""
    p = saved_params(d)
    pca = read_jsonl(d / "pca.json")
    assert pca, f"{d}: no pca.json; run ./pull.sh?"
    src = Path(p["init_from"])
    steps, eflop = run_compute(src)
    ap = boundary_ap(probe_stats(src)) if (src / "probe.json").is_file() else None
    c = (p["patch_size"][0], p["global_size"][0], p["local_size"][0])
    i = int(d.name[1:])
    return {"run": d.name, "source": src.name, "config": c, "repeat": 1 + i // N_CONFIGS, "steps": steps, "EFLOP": eflop,
            "batch": saved_params(src)["batch_size"], "between_tile": pca[-1]["between_tile_variance"],
            "effective_rank": pca[-1]["effective_rank"], "centered_effective_rank": pca[-1]["centered_effective_rank"],
            "ap_short": ap and ap["short (+1)"], "ap_long": ap and ap["long (+10)"], "pca2": d / "pca2.png"}


def axis_of(c) -> str:
    """The axis a config lies on, for colouring: the base lies on all of them."""
    return "base" if c == (128, 96, 64) else next(name for name, (_, on, _) in AXES.items() if on(c))


def build() -> tuple:
    """Return (tiles, sections) for page.write."""
    runs = [run_info(d) for d in run_dirs(SWEEP)]
    probed = [r for r in runs if r["ap_short"] is not None]
    by_config = {}
    for r in runs:
        by_config.setdefault(r["config"], {})[r["repeat"]] = r
    noise = [abs(rs[1]["between_tile"] - rs[2]["between_tile"]) for rs in by_config.values() if len(rs) == 2]
    df = pandas.DataFrame(runs)
    most = max(runs, key=lambda r: r["between_tile"])
    tiles = [
        (f"{len(runs)}/{len(run_dirs(SWEEP))}", "runs with PCA maps"),
        (f"{df.between_tile.min():.2f}–{df.between_tile.max():.2f}", "between-tile variance, min–max"),
        (f"{statistics.median(noise):.3f}", "repeat noise: median |Δ between-tile|"),
        (label(most["config"]), f"most tile code ({most['between_tile']:.2f})"),
        (f"{df.between_tile.corr(df.ap_short, method='spearman'):+.2f}", f"Spearman between-tile vs short AP ({len(probed)} probed)"),
        (f"{df.between_tile.corr(df.steps, method='spearman'):+.2f}", "Spearman between-tile vs training steps"),
    ]
    maps, tile_code, probe = [], [], []
    for name, (xlabel, on, x) in AXES.items():
        configs = sorted((c for c in by_config if on(c)), key=x)
        grid = {f"repeat {k}": {label(c): [by_config[c][k]["pca2"]] for c in configs if k in by_config[c]} for k in (1, 2)}
        # Linked, not embedded (42 MB of maps): the page works from this checkout, next to outdir/.
        maps.append((f"{xlabel}: input/global/local per column", pi.images(grid, base=page.out_path(SWEEP).parent)))
        series = lambda key, k: ([x(c) for c in configs if k in by_config[c] and by_config[c][k][key] is not None],
                                 [by_config[c][k][key] for c in configs if k in by_config[c] and by_config[c][k][key] is not None])
        tile_code.append((xlabel, pi.lines({f"repeat {k}": series("between_tile", k) for k in (1, 2)}, xlabel, "between-tile variance")))
        probe.append((xlabel, pi.lines({f"{rng} AP, repeat {k}": series(f"ap_{rng}", k) for rng in ("short", "long") for k in (1, 2)},
                                       xlabel, "linear-probe boundary AP")))
    random_ap = boundary_ap(probe_stats(LINEAR_RANDOM))
    point = lambda r, y: {"between-tile": r["between_tile"], y: r[y], "config": label(r["config"]), "repeat": r["repeat"],
                          "run": r["run"], "steps": r["steps"]}
    groups = lambda y, rows: {a: [point(r, y) for r in rows if axis_of(r["config"]) == a] for a in ["base", *AXES]}
    vs_probe = [(f"Between-tile variance vs {rng}-range linear-probe AP (gray: a random encoder's)",
                 pi.scatter(groups(f"ap_{rng}", probed), "between-tile", f"ap_{rng}",
                            hlines={"random encoder, linear": random_ap[f"{rng} (+{1 if rng == 'short' else 10})"]}))
                for rng in ("short", "long")]
    steps_point = lambda r, y: {"steps": r["steps"], y: r[y], "config": label(r["config"]), "repeat": r["repeat"], "run": r["run"]}
    steps_groups = lambda y, rows: {a: [steps_point(r, y) for r in rows if axis_of(r["config"]) == a] for a in ["base", *AXES]}
    confound = [("Between-tile variance vs training steps (8 h each: bigger views and batches train fewer steps)",
                 pi.scatter(steps_groups("between_tile", runs), "steps", "between_tile")),
                ("Short-range linear-probe AP vs training steps", pi.scatter(steps_groups("ap_short", probed), "steps", "ap_short"))]
    dash = lambda v: "—" if v is None else v
    table = pi.table(
        ["Run", "Source", "Input/global/local", "Axis", "Repeat", "Batch", "Steps", "Between-tile", "Effective rank",
         "Within-tile eff. rank", "AP short", "AP long"],
        [[r["run"], r["source"], label(r["config"]), axis_of(r["config"]), r["repeat"], r["batch"], r["steps"] or "—",
          r["between_tile"], r["effective_rank"], r["centered_effective_rank"], dash(r["ap_short"]), dash(r["ap_long"])] for r in runs])
    missing = sorted(r["source"] for r in runs if r["ap_short"] is None)
    sections = [
        ("PCA maps", maps),
        ("Tile code vs view size", tile_code),
        ("Probe vs view size", probe),
        ("Tile code vs probe", vs_probe),
        ("Training length", confound),
        ("All runs", [(f"Per run (AP: viewsizes-v2's own linear probe; none for {', '.join(missing) or 'no run'})", table)]),
    ]
    return tiles, sections


def main():
    page.write(SWEEP, __doc__, *build())


if __name__ == "__main__":
    main()
