"""e00/e01-viewsizes-2d: a 2D grid of input patch x global view size (local view = global view; batch 32, 12x512, 40k
steps), each cell a constant-lr trunk that, at every 10% of its steps, spawns an anneal of that milestone (10% of the
milestone's steps, cosine lr to 1%, Adam moments carried over), which then gets PCA maps and a linear boundary probe.
Do large views and large input patches help or hurt the linear probe's AP, and does AP keep rising along a trunk or
peak and fall as the scaling-law models did? Does the share of token variance that is just each view's own code
(between-tile) grow with training, and do the invariance and SIGReg terms, now logged apart, trade off? Trunk lengths
differ in wall-clock (max_hours is a 2x cap on an estimate): see the coverage section before reading a missing cell."""

from pathlib import Path

import pandas

import plots_interactive as pi
from analysis_plots import LINEAR_RANDOM, frame, series_by
from analysis_utils import boundary_ap, probe_stats, read_jsonl, run_dirs, saved_params
from reports import page

SWEEP = "e00/e01-viewsizes-2d"
LOSS_TERMS = ["inv", "sigreg", "resid_norm"]


def run_info(d: Path) -> dict:
    """One run dir: its config, whether it's a trunk or an anneal (and of which trunk step), where it got to, its
    probe's boundary AP, its PCA stats and the mean of its last 10 logged losses. Missing files give NaN."""
    p = saved_params(d)
    m = pandas.DataFrame([r for r in read_jsonl(d / "metrics.json") if r["tbl"] == "metrics"])
    pca = read_jsonl(d / "pca.json")
    ap = boundary_ap(probe_stats(d)) if (d / "probe.json").is_file() else {}
    anneal = p["anneal_from"]
    last = m[["loss", *LOSS_TERMS]].tail(10).mean() if len(m) else {}
    return {"run": d.name, "input": p["patch_size"][0], "global": p["global_size"][0], "kind": "anneal" if anneal else "trunk",
            "step": int(Path(anneal).stem.split("_")[1]) if anneal else (int(m.idx_step.max()) + 1 if len(m) else 0),
            "max_hours": p["max_hours"], "hours": m.time.max() / 3600 if len(m) else float("nan"), "logged": len(m) > 0,
            "ap_short": ap.get("short (+1)"), "ap_long": ap.get("long (+10)"),
            "between_tile": pca[-1]["between_tile_variance"] if pca else None, "effective_rank": pca[-1]["effective_rank"] if pca else None,
            **{k: last.get(k) for k in ["loss", *LOSS_TERMS]}, "pca2": d / "pca2.png"}


def trunk_curves(d: Path) -> pandas.DataFrame:
    """A trunk's inv, sigreg and pre-LN residual norm vs step, smoothed over 100 steps and thinned to every 100th."""
    m = pandas.DataFrame([r for r in read_jsonl(d / "metrics.json") if r["tbl"] == "metrics"])
    p = saved_params(d)
    out = m[LOSS_TERMS].rolling(10).mean().iloc[9::10].copy()
    out["step"] = m.idx_step.iloc[9::10]
    out["config"] = f'{p["patch_size"][0]}/{p["global_size"][0]}'
    return out


def build() -> tuple:
    """Return (tiles, sections) for page.write."""
    runs = pandas.DataFrame([run_info(d) for d in run_dirs(SWEEP)])
    runs["config"] = runs.input.astype(str) + "/" + runs["global"].astype(str)
    runs = runs.sort_values(["global", "input", "kind", "step"], ascending=[True, True, False, True])  # trunk first
    trunks, anneals = runs.query("kind == 'trunk'"), runs.query("kind == 'anneal'")
    scored = anneals.dropna(subset=["ap_short"])
    best_s, best_l = scored.loc[scored.ap_short.idxmax()], scored.loc[scored.ap_long.idxmax()]
    random_ap = boundary_ap(probe_stats(LINEAR_RANDOM))
    tiles = [
        (f"{len(runs)}", f"run dirs pulled ({len(trunks)} trunks, {len(anneals)} anneals)"),
        (f"{len(scored)}", "anneals with a probe"),
        (f"{best_s.ap_short:.3f}", f"best short AP: {best_s.config} @ {best_s.step}"),
        (f"{best_l.ap_long:.3f}", f"best long AP: {best_l.config} @ {best_l.step}"),
        (f"{random_ap['short (+1)']:.3f} / {random_ap['long (+10)']:.3f}", "random encoder, linear probe: short / long"),
    ]
    coverage = pi.table(["Run", "Input/global", "Max hours", "Hours logged", "Steps logged", "Probed"],
                        [[r.run, r.config, r.max_hours, r.hours, r.step, "yes" if pandas.notna(r.ap_short) else "no"] for r in trunks.itertuples()])
    over_steps = lambda y, rows: {c: (g.step.tolist(), g[y].tolist()) for c, g in rows.dropna(subset=[y]).groupby("config", sort=False)}
    curves = pandas.concat([trunk_curves(Path("outdir", SWEEP, r)) for r in trunks.query("logged").run])
    steps = sorted(anneals.step.unique())
    pivot = lambda y: frame(scored.pivot(index="config", columns="step", values=y).reindex(columns=steps).reset_index().rename(columns={"config": "input/global"}))
    both = trunks.merge(anneals, on=["config", "step"], suffixes=("_trunk", "_anneal")).dropna(subset=["ap_short_trunk", "ap_short_anneal"])
    effect = pi.table(["Input/global", "Step", "Short AP trunk", "Short AP annealed", "Long AP trunk", "Long AP annealed"],
                      [[r.config, r.step, r.ap_short_trunk, r.ap_short_anneal, r.ap_long_trunk, r.ap_long_anneal] for r in both.itertuples()])
    maps = {}
    for c, g in runs.groupby("config", sort=False):
        cells = {str(r.step): [r.pca2] if r.pca2.is_file() else [] for r in g.itertuples() if r.kind == "anneal"}
        end = g.query("kind == 'trunk'").iloc[0]
        maps[c] = {**{str(s): cells.get(str(s), []) for s in steps}, "trunk end (no anneal)": [end.pca2] if end.pca2.is_file() else []}
    table = pi.table(["Run", "Input/global", "Kind", "Trunk step", "AP short", "AP long", "Between-tile", "Effective rank", "Loss", "Inv", "SIGReg", "Resid norm"],
                     [[r.run, r.config, r.kind, r.step, r.ap_short, r.ap_long, r.between_tile, r.effective_rank, r.loss, r.inv, r.sigreg, r.resid_norm]
                      for r in runs.itertuples()])
    dims = ["resid_norm", "inv", "sigreg", "effective_rank", "ap_short", "ap_long"]
    pairs = {f"global {g}": s[["run", "config", "step", *dims]].to_dict("records") for g, s in scored.groupby("global")}
    heat = lambda y: [(f"Trunk step {s}", pi.heatmap(scored.query("step == @s").set_index(["global", "input"])[y].to_dict(), sorted(runs["global"].unique()),
                                                     sorted(runs.input.unique()), "global size", "input size", (scored[y].min(), scored[y].max())))
                      for s in steps]  # one colour scale across the steps
    sections = [
        ("Coverage", [("Trunks: how far each got (max_hours is a 2x cap on an estimate; a trunk past it never spawns its later anneals)", coverage)]),
        ("Probe vs trunk step", [(f"{rng} boundary AP of the annealed milestone, one line per input/global", pi.lines(over_steps(y, scored), "trunk step", y))
                                 for rng, y in [("Short-range", "ap_short"), ("Long-range", "ap_long")]]),
        ("Short-range AP heatmaps", heat("ap_short")),
        ("Long-range AP heatmaps", heat("ap_long")),
        ("Tile code vs trunk step", [(title, pi.lines(over_steps(y, anneals), "trunk step", y))
                                     for title, y in [("Between-tile variance of the annealed milestone", "between_tile"), ("Effective rank of its tokens", "effective_rank")]]),
        ("Trunk losses", [(f"{t} vs step (100-step mean), one line per input/global", pi.lines(series_by(curves.dropna(), "config", "step", t), "step", t)) for t in LOSS_TERMS]),
        ("Losses vs probe", page.wide([("Every probed anneal: its last-10 mean loss terms, its tokens' effective rank and boundary AP, colored by global view", pi.splom(pairs, dims))])),
        ("Annealing", [("Where a trunk finished and its same-step anneal exists: does annealing help?", effect)]),
        ("AP by milestone", [("Short-range AP, input/global x trunk step", pivot("ap_short")), ("Long-range AP, input/global x trunk step", pivot("ap_long"))]),
        ("PCA maps", [("Rows input/global, columns the milestone's trunk step; last column the trunk's own (constant lr) end",
                       pi.images(maps, base=page.out_path(SWEEP).parent))]),
        ("All runs", [("Per run (loss terms: mean of the last 10 logged)", table)]),
    ]
    return tiles, sections


def main():
    page.write(SWEEP, __doc__, *build())


if __name__ == "__main__":
    main()
