"""e00/e02-maskloss-sweep: the masked-token objective (one 96^3 view of a 128^3 input, a cube of its tokens hidden from the
encoder and predicted from the rest; m, batch 32) over the cube's size (mask_block: its side as a fraction of the 12-token
grid side) and SIGReg's weight (lamb), each pair twice, every run trained for a time budget (8 h on one RTX PRO 6000) as a
constant-lr trunk whose milestones (every 10k steps) each get PCA maps and the linear boundary probe as saved, with no anneal.
Does any block size or lamb give tokens a linear probe can use, beyond a random encoder's? Does AP keep rising along a
trunk? Does the prediction error (inv, the old invariance term's slot) trade off against SIGReg, and does the share of
token variance that is just each tile's own code (between-tile) track either? Chance for the probe is the boundary base rate
(the tiles): check every AP against it, and the random encoder's, before reading a difference between cells."""

import statistics
from pathlib import Path

import pandas

import plots_interactive as pi
from analysis_plots import LINEAR_RANDOM
from analysis_utils import boundary_ap, probe_stats, read_jsonl, run_dirs, saved_params
from reports import page

SWEEP = "e00/e02-maskloss-sweep"
LOSS_TERMS = ["inv", "sigreg", "resid_norm"]
SHORT_CHANNELS = ["(1, 0, 0)", "(0, 1, 0)", "(0, 0, 1)"]


def trunk_losses(d: Path) -> pandas.DataFrame:
    """A trunk's logged loss, inv, sigreg and pre-LN residual norm per step (idx_step + 1)."""
    m = pandas.DataFrame([r for r in read_jsonl(d / "metrics.json") if r["tbl"] == "metrics"])
    return m.assign(step=m.idx_step + 1)[["step", "loss", *LOSS_TERMS]] if len(m) else pandas.DataFrame(columns=["step", "loss", *LOSS_TERMS])


def run_info(d: Path, trunk_dir: Path, losses: pandas.DataFrame) -> dict:
    """One run dir: a trunk (evaluated at its own end), or an eval of one of its milestones (init_from). Its config, the trunk
    step it evaluates, its probe's boundary AP and the boundary base rate, its PCA stats, and the mean of its trunk's last 10
    logged losses up to that step. Missing files give None."""
    p = saved_params(d)
    pca = read_jsonl(d / "pca.json")
    probe = probe_stats(d) if (d / "probe.json").is_file() else {}
    ap = boundary_ap(probe) if probe else {}
    m = pandas.DataFrame([r for r in read_jsonl(d / "metrics.json") if r["tbl"] == "metrics"])
    step = int(Path(p["init_from"]).stem.split("_")[1]) if p["init_from"] else (int(m.idx_step.max()) + 1 if len(m) else 0)
    near = losses[losses.step <= step].tail(10)
    log = next(iter(sorted(d.glob("job_run_*.log"))), None)
    return {"run": d.name, "trunk": int(trunk_dir.name[1:]), "kind": "eval" if p["init_from"] else "trunk", "block": p["mask_block"], "lamb": p["lamb"],
            "step": step, "max_hours": p["max_hours"], "hours": m.time.max() / 3600 if len(m) else None,
            "quota": log is not None and "Disk quota exceeded" in log.read_text(errors="ignore"),
            "ap_short": ap.get("short (+1)"), "ap_long": ap.get("long (+10)"),
            "base_short": sum(probe[f"boundary_frac_{c}"] for c in SHORT_CHANNELS) / 3 if probe else None,
            "between_tile": pca[-1]["between_tile_variance"] if pca else None, "effective_rank": pca[-1]["effective_rank"] if pca else None,
            **{k: near[k].mean() if len(near) else None for k in ["loss", *LOSS_TERMS]}, "pca2": d / "pca2.png"}


def build() -> tuple:
    """Return (tiles, sections) for page.write."""
    dirs = [d for d in run_dirs(SWEEP) if saved_params(d)]  # a dir with no runs.json is a job that hasn't started
    trunk_of = lambda d: Path(saved_params(d)["init_from"]).parents[2] if saved_params(d)["init_from"] else d  # the trunk dir an eval reads
    all_losses = {d.name: trunk_losses(d) for d in dirs if not saved_params(d)["init_from"]}
    runs = pandas.DataFrame([run_info(d, trunk_of(d), all_losses[trunk_of(d).name]) for d in dirs if trunk_of(d).name in all_losses])
    trunks = runs.query("kind == 'trunk'").sort_values("trunk")
    repeat = trunks.groupby(["block", "lamb"]).cumcount() + 1  # a trunk's repeat: its rank among the trunks of its (block, lamb)
    runs["repeat"] = runs.trunk.map(dict(zip(trunks.trunk, repeat)))
    runs = runs.sort_values(["block", "lamb", "repeat", "kind", "step"], ascending=[True, True, True, False, True])
    trunks, evals = runs.query("kind == 'trunk'"), runs.query("kind == 'eval'")
    scored = runs.dropna(subset=["ap_short"])  # evals and probed trunk ends
    steps = sorted(evals.step.unique())
    blocks, lambs = sorted(runs.block.unique()), sorted(runs.lamb.unique())
    best_s, best_l = scored.loc[scored.ap_short.idxmax()], scored.loc[scored.ap_long.idxmax()]
    random_ap = boundary_ap(probe_stats(LINEAR_RANDOM))
    full = trunks[trunks.hours >= 0.99 * trunks.max_hours]
    tiles = [
        (f"{len(full)}/{len(trunks)}", "trunks that ran their full time budget"),
        (f"{trunks.quota.sum()} + {(trunks.hours < 0.99 * trunks.max_hours).sum() - trunks.quota.sum()}", "trunks stopped by a full disk quota + otherwise incomplete (still running at the pull, or killed)"),
        (f"{len(scored)}", f"probed checkpoints ({len(evals.dropna(subset=['ap_short']))} milestones, {len(trunks.dropna(subset=['ap_short']))} trunk ends)"),
        (f"{best_s.ap_short:.3f}", f"best short AP: block {best_s.block}, lamb {best_s.lamb}, {best_s.run} @ {best_s.step}"),
        (f"{best_l.ap_long:.3f}", f"best long AP: block {best_l.block}, lamb {best_l.lamb}, {best_l.run} @ {best_l.step}"),
        (f"{random_ap['short (+1)']:.3f} / {random_ap['long (+10)']:.3f}", "random encoder, linear probe: short / long"),
        (f"{scored.base_short.mean():.3f}", "boundary base rate (short): a probe that finds nothing scores this"),
        (f"{statistics.median(full.step) / 1000:.0f}k", "median steps of a full-budget trunk"),
    ]
    coverage = pi.table(["Run", "Block", "Lamb", "Repeat", "Steps logged", "Hours logged", "Max hours", "Status", "Probed at end"],
                        [[r.run, r.block, r.lamb, r.repeat, r.step, r.hours, r.max_hours,
                          "disk quota exceeded" if r.quota else "full budget" if r.hours >= 0.99 * r.max_hours else "incomplete: running at the pull, or killed",
                          "yes" if pandas.notna(r.ap_short) else "no"] for r in trunks.itertuples()])
    # Heatmaps: x = lamb, y = block, the mean over the repeats; one colour scale across the figures of a range.
    def heat(rows: pandas.DataFrame, y: str, zrange: tuple) -> str:
        cells = rows.dropna(subset=[y]).groupby(["lamb", "block"])[y].mean().to_dict()
        return pi.heatmap(cells, lambs, blocks, "lamb", "mask block", zrange)
    scale = lambda y, rows: (rows[y].min(), rows[y].max())
    facets = [(f"Trunk step {s}", evals.query("step == @s")) for s in steps] + [("Trunk end (8 h, no milestone)", trunks)]
    heatmaps = lambda y: [(title, heat(rows, y, scale(y, scored))) for title, rows in facets]
    ends = [(title, heat(trunks, y, scale(y, trunks))) for title, y in
            [("Between-tile variance at the trunk end", "between_tile"), ("Effective rank of its tokens", "effective_rank"),
             ("Prediction error (inv), last 100 steps", "inv"), ("SIGReg, last 100 steps", "sigreg"),
             ("Pre-LN residual norm, last 100 steps", "resid_norm")]]
    # Probe vs step: one card per block, one line per lamb (the repeats averaged; a trunk's end at its mean step).
    def over_steps(y: str, block: float) -> str:
        rows = scored[scored.block == block]
        pts = pandas.concat([rows.query("kind == 'eval'").groupby(["lamb", "step"], as_index=False)[y].mean(),
                             rows.query("kind == 'trunk'").groupby("lamb", as_index=False)[["step", y]].mean()]).sort_values(["lamb", "step"])
        return pi.lines({str(l): (g.step.tolist(), g[y].tolist()) for l, g in pts.groupby("lamb")}, "trunk step", y)
    pairs = {f"block {b}": g[["run", "lamb", "step", "kind", "inv", "sigreg", "resid_norm", "effective_rank", "between_tile", "ap_short", "ap_long"]].dropna().to_dict("records")
             for b, g in scored.groupby("block")}
    # Repeat noise: the two repeats of every (block, lamb), at the trunk end.
    ends_by = trunks.dropna(subset=["ap_long"]).pivot(index=["block", "lamb"], columns="repeat", values="ap_long").dropna().reset_index()
    noise = pi.scatter({f"block {b}": [{"repeat 1": r[1], "repeat 2": r[2], "lamb": r.lamb, "block": b} for _, r in g.iterrows()] for b, g in ends_by.groupby("block")},
                       "repeat 1", "repeat 2", hlines={"random encoder": random_ap["long (+10)"]})
    maps = {}
    for (b, l), g in runs.query("repeat == 1").groupby(["block", "lamb"], sort=False):
        cells = {str(r.step): [r.pca2] if r.pca2.is_file() else [] for r in g.itertuples() if r.kind == "eval"}
        end = g.query("kind == 'trunk'").iloc[0]
        maps[f"{b}/{l}"] = {**{str(s): cells.get(str(s), []) for s in steps}, "trunk end": [end.pca2] if end.pca2.is_file() else []}
    table = pi.table(["Run", "Kind", "Block", "Lamb", "Repeat", "Trunk step", "AP short", "AP long", "Base rate", "Between-tile", "Effective rank", "Loss", "Inv", "SIGReg", "Resid norm"],
                     [[r.run, r.kind, r.block, r.lamb, r.repeat, r.step, r.ap_short, r.ap_long, r.base_short, r.between_tile, r.effective_rank, r.loss, r.inv, r.sigreg, r.resid_norm]
                      for r in runs.itertuples()])
    sections = [
        ("Coverage", [("Trunks: how far each got (a trunk's later milestones are never evaluated if it stopped early)", coverage)]),
        ("Short-range AP heatmaps", heatmaps("ap_short")),
        ("Long-range AP heatmaps", heatmaps("ap_long")),
        ("Probe vs trunk step", [(f"Short-range AP, block {b}, one line per lamb", over_steps("ap_short", b)) for b in blocks]
                                + [(f"Long-range AP, block {b}, one line per lamb", over_steps("ap_long", b)) for b in blocks]),
        ("Trunk end", ends),
        ("Losses vs probe", page.wide([("Every probed checkpoint: its trunk's loss terms around that step, its tokens' effective rank and tile code, and boundary AP, colored by block", pi.splom(pairs, ["inv", "sigreg", "resid_norm", "effective_rank", "between_tile", "ap_short", "ap_long"]))])),
        ("Repeat noise", [("Long-range AP at the trunk end, repeat 1 vs 2, per (block, lamb)", noise)]),
        ("PCA maps", [("Rows block/lamb (repeat 1), columns the milestone's trunk step; last column the trunk's own end",
                       pi.images(maps, base=page.out_path(SWEEP).parent))]),
        ("All runs", [("Per run (loss terms: mean of the trunk's last 10 logged values up to that step)", table)]),
    ]
    return tiles, sections


def main():
    page.write(SWEEP, __doc__, *build())


if __name__ == "__main__":
    main()
