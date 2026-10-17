"""e00/e03-anticollapse: e02-maskloss-sweep's masked-token objective collapsed to a code of token position alone. Two fixes,
six 2 h trunks each: SIGReg over each position's batch samples (batch), or detached targets with e02's per-view SIGReg
(stopgrad), each over mask block (0.3, 0.6) x SIGReg pull (lamb x its sample count: 1, 3.2, 10), every 5k-step milestone
evaluated as saved. Does either give tokens a linear probe can use, above the boundary base rate (a probe that finds nothing)
and a random encoder? In the PCA maps the position code is a lattice identical on every crop."""

from pathlib import Path

import pandas

import plots_interactive as pi
from analysis_plots import LINEAR_RANDOM, trunk_loss_curves
from analysis_utils import BOUNDARY_CHANNELS, boundary_ap, probe_stats, read_jsonl, run_dirs, run_params, tflop_per_step, trunk_metrics
from reports import page

SWEEP = "e00/e03-anticollapse"
TOKEN_SIDE = 8  # the encoder's patch side (lib.encoders.EncoderConfig's default, which lejepa_config keeps)


def cell(p: dict) -> str:
    """A trunk's cell: its fix (stopgrad, or SIGReg's samples), mask block, SIGReg pull (lamb x SIGReg's sample count: the
    batch, or the view's tokens; "both" scales its view term to the batch's pull, lib.models.Lejepa.sigreg_term) and, since
    e05, the target view's shift."""
    over = p["sigreg_over"]
    n = (p["global_size"][0] // TOKEN_SIDE) ** 3 if over == "view" else p["batch_size"]
    shift = f" shift {p['mask_shift']}" if p.get("mask_shift") else ""
    return f"{'stopgrad' if p['stop_grad'] else over} block {p['mask_block']} pull {p['lamb'] * n:.3g}{shift}"


def build(sweep: str = SWEEP, label=cell) -> tuple:
    """Return (tiles, sections) for page.write: of this sweep, or another of the same shape (reports/sigreg_both.py), its
    trunks named by label(params) (cell's names for e03-e05's masked-token grids)."""
    # Since e04 a trunk's milestone evals nest in it (<trunk>/step_*/); before, they were sibling dN/ dirs.
    params = {e: run_params(e) for d in run_dirs(sweep) for e in [d, *sorted(d.glob("step_*/"))]}
    params = {d: p for d, p in params.items() if p}  # a dir with no runs.json is a job that hasn't started
    labels = {d.name: f"{d.name} {label(p)}" for d, p in params.items() if not p["init_from"]}
    trunk = lambda d, p: Path(p["init_from"]).parents[2] if p["init_from"] else d  # a milestone's: outdir/<sweep>/dN/checkpoints/milestones/
    # Every probe of a trunk's weights: its milestones', and its own end's if it stopped off one (max_hours). Its training
    # compute is the trunk's FLOPs per step (performance.json's benchmark window) x the step.
    evals = pandas.DataFrame([{"trunk": labels[trunk(d, p).name], "step": probe_stats(d)["step"],
                               "PFLOP": probe_stats(d)["step"] * tflop_per_step(trunk(d, p)) / 1e3,
                               **boundary_ap(probe_stats(d)), "pca2": d / "pca2.png", "between-tile variance": read_jsonl(d / "pca.json")[-1]["between_tile_variance"],
                               **{f"base {k}": sum(probe_stats(d)[f"boundary_frac_{c}"] for c in cs) / 3 for k, cs in BOUNDARY_CHANNELS.items()}}
                              for d, p in params.items() if (d / "probe.json").is_file() and (d / "pca.json").is_file()])
    random_ap = boundary_ap(probe_stats(LINEAR_RANDOM))
    tiles = [(f"{len(labels)}", "trunks started"), (f"{len(evals)}", "checkpoints probed"),
             (f"{random_ap['short (+1)']:.3f} / {random_ap['long (+10)']:.3f}", "random encoder, linear probe: short / long AP")]
    sections = [("Loss trajectories", trunk_loss_curves({labels[d.name]: trunk_metrics(d) for d in params if d.name in labels},
                                                        ["loss", "inv", "sigreg", "resid_norm"]))]
    if len(evals):
        evals = evals.sort_values(["trunk", "step"])
        best = evals.loc[evals["short (+1)"].idxmax()]
        tiles += [(f"{best['short (+1)']:.3f}", f"best short AP: {best.trunk} @ {best.step}"),
                  (f"{evals['base short (+1)'].mean():.3f} / {evals['base long (+10)'].mean():.3f}", "boundary base rate: short / long")]
        sections += [
            ("Probe vs trunk step", [(f"{k} boundary AP, one line per trunk (random encoder {random_ap[k]:.3f}, base rate {evals[f'base {k}'].mean():.3f})",
                                      pi.lines({t: (g.step.tolist(), g[k].tolist()) for t, g in evals.groupby("trunk")}, "trunk step", k))
                                     for k in BOUNDARY_CHANNELS]
                                    + [(f"{k} boundary AP vs training compute (trunks differ in FLOPs per step), one line per trunk",
                                        pi.lines({t: (g.PFLOP.tolist(), g[k].tolist()) for t, g in evals.groupby("trunk")}, "training PFLOP", k, logx=True))
                                       for k in BOUNDARY_CHANNELS]
                                    + [("Between-tile variance: the share of token variance that is each tile's mean token (a code of the sample)",
                                        pi.lines({t: (g.step.tolist(), g["between-tile variance"].tolist()) for t, g in evals.groupby("trunk")},
                                                 "trunk step", "between-tile variance"))]),
            ("PCA maps", [("Rows the trunks, columns the milestone's trunk step",
                           pi.images({t: {str(r.step): [r.pca2] if r.pca2.is_file() else [] for r in g.itertuples()} for t, g in evals.groupby("trunk")},
                                     base=page.out_path(sweep).parent))]),
        ]
    return tiles, sections


def main():
    page.write(SWEEP, __doc__, *build())


if __name__ == "__main__":
    main()
