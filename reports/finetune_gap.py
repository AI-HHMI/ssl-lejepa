"""e00/e10-finetune-gap: is pretraining worth it once labels are used to train? Test-block boundary AP of a UNETR decoder
fitted on GT affinities (Finetune), as a function of fit compute (steps 100 to 10k, scored at doubling steps) and of GT (the
fit block cut to a cube of side 128, 256, 512 or whole), on the four lmd_ssl_v1 families, with the encoder frozen or trained
with the decoder. Four encoders trained here at one budget (6 h of m on one H200): MAE (decoder 512x8, view 96), masked
tokens at view 64, multi-view invariance (view 96), and masked tokens at view 64 on all four families (lmd4); each beside a
random encoder at its own view. The gap is pretrained minus random, same view, same GT, same fit steps."""

import json
from pathlib import Path
from statistics import mean

import plots_interactive as pi
from analysis_utils import boundary_ap, job_text, job_walltime_s, read_jsonl, run_dirs, saved_params
from reports import page

SWEEP = "e00/e10-finetune-gap"
TRUNKS = {"d0": "MAE", "d1": "masked 64", "d2": "views", "d3": "masked 64, lmd4"}
MATCH = {"MAE": "random 96", "views": "random 96", "masked 64": "random 64", "masked 64, lmd4": "random 64"}  # same view
ENCODERS = [*TRUNKS.values(), "random 96", "random 64"]
SETS = {"hemibrain_eb": "hemibrain EB", "fish2": "fish2", "liconn_mouse": "LICONN", "kasthuri15": "Kasthuri15"}
SIDES, MODES = (128, 256, 512, 896), ("frozen", "trained")
REFERENCE = {"d0": ("e08-mae-fix/d3, 4 h of B300", 0.296, 0.869), "d1": ("e06-viewsize-vs-baselines/d0, 8 h of B300", 0.205, 0.847),
             "d2": ("e06-viewsize-vs-baselines/d5, 8 h of B300", 0.113, 0.544)}
SHORT, LONG = "short (+1)", "long (+10)"

# Read from the runs' finetune.json, probe.json and job logs (2026-10-21).
FINDINGS = """<p><b>Pretraining helps at every amount of GT tested, and the gap closes with GT more than with fit compute.</b>
At 10k fit steps the MAE encoder beats the random one in 23 of 24 (set, GT, frozen or trained) cells, on short and on long
AP; all four pretrained encoders together in 79 of 96 (short) and 87 of 96 (long).</p>
<ul>
<li><b>Near eval(0, *) the gap is large everywhere</b>, as the linear probe implied: at 100 fit steps MAE leads its random
encoder by 0.08 short / 0.27 long AP on average (hemibrain EB, 512^3 of GT, frozen: 0.124 against 0.034 short, 0.875 against
0.406 long).</li>
<li><b>With enough GT, fit compute closes most of it.</b> On hemibrain EB at 512^3, random reaches MAE's 100-step AP after 400
(short) and 1600 (long) steps, and at 10k is within 0.016-0.025 short and 0.020-0.028 long: one seed, so near noise. That is
the largest GT that ran on EB (the whole-block fits failed; see Missing runs).</li>
<li><b>With little GT it doesn't.</b> At 128^3 random never reaches MAE's 100-step AP within 10k steps on hemibrain, fish2 or
Kasthuri; the long-AP gap at 10k (all four pretrained encoders) is 0.12 on average at 128^3, against 0.02-0.05 at 512^3 and
whole blocks. At 128^3 pretrained fits peak early (MAE at a median 800 steps) and then overfit: their best point beats random's
best by about as much as at 10k.</li>
<li><b>Kasthuri15 gains the most</b>, from encoders that never saw it (all but lmd4 are hemibrain-only): +0.19-0.21 short and
+0.17-0.20 long AP for MAE at 256^3 and 10k steps. <b>LICONN gains least</b> (short within 0.05), and the lmd4 encoder, the
only one that saw LICONN data, is not clearly better there than the same recipe on hemibrain alone (within +-0.025 at 10k).</li>
<li><b>Ranking: MAE > masked 64 = masked 64 on lmd4 > views > random</b>, by every summary in Encoder ranking. The views
encoder's gap is mostly gone by 10k steps (positive in 13 of 24 short cells).</li>
<li><b>Training the encoder with the decoder adds little</b> at these GT sizes: 10k-step AP trained minus frozen is -0.003 to
+0.026 short per encoder, and random encoders gain nothing from it (+0.006 at view 64, -0.002 at 96). The decoder does the
fitting.</li>
<li><b>The encoders reproduce</b> on their linear probes (Encoders): MAE 0.297 / 0.877 short / long AP (e08: 0.296 / 0.869)
and masked 64 0.203 / 0.858 (e06: 0.205 / 0.847), in 1.5-1.9x those runs' steps; views probes better than e06's (0.176 /
0.616 against 0.113 / 0.544) in half its steps; masked 64 on lmd4 reaches 0.211 / 0.847 on EB, level with hemibrain alone.</li>
</ul>
<p><b>Caveats.</b> One seed per cell: fish2's curves move by up to 0.05 between doubling steps, so read sub-0.03 gaps as ties.
A fit step at view 96 decodes 3.4x the voxels of one at view 64, so steps compare within an encoder's pair, not across views.
Best-of-curve numbers pick the step on the test blocks themselves. The whole-block fits of hemibrain EB, LICONN and
Kasthuri15 (and Kasthuri's 512^3) are missing, so the large-GT end rests on hemibrain and LICONN at 512^3 and fish2.</p>
<p><b>Next.</b> Rerun the 48 failed fits (fixed), then larger GT on hemibrain (several 896^3 blocks: the regime where random
should catch up), 2-3 seeds at the closing cells, and the gap as a function of pretraining compute (keep the milestones).</p>"""


def existing(path: Path) -> list:
    """An images cell: path, or nothing (a dash) where a job wrote none."""
    return [path] if path.is_file() else []


def curve(d: Path) -> list[dict]:
    """A finetune's finetune_curve.json rows; [] without results. e10's first jobs wrote theirs to finetune.json, over their
    spec (the same name then), so there it is results only if it isn't still the spec."""
    if (d / "finetune_curve.json").is_file():
        return read_jsonl(d / "finetune_curve.json")
    lines = (d / "finetune.json").read_text().splitlines()
    return [json.loads(l) for l in lines] if lines and lines[0].startswith('{"tbl"') else []


def finetunes() -> dict:
    """(encoder, set, mode, side) -> its finetune: run dir, job minutes, the GT it fit (labelled voxels; None without results)
    and its curve, fit step -> boundary AP per range ({} without results)."""
    out = {}
    for d in run_dirs(SWEEP):
        if d.name in TRUNKS:
            continue
        p = saved_params(d)
        view = p["train"]["global_size"][0]
        enc = f"random {view}" if p["init_from"] == "random" else TRUNKS[Path(p["init_from"]).name]
        rows = curve(d)
        out[(enc, SETS[p["probe_on"]], "frozen" if p["freeze_encoder"] else "trained", p["fit_side"] or 896)] = dict(
            run=d, minutes=(job_walltime_s(job_text(d, "finetune")) or 0) / 60, labelled=rows[0]["fit_labelled"] if rows else None,
            curve={r["step"]: boundary_ap(r) for r in rows[1:] if "boundary_ap_short" in r})
    return out


def gaps(runs: dict, metric: str) -> list[dict]:
    """Per (set, mode, side, pretrained encoder) with results for it and its random encoder: its AP minus random's at 100,
    1600 and 10k steps and at each curve's best, and the first step at which random reaches the pretrained encoder's
    100-step AP (None: not within 10k)."""
    out = []
    for (e, s, mode, side), f in runs.items():
        r = runs.get((MATCH.get(e), s, mode, side))
        if e not in MATCH or not f["curve"] or not r or not r["curve"]:
            continue
        c, b = ({k: v[metric] for k, v in x["curve"].items()} for x in (f, r))
        out.append(dict(enc=e, set=s, mode=mode, side=side, **{f"gap {k}": c[k] - b[k] for k in (100, 1600, 10000)},
                        best=max(c.values()) - max(b.values()), catch=next((k for k in sorted(b) if b[k] >= c[100]), None)))
    return out


def ranking(g: list[dict], metric: str, by: str) -> tuple:
    keys = list(dict.fromkeys(x[by] for x in g))
    rows = [[k, *(mean(x[c] for x in g if x[by] == k) for c in ("gap 100", "gap 1600", "gap 10000", "best")),
             f"{sum(x['gap 10000'] > 0 for x in g if x[by] == k)}/{sum(x[by] == k for x in g)}",
             f"{sum(x['catch'] is None for x in g if x[by] == k)}/{sum(x[by] == k for x in g)}"] for k in keys]
    return (f"{metric} AP: mean gap by {by} (positive: cells where pretrained leads at 10k; never caught: random never reaches "
            "its 100-step AP within 10k)", pi.table([by, "gap 100", "gap 1600", "gap 10k", "best", "positive", "never caught"], rows))


def trunk_cards() -> list:
    """The pretraining runs: linear probe at each milestone and at the end, against the run each repeats."""
    rows, series = [], {}
    for d in run_dirs(SWEEP)[:4]:
        aps = [(r["step"], boundary_ap(r)) for r in [read_jsonl(f)[-1] for f in sorted(d.glob("step_*/probe.json"))] + read_jsonl(d / "probe.json")]
        series[TRUNKS[d.name]] = ([s for s, _ in aps], [a[SHORT] for _, a in aps])
        ref, rs, rl = REFERENCE.get(d.name, ("-", None, None))
        rows.append([TRUNKS[d.name], aps[-1][0], aps[-1][1][SHORT], aps[-1][1][LONG], ref, rs, rl])
    return [("Linear probe on hemibrain EB at the end of pretraining, beside the run each repeats",
             pi.table(["encoder", "steps", "short AP", "long AP", "repeats", "its short AP", "its long AP"], rows)),
            ("Linear-probe short AP over pretraining (milestones)", pi.lines(series, "pretraining step", "short AP", logx=True))]


def build() -> tuple:
    """Return (tiles, sections) for page.write."""
    runs = finetunes()
    done = {k: f for k, f in runs.items() if f["curve"]}
    g = {m: gaps(runs, m) for m in (SHORT, LONG)}
    mae = {m: [x for x in g[m] if x["enc"] == "MAE"] for m in g}
    tiles = [(f"{len(done)}/{len(runs)}", "finetunes with results"),
             (f"{sum(x['gap 10000'] > 0 for x in mae[SHORT])}/{len(mae[SHORT])}", "cells where MAE beats random at 10k steps (short AP)"),
             *((f"{mean(x['gap 100'] for x in mae[m]):+.3f} / {mean(x['gap 10000'] for x in mae[m]):+.3f}", f"MAE minus random, {m} AP: 100 / 10k steps")
               for m in (SHORT, LONG))]
    steps = sorted({k for f in done.values() for k in f["curve"]})

    def heat(metric: str, s: str, zr: tuple) -> tuple:  # MAE minus random 96 over fit steps x (GT, mode)
        z = {(k, f"{side} {mode}"): runs[("MAE", s, mode, side)]["curve"][k][metric] - runs[("random 96", s, mode, side)]["curve"][k][metric]
             for side in SIDES for mode in MODES for k in steps
             if runs[("MAE", s, mode, side)]["curve"] and runs[("random 96", s, mode, side)]["curve"]}
        return (f"{s}: MAE minus random, {metric} AP (rows GT side and mode, columns fit steps)",
                pi.heatmap(z, steps, [f"{side} {mode}" for side in SIDES for mode in MODES], "fit steps", "GT", zr))

    def lines(metric: str, s: str, side: int) -> tuple:
        series = {f"{e}, {mode}": (sorted(f["curve"]), [f["curve"][k][metric] for k in sorted(f["curve"])])
                  for e in ENCODERS for mode in MODES if (f := done.get((e, s, mode, side)))}
        return (f"{s}, {side}^3 of GT: {metric} AP over fit steps", pi.lines(series, "fit steps", f"{metric} AP", logx=True))

    cells = [(s, side) for s in SETS.values() for side in SIDES if any(k[1] == s and k[3] == side for k in done)]
    missing = [[f["run"].name, *k] for k, f in runs.items() if not f["curve"]]
    sections = [
        ("Findings", [("What e10 shows", FINDINGS)]),
        ("The gap", [heat(m, s, zr) for m, zr in ((SHORT, (-0.1, 0.25)), (LONG, (-0.1, 0.5))) for s in SETS.values()]),
        ("Encoder ranking", [ranking(g[m], m, by) for m in (SHORT, LONG) for by in ("enc", "side", "set")]),
        ("Curves: short AP", page.wide([lines(SHORT, s, side) for s, side in cells])),
        ("Curves: long AP", page.wide([lines(LONG, s, side) for s, side in cells])),
        ("Encoders", trunk_cards()),
        ("Predictions", [("Test block after 10k steps, frozen encoder, 256^3 of GT: EM | true boundaries | predicted (rows the encoder)",
                          pi.images({e: {s: existing(runs[(e, s, "frozen", 256)]["run"] / "finetune.png") for s in SETS.values()} for e in ENCODERS},
                                    base=page.out_path(SWEEP).parent))]),
        ("All finetunes", [("AP at 10k fit steps, the GT it fit (labelled voxels) and its job's minutes",
                            pi.table(["run", "encoder", "set", "mode", "side", "labelled", SHORT, LONG, "minutes"],
                                     [[f["run"].name, *k, f["labelled"], f["curve"][10000][SHORT], f["curve"][10000][LONG], f["minutes"]]
                                      for k, f in done.items()]))]),
        ("Missing runs", [(f"{len(missing)} finetunes without results",
                           "<p>Each ran its 10k steps, then failed writing finetune.json: its fit block's size was numpy integers wherever the "
                           "block is smaller than the side asked for (the whole blocks of hemibrain EB, LICONN and Kasthuri15, and Kasthuri's "
                           "512^3, whose z is 362), which json can't write. fish2's block is larger than every side, so all of its ran.</p>"
                           + pi.table(["run", "encoder", "set", "mode", "side"], missing))]),
    ]
    return tiles, sections


def main():
    page.write(SWEEP, __doc__, *build())


if __name__ == "__main__":
    main()
