"""e00/e11-pretrain-b300-48h: e10's four pretraining runs (MAE, masked tokens at view 64, multi-view invariance, masked tokens on
lmd4), the same recipes at 7-8x the GPUs for 48 h instead of 6 h on one H200: a node of 7 H200s or 8 B300s each, batch 32 per GPU,
every doubling-step milestone probed and PCA-mapped as saved. A fifth run, d4, samples lmd4's families by voxel count ^ 0.4
instead of equally. Does the linear probe keep improving with 8x the samples and 8x the steps?"""

import itertools
from pathlib import Path

import plots_interactive as pi
from analysis_utils import boundary_ap, probe_stats, read_jsonl, run_dirs, run_params
from reports import anticollapse, page

SWEEP = "e00/e11-pretrain-b300-48h"
E10 = "e00/e10-finetune-gap"
E10_TRUNKS = {"d0": "MAE", "d1": "masked 64", "d2": "views", "d3": "masked 64 lmd4"}  # e10's d0-d3 are e11's d0-d3
OBJECTIVE = {"mae": "MAE", "mask": "masked 64", "views": "views"}
SHORT, LONG = "short (+1)", "long (+10)"

# Read from the runs' metrics.json, probe.json, pca.json and performance.json (2026-10-24, the trunks 19 h into their 48).
FINDINGS = """<p><b>The linear probe peaks early and then stops improving, or declines, however many more samples it sees.</b>
At 19 h the four runs have seen 117M (MAE), 77M (masked 64), 37M (views) and 180M (masked 64 on lmd4) samples, 9-22x e10's 6 h,
and 0.5M, 0.3M, 0.16M and 0.8M steps.</p>
<ul>
<li><b>MAE</b> peaks at 80k steps (0.302 short / 0.878 long AP, e10: 0.296 / 0.876 at 160k) and falls after: 0.297 at 160k,
0.282 / 0.866 at 320k. Its loss keeps creeping down (0.534 at 50k, 0.507 at 500k) and its gradient norm stays near 0.05-0.06 after 50k:
the reconstruction gets better and the boundary content in its tokens slightly worse.</li>
<li><b>Masked 64 is worse here than in e10, not better.</b> Short AP peaks at 0.180 (80k) and is 0.176 at 160k, against e10's 0.205
at 160k, though it is ahead at 5k (0.164 against 0.132). Long AP matches e10 (0.852 against 0.853 at 160k). Its loss keeps
falling (0.505 at 50k, 0.440 at 300k) and so does its prediction error (inv 0.143 to 0.124), while the share of token variance
that is each tile's mean token (the sample code of e03) rises 0.11 to 0.19, e10's masked run ended at 0.16. The masked-token
runs seem to buy lower loss with more of that code. The one difference from e10 besides compute is the global batch (256, with
the learning rate unchanged), which I haven't separated from the longer training.</li>
<li><b>Masked 64 on lmd4</b> (equal family weights, 7 H200s) behaves the same: 0.185 short at 80k, then 0.183, 0.173 (320k)
and 0.167 (640k); long AP flat at 0.85. It is again behind e10's short AP from 80k (0.208 at 160k) and ahead of it on long
AP at every milestone (0.852 against 0.824 at 160k). Its grad-norm spikes are larger than the others' (max 52, p99 2.9).</li>
<li><b>Views</b> gains most from the extra compute: 0.216 short at 40k (e10: 0.170 at 40k, when it stopped) and a long AP still
rising at 160k (0.757, from 0.699 at 40k). It is the slowest run (0.41 s per step on 7 H200s, 826 TFLOP a step), so it has the
fewest steps, and the only one still on its way up.</li>
<li><b>No run was unstable.</b> No skipped steps; the largest gradient norms are 2.4 (MAE), 8.9 (masked), 4.8 (views) and 52
(lmd4).</li>
</ul>
<p><b>Throughput.</b> MAE 1760 samples/s (7 H200s), views 545 (7 H200s), masked 64 on lmd4 2630 (7 H200s, 0.085 s a step), but
masked 64 on hemibrain_wide only 1120 on 8 B300s (0.229 s a step). The recipe is the same, with different data and one more GPU, and it is 2.3x slower. Either the B300 node
or hemibrain_wide's storage (512^3 shards of poorly compressed 128^3 zstd chunks, about 10 MB read for each 0.7 MB patch) is the
limit: d4 (lmd4 with new family weights, on a B300 node) separates them.</p>
<p><b>Caveats.</b> The runs are 40% of the way through their 48 h, and these are single seeds. Steps compare across e10 and
e11 only as a recipe's age: an e11 step sees 7-8x the samples. d3 shares a directory with a B300 job that crashed at step 5k
(two jobs, one directory); its metrics and checkpoints are the H200 job's, as are the evals. d4 has just started and has no
milestone yet.</p>
<p><b>Next.</b> The probe says more compute does not help after about 80k steps on masked 64 and MAE. Finetuning is the fairer
test of a longer-trained encoder (e10's gap, on these encoders); then lowering the learning rate or batch to see whether the
masked drop is from the batch size.</p>"""


def label(p: dict) -> str:
    """A trunk's name: its objective, data and GPUs."""
    gpus = f"{p['n_gpus']}x{p['queue'].removeprefix('gpu_').upper()}"
    return f"{OBJECTIVE[p['objective']]}{'' if p['data'] == 'hemibrain_wide' else ' ' + p['data']} {gpus}"


def milestones(d: Path) -> tuple[list, dict]:
    """A trunk's probe at each milestone: its steps and {short, long AP: values}."""
    aps = [(probe_stats(e)["step"], boundary_ap(probe_stats(e))) for e in sorted(d.glob("step_*/")) if (e / "probe.json").is_file()]
    return [s for s, _ in aps], {k: [a[k] for _, a in aps] for k in (SHORT, LONG)}


def against_e10() -> list:
    """Each recipe's linear-probe AP at its milestones, here beside e10's."""
    out = []
    for k in (SHORT, LONG):
        series = {}
        for d in run_dirs(SWEEP):
            if d.name in E10_TRUNKS and run_params(d):
                for sweep, tag in ((SWEEP, "e11"), (E10, "e10")):
                    steps, aps = milestones(Path("outdir") / sweep / d.name)
                    series[(f"{E10_TRUNKS[d.name]} {tag}", E10_TRUNKS[d.name])] = (steps, aps[k])
        out.append((f"{k} boundary AP: e11 (48 h) beside e10 (6 h), the same recipes", pi.lines(series, "trunk step", k, logx=True)))
    return out


def probe_fits() -> list:
    """The probe's fit curve at each milestone (probe_fit.json, every 100 steps: training BCE, and BCE and boundary AP on the
    test block's tokens, monitoring only), one card per metric with a radio for the trunk and a line per milestone; then each
    fit's last point against the milestone's step."""
    fits = {}  # trunk label -> {milestone step: probe_fit rows}
    for d in run_dirs(SWEEP):
        if run_params(d):
            fits[f"{d.name} {label(run_params(d))}"] = {int(e.name.removeprefix("step_")): read_jsonl(e / "probe_fit.json")
                                                        for e in sorted(d.glob("step_*/")) if (e / "probe_fit.json").is_file()}
    steps = sorted({k for f in fits.values() for k in f})
    colors = dict(zip(map(str, steps), itertools.cycle(pi.COLORS)))  # a milestone keeps its colour in every trunk
    cards = [(f"Probe fit: {m} over the probe's steps, a line per milestone (the radio picks the trunk)",
              pi.lines({(str(k), t): ([r["step"] for r in rows], [r[m] for r in rows]) for t, f in fits.items() for k, rows in f.items()},
                       "probe fit step", m, colors=colors))
             for m in ("loss", "held_bce", "held_boundary_ap")]
    last = {t: ([k for k in f], [f[k][-1] for k in f]) for t, f in fits.items() if f}
    cards += [(f"Probe fit's last {m} against the milestone's trunk step (a probe that overfit would fit better as it tested worse)",
               pi.lines({t: (ks, [r[m] for r in rows]) for t, (ks, rows) in last.items()}, "trunk step", m, logx=True))
              for m in ("loss", "held_bce")]
    cards.append(("Test-block BCE minus training BCE at the end of the fit, against the milestone's trunk step",
                  pi.lines({t: (ks, [r["held_bce"] - r["loss"] for r in rows]) for t, (ks, rows) in last.items()}, "trunk step", "held_bce - loss", logx=True)))
    return cards


def throughput() -> tuple:
    """Each run's speed over its last 300 steps."""
    rows = []
    for d in run_dirs(SWEEP):
        p = run_params(d)
        m = read_jsonl(d / "metrics.json") if (d / "metrics.json").is_file() else []
        if p and len(m) > 31:
            a, b = m[-31], m[-1]
            s = (b["time"] - a["time"]) / (b["idx_step"] - a["idx_step"])
            rows.append([d.name, label(p), b["idx_step"], b["time"] / 3600, s, p["batch_size"] * p["n_gpus"] / s, p["batch_size"] * p["n_gpus"] * b["idx_step"] / 1e6])
    return ("Speed over the last 300 steps, and what each run has seen",
            pi.table(["run", "recipe", "step", "hours", "s/step", "samples/s", "M samples"], rows))


def main():
    tiles, sections = anticollapse.build(SWEEP, label=label)
    sections = [("Findings", [("What e11 shows so far", FINDINGS)]), ("Against e10", against_e10() + [throughput()]), ("Probe fit trajectories", probe_fits()), *sections]
    page.write(SWEEP, __doc__, tiles, sections)


if __name__ == "__main__":
    main()
