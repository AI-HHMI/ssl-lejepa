"""e00/e11-pretrain-b300-48h: e10's four pretraining runs (MAE, masked tokens at view 64, multi-view invariance, masked tokens on
lmd4), the same recipes at 7-8x the GPUs for 48 h instead of 6 h on one H200: a node of 7 H200s or 8 B300s each, batch 32 per GPU,
every doubling-step milestone probed and PCA-mapped as saved. A fifth run, d4, was to sample lmd4's families by voxel count ^ 0.4
instead of equally (it never started). Does the linear probe keep improving with 8x the samples and 8x the steps?"""

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

# Read from the runs' metrics.json, probe.json and performance.json (2026-10-26, the four trunks finished: 48 h each).
FINDINGS = """<p><b>The linear probe peaks early and then declines, however many more samples the encoder sees; and two of the four runs
went unstable late.</b> In 48 h the runs saw 295M (MAE, 1.32M steps), 192M (masked 64, 0.75M), 93M (views, 0.42M) and 456M
(masked 64 on lmd4, 2.04M) samples, 23-55x e10's 6 h on one GPU. d4 (lmd4 weighted by voxel count ^ 0.4) never trained: its job
failed validating its spec 27 s in (types.py lacked the name), so the weighting is untested.</p>
<ul>
<li><b>MAE</b> peaks at 80k steps (0.302 short / 0.878 long AP; e10 0.296 / 0.876 at 160k) and falls after: 0.282 / 0.866 at 320k,
0.274 / 0.862 at 640k, 0.266 / 0.862 at 1.28M. Its loss is flat from 50k (0.531 to 0.505 at 1M) and its gradient norm sits at 0.06
until step 1.06M. Then it rises to 0.3-0.5 with spikes to 72, and the activation norm (resid_norm) climbs from 51 to 236; the loss
does not move (0.505 to 0.513). The 1.28M milestone comes from this unstable stretch.</li>
<li><b>Masked 64</b> (8 B300s) is worse than in e10 at every milestone after 5k. Short AP peaks at 0.180 (80k; e10 0.205 at 160k)
and falls to 0.135 at 640k; long AP peaks at 0.852 (80-160k, e10 0.853) and falls to 0.816. Its loss falls (0.508 at 47k, 0.41 at
750k) and its prediction error with it (inv 0.151 to 0.11-0.13), while the share of token variance that is each tile's mean token
(the sample code of e03) rose from 0.11 to 0.19 in the first 19 h. No instability: gradient norm 0.25-0.27 throughout, no spike
above 0.45, but its resid_norm rises steadily from 125 to 314.</li>
<li><b>Masked 64 on lmd4</b> (equal family weights, 7 H200s) behaves the same until step 1.0M: 0.185 short at 80k, 0.167 at 640k,
long AP 0.85 (e10: 0.208 short, 0.824 long at 160k). Gradient spikes of 100-300 begin around step 1.0M (24 h) and at step
1,198,060 (28.4 h) it diverges for good: loss 0.37 to about 1.05, resid_norm 110 to over 1e6. It then trains 0.8M steps in that
state. Its 1.28M milestone (0.032 short, 0.302 long) and its final weights are of the diverged model, and its run's end-of-training
results are meaningless.</li>
<li><b>Views</b> has the fewest steps, and its long AP peaks too: 0.757 at 160k, then 0.739 at 320k, with short AP 0.216 at 40k
falling to 0.181. It is far ahead of e10's 40k stop (0.170 / 0.604). No instability (gradient norm 0.12 falling to 0.05), resid_norm 157 to 278.</li>
<li><b>No step was skipped</b> (the skip applies only to a non-finite gradient; grad-norm clipping at 1.0 is always on).</li>
</ul>
<p><b>Why the probe declines.</b> The decline starts at 80k steps in every run, long before the instabilities (d0 at 1.06M, d3
at 1.0M), so the instability does not explain it. The probe-fit curves below show it is not the probe overfitting: its training
BCE rises with its test BCE and the gap stays at about 0.004 (MAE). The features lose boundary information that a linear readout
can use while the pretraining loss falls. Untested: a held-out pretraining loss (is it overfitting the pretraining data?), refitting
the probe with 10x the steps, the encoder's feature scale against the probe's loss.</p>
<p><b>Optimizer.</b> AdamW, weight decay 0.05 on matrices, constant lr 1e-4 after a 1,000-step warmup (no decay), beta2 0.95. The
decay's time constant is 200k steps, so the weights had been at equilibrium for 5 time constants before d0 and d3 went
unstable. The three candidates I can name are untested: weight decay shrinking weights the loss is scale-blind to (d3's
resid_norm fell from 150 to 110 before it diverged, but d0, d1 and d2's rise), no lr decay at all over more than 1M steps, and
the pairing of the two. No pre-divergence checkpoint exists: milestone weights are deleted once probed (release_milestone).</p>
<p><b>Throughput.</b> MAE 1760 samples/s (7 H200s), views 545 (7 H200s), masked 64 on lmd4 2630 (7 H200s, 0.083 s a step), but
masked 64 on hemibrain_wide only 1120 on 8 B300s (0.230 s a step), with the same recipe and one more GPU: 2.3x slower. Either the
B300 node or hemibrain_wide's storage (512^3 shards of poorly compressed 128^3 zstd chunks, about 10 MB read for each 0.7 MB
patch) is the limit; d4 would have separated them.</p>
<p><b>Caveats.</b> Single seeds. Steps compare across e10 and e11 only as a recipe's age: an e11 step sees 7-8x the samples (batch
224-256 against 32, the learning rate unchanged, which I haven't separated from the longer training). d3 shares a directory with a
B300 job that crashed at step 5k (two jobs, one directory); its metrics and checkpoints are the H200 job's, as are the evals.</p>
<p><b>Next.</b> The probe says more compute does not help after about 80k steps. Finetuning is the fairer test of a longer-trained
encoder (e10's gap, on the 80k-160k and 640k milestones, which I no longer have as weights: only d0's rolling checkpoints
remain). Before another long run: keep the milestone checkpoints, log per-layer weight norms, and decide on an lr decay or a
smaller weight decay or batch; then run d4.</p>"""


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
