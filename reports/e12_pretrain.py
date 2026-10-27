"""e00/e12-cellmap-qknorm-lrdecay-12h: e11's recipes again for 12 h, with CellMap's organelle EM added to the data (lmd5_pow04),
masked tokens at the MAE's context (view 96 in 128), qk_norm and the lr falling linearly to 0 (lr_decay), and a validation loss
(held-out blocks beside training volumes) at every milestone. d0 MAE, d1 masked and d2 views on lmd5_pow04; d3 masked on lmd4_pow04
(no CellMap), d4 masked on CellMap alone, and d5 d1 with e11's smaller masked cube (5^3 tokens, not 7^3). Do the probe's early peak and fall, and e11's instabilities, survive the changes?"""

import json
from pathlib import Path

import plots_interactive as pi
from analysis_utils import boundary_ap, probe_stats, read_jsonl, run_dirs, run_params
from reports import e03_anticollapse, page

SWEEP = "e00/e12-cellmap-qknorm-lrdecay-12h"
E11 = "e00/e11-pretrain-b300-48h"
NAMES = {"d0": "MAE", "d1": "masked 96 mask 7", "d2": "views", "d5": "masked 96 mask 5"}  # e11's d0-d2 are the same objectives: masked 64 and hemibrain_wide only
SHORT, LONG = "short (+1)", "long (+10)"
FAMILIES = ["hemibrain_wide", "fish2", "liconn_mouse", "microns", "cellmap"]

# Read from the runs' metrics.json, probe.json and val.json (2026-10-27, as pulled: d1 finished; d0 and d2 at 11.2 h of 12 and
# still running; d3 and d4 pending in the queue since submission, 13 h, never started).
FINDINGS = """<p><b>Only three of the five runs have trained, and the probe says the changes did not help it.</b> d3 (masked on lmd4_pow04,
no CellMap) and d4 (masked on CellMap alone) have waited in the queue since submission and have not started, so the CellMap
comparison is not made. d1 finished its 62.5k steps (5.7 h on 8 B300s, 16M samples); d0 (259k of 325k steps, 58M samples) and
d2 (87k of 100k, 20M) are at 11.2 h of their 12 h and will stop before the lr reaches 0 (now 2.0e-5 and 1.3e-5): they ran 0.149
and 0.443 s a step against the 0.127 and 0.41 of e11.</p>
<ul>
<li><b>Training is stable.</b> No skipped step; after step 10k the gradient norm stays under 2.1 (MAE), 1.0 (masked) and 0.35 (views),
against e11's spikes to 72-300 late. The MAE's resid_norm falls from 32 to 19-35 (e11: 51 to 236), the masked run's levels off at
60-100 (e11: 125 to 314), views' wanders at 166-348 (e11: 157 to 278). qk_norm and the lr decay did not cost anything, but the
1M-step instabilities of e11 are beyond what these runs have reached (259k), so they are not shown to be fixed.</li>
<li><b>MAE</b> probes as e11's did: 0.293 short / 0.865 long at 80k, 0.292 / 0.867 at 160k (e11: 0.302 / 0.878 at 80k, 0.297 / 0.876 at
160k). e11's decline began after 160k, so there is nothing to compare yet: the 320k milestone does not exist, and the run's end
(step 259k) is not probed.</li>
<li><b>Masked 96</b> is worse than e11's masked 64 at every milestone: short AP 0.151 at 20k and 0.128 at the end (e11: 0.172 at
20k, 0.177 at 40k), long AP 0.752 at 40k (e11: 0.841). It peaks at 20k and falls although the lr decays, so the fall is not the
constant lr. It differs from e11's in three ways: a bigger view (12^3 tokens against 8^3), a bigger masked cube (mask_block 0.6 of
12 is 7^3 tokens, 343 predicted, against e11's 5^3, 125) and a quarter hemibrain_wide, the probe's own family. Confounded: it
does not say which of the three costs the probe. d5 (d1 with a 5^3 cube) is added to separate the mask's size; it has not run.</li>
<li><b>Views</b> at 80k steps: 0.180 short, 0.667 long, against e11's 0.213 and 0.721. Its long AP is still rising at 80k.</li>
<li><b>Validation loss</b> (32 fixed windows a family; the held-out blocks against training volumes). MAE: no gap on hemibrain_wide
(0.510 against 0.525 at 160k), microns (0.417 both) or CellMap (0.632 against 0.600); on liconn_mouse the gap widens from 0.07
(5k) to 0.15 (0.578 against 0.432, 160k) while both fall. Masked: gaps of 0.04 (hemibrain_wide), 0.11 (fish2), 0.09 (microns),
none for CellMap or liconn_mouse at the end. Views: the held-out loss <i>rises</i> while the training loss falls on three
families: hemibrain_wide 0.181 to 0.310 (training 0.138 to 0.101), microns 0.223 to 0.428 (0.207 to 0.231), liconn_mouse 0.214 to 0.273
(0.168 to 0.140): the invariance objective does overfit its training volumes, the one run with a clear overfitting signal. The
probe's fall in e11 was not shown to be this: the MAE and masked losses fall on held-out blocks while the probe falls.</li>
</ul>
<p><b>Caveats.</b> Single seeds. The held-out blocks are single volumes, each a different distribution from its family's training
volumes, so a gap is partly distribution shift; CellMap's own are labelled crops inside the training volumes' footprint. The
probe is the unstandardized, 3000-step fit of e11 (Eval.standardize off), so the milestone numbers compare with e11's and not
with a converged probe. d0 and d2 stop before lr 0 and have no end evals yet.</p>
<p><b>Next.</b> Run d3 and d4 (the data comparison), re-pull when d0 and d2 finish, and refit the probes with
<code>Eval.standardize</code> on the kept milestones to separate the probe's fall from the encoder's.</p>"""


def milestones(d: Path) -> tuple[list, dict]:
    """A trunk's probe at each of its milestones and its end: its steps and {short, long AP: values}."""
    aps = [(probe_stats(e)["step"], boundary_ap(probe_stats(e))) for e in [*sorted(d.glob("step_*/")), d] if (e / "probe.json").is_file()]
    return [s for s, _ in aps], {k: [a[k] for _, a in aps] for k in (SHORT, LONG)}


def against_e11() -> list:
    """Each recipe's linear-probe AP at its milestones, here beside e11's."""
    out = []
    for k in (SHORT, LONG):
        series = {}
        for d, name in NAMES.items():
            for sweep, tag in ((SWEEP, "e12"), (E11, "e11")):
                steps, aps = milestones(Path("outdir") / sweep / d)
                series[(f"{name} {tag}", name)] = (steps, aps[k])
        out.append((f"{k} boundary AP: e12 (12 h) beside e11 (48 h), the same objectives", pi.lines(series, "trunk step", k, logx=True)))
    return out


def val_losses() -> list:
    """The pretraining loss on held-out blocks and on training volumes at each milestone (val.json), a card per run with a
    radio for the family."""
    out = []
    for d, name in NAMES.items():
        series = {}
        for e in [*sorted((Path("outdir") / SWEEP / d).glob("step_*/")), Path("outdir") / SWEEP / d]:
            if (e / "val.json").is_file():
                v = json.loads((e / "val.json").read_text())
                for f in FAMILIES:
                    for split in ("val", "train"):
                        xs, ys = series.setdefault((f"{split} {f}", f), ([], []))
                        xs.append(v["step"]); ys.append(v[split][f]["loss"])
        out.append((f"{name}: pretraining loss on held-out blocks (val) and on training volumes (train), the radio picks the family",
                    pi.lines(series, "trunk step", "loss", logx=True)))
    return out


def throughput() -> tuple:
    """Each run's speed over its last 300 steps."""
    rows = []
    for d in run_dirs(SWEEP):
        p = run_params(d)
        m = read_jsonl(d / "metrics.json") if (d / "metrics.json").is_file() else []
        if p and len(m) > 31:
            a, b = m[-31], m[-1]
            s = (b["time"] - a["time"]) / (b["idx_step"] - a["idx_step"])
            n = p["batch_size"] * p["n_gpus"]
            rows.append([d.name, f"{p['objective']} {p['data']} {p['n_gpus']}x{p['queue'].removeprefix('gpu_').upper()}", b["idx_step"], b["time"] / 3600, s, n / s, n * b["idx_step"] / 1e6])
    return ("Speed over the last 300 steps, and what each run has seen",
            pi.table(["run", "recipe", "step", "hours", "s/step", "samples/s", "M samples"], rows))


def label(p: dict) -> str:
    """A trunk's name: its objective, data and GPUs."""
    return f"{p['objective']} {p['data']} {p['n_gpus']}x{p['queue'].removeprefix('gpu_').upper()}"


def main():
    tiles, sections = e03_anticollapse.build(SWEEP, label=label)
    sections = [("Findings", [("What e12 shows so far", FINDINGS)]), ("Against e11", against_e11() + [throughput()]),
                ("Validation loss", val_losses()), *sections]
    page.write(SWEEP, __doc__, tiles, sections)


if __name__ == "__main__":
    main()
