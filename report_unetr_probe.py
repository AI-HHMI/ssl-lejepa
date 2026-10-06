"""One-page HTML report of e00/unetr-probe (a UNETR decoder vs the linear probe on the scaling-law encoders and random
encoders), in lmd-catalog's report style (scripts/analysis/report.py: plots.py, tiles, cards, contents on the side).
From pulled artifacts in outdir/ only. Usage: uv sync --extra analysis && uv run python report_unetr_probe.py
   ->  results/e00/unetr-probe/report.html
"""

import json
import math
import re
import statistics
from pathlib import Path

import plots
from analysis import boundary_ap, probe_source, probe_stats, probe_walltime_s, probed_dirs, read_jsonl, run_compute, run_dirs, saved_params

SWEEP = "e00/unetr-probe"
LINEAR_RANDOM = Path("outdir/e00/probe-test/d1")  # the linear probe on a random encoder, same probe blocks
OUT = Path("results") / SWEEP / "report.html"
SIZES = {(4, 256): "xs", (6, 384): "s", (12, 512): "m", (16, 768): "l"}  # (n_layers, width)
INTRO = """UNETR decoder (skip connections into the encoder's own blocks) vs the linear probe, on the encoders of
scaling-law's budget-c4 runs and on random encoders. AP is the average precision of ranking different-object voxel pairs
first, per affinity channel: short range (+1 voxel, membranes; 2% of pairs are boundaries) and long range (+10 voxels;
17-22%), each the mean of its three axes, on the held-out EB test block. Higher is better."""
# Copied from lmd-catalog's scripts/analysis/report.py (CSS and TOC_JS verbatim).
CSS = """
body{font:15px system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:16px;color:#222;background:#fff}
h1{margin-bottom:4px} h2{margin-top:36px;border-bottom:1px solid #ddd;padding-bottom:4px}
.tiles{display:flex;flex-wrap:wrap;gap:12px} .tile{border:1px solid #ddd;border-radius:6px;padding:10px 16px}
.tile b{display:block;font-size:24px} .grid{display:flex;flex-wrap:wrap;gap:20px}
.card{flex:1 1 460px;min-width:0} .card.wide{flex-basis:100%} .card svg{max-width:100%;height:auto} h3{margin:8px 0 4px;font-size:15px}
table{border-collapse:collapse;font-size:13px} th,td{padding:3px 10px;border-bottom:1px solid #eee;text-align:left}
td.n,th{text-align:right} th:first-child{text-align:left}
h1,h2{scroll-margin-top:16px} nav.toc{display:none}
@media (min-width:1100px){
  body{max-width:none}
  .layout{display:grid;grid-template-columns:230px minmax(0,1200px);gap:32px;max-width:1500px;margin:0 auto}
  nav.toc{display:block;position:sticky;top:16px;align-self:start;max-height:calc(100vh - 32px);overflow:auto;font-size:14px}
  nav.toc b{display:block;margin:0 0 8px 12px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:#888}
  nav.toc a{display:block;padding:5px 10px 5px 12px;color:#555;text-decoration:none;border-left:2px solid #eee}
  nav.toc a:hover{color:#222;border-left-color:#bbb} nav.toc a.on{color:#0072B2;border-left-color:#0072B2;font-weight:600}
}
"""
TOC_JS = """
const links = [...document.querySelectorAll('nav.toc a')];
const targets = links.map(a => document.querySelector(a.getAttribute('href')));
function mark() {
  let cur = 0;
  targets.forEach((t, i) => { if (t.getBoundingClientRect().top <= 120) cur = i; });
  links.forEach((a, i) => a.classList.toggle('on', i === cur));
}
addEventListener('scroll', mark, {passive: true});
mark();
"""


def card(title: str, content: str) -> str:
    wide = " wide" if content.startswith("<table") else ""  # tables get a row of their own
    return f"<div class='card{wide}'><h3>{title}</h3>{content}</div>"


def anchor(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def run_info(d: Path) -> dict:
    """A probed unetr-probe run beside the linear probe on the same encoder (its source scaling-law run, or the random
    baseline), and its mia-evals record if it has been scored."""
    p = saved_params(d)
    src, sp, init = probe_source(d)
    random = init == "random"
    name = SIZES[(sp["n_layers"], sp["width"])]
    kind = "pretrained" if not random else "random, encoder " + ("frozen" if p.get("unetr_freeze_encoder", True) else "trained")
    steps, eflop = (None, None) if random else run_compute(src)
    walltime = probe_walltime_s(d)
    assert walltime, f"{d}: no probe job log; run ./pull.sh?"
    records = sorted(d.glob("mia_evals/*/records/*.json"))
    return dict(
        run=d.name, name=name, kind=kind, label=f"{name} ({src.name})" if not random else f"{name}, {kind}",
        source="-" if random else f"{src.parent.name}/{src.name}", steps=steps, eflop=eflop, minutes=walltime / 60,
        unetr=boundary_ap(probe_stats(d)), linear=boundary_ap(probe_stats(LINEAR_RANDOM if random else src)),
        fit=read_jsonl(d / "probe_fit.json"), score=json.loads(records[-1].read_text())["scores"]["voxel_instance"] if records else None,
    )


def ap_bars(runs, linear_random, key: str) -> str:
    """UNETR bar for every run, a linear bar beside each pretrained one, and one linear bar for the random encoder."""
    rows = []
    for r in runs:
        rows.append((f"{r['label']}: UNETR", r["unetr"][key], "UNETR"))
        if r["kind"] == "pretrained":
            rows.append((f"{r['label']}: linear", r["linear"][key], "Linear"))
    rows.append(("random encoder: linear", linear_random[key], "Linear"))
    return plots.group_bar(rows, "boundary AP")


def ap_vs_compute(runs, key: str) -> str:
    pre = [r for r in runs if r["kind"] == "pretrained"]
    groups = {dec: ([math.log10(r["eflop"]) for r in pre], [r[field][key] for r in pre]) for dec, field in [("UNETR", "unetr"), ("Linear", "linear")]}
    return plots.scatter(groups, "log10 pretraining EFLOP", f"{key} boundary AP")


def build() -> tuple:
    """Return (tiles, sections): tiles are (number, label); sections are (title, [(card title, svg or table html)])."""
    runs = [run_info(d) for d in probed_dirs(SWEEP)]
    linear_random = boundary_ap(probe_stats(LINEAR_RANDOM))
    frozen = next(r for r in runs if r["kind"] == "random, encoder frozen")
    pre = [r for r in runs if r["kind"] == "pretrained"]
    best = max(pre, key=lambda r: r["unetr"]["short (+1)"])
    scored = [r for r in runs if r["score"]]
    tiles = [
        (f"{len(runs)}/{len(run_dirs(SWEEP))}", "runs probed"),
        (f"{best['unetr']['short (+1)']:.3f}", f"best pretrained short AP ({best['name']})"),
        (f"{frozen['unetr']['short (+1)']:.3f}", "random frozen encoder, short AP"),
        (f"{sum(r['unetr']['short (+1)'] > frozen['unetr']['short (+1)'] for r in pre)}/{len(pre)}", "pretrained beat random (short AP)"),
        (f"{statistics.median(r['minutes'] for r in runs):.0f}", "min per probe (median)"),
        (f"{len(scored)}/{len(runs)}", "scored with mia-evals"),
    ]
    dash = lambda v, f: "—" if v is None else f.format(v)
    table = plots.table(
        ["Run", "Model", "Source", "Pretrain steps", "EFLOP", "UNETR short AP", "UNETR long AP", "Linear short AP", "Linear long AP", "Probe min"],
        [[r["run"], r["label"] if r["kind"] != "pretrained" else r["name"], r["source"], r["steps"] or "—", r["eflop"] or "—",
          f"{r['unetr']['short (+1)']:.3f}", f"{r['unetr']['long (+10)']:.3f}", f"{r['linear']['short (+1)']:.3f}",
          f"{r['linear']['long (+10)']:.3f}", f"{r['minutes']:.1f}"] for r in runs],
    )
    score_table = plots.table(
        ["Run", "Model", "PQ", "VOI split", "VOI merge", "Adapted Rand error"],
        [[r["run"], r["label"], *(dash(r["score"] and r["score"][k], "{:.3f}") for k in ["pq", "voi_split", "voi_merge", "adapted_rand_error"])] for r in runs],
    )
    fits = {r["label"]: ([row["step"] for row in r["fit"]], [row["held_boundary_ap"] for row in r["fit"]]) for r in runs}
    sections = [
        ("Results", [
            ("Runs (linear = linear probe on the same encoder; random rows compare with the random linear baseline)", table),
            ("Short-range boundary AP (+1 voxel: membranes)", ap_bars(runs, linear_random, "short (+1)")),
            ("Long-range boundary AP (+10 voxels: same neuron?)", ap_bars(runs, linear_random, "long (+10)")),
        ]),
        ("Compute", [
            ("Short-range boundary AP vs pretraining compute", ap_vs_compute(runs, "short (+1)")),
            ("Long-range boundary AP vs pretraining compute", ap_vs_compute(runs, "long (+10)")),
        ]),
        ("Probe fit", [("Held-out boundary AP vs probe step (every 100 steps; monitoring only)", plots.lines(fits, "held-out boundary AP"))]),
        ("Cost", [("Probe job walltime (minutes, whole job: startup, fit, test inference, artifacts)", plots.bar({r["label"]: r["minutes"] for r in runs}, "minutes"))]),
        ("Scoring", [("mia-evals neuron instance segmentation of the probe's affinities (test block; — = not scored)", score_table)]),
    ]
    return tiles, sections


def main():
    tiles, sections = build()
    toc = [("overview", "Overview")] + [(anchor(title), title) for title, _ in sections]
    nav = "<nav class='toc'><b>Contents</b>" + "".join(f"<a href='#{a}'>{t}</a>" for a, t in toc) + "</nav>"
    main_html = (
        f"<main><h1 id='overview'>{SWEEP} report</h1><p>{INTRO}</p><div class='tiles'>"
        + "".join(f"<div class='tile'><b>{n}</b>{label}</div>" for n, label in tiles) + "</div>"
        + "".join(f"<h2 id='{anchor(title)}'>{title}</h2><div class='grid'>" + "".join(card(t, c) for t, c in cards) + "</div>" for title, cards in sections)
        + "</main>"
    )
    body = f"<div class='layout'>{nav}{main_html}</div><script>{TOC_JS}</script>"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{SWEEP} report</title><style>{CSS}</style></head><body>{body}</body></html>"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
