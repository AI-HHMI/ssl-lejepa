"""The experiment-independent half of a one-page HTML report in lmd-catalog's style (scripts/analysis/report.py):
CSS, the side contents bar, tiles and cards. An experiment's report_*.py builds (tiles, sections) from plots.py (static
SVG) or plots_interactive.py fragments and calls write()."""

import re
from pathlib import Path

import plots_interactive

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


def write(out: Path, title: str, intro: str, tiles: list, sections: list):
    """tiles: (number, label); sections: (title, [(card title, svg or table html)])."""
    toc = [("overview", "Overview")] + [(anchor(t), t) for t, _ in sections]
    nav = "<nav class='toc'><b>Contents</b>" + "".join(f"<a href='#{a}'>{t}</a>" for a, t in toc) + "</nav>"
    main_html = (
        f"<main><h1 id='overview'>{title}</h1><p>{intro}</p><div class='tiles'>"
        + "".join(f"<div class='tile'><b>{n}</b>{label}</div>" for n, label in tiles) + "</div>"
        + "".join(f"<h2 id='{anchor(t)}'>{t}</h2><div class='grid'>" + "".join(card(ct, c) for ct, c in cards) + "</div>" for t, cards in sections)
        + "</main>"
    )
    runtime = plots_interactive.RUNTIME if plots_interactive.MARK in main_html else ""
    body = f"<div class='layout'>{nav}{main_html}</div><script>{TOC_JS}</script>{runtime}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>"
    )
    print(f"wrote {out}")
