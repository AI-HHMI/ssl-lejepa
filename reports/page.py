"""The experiment-independent half of a one-page HTML report in lmd-catalog's style (scripts/analysis/report.py):
CSS, the side contents bar, tiles and cards. An experiment's reports/<sweep>.py builds (tiles, sections) from
analysis_plots' cards (plots_interactive fragments) and calls write()."""

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
# Ours: ul.entries, a list of links to pages (reports/index.py): date on the left, title, then a few lines of intro.
CSS += """
ul.entries{list-style:none;margin:6px 0 0;padding:0}
ul.entries li{display:grid;grid-template-columns:96px 1fr;column-gap:16px;padding:10px 14px;margin:0 0 8px;
  border-left:3px solid #0072B2;border-radius:0 6px 6px 0;background:#fafafa;transition:background .15s}
ul.entries li:hover{background:#eef5fb}
ul.entries time{grid-row:span 2;color:#888;font:12px ui-monospace,monospace;padding-top:3px;line-height:1.5}
ul.entries time code{color:#aaa;font-size:11px}
ul.entries a{font-weight:600;color:#0072B2;text-decoration:none} ul.entries a:hover{text-decoration:underline}
ul.entries p{margin:3px 0 0;color:#555;font-size:13.5px;line-height:1.45;
  display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
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
    wide = " wide" if content.startswith(("<table", "<ul", "<div class='wide'>")) else ""  # tables, lists, wide(): a row of their own
    return f"<div class='card{wide}'><h3>{title}</h3>{content}</div>"


def wide(cards: list) -> list:
    """cards, each on a row of its own instead of side by side (e.g. charts with long labels or legends)."""
    return [(title, f"<div class='wide'>{content}</div>") for title, content in cards]


def anchor(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def out_path(sweep: str) -> Path:
    """Where write() puts a sweep's page: results/<sweep>/report.html."""
    return Path("results") / sweep / "report.html"


def write(sweep: str, intro: str | None, tiles: list, sections: list, interactive: bool = True) -> Path:
    """The sweep's page at out_path(sweep), titled after it. intro: what the experiment asks (a report's docstring);
    tiles: (number, label); sections: (title, [(card title, html fragment)]). interactive: whether a page with
    plots_interactive fragments opens interactive or static (its corner toggle switches either way)."""
    assert intro, f"{sweep}: a report's docstring is its page's intro: say what the experiment asks"
    out = out_path(sweep)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html(f"{sweep} report", intro, tiles, sections, interactive))
    print(f"wrote {out}")
    return out


def html(title: str, intro: str, tiles: list, sections: list, interactive: bool = True) -> str:
    """A whole page: title and intro, tiles, then sections of cards, with the contents bar on the side (write())."""
    toc = [("overview", "Overview")] + [(anchor(t), t) for t, _ in sections]
    nav = "<nav class='toc'><b>Contents</b>" + "".join(f"<a href='#{a}'>{t}</a>" for a, t in toc) + "</nav>"
    main_html = (
        f"<main><h1 id='overview'>{title}</h1><p>{intro}</p><div class='tiles'>"
        + "".join(f"<div class='tile'><b>{n}</b>{label}</div>" for n, label in tiles) + "</div>"
        + "".join(f"<h2 id='{anchor(t)}'>{t}</h2><div class='grid'>" + "".join(card(ct, c) for ct, c in cards) + "</div>" for t, cards in sections)
        + "</main>"
    )
    runtime = plots_interactive.runtime(interactive) if plots_interactive.MARK in main_html else ""
    body = f"<div class='layout'>{nav}{main_html}</div><script>{TOC_JS}</script>{runtime}"
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>")
