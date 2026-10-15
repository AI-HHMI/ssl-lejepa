"""The site that serves the reports from this checkout (serve.sh): site/index.html, one entry per report page in
results/ (its title, intro, and the experiment's original commit and its date), grouped by research thread (THREADS),
next to links site/results -> results/ and site/outdir -> outdir/, so the pages' relative links (e.g. to outdir/'s PCA
maps) resolve. Only site/ is served: never the repo root, whose .git/, .env and .venv/ aren't for the web.
Usage: python -m reports.index (serve.sh runs it first).
"""

import html
import re
import subprocess
import time
from pathlib import Path

from analysis_utils import read_jsonl
from reports import page

SITE = Path("site")
LINKED = ["results", "outdir"]  # all the site serves, besides its index
# The index's sections: research threads and their pages, by key (a report's sweep, or a top-level page's name), shown
# newest first. A page no thread lists goes under "Unsorted": add it here.
THREADS = {
    "Performance optimization": ["perf_journey", "e00/b300-compile", "e00/cudagraph-fix", "e00/nanhunt_flash"],
    "View creation": ["e00/e01-viewsizes-2d", "e00/viewsizes-v2", "e00/viewsizes-pca"],
    "Architecture and scaling": ["e00/b300-train8h-dynamic", "e00/scaling-law-calib", "e00/scaling-law", "e00/scaling-law-pca"],
    "Evaluation": ["e00/probe-test", "e00/unetr-probe"],
    "Masked-token objective": ["e00/e02-maskloss-sweep"],
}


def pages() -> list[Path]:
    """The report pages: results/<sweep>/report.html, and results/*.html (e.g. perf_journey.html)."""
    return sorted(Path("results").glob("**/report.html")) + sorted(Path("results").glob("*.html"))

def key(path: Path) -> str:
    """A page's key in THREADS: its sweep for results/<sweep>/report.html, else its name (results/perf_journey.html)."""
    return str(path.parent.relative_to("results")) if path.name == "report.html" else path.stem

def git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True)
    assert out.returncode == 0, f"git {' '.join(args)}: {out.stderr.strip()}"
    return out.stdout.strip()

def origin(path: Path) -> tuple[float, str]:
    """(commit time, short commit id) of the experiment behind a page: the oldest commit any of its runs ran from
    (outdir/<sweep>/d*/runs.json's commit_id; later pca/probe/score jobs ran from newer ones). A page that isn't one
    sweep's (perf_journey) gets the commit that first added it."""
    ids = {r["commit_id"] for f in Path("outdir", key(path)).glob("d*/runs.json") for r in read_jsonl(f) if r.get("commit_id")}
    if not ids:
        lines = git("log", "--format=%at %h", "--", str(path)).splitlines()  # author time: survives jj rewrites
        assert lines, f"{path}: no runs.json in outdir/{key(path)}/d*/ and no commit of the page itself"
        t, c = lines[-1].split()  # git log lists newest first
        return float(t), c
    times = {c: float(git("show", "-s", "--format=%ct", c)) for c in ids}  # commit times: the code as it ran
    c = min(ids, key=lambda c: times[c])
    return times[c], c[:8]

def entry(path: Path, when: tuple[float, str]) -> str:
    """One page as a list item: its experiment's original commit and date, its <title> as the link, and its first
    paragraph (write()'s intro) as plain text, clamped to a few lines (the full text on hover)."""
    text = path.read_text(errors="ignore")
    title = re.search(r"<title>(.*?)</title>", text, re.S)
    intro = re.search(r"<p>(.*?)</p>", text, re.S)
    plain = html.escape(html.unescape(re.sub(r"<[^>]+>", "", intro[1])), quote=True) if intro else ""
    name = title[1].removesuffix(" report") if title else path.parent.name  # write()'s "<sweep> report": the sweep
    date = time.strftime("%Y-%m-%d", time.localtime(when[0]))
    return (f"<li><time>{date}<br><code>{when[1]}</code></time><a href='{html.escape(str(path), quote=True)}'>{name}</a>"
            f"<p title='{plain}'>{plain}</p></li>")

def main():
    SITE.mkdir(exist_ok=True)
    for name in LINKED:
        link = SITE / name
        if not link.is_symlink():
            assert not link.exists(), f"{link} exists and isn't a link: move it out of site/"
            link.symlink_to(Path("..") / name)
    found = pages()
    assert found, "no report pages in results/: run a report first, e.g. python -m reports.viewsizes_pca"
    when = {p: origin(p) for p in found}
    listed = {k for ks in THREADS.values() for k in ks}
    groups = [(t, [p for p in found if key(p) in ks]) for t, ks in THREADS.items()] + [("Unsorted", [p for p in found if key(p) not in listed])]
    sections = [(t, [(f"{len(ps)} experiment{'s' * (len(ps) > 1)}, newest first",
                      "<ul class='entries'>" + "".join(entry(p, when[p]) for p in sorted(ps, key=lambda p: -when[p][0])) + "</ul>")])
                for t, ps in groups if ps]
    tiles = [(str(len(found)), "report pages"), (str(len(sections)), "research threads"),
             (time.strftime("%Y-%m-%d", time.localtime(max(t for t, _ in when.values()))), "newest experiment")]
    (SITE / "index.html").write_text(page.html("ssl-lejepa results", "Experiment reports by research thread, from this checkout's outdir/ mirror; "
                                               "each dated by its experiment's original commit.", tiles, sections))
    print(f"wrote {SITE / 'index.html'}: {len(found)} report pages")


if __name__ == "__main__":
    main()
