"""Mermaid diagrams of the project (diagrams/*.mmd), drawn in the browser by pele (pele.run): the entities of an experiment,
the dataflow from volumes to reports, and the three pretraining models.
Usage: python -m reports.diagrams  (writes results/diagrams.html)"""

import html
from pathlib import Path

from reports import page

PELE = "https://cdn.jsdelivr.net/npm/pele@1.10.0/dist/pele.global.js"
OUT = Path("results/diagrams.html")
SECTIONS = {
    "Entities": [("An experiment's entities and how they relate (erDiagram)", "entities")],
    "Dataflow": [("From volumes on disk to the report (flowchart)", "dataflow")],
    "Models": [("MAE: encode a quarter of the tokens, decode the rest to voxels", "model-mae"),
               ("Masked tokens: predict a hidden cube of a shifted view", "model-masked"),
               ("Views: invariance across crops of one volume", "model-views")],
}
DRAW = "document.querySelectorAll('pre.pele').forEach(p => { p.outerHTML = pele.render(p.textContent).svg; });"


def main():
    sections = [(title, [(card, f"<div class='wide'><pre class='pele'>{html.escape(Path(f'diagrams/{name}.mmd').read_text())}</pre></div>")
                         for card, name in cards]) for title, cards in SECTIONS.items()]
    intro = "High-level diagrams of how the project fits together: what an experiment is made of, how data flows through it, and what each pretraining objective trains."
    doc = page.html("Diagrams", intro, [], sections, home="../index.html")
    OUT.write_text(doc.replace("</body>", f"<script src='{PELE}'></script><script>{DRAW}</script></body>"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
