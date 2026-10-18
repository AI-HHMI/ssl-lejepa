"""e00/e06-01-test-tile-size: does an encoder's view of the data depend on the input size at test time? e06-viewsize-vs-
baselines' best encoder (masked tokens, trained on views of 64 voxels = 8^3 tokens, its last checkpoint) run on one held-out
EB val crop (512 x 1024 x 1024 voxels, z y x) in non-overlapping cubic tiles of 16 to 512 voxels (2^3 to 64^3 tokens): 4
tiles of the largest, 131072 of the smallest. Its position embedding has only seen positions 0-7 per axis. Each map is the
same z-slice: EM | PCA of the tokens | PCA after subtracting each tile's mean token, fitted on that slice."""

from pathlib import Path

import plots_interactive as pi
from analysis_utils import read_jsonl
from reports import page

SWEEP = "e00/e06-01-test-tile-size"

# Read from pca_sizes.json and the pca_tile_*.png maps (2026-10-18).
FINDINGS = """<ul>
<li><b>Larger tiles than the encoder trained on make its tokens more semantic, not less.</b> At its own 64 voxels the maps
are smooth regions, one per neurite cross-section. At 256 and 512 the top three components separate kinds of
structure across the whole crop instead: cytoplasm in one colour, membranes in another, mitochondria (the dark bodies in the
EM) in a third, consistent from tile to tile. The share of token variance that is each tile's own mean falls from 0.15 at
64 to 0.015 at 256 and 0.006 at 512.</li>
<li><b>But each tile's border gets a code of its own.</b> At 128 and up, a band a few tokens wide along every tile edge is a
colour of its own, whatever the image there; at 128 it shows as stripes across the crop. Two likely causes: the position
embedding reaches positions it never saw (up to 63 per axis), and border tokens have context on one side only.</li>
<li><b>Smaller tiles lose the content.</b> At 32 the maps still follow the cells, but each 4^3-token tile is visible; at 16
(2^3 tokens) the tokens mostly follow the EM's brightness, the tile code holds a quarter of the variance (0.25) and the
centred map is a lattice.</li>
<li><b>The spectrum is widest at the training size.</b> Effective rank 291 at 64, against 204-264 elsewhere; the top three
components hold 14% of the variance at 64 and 31-43% at every other size, where a few directions (kinds of structure at large
tiles, brightness at small ones) dominate.</li>
</ul>
<p><b>Next.</b> Whether the larger-tile tokens probe better is open: the boundary probe's context leaves the EB crop at tiles of
256 and up, so it needs a smaller test block, or tiles overlapped so their borders are discarded. Overlapping tiles would
also remove the border code from the maps.</p>"""


def main():
    rows = read_jsonl(Path("outdir", SWEEP, "d0", "pca_sizes.json"))
    tiles = [r["tile"] for r in rows]
    stats = {"effective rank": [r["effective_rank"] for r in rows],
             "between-tile variance": [r["between_tile_variance"] for r in rows],
             "top-3 explained variance": [sum(r["explained_variance"][:3]) for r in rows]}
    head = [(f"{min(tiles)}-{max(tiles)}", "tile sides (voxels)"), (f"{rows[0]['step']}", "encoder's training step"),
            (f"{max(stats['effective rank']):.0f} @ {tiles[stats['effective rank'].index(max(stats['effective rank']))]}", "widest spectrum: effective rank @ tile")]
    sections = [("Findings", [("What e06-01 shows", FINDINGS)]),
                ("Statistics vs tile size", [(f"{k} vs test tile side (the encoder trained at 64)", pi.lines({k: (tiles, v)}, "tile side (voxels)", k, logx=True))
                                             for k, v in stats.items()]),
                ("Maps", [("Rows the tile side; each map EM | PCA | PCA minus each tile's mean, the same z-slice",
                           pi.images({f"tile {t}": {"map": [Path("outdir", SWEEP, "d0", f"pca_tile_{t:03d}.png")]} for t in tiles},
                                     base=page.out_path(SWEEP).parent))])]
    page.write(SWEEP, __doc__, head, sections)


if __name__ == "__main__":
    main()
