"""e00/e07-mask-side: how big should the masked cube be? e05-shifted-target's best cell (view 96 = 12^3 tokens, targets
from a second view shifted 32 voxels, pull 10, SIGReg "both") with the cube's side swept over every size from 1 to 11
tokens (0.06% to 77% of the view's tokens), 2 h on one B300 each. The cube must fit both views, so sides 9, 10 and 11 get
shifts of 24, 16 and 8 voxels instead of 32 and differ in the shift as well. Side 7 is e05's d7 on another GPU."""

from reports import anticollapse, page

SWEEP = "e00/e07-mask-side"

# Read from the runs' pca.json, probe.json, metrics.json and pca2.png (2026-10-17).
FINDINGS = """<ul>
<li><b>AP rises with the cube's side up to 8 tokens of 12</b> (30% of the tokens hidden): final short / long AP 0.111 / 0.596
at side 2, 0.134 / 0.651 at 4, 0.155 / 0.701 at 6, 0.166 / 0.732 at 7 and 0.172 / 0.748 at 8 (random encoder 0.124 / 0.481).
The order holds at every milestone from 20k steps on, and every run is still rising at 2 h.</li>
<li><b>A single hidden token teaches nothing useful.</b> At side 1, AP falls below a random encoder's with training (0.101 to
0.095 short), the prediction error stays at 0.73 and effective rank drops to 32; its PCA maps are token-sized noise with the
membranes barely showing. Sides 2-3 end near or below the random encoder's short AP.</li>
<li><b>Past side 8 the shift had to shrink, and long-range AP falls with it.</b> Side 9 (shift 24) 0.168 / 0.718, side 10
(shift 16) 0.156 / 0.644, side 11 (shift 8) 0.169 / 0.616. Between-tile variance rises as the shift shrinks (0.10-0.17
against 0.04-0.07 at shift 32): a code of the sample creeps back when the two views nearly coincide. Bigger cubes are not
ruled out, but this sweep cannot separate them from the smaller shift.</li>
<li><b>e05 reproduces on the B300.</b> Side 7 at 20k steps: 0.140 / 0.660, against e05's d7 at 20.7k steps on an RTX PRO 6000:
0.146 / 0.679. The B300 simply takes more steps in 2 h (31.8k).</li>
</ul>
<p><b>Next.</b> Hide about a quarter to a third of the tokens. To hide more without shrinking the shift, use a bigger view
(e06: smaller views probe better, so this trades against view size) or several target blocks per view, as I-JEPA does.</p>"""


def label(p: dict) -> str:
    """A trunk's name: its cube's side in tokens and its shift."""
    return f"side {round(p['mask_block'] * p['global_size'][0] / 8)} shift {p['mask_shift']}"


def main():
    tiles, sections = anticollapse.build(SWEEP, label=label)
    page.write(SWEEP, __doc__, tiles, [("Findings", [("What e07 shows", FINDINGS)]), *sections])


if __name__ == "__main__":
    main()
