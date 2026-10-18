"""e00/e08-mae-fix: e06-viewsize-vs-baselines' MAE (view 96 of input 128, decoder 192 wide and 2 deep, lr 1e-4) probed
0.19-0.20 short AP, then went unstable: its loss rose after ~40k steps, its gradient norm and residual stream grew, its
effective rank fell. Two suspects, swept: the learning rate (1e-4, 3e-5, 1e-5, held after warmup; MAE's own rule gives
1.9e-5 at batch 32) and the decoder (mia-train's default 192 x 2, the MAE paper's 512 x 8, which it found gives a better
linear probe). 4 h on one B300 each, batch 32."""

from reports import anticollapse, page

SWEEP = "e00/e08-mae-fix"

# Read from the runs' pca.json, probe.json, metrics.json, performance.json and pca2.png (2026-10-18).
FINDINGS = """<ul>
<li><b>The MAE paper's decoder fixes it, and gives the best encoder of any objective so far.</b> Decoder 512 x 8 at lr 1e-4
ends at short / long AP 0.296 / 0.869 (random encoder 0.124 / 0.481), against 0.205 / 0.847 for e06's best masked-token
LeJEPA and 0.216 / 0.764 for the best invariance run. It passes 0.2 by 5k steps, reaches 0.278 at 20k and creeps up after
(0.290 at 40k, 0.296 at 117k). Its training is smooth: gradient norm ~0.07 throughout (1-6 in every other run), loss
0.51-0.55, residual stream 64, effective rank rising from 157 to 328. Its PCA maps draw membranes as their own colour and
cell interiors as textured regions, with no position code.</li>
<li><b>A lower learning rate does not fix it.</b> With the small decoder every lr peaks and then falls: 0.219 at 20k steps
(1e-4), 0.215 at 80k (3e-5) and 0.186 at 80k (1e-5), ending at 0.136, 0.125 and 0.153. At 1e-4 and 3e-5 the loss turns
upward (to 0.92), with gradient spikes of 43 and 273; at 1e-5 the loss keeps falling, yet AP still drops and the PCA maps
are a code of token position with scattered high-norm tokens. The big decoder needs the high lr: 0.234 at 3e-5 (still
rising, a gradient spike of 570) and 0.125 at 1e-5 (loss rising from 80k).</li>
<li><b>At matched compute the MAE is now well ahead of LeJEPA.</b> The 512 x 8 decoder costs 0.53 TFLOP a sample (0.14 with the
small one; masked view 64 0.30, view 96 1.35). At ~1.3 EFLOP the MAE probes 0.294 against masked view 64's 0.205, and it
reaches 0.205 at 0.08 EFLOP, ~15x less.</li>
<li>d0 repeats e06's MAE: peak 0.219 here against 0.194 there, the same decline after.</li>
</ul>
<p><b>Next.</b> The MAE baseline is decoder 512 x 8 at lr 1e-4. LeJEPA's analogue of the decoder is the masked-token
predictor (4 blocks at the encoder's width): a deeper or wider predictor is the first thing to try on that side. And this
probe reads boundaries, a low-level target an MAE is built for; the semantic (CellMap) and tracing (NERL) evals decide
the comparison Jakob asked for.</p>"""


def label(p: dict) -> str:
    """A trunk's name: its decoder (width x depth) and learning rate."""
    width, depth = p["mae_decoder"]
    return f"MAE decoder {width}x{depth} lr {p['lr']:g}"


def main():
    tiles, sections = anticollapse.build(SWEEP, label=label)
    page.write(SWEEP, __doc__, tiles, [("Findings", [("What e08 shows", FINDINGS)]), *sections])


if __name__ == "__main__":
    main()
