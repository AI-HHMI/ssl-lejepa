"""The cross-experiment performance page, results/perf_journey.html: fills in its run-derived numbers (the steps /
phases / scaling arrays between the BEGIN/END perf_journey() markers in its <script>); its tiles and prose are
hand-written. Usage: uv run --extra analysis python -m reports.perf_journey
"""

import json
from pathlib import Path

from analysis import read_jsonl


def perf_journey():
    """Write the run-derived numbers in results/perf_journey.html: its steps / phases / scaling arrays, between the
    BEGIN/END perf_journey() markers in its <script>. Tiles and prose there are still hand-written.

    Bars are listed here by hand, per phase (a phase = one GPU type and setup). Each bar's ktok/s per GPU, total
    tok/s and MFU come from its run's last throughput row in performance.json (run ./pull.sh first). Its change
    label compares it with the `vs` run: same GPU count -> "+14%" (or "x3.1" at 2x+); 8 GPUs vs 1 -> per-GPU
    scaling efficiency, "94%".
    """
    B3, H2 = "B300", "H200"
    phases = [  # (label, [(bar label with | line breaks, run, GPU type, vs run, notes)])
        ("B300 · flyliconn, basic views", [
            ("fp32", "compile-amp-tok_s/d0", B3, None, "fp32, batch 42, 4 workers, eager. Attention on fp32 mem-efficient kernel."),
            ("+ bf16", "compile-amp-tok_s/d5", B3, "compile-amp-tok_s/d0", "bf16 autocast (SIGReg fp32), batch 84. cuDNN flash attention."),
            ("+ compile", "compile-amp-tok_s/d7", B3, "compile-amp-tok_s/d5", "torch.compile(dynamic=True) on the encoder. Now data-loader bound (~170 samples/s)."),
            ("+ 8|workers", "ddp/d0", B3, "compile-amp-tok_s/d7", "8 DataLoader workers: data wait gone, 88% GPU busy (≈ workers/d1)."),
            ("DDP", "ddp/d3", B3, "ddp/d0", "8×B300 with DDP."),
        ]),
        ("H200 · hemibrain EB 128³, displace views", [
            ("new|setup", "cudagraphs/d0", H2, None, "H200, hemibrain EB 128³, displace 96³/64³, compile(dynamic), workers active through profile."),
            ("+ CUDA|graphs", "cudagraphs/d4", H2, "cudagraphs/d0", "compile(mode='reduce-overhead'): replay recorded kernel sequences. 99% GPU busy."),
            ("+ batch|views", "batchviews/d1", H2, "cudagraphs/d4", "2 encoder calls per step (all globals, all locals) instead of 6."),
            ("+ uint8", "width-defer/d3", H2, "batchviews/d1", "defer_image_ops: workers ship uint8, GPU normalizes."),
            ("+ Linear|embed", "patchembed-linear/d2", H2, "width-defer/d3", "Patch embedding as reshape + one Linear instead of Conv3d."),
            ("DDP", "cudagraphs/d7", H2, "cudagraphs/d4", "8×H200 with CUDA graphs, 72 cores."),
            ("+ full|node", "allreduce/d2", H2, "allreduce/d0", "12 cores/GPU → all 96 cores. bf16 grad compression: no further gain."),
            ("+ batch|views", "batchviews/d3", H2, "batchviews/d1", "Batched encoder calls at 8 GPUs."),
            ("+ uint8", "width-defer/d5", H2, "width-defer/d3", "uint8 transfer at 8 GPUs."),
        ]),
        ("B300 · H200 setup", [
            ("H200|setup", "b300-revisit/d0", B3, None, "Everything from the H200 phase, on one B300. Conv3d patch embed ~15–17% of GPU time."),
            ("DDP", "b300-revisit/d1", B3, "b300-revisit/d0", "8×B300."),
            ("+ Linear|embed", "patchembed-linear/d0", B3, "b300-revisit/d0", "Patch embedding as reshape + one Linear (cuDNN attention)."),
            ("safe|stack", "cudagraph-fix/d3", B3, "patchembed-linear/d0", "What training runs now: flash attention (cuDNN's "
             "backward returned NaN grads, nanhunt_flash) and the patch embed outside the compiled graph (its Triton kernel "
             "miscompiled under cudagraphs, b300-compile). Nearly all of the cost is flash."),
        ]),
    ]
    scaling = [  # (label, run, GPU type, 1-GPU run whose 8x is ideal)
        ("B300 old · 1 GPU", "ddp/d0", B3, None), ("B300 old · 8 GPUs", "ddp/d3", B3, "ddp/d0"),
        ("H200 · 1 GPU", "width-defer/d3", H2, None), ("H200 · 8 GPUs", "width-defer/d5", H2, "width-defer/d3"),
        ("B300 · 1 GPU", "b300-revisit/d0", B3, None), ("B300 · 8 GPUs", "b300-revisit/d1", B3, "b300-revisit/d0"),
    ]

    def perf(run):  # (total tok/s, n_gpus, mfu or None) from the run's last throughput row
        f = Path("outdir/e00") / run / "performance.json"
        tp = [r for r in read_jsonl(f) if r["tbl"] == "throughput"]
        assert tp, f"no throughput row in {f}; run ./pull.sh?"
        r = tp[-1]
        return r["tokens_per_second"], r.get("world_size") or r["params"].get("n_gpus", 1), r.get("mfu")  # pre-DDP: 1 GPU

    steps = []
    for _, bars in phases:
        for k, run, hw, vs, d in bars:
            tok, g, mfu = perf(run)
            step = {"k": k, "g": g, "v": round(tok / g / 1e3), "hw": hw, "run": run,
                    "d": d + (f" {tok / 1e6:.2f}M tok/s total." if g > 1 else "") + (f" {100 * mfu:.1f}% MFU." if mfu else "")}
            if vs:
                vs_tok, vs_g, _ = perf(vs)
                r = (tok / g) / (vs_tok / vs_g)
                step["x"] = f"{r:.0%}" if g != vs_g else f"×{r:.1f}" if r >= 2 else f"{r - 1:+.0%}"
                if g != vs_g:  # the 1-GPU run this bar's scaling % is against, drawn as an outline behind it
                    step["ref"] = round(vs_tok / vs_g / 1e3)
            steps.append(step)
    nodes = []
    for k, run, hw, vs in scaling:
        tok, g, _ = perf(run)
        nodes.append({"k": k, "v": round(tok / 1e6, 3), "hw": hw, "run": run} | ({"ideal": round(g * perf(vs)[0] / 1e6, 3)} if vs else {}))

    path = Path("results/perf_journey.html")
    html = path.read_text()
    BEGIN, END = "// BEGIN perf_journey() data", "// END perf_journey() data"
    assert html.count(BEGIN) == 1 and html.count(END) == 1, f"{path} needs one {BEGIN!r} ... {END!r} block"
    rows = lambda xs: "[\n" + ",\n".join("  " + json.dumps(x, ensure_ascii=False) for x in xs) + ",\n]"
    data = (f"{BEGIN}: written by reports/perf_journey.py perf_journey(); edit the lists there, not here.\n"
            f"const steps = {rows(steps)};\n"
            f"// One label per phase; a new phase starts wherever the GPU type changes.\n"
            f"const phases = {json.dumps([label for label, _ in phases], ensure_ascii=False)};\n"
            f"const scaling = {rows(nodes)};\n")
    path.write_text(html[:html.index(BEGIN)] + data + html[html.index(END):])
    print(f"Wrote {len(steps)} bars and {len(nodes)} node results to {path}")


if __name__ == "__main__":
    perf_journey()
