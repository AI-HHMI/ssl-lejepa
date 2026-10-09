"""Free disk in the cluster's outdir/ by deleting data that's cheap to recreate: the probe's affinity artifacts,
savedir/probe/{fit,test}/*.zarr (~9 GB per run, ~0.9 TB of e00's 1.1 TB). probe(n) rewrites them from the same
checkpoint in ~10 GPU minutes, identically (its fits are seeded), from the run's own commit (runs.json). Only artifacts
whose source checkpoint still exists (or that probed a random init) go; probe.json, probe.png and mia-evals records stay.
Re-probe before re-scoring: score() reads these artifacts. Run on the cluster, from the repo root:
  uv run python cleanup.py zarrs          # list what would be deleted, with sizes
  uv run python cleanup.py delete_zarrs   # delete it
"""

import shutil
import sys
from pathlib import Path

from analysis import saved_params
from lib.util import call_entrypoint, pick_entrypoint


def size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9

def recreatable() -> list[Path]:
    """Every probe artifact under outdir/ (not .trash/) whose source checkpoint still exists: the run's own, or its
    init_from run's; a random init needs none."""
    out = []
    for z in sorted(Path("outdir").glob("*/*/d*/probe/*/*.zarr")):  # outdir/<series>/<sweep>/dN/probe/<split>/*.zarr
        if z.parts[1] == ".trash":
            continue
        run = z.parents[2]
        init = saved_params(run).get("init_from", "")
        b1 = init == "random"
        b2 = any((Path(init) if init else run).glob("checkpoints/*.pt"))
        if b1 or b2:
            out.append(z)
    return out

def zarrs():
    """List the recreatable probe artifacts and what deleting them would free."""
    total = 0.0
    for z in recreatable():
        gb = size_gb(z)
        total += gb
        print(f"{gb:7.1f} GB  {z}")
    print(f"{total:.0f} GB in recreatable probe artifacts; `python cleanup.py delete_zarrs` deletes them")

def delete_zarrs():
    """Delete the recreatable probe artifacts (zarrs())."""
    zs = recreatable()
    for z in zs:
        shutil.rmtree(z)
        print(f"deleted {z}")
    print(f"deleted {len(zs)} probe artifacts")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        pick_entrypoint()
    else:
        call_entrypoint(sys.argv[1], *sys.argv[2:])
