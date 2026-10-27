"""Cluster side of reports/metric_examples.py: a real hemibrain EB test crop with eight kinds of mistake added to its
ground-truth labels one at a time, each scored with mia-evals' voxel_instance metrics (PQ, SQ, RQ, VOI split and merge, adapted
Rand error) and with boundary AP as a hard prediction (the probe's metric: short is the mean of the +1 channels, long of the +10),
and drawn against the EM. Run where the data is, from the experiment checkout (it imports lib):
  PYTHONPATH=. uv run python metric_examples.py   -> outdir/e00/metric-examples/{<case>.png, metrics.json} (pull.sh brings them)"""

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path("outdir/e00/metric-examples")
SIDE = 128  # the crop's side, 8 nm voxels
SCALE = 3  # panel pixels per voxel
MIA_EVALS = Path.home() / "proj/mia-evals/src"
METRICS = ["pq", "sq", "rq", "voi_split", "voi_merge", "adapted_rand_error", "ap_short", "ap_long"]
CASES = {  # name: what was done to the labels
    "exact": "The ground truth itself: every metric at its best.",
    "shifted": "Every object moved 2 voxels in x: all boundaries in the right place up to 2 voxels, no object wrong.",
    "eroded": "A 1-voxel gap (label 0) between touching objects, as thresholding or watershed leave: objects slightly small.",
    "split": "The largest object cut in two by an x plane: one split.",
    "merge": "The largest object merged with the neighbour it touches most: one merge.",
    "shattered": "Every object cut into pieces by a 32-voxel grid: many splits, each piece a plausible object.",
    "dust": "300 small (2^3) fragments of new ids inside objects: a speck is a false object.",
    "missed small": "The smaller half of the objects (by count) deleted: tiny share of voxels, half of the objects.",
}


def edit(name: str, lab: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """lab (Z Y X, compact ids, 0 background) with mistake name made; and the z to draw it at."""
    pred, zc, k = lab.copy(), SIDE // 2, lab.max()
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    big = sizes.argmax()
    if name == "shifted":
        pred = np.roll(lab, 2, axis=2)
        pred[:, :, :2] = 0
    elif name == "eroded":
        same = np.ones(lab.shape, bool)
        for ax in range(3):
            a, b = [slice(None)] * 3, [slice(None)] * 3
            a[ax], b[ax] = slice(1, None), slice(None, -1)
            ne = lab[tuple(a)] != lab[tuple(b)]
            same[tuple(a)] &= ~ne
            same[tuple(b)] &= ~ne
        pred[~same] = 0
    elif name == "split":
        x = np.arange(SIDE)[None, None, :]
        pred[(lab == big) & (x >= np.median(np.nonzero((lab == big).any((0, 1)))[0]))] = k + 1  # an x plane: a line in the slice
        zc = int(np.median(np.nonzero((lab == big).any((1, 2)))[0]))
    elif name == "merge":
        left, right = lab[:, :, :-1], lab[:, :, 1:]
        touch = (left != right) & (left > 0) & (right > 0) & ((left == big) | (right == big))
        other = np.where(left[touch] == big, right[touch], left[touch])
        mate = np.bincount(other).argmax()
        pred[lab == mate] = big
        zc = int(np.median(np.nonzero((lab == mate).any((1, 2)))[0]))
    elif name == "shattered":
        z, y, x = np.indices(lab.shape) // 32
        pred = np.where(lab > 0, lab * 64 + z * 16 + y * 4 + x, 0)
    elif name == "dust":
        for i in range(300):
            z, y, x = [int(v) for v in rng.integers(0, SIDE - 2, 3)]
            pred[z:z + 2, y:y + 2, x:x + 2] = k + 1 + i
        zc = int(np.bincount(np.nonzero(pred > k)[0]).argmax())
    elif name == "missed small":
        present = np.nonzero(sizes)[0]
        pred[np.isin(lab, present[sizes[present] <= np.median(sizes[present])])] = 0
    return pred, zc


def score(lab: np.ndarray, pred: np.ndarray) -> dict:
    """The mistakes' metrics: mia-evals' voxel_instance and the hard prediction's boundary AP over the truth's valid pairs."""
    import torch
    sys.path.insert(0, str(MIA_EVALS))
    from metrics.voxel_instance import VoxelInstance
    from lib.probe import AFFINITY_OFFSETS_XYZ, affinities, average_precision
    s = VoxelInstance()(pred, lab)
    offsets = [o[::-1] for o in AFFINITY_OFFSETS_XYZ]
    t, valid = affinities(torch.from_numpy(lab), offsets)
    q, _ = affinities(torch.from_numpy(pred), offsets)
    ap = [average_precision(1 - q[c][valid[c]].float(), 1 - t[c][valid[c]].float()) for c in range(6)]
    return {**s, "ap_short": sum(ap[:3]) / 3, "ap_long": sum(ap[3:]) / 3}


def colors(lab: np.ndarray) -> np.ndarray:
    """lab (Y X) as RGB: a fixed random colour per id (mod 4096), black for 0."""
    palette = np.random.default_rng(0).integers(50, 255, (4096, 3)).astype(np.uint8)
    palette[0] = 0
    return palette[lab % 4096]


def edges(lab: np.ndarray) -> np.ndarray:
    """Where a voxel's label differs from its next voxel's in x or y (Y X bool)."""
    e = np.zeros(lab.shape, bool)
    e[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    e[:-1, :] |= lab[:-1, :] != lab[1:, :]
    return e


def strip(path: Path, em: np.ndarray, lab: np.ndarray, pred: np.ndarray, z: int):
    """Slice z of EM | truth | labelling | its boundaries against the truth's (green both, red the truth's only: a missed
    boundary, blue the labelling's only: an extra one)."""
    from PIL import Image
    gt, p = edges(lab[z]), edges(pred[z])
    err = np.zeros((*gt.shape, 3), np.uint8)
    err[gt & p] = (60, 200, 60)
    err[gt & ~p] = (230, 40, 40)
    err[~gt & p] = (50, 90, 240)
    panels = [np.repeat((em[z][..., None] * 255).astype(np.uint8), 3, -1), colors(lab[z]), colors(pred[z]), err]
    gap = np.full((SIDE, 4, 3), 255, np.uint8)
    row = np.concatenate([x for q in panels for x in (q, gap)][:-1], 1)
    Image.fromarray(row).resize((row.shape[1] * SCALE, row.shape[0] * SCALE), Image.Resampling.NEAREST).save(path)


def make():
    """Read the crop and write OUT/<case>.png and OUT/metrics.json."""
    from lib.data import probe_set
    from lib.readers import read_window
    vol = probe_set("hemibrain_eb")["test"][0]
    center = np.array([(b[0] + b[1]) / 2 for b in vol.box][::-1])  # z y x, level-0 voxels
    em, lab = read_window(vol, center, [SIDE] * 3)
    em, lab = em.numpy(), lab.numpy()
    lab = np.unique(lab, return_inverse=True)[1].reshape(lab.shape) + 1 - (lab == 0).any()  # compact ids, 0 stays background
    OUT.mkdir(parents=True, exist_ok=True)
    rng, rows = np.random.default_rng(0), []
    for name, what in CASES.items():
        pred, z = edit(name, lab, rng) if name != "exact" else (lab, SIDE // 2)
        strip(OUT / f"{name.replace(' ', '_')}.png", em, lab, pred, z)
        rows.append({"case": name, "what": what, "objects": int(len(np.unique(lab)) - 1), "z": z, **score(lab, pred)})
        print(name, {k: round(rows[-1][k], 3) for k in METRICS}, flush=True)
    (OUT / "metrics.json").write_text(json.dumps(rows) + "\n")


if __name__ == "__main__":
    make()
