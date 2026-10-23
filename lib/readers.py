"""Reading eval data as training reads it: held-out and probe-set windows of the EM volumes at 8 nm through miao (resampled,
normalized; labels nearest), with context around a box for tiled inference."""

import math

import numpy as np

import lmd_catalog as lmd
from miao import VolumeDataset
from miao.config import MiaoConfig

from lib.data import HEMIBRAIN_EB, ProbeVolume

PROBE_SIDE = 896  # probe: a block's largest side, 8 nm voxels (its box cut to it about its centre; hemibrain_eb's are exactly it)
PROBE_MARGIN = 16  # read_window: the extra read around a window, 8 nm voxels (miao's resampling and label-quantization slack)


def eb_val_crop(shape: tuple[int, ...]):
    """A shape (Z Y X) window of EB at (3400, 2000, 2000) z y x, inside its val slab (EB z 3000-4000), held out from
    training: 1 Z Y X in [0, 1], through miao (training's data path and normalization). miao needs the box strictly larger
    than the patch: one extra voxel per axis leaves exactly one window."""
    origin = (3400, 2000, 2000)
    assert origin[0] + shape[0] <= 4000, f"crop {shape} leaves EB's val slab"
    vol = lmd.get(HEMIBRAIN_EB).to_miao(bounding_box=[[o, o + s + 1] for o, s in zip(origin, shape)])
    mcfg = MiaoConfig(volumes=[vol], patch_size=list(shape), resolutions=[[8.0, 8.0, 8.0]],
                      samples_per_epoch=1, sampling="random", output_axes="lzyx")
    return VolumeDataset(mcfg)[0]["img"]

def probe_volume_config(vol: ProbeVolume, box: list[list[int]] | None, patch: list[int]) -> MiaoConfig:
    """miao reading patches (Z Y X, 8 nm voxels) of vol within box (Z Y X, level-0 voxels; None: anywhere), image and
    labels, on a grid (sequential): the first is the one nearest the box's low corner miao's read margins allow."""
    from miao.config import VolumeConfig
    lo, hi = vol.clip or (None, None)
    vc = VolumeConfig(name=vol.name, path=vol.path, image_key="raw", label_key=f"labels/{vol.label_key}", zarr_version="zarr3",
                      normalize_min=lo, normalize_max=hi, bounding_box=box)
    return MiaoConfig(volumes=[vc], patch_size=patch, resolutions=[[8.0, 8.0, 8.0]], samples_per_epoch=1, sampling="sequential", output_axes="lzyx")

def volume_grid(vol: ProbeVolume) -> tuple[np.ndarray, np.ndarray]:
    """(level-0 voxel size in nm, level-0 shape), each Z Y X, of vol's image, as miao resolves them (its OME transforms,
    expansion-corrected for ExM). Read off miao's VolumeInfo, which has no public accessor."""
    info = VolumeDataset(probe_volume_config(vol, None, [8, 8, 8]).model_copy(update={"sampling": "random"}))._volumes[0]  # no grid
    order = [info.img_spatial_axes.index(a) for a in "zyx"]
    return np.asarray(info.finest_voxel_size)[order], np.asarray(info.image_meta.scales[0].shape)[info.img_spatial_idx][order]

def read_window(vol: ProbeVolume, center, size: list[int]):
    """(img in [0, 1], labels int64), each Z Y X: size voxels at 8 nm about center (Z Y X, level-0 voxels) of vol, read as
    training reads (miao: resampled to 8 nm, normalized; labels nearest). What lies outside the volume is edge-replicated
    in the image and 0 (ignored) in the labels, as is the volume's outermost voxel (miao keeps a voxel's slack inside the
    box it reads from: checked on hemibrain, where 8 nm is native and every other voxel equals the stored one)."""
    import torch.nn.functional as F
    vox, shape = volume_grid(vol)
    want = np.array(size)
    low = center - want * 4 / vox  # the window's low corner, level-0 voxels
    # The box: the window and 2 margins (8 nm voxels) below it, 4 above, cut to the volume. miao's first read of it (its grid,
    # sequential) starts a little past the box's corner (its slack for resampling and a coarser label pyramid), under 2
    # margins before the window, so a patch 3 margins longer than the window covers it, if the box holds it with that
    # slack: at the volume's edge, the patch is cut to the most the box holds, the least slack miao accepts (it says how
    # little by refusing, its bounds being private). Then the window is cut out of the read, where it landed.
    lo = np.clip(np.floor(low - 2 * PROBE_MARGIN * 8 / vox), 0, shape).astype(int)
    hi = np.clip(np.ceil(low + (want + 4 * PROBE_MARGIN) * 8 / vox), 0, shape).astype(int)
    box = [[int(l), int(h)] for l, h in zip(lo, hi)]

    def first_read():  # (patch, its sample) at the least slack miao accepts
        for slack in (2, 4, 8, PROBE_MARGIN):
            patch = np.minimum(want + 3 * PROBE_MARGIN, ((hi - lo) * vox / 8).astype(int) - slack)
            try:
                return patch, VolumeDataset(probe_volume_config(vol, box, patch.tolist()))[0]
            except ValueError:  # "bounding_box ... too small to contain the requested window"
                pass
        assert False, f"{vol.name}: miao reads no window from {box}, even {PROBE_MARGIN} voxels smaller"

    patch, sample = first_read()
    img, lab = sample["img"][0], sample["label"][0].long()
    start = np.round((np.asarray(sample["bbox"][0][0]) / vox - low) * vox / 8).astype(int)  # the read's corner in the window
    a, b = np.clip(-start, 0, patch), np.clip(want - start, 0, patch)  # the read's part inside the window
    img, lab = img[a[0]:b[0], a[1]:b[1], a[2]:b[2]], lab[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
    before = np.maximum(start, 0)
    after = want - before - (b - a)  # what the volume doesn't hold, padded
    pad = [int(n) for lo_, hi_ in zip(before[::-1], after[::-1]) for n in (lo_, hi_)]  # F.pad: last axis first
    return F.pad(img[None, None], pad, mode="replicate")[0, 0], F.pad(lab, pad, value=0)

def inner(x, m, size):
    """x's (... Z Y X) box of size (Z Y X) at offset m (Z Y X): read_block's box, within its padded region."""
    return x[..., m[0]:m[0] + size[0], m[1]:m[1] + size[1], m[2]:m[2] + size[2]]

def read_block(vol: ProbeVolume, tile, p, device, side: int = PROBE_SIDE, context: bool = False):
    """vol's box cut to whole patches (p, Z Y X) at most side a side about its centre, read at 8 nm (read_window) with whole
    tiles of context around it: (img Z Y X [the padded region, a multiple of tile, on device], labels Z Y X [the box's],
    m [the box's offset in img, whole patches], size [the box's]). A model tiles img (tile_features, tile_logits); inner(x, m,
    size) trims what it makes, or img itself (the box's EM), back to the box. context: at least half a tile of it on every side,
    which tile_logits' blending needs."""
    vox, _ = volume_grid(vol)
    lo, hi = np.array([b[0] for b in vol.box][::-1]), np.array([b[1] for b in vol.box][::-1])  # z y x, level-0 voxels
    size = [int(min(s - s % q, side)) for s, q in zip(((hi - lo) * vox / 8).astype(int), p)]  # ints, not numpy's: json writes them
    region = [math.ceil((s + context * t) / t) * t for s, t in zip(size, tile)]  # whole tiles around the box
    m = [(r - s) // 2 // q * q for r, s, q in zip(region, size, p)]
    img, lab = read_window(vol, (lo + hi) / 2, region)
    return img.to(device), inner(lab, m, size).contiguous().to(device), m, size
