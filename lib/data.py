"""The datasets: each a few crops of the lmd catalog, cut into volumes by what they are for. Every dataset has the same parts:
  train  the crops' images that pretraining samples, never any volume below
  val    held-out images: the pretraining loss is measured on them, and the PCA maps are drawn on the first
  fit    a labelled volume the readout (the linear probe, the decoder) is fit on          } instance labels, scored as mia-evals'
  test   labelled volumes the fitted readout is scored on, averaged                       } neuron-instance task; a dataset without
  tune   a labelled volume where mia-evals chooses its post-processing's parameters, for `score` } them can't be read out
Every volume is read at 8 nm (miao resamples), whatever it is stored at."""

from __future__ import annotations

import math
from typing import NamedTuple

LMD_DATA = "/groups/miaai/miaai/lmd-v0.0.1/data"  # the catalog's stores, <crop>.zarr

Box = list[list[int]]  # [lo, hi) level-0 voxels, x y z


class Volume(NamedTuple):
    """A box of one catalog crop; labelled if it has instance labels."""
    crop: str  # catalog name
    box: Box
    label: str = ""  # labels/<label> of the store
    clip: tuple[float, float] | None = None  # uint16 images: the [lo, hi] mapped to [0, 1] (miao's normalize_min / _max)
    sparse: bool = False  # labels are whole cells but not every cell: labelled against unlabelled is a boundary (lib.probe.affinities)
    annotated: Box | None = None  # mia-evals' annotated box around it (its artifact's provenance)
    name: str = ""  # its name in mia-evals' data config, which names its artifact: only a volume mia-evals scores has one

    @property
    def path(self) -> str:
        return f"{LMD_DATA}/{self.crop}.zarr"

    @property
    def zyx(self) -> Box:
        return self.box[::-1]  # miao's order

    @property
    def voxels(self) -> int:
        return math.prod(hi - lo for lo, hi in self.box)


class Dataset(NamedTuple):
    train: list[Volume]
    val: list[Volume]
    fit: Volume | None = None
    test: tuple[Volume, ...] | list[Volume] = ()
    tune: Volume | None = None


def volumes(crops: dict[str, Box]) -> list[Volume]:
    return [Volume(crop, box) for crop, box in crops.items()]


def sparse(crop: str, label: str, box: Box, clip: tuple[float, float] | None = None) -> Volume:
    """mia-evals' lmd_ssl_v1 blocks (configs/lmd_ssl_v1_neuron_instance/data, copied 2026-10-18; boxes and clip ranges as there). Their
    labels (proofread, or manual "final") label some cells whole: scored on labelled pairs only, +1 voxel boundaries were 0.1-1% of
    pairs and every encoder's short AP sat at that base rate."""
    return Volume(crop, box, label, clip, sparse=True)


# FlyEM hemibrain, 8 nm, global voxels (x y z): the Ellipsoid Body crop-001 (x 24000-29000 y 23000-28000 z 17000-22000) is
# inside crop-002 (11k, x 18000-29000 y 17000-28000 z 11000-22000), beside crop-003 (10k, x 8000-18000 y 17000-27000 z 11000-21000).
# crop-002 is cut at its z < 9000 (global z < 20000), so EB's z >= 3000 (the three labelled blocks, mia-evals' gary_comparison
# blocks and ours) never enter training. fit is ours, in EB's val slab (z 3000-4000); tune and test are mia-evals'
# (docs/scoring_third_party_affinities.md: tune fits the size filter, test is reported).
EB = "em-drosophila-flyem-hemibrain/crop-001_EllipsoidBody_x24000_y23000_z17000"
EB_LABEL = "proofread-cell-hemibrain-v1.2"
HEMIBRAIN_FIT = Volume(EB, [[3052, 3948], [3052, 3948], [3052, 3948]], EB_LABEL)
HEMIBRAIN_TEST = Volume(EB, [[4052, 4948], [4052, 4948], [4052, 4948]], EB_LABEL, annotated=[[4000, 5000], [4000, 5000], [4000, 5000]], name="hemibrain_eb_test")
HEMIBRAIN_TUNE = Volume(EB, [[4052, 4948], [4052, 4948], [3052, 3948]], EB_LABEL, annotated=[[4000, 5000], [4000, 5000], [3000, 4000]], name="hemibrain_eb_fit")

# lmd_ssl_v1's other families, read at 8 nm: fish2 8 x 8 x 7.5 nm, LICONN's 9.7 x 9.7 x 25.9 nm (expansion-corrected), MICrONS' 8 x 8 x 40 nm.
FISH2_FIT = sparse("em-zebrafish-fish2/crop-003_quadcube1_x23908_y18314_z0", "proofread-cell-seg230927", [[0, 2000], [0, 2000], [0, 1364]])
FISH2_TEST = sparse("em-zebrafish-fish2/crop-001_doublecube1_x52000_y42500_z0", "proofread-cell-seg230927", [[0, 2000], [0, 2000], [0, 2112]])
LICONN_FIT = sparse("exm-mouse-liconn-DG-20250809_ExPID19-02_2ndGel_C5_Atto488_40XW_002/crop-001", "manual_gt-cell-final", [[1416, 1892], [516, 996], [83, 276]], (301, 1130))
LICONN_TEST = [
    sparse("exm-mouse-liconn-hippocampus-202507013_ExPID10-02_2percentAA_S2_Atto488_40XW_004/crop-001", "manual_gt-cell-final", [[496, 976], [496, 976], [44, 236]], (219, 919)),
    sparse("exm-mouse-liconn-ExPID82-1/crop-002_sub", "proofread-cell-agglom-dense-v20240123", [[1229, 2253], [1642, 2666], [0, 534]]),
]
# Kasthuri15 (mouse cortex EM, 0.3 Gvox) is all eval volumes; MICrONS trains the dataset, the closest in tissue and anisotropy.
KASTHURI_FIT = sparse("em-mouse-Kasthuri15-ac3ac4-cortex/crop-001_ac3_100slices", "public_gt-cell-kasthuri15", [[0, 1024], [0, 1024], [0, 100]])
KASTHURI_TEST = sparse("em-mouse-Kasthuri15-ac3ac4-cortex/crop-002_ac4_100slices", "public_gt-cell-kasthuri15", [[0, 1024], [0, 1024], [0, 100]])

DATASETS: dict[str, Dataset] = {
    "hemibrain": Dataset(  # ~2.1 Tvox
        train=volumes({
            "em-drosophila-flyem-hemibrain/crop-002_11k_x18000_y17000_z11000": [[0, 11000], [0, 11000], [0, 9000]],
            "em-drosophila-flyem-hemibrain/crop-003_10k_x8000_y17000_z11000": [[0, 10000], [0, 10000], [0, 10000]],
        }),
        val=[HEMIBRAIN_FIT, HEMIBRAIN_TEST], fit=HEMIBRAIN_FIT, test=[HEMIBRAIN_TEST], tune=HEMIBRAIN_TUNE),
    "fish2": Dataset(  # zebrafish EM, ~25 Gvox: every crop but the fit and test ones
        train=volumes({
            "em-zebrafish-fish2/crop-002_doublecube2_x120000_y45500_z0": [[0, 2000], [0, 2000], [0, 2120]],
            "em-zebrafish-fish2/crop-004_quadcube2_x15114_y10046_z0": [[0, 2000], [0, 2000], [0, 1364]],
            "em-zebrafish-fish2/crop-005_quadcube3_x9638_y10314_z0": [[0, 2000], [0, 2000], [0, 1364]],
            "em-zebrafish-fish2/crop-006_quadcube4_x48294_y6026_z0": [[0, 2000], [0, 2000], [0, 1364]],
        }),
        val=[FISH2_FIT, FISH2_TEST], fit=FISH2_FIT, test=[FISH2_TEST]),
    "liconn_mouse": Dataset(  # mouse LICONN ExM, ~666 Gvox: the three animals with no eval volume (ExPID82-1's full volume is left out: its crop-002 is a test block)
        train=volumes({
            "exm-mouse-liconn-ExPID124/crop-001": [[0, 39000], [0, 10000], [0, 752]],
            "exm-mouse-liconn-ExPID146/crop-001": [[0, 10762], [0, 10680], [0, 1695]],
            "exm-mouse-liconn-ExPID150/crop-001": [[0, 4900], [0, 4700], [0, 7720]],
        }),
        val=[LICONN_FIT, *LICONN_TEST], fit=LICONN_FIT, test=LICONN_TEST),
    "microns": Dataset(  # mouse cortex EM, MICrONS minnie65, 69 Gvox; read out on Kasthuri15
        train=volumes({"em-mouse-MICrONS-minnie65/crop-001": [[0, 8192], [0, 8192], [0, 1024]]}),
        val=[KASTHURI_FIT, KASTHURI_TEST], fit=KASTHURI_FIT, test=[KASTHURI_TEST]),
    # CellMap's FIB-SEM organelle volumes (HHMI Janelia: cells, tissues, 4-8 nm), ~10 Tvox at 8 nm, uint8. Train: each volume's largest slab
    # (x y z, whole in the other two axes) with no manual organelle crop in it (labels/manual_gt-*), so the regions CellMap's organelle
    # labels cover stay out of training; the labelled-crop boxes are those in the catalog's labels, not CellMap's unreleased test crops.
    # Left out: volumes with under 0.5 Gvox free at 8 nm (nacc-1, cos7). Val: the first (up to) 512 voxels at 8 nm a side of each
    # volume's largest manual organelle crop, where one holds a 128 window. CellMap's labels aren't affinities: no fit, test or tune.
    "cellmap": Dataset(
        train=volumes({  # Gvox at 8 nm
            "em-drosophila-CellMap-jrc-fly-mb-1a/crop-001_fullvol_tissuecrop": [[1087, 3717], [0, 3430], [0, 7723]],  # 8.7
            "em-drosophila-CellMap-jrc-fly-vnc-1/crop-001_fullvol_tissuecrop": [[1363, 3488], [0, 2067], [0, 1471]],  # 0.8
            "em-human-CellMap-jrc-hela-2/crop-001_fullvol_tissuecrop": [[8580, 10333], [0, 1312], [0, 5742]],  # 2.2
            "em-human-CellMap-jrc-hela-3/crop-001_fullvol_tissuecrop": [[0, 10596], [0, 931], [5039, 8953]],  # 3.9
            "em-human-CellMap-jrc-jurkat-1/crop-001_fullvol_tissuecrop": [[3312, 5014], [0, 2840], [0, 4212]],  # 2.2
            "em-human-CellMap-jrc-macrophage-2/crop-001_fullvol_tissuecrop": [[0, 9526], [0, 1590], [5147, 6562]],  # 2.3
            "em-human-CellMap-jrc-sum159-1/crop-001_fullvol_tissuecrop": [[8334, 13515], [0, 2469], [0, 4067]],  # 7.4
            "em-human-CellMap-jrc-sum159-4/crop-001_fullvol_tissuecrop": [[2986, 6865], [0, 847], [0, 3772]],  # 12.4
            "em-human-CellMap-jrc-ut21-1413-003/crop-001_fullvol_tissuecrop": [[0, 9996], [307, 3007], [0, 16301]],  # 440.0
            "em-mouse-CellMap-jrc-ctl-id8-1/crop-001_fullvol_tissuecrop": [[693, 3593], [0, 1895], [0, 6428]],  # 3.8
            "em-mouse-CellMap-jrc-mus-heart-1/crop-001_fullvol_tissuecrop": [[0, 11226], [0, 3528], [0, 6591]],  # 261.0
            "em-mouse-CellMap-jrc-mus-kidney-3/crop-001_fullvol_tissuecrop": [[0, 6105], [0, 5202], [3674, 7530]],  # 122.5
            "em-mouse-CellMap-jrc-mus-kidney-glomerulus-2/crop-001_fullvol_tissuecrop": [[2734, 3389], [0, 3346], [0, 2421]],  # 0.7
            "em-mouse-CellMap-jrc-mus-kidney/crop-001_fullvol_tissuecrop": [[0, 9867], [0, 7190], [616, 4252]],  # 258.0
            "em-mouse-CellMap-jrc-mus-liver-3/crop-001_fullvol_tissuecrop": [[5128, 9747], [0, 7585], [0, 8778]],  # 307.5
            "em-mouse-CellMap-jrc-mus-liver-zon-1/crop-001_fullvol_tissuecrop": [[0, 20208], [0, 15828], [30060, 39660]],  # 3070.6
            "em-mouse-CellMap-jrc-mus-liver-zon-2/crop-001_fullvol_tissuecrop": [[0, 21800], [0, 18280], [6584, 17740]],  # 4445.7
            "em-mouse-CellMap-jrc-mus-liver/crop-001_fullvol_tissuecrop": [[0, 10037], [7855, 10174], [0, 8810]],  # 205.1
            "em-zebrafish-CellMap-jrc-zf-cardiac-1/crop-001_fullvol_tissuecrop": [[0, 7000], [5200, 10400], [0, 23720]],  # 863.4
        }),
        val=volumes({
            "em-human-CellMap-jrc-hela-2/crop-001_fullvol_tissuecrop": [[7454, 7854], [583, 983], [1415, 1815]],
            "em-human-CellMap-jrc-sum159-4/crop-001_fullvol_tissuecrop": [[1006, 1518], [47, 559], [2076, 2588]],
            "em-human-CellMap-jrc-ut21-1413-003/crop-001_fullvol_tissuecrop": [[3605, 3805], [8877, 9077], [706, 906]],
            "em-mouse-CellMap-jrc-mus-heart-1/crop-001_fullvol_tissuecrop": [[7248, 7760], [3528, 4040], [908, 1420]],
            "em-mouse-CellMap-jrc-mus-kidney-3/crop-001_fullvol_tissuecrop": [[2418, 2930], [1858, 2370], [2474, 2986]],
            "em-mouse-CellMap-jrc-mus-kidney-glomerulus-2/crop-001_fullvol_tissuecrop": [[534, 1558], [632, 1656], [86, 1110]],
            "em-mouse-CellMap-jrc-mus-kidney/crop-001_fullvol_tissuecrop": [[2820, 3332], [3178, 3690], [4252, 4764]],
            "em-mouse-CellMap-jrc-mus-liver-3/crop-001_fullvol_tissuecrop": [[3928, 4440], [3262, 3774], [3854, 4366]],
            "em-mouse-CellMap-jrc-mus-liver-zon-1/crop-001_fullvol_tissuecrop": [[11808, 12320], [6600, 7112], [9188, 9700]],
            "em-mouse-CellMap-jrc-mus-liver-zon-2/crop-001_fullvol_tissuecrop": [[7000, 7512], [5480, 5992], [25740, 26252]],
            "em-mouse-CellMap-jrc-mus-liver/crop-001_fullvol_tissuecrop": [[130, 642], [162, 674], [680, 1192]],
            "em-zebrafish-CellMap-jrc-zf-cardiac-1/crop-001_fullvol_tissuecrop": [[2000, 2512], [0, 512], [15520, 16032]],
        })),
}


def dataset(name: str) -> Dataset:
    assert name in DATASETS, f"no dataset {name!r}: {list(DATASETS)}"
    return DATASETS[name]


def sampling(names: tuple[str, ...], power: float) -> list[tuple[Volume, float]]:
    """Each train volume of the named datasets with its sampling weight, summing to 1. A dataset's share goes as its voxel count
    ^ power, and a volume's within its dataset as its own ^ power (native voxels, not 8 nm's): power 0 samples datasets equally and
    each one's volumes equally, 1 by size, 0.4 flattens CellMap's 0.7 to 4,400 Gvox volumes and hemibrain's 2.1 Tvox against fish2's 25 Gvox."""
    assert names, "no training data: name datasets"
    train = {n: dataset(n).train for n in names}
    share = {n: sum(v.voxels for v in vs) ** power for n, vs in train.items()}
    return [(v, share[n] / sum(share.values()) * v.voxels ** power / sum(u.voxels ** power for u in vs)) for n, vs in train.items() for v in vs]
