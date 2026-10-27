"""Training/eval volume definitions (lmd catalog + miao): hemibrain, and the four families of mia-evals' lmd_ssl_v1.
A dataset is a few crops, each with raw and labels, and has the same three views here, by its name:
  TRAIN_BOXES / train_volumes(name)   its crops' images that pretraining samples (never its eval blocks)
  PROBE_SETS / probe_set(name)        labelled blocks: "train" (the probe or decoder fits on it), "test" (scored), and for a set
                                      mia-evals scores, "fit" (tunes its post-processing)
  HELD_OUT / held_out(name)           a region no training sees, and the window of it that PCA maps are drawn on"""

from __future__ import annotations

import math
from typing import NamedTuple

from lib.types import TrainData  # keys of TRAIN_BOXES, or JOINT_DATA

DEFAULT_DATA = "hemibrain_eb"  # the dataset a spec trains on, probes and maps unless it says otherwise (every run so far)

LMD_DATA = "/groups/miaai/miaai/lmd-v0.0.1/data"  # the catalog's stores, <catalog name>.zarr

# FlyEM hemibrain Ellipsoid Body splits from orhane's gary_comparison runs.
HEMIBRAIN_EB = "em-drosophila-flyem-hemibrain/crop-001_EllipsoidBody_x24000_y23000_z17000"
HEMIBRAIN_EB_BOXES = {  # [lo, hi) level-0 voxels, x y z
    "train": [[0, 5000], [0, 5000], [0, 3000]],
    "val": [[0, 5000], [0, 5000], [3000, 4000]],
    "test": [[4000, 5000], [4000, 5000], [4000, 5000]],
}
# Linear affinity probe (experiment.probe): 896^3 blocks of EB, [lo, hi) level-0 voxels, x y z. fit and test are
# mia-evals' gary_comparison_neuron_instance blocks (docs/scoring_third_party_affinities.md: fit tunes the size filter,
# test is reported); the probe itself trains on a third block in our val slab, so neither mia-evals block is its
# training data. None of the three is in our encoder's training region (EB z < 3000).
HEMIBRAIN_EB_LABELS = "labels/proofread-cell-hemibrain-v1.2"
HEMIBRAIN_EB_PROBE_BOXES = {
    "train": [[3052, 3948], [3052, 3948], [3052, 3948]],
    "fit": [[4052, 4948], [4052, 4948], [3052, 3948]],
    "test": [[4052, 4948], [4052, 4948], [4052, 4948]],
}
# mia-evals' annotated boxes around fit and test (artifact provenance: annotated_box)
HEMIBRAIN_EB_PROBE_ANNOTATED = {"fit": [[4000, 5000], [4000, 5000], [3000, 4000]], "test": [[4000, 5000], [4000, 5000], [4000, 5000]]}

# Hemibrain crops in global hemibrain voxels (x y z), 8 nm, from their catalog names and shapes:
#   crop-001 EB   x 24000-29000  y 23000-28000  z 17000-22000   (5000^3)
#   crop-002 11k  x 18000-29000  y 17000-28000  z 11000-22000   (11000^3, CONTAINS crop-001)
#   crop-003 10k  x  8000-18000  y 17000-27000  z 11000-21000   (10000^3, beside crop-002)
# crop-002 is cut at its local z < 9000 (global z < 20000) so EB's val slab and test corner
# (EB z >= 3000) never enter training; EB's train region (EB z < 3000) stays inside.
HEMIBRAIN_002 = "em-drosophila-flyem-hemibrain/crop-002_11k_x18000_y17000_z11000"
HEMIBRAIN_003 = "em-drosophila-flyem-hemibrain/crop-003_10k_x8000_y17000_z11000"

# Training volumes per TrainData choice: catalog name -> [lo, hi) level-0 box in that crop, x y z.
TRAIN_BOXES: dict[TrainData, dict[str, list[list[int]]]] = {
    "hemibrain_eb": {
        HEMIBRAIN_EB: HEMIBRAIN_EB_BOXES["train"]
    },  # 75 Gvox
    "hemibrain_wide": {  # ~2.1 Tvox, EB train included via crop-002
        HEMIBRAIN_002: [[0, 11000], [0, 11000], [0, 9000]],
        HEMIBRAIN_003: [[0, 10000], [0, 10000], [0, 10000]],
    },
    # The other three families of lmd_ssl_v1 (PROBE_SETS), each its whole crops outside that set's eval volumes. All uint8,
    # stored x y z; miao reads them at the 8 nm the dataloader asks for, like mia-evals: fish2 8 x 8 x 7.5 nm, LICONN's
    # 9.7 x 9.7 x 25.9 nm (expansion-corrected) and MICrONS' 8 x 8 x 40 nm resampled to 8 nm isotropic.
    "fish2": {  # zebrafish EM, ~25 Gvox: all fish2 crops but quadcube1 (lmd_ssl_v1 fit) and doublecube1 (test)
        "em-zebrafish-fish2/crop-002_doublecube2_x120000_y45500_z0": [[0, 2000], [0, 2000], [0, 2120]],
        "em-zebrafish-fish2/crop-004_quadcube2_x15114_y10046_z0": [[0, 2000], [0, 2000], [0, 1364]],
        "em-zebrafish-fish2/crop-005_quadcube3_x9638_y10314_z0": [[0, 2000], [0, 2000], [0, 1364]],
        "em-zebrafish-fish2/crop-006_quadcube4_x48294_y6026_z0": [[0, 2000], [0, 2000], [0, 1364]],
    },
    # mouse LICONN ExM, ~666 Gvox: three animals with no eval volume (ExPID82-1's full volume is left out: its crop-002 is
    # lmd_ssl_v1 test data, cut from it)
    "liconn_mouse": {
        "exm-mouse-liconn-ExPID124/crop-001": [[0, 39000], [0, 10000], [0, 752]],
        "exm-mouse-liconn-ExPID146/crop-001": [[0, 10762], [0, 10680], [0, 1695]],
        "exm-mouse-liconn-ExPID150/crop-001": [[0, 4900], [0, 4700], [0, 7720]],
    },
    # mouse cortex EM for the Kasthuri15 family, whose 0.3 Gvox are all eval volumes: MICrONS minnie65, 69 Gvox, the
    # closest in tissue (mouse cortex) and anisotropy (8 x 8 x 40 nm against 6 x 6 x 29)
    "microns": {
        "em-mouse-MICrONS-minnie65/crop-001": [[0, 8192], [0, 8192], [0, 1024]],
    },
    # CellMap's FIB-SEM organelle volumes (HHMI Janelia: cells, tissues, 4-8 nm), ~10 Tvox at 8 nm. Each volume's largest slab (x y z,
    # whole in the other two axes) with no manual organelle crop in it (labels/manual_gt-*), so the regions that CellMap's organelle
    # labels cover (CELLMAP_VAL) stay out of training; the labelled-crop boxes are those in the catalog's labels, not CellMap's
    # unreleased test crops. Left out: volumes with under 0.5 Gvox free at 8 nm (nacc-1, cos7). Native voxel sizes 4-8 nm (3.2-5.2 along z
    # for some cell lines), read at 8 nm (miao resamples), uint8. The Gvox at 8 nm in the comments.
    "cellmap": {
        "em-drosophila-CellMap-jrc-fly-mb-1a/crop-001_fullvol_tissuecrop": [[1087, 3717], [0, 3430], [0, 7723]],  # 8.7 Gvox at 8 nm
        "em-drosophila-CellMap-jrc-fly-vnc-1/crop-001_fullvol_tissuecrop": [[1363, 3488], [0, 2067], [0, 1471]],  # 0.8 Gvox at 8 nm
        "em-human-CellMap-jrc-hela-2/crop-001_fullvol_tissuecrop": [[8580, 10333], [0, 1312], [0, 5742]],  # 2.2 Gvox at 8 nm
        "em-human-CellMap-jrc-hela-3/crop-001_fullvol_tissuecrop": [[0, 10596], [0, 931], [5039, 8953]],  # 3.9 Gvox at 8 nm
        "em-human-CellMap-jrc-jurkat-1/crop-001_fullvol_tissuecrop": [[3312, 5014], [0, 2840], [0, 4212]],  # 2.2 Gvox at 8 nm
        "em-human-CellMap-jrc-macrophage-2/crop-001_fullvol_tissuecrop": [[0, 9526], [0, 1590], [5147, 6562]],  # 2.3 Gvox at 8 nm
        "em-human-CellMap-jrc-sum159-1/crop-001_fullvol_tissuecrop": [[8334, 13515], [0, 2469], [0, 4067]],  # 7.4 Gvox at 8 nm
        "em-human-CellMap-jrc-sum159-4/crop-001_fullvol_tissuecrop": [[2986, 6865], [0, 847], [0, 3772]],  # 12.4 Gvox at 8 nm
        "em-human-CellMap-jrc-ut21-1413-003/crop-001_fullvol_tissuecrop": [[0, 9996], [307, 3007], [0, 16301]],  # 440.0 Gvox at 8 nm
        "em-mouse-CellMap-jrc-ctl-id8-1/crop-001_fullvol_tissuecrop": [[693, 3593], [0, 1895], [0, 6428]],  # 3.8 Gvox at 8 nm
        "em-mouse-CellMap-jrc-mus-heart-1/crop-001_fullvol_tissuecrop": [[0, 11226], [0, 3528], [0, 6591]],  # 261.0 Gvox at 8 nm
        "em-mouse-CellMap-jrc-mus-kidney-3/crop-001_fullvol_tissuecrop": [[0, 6105], [0, 5202], [3674, 7530]],  # 122.5 Gvox at 8 nm
        "em-mouse-CellMap-jrc-mus-kidney-glomerulus-2/crop-001_fullvol_tissuecrop": [[2734, 3389], [0, 3346], [0, 2421]],  # 0.7 Gvox at 8 nm
        "em-mouse-CellMap-jrc-mus-kidney/crop-001_fullvol_tissuecrop": [[0, 9867], [0, 7190], [616, 4252]],  # 258.0 Gvox at 8 nm
        "em-mouse-CellMap-jrc-mus-liver-3/crop-001_fullvol_tissuecrop": [[5128, 9747], [0, 7585], [0, 8778]],  # 307.5 Gvox at 8 nm
        "em-mouse-CellMap-jrc-mus-liver-zon-1/crop-001_fullvol_tissuecrop": [[0, 20208], [0, 15828], [30060, 39660]],  # 3070.6 Gvox at 8 nm
        "em-mouse-CellMap-jrc-mus-liver-zon-2/crop-001_fullvol_tissuecrop": [[0, 21800], [0, 18280], [6584, 17740]],  # 4445.7 Gvox at 8 nm
        "em-mouse-CellMap-jrc-mus-liver/crop-001_fullvol_tissuecrop": [[0, 10037], [7855, 10174], [0, 8810]],  # 205.1 Gvox at 8 nm
        "em-zebrafish-CellMap-jrc-zf-cardiac-1/crop-001_fullvol_tissuecrop": [[0, 7000], [5200, 10400], [0, 23720]],  # 863.4 Gvox at 8 nm
    },
}
# CellMap's labelled crops, by volume: the first (up to) 512 voxels at 8 nm a side of its largest manual organelle crop (x y z, level-0 voxels), for the volumes where one holds a 128 window, validation images
# for the pretraining loss (lib.evals.val_loss; CellMap's labels aren't affinities, so there is no probe set). Outside TRAIN_BOXES["cellmap"].
CELLMAP_VAL: dict[str, list[list[int]]] = {
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
}
WITHIN_EXPONENT = {"cellmap": 0.4}  # a family's volumes sampled by voxel count ^ this (default 0: equally): CellMap's go from 0.7 to 4,400 Gvox
# A joint set: its families' volumes, and the exponent e of its family weights: a family's share of the samples goes as its
# voxel count ^ e (0: equal, as lmd_ssl_v1 weights its volumes, and as e09 and e11's d3 ran; 1: by size, which hemibrain_wide would
# take 3/4 of); each family's volumes equally within it.
LMD4 = ("hemibrain_wide", "fish2", "liconn_mouse", "microns")
JOINT_DATA: dict[TrainData, tuple[tuple[TrainData, ...], float]] = {"lmd4": (LMD4, 0.0), "lmd4_pow04": (LMD4, 0.4), "lmd5_pow04": ((*LMD4, "cellmap"), 0.4)}


def family_voxels(family: TrainData) -> int:
    return sum(math.prod(hi - lo for lo, hi in box) for box in TRAIN_BOXES[family].values())


def train_volumes(data: TrainData) -> list[tuple[str, list[list[int]], float]]:
    """(catalog name, [lo, hi) box x y z, sampling weight) of each training volume of data; the weights sum to 1: a family's share
    (its voxel count ^ the joint set's exponent), split among its volumes (equally, or by WITHIN_EXPONENT)."""
    families, exponent = JOINT_DATA.get(data, ((data,), 0.0))
    shares = {f: family_voxels(f) ** exponent for f in families}
    within = {name: math.prod(hi - lo for lo, hi in box) ** WITHIN_EXPONENT.get(f, 0.0) for f in families for name, box in TRAIN_BOXES[f].items()}
    return [(name, box, shares[f] / sum(shares.values()) * within[name] / sum(within[n] for n in TRAIN_BOXES[f])) for f in families for name, box in TRAIN_BOXES[f].items()]


class ProbeVolume(NamedTuple):
    """One labeled block the affinity probe fits on or is scored on (experiment.probe)."""
    name: str
    path: str  # OME-zarr store with raw/ and labels/<label_key>
    label_key: str
    box: list[list[int]]  # [lo, hi) level-0 voxels, x y z
    clip: tuple[float, float] | None = None  # uint16 images: the [lo, hi] mapped to [0, 1] (miao's normalize_min / _max)
    sparse: bool = False  # its labels are whole cells but not every cell: labelled against unlabelled is a boundary (lib.probe.affinities)
    annotated: list[list[int]] | None = None  # mia-evals' annotated box around a fit or test block (its artifact's provenance), x y z


def _lmd_ssl_v1(name: str, catalog: str, label_key: str, box: list[list[int]], clip: tuple[float, float] | None = None) -> ProbeVolume:
    # lmd_ssl_v1's labels (proofread, or manual "final") label some cells whole: e09's probes there scored labelled pairs only,
    # where +1 voxel boundaries were 0.1-1% of pairs and every encoder's short AP sat at that base rate.
    return ProbeVolume(name, f"{LMD_DATA}/{catalog}.zarr", label_key, box, clip, sparse=True)


class ProbeSet(NamedTuple):
    """A labelled dataset's blocks, by the role each plays: what lib.evals.probe and finetune fit and score, and what `score` tunes on."""
    fit: ProbeVolume  # the readout (the linear probe, the decoder) is fit on it
    test: list[ProbeVolume]  # the fitted readout is scored on these, averaged
    tune: ProbeVolume | None = None  # `score` only: where mia-evals chooses its post-processing's parameters (its "fit" block), never test


def _hemibrain_eb(split: str) -> ProbeVolume:
    return ProbeVolume(f"hemibrain_eb_{split}", f"{LMD_DATA}/{HEMIBRAIN_EB}.zarr", HEMIBRAIN_EB_LABELS.removeprefix("labels/"),
                       HEMIBRAIN_EB_PROBE_BOXES[split], annotated=HEMIBRAIN_EB_PROBE_ANNOTATED.get(split))


# Probe sets (Eval.probe_on). hemibrain_eb is the probe of every run so far; its volumes are named for the splits of its boxes
# (HEMIBRAIN_EB_PROBE_BOXES): hemibrain_eb_train is the fit block, hemibrain_eb_fit the tune block (mia-evals' name for it). The others
# are mia-evals' lmd_ssl_v1_neuron_instance (configs/lmd_ssl_v1_neuron_instance/data, copied 2026-10-18): its fit half is the set's
# fit block, its test half the test blocks, boxes and clip ranges as there; they have no tune block, so they can't be scored.
PROBE_SETS: dict[str, ProbeSet] = {
    "hemibrain_eb": ProbeSet(fit=_hemibrain_eb("train"), test=[_hemibrain_eb("test")], tune=_hemibrain_eb("fit")),
    "kasthuri15": ProbeSet(
        fit=_lmd_ssl_v1("kasthuri15_ac3", "em-mouse-Kasthuri15-ac3ac4-cortex/crop-001_ac3_100slices", "public_gt-cell-kasthuri15", [[0, 1024], [0, 1024], [0, 100]]),
        test=[_lmd_ssl_v1("kasthuri15_ac4", "em-mouse-Kasthuri15-ac3ac4-cortex/crop-002_ac4_100slices", "public_gt-cell-kasthuri15", [[0, 1024], [0, 1024], [0, 100]])]),
    "fish2": ProbeSet(
        fit=_lmd_ssl_v1("zebrafish_fish2_quadcube1", "em-zebrafish-fish2/crop-003_quadcube1_x23908_y18314_z0", "proofread-cell-seg230927", [[0, 2000], [0, 2000], [0, 1364]]),
        test=[_lmd_ssl_v1("zebrafish_fish2_doublecube1", "em-zebrafish-fish2/crop-001_doublecube1_x52000_y42500_z0", "proofread-cell-seg230927", [[0, 2000], [0, 2000], [0, 2112]])]),
    "liconn_mouse": ProbeSet(
        fit=_lmd_ssl_v1("liconn_mouse_dg", "exm-mouse-liconn-DG-20250809_ExPID19-02_2ndGel_C5_Atto488_40XW_002/crop-001", "manual_gt-cell-final",
                        [[1416, 1892], [516, 996], [83, 276]], (301, 1130)),
        test=[_lmd_ssl_v1("liconn_mouse_hippocampus", "exm-mouse-liconn-hippocampus-202507013_ExPID10-02_2percentAA_S2_Atto488_40XW_004/crop-001", "manual_gt-cell-final",
                          [[496, 976], [496, 976], [44, 236]], (219, 919)),
              _lmd_ssl_v1("liconn_expid82", "exm-mouse-liconn-ExPID82-1/crop-002_sub", "proofread-cell-agglom-dense-v20240123", [[1229, 2253], [1642, 2666], [0, 534]])]),
}


def probe_set(name: str) -> ProbeSet:
    """The labelled blocks of probe set name, by role."""
    assert name in PROBE_SETS, f"no probe set {name!r}: {list(PROBE_SETS)}"
    return PROBE_SETS[name]


class HeldOut(NamedTuple):
    """A region of one crop that no training sees, and where PCA maps look in it."""
    path: str  # catalog name of the crop
    region: list[list[int]]  # [lo, hi) level-0 voxels, z y x
    origin: tuple[int, int, int]  # z y x: a window's low corner


HELD_OUT: dict[str, HeldOut] = {
    "hemibrain_eb": HeldOut(HEMIBRAIN_EB, [[3000, 4000], [0, 5000], [0, 5000]], (3400, 2000, 2000)),  # its val slab, EB z 3000-4000
}


# A training family's validation images: the blocks of a probe set (their images: no label is read), which pretraining never
# samples (TRAIN_BOXES leaves out every eval volume). MICrONS has none of its own: Kasthuri15 is mouse cortex EM, as it is.
VAL_SETS = {"hemibrain_eb": "hemibrain_eb", "hemibrain_wide": "hemibrain_eb", "fish2": "fish2", "liconn_mouse": "liconn_mouse", "microns": "kasthuri15"}


def val_blocks(family: TrainData) -> list[ProbeVolume]:
    """The held-out blocks of a training family (not a joint set) whose validation loss is measured (VAL_SETS)."""
    blocks = probe_set(VAL_SETS[family])
    return [blocks.fit, *blocks.test]


def held_out(name: str) -> HeldOut:
    assert name in HELD_OUT, f"no held-out region for {name!r} (add it to lib.data.HELD_OUT): {list(HELD_OUT)}"
    return HELD_OUT[name]
