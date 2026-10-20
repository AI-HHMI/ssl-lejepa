"""Training/eval volume definitions (lmd catalog + miao): hemibrain, and the four families of mia-evals' lmd_ssl_v1."""

from __future__ import annotations

from typing import NamedTuple

import lmd_catalog as lmd
from miao.config import MiaoConfig

# from lib.types import TrainData
from typing import Literal
TrainData = Literal["hemibrain_eb", "hemibrain_wide", "fish2", "liconn_mouse", "microns", "lmd4"]  # keys of lib.data.TRAIN_BOXES, or JOINT_DATA

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
}
# A joint set: its families' volumes, the families weighted equally (as lmd_ssl_v1 weights its volumes), each family's
# volumes equally within it.
JOINT_DATA: dict[TrainData, tuple[TrainData, ...]] = {"lmd4": ("hemibrain_wide", "fish2", "liconn_mouse", "microns")}


def train_volumes(data: TrainData) -> list[tuple[str, list[list[int]], float]]:
    """(catalog name, [lo, hi) box x y z, sampling weight) of each training volume of data."""
    families = JOINT_DATA.get(data, (data,))
    return [(name, box, 1 / len(families) / len(TRAIN_BOXES[f])) for f in families for name, box in TRAIN_BOXES[f].items()]


class ProbeVolume(NamedTuple):
    """One labeled block the affinity probe fits on or is scored on (experiment.probe)."""
    name: str
    path: str  # OME-zarr store with raw/ and labels/<label_key>
    label_key: str
    box: list[list[int]]  # [lo, hi) level-0 voxels, x y z
    clip: tuple[float, float] | None = None  # uint16 images: the [lo, hi] mapped to [0, 1] (miao's normalize_min / _max)


def _lmd_ssl_v1(name: str, catalog: str, label_key: str, box: list[list[int]], clip: tuple[float, float] | None = None) -> ProbeVolume:
    return ProbeVolume(name, f"{LMD_DATA}/{catalog}.zarr", label_key, box, clip)


# Probe sets (Eval.probe_on): a "train" block the probe fits on, and the "test" blocks it is scored on. hemibrain_eb is the
# probe of every run so far. The others are mia-evals' lmd_ssl_v1_neuron_instance (configs/lmd_ssl_v1_neuron_instance/data,
# copied 2026-10-18): its fit half the probe's train, its test half the test, boxes and clip ranges as there.
PROBE_SETS: dict[str, dict[str, list[ProbeVolume]]] = {
    "hemibrain_eb": {split: [ProbeVolume(f"hemibrain_eb_{split}", f"{LMD_DATA}/{HEMIBRAIN_EB}.zarr", HEMIBRAIN_EB_LABELS.removeprefix("labels/"),
                                         HEMIBRAIN_EB_PROBE_BOXES[split])] for split in ("train", "test")},
    "kasthuri15": {
        "train": [_lmd_ssl_v1("kasthuri15_ac3", "em-mouse-Kasthuri15-ac3ac4-cortex/crop-001_ac3_100slices", "public_gt-cell-kasthuri15", [[0, 1024], [0, 1024], [0, 100]])],
        "test": [_lmd_ssl_v1("kasthuri15_ac4", "em-mouse-Kasthuri15-ac3ac4-cortex/crop-002_ac4_100slices", "public_gt-cell-kasthuri15", [[0, 1024], [0, 1024], [0, 100]])],
    },
    "fish2": {
        "train": [_lmd_ssl_v1("zebrafish_fish2_quadcube1", "em-zebrafish-fish2/crop-003_quadcube1_x23908_y18314_z0", "proofread-cell-seg230927", [[0, 2000], [0, 2000], [0, 1364]])],
        "test": [_lmd_ssl_v1("zebrafish_fish2_doublecube1", "em-zebrafish-fish2/crop-001_doublecube1_x52000_y42500_z0", "proofread-cell-seg230927", [[0, 2000], [0, 2000], [0, 2112]])],
    },
    "liconn_mouse": {
        "train": [_lmd_ssl_v1("liconn_mouse_dg", "exm-mouse-liconn-DG-20250809_ExPID19-02_2ndGel_C5_Atto488_40XW_002/crop-001", "manual_gt-cell-final",
                              [[1416, 1892], [516, 996], [83, 276]], (301, 1130))],
        "test": [_lmd_ssl_v1("liconn_mouse_hippocampus", "exm-mouse-liconn-hippocampus-202507013_ExPID10-02_2percentAA_S2_Atto488_40XW_004/crop-001", "manual_gt-cell-final",
                             [[496, 976], [496, 976], [44, 236]], (219, 919)),
                 _lmd_ssl_v1("liconn_expid82", "exm-mouse-liconn-ExPID82-1/crop-002_sub", "proofread-cell-agglom-dense-v20240123", [[1229, 2253], [1642, 2666], [0, 534]])],
    },
}

def hemibrain_eb_config(split: str) -> MiaoConfig:
    """MiaoConfig equivalent to gary_comparison's hemibrain_eb_{split}.yaml (minus defer_image_ops)."""
    assert split in HEMIBRAIN_EB_BOXES, f"split must be one of {list(HEMIBRAIN_EB_BOXES)}, got {split!r}"
    vol = lmd.get(HEMIBRAIN_EB).to_miao(
        spatial_axes="xyz",
        label_key="labels/proofread-cell-hemibrain-v1.2",
        bounding_box=HEMIBRAIN_EB_BOXES[split],
    )
    return MiaoConfig(
        volumes=[vol],
        resolutions=[[8.0, 8.0, 8.0]],
        patch_size=[256, 256, 256],
        output_axes="lcxyz",
        samples_per_epoch=100000 if split == "train" else 32,
    )

def hemibrain_wide_config(split: str) -> MiaoConfig:
    """MiaoConfig over TRAIN_BOXES["hemibrain_wide"]'s volumes (crop-002 + crop-003, ~2.1 Tvox combined).

    Only "train" is defined: no held-out region has been carved out of this corpus, so training loss here reflects
    repeated passes over a fixed corpus at the smaller end of a compute budget, not a genuine held-out metric."""
    assert split == "train", f"split must be 'train' (no val/test defined for hemibrain_wide yet), got {split!r}"
    volumes = [lmd.get(name).to_miao(spatial_axes="xyz", bounding_box=box) for name, box in TRAIN_BOXES["hemibrain_wide"].items()]
    return MiaoConfig(
        volumes=volumes,
        resolutions=[[8.0, 8.0, 8.0]],
        patch_size=[256, 256, 256],
        output_axes="lcxyz",
        samples_per_epoch=100000,
    )
