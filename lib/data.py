"""Training/eval volume definitions (lmd catalog + miao) for hemibrain."""

from __future__ import annotations

import lmd_catalog as lmd
from miao.config import MiaoConfig

from lib.types import TrainData

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
