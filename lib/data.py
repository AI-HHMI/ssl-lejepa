"""Data utilities and test volume generators for local development."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import zarr
from zarr.storage import LocalStore
from miao.config import MiaoConfig, VolumeConfig
from lib.types import TrainData, Tup3Int
import lmd_catalog as lmd

def create_mock_ome_zarr(
    root_path: Path | str,
    shape: Tup3Int = (128, 256, 256),
    resolutions: Tuple[float, float, float] = (25.0, 10.0, 10.0),
    group_key: str = "raw",
    dtype: str = "uint16",
) -> Path:
    """Create a minimal OME-NGFF Zarr (v2) volume on local disk for testing.

    Args:
        root_path: Directory path for the Zarr store.
        shape: (Z, Y, X) voxel shape.
        resolutions: (Z, Y, X) voxel sizes in nanometers.
        group_key: Subgroup key for raw imagery (default 'raw').
        dtype: Numerical data type.

    Returns:
        Path to created zarr directory.
    """
    root_path = Path(root_path)
    root_path.mkdir(parents=True, exist_ok=True)

    store = LocalStore(str(root_path))
    root = zarr.open_group(store, mode="a", zarr_format=2)

    parts = group_key.split("/")
    grp = root
    for part in parts:
        grp = grp.create_group(part, overwrite=False) if part not in grp else grp[part]

    # Level 0 array
    rng = np.random.default_rng(42)
    data = (rng.random(shape) * 65535).astype(dtype)

    arr = grp.create_array(
        "s0",
        shape=shape,
        chunks=(32, 64, 64),
        dtype=dtype,
        overwrite=True,
    )
    arr[:] = data

    # OME-Zarr multiscales metadata
    axes = [
        {"name": "z", "type": "space", "unit": "nanometer"},
        {"name": "y", "type": "space", "unit": "nanometer"},
        {"name": "x", "type": "space", "unit": "nanometer"},
    ]
    multiscales = [{
        "version": "0.4",
        "name": "image",
        "axes": axes,
        "datasets": [{
            "path": "s0",
            "coordinateTransformations": [{
                "type": "scale",
                "scale": list(resolutions),
            }],
        }],
    }]
    grp.attrs["multiscales"] = multiscales
    return root_path


def get_mock_volume_config(
    mock_dir: Path | str = "/tmp/mock_flyliconn.zarr",
    shape: Tup3Int = (128, 256, 256),
    resolutions: Tuple[float, float, float] = (25.0, 10.0, 10.0),
) -> VolumeConfig:
    """Return a VolumeConfig pointing to a local mock volume, creating it if needed."""
    path = Path(mock_dir)
    if not path.exists():
        create_mock_ome_zarr(path, shape=shape, resolutions=resolutions)

    return VolumeConfig(
        name="mock_flyliconn",
        path=str(path),
        image_key="raw",
        zarr_version="zarr2",
    )

# FlyEM hemibrain Ellipsoid Body splits from orhane's gary_comparison runs.
HEMIBRAIN_EB = "em-drosophila-flyem-hemibrain/crop-001_EllipsoidBody_x24000_y23000_z17000"
HEMIBRAIN_EB_BOXES = {  # [lo, hi) level-0 voxels, x y z
    "train": [[0, 5000], [0, 5000], [0, 3000]],
    "val": [[0, 5000], [0, 5000], [3000, 4000]],
    "test": [[4000, 5000], [4000, 5000], [4000, 5000]],
}
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
    "hemibrain_eb": {HEMIBRAIN_EB: HEMIBRAIN_EB_BOXES["train"]},  # 75 Gvox
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
