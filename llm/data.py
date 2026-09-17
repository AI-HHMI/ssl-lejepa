"""Data utilities and test volume generators for local development."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import zarr
from zarr.storage import LocalStore
from miao.config import VolumeConfig


def create_mock_ome_zarr(
    root_path: Path | str,
    shape: Tuple[int, int, int] = (128, 256, 256),
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
    shape: Tuple[int, int, int] = (128, 256, 256),
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
