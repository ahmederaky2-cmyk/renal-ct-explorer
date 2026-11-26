"""Reduced-resolution loading for the free public demo.

The public Streamlit server has little memory, so the *Online demo* reads every scan at
about 1.5 mm. Each file is read with strided slices of its NIfTI proxy before reorientation.
nibabel may still read much of the compressed stream, but the full-resolution float volume
is never built.

The sampling grid is defined in canonical (RAS) voxel space: canonical index 0, s, 2s, … on
every axis. Each file is sliced on its own native axes so that it lands on exactly that grid
(flipped axes start at ``(n - 1) % s``). The CT, its labels and any full-resolution
prediction therefore always keep the same voxels, and reducing a canonical array is plain
slicing.

Patient mode and server datasets always load at full resolution: measurements used for
planning must never come from a reduced grid.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

DEFAULT_DEMO_SPACING_MM = 1.5
ONLINE_DEMO = "Online demo"


def demo_target_spacing(access_mode: str, patient_mode: bool) -> float | None:
    """Target spacing for this session, or None for full resolution.

    Full resolution whenever patient mode is on, whatever data source is selected.
    ``RENAL_DEMO_SPACING_MM=0`` turns reduction off for the demo too.
    """

    if patient_mode or access_mode != ONLINE_DEMO:
        return None
    try:
        target = float(os.environ.get("RENAL_DEMO_SPACING_MM", DEFAULT_DEMO_SPACING_MM))
    except ValueError:
        target = DEFAULT_DEMO_SPACING_MM
    return target if target > 0 else None


def decimation_strides(zooms, target_mm: float | None) -> tuple[int, int, int]:
    """Integer step per axis so the new spacing is close to (and rarely far above) target.

    0.78 mm → ×2 (1.56 mm), 0.5 mm → ×3 (1.5 mm), 1.0 mm → ×1, 5 mm → ×1.
    """

    if not target_mm:
        return (1, 1, 1)
    return tuple(  # type: ignore[return-value]
        max(1, int(math.floor(target_mm / float(zoom) + 0.35))) for zoom in list(zooms)[:3]
    )


def full_canonical_geometry(image) -> dict[str, Any]:
    """Shape/affine/spacing that ``nib.as_closest_canonical`` gives, without loading data."""

    from nibabel import orientations

    orientation = orientations.io_orientation(image.affine)
    shape = tuple(int(value) for value in image.shape[:3])
    canonical_shape = [0, 0, 0]
    for source_axis, (target_axis, _flip) in enumerate(orientation):
        canonical_shape[int(target_axis)] = shape[source_axis]
    affine = np.asarray(image.affine, dtype=float) @ orientations.inv_ornt_aff(orientation, shape)
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return {
        "shape": tuple(canonical_shape),
        "affine": affine,
        "spacing": tuple(float(value) for value in spacing),
        "orientation": orientation,
    }


def read_reduced(image, canonical_strides: tuple[int, int, int]):
    """Strided copy of a nibabel image that lands on the canonical grid 0, s, 2s, …"""

    import nibabel as nib
    from nibabel import orientations

    if tuple(canonical_strides) == (1, 1, 1):
        return image
    orientation = orientations.io_orientation(image.affine)
    shape = image.shape[:3]
    slices = []
    starts = []
    steps = []
    for axis in range(3):
        canonical_axis = int(orientation[axis, 0])
        step = int(canonical_strides[canonical_axis])
        flipped = orientation[axis, 1] < 0
        start = (shape[axis] - 1) % step if flipped else 0
        slices.append(slice(start, None, step))
        starts.append(start)
        steps.append(step)
    data = np.asanyarray(image.dataobj[tuple(slices)])
    shift = np.eye(4)
    shift[:3, 3] = starts
    affine = np.asarray(image.affine, dtype=float) @ shift @ np.diag([*steps, 1.0])
    return nib.Nifti1Image(data, affine)


def reduce_canonical_like(array: np.ndarray, resolution: dict[str, Any]) -> np.ndarray:
    """Apply the demo's sampling to a full-resolution canonical array (plain slicing)."""

    strides = tuple(int(value) for value in resolution.get("strides", (1, 1, 1)))
    if strides == (1, 1, 1):
        return array
    return np.ascontiguousarray(array[tuple(slice(None, None, step) for step in strides)])


def grid_token(shape, affine, resolution: dict[str, Any] | None = None) -> str:
    """Short identity of a voxel grid, for cache keys (same file, different grid ≠ same)."""

    import hashlib

    digest = hashlib.sha1()
    digest.update(str(tuple(int(value) for value in shape)).encode())
    digest.update(np.round(np.asarray(affine, dtype=float), 4).tobytes())
    strides = tuple(int(value) for value in (resolution or {}).get("strides", (1, 1, 1)))
    digest.update(str(strides).encode())
    return digest.hexdigest()[:12]
