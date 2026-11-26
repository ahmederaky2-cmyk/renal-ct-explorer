"""Reduced-resolution loading for the free public demo.

The public Streamlit server has little memory, so the *Online demo* reads every scan at
about 1.5 mm. Decimation happens while reading (strided slices of the NIfTI proxy, before
reorientation), so the full-resolution volume is never held in memory.

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


def read_reduced(image, strides: tuple[int, int, int]):
    """A strided copy of a nibabel image, read lazily from its data proxy."""

    import nibabel as nib

    if strides == (1, 1, 1):
        return image
    step = tuple(slice(None, None, value) for value in strides)
    data = np.asanyarray(image.dataobj[step])
    affine = np.asarray(image.affine, dtype=float) @ np.diag([*strides, 1.0])
    return nib.Nifti1Image(data, affine, header=None)


def full_canonical_geometry(image) -> dict[str, Any]:
    """Shape/affine that ``nib.as_closest_canonical`` would give, without loading data."""

    from nibabel import orientations

    orientation = orientations.io_orientation(image.affine)
    shape = tuple(int(value) for value in image.shape[:3])
    canonical_shape = [0, 0, 0]
    for source_axis, (target_axis, _flip) in enumerate(orientation):
        canonical_shape[int(target_axis)] = shape[source_axis]
    affine = np.asarray(image.affine, dtype=float) @ orientations.inv_ornt_aff(orientation, shape)
    return {
        "shape": tuple(canonical_shape),
        "affine": affine,
        "orientation": orientation.tolist(),
    }


def reduce_canonical_like(array: np.ndarray, resolution: dict[str, Any]) -> np.ndarray:
    """Apply the demo's decimation to a full-resolution canonical array.

    Used for imported predictions, which are verified against the full-resolution CT
    geometry and must then land on exactly the voxels that the reduced CT kept.
    """

    from nibabel import orientations

    strides = tuple(resolution["strides"])
    if strides == (1, 1, 1):
        return array
    orientation = np.asarray(resolution["orientation"], dtype=float)
    to_original = orientations.ornt_transform(orientations.axcodes2ornt("RAS"), orientation)
    original = orientations.apply_orientation(array, to_original)
    reduced = original[tuple(slice(None, None, value) for value in strides)]
    return np.ascontiguousarray(orientations.apply_orientation(reduced, orientation))
