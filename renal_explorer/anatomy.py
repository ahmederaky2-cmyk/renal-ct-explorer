"""Anatomical reference helpers (laterality without trusting the image field of view)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage as ndi

from .labels import CanonicalLabel

UNKNOWN = "Unknown"


def world_centroid(mask: np.ndarray, affine: np.ndarray) -> np.ndarray | None:
    if not mask.any():
        return None
    voxel = np.asarray(ndi.center_of_mass(mask), dtype=float)
    return np.asarray(affine, dtype=float)[:3, :3] @ voxel + np.asarray(affine, dtype=float)[:3, 3]


def great_vessel_midline_x(segmentation: np.ndarray, affine: np.ndarray) -> float | None:
    """Patient-midline estimate from the aorta and/or IVC (RAS x, mm)."""

    xs = []
    for label in (CanonicalLabel.AORTA, CanonicalLabel.IVC):
        centroid = world_centroid(segmentation == int(label), affine)
        if centroid is not None:
            xs.append(float(centroid[0]))
    return float(np.mean(xs)) if xs else None


def side_of(x: float, reference_x: float, tolerance_mm: float = 5.0) -> str:
    """Canonical RAS+: x grows towards the patient's right."""

    if x > reference_x + tolerance_mm:
        return "Right"
    if x < reference_x - tolerance_mm:
        return "Left"
    return UNKNOWN


def x_extent(mask: np.ndarray, affine: np.ndarray) -> tuple[float, float] | None:
    """World (RAS) x range of a mask's bounding box, from its eight corners."""

    if not mask.any():
        return None
    bounds = []
    for axis in range(3):
        other = tuple(value for value in range(3) if value != axis)
        indices = np.flatnonzero(np.any(mask, axis=other))
        bounds.append((indices[0] - 0.5, indices[-1] + 0.5))
    corners = np.array(np.meshgrid(*bounds, indexing="ij")).reshape(3, -1).T
    xs = corners @ np.asarray(affine, dtype=float)[0, :3] + float(affine[0, 3])
    return float(xs.min()), float(xs.max())


def kidney_reference_x(
    components: Sequence[dict[str, float]], midline_x: float | None
) -> tuple[float | None, str]:
    """Reference x separating right from left, and how it was found.

    ``components`` are kidney components with keys ``x`` (centroid), ``x_min``, ``x_max``
    and ``volume``. Priority:

    1. the aorta/IVC midline;
    2. two substantial kidney components whose x ranges do not overlap (left and right
       kidneys never overlap left-to-right; pieces of one kidney almost always do);
    3. nothing: laterality is Unknown. A wrong side is worse than no side.
    """

    if midline_x is not None:
        return midline_x, "great vessels"
    if len(components) < 2:
        return None, "unknown"
    ordered = sorted(components, key=lambda item: item["volume"], reverse=True)
    largest = ordered[0]
    for other in ordered[1:]:
        if other["volume"] < 0.25 * largest["volume"]:
            break
        if other["x_min"] > largest["x_max"] or other["x_max"] < largest["x_min"]:
            return 0.5 * (largest["x"] + other["x"]), "both kidneys"
    return None, "unknown"


def kidney_sides(
    components: Sequence[dict[str, float]], midline_x: float | None
) -> list[str]:
    reference, source = kidney_reference_x(components, midline_x)
    if reference is None:
        return [UNKNOWN] * len(components)
    tolerance = 5.0 if source == "great vessels" else 0.0
    return [side_of(item["x"], reference, tolerance) for item in components]
