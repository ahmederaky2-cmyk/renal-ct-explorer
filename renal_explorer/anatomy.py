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


MIN_KIDNEY_ML = 40.0  # smaller pieces are not trusted as a separate kidney
MIN_KIDNEY_GAP_MM = 30.0  # left and right kidneys are separated by the spine and psoas


def kidney_reference_x(
    components: Sequence[dict[str, float]], midline_x: float | None
) -> tuple[float | None, str]:
    """Reference x separating right from left, and how it was found.

    ``components`` are kidney components with keys ``x`` (centroid), ``x_min``, ``x_max``
    and ``volume`` (mL). Priority:

    1. the aorta/IVC midline;
    2. exactly two kidney-sized components (each ≥ ``MIN_KIDNEY_ML``) whose x ranges are at
       least ``MIN_KIDNEY_GAP_MM`` apart. A piece of the same kidney, even a large one,
       does not sit a spine's width away from it;
    3. anything else (one kidney, fragments, three pieces): Unknown. A wrong side is
       worse than no side, and split volumes are then withheld.
    """

    if midline_x is not None:
        return midline_x, "great vessels"
    kidneys = [item for item in components if item["volume"] >= MIN_KIDNEY_ML]
    if len(kidneys) != 2:
        return None, "unknown"
    first, second = sorted(kidneys, key=lambda item: item["x"])
    if second["x_min"] - first["x_max"] < MIN_KIDNEY_GAP_MM:
        return None, "unknown"
    return 0.5 * (first["x"] + second["x"]), "both kidneys"


def kidney_sides(
    components: Sequence[dict[str, float]], midline_x: float | None
) -> list[str]:
    reference, source = kidney_reference_x(components, midline_x)
    if reference is None:
        return [UNKNOWN] * len(components)
    tolerance = 5.0 if source == "great vessels" else 0.0
    return [side_of(item["x"], reference, tolerance) for item in components]
