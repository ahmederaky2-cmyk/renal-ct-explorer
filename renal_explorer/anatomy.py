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


def kidney_sides(
    kidney_centroids_x: Sequence[float],
    kidney_volumes_ml: Sequence[float],
    midline_x: float | None,
) -> list[str]:
    """Sides for kidney components: by the two largest kidneys, else the great vessels.

    Never by the image centre, which moves whenever a scan is cropped.
    """

    count = len(kidney_centroids_x)
    if count == 0:
        return []
    order = np.argsort(np.asarray(kidney_volumes_ml))[::-1]
    if count >= 2:
        first, second = kidney_centroids_x[order[0]], kidney_centroids_x[order[1]]
        if abs(first - second) >= 30.0:  # two separate kidneys, not fragments of one
            reference = 0.5 * (first + second)
            return [side_of(x, reference, tolerance_mm=0.0) for x in kidney_centroids_x]
    if midline_x is not None:
        return [side_of(x, midline_x) for x in kidney_centroids_x]
    return [UNKNOWN] * count
