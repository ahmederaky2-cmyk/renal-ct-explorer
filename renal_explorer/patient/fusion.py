"""Combine per-structure masks from several models into one canonical labelmap."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..labels import CanonicalLabel

# Later entries overwrite earlier ones.
FUSION_ORDER = (
    CanonicalLabel.LIVER,
    CanonicalLabel.SPLEEN,
    CanonicalLabel.PANCREAS,
    CanonicalLabel.COLON,
    CanonicalLabel.DUODENUM,
    CanonicalLabel.PSOAS,
    CanonicalLabel.ADRENAL,
    CanonicalLabel.AORTA,
    CanonicalLabel.IVC,
    CanonicalLabel.KIDNEY,
    CanonicalLabel.RENAL_VEIN,
    CanonicalLabel.RENAL_ARTERY,
    CanonicalLabel.CYST,
    CanonicalLabel.COLLECTING_SYSTEM,
    CanonicalLabel.TUMOR,
)
RENAL_ONLY = (CanonicalLabel.TUMOR, CanonicalLabel.CYST, CanonicalLabel.COLLECTING_SYSTEM)


def restrict_to_kidney(
    mask: np.ndarray, kidney: np.ndarray, spacing, max_distance_mm: float
) -> np.ndarray:
    """Keep connected components that touch or lie within ``max_distance_mm`` of the kidney."""

    if not mask.any() or not kidney.any():
        return np.zeros_like(mask, dtype=bool)
    near = ndi.distance_transform_edt(~kidney, sampling=spacing) <= max_distance_mm
    components, count = ndi.label(mask, structure=ndi.generate_binary_structure(3, 2))
    keep = np.unique(components[near & mask])
    keep = keep[keep > 0]
    return np.isin(components, keep)


def fuse(masks: dict[CanonicalLabel, np.ndarray], spacing, renal_distance_mm: float = 10.0):
    """Return (uint8 labelmap, per-label voxel counts, list of notes)."""

    shape = next(iter(masks.values())).shape
    labels = np.zeros(shape, dtype=np.uint8)
    notes: list[str] = []
    kidney = masks.get(CanonicalLabel.KIDNEY)
    kidney_or_tumor = kidney.copy() if kidney is not None else np.zeros(shape, bool)
    if CanonicalLabel.TUMOR in masks and kidney is not None:
        kidney_or_tumor |= masks[CanonicalLabel.TUMOR]

    for label in FUSION_ORDER:
        mask = masks.get(label)
        if mask is None or not mask.any():
            continue
        mask = mask.astype(bool, copy=False)
        if label in RENAL_ONLY:
            if kidney is None or not kidney.any():
                notes.append(f"{label.name.lower()} dropped: no kidney segmentation.")
                continue
            reference = kidney if label == CanonicalLabel.TUMOR else kidney_or_tumor
            restricted = restrict_to_kidney(mask, reference, spacing, renal_distance_mm)
            dropped = int(mask.sum() - restricted.sum())
            if dropped:
                notes.append(
                    f"{label.name.lower()}: {dropped} voxels away from the kidneys were removed."
                )
            mask = restricted
        labels[mask] = int(label)

    counts = {
        CanonicalLabel(int(value)).name: int(count)
        for value, count in zip(*np.unique(labels, return_counts=True))  # noqa: B905
        if value != 0
    }
    return labels, counts, notes
