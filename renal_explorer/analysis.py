from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .labels import LABEL_NAMES, CanonicalLabel
from .models import VolumeCase


def region_statistics(
    image: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, Any]:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {}

    values = image[mask]
    voxel_volume_mm3 = float(np.prod(spacing))
    volume_mm3 = count * voxel_volume_mm3
    coordinates = np.argwhere(mask)
    mins = coordinates.min(axis=0)
    maxs = coordinates.max(axis=0) + 1
    extents_mm = (maxs - mins) * np.asarray(spacing)

    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {
        "voxels": count,
        "volume_ml": float(volume_mm3 / 1000.0),
        "mean_hu": float(np.mean(values)),
        "std_hu": float(np.std(values)),
        "min_hu": float(np.min(values)),
        "max_hu": float(np.max(values)),
        "median_hu": float(median),
        "q25_hu": float(q25),
        "q75_hu": float(q75),
        "iqr_hu": float(q75 - q25),
        "equivalent_diameter_mm": float(np.cbrt(6.0 * volume_mm3 / np.pi)),
        "extent_mm": [float(value) for value in extents_mm],
    }


def voxel_to_world(affine: np.ndarray, voxel: np.ndarray) -> np.ndarray:
    homogeneous = np.append(np.asarray(voxel, dtype=float), 1.0)
    return (affine @ homogeneous)[:3]


def component_measurements(
    case: VolumeCase,
    mask: np.ndarray,
    minimum_volume_ml: float = 0.1,
) -> list[dict[str, Any]]:
    structure = ndi.generate_binary_structure(rank=3, connectivity=2)
    components, component_count = ndi.label(mask, structure=structure)
    measurements: list[dict[str, Any]] = []
    center_x = (case.image.shape[0] - 1) / 2.0

    for component_id in range(1, component_count + 1):
        component_mask = components == component_id
        stats = region_statistics(case.image, component_mask, case.spacing)
        if not stats or stats["volume_ml"] < minimum_volume_ml:
            continue
        centroid_voxel = np.asarray(ndi.center_of_mass(component_mask), dtype=float)
        centroid_world = voxel_to_world(case.affine, centroid_voxel)
        # nib.as_closest_canonical produces RAS+ voxel axes; increasing axis 0 is rightward.
        laterality = "Right" if centroid_voxel[0] >= center_x else "Left"
        measurements.append(
            {
                "component": int(component_id),
                "laterality": laterality,
                "centroid_voxel": [float(value) for value in centroid_voxel],
                "centroid_world_mm": [float(value) for value in centroid_world],
                **stats,
            }
        )

    measurements.sort(key=lambda item: item["volume_ml"], reverse=True)
    return measurements


def analyze_case(case: VolumeCase) -> dict[str, Any]:
    if case.segmentation is None:
        return {"regions": {}, "kidney_components": [], "tumor_components": []}

    segmentation = case.segmentation
    regions: dict[str, dict[str, Any]] = {}
    for label in CanonicalLabel:
        if label == CanonicalLabel.BACKGROUND:
            continue
        stats = region_statistics(case.image, segmentation == int(label), case.spacing)
        if stats:
            regions[LABEL_NAMES[label]] = stats

    kidneys = component_measurements(case, segmentation == int(CanonicalLabel.KIDNEY), 0.1)
    tumors = component_measurements(case, segmentation == int(CanonicalLabel.TUMOR), 0.01)

    for tumor in tumors:
        tumor_world = np.asarray(tumor["centroid_world_mm"])
        if kidneys:
            nearest = min(
                kidneys,
                key=lambda kidney: np.linalg.norm(
                    tumor_world - np.asarray(kidney["centroid_world_mm"])
                ),
            )
            distance = np.linalg.norm(tumor_world - np.asarray(nearest["centroid_world_mm"]))
            tumor["nearest_kidney"] = nearest["laterality"]
            tumor["kidney_centroid_distance_mm"] = float(distance)

    return {
        "regions": regions,
        "kidney_components": kidneys,
        "tumor_components": tumors,
    }


def histogram_table(case: VolumeCase, bins: int = 160) -> dict[str, list[float]]:
    edges = np.linspace(-1000, 1000, bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()
    output: dict[str, list[float]] = {"HU": centers}

    all_counts, _ = np.histogram(case.image, bins=edges)
    output["All voxels"] = np.log1p(all_counts).astype(float).tolist()

    if case.segmentation is not None:
        for label in (
            CanonicalLabel.KIDNEY,
            CanonicalLabel.TUMOR,
            CanonicalLabel.CYST,
        ):
            values = case.image[case.segmentation == int(label)]
            if values.size:
                counts, _ = np.histogram(values, bins=edges)
                output[LABEL_NAMES[label]] = np.log1p(counts).astype(float).tolist()
    return output
