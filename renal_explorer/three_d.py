from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import frangi
from skimage.measure import marching_cubes
from skimage.morphology import remove_small_objects

from .labels import CanonicalLabel

MeshData = dict[str, Any]

VESSEL_SENSITIVITY_QUANTILES = {
    "Strict": 99.7,
    "Balanced": 99.2,
    "Sensitive": 98.7,
}

VESSEL_TARGET_SPACING_MM = {
    "Fast": 3.0,
    "Balanced": 2.0,
    "Detailed": 1.5,
}


def _mesh_from_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    origin_voxels: np.ndarray,
    voxel_scale: np.ndarray,
    step_size: int,
) -> MeshData | None:
    if mask.ndim != 3 or not np.any(mask):
        return None

    padded = np.pad(mask.astype(np.uint8, copy=False), 1)
    vertices, faces, _normals, _values = marching_cubes(
        padded,
        level=0.5,
        step_size=max(1, int(step_size)),
        allow_degenerate=False,
    )
    voxel_vertices = (vertices - 1.0) * voxel_scale + origin_voxels
    affine_array = np.asarray(affine, dtype=float)
    world_vertices = np.empty_like(voxel_vertices, dtype=float)
    for world_axis in range(3):
        world_vertices[:, world_axis] = affine_array[world_axis, 3]
        for voxel_axis in range(3):
            world_vertices[:, world_axis] += (
                voxel_vertices[:, voxel_axis] * affine_array[world_axis, voxel_axis]
            )
    return {
        "vertices": world_vertices.astype(np.float32),
        "faces": faces.astype(np.int32),
        "source_voxels": int(np.count_nonzero(mask)),
    }


def surface_mesh(
    segmentation: np.ndarray,
    affine: np.ndarray,
    label_value: int,
    step_size: int = 2,
) -> MeshData | None:
    """Create a physical-coordinate surface for one segmentation label."""

    mask = segmentation == int(label_value)
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return None

    minimum = np.maximum(coordinates.min(axis=0) - 1, 0)
    maximum = np.minimum(coordinates.max(axis=0) + 2, np.asarray(mask.shape))
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(minimum, maximum))
    cropped = mask[slices]
    return _mesh_from_mask(
        cropped,
        affine,
        origin_voxels=minimum.astype(float),
        voxel_scale=np.ones(3, dtype=float),
        step_size=step_size,
    )


def candidate_vessel_mesh(
    image: np.ndarray,
    segmentation: np.ndarray,
    affine: np.ndarray,
    spacing: tuple[float, float, float],
    sensitivity: str = "Balanced",
    detail: str = "Balanced",
) -> MeshData | None:
    """Estimate bright tubular structures near the kidneys for research visualization.

    This is deliberately returned as an unlabeled candidate-vessel surface. It cannot
    distinguish renal arteries from veins and is not a substitute for vessel annotations.
    """

    renal_labels = (
        int(CanonicalLabel.KIDNEY),
        int(CanonicalLabel.TUMOR),
        int(CanonicalLabel.CYST),
    )
    renal_mask = np.isin(segmentation, renal_labels)
    coordinates = np.argwhere(renal_mask)
    if coordinates.size == 0:
        return None

    spacing_array = np.asarray(spacing, dtype=float)
    padding_voxels = np.ceil(30.0 / spacing_array).astype(int)
    minimum = np.maximum(coordinates.min(axis=0) - padding_voxels, 0)
    maximum = np.minimum(
        coordinates.max(axis=0) + padding_voxels + 1,
        np.asarray(image.shape),
    )
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(minimum, maximum))
    image_crop = image[slices]
    renal_crop = renal_mask[slices]

    target_spacing = VESSEL_TARGET_SPACING_MM.get(detail, 2.0)
    zoom_factors = np.minimum(1.0, spacing_array / target_spacing)
    image_small = ndi.zoom(
        image_crop,
        zoom=tuple(zoom_factors),
        order=1,
        prefilter=False,
    )
    renal_small = ndi.zoom(
        renal_crop.astype(np.uint8),
        zoom=tuple(zoom_factors),
        order=0,
        prefilter=False,
    ).astype(bool)

    effective_spacing = spacing_array / zoom_factors
    distance_from_renal = ndi.distance_transform_edt(
        ~renal_small,
        sampling=tuple(effective_spacing),
    )
    region = distance_from_renal <= 24.0

    clipped = np.clip(image_small, -50.0, 500.0)
    normalized = ((clipped + 50.0) / 550.0).astype(np.float32)
    vesselness = frangi(
        normalized,
        sigmas=(0.8, 1.4, 2.2),
        black_ridges=False,
    )

    enhanced_tissue = region & (image_small >= 80.0) & (image_small <= 650.0)
    values = vesselness[enhanced_tissue]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 20:
        return None

    quantile = VESSEL_SENSITIVITY_QUANTILES.get(sensitivity, 99.2)
    threshold = float(np.percentile(values, quantile))
    candidates = enhanced_tissue & (vesselness >= threshold)
    candidates = ndi.binary_closing(candidates, iterations=1)
    candidates = remove_small_objects(candidates, min_size=6)
    if not np.any(candidates):
        return None

    mesh = _mesh_from_mask(
        candidates,
        affine,
        origin_voxels=minimum.astype(float),
        voxel_scale=1.0 / zoom_factors,
        step_size=1,
    )
    if mesh is not None:
        mesh["vesselness_threshold"] = threshold
        mesh["effective_spacing_mm"] = effective_spacing.astype(np.float32)
    return mesh
