"""Research-only partial nephrectomy planning metrics.

Everything in this module is computed from labelmaps in physical millimetres. The scores are
automated approximations of published nephrometry systems (RENAL, PADUA). They have not been
validated against expert scoring and must not be used for clinical decisions.

The engine works on a crop around the selected tumour's renal unit so it stays within the
memory limits of small hosted instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .anatomy import UNKNOWN, great_vessel_midline_x, side_of
from .labels import CanonicalLabel

STRUCTURE_18 = ndi.generate_binary_structure(rank=3, connectivity=2)

CROP_MARGIN_MM = 30.0
FINEST_WORK_SPACING_MM = 0.6
ENVELOPE_CLOSING_MM = 15.0
SINUS_MIN_DEPTH_MM = 4.0
SINUS_FAT_HU = -30.0  # fat inside a kidney label; medulla and urine overlap at 0-50 HU
MIN_TUMOR_ML = 0.01
MIN_CONTRALATERAL_ML = 10.0
CONTACT_TOLERANCE_MM = 1.0
BRANCH_MIN_LENGTH_MM = 8.0
ROOT_MIN_EXTRARENAL_MM = 10.0  # a renal (or accessory) artery must start outside the kidney
FEEDING_BRANCH_MIN_TUMOR_FRACTION = 0.05

RENAL_RISK = ((4, 6, "Low"), (7, 9, "Intermediate"), (10, 12, "High"))
PADUA_RISK = ((6, 7, "Low"), (8, 9, "Intermediate"), (10, 14, "High"))

DISTANCE_TARGETS = (
    ("Collecting system", CanonicalLabel.COLLECTING_SYSTEM),
    ("Renal artery", CanonicalLabel.RENAL_ARTERY),
    ("Renal vein", CanonicalLabel.RENAL_VEIN),
    ("Aorta", CanonicalLabel.AORTA),
    ("Inferior vena cava", CanonicalLabel.IVC),
    ("Duodenum", CanonicalLabel.DUODENUM),
    ("Colon", CanonicalLabel.COLON),
    ("Pancreas", CanonicalLabel.PANCREAS),
    ("Spleen", CanonicalLabel.SPLEEN),
    ("Liver", CanonicalLabel.LIVER),
    ("Adrenal gland", CanonicalLabel.ADRENAL),
    ("Psoas", CanonicalLabel.PSOAS),
)


class PlanningError(ValueError):
    pass


@dataclass
class RenalUnit:
    """Masks and geometry for one tumour and its kidney, on a cropped working grid."""

    tumor_index: int
    side: str
    affine: np.ndarray  # working-grid voxel -> world (RAS mm)
    spacing: tuple[float, float, float]
    crop_start: tuple[int, int, int]
    stride: tuple[int, int, int]
    shape: tuple[int, int, int]
    tumor: np.ndarray
    other_tumor: np.ndarray
    kidney: np.ndarray
    cyst: np.ndarray
    labels: np.ndarray  # cropped canonical segmentation
    image: np.ndarray | None
    contralateral_parenchyma_ml: float
    envelope: np.ndarray = field(init=False)
    sinus: np.ndarray | None = field(init=False, default=None)
    sinus_source: str = field(init=False, default="unavailable")
    tumor_distance: np.ndarray = field(init=False)
    tumor_nearest: np.ndarray | None = field(init=False)
    axes: dict[str, np.ndarray] = field(init=False)
    center: np.ndarray = field(init=False)

    @property
    def voxel_ml(self) -> float:
        return float(np.prod(self.spacing)) / 1000.0

    def mask(self, label: CanonicalLabel) -> np.ndarray:
        return self.labels == int(label)

    def world(self, voxels: np.ndarray) -> np.ndarray:
        voxels = np.atleast_2d(np.asarray(voxels, dtype=float))
        return voxels @ self.affine[:3, :3].T + self.affine[:3, 3]


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------


def _bounds(mask: np.ndarray) -> list[tuple[int, int]] | None:
    if not mask.any():
        return None
    bounds = []
    for axis in range(mask.ndim):
        other = tuple(value for value in range(mask.ndim) if value != axis)
        indices = np.flatnonzero(np.any(mask, axis=other))
        bounds.append((int(indices[0]), int(indices[-1]) + 1))
    return bounds


def _edt(mask: np.ndarray, spacing, return_indices: bool = False):
    """Distance (mm) from every True voxel to the nearest False voxel."""

    return ndi.distance_transform_edt(
        mask, sampling=tuple(float(value) for value in spacing), return_indices=return_indices
    )


def _closing_mm(mask: np.ndarray, spacing, radius_mm: float) -> np.ndarray:
    """Exact Euclidean morphological closing using two distance transforms."""

    if not mask.any():
        return mask.copy()
    pad = [int(np.ceil(radius_mm / float(value))) + 2 for value in spacing]
    padded = np.pad(mask, [(value, value) for value in pad])
    dilated = _edt(~padded, spacing) <= radius_mm
    closed = _edt(dilated, spacing) > radius_mm
    unpad = tuple(slice(value, -value) for value in pad)
    return closed[unpad] | mask


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(mask)


def _gap_mm(first: np.ndarray, second: np.ndarray, spacing) -> float:
    """Approximate surface gap between two voxel centres (centre distance − voxel extent)."""

    vector = (np.asarray(first, float) - np.asarray(second, float)) * np.asarray(spacing, float)
    distance = float(np.linalg.norm(vector))
    if distance == 0.0:
        return 0.0
    extent = float(np.sum(np.abs(vector / distance) * np.asarray(spacing, float)))
    return max(0.0, distance - extent)


def _risk(total: int, table) -> str:
    for low, high, name in table:
        if low <= total <= high:
            return name
    return "—"


def nearness_points(gap_mm: float) -> int:
    """R.E.N.A.L. N: ≥7 mm = 1, >4 to <7 mm = 2, ≤4 mm = 3."""

    return 1 if gap_mm >= 7.0 else 2 if gap_mm > 4.0 else 3


def _risk_range(low: int, high: int, table) -> str:
    first, second = _risk(low, table), _risk(high, table)
    return first if first == second else f"{first}–{second}"


def _principal_axes(points: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    center = points.mean(axis=0)
    if points.shape[0] < 4:
        identity = np.eye(3)
        return center, {"long": identity[2], "middle": identity[0], "short": identity[1]}
    values, vectors = np.linalg.eigh(np.cov((points - center).T))
    order = np.argsort(values)[::-1]
    long_axis, middle_axis, short_axis = (vectors[:, index] for index in order)
    if long_axis[2] < 0:  # point towards superior
        long_axis = -long_axis
    if short_axis[1] < 0:  # point towards anterior
        short_axis = -short_axis
    middle_axis = np.cross(short_axis, long_axis)
    return center, {"long": long_axis, "middle": middle_axis, "short": short_axis}


def _sample(points: np.ndarray, maximum: int = 200_000, seed: int = 0) -> np.ndarray:
    if points.shape[0] <= maximum:
        return points
    generator = np.random.default_rng(seed)
    return points[generator.choice(points.shape[0], maximum, replace=False)]


# --------------------------------------------------------------------------------------
# Renal unit extraction
# --------------------------------------------------------------------------------------


def tumor_components(
    segmentation: np.ndarray, spacing
) -> list[dict[str, Any]]:
    """Tumour components sorted by volume, matching ``analysis.component_measurements``."""

    tumor = segmentation == int(CanonicalLabel.TUMOR)
    bounds = _bounds(tumor)
    if bounds is None:
        return []
    crop = tuple(slice(low, high) for low, high in bounds)
    components, count = ndi.label(tumor[crop], structure=STRUCTURE_18)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    sizes = ndi.sum_labels(np.ones_like(components), components, index=np.arange(1, count + 1))
    items = []
    for component_id, voxels in zip(range(1, count + 1), sizes):  # noqa: B905
        volume = float(voxels) * voxel_ml
        if volume >= MIN_TUMOR_ML:
            items.append({"component": component_id, "volume_ml": volume})
    items.sort(key=lambda item: item["volume_ml"], reverse=True)
    return items


def extract_renal_unit(
    segmentation: np.ndarray,
    affine: np.ndarray,
    spacing,
    image: np.ndarray | None = None,
    tumor_index: int = 0,
) -> RenalUnit:
    spacing = tuple(float(value) for value in spacing)
    renal_labels = [
        int(CanonicalLabel.KIDNEY),
        int(CanonicalLabel.TUMOR),
        int(CanonicalLabel.CYST),
        int(CanonicalLabel.COLLECTING_SYSTEM),
    ]
    renal = np.isin(segmentation, renal_labels)
    bounds = _bounds(renal)
    if bounds is None:
        raise PlanningError("No kidney or tumour labels are present.")
    first_crop = tuple(slice(low, high) for low, high in bounds)
    first_offset = np.array([low for low, _high in bounds])
    seg_first = segmentation[first_crop]

    tumor_mask = seg_first == int(CanonicalLabel.TUMOR)
    tumor_labels, tumor_count = ndi.label(tumor_mask, structure=STRUCTURE_18)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    sizes = ndi.sum_labels(
        np.ones_like(tumor_labels), tumor_labels, index=np.arange(1, tumor_count + 1)
    )
    ranked = [
        component_id
        for component_id, voxels in sorted(
            zip(range(1, tumor_count + 1), sizes),  # noqa: B905
            key=lambda item: item[1],
            reverse=True,
        )
        if voxels * voxel_ml >= MIN_TUMOR_ML
    ]
    if not ranked:
        raise PlanningError("This case has no tumour label to plan around.")
    if tumor_index < 0 or tumor_index >= len(ranked):
        raise PlanningError(f"Tumour {tumor_index + 1} does not exist.")
    chosen_tumor = tumor_labels == ranked[tumor_index]

    renal_components, renal_count = ndi.label(
        np.isin(seg_first, renal_labels), structure=STRUCTURE_18
    )
    unit_ids = np.unique(renal_components[chosen_tumor])
    unit_ids = unit_ids[unit_ids > 0]
    unit = np.isin(renal_components, unit_ids)
    kidney_first = seg_first == int(CanonicalLabel.KIDNEY)
    if not np.any(unit & kidney_first):
        # Tumour not connected to parenchyma: attach the nearest kidney component.
        tumor_centroid = np.array(ndi.center_of_mass(chosen_tumor))
        best_id, best_distance = None, np.inf
        for component_id in range(1, renal_count + 1):
            component = renal_components == component_id
            if not np.any(component & kidney_first):
                continue
            centroid = np.array(ndi.center_of_mass(component))
            distance = np.linalg.norm((centroid - tumor_centroid) * np.asarray(spacing))
            if distance < best_distance:
                best_id, best_distance = component_id, distance
        if best_id is None:
            raise PlanningError("No kidney parenchyma was found for this tumour.")
        unit |= renal_components == best_id

    contralateral_voxels = 0
    contralateral_centroids: list[tuple[int, np.ndarray]] = []
    for component_id in range(1, renal_count + 1):
        component = renal_components == component_id
        if np.any(component & unit):
            continue
        kidney_voxels = int(np.count_nonzero(component & kidney_first))
        if kidney_voxels * voxel_ml >= MIN_CONTRALATERAL_ML:
            contralateral_voxels += kidney_voxels
            contralateral_centroids.append(
                (kidney_voxels, np.array(ndi.center_of_mass(component & kidney_first)))
            )

    # Working crop: renal unit plus a physical margin, clipped to the volume.
    unit_bounds = _bounds(unit)
    margin = [int(np.ceil(CROP_MARGIN_MM / value)) for value in spacing]
    start = [
        max(0, unit_bounds[axis][0] + first_offset[axis] - margin[axis]) for axis in range(3)
    ]
    stop = [
        min(segmentation.shape[axis], unit_bounds[axis][1] + first_offset[axis] + margin[axis])
        for axis in range(3)
    ]
    stride = tuple(
        max(1, int(np.floor(1.0 / value))) if value < FINEST_WORK_SPACING_MM else 1
        for value in spacing
    )
    work = tuple(slice(start[axis], stop[axis], stride[axis]) for axis in range(3))
    labels = np.ascontiguousarray(segmentation[work])

    def to_work(first_mask: np.ndarray) -> np.ndarray:
        full = np.zeros(tuple(stop[axis] - start[axis] for axis in range(3)), dtype=bool)
        source = []
        target = []
        for axis in range(3):
            low = max(start[axis], first_offset[axis])
            high = min(stop[axis], first_offset[axis] + first_mask.shape[axis])
            source.append(slice(low - first_offset[axis], high - first_offset[axis]))
            target.append(slice(low - start[axis], high - start[axis]))
        full[tuple(target)] = first_mask[tuple(source)]
        return np.ascontiguousarray(full[:: stride[0], :: stride[1], :: stride[2]])

    unit_work = to_work(unit)
    tumor_work = to_work(chosen_tumor)
    if not tumor_work.any():
        raise PlanningError("The selected tumour is smaller than the working grid.")

    work_spacing = tuple(spacing[axis] * stride[axis] for axis in range(3))
    translate = np.eye(4)
    translate[:3, 3] = start
    scale = np.diag([*stride, 1.0])
    work_affine = np.asarray(affine, dtype=float) @ translate @ scale

    kidney_work = (labels == int(CanonicalLabel.KIDNEY)) & unit_work
    if not kidney_work.any():
        raise PlanningError("No kidney parenchyma was found for this tumour.")
    # Laterality from anatomy (opposite kidney, else aorta/IVC), never the image centre.
    kidney_world = work_affine[:3, :3] @ np.array(ndi.center_of_mass(kidney_work)) + work_affine[
        :3, 3
    ]
    if contralateral_centroids:
        _voxels, other_voxel = max(contralateral_centroids, key=lambda item: item[0])
        other_world = np.asarray(affine, dtype=float) @ np.append(other_voxel + first_offset, 1.0)
        side = side_of(float(kidney_world[0]), 0.5 * (kidney_world[0] + other_world[0]), 0.0)
    else:
        midline = great_vessel_midline_x(segmentation, affine)
        side = side_of(float(kidney_world[0]), midline) if midline is not None else UNKNOWN

    renal_unit = RenalUnit(
        tumor_index=tumor_index,
        side=side,
        affine=work_affine,
        spacing=work_spacing,
        crop_start=tuple(int(value) for value in start),
        stride=stride,
        shape=labels.shape,
        tumor=tumor_work,
        other_tumor=(labels == int(CanonicalLabel.TUMOR)) & unit_work & ~tumor_work,
        kidney=kidney_work,
        cyst=(labels == int(CanonicalLabel.CYST)) & unit_work,
        labels=labels,
        image=(
            np.ascontiguousarray(image[work]).astype(np.float32, copy=False)
            if image is not None
            else None
        ),
        contralateral_parenchyma_ml=contralateral_voxels * voxel_ml,
    )
    _derive_geometry(renal_unit)
    return renal_unit


def _derive_geometry(unit: RenalUnit) -> None:
    spacing = unit.spacing
    collecting = unit.mask(CanonicalLabel.COLLECTING_SYSTEM)
    base = unit.kidney | collecting | unit.other_tumor
    unit.envelope = _fill_holes(_closing_mm(base, spacing, ENVELOPE_CLOSING_MM)) | base

    depth = _edt(unit.envelope, spacing)
    vessels = unit.mask(CanonicalLabel.RENAL_ARTERY) | unit.mask(CanonicalLabel.RENAL_VEIN)
    candidate = unit.envelope & ~unit.tumor & ~unit.other_tumor & ~unit.cyst & ~vessels
    candidate &= depth > SINUS_MIN_DEPTH_MM
    if unit.image is not None:
        # Inside the kidney label only fat counts: in corticomedullary scans the unenhanced
        # medulla (and urine) share 0-50 HU and were misread as sinus (review finding).
        candidate &= (~unit.kidney) | (unit.image < SINUS_FAT_HU)
        source = "Image-derived proxy (non-parenchymal space and fat inside the renal envelope)"
    else:
        candidate &= ~unit.kidney
        source = "Label-derived proxy (non-parenchymal space inside the renal envelope)"
    candidate |= collecting
    components, count = ndi.label(candidate, structure=STRUCTURE_18)
    sinus = None
    if count:
        sizes = ndi.sum_labels(
            np.ones_like(components), components, index=np.arange(1, count + 1)
        ) * unit.voxel_ml
        largest = int(np.argmax(sizes)) + 1
        keep = [largest] + [
            index + 1 for index, size in enumerate(sizes) if size >= 2.0 and index + 1 != largest
        ]
        if sizes[largest - 1] >= 0.5:
            sinus = np.isin(components, keep)
    unit.sinus = sinus
    unit.sinus_source = source if sinus is not None else "unavailable"

    distance, nearest = _edt(~unit.tumor, spacing, return_indices=True)
    unit.tumor_distance = distance.astype(np.float32)
    unit.tumor_nearest = nearest
    del distance

    renal_points = unit.world(_sample(np.argwhere(unit.kidney | unit.tumor)))
    unit.center, unit.axes = _principal_axes(renal_points)
    if unit.sinus is not None:
        sinus_center = unit.world(_sample(np.argwhere(unit.sinus), 50_000)).mean(axis=0)
        if np.dot(sinus_center - unit.center, unit.axes["middle"]) < 0:
            unit.axes["middle"] = -unit.axes["middle"]


# --------------------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------------------


def _nearest_to_tumor(unit: RenalUnit, target: np.ndarray) -> dict[str, Any] | None:
    if target is None or not target.any():
        return None
    distances = np.where(target, unit.tumor_distance, np.inf)
    flat = int(np.argmin(distances))
    target_voxel = np.array(np.unravel_index(flat, target.shape))
    tumor_voxel = unit.tumor_nearest[(slice(None), *target_voxel)]
    if unit.tumor[tuple(target_voxel)]:
        gap = 0.0
    else:
        gap = _gap_mm(tumor_voxel, target_voxel, unit.spacing)
    return {
        "gap_mm": float(gap),
        "tumor_point_mm": unit.world(tumor_voxel)[0].tolist(),
        "target_point_mm": unit.world(target_voxel)[0].tolist(),
    }


def max_diameter_mm(unit: RenalUnit) -> float:
    from scipy.spatial import ConvexHull, QhullError

    surface = unit.tumor & ~ndi.binary_erosion(unit.tumor)
    voxels = np.argwhere(surface)
    if voxels.shape[0] < 2:
        return float(max(unit.spacing))
    points = voxels * np.asarray(unit.spacing)
    try:
        hull_points = points[ConvexHull(points).vertices] if points.shape[0] >= 5 else points
    except QhullError:
        hull_points = points
    hull_points = _sample(hull_points, 3000)
    deltas = hull_points[:, None, :] - hull_points[None, :, :]
    lengths = np.linalg.norm(deltas, axis=2)
    first, second = np.unravel_index(int(np.argmax(lengths)), lengths.shape)
    vector = deltas[first, second]
    length = float(lengths[first, second])
    if length == 0.0:
        return float(max(unit.spacing))
    extent = float(np.sum(np.abs(vector / length) * np.asarray(unit.spacing)))
    return length + extent


def contact_surface(unit: RenalUnit) -> dict[str, float]:
    """Tumour surface area and the part in contact with renal parenchyma (cm²)."""

    from skimage.measure import marching_cubes

    bounds = _bounds(unit.tumor)
    pad = 2
    crop = tuple(
        slice(max(0, low - pad), min(unit.shape[axis], high + pad))
        for axis, (low, high) in enumerate(bounds)
    )
    tumor = np.pad(unit.tumor[crop], 1).astype(np.float32)
    parenchyma = np.pad(unit.kidney[crop], 1)
    vertices, faces, _normals, _values = marching_cubes(
        tumor, level=0.5, spacing=unit.spacing, method="lewiner"
    )
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    if not parenchyma.any():
        return {"tumor_surface_cm2": float(areas.sum() / 100.0), "contact_cm2": 0.0}
    parenchyma_distance = _edt(~parenchyma, unit.spacing)
    centroids_voxel = triangles.mean(axis=1) / np.asarray(unit.spacing)
    sampled = ndi.map_coordinates(parenchyma_distance, centroids_voxel.T, order=1, mode="nearest")
    tolerance = CONTACT_TOLERANCE_MM + 0.5 * max(unit.spacing)
    contact = sampled <= tolerance
    total = float(areas.sum() / 100.0)
    touching = float(areas[contact].sum() / 100.0)
    return {
        "tumor_surface_cm2": total,
        "contact_cm2": touching,
        "contact_fraction": touching / total if total else 0.0,
    }


def _projections(unit: RenalUnit, mask: np.ndarray, axis: str) -> np.ndarray:
    points = unit.world(_sample(np.argwhere(mask)))
    return (points - unit.center) @ unit.axes[axis]


def nephrometry(unit: RenalUnit, distances: dict[str, Any]) -> dict[str, Any]:
    notes: list[str] = []
    diameter = max_diameter_mm(unit)
    tumor_voxels = int(np.count_nonzero(unit.tumor))
    exophytic_fraction = float(np.count_nonzero(unit.tumor & ~unit.envelope)) / tumor_voxels

    # --- R
    radius_cm = diameter / 10.0
    r_points = 1 if radius_cm <= 4.0 else 2 if radius_cm < 7.0 else 3

    # --- E (endophytic if essentially nothing outside the envelope)
    if exophytic_fraction >= 0.5:
        e_points = 1
    elif exophytic_fraction >= 0.02:
        e_points = 2
    else:
        e_points = 3

    # --- N: nearest of collecting system (label) and renal sinus (only ever a proxy).
    collecting = distances.get("Collecting system")
    sinus = distances.get("Renal sinus")
    surfaces = [
        (name, item)
        for name, item in (("Collecting system", collecting), ("Renal sinus (proxy)", sinus))
        if item is not None
    ]
    n_points: int | None = None
    n_estimate: int | None = None
    n_source: str | None = None
    n_gap: float | None = None
    n_low, n_high = 1, 3
    if surfaces:
        n_source, nearest_surface = min(surfaces, key=lambda entry: entry[1]["gap_mm"])
        n_gap = float(nearest_surface["gap_mm"])
        n_estimate = nearness_points(n_gap)
    if collecting is not None:
        # The true sinus can only be nearer than the collecting system, never farther
        # than a measured label: N lies between N(collecting system) and 3.
        n_low = nearness_points(collecting["gap_mm"])
    if n_low == n_high:
        n_points = n_low
    if n_points is None:
        notes.append(
            "N is a range: the renal sinus has no reviewed outline. The estimate "
            f"({n_estimate if n_estimate is not None else '—'}, from {n_source or 'nothing'}) "
            "uses an image-derived proxy; check it on the images."
        )

    # --- A
    anterior = _projections(unit, unit.tumor, "short")
    anterior_fraction = float(np.mean(anterior > 0))
    a_letter = "a" if anterior_fraction >= 0.75 else "p" if anterior_fraction <= 0.25 else "x"

    # --- L (polar lines from the sinus extent along the kidney long axis)
    tumor_long = _projections(unit, unit.tumor, "long")
    renal_long = _projections(unit, unit.kidney | unit.tumor, "long")
    midline = 0.5 * (float(renal_long.min()) + float(renal_long.max()))
    if unit.sinus is not None:
        sinus_long = _projections(unit, unit.sinus, "long")
        lower_line, upper_line = np.percentile(sinus_long, [1, 99])
        polar_source = "renal sinus extent"
    else:
        span = float(renal_long.max() - renal_long.min())
        lower_line = float(renal_long.min()) + span / 3.0
        upper_line = float(renal_long.max()) - span / 3.0
        polar_source = "kidney thirds (no sinus proxy)"
        notes.append("Polar lines approximated by kidney thirds.")
    between_fraction = float(np.mean((tumor_long >= lower_line) & (tumor_long <= upper_line)))
    crosses_midline = float(tumor_long.min()) < midline < float(tumor_long.max())
    if between_fraction <= 0.01:
        l_points = 1
    elif between_fraction > 0.5 or crosses_midline or between_fraction >= 0.99:
        l_points = 3
    else:
        l_points = 2
    tumor_center_long = float(np.mean(tumor_long))
    if tumor_center_long > upper_line:
        pole = "Upper pole"
    elif tumor_center_long < lower_line:
        pole = "Lower pole"
    else:
        pole = "Interpolar"

    # --- hilar suffix
    hilar: bool | None = None
    vessels = unit.mask(CanonicalLabel.RENAL_ARTERY) | unit.mask(CanonicalLabel.RENAL_VEIN)
    if vessels.any():
        hilar_region = ~unit.envelope
        if unit.sinus is not None:
            hilar_region = hilar_region | unit.sinus
        main_vessels = vessels & hilar_region
        nearest_main = _nearest_to_tumor(unit, main_vessels)
        hilar = nearest_main is not None and nearest_main["gap_mm"] <= CONTACT_TOLERANCE_MM
    else:
        notes.append("Hilar (h) suffix not assessable without renal artery/vein labels.")

    base_points = r_points + e_points + l_points
    renal_total_min = base_points + (n_points if n_points is not None else n_low)
    renal_total_max = base_points + (n_points if n_points is not None else n_high)
    renal_total_estimate = (
        base_points + (n_points if n_points is not None else n_estimate)
        if (n_points is not None or n_estimate is not None)
        else None
    )
    suffix = a_letter + ("h" if hilar else "")
    if renal_total_min == renal_total_max:
        renal_string = f"{renal_total_min}{suffix}"
    else:
        renal_string = f"{renal_total_min}–{renal_total_max}{suffix}"
        if renal_total_estimate is not None:
            renal_string += f" (est. {renal_total_estimate})"

    # --- PADUA
    middle = _projections(unit, unit.tumor, "middle")
    medial_fraction = float(np.mean(middle > 0)) if unit.sinus is not None else None
    padua_items: dict[str, int | None] = {
        "Longitudinal location": 2 if lower_line <= tumor_center_long <= upper_line else 1,
        "Exophytic rate": e_points,
        "Renal rim": (
            None if medial_fraction is None else (2 if medial_fraction > 0.5 else 1)
        ),
        # Sinus involvement needs a reviewed sinus outline; the proxy is only an estimate.
        "Renal sinus": None,
        "Urinary collecting system": (
            None
            if collecting is None
            else (2 if collecting["gap_mm"] <= CONTACT_TOLERANCE_MM else 1)
        ),
        # PADUA: ≤4 cm, 4.1–7 cm, >7 cm (R.E.N.A.L. scores exactly 7 cm as 3; PADUA does not).
        "Tumour size": 1 if radius_cm <= 4.0 else 2 if radius_cm <= 7.0 else 3,
    }
    known_padua = sum(value for value in padua_items.values() if value is not None)
    unknown_padua = sum(1 for value in padua_items.values() if value is None)
    padua_min = known_padua + unknown_padua
    padua_max = known_padua + 2 * unknown_padua
    if padua_items["Urinary collecting system"] is None:
        notes.append("PADUA collecting-system item needs a collecting system outline.")
    padua_sinus_estimate = (
        None if sinus is None else (2 if sinus["gap_mm"] <= CONTACT_TOLERANCE_MM else 1)
    )

    return {
        "max_diameter_mm": diameter,
        "exophytic_fraction": exophytic_fraction,
        "anterior_fraction": anterior_fraction,
        "between_polar_lines_fraction": between_fraction,
        "crosses_midline": bool(crosses_midline),
        "polar_line_source": polar_source,
        "location": pole,
        "renal": {
            "R": r_points,
            "E": e_points,
            "N": n_points,
            "N_range": [n_points, n_points] if n_points is not None else [n_low, n_high],
            "N_estimate": n_points if n_points is not None else n_estimate,
            "N_source": n_source,
            "N_gap_mm": n_gap,
            "total_estimate": renal_total_estimate,
            "A": a_letter,
            "L": l_points,
            "hilar": hilar,
            "total_min": renal_total_min,
            "total_max": renal_total_max,
            "score": renal_string,
            "complexity": _risk_range(renal_total_min, renal_total_max, RENAL_RISK),
        },
        "padua": {
            "items": padua_items,
            "renal_sinus_estimate": padua_sinus_estimate,
            "face": a_letter,
            "total_min": padua_min,
            "total_max": padua_max,
            "score": (
                f"{padua_min}" if padua_min == padua_max else f"{padua_min}–{padua_max}"
            ) + a_letter,
            "risk": _risk_range(padua_min, padua_max, PADUA_RISK),
        },
        "notes": notes,
    }


def structure_distances(unit: RenalUnit) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if unit.sinus is not None:
        nearest = _nearest_to_tumor(unit, unit.sinus & ~unit.mask(CanonicalLabel.COLLECTING_SYSTEM))
        if nearest is None:
            nearest = _nearest_to_tumor(unit, unit.sinus)
        if nearest is not None:
            results["Renal sinus"] = {**nearest, "source": "proxy"}
    for name, label in DISTANCE_TARGETS:
        nearest = _nearest_to_tumor(unit, unit.mask(label))
        if nearest is not None:
            results[name] = {**nearest, "source": "label"}
    return results


def simulate_resection(
    unit: RenalUnit, margin_mm: float, distances: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    margin_mm = max(0.0, float(margin_mm))
    resected_parenchyma = unit.kidney & (unit.tumor_distance <= margin_mm) if margin_mm else (
        np.zeros_like(unit.kidney)
    )
    voxel_ml = unit.voxel_ml
    ipsilateral = float(np.count_nonzero(unit.kidney)) * voxel_ml
    removed = float(np.count_nonzero(resected_parenchyma)) * voxel_ml
    preserved = ipsilateral - removed
    contralateral = unit.contralateral_parenchyma_ml
    total_before = ipsilateral + contralateral
    total_after = preserved + contralateral
    warnings = []
    for name, item in distances.items():
        gap = item["gap_mm"]
        if gap <= margin_mm:
            warnings.append(
                {"structure": name, "gap_mm": gap, "level": "inside margin"}
            )
        elif gap <= margin_mm + 2.0:
            warnings.append({"structure": name, "gap_mm": gap, "level": "within 2 mm of margin"})
    return {
        "margin_mm": margin_mm,
        "technique": "Enucleation" if margin_mm == 0 else f"Resection with {margin_mm:g} mm margin",
        "tumor_ml": float(np.count_nonzero(unit.tumor)) * voxel_ml,
        "removed_parenchyma_ml": removed,
        "ipsilateral_parenchyma_ml": ipsilateral,
        "preserved_parenchyma_ml": preserved,
        "preserved_ipsilateral_pct": 100.0 * preserved / ipsilateral if ipsilateral else 0.0,
        "contralateral_parenchyma_ml": contralateral,
        "preserved_total_pct": 100.0 * total_after / total_before if total_before else 0.0,
        "ipsilateral_share_before_pct": (
            100.0 * ipsilateral / total_before if contralateral and total_before else None
        ),
        "ipsilateral_share_after_pct": (
            100.0 * preserved / total_after if contralateral and total_after else None
        ),
        "warnings": warnings,
        "mask": resected_parenchyma,
    }


# --------------------------------------------------------------------------------------
# Arterial territories
# --------------------------------------------------------------------------------------


def _skeleton_graph(mask: np.ndarray, spacing):
    import networkx as nx
    from skimage.morphology import skeletonize

    skeleton = skeletonize(mask)
    coordinates = np.argwhere(skeleton)
    graph = nx.Graph()
    if coordinates.shape[0] == 0:
        return graph, coordinates
    index = -np.ones(mask.shape, dtype=np.int64)
    index[tuple(coordinates.T)] = np.arange(coordinates.shape[0])
    graph.add_nodes_from(range(coordinates.shape[0]))
    offsets = [
        np.array(offset)
        for offset in np.ndindex(3, 3, 3)
        if offset > (1, 1, 1)
    ]
    for raw in offsets:
        offset = raw - 1
        neighbours = coordinates + offset
        inside = np.all((neighbours >= 0) & (neighbours < np.asarray(mask.shape)), axis=1)
        source = np.flatnonzero(inside)
        target = index[tuple(neighbours[inside].T)]
        valid = target >= 0
        weight = float(np.linalg.norm(offset * np.asarray(spacing)))
        graph.add_weighted_edges_from(
            (int(a), int(b), weight) for a, b in zip(source[valid], target[valid])  # noqa: B905
        )
    return graph, coordinates


BRANCH_COLORS = (
    (230, 159, 0),
    (86, 180, 233),
    (0, 158, 115),
    (240, 228, 66),
    (0, 114, 178),
    (213, 94, 0),
    (204, 121, 167),
    (150, 150, 150),
    (120, 60, 200),
    (60, 200, 160),
)


def arterial_territories(unit: RenalUnit, depth: int = 1) -> dict[str, Any] | None:
    """Approximate supply territories by nearest-branch (Voronoi) assignment.

    The artery centreline is turned into a tree rooted at the point farthest from the kidney.
    A bifurcation counts only when both child subtrees are at least
    ``BRANCH_MIN_LENGTH_MM`` long *and* reach the renal envelope, so capsular, adrenal and
    skeleton spurs are ignored. ``depth`` selects how many significant bifurcations to pass
    (1 ≈ anterior/posterior divisions, 2 ≈ segmental arteries).
    """

    import networkx as nx

    artery = unit.mask(CanonicalLabel.RENAL_ARTERY)
    if not artery.any():
        return None
    envelope_distance = _edt(~unit.envelope, unit.spacing)
    graph, coordinates = _skeleton_graph(artery, unit.spacing)
    if coordinates.shape[0] == 0:
        return None
    node_renal_distance = envelope_distance[tuple(coordinates.T)]
    reaches = node_renal_distance <= 2.0

    lineage = np.zeros(coordinates.shape[0], dtype=np.int32)
    lineage_info: dict[int, dict[str, Any]] = {}
    next_id = 1
    centerline_segments: dict[int, list[np.ndarray]] = {}
    component_rank = 0

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    fragments = 0
    for component in components:
        nodes = np.fromiter(component, dtype=np.int64)
        if not reaches[nodes].any():
            continue
        root = int(nodes[np.argmax(node_renal_distance[nodes])])
        if node_renal_distance[root] < ROOT_MIN_EXTRARENAL_MM:
            # Starts inside/at the kidney: a disconnected piece of the tree (segmentation
            # gap), not an accessory artery. Never give it a territory of its own.
            fragments += 1
            continue
        predecessors, distance = nx.dijkstra_predecessor_and_distance(graph.subgraph(nodes), root)
        parent = {node: (pred[0] if pred else None) for node, pred in predecessors.items()}
        children: dict[int, list[int]] = {node: [] for node in parent}
        for node, node_parent in parent.items():
            if node_parent is not None:
                children[node_parent].append(node)
        order_desc = sorted(distance, key=distance.get, reverse=True)
        subtree_length = {node: 0.0 for node in parent}
        subtree_reaches = {node: bool(reaches[node]) for node in parent}
        for node in order_desc:
            node_parent = parent[node]
            if node_parent is None:
                continue
            edge = graph[node_parent][node]["weight"]
            subtree_length[node_parent] += subtree_length[node] + edge
            subtree_reaches[node_parent] |= subtree_reaches[node]

        def significant(
            child: int,
            node: int,
            lengths: dict[int, float] = subtree_length,
            reach: dict[int, bool] = subtree_reaches,
        ) -> bool:
            return (
                reach[child]
                and lengths[child] + graph[node][child]["weight"] >= BRANCH_MIN_LENGTH_MM
            )

        accessory = component_rank > 0
        component_rank += 1
        root_id = next_id
        next_id += 1
        lineage_info[root_id] = {
            "order": 0,
            "accessory": accessory,
            "component": component_rank,
            "parent": None,
        }
        order = {root: 0}
        lineage[root] = root_id
        for node in sorted(distance, key=distance.get):
            kids = children[node]
            strong = [child for child in kids if significant(child, node)]
            is_bifurcation = len(strong) >= 2
            for child in kids:
                if is_bifurcation and child in strong and order[node] < depth:
                    order[child] = order[node] + 1
                    lineage[child] = next_id
                    lineage_info[next_id] = {
                        "order": order[child],
                        "accessory": accessory,
                        "component": component_rank,
                        "parent": int(lineage[node]),
                    }
                    next_id += 1
                else:
                    order[child] = order[node]
                    lineage[child] = lineage[node]
                segment = unit.world(coordinates[[node, child]])
                centerline_segments.setdefault(int(lineage[child]), []).append(segment)

    # Only terminal lineages (those not divided further) supply territory.
    divided = {info["parent"] for info in lineage_info.values() if info["parent"] is not None}
    terminal = np.array([value > 0 and int(value) not in divided for value in lineage])
    seeds = np.zeros(unit.shape, dtype=np.int32)
    seed_nodes = np.flatnonzero(terminal & reaches)
    if seed_nodes.size == 0:
        return None
    seeds[tuple(coordinates[seed_nodes].T)] = lineage[seed_nodes]
    _distance, nearest = _edt(seeds == 0, unit.spacing, return_indices=True)
    territory_map = seeds[tuple(nearest)]
    del nearest

    renal = unit.kidney | unit.tumor
    voxel_ml = unit.voxel_ml
    tumor_total = float(np.count_nonzero(unit.tumor)) * voxel_ml
    parenchyma_total = float(np.count_nonzero(unit.kidney)) * voxel_ml
    span = float(np.ptp(_projections(unit, renal, "long"))) or 1.0
    branches = []
    for lineage_id in np.unique(territory_map[renal]):
        if lineage_id == 0:
            continue
        region = territory_map == lineage_id
        parenchyma_ml = float(np.count_nonzero(region & unit.kidney)) * voxel_ml
        tumor_ml = float(np.count_nonzero(region & unit.tumor)) * voxel_ml
        points = unit.world(_sample(np.argwhere(region & renal), 50_000))
        relative = points.mean(axis=0) - unit.center
        along = float(np.dot(relative, unit.axes["long"]))
        front = float(np.dot(relative, unit.axes["short"]))
        vertical = "upper" if along > span / 6 else "lower" if along < -span / 6 else "middle"
        facing = "anterior" if front > 2.0 else "posterior" if front < -2.0 else "central"
        info = lineage_info.get(int(lineage_id), {})
        branches.append(
            {
                "id": int(lineage_id),
                "region": f"{vertical}, {facing}",
                "accessory": bool(info.get("accessory", False)),
                "order": int(info.get("order", 0)),
                "parenchyma_ml": parenchyma_ml,
                "parenchyma_pct": (
                    100.0 * parenchyma_ml / parenchyma_total if parenchyma_total else 0.0
                ),
                "tumor_ml": tumor_ml,
                "tumor_pct": 100.0 * tumor_ml / tumor_total if tumor_total else 0.0,
            }
        )
    branches.sort(key=lambda item: item["parenchyma_ml"], reverse=True)
    for number, branch in enumerate(branches, start=1):
        branch["name"] = f"{'Accessory ' if branch['accessory'] else ''}Branch {number}"
        branch["color"] = BRANCH_COLORS[(number - 1) % len(BRANCH_COLORS)]
        branch["feeds_tumor"] = branch["tumor_pct"] >= 100 * FEEDING_BRANCH_MIN_TUMOR_FRACTION

    feeding = [branch for branch in branches if branch["feeds_tumor"]]
    selective_ml = sum(branch["parenchyma_ml"] for branch in feeding)
    artery_count = sum(
        1 for info in lineage_info.values() if info["order"] == 0
    )
    warnings = []
    if fragments:
        warnings.append(
            f"{fragments} disconnected artery fragment(s) inside the kidney were ignored; "
            "the vessel segmentation has gaps, so nearby territories are unreliable."
        )
    if len(branches) < 2:
        warnings.append("Fewer than two branches resolved: no selective-clamping estimate.")
    reliable = not warnings
    return {
        "depth": depth,
        "renal_artery_count": artery_count,
        "ignored_fragments": fragments,
        "warnings": warnings,
        # Ischaemic volumes are only shown when the tree looks complete.
        "ischemia_estimate_reliable": reliable,
        "branches": branches,
        "feeding_branches": [branch["name"] for branch in feeding],
        "main_clamp_ischemic_ml": parenchyma_total,
        "selective_clamp_ischemic_ml": selective_ml,
        "selective_clamp_ischemic_pct": (
            100.0 * selective_ml / parenchyma_total if parenchyma_total else 0.0
        ),
        "map": territory_map,
        "centerlines": {
            int(key): np.concatenate(
                [np.vstack((segment, np.full((1, 3), np.nan))) for segment in value]
            )
            for key, value in centerline_segments.items()
        },
        "method": "Centreline tree + nearest-branch (Voronoi) assignment; not a perfusion study",
    }


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------


def plan_geometry(
    segmentation: np.ndarray,
    affine: np.ndarray,
    spacing,
    image: np.ndarray | None = None,
    tumor_index: int = 0,
    branch_depth: int = 1,
) -> dict[str, Any]:
    """Margin-independent (expensive) part of the plan. Safe to cache; never mutated."""

    unit = extract_renal_unit(segmentation, affine, spacing, image, tumor_index)
    distances = structure_distances(unit)
    geometry = {
        "unit": unit,
        "distances": distances,
        "scores": nephrometry(unit, distances),
        "contact": contact_surface(unit),
        "territories": arterial_territories(unit, branch_depth),
    }
    # Release large intermediates that later (cached) steps do not need.
    unit.tumor_nearest = None
    unit.image = None
    return geometry


def assemble_plan(geometry: dict[str, Any], margin_mm: float) -> dict[str, Any]:
    """Cheap margin-dependent part; returns JSON-safe ``summary`` and display arrays."""

    unit: RenalUnit = geometry["unit"]
    resection = simulate_resection(unit, margin_mm, geometry["distances"])
    resection_mask = resection.pop("mask")
    territories = geometry["territories"]
    territory_summary = (
        {key: value for key, value in territories.items() if key not in ("map", "centerlines")}
        if territories
        else None
    )
    display = {
        "affine": unit.affine,
        "spacing": unit.spacing,
        "crop_start": unit.crop_start,
        "stride": unit.stride,
        "tumor": unit.tumor,
        "kidney": unit.kidney,
        "resection": resection_mask,
        "sinus": unit.sinus,
        "structures": {
            name: unit.mask(label)
            for name, label in (
                ("Renal artery", CanonicalLabel.RENAL_ARTERY),
                ("Renal vein", CanonicalLabel.RENAL_VEIN),
                ("Collecting system", CanonicalLabel.COLLECTING_SYSTEM),
            )
            if unit.mask(label).any()
        },
        "territory_map": territories["map"] if territories else None,
        "centerlines": territories["centerlines"] if territories else None,
        "distance_lines": {
            name: np.array([item["tumor_point_mm"], item["target_point_mm"]])
            for name, item in geometry["distances"].items()
        },
    }
    summary = {
        "tumor_index": unit.tumor_index,
        "side": unit.side,
        "working_spacing_mm": list(unit.spacing),
        "sinus_source": unit.sinus_source,
        "scores": geometry["scores"],
        "contact_surface": geometry["contact"],
        "distances": geometry["distances"],
        "resection": resection,
        "territories": territory_summary,
        "research_use_only": True,
    }
    return {"summary": summary, "display": display}


def build_plan(
    segmentation: np.ndarray,
    affine: np.ndarray,
    spacing,
    image: np.ndarray | None = None,
    tumor_index: int = 0,
    margin_mm: float = 3.0,
    branch_depth: int = 1,
) -> dict[str, Any]:
    """Compute a complete research plan for one tumour."""

    geometry = plan_geometry(segmentation, affine, spacing, image, tumor_index, branch_depth)
    return assemble_plan(geometry, margin_mm)


def tumor_center_voxel(plan: dict[str, Any]) -> tuple[int, int, int]:
    """Tumour centroid in the original (full-volume) voxel grid."""

    display = plan["display"]
    centroid = np.array(ndi.center_of_mass(display["tumor"]))
    full = np.asarray(display["crop_start"]) + centroid * np.asarray(display["stride"])
    return tuple(int(round(value)) for value in full)


def resect_with_margin(plan: dict[str, Any], margin_mm: float) -> np.ndarray:
    """Resection mask for a new margin without recomputing the plan (cheap)."""

    display = plan["display"]
    distance = _edt(~display["tumor"], display["spacing"])
    return display["kidney"] & (distance <= margin_mm) if margin_mm > 0 else np.zeros_like(
        display["kidney"]
    )
