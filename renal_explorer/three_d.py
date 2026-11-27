from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import frangi, sato
from skimage.measure import marching_cubes
from skimage.morphology import remove_small_objects, skeletonize

from .labels import CanonicalLabel

MeshData = dict[str, Any]

SURFACE_PROFILES = {
    "Fast": {
        "target_spacing_mm": 1.5,
        "smoothing_factor": 0.30,
        "max_faces": 60_000,
        "max_displacement_p95_mm": 1.25,
        "volume_tolerance_pct": 4.0,
    },
    "Balanced": {
        "target_spacing_mm": 1.0,
        "smoothing_factor": 0.35,
        "max_faces": 120_000,
        "max_displacement_p95_mm": 0.90,
        "volume_tolerance_pct": 3.0,
    },
    "Detailed": {
        "target_spacing_mm": 0.75,
        "smoothing_factor": 0.25,
        "max_faces": 240_000,
        "max_displacement_p95_mm": 0.60,
        "volume_tolerance_pct": 2.5,
    },
}

# Display-only smoothing. Volumes, distances and scores come from the voxel labels, never
# from these surfaces, so surfaces may be smoothed enough to remove slice "terraces".
# The signed-distance field is blurred by this many voxels along each axis (so a 5 mm slice
# axis is smoothed at its own scale), and smoothing may move the surface by up to half of
# the coarsest voxel, the height of a terrace. Volume stays calibrated by the iso-level.
DISPLAY_SDF_SIGMA_VOXELS = 0.6
DISPLAY_MAX_DISPLACEMENT_FRACTION = 0.5

VESSEL_THRESHOLDS = {
    "Strict": (99.6, 99.0),
    "Balanced": (99.0, 97.8),
    "Sensitive": (98.5, 96.5),
}

# "Reference" is the geometrically authoritative surface used for exports: marching cubes
# on the native grid, with no blur, smoothing or decimation.
REFERENCE_QUALITY = "Reference"

VESSEL_TARGET_SPACING_MM = {
    "Fast": 2.0,
    "Balanced": 1.5,
    "Detailed": 1.0,
}

VESSEL_LABELS = {
    int(CanonicalLabel.RENAL_ARTERY),
    int(CanonicalLabel.RENAL_VEIN),
    int(CanonicalLabel.AORTA),
    int(CanonicalLabel.IVC),
}


def _voxel_to_world(vertices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    affine_array = np.asarray(affine, dtype=float)
    world_vertices = np.empty_like(vertices, dtype=float)
    for world_axis in range(3):
        world_vertices[:, world_axis] = affine_array[world_axis, 3]
        for voxel_axis in range(3):
            world_vertices[:, world_axis] += (
                vertices[:, voxel_axis] * affine_array[world_axis, voxel_axis]
            )
    return world_vertices


def _mesh_volume_mm3(vertices: np.ndarray, faces: np.ndarray) -> float:
    if vertices.size == 0 or faces.size == 0:
        return 0.0
    centered = vertices.astype(np.float64) - np.mean(vertices, axis=0, dtype=np.float64)
    signed = np.einsum(
        "ij,ij->i",
        centered[faces[:, 0]],
        np.cross(centered[faces[:, 1]], centered[faces[:, 2]]),
    )
    return abs(float(np.sum(signed) / 6.0))


def _mesh_area_mm2(vertices: np.ndarray, faces: np.ndarray) -> float:
    first = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    second = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    return float(np.sum(np.linalg.norm(np.cross(first, second), axis=1)) / 2.0)


def _topology_metrics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    triangle_vectors = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    degenerate = int(np.count_nonzero(np.linalg.norm(triangle_vectors, axis=1) <= 1e-8))
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
        axis=0,
    )
    edges.sort(axis=1)
    _unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = int(np.count_nonzero(edge_counts == 1))
    nonmanifold_edges = int(np.count_nonzero(edge_counts > 2))
    return {
        "degenerate_triangles": degenerate,
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "watertight": boundary_edges == 0 and nonmanifold_edges == 0 and degenerate == 0,
    }


def _polydata(vertices: np.ndarray, faces: np.ndarray) -> Any:
    # VTK is intentionally imported only when a 3D mesh is requested. Loading its
    # native libraries while Streamlit starts can exhaust a small Community Cloud
    # instance before the first page is rendered.
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

    points = vtkPoints()
    points.SetData(numpy_to_vtk(vertices.astype(np.float64), deep=True))
    cells = vtkCellArray()
    offsets = np.arange(0, (faces.shape[0] + 1) * 3, 3, dtype=np.int64)
    connectivity = faces.astype(np.int64, copy=False).reshape(-1)
    cells.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(connectivity, deep=True),
    )
    polydata = vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    return polydata


def _polydata_arrays(polydata: Any) -> tuple[np.ndarray, np.ndarray]:
    from vtkmodules.util.numpy_support import vtk_to_numpy

    vertices = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64, copy=True)
    connectivity = vtk_to_numpy(
        polydata.GetPolys().GetConnectivityArray()
    ).astype(np.int64, copy=False)
    faces = connectivity.reshape(-1, 3).astype(np.int32, copy=True)
    return vertices, faces


def _windowed_sinc_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    smoothing_factor: float,
) -> np.ndarray:
    from vtkmodules.vtkFiltersCore import vtkWindowedSincPolyDataFilter

    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(_polydata(vertices, faces))
    smoother.SetNumberOfIterations(int(round(20 + smoothing_factor * 40)))
    smoother.SetPassBand(10.0 ** (-4.0 * smoothing_factor))
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    smoothed_vertices, _faces = _polydata_arrays(smoother.GetOutput())
    return smoothed_vertices


def _constrain_smoothing(
    raw_vertices: np.ndarray,
    smoothed_vertices: np.ndarray,
    faces: np.ndarray,
    source_volume_mm3: float,
    maximum_p95_mm: float,
    volume_tolerance_pct: float,
) -> tuple[np.ndarray, float]:
    displacement = np.linalg.norm(smoothed_vertices - raw_vertices, axis=1)
    p95 = float(np.percentile(displacement, 95)) if displacement.size else 0.0
    alpha = min(1.0, maximum_p95_mm / p95) if p95 > 0 else 1.0

    best = raw_vertices
    for candidate_alpha in np.linspace(alpha, 0.0, 13):
        candidate = raw_vertices + candidate_alpha * (smoothed_vertices - raw_vertices)
        candidate_volume = _mesh_volume_mm3(candidate, faces)
        volume_error_pct = (
            abs(candidate_volume - source_volume_mm3) / source_volume_mm3 * 100.0
            if source_volume_mm3 > 0
            else 0.0
        )
        best = candidate
        triangle_vectors = np.cross(
            candidate[faces[:, 1]] - candidate[faces[:, 0]],
            candidate[faces[:, 2]] - candidate[faces[:, 0]],
        )
        has_degenerate_triangles = np.any(
            np.linalg.norm(triangle_vectors, axis=1) <= 1e-8
        )
        if volume_error_pct <= volume_tolerance_pct and not has_degenerate_triangles:
            break

    final_p95 = float(
        np.percentile(np.linalg.norm(best - raw_vertices, axis=1), 95)
    )
    return best, final_p95


def _decimate(
    vertices: np.ndarray,
    faces: np.ndarray,
    maximum_faces: int,
) -> tuple[np.ndarray, np.ndarray]:
    if faces.shape[0] <= maximum_faces:
        return vertices, faces
    from vtkmodules.vtkFiltersCore import vtkDecimatePro

    decimator = vtkDecimatePro()
    decimator.SetInputData(_polydata(vertices, faces))
    decimator.SetTargetReduction(1.0 - maximum_faces / faces.shape[0])
    decimator.PreserveTopologyOn()
    decimator.SplittingOff()
    decimator.SetFeatureAngle(60.0)
    decimator.SetMaximumError(1.0)
    decimator.Update()
    return _polydata_arrays(decimator.GetOutput())


def _quality_report(
    vertices: np.ndarray,
    faces: np.ndarray,
    source_volume_mm3: float,
    smoothing_p95_mm: float,
    target_spacing_mm: float,
    method: str,
) -> dict[str, Any]:
    mesh_volume_mm3 = _mesh_volume_mm3(vertices, faces)
    report = {
        "method": method,
        "vertices": int(vertices.shape[0]),
        "triangles": int(faces.shape[0]),
        "source_volume_ml": source_volume_mm3 / 1000.0,
        "mesh_volume_ml": mesh_volume_mm3 / 1000.0,
        "volume_error_pct": (
            abs(mesh_volume_mm3 - source_volume_mm3) / source_volume_mm3 * 100.0
            if source_volume_mm3 > 0
            else 0.0
        ),
        "surface_area_cm2": _mesh_area_mm2(vertices, faces) / 100.0,
        "smoothing_displacement_p95_mm": smoothing_p95_mm,
        "target_spacing_mm": target_spacing_mm,
    }
    report.update(_topology_metrics(vertices, faces))
    return report


def _output_shape_for_spacing(
    shape: tuple[int, int, int],
    spacing: np.ndarray,
    target_spacing_mm: float,
) -> np.ndarray:
    physical_extent = (np.asarray(shape, dtype=float) - 1.0) * spacing
    return np.maximum(2, np.rint(physical_extent / target_spacing_mm).astype(int) + 1)


def _resample(
    array: np.ndarray,
    output_shape: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    input_shape = np.asarray(array.shape, dtype=float)
    zoom = output_shape.astype(float) / input_shape
    resampled = ndi.zoom(
        array,
        zoom=tuple(zoom),
        order=order,
        prefilter=order > 1,
        grid_mode=False,
    )
    input_per_output = (input_shape - 1.0) / (np.asarray(resampled.shape) - 1.0)
    return resampled, input_per_output


def _calibrated_surface(
    signed_distance: np.ndarray,
    input_per_output: np.ndarray,
    origin: np.ndarray,
    affine: np.ndarray,
    source_volume_mm3: float,
    target_spacing_mm: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    lower = max(float(np.min(signed_distance)) + 1e-5, -1.5 * target_spacing_mm)
    upper = min(float(np.max(signed_distance)) - 1e-5, 1.5 * target_spacing_mm)
    best_vertices: np.ndarray | None = None
    best_faces: np.ndarray | None = None
    best_level = 0.0
    best_error = float("inf")

    for level in (0.0, *np.linspace(lower, upper, 11)):
        vertices, faces, _normals, _values = marching_cubes(
            signed_distance,
            level=float(level),
            step_size=1,
            allow_degenerate=False,
            method="lewiner",
        )
        voxel_vertices = vertices * input_per_output + origin
        world_vertices = _voxel_to_world(voxel_vertices, affine)
        volume_error = abs(_mesh_volume_mm3(world_vertices, faces) - source_volume_mm3)
        if volume_error < best_error:
            best_vertices = world_vertices
            best_faces = faces
            best_level = float(level)
            best_error = volume_error

    if best_vertices is None or best_faces is None:
        raise ValueError("Unable to extract a surface from this labelmap")

    search_half_width = (upper - lower) / 10.0
    for _iteration in range(3):
        levels = np.linspace(
            max(lower, best_level - search_half_width),
            min(upper, best_level + search_half_width),
            7,
        )
        for level in levels:
            vertices, faces, _normals, _values = marching_cubes(
                signed_distance,
                level=float(level),
                step_size=1,
                allow_degenerate=False,
                method="lewiner",
            )
            voxel_vertices = vertices * input_per_output + origin
            world_vertices = _voxel_to_world(voxel_vertices, affine)
            volume_error = abs(
                _mesh_volume_mm3(world_vertices, faces) - source_volume_mm3
            )
            if volume_error < best_error:
                best_vertices = world_vertices
                best_faces = faces
                best_level = float(level)
                best_error = volume_error
        search_half_width /= 3.0

    return best_vertices, best_faces, best_level


def _surface_from_binary_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    spacing: tuple[float, float, float],
    quality: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    reference = quality == REFERENCE_QUALITY
    profile = SURFACE_PROFILES.get(quality, SURFACE_PROFILES["Balanced"])
    target_spacing_mm = float(profile["target_spacing_mm"])
    spacing_array = np.asarray(spacing, dtype=float)
    source_volume_mm3 = float(np.count_nonzero(mask) * np.prod(spacing_array))
    if not reference and source_volume_mm3 < 5_000.0:
        small_structure_spacing = {
            "Fast": 1.0,
            "Balanced": 0.75,
            "Detailed": 0.60,
        }
        target_spacing_mm = min(
            target_spacing_mm,
            small_structure_spacing.get(quality, 0.75),
        )
    coordinates = np.argwhere(mask)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0) + 1
    padding = np.maximum(2, np.ceil(3.0 / spacing_array).astype(int))
    minimum = np.maximum(minimum - padding, 0)
    maximum = np.minimum(maximum + padding, np.asarray(mask.shape))
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(minimum, maximum))
    cropped = np.pad(mask[slices], 1)
    origin = minimum.astype(float) - 1.0

    signed_distance = ndi.distance_transform_edt(
        cropped,
        sampling=tuple(spacing_array),
    ) - ndi.distance_transform_edt(~cropped, sampling=tuple(spacing_array))
    mask_distance = signed_distance
    if reference:
        # Fixed, documented rule (no search, resampling, smoothing or decimation):
        # Lewiner marching cubes at level 0 of the native-grid signed distance.
        vertices, faces, _normals, _values = marching_cubes(
            signed_distance.astype(np.float32),
            level=0.0,
            step_size=1,
            allow_degenerate=False,
            method="lewiner",
        )
        raw_world = _voxel_to_world(vertices + origin, affine)
        surface_to_mask = _surface_to_mask_p95(raw_world, mask_distance, affine, origin)
        return (
            raw_world.astype(np.float32),
            faces.astype(np.int32),
            source_volume_mm3,
            0.0,
            signed_distance,
            origin,
            np.ones(3),
            float(np.min(spacing_array)),
            0.0,
            surface_to_mask,
        )
    # Display preview only. Blur scale in mm per axis (sigma_voxels x spacing), so a
    # 5 mm slice axis is smoothed over about 3 mm and a 0.8 mm axis over about 0.5 mm.
    signed_distance = ndi.gaussian_filter(
        signed_distance, sigma=DISPLAY_SDF_SIGMA_VOXELS, mode="nearest"
    )
    output_shape = _output_shape_for_spacing(
        signed_distance.shape,
        spacing_array,
        target_spacing_mm,
    )
    isotropic_distance, input_per_output = _resample(
        signed_distance.astype(np.float32),
        output_shape,
        order=1,
    )
    raw_world, faces, iso_level = _calibrated_surface(
        isotropic_distance,
        input_per_output,
        origin,
        affine,
        source_volume_mm3,
        target_spacing_mm,
    )

    smoothed_world = _windowed_sinc_smooth(
        raw_world,
        faces,
        float(profile["smoothing_factor"]),
    )
    volume_tolerance = float(profile["volume_tolerance_pct"])
    if source_volume_mm3 < 5_000.0:
        volume_tolerance = max(volume_tolerance, 6.0)
    maximum_displacement = max(
        float(profile["max_displacement_p95_mm"]),
        DISPLAY_MAX_DISPLACEMENT_FRACTION * float(np.max(spacing_array)),
    )
    constrained_world, smoothing_p95 = _constrain_smoothing(
        raw_world,
        smoothed_world,
        faces,
        source_volume_mm3,
        maximum_displacement,
        volume_tolerance,
    )
    final_vertices, final_faces = _decimate(
        constrained_world,
        faces,
        int(profile["max_faces"]),
    )

    before_decimation_error = abs(
        _mesh_volume_mm3(constrained_world, faces) - source_volume_mm3
    )
    after_decimation_error = abs(
        _mesh_volume_mm3(final_vertices, final_faces) - source_volume_mm3
    )
    final_topology = _topology_metrics(final_vertices, final_faces)
    if after_decimation_error > before_decimation_error + source_volume_mm3 * 0.01 or not (
        final_topology["watertight"]
    ):
        final_vertices, final_faces = constrained_world, faces

    return (
        final_vertices.astype(np.float32),
        final_faces.astype(np.int32),
        source_volume_mm3,
        smoothing_p95,
        isotropic_distance,
        origin,
        input_per_output,
        target_spacing_mm,
        iso_level,
        _surface_to_mask_p95(final_vertices, mask_distance, affine, origin),
    )


def _surface_to_mask_p95(
    world_vertices: np.ndarray,
    mask_distance: np.ndarray,
    affine: np.ndarray,
    origin_voxels: np.ndarray,
) -> float:
    """95th percentile distance (mm) from mesh vertices to the voxel-mask boundary.

    The unblurred signed distance of the source mask is sampled at each vertex. Its
    zero level lies half a voxel outside boundary voxel centres, so this measures how
    far the displayed surface departs from the mask, not just from a smoothed copy.
    """

    if world_vertices.size == 0:
        return 0.0
    homogeneous = np.c_[world_vertices.astype(float), np.ones(world_vertices.shape[0])]
    voxel = (np.linalg.inv(affine) @ homogeneous.T)[:3].T - np.asarray(origin_voxels)
    sampled = ndi.map_coordinates(mask_distance, voxel.T, order=1, mode="nearest")
    return float(np.percentile(np.abs(sampled), 95))


def _line_segments_from_skeleton(
    mask: np.ndarray,
    affine: np.ndarray,
    origin_voxels: np.ndarray,
    voxel_scale: np.ndarray,
) -> np.ndarray | None:
    skeleton = skeletonize(mask, method="lee").astype(bool)
    if not np.any(skeleton):
        return None

    segments = []
    half_neighborhood = [
        offset
        for offset in product((-1, 0, 1), repeat=3)
        if offset != (0, 0, 0) and offset > (0, 0, 0)
    ]
    shape = np.asarray(skeleton.shape)
    for offset_tuple in half_neighborhood:
        offset = np.asarray(offset_tuple)
        first_start = np.maximum(0, -offset)
        first_stop = np.minimum(shape, shape - offset)
        first_slices = tuple(
            slice(int(start), int(stop)) for start, stop in zip(first_start, first_stop)
        )
        second_slices = tuple(
            slice(int(start + delta), int(stop + delta))
            for start, stop, delta in zip(first_start, first_stop, offset)
        )
        connected = skeleton[first_slices] & skeleton[second_slices]
        first_points = np.argwhere(connected) + first_start
        if first_points.size == 0:
            continue
        second_points = first_points + offset
        first_voxels = first_points * voxel_scale + origin_voxels
        second_voxels = second_points * voxel_scale + origin_voxels
        first_world = _voxel_to_world(first_voxels, affine)
        second_world = _voxel_to_world(second_voxels, affine)
        nan_rows = np.full((first_world.shape[0], 3), np.nan)
        segments.append(
            np.stack((first_world, second_world, nan_rows), axis=1).reshape(-1, 3)
        )
    return np.concatenate(segments).astype(np.float32) if segments else None


def surface_mesh(
    segmentation: np.ndarray,
    affine: np.ndarray,
    spacing: tuple[float, float, float],
    label_value: int,
    quality: str = "Balanced",
) -> MeshData | None:
    """Create a surface for one mapped label.

    ``quality="Reference"`` gives the unsmoothed native-grid surface used for exports;
    the other profiles are display previews whose departure from the voxel mask is
    reported as ``surface_to_mask_p95_mm``.
    """

    mask = segmentation == int(label_value)
    if not np.any(mask):
        return None
    (
        vertices,
        faces,
        source_volume,
        smoothing_p95,
        isotropic_distance,
        isotropic_origin,
        input_per_output,
        target_spacing_mm,
        iso_level,
        surface_to_mask_p95,
    ) = _surface_from_binary_mask(mask, affine, spacing, quality)
    mesh: MeshData = {
        "vertices": vertices,
        "faces": faces,
        "source_voxels": int(np.count_nonzero(mask)),
        "quality": _quality_report(
            vertices,
            faces,
            source_volume,
            smoothing_p95,
            target_spacing_mm,
            (
                "Reference: Lewiner marching cubes at level 0 of the native-grid signed "
                "distance; no resampling, calibration, smoothing or decimation"
                if quality == REFERENCE_QUALITY
                else "Display preview: blurred signed distance + Lewiner + VTK windowed sinc"
            ),
        ),
    }
    mesh["quality"]["calibrated_iso_level_mm"] = iso_level
    mesh["quality"]["surface_to_mask_p95_mm"] = surface_to_mask_p95
    mesh["quality"]["role"] = "reference" if quality == REFERENCE_QUALITY else "display"
    if int(label_value) in VESSEL_LABELS:
        isotropic_mask = isotropic_distance >= 0.0
        mesh["centerline_points"] = _line_segments_from_skeleton(
            isotropic_mask,
            affine,
            isotropic_origin,
            input_per_output,
        )
    return mesh


def _normalized_response(response: np.ndarray, region: np.ndarray) -> np.ndarray:
    values = response[region]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return np.zeros_like(response, dtype=np.float32)
    scale = float(np.percentile(values, 99.8))
    if scale <= 0:
        return np.zeros_like(response, dtype=np.float32)
    return np.clip(response / scale, 0.0, 1.0).astype(np.float32)


def _filter_vessel_components(
    candidates: np.ndarray,
    distance_from_renal: np.ndarray,
    effective_spacing: np.ndarray,
    maximum_components: int = 24,
) -> np.ndarray:
    labels, component_count = ndi.label(
        candidates,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    if component_count == 0:
        return candidates

    voxel_volume_mm3 = float(np.prod(effective_spacing))
    ranked = []
    for component_id, component_slice in enumerate(
        ndi.find_objects(labels),
        start=1,
    ):
        if component_slice is None:
            continue
        component = labels[component_slice] == component_id
        voxel_count = int(np.count_nonzero(component))
        volume_mm3 = voxel_count * voxel_volume_mm3
        extent_mm = np.asarray(component.shape) * effective_spacing
        minimum_renal_distance = float(
            np.min(distance_from_renal[component_slice][component])
        )
        if (
            volume_mm3 < 10.0
            or float(np.max(extent_mm)) < 6.0
            or minimum_renal_distance > 8.0
        ):
            continue
        score = volume_mm3 * (1.0 + float(np.max(extent_mm)) / 12.0) / (
            1.0 + minimum_renal_distance
        )
        ranked.append((score, component_id, component_slice))

    filtered = np.zeros_like(candidates, dtype=bool)
    for _score, component_id, component_slice in sorted(ranked, reverse=True)[
        :maximum_components
    ]:
        filtered[component_slice] |= labels[component_slice] == component_id
    return filtered


def candidate_vessel_mesh(
    image: np.ndarray,
    segmentation: np.ndarray,
    affine: np.ndarray,
    spacing: tuple[float, float, float],
    sensitivity: str = "Balanced",
    detail: str = "Balanced",
) -> MeshData | None:
    """Estimate unlabeled tubular structures near kidneys for research visualization.

    The response fuses physical-scale Frangi and Sato filters, then applies hysteresis
    connectivity. It intentionally remains an unlabeled candidate layer: it cannot
    distinguish arteries from veins and is not a substitute for expert annotations.
    """

    renal_mask = np.isin(
        segmentation,
        (
            int(CanonicalLabel.KIDNEY),
            int(CanonicalLabel.TUMOR),
            int(CanonicalLabel.CYST),
        ),
    )
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

    target_spacing = VESSEL_TARGET_SPACING_MM.get(detail, 1.5)
    output_shape = _output_shape_for_spacing(
        image_crop.shape,
        spacing_array,
        target_spacing,
    )
    image_small, input_per_output = _resample(
        image_crop.astype(np.float32),
        output_shape,
        order=1,
    )
    renal_small, _mapping = _resample(
        renal_crop.astype(np.uint8),
        output_shape,
        order=0,
    )
    renal_small = renal_small.astype(bool)
    effective_spacing = spacing_array * input_per_output
    distance_from_renal = ndi.distance_transform_edt(
        ~renal_small,
        sampling=tuple(effective_spacing),
    )
    region = distance_from_renal <= 24.0

    clipped = np.clip(image_small, -50.0, 500.0)
    normalized = ((clipped + 50.0) / 550.0).astype(np.float32)
    physical_sigmas_mm = np.asarray((0.8, 1.2, 1.8, 2.6, 3.5))
    sigma_pixels = tuple(
        float(max(0.5, sigma / float(np.mean(effective_spacing))))
        for sigma in physical_sigmas_mm
    )
    frangi_response = frangi(
        normalized,
        sigmas=sigma_pixels,
        black_ridges=False,
    )
    sato_response = sato(
        normalized,
        sigmas=sigma_pixels,
        black_ridges=False,
    )

    enhanced_tissue = region & (image_small >= 80.0) & (image_small <= 650.0)
    fused = np.maximum(
        _normalized_response(frangi_response, enhanced_tissue),
        0.85 * _normalized_response(sato_response, enhanced_tissue),
    )
    values = fused[enhanced_tissue]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 20:
        return None

    high_quantile, low_quantile = VESSEL_THRESHOLDS.get(
        sensitivity,
        VESSEL_THRESHOLDS["Balanced"],
    )
    high_threshold = float(np.percentile(values, high_quantile))
    low_threshold = float(np.percentile(values, low_quantile))
    seeds = enhanced_tissue & (fused >= high_threshold)
    support = enhanced_tissue & (fused >= low_threshold)
    candidates = ndi.binary_propagation(seeds, mask=support)
    candidates = ndi.binary_closing(candidates, iterations=1)
    minimum_component_voxels = max(
        3,
        int(round(6.0 / float(np.prod(effective_spacing)))),
    )
    candidates = remove_small_objects(
        candidates,
        min_size=minimum_component_voxels,
    )
    candidates = _filter_vessel_components(
        candidates,
        distance_from_renal,
        effective_spacing,
    )
    if not np.any(candidates):
        return None

    padded = np.pad(candidates, 1)
    signed_distance = ndi.distance_transform_edt(
        padded,
        sampling=tuple(effective_spacing),
    ) - ndi.distance_transform_edt(
        ~padded,
        sampling=tuple(effective_spacing),
    )
    source_volume_mm3 = float(np.count_nonzero(candidates) * np.prod(effective_spacing))
    raw_world, faces, iso_level = _calibrated_surface(
        signed_distance,
        input_per_output,
        minimum.astype(float) - input_per_output,
        affine,
        source_volume_mm3,
        float(np.mean(effective_spacing)),
    )
    smoothed_world = _windowed_sinc_smooth(raw_world, faces, 0.18)
    constrained_world, smoothing_p95 = _constrain_smoothing(
        raw_world,
        smoothed_world,
        faces,
        source_volume_mm3,
        maximum_p95_mm=0.55,
        volume_tolerance_pct=8.0,
    )

    labeled_components, component_count = ndi.label(
        candidates,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    component_sizes = np.bincount(labeled_components.ravel())[1:]
    largest_component_fraction = (
        float(component_sizes.max() / component_sizes.sum())
        if component_sizes.size
        else 0.0
    )
    mesh: MeshData = {
        "vertices": constrained_world.astype(np.float32),
        "faces": faces.astype(np.int32),
        "source_voxels": int(np.count_nonzero(candidates)),
        "centerline_points": _line_segments_from_skeleton(
            candidates,
            affine,
            minimum.astype(float),
            input_per_output,
        ),
        "quality": _quality_report(
            constrained_world,
            faces,
            source_volume_mm3,
            smoothing_p95,
            float(np.mean(effective_spacing)),
            "Physical-scale Frangi/Sato fusion + connectivity hysteresis",
        ),
    }
    mesh["quality"].update(
        {
            "candidate_components": int(component_count),
            "largest_component_fraction": largest_component_fraction,
            "high_vesselness_threshold": high_threshold,
            "low_vesselness_threshold": low_threshold,
            "calibrated_iso_level_mm": iso_level,
        }
    )
    return mesh
