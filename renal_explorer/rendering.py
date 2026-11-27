from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import ndimage as ndi

from .labels import LABEL_COLORS, LABEL_NAMES, CanonicalLabel

VIEW_AXES = {"Axial": 2, "Coronal": 1, "Sagittal": 0}

# Radiological convention (as on every PACS): axial and coronal views show the patient's
# right on the image left; sagittal views show anterior on the image left. The volume is
# canonical RAS+ (x grows to the patient's right, y anterior, z superior).
# Labels for the (left, right, top, bottom) image edges:
ORIENTATION_MARKERS = {
    "Axial": ("R", "L", "A", "P"),
    "Coronal": ("R", "L", "S", "I"),
    "Sagittal": ("A", "P", "S", "I"),
}


# 3D camera presets in RAS+ world space. "Anterior" faces the patient, so the patient's
# right is on the screen's left, matching the radiological slice layout.
CAMERA_PRESETS = {
    "Anterior": {"eye": {"x": 0.0, "y": 2.1, "z": 0.25}, "up": {"x": 0, "y": 0, "z": 1}},
    "Posterior": {"eye": {"x": 0.0, "y": -2.1, "z": 0.25}, "up": {"x": 0, "y": 0, "z": 1}},
    "From patient's right": {
        "eye": {"x": 2.1, "y": 0.0, "z": 0.25},
        "up": {"x": 0, "y": 0, "z": 1},
    },
    "From patient's left": {
        "eye": {"x": -2.1, "y": 0.0, "z": 0.25},
        "up": {"x": 0, "y": 0, "z": 1},
    },
    "Superior": {"eye": {"x": 0.0, "y": 0.0, "z": 2.1}, "up": {"x": 0, "y": 1, "z": 0}},
    "Inferior": {"eye": {"x": 0.0, "y": 0.0, "z": -2.1}, "up": {"x": 0, "y": 1, "z": 0}},
}


def apply_ct_window(image: np.ndarray, level: float, width: float) -> np.ndarray:
    width = max(float(width), 1.0)
    lower = float(level) - width / 2.0
    normalized = np.clip((image - lower) / width, 0.0, 1.0)
    return np.round(normalized * 255.0).astype(np.uint8)


def _extract_plane(volume: np.ndarray, orientation: str, index: int) -> np.ndarray:
    # rot90 puts superior/anterior at the top with increasing x (or y) to the right;
    # fliplr then gives the radiological left-right (and anterior-left sagittal) layout.
    return np.fliplr(np.rot90(np.take(volume, index, axis=VIEW_AXES[orientation])))


def _display_spacing(
    spacing: tuple[float, float, float], orientation: str
) -> tuple[float, float]:
    # Row/column spacing after the 90-degree rotation performed in _extract_plane.
    if orientation == "Axial":
        return spacing[1], spacing[0]
    if orientation == "Coronal":
        return spacing[2], spacing[0]
    return spacing[2], spacing[1]


def _resample_for_display(
    plane: np.ndarray,
    segmentation: np.ndarray | None,
    pixel_spacing: tuple[float, float],
    max_dimension: int = 900,
) -> tuple[np.ndarray, np.ndarray | None, tuple[float, float]]:
    minimum_spacing = min(pixel_spacing)
    factors = np.asarray(pixel_spacing, dtype=float) / minimum_spacing
    proposed = np.asarray(plane.shape, dtype=float) * factors
    if proposed.max() > max_dimension:
        factors *= max_dimension / proposed.max()

    if np.allclose(factors, 1.0, atol=0.01):
        return plane, segmentation, (1.0, 1.0)

    resized_plane = ndi.zoom(plane, zoom=tuple(factors), order=1, prefilter=False)
    resized_segmentation = None
    if segmentation is not None:
        resized_segmentation = ndi.zoom(
            segmentation, zoom=tuple(factors), order=0, prefilter=False
        )
    return resized_plane, resized_segmentation, (float(factors[0]), float(factors[1]))


def _crosshair_position(
    orientation: str,
    indices: dict[str, int],
    shape: tuple[int, int, int],
) -> tuple[int, int]:
    x_index = indices["Sagittal"]
    y_index = indices["Coronal"]
    z_index = indices["Axial"]
    if orientation == "Axial":
        return shape[1] - 1 - y_index, shape[0] - 1 - x_index
    if orientation == "Coronal":
        return shape[2] - 1 - z_index, shape[0] - 1 - x_index
    return shape[2] - 1 - z_index, shape[1] - 1 - y_index


def _draw_crosshair(image: np.ndarray, row: int, column: int) -> None:
    height, width = image.shape[:2]
    row = int(np.clip(row, 0, height - 1))
    column = int(np.clip(column, 0, width - 1))
    color = np.asarray((245, 245, 245), dtype=np.uint8)
    image[max(0, row - 1) : min(height, row + 1), :, :] = color
    image[:, max(0, column - 1) : min(width, column + 1), :] = color


def render_view(
    image: np.ndarray,
    segmentation: np.ndarray | None,
    spacing: tuple[float, float, float],
    orientation: str,
    index: int,
    level: float,
    width: float,
    visible_labels: Iterable[CanonicalLabel],
    overlay_mode: str,
    opacity: float,
    linked_indices: dict[str, int] | None = None,
    supplemental_segmentation: np.ndarray | None = None,
    markers: bool = True,
) -> np.ndarray:
    image_plane = _extract_plane(image, orientation, index)
    segmentation_plane = (
        _extract_plane(segmentation, orientation, index)
        if segmentation is not None
        else None
    )
    image_plane, segmentation_plane, scale = _resample_for_display(
        image_plane,
        segmentation_plane,
        _display_spacing(spacing, orientation),
    )
    gray = apply_ct_window(image_plane, level, width)
    rgb = np.repeat(gray[..., np.newaxis], 3, axis=2)

    supplemental_plane = None
    if supplemental_segmentation is not None:
        supplemental_plane = _extract_plane(
            supplemental_segmentation,
            orientation,
            index,
        )
        if not np.allclose(scale, 1.0, atol=0.01):
            supplemental_plane = ndi.zoom(
                supplemental_plane,
                zoom=scale,
                order=0,
                prefilter=False,
            )

    if overlay_mode != "Off":
        for overlay in (segmentation_plane, supplemental_plane):
            if overlay is None:
                continue
            for label in visible_labels:
                mask = overlay == int(label)
                if not np.any(mask):
                    continue
                if overlay_mode == "Outline":
                    mask = mask & ~ndi.binary_erosion(mask)
                    rgb[mask] = np.asarray(LABEL_COLORS[label], dtype=np.uint8)
                else:
                    color = np.asarray(LABEL_COLORS[label], dtype=float)
                    blended = rgb[mask].astype(float) * (1.0 - opacity) + color * opacity
                    rgb[mask] = np.clip(blended, 0, 255).astype(np.uint8)

    if linked_indices is not None:
        row, column = _crosshair_position(orientation, linked_indices, image.shape)
        _draw_crosshair(rgb, round(row * scale[0]), round(column * scale[1]))
    if markers:
        rgb = draw_orientation_markers(rgb, orientation)
    return rgb


def draw_orientation_markers(rgb: np.ndarray, orientation: str) -> np.ndarray:
    """Burn R/L/A/P/S/I letters into the image edges (they stay on screenshots and PDFs)."""

    from PIL import Image, ImageDraw, ImageFont

    if min(rgb.shape[:2]) < 96:  # too small for legible letters (thumbnails, tests)
        return rgb
    left, right, top, bottom = ORIENTATION_MARKERS[orientation]
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    size = max(12, int(min(image.size) * 0.06))
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    width, height = image.size
    margin = max(3, size // 4)
    style = {"fill": (255, 214, 64), "font": font, "stroke_width": 2, "stroke_fill": (0, 0, 0)}
    draw.text((margin, height / 2), left, anchor="lm", **style)
    draw.text((width - margin, height / 2), right, anchor="rm", **style)
    draw.text((width / 2, margin), top, anchor="mt", **style)
    draw.text((width / 2, height - margin), bottom, anchor="mb", **style)
    return np.asarray(image)


def legend_html(labels: Iterable[CanonicalLabel]) -> str:
    chips: list[str] = []
    for label in labels:
        red, green, blue = LABEL_COLORS[label]
        chips.append(
            "<span style='display:inline-flex;align-items:center;margin:0 12px 6px 0;'>"
            f"<span style='width:12px;height:12px;border-radius:3px;margin-right:5px;"
            f"background:rgb({red},{green},{blue});'></span>{LABEL_NAMES[label]}</span>"
        )
    return "<div style='font-size:0.82rem'>" + "".join(chips) + "</div>"
