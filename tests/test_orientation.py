"""Display orientation: radiological convention with burned-in markers."""

from __future__ import annotations

import numpy as np

from renal_explorer.labels import LABEL_COLORS, CanonicalLabel
from renal_explorer.rendering import CAMERA_PRESETS, render_view


def _volume_with_right_sided_mass():
    # Canonical RAS+: x grows towards the patient's right, y anterior, z superior.
    shape = (40, 30, 20)
    image = np.zeros(shape, dtype=np.float32)
    segmentation = np.zeros(shape, dtype=np.uint8)
    segmentation[30:36, 4:10, 14:18] = int(CanonicalLabel.TUMOR)  # right, posterior, superior
    return image, segmentation


def _red_columns_rows(view):
    red = np.all(view == LABEL_COLORS[CanonicalLabel.TUMOR], axis=2)
    rows, columns = np.nonzero(red)
    return rows, columns, view.shape


def _render(orientation, index, image, segmentation, markers=False):
    return render_view(
        image=image,
        segmentation=segmentation,
        spacing=(1.0, 1.0, 1.0),
        orientation=orientation,
        index=index,
        level=40,
        width=400,
        visible_labels=[CanonicalLabel.TUMOR],
        overlay_mode="Fill",
        opacity=1.0,
        markers=markers,
    )


def test_patient_right_is_on_the_image_left_in_axial_and_coronal() -> None:
    image, segmentation = _volume_with_right_sided_mass()
    rows, columns, shape = _red_columns_rows(_render("Axial", 15, image, segmentation))
    assert columns.max() < shape[1] / 2  # patient right -> image left
    assert rows.min() > shape[0] / 2  # posterior -> image bottom
    rows, columns, shape = _red_columns_rows(_render("Coronal", 6, image, segmentation))
    assert columns.max() < shape[1] / 2
    assert rows.max() < shape[0] / 2  # superior -> image top


def test_sagittal_shows_anterior_on_the_left() -> None:
    image, segmentation = _volume_with_right_sided_mass()
    rows, columns, shape = _red_columns_rows(_render("Sagittal", 32, image, segmentation))
    assert columns.min() > shape[1] / 2  # posterior -> image right
    assert rows.max() < shape[0] / 2


def test_crosshair_passes_through_the_selected_point() -> None:
    image, segmentation = _volume_with_right_sided_mass()
    view = render_view(
        image=image,
        segmentation=None,
        spacing=(1.0, 1.0, 1.0),
        orientation="Axial",
        index=15,
        level=40,
        width=400,
        visible_labels=[],
        overlay_mode="Off",
        opacity=1.0,
        linked_indices={"Sagittal": 33, "Coronal": 6, "Axial": 15},
        markers=False,
    )
    white = np.all(view == (245, 245, 245), axis=2)
    column = int(np.argmax(white.sum(axis=0)))
    row = int(np.argmax(white.sum(axis=1)))
    assert abs(column - (40 - 1 - 33)) <= 1
    assert abs(row - (30 - 1 - 6)) <= 1


def test_orientation_markers_are_drawn() -> None:
    image, segmentation = _volume_with_right_sided_mass()
    plain = _render("Axial", 15, np.zeros((160, 160, 20), np.float32), None)
    marked = _render("Axial", 15, np.zeros((160, 160, 20), np.float32), None, markers=True)
    assert plain.shape == marked.shape
    assert np.any(np.all(marked == (255, 214, 64), axis=2))


def test_anterior_camera_puts_patient_right_on_screen_left() -> None:
    camera = CAMERA_PRESETS["Anterior"]
    eye = np.array([camera["eye"][axis] for axis in "xyz"])
    up = np.array([camera["up"][axis] for axis in "xyz"])
    forward = -eye / np.linalg.norm(eye)
    screen_right = np.cross(forward, up)
    assert screen_right[0] < -0.9  # screen right = patient left (-x)
