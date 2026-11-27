"""Independent orientation gate: DICOM fiducial -> intake -> loader -> displayed pixels.

The expected screen position is derived only from DICOM geometry (ImageOrientationPatient,
ImagePositionPatient, PixelSpacing) and the radiological display convention, never from
the app's own coordinate helpers. A bright box sits in the patient's right-anterior-
superior octant; on screen it must appear:

* axial:    image left (patient right) and top (anterior)
* coronal:  image left (patient right) and top (superior)
* sagittal: image left (anterior) and top (superior)

Each series variant exercises a different geometry path: standard axial, reversed file
order, flipped row/column directions, an oblique (12 degree) acquisition, and a
coronally acquired stack. A mutation test proves the check fails on a mirrored renderer.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
pytest.importorskip("SimpleITK")

from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid  # noqa: E402

from renal_explorer import rendering  # noqa: E402
from renal_explorer.datasets import KiTS23Adapter  # noqa: E402
from renal_explorer.models import CaseFiles  # noqa: E402
from renal_explorer.patient.dicom_intake import convert_series, list_series  # noqa: E402

# Fiducial box in DICOM patient coordinates (LPS: +x left, +y posterior, +z superior).
# x < 0 -> patient right, y < 0 -> anterior, z > 0 -> superior.
BOX_LPS = ((-80.0, -40.0), (-80.0, -40.0), (20.0, 60.0))
FIELD_MM = 100.0  # the volume covers -100..+100 mm in-plane
SLICES = 41
SLICE_STEP_MM = 4.0
PIXEL_MM = 2.0

EXPECTED = {
    # orientation: (expected horizontal half, expected vertical half)
    "Axial": ("left", "top"),
    "Coronal": ("left", "top"),
    "Sagittal": ("left", "top"),
}


def _inside_box(points: np.ndarray) -> np.ndarray:
    inside = np.ones(points.shape[:-1], dtype=bool)
    for axis, (low, high) in enumerate(BOX_LPS):
        inside &= (points[..., axis] >= low) & (points[..., axis] <= high)
    return inside


def _rotation_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0, 0, 1],
        ]
    )


def _write_series(folder: Path, row_dir, col_dir, *, reverse_files: bool = False) -> None:
    """Write a CT series whose voxel values are sampled from the patient-space fiducial."""

    row_dir = np.asarray(row_dir, float)
    col_dir = np.asarray(col_dir, float)
    normal = np.cross(row_dir, col_dir)
    size = int(2 * FIELD_MM / PIXEL_MM)
    # The slice stack is centred on the origin along the normal.
    offsets = (np.arange(SLICES) - (SLICES - 1) / 2) * SLICE_STEP_MM
    corner = -FIELD_MM * row_dir - FIELD_MM * col_dir
    study, series = generate_uid(), generate_uid()
    columns, rows = np.meshgrid(np.arange(size), np.arange(size))
    order = list(range(SLICES))
    if reverse_files:
        order = order[::-1]
    folder.mkdir(parents=True, exist_ok=True)
    for file_number, k in enumerate(order):
        origin = corner + offsets[k] * normal
        # DICOM: first IOP triplet is the direction of increasing column index,
        # second triplet the direction of increasing row index.
        points = (
            origin + columns[..., None] * PIXEL_MM * row_dir + rows[..., None] * PIXEL_MM * col_dir
        )
        pixels = np.where(_inside_box(points), 1000, 0).astype(np.int16)

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset = FileDataset(
            str(folder / f"im{file_number:03d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128
        )
        dataset.SOPClassUID = CTImageStorage
        dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        dataset.StudyInstanceUID = study
        dataset.SeriesInstanceUID = series
        dataset.Modality = "CT"
        dataset.SeriesDescription = "orientation fixture"
        dataset.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        dataset.PatientName = "Fixture^Orientation"
        dataset.PatientID = "FIXTURE"
        dataset.InstanceNumber = k + 1
        dataset.ImagePositionPatient = [float(value) for value in origin]
        dataset.ImageOrientationPatient = [float(value) for value in (*row_dir, *col_dir)]
        dataset.PixelSpacing = [PIXEL_MM, PIXEL_MM]
        dataset.SliceThickness = SLICE_STEP_MM
        dataset.Rows = size
        dataset.Columns = size
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1
        dataset.RescaleIntercept = 0
        dataset.RescaleSlope = 1
        dataset.PixelData = pixels.tobytes()
        dataset.save_as(str(folder / f"im{file_number:03d}.dcm"), enforce_file_format=True)


VARIANTS = {
    "axial": ((1, 0, 0), (0, 1, 0), False),
    "axial_reversed_files": ((1, 0, 0), (0, 1, 0), True),
    "axial_flipped_rows_columns": ((-1, 0, 0), (0, -1, 0), False),
    "axial_oblique_12deg": (
        tuple(_rotation_z(12) @ np.array([1.0, 0, 0])),
        tuple(_rotation_z(12) @ np.array([0, 1.0, 0])),
        False,
    ),
    "coronal_acquisition": ((1, 0, 0), (0, 0, -1), False),
}


def _load_through_app(tmp_path: Path, variant: str):
    row_dir, col_dir, reverse = VARIANTS[variant]
    dicom_dir = tmp_path / "dicom"
    _write_series(dicom_dir, row_dir, col_dir, reverse_files=reverse)
    series = list_series(dicom_dir)
    assert len(series) == 1
    nifti = tmp_path / "imaging.nii.gz"
    convert_series(series[0].files, nifti)
    return KiTS23Adapter().load_case(CaseFiles("fixture", nifti, None))


def _bright_half(view: np.ndarray) -> tuple[str, str]:
    bright = view.mean(axis=2) > 128
    assert bright.any(), "fiducial not visible on this slice"
    rows, columns = np.nonzero(bright)
    horizontal = "left" if columns.mean() < view.shape[1] / 2 else "right"
    vertical = "top" if rows.mean() < view.shape[0] / 2 else "bottom"
    return horizontal, vertical


def _displayed_positions(case) -> dict[str, tuple[str, str]]:
    bright = case.image > 500
    centre = np.round(np.argwhere(bright).mean(axis=0)).astype(int)  # a slice through the box
    positions = {}
    for orientation, axis in rendering.VIEW_AXES.items():
        view = rendering.render_view(
            image=case.image,
            segmentation=None,
            spacing=case.spacing,
            orientation=orientation,
            index=int(centre[axis]),
            level=500,
            width=1000,
            visible_labels=[],
            overlay_mode="Off",
            opacity=0.0,
            markers=False,
        )
        positions[orientation] = _bright_half(view)
    return positions


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_patient_right_anterior_superior_fiducial_is_displayed_correctly(tmp_path, variant):
    case = _load_through_app(tmp_path, variant)
    assert _displayed_positions(case) == EXPECTED


def test_gate_fails_when_the_renderer_is_mirrored(tmp_path, monkeypatch):
    case = _load_through_app(tmp_path, "axial")
    monkeypatch.setattr(
        rendering,
        "_extract_plane",
        lambda volume, orientation, index: np.rot90(
            np.take(volume, index, axis=rendering.VIEW_AXES[orientation])
        ),
    )
    positions = _displayed_positions(case)
    assert positions["Axial"] != EXPECTED["Axial"]
    assert positions["Coronal"] != EXPECTED["Coronal"]


def test_gate_fails_when_rows_are_flipped(tmp_path, monkeypatch):
    case = _load_through_app(tmp_path, "axial")
    original = rendering._extract_plane
    monkeypatch.setattr(
        rendering,
        "_extract_plane",
        lambda volume, orientation, index: np.flipud(original(volume, orientation, index)),
    )
    positions = _displayed_positions(case)
    assert all(positions[name][1] == "bottom" for name in positions)
