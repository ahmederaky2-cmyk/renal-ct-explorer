from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from renal_explorer.resolution import decimation_strides, demo_target_spacing


def test_full_resolution_unless_online_demo_without_patient_mode(monkeypatch) -> None:
    monkeypatch.delenv("RENAL_DEMO_SPACING_MM", raising=False)
    assert demo_target_spacing("Online demo", patient_mode=False) == 1.5
    # Patient mode always means full resolution, whatever source is selected.
    assert demo_target_spacing("Online demo", patient_mode=True) is None
    assert demo_target_spacing("Patient cases", patient_mode=True) is None
    assert demo_target_spacing("Server datasets", patient_mode=False) is None
    monkeypatch.setenv("RENAL_DEMO_SPACING_MM", "0")
    assert demo_target_spacing("Online demo", patient_mode=False) is None
    monkeypatch.setenv("RENAL_DEMO_SPACING_MM", "not-a-number")
    assert demo_target_spacing("Online demo", patient_mode=False) == 1.5


def test_decimation_strides() -> None:
    assert decimation_strides((0.78, 0.78, 0.5), 1.5) == (2, 2, 3)
    assert decimation_strides((1.0, 1.0, 5.0), 1.5) == (1, 1, 1)
    assert decimation_strides((0.7, 0.6, 3.0), 1.5) == (2, 2, 1)
    assert decimation_strides((0.5, 0.5, 0.5), None) == (1, 1, 1)


def test_reduced_load_matches_full_geometry_and_prediction_reduction(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    if not hasattr(nib, "orientations"):
        pytest.skip("needs real nibabel")
    from renal_explorer.datasets import get_adapter
    from renal_explorer.models import CaseFiles
    from renal_explorer.resolution import reduce_canonical_like

    # A deliberately non-canonical scan: axes swapped and flipped (LPS-like, slices first).
    shape = (40, 64, 60)
    affine = np.array(
        [[0.0, -0.78, 0.0, 120.0], [0.0, 0.0, -0.78, 90.0], [0.5, 0.0, 0.0, -40.0], [0, 0, 0, 1]]
    )
    rng = np.random.default_rng(1)
    image = rng.normal(0, 100, shape).astype(np.int16)
    labels = np.zeros(shape, np.uint8)
    labels[10:30, 20:50, 15:45] = 1  # KiTS kidney
    labels[12:20, 25:35, 20:30] = 2  # KiTS tumour
    case_dir = tmp_path / "case_00001"
    case_dir.mkdir()
    nib.save(nib.Nifti1Image(image, affine), str(case_dir / "imaging.nii.gz"))
    nib.save(nib.Nifti1Image(labels, affine), str(case_dir / "segmentation.nii.gz"))
    files = CaseFiles("case_00001", case_dir / "imaging.nii.gz", case_dir / "segmentation.nii.gz")
    adapter = get_adapter("kits23")

    full = adapter.load_case(files)
    reduced = adapter.load_case(files, 1.5)
    assert full.metadata["resolution"]["reduced"] is False
    resolution = reduced.metadata["resolution"]
    assert resolution["reduced"] is True
    assert tuple(resolution["full_shape"]) == full.image.shape
    assert np.allclose(resolution["full_affine"], full.affine)
    assert all(value >= 1.4 for value in reduced.spacing)

    # A prediction verified on the full grid, reduced like the CT, lands on the same voxels
    # as the CT's own reduced labels.
    reduced_labels = reduce_canonical_like(full.segmentation, resolution)
    assert np.array_equal(reduced_labels, reduced.segmentation)
    assert np.array_equal(
        reduce_canonical_like(full.image, resolution).astype(np.float32), reduced.image
    )
