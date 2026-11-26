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


def test_patient_and_server_adapters_refuse_reduction(tmp_path: Path) -> None:
    from renal_explorer.datasets import DatasetError, get_adapter
    from renal_explorer.models import CaseFiles

    files = CaseFiles("PN-00000000", tmp_path / "imaging.nii.gz", None)
    for key in ("patient", "kipa22"):
        with pytest.raises(DatasetError, match="full resolution"):
            get_adapter(key).load_case(files, 1.5)


def test_grid_token_separates_demo_and_full_grids() -> None:
    from renal_explorer.resolution import grid_token, reduce_canonical_like

    affine = np.diag([0.78, 0.78, 0.5, 1.0])
    full = grid_token((512, 512, 600), affine, {"strides": [1, 1, 1]})
    demo = grid_token((256, 256, 200), affine @ np.diag([2, 2, 3, 1]), {"strides": [2, 2, 3]})
    assert full != demo
    assert grid_token((512, 512, 600), affine, None) == full
    array = np.arange(6 * 6 * 6).reshape(6, 6, 6)
    assert reduce_canonical_like(array, {"strides": [2, 3, 1]}).shape == (3, 2, 6)
    assert reduce_canonical_like(array, {"strides": [1, 1, 1]}) is array


def test_differently_oriented_mask_and_mismatched_slices(tmp_path: Path) -> None:
    """Regression (third review): orientation-dependent sampling and hidden mismatches."""

    nib = pytest.importorskip("nibabel")
    if not hasattr(nib, "orientations"):
        pytest.skip("needs real nibabel")
    from renal_explorer.datasets import DatasetError, get_adapter
    from renal_explorer.models import CaseFiles

    shape = (41, 64, 60)  # odd length on the flipped axis exercises the start offset
    affine = np.array(
        [[0.0, -0.78, 0.0, 120.0], [0.0, 0.0, -0.78, 90.0], [0.5, 0.0, 0.0, -40.0], [0, 0, 0, 1]]
    )
    image = np.random.default_rng(2).normal(0, 100, shape).astype(np.int16)
    labels = np.zeros(shape, np.uint8)
    labels[10:30, 20:50, 15:45] = 1
    labels[12:20, 25:35, 20:30] = 2
    # Store the mask in canonical orientation instead of the CT's native one.
    mask_canonical = nib.as_closest_canonical(nib.Nifti1Image(labels, affine))
    case_dir = tmp_path / "case_00002"
    case_dir.mkdir()
    nib.save(nib.Nifti1Image(image, affine), str(case_dir / "imaging.nii.gz"))
    nib.save(mask_canonical, str(case_dir / "segmentation.nii.gz"))
    files = CaseFiles("case_00002", case_dir / "imaging.nii.gz", case_dir / "segmentation.nii.gz")
    adapter = get_adapter("kits23")
    full = adapter.load_case(files)
    reduced = adapter.load_case(files, 1.5)
    from renal_explorer.resolution import reduce_canonical_like

    resolution = reduced.metadata["resolution"]
    assert np.array_equal(
        reduce_canonical_like(full.segmentation, resolution), reduced.segmentation
    )

    # A 7-slice CT with an 8-slice mask must be rejected, also when both would reduce to 4.
    small = case_dir.parent / "case_00003"
    small.mkdir()
    nib.save(nib.Nifti1Image(np.zeros((7, 8, 8), np.int16), np.diag([0.5, 0.5, 0.5, 1.0])),
             str(small / "imaging.nii.gz"))
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.uint8), np.diag([0.5, 0.5, 0.5, 1.0])),
             str(small / "segmentation.nii.gz"))
    mismatched = CaseFiles("case_00003", small / "imaging.nii.gz", small / "segmentation.nii.gz")
    for target in (None, 1.5):
        with pytest.raises(DatasetError, match="differ"):
            adapter.load_case(mismatched, target)
