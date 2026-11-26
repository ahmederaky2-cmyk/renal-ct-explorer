"""Rigid registration of additional CT phases onto the main planning phase.

Kidneys move with respiration between phases, so registration is restricted to a region of
interest around the kidneys segmented on the main phase. One rigid transform per kidney side
would be more accurate; one transform for the renal region is used here and its quality is
reported (mutual information before/after) so poor alignments are visible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class RegistrationError(RuntimeError):
    pass


def _sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover
        raise RegistrationError("SimpleITK is required for phase registration.") from exc
    return sitk


def register_phase(
    fixed_path: Path,
    moving_path: Path,
    output_path: Path,
    roi_mask: np.ndarray | None = None,
    roi_affine: np.ndarray | None = None,
) -> dict[str, Any]:
    """Register ``moving`` to ``fixed`` and resample it onto the fixed grid.

    ``roi_mask`` (bool, nibabel RAS voxel order on the fixed image) limits the metric to the
    renal region. It is passed through a NIfTI round trip so orientation stays consistent.
    """

    sitk = _sitk()
    fixed = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.05, seed=7)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-3, numberOfIterations=200, relaxationFactor=0.6
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if roi_mask is not None and roi_affine is not None and roi_mask.any():
        import tempfile

        import nibabel as nib

        with tempfile.TemporaryDirectory() as folder:
            mask_path = Path(folder) / "roi.nii.gz"
            nib.save(nib.Nifti1Image(roi_mask.astype(np.uint8), roi_affine), str(mask_path))
            mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
        mask = sitk.Resample(mask, fixed, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
        registration.SetMetricFixedMask(mask)

    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration.SetInitialTransform(initial, inPlace=False)
    metric_before = registration.MetricEvaluate(fixed, moving)
    try:
        transform = registration.Execute(fixed, moving)
    except RuntimeError as exc:
        raise RegistrationError(f"Registration failed: {exc}") from exc
    metric_after = registration.GetMetricValue()

    resampled = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, -1024.0)
    sitk.WriteImage(sitk.Cast(resampled, sitk.sitkInt16), str(output_path), useCompression=True)
    try:
        parameters = list(transform.GetParameters())
    except (AttributeError, RuntimeError):
        parameters = []
    translation = parameters[3:6] if len(parameters) >= 6 else []
    return {
        "method": "Rigid (Euler 3D), Mattes MI, renal ROI, 3-level pyramid",
        "metric_before": float(metric_before),
        "metric_after": float(metric_after),
        "improved": bool(metric_after < metric_before),
        "translation_mm": [float(value) for value in translation],
        "iterations": int(registration.GetOptimizerIteration()),
    }
