"""End-to-end processing of one patient case (runs inside the background worker)."""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from ..labels import CanonicalLabel
from .dicom_intake import (
    consume_inbox_source,
    convert_series,
    list_series,
    main_phase,
    read_series_index,
)
from .fusion import fuse
from .registration import register_phase
from .segmenters import (
    PipelineSettings,
    collecting_system_from_excretory,
    map_totalseg_labels,
    run_atlasnet_tumors,
    run_kidney_cysts,
    run_totalsegmentator,
    totalseg_class_map,
)
from .store import PatientCase


class PipelineError(RuntimeError):
    pass


def to_ras(array: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, tuple[float, ...]]:
    """Reorder voxel axes like ``nibabel.as_closest_canonical``; returns array and spacing."""

    from nibabel import orientations as orient

    source = orient.io_orientation(affine)
    transform = orient.ornt_transform(source, orient.axcodes2ornt("RAS"))
    zooms = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))
    order = transform[:, 0].astype(int)
    spacing = tuple(float(zooms[np.flatnonzero(order == axis)[0]]) for axis in range(3))
    return np.ascontiguousarray(orient.apply_orientation(array, transform)), spacing


def from_ras(array: np.ndarray, affine: np.ndarray) -> np.ndarray:
    from nibabel import orientations as orient

    target = orient.io_orientation(affine)
    transform = orient.ornt_transform(orient.axcodes2ornt("RAS"), target)
    return np.ascontiguousarray(orient.apply_orientation(array, transform))


def segment_main_phase(
    image_path: Path,
    work_dir: Path,
    settings: PipelineSettings,
    log,
    step,
    warn,
    provenance: dict[str, Any],
) -> dict[CanonicalLabel, np.ndarray]:
    """TotalSegmentator + cyst subtask + tumour model on one NIfTI CT (no DICOM needed)."""

    masks: dict[CanonicalLabel, np.ndarray] = {}
    step("TotalSegmentator: kidneys, great vessels and nearby organs")
    totalseg_labels, totalseg_info = run_totalsegmentator(image_path, settings, log)
    provenance["totalsegmentator"] = totalseg_info
    for name, mask in map_totalseg_labels(totalseg_labels, totalseg_class_map()).items():
        masks[CanonicalLabel[name]] = mask
    del totalseg_labels
    if CanonicalLabel.KIDNEY not in masks:
        raise PipelineError("No kidney was found on the main phase.")

    step("Kidney cysts")
    try:
        cysts = run_kidney_cysts(image_path, settings, log)
        if cysts is not None:
            masks[CanonicalLabel.CYST] = masks.get(CanonicalLabel.CYST, False) | cysts
            provenance["kidney_cysts"] = "TotalSegmentator kidney_cysts subtask"
    except Exception as exc:  # noqa: BLE001 - optional step
        warn(f"Kidney cyst subtask failed: {exc}")

    step("Kidney tumours (AbdomenAtlasNet)")
    try:
        tumors = run_atlasnet_tumors(image_path, work_dir, settings, log)
        if tumors is None:
            warn(
                "No automatic tumour model installed: outline the tumour (e.g. 3D Slicer) and "
                "upload the corrected segmentation."
            )
        else:
            tumor_mask, atlas_cysts, tumor_info = tumors
            masks[CanonicalLabel.TUMOR] = tumor_mask
            if atlas_cysts.any():
                masks[CanonicalLabel.CYST] = masks.get(CanonicalLabel.CYST, False) | atlas_cysts
            provenance["tumor_model"] = tumor_info
            if not tumor_mask.any():
                warn("The tumour model found no kidney tumour on this scan.")
    except Exception as exc:  # noqa: BLE001
        warn(f"Tumour model failed: {exc}")
    return masks


def run_case(case: PatientCase, settings: PipelineSettings | None = None) -> dict[str, Any]:
    import nibabel as nib

    settings = settings or PipelineSettings()
    request = case.request()
    assignment: dict[str, str] = request.get("assignment", {})
    main = request.get("main_phase") or main_phase(assignment)
    if not main or main not in assignment:
        raise PipelineError("No main phase was selected.")

    provenance: dict[str, Any] = {
        "case_id": case.case_id,
        "started": time.time(),
        "main_phase": main,
        "phases": {},
        "steps": [],
        "warnings": [],
        "research_use_only": True,
    }

    def log(message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with open(case.log_path, "a") as handle:
            handle.write(f"[{stamp}] {message}\n")

    def step(name: str) -> None:
        log(name)
        provenance["steps"].append(name)
        case.update_status(step=name)

    def warn(message: str) -> None:
        log(f"WARNING: {message}")
        provenance["warnings"].append(message)

    # 1. DICOM -> NIfTI, then delete the raw upload.
    step("Converting DICOM series")
    case.phases_dir.mkdir(exist_ok=True)
    native_paths: dict[str, Path] = {}
    try:
        index = read_series_index(case.raw_dir)
        series = {item.series_uid: item for item in list_series(case.raw_dir)}
        for phase, handle in assignment.items():
            uid = index.get(handle, {}).get("series_uid", handle)
            if uid not in series:
                raise PipelineError(f"Series for {phase} phase not found in the upload.")
            target = (
                case.image_path if phase == main else case.phases_dir / f"{phase}_native.nii.gz"
            )
            info = convert_series(series[uid].files, target)
            native_paths[phase] = target
            provenance["phases"][phase] = {
                **info,
                "slice_thickness_mm": series[uid].slice_thickness_mm,
            }
        inbox = Path(os.environ.get("RENAL_INBOX", "~/RenalPlanner/inbox")).expanduser()
        if consume_inbox_source(case.raw_dir, inbox):
            log("Inbox study deleted after successful conversion.")
    finally:
        case.discard_raw()
        log("Raw DICOM upload deleted.")

    # 2. Segment the main phase.
    reference = nib.load(str(case.image_path))
    spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
    masks = segment_main_phase(
        case.image_path, case.path / "work", settings, log, step, warn, provenance
    )

    # 3. Register the other phases onto the main phase (renal region of interest).
    renal = masks[CanonicalLabel.KIDNEY] | masks.get(CanonicalLabel.TUMOR, False)
    from scipy import ndimage as ndi

    roi = ndi.distance_transform_edt(~renal, sampling=spacing) <= 30.0
    registered: dict[str, Path] = {}
    for phase, path in native_paths.items():
        if phase == main:
            continue
        step(f"Registering {phase} phase")
        target = case.phases_dir / f"{phase}.nii.gz"
        try:
            provenance["phases"][phase]["registration"] = register_phase(
                case.image_path, path, target, roi, reference.affine
            )
            registered[phase] = target
            path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            warn(f"{phase} phase registration failed: {exc}")

    # 4. Collecting system from the excretory phase.
    if "excretory" in registered:
        step("Collecting system from the excretory phase")
        try:
            excretory = np.asanyarray(nib.load(str(registered["excretory"])).dataobj).astype(
                np.float32
            )
            collecting = collecting_system_from_excretory(
                excretory, renal, spacing, settings.collecting_threshold_hu
            )
            masks[CanonicalLabel.COLLECTING_SYSTEM] = collecting
            provenance["collecting_system"] = {
                "method": "Excretory-phase threshold within the renal envelope",
                "threshold_hu": settings.collecting_threshold_hu,
                "voxels": int(collecting.sum()),
            }
            if not collecting.any():
                warn("No contrast-filled collecting system found on the excretory phase.")
        except Exception as exc:  # noqa: BLE001
            warn(f"Collecting system extraction failed: {exc}")

    # 5. Renal artery/vein from the arterial phase with the validated KiPA22 model.
    if "arterial" in registered:
        step("Renal artery and vein (KiPA22 model, arterial phase)")
        try:
            from ..vessel_ml import VesselModelError, predict_vessels, resolve_model_bundle

            try:
                bundle = resolve_model_bundle(None)
            except VesselModelError as exc:
                bundle = None
                warn(f"Vessel model unavailable: {exc}")
            if bundle is not None:
                arterial = np.asanyarray(nib.load(str(registered["arterial"])).dataobj).astype(
                    np.float32
                )
                # The CNN was trained on canonical RAS arrays (as the app loads them).
                arterial_ras, ras_spacing = to_ras(arterial, reference.affine)
                renal_ras, _ = to_ras(renal.astype(np.uint8), reference.affine)
                prediction = predict_vessels(arterial_ras, ras_spacing, renal_ras, bundle)
                vessels = from_ras(prediction.segmentation, reference.affine)
                masks[CanonicalLabel.RENAL_ARTERY] = vessels == int(CanonicalLabel.RENAL_ARTERY)
                masks[CanonicalLabel.RENAL_VEIN] = vessels == int(CanonicalLabel.RENAL_VEIN)
                provenance["vessels"] = prediction.provenance
        except Exception as exc:  # noqa: BLE001
            warn(f"Vessel model failed: {exc}")
    else:
        warn(
            "No arterial phase: arterial territories need an imported BA-Net prediction or a "
            "KiPA22 vessel model."
        )

    # 6. Fuse and save on the main-phase grid.
    step("Combining labels")
    labels, counts, notes = fuse(masks, spacing)
    for note in notes:
        warn(note)
    image = nib.Nifti1Image(labels, reference.affine)
    image.header.set_data_dtype(np.uint8)
    nib.save(image, str(case.segmentation_path))
    provenance["label_voxels"] = counts
    provenance["finished"] = time.time()
    case.write_provenance(provenance)
    log("Done.")
    return provenance


def run_and_record(case: PatientCase, settings: PipelineSettings | None = None) -> None:
    case.update_status(state="running", started=time.time(), error=None)
    try:
        provenance = run_case(case, settings)
        case.update_status(
            state="done",
            step="Ready",
            finished=time.time(),
            warnings=len(provenance["warnings"]),
        )
    except Exception as exc:  # noqa: BLE001 - surface every failure to the UI
        with open(case.log_path, "a") as handle:
            handle.write(traceback.format_exc())
        case.discard_raw()
        case.update_status(state="failed", error=str(exc)[:500], finished=time.time())
