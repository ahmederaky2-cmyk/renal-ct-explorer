"""Automatic segmentation backends for patient cases.

Every backend is optional and imported lazily:

* TotalSegmentator: kidneys, aorta, IVC, adrenals and context organs. Uses the native Apple
  Silicon port (``totalsegmentator-mlx``) when installed, otherwise the PyTorch package
  (``TotalSegmentator``) on MPS/CUDA/CPU.
* TotalSegmentator ``kidney_cysts`` subtask (PyTorch package only).
* AbdomenAtlasNet-Tumors (nnU-Net v2, CC BY-NC 4.0): kidney tumours. Label IDs are read from
  the model's ``dataset.json`` because the release uses region-based labels.
* Collecting system: contrast-filled urine on a registered excretory phase (no model).
* Renal artery/vein: the repository's validated KiPA22 ONNX model on the arterial phase.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from ..labels import CanonicalLabel

TOTALSEG_ROI: dict[str, CanonicalLabel] = {
    "kidney_left": CanonicalLabel.KIDNEY,
    "kidney_right": CanonicalLabel.KIDNEY,
    "kidney_cyst_left": CanonicalLabel.CYST,
    "kidney_cyst_right": CanonicalLabel.CYST,
    "aorta": CanonicalLabel.AORTA,
    "inferior_vena_cava": CanonicalLabel.IVC,
    "adrenal_gland_left": CanonicalLabel.ADRENAL,
    "adrenal_gland_right": CanonicalLabel.ADRENAL,
    "liver": CanonicalLabel.LIVER,
    "spleen": CanonicalLabel.SPLEEN,
    "pancreas": CanonicalLabel.PANCREAS,
    "duodenum": CanonicalLabel.DUODENUM,
    "colon": CanonicalLabel.COLON,
    "iliopsoas_left": CanonicalLabel.PSOAS,
    "iliopsoas_right": CanonicalLabel.PSOAS,
}

TUMOR_NAME = re.compile(r"kidney|renal", re.IGNORECASE)
LESION_NAME = re.compile(r"lesion|tumou?r|cancer|mass|neoplasm", re.IGNORECASE)
CYST_NAME = re.compile(r"cyst", re.IGNORECASE)
CHECKPOINT_NAMES = ("checkpoint_final.pth", "checkpoint_best.pth")


class SegmentationError(RuntimeError):
    pass


@dataclass
class PipelineSettings:
    totalseg_backend: str = os.environ.get("RENAL_TOTALSEG_BACKEND", "auto")
    totalseg_fast: bool = os.environ.get("RENAL_TOTALSEG_FAST", "0") == "1"
    totalseg_mlx_models: Path = Path(
        os.environ.get("RENAL_TOTALSEG_MLX_MODELS", "~/RenalPlanner/models/totalsegmentator-mlx")
    ).expanduser()
    device: str = os.environ.get("RENAL_DEVICE", "auto")
    run_kidney_cysts: bool = os.environ.get("RENAL_RUN_KIDNEY_CYSTS", "1") == "1"
    atlasnet_tumor_dir: Path = Path(
        os.environ.get("RENAL_ATLASNET_TUMOR_DIR", "~/RenalPlanner/models/AbdomenAtlasNetTumors")
    ).expanduser()
    atlasnet_folds: str = os.environ.get("RENAL_ATLASNET_FOLDS", "all")
    collecting_threshold_hu: float = float(os.environ.get("RENAL_CS_THRESHOLD_HU", "400"))
    extra: dict[str, Any] = field(default_factory=dict)


def torch_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "gpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------------------
# TotalSegmentator
# --------------------------------------------------------------------------------------


def totalseg_class_map() -> dict[int, str]:
    """Label id -> name for the TotalSegmentator ``total`` task."""

    try:
        from totalsegmentator.map_to_binary import class_map

        return {int(key): str(value) for key, value in class_map["total"].items()}
    except ImportError:
        pass
    try:  # the MLX port stores the upstream table; look for it without hard-coding a name
        import totalsegmentator_mlx as mlx_package

        for value in vars(mlx_package).values():
            if isinstance(value, dict) and "kidney_left" in value.values():
                return {int(key): str(name) for key, name in value.items()}
    except ImportError:
        pass
    raise SegmentationError(
        "Cannot resolve TotalSegmentator label names; install the TotalSegmentator package."
    )


def map_totalseg_labels(labels: np.ndarray, class_map: dict[int, str]) -> dict[str, np.ndarray]:
    """Split a TotalSegmentator multilabel map into canonical-label masks (by name)."""

    by_name = {name: label_id for label_id, name in class_map.items()}
    masks: dict[str, np.ndarray] = {}
    for name, canonical in TOTALSEG_ROI.items():
        label_id = by_name.get(name)
        if label_id is None:
            continue
        mask = labels == label_id
        if mask.any():
            key = canonical.name
            masks[key] = masks[key] | mask if key in masks else mask
    return masks


def _available(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


def run_totalsegmentator(
    ct_path: Path, settings: PipelineSettings, log
) -> tuple[np.ndarray, dict[str, Any]]:
    """Returns the multilabel array on the input grid (input voxel order) and provenance."""

    import nibabel as nib

    roi = list(TOTALSEG_ROI)
    backend = settings.totalseg_backend
    if backend == "auto":
        backend = "mlx" if _available("totalsegmentator_mlx") else "pytorch"

    class_map = totalseg_class_map()
    valid = set(class_map.values())
    unknown = [name for name in roi if name not in valid]
    roi = [name for name in roi if name in valid]
    if unknown:
        log(f"TotalSegmentator does not know {unknown}; skipped.")

    automatic = settings.totalseg_backend == "auto"
    if backend == "mlx":
        try:
            from totalsegmentator_mlx import segment

            log("TotalSegmentator (MLX, Apple Silicon) …")
            image = segment(
                str(ct_path),
                model_dir=str(settings.totalseg_mlx_models),
                mode="fast" if settings.totalseg_fast else "standard",
                roi_subset=roi,
            )
            version = _package_version("totalsegmentator-mlx")
        except Exception as exc:  # noqa: BLE001
            if not automatic:
                raise
            log(f"MLX backend failed ({exc}); falling back to PyTorch.")
            backend = "pytorch"
    if backend == "pytorch":
        if not _available("totalsegmentator"):
            raise SegmentationError(
                "Neither totalsegmentator-mlx nor TotalSegmentator is installed."
            )
        from totalsegmentator.python_api import totalsegmentator

        device = torch_device(settings.device)
        log(f"TotalSegmentator (PyTorch, {device}) …")
        image = totalsegmentator(
            str(ct_path),
            None,
            ml=True,
            fast=settings.totalseg_fast,
            roi_subset=roi,
            device=device,
            quiet=True,
        )
        version = _package_version("TotalSegmentator")
    elif backend != "mlx":
        raise SegmentationError(f"Unknown TotalSegmentator backend {backend!r}.")

    reference = nib.load(str(ct_path))
    labels = np.asanyarray(image.dataobj).astype(np.uint16)
    if labels.shape != reference.shape:
        raise SegmentationError("TotalSegmentator output grid differs from the input CT.")
    return labels, {
        "backend": backend,
        "version": version,
        "fast": settings.totalseg_fast,
        "roi_subset": roi,
        "licence": "Apache-2.0 (code and open-task weights)",
    }


def run_kidney_cysts(ct_path: Path, settings: PipelineSettings, log) -> np.ndarray | None:
    if not settings.run_kidney_cysts or not _available("totalsegmentator"):
        return None
    from totalsegmentator.map_to_binary import class_map
    from totalsegmentator.python_api import totalsegmentator

    if "kidney_cysts" not in class_map:
        return None
    log("TotalSegmentator kidney_cysts subtask …")
    image = totalsegmentator(
        str(ct_path), None, ml=True, task="kidney_cysts", device=torch_device(settings.device),
        quiet=True,
    )
    labels = np.asanyarray(image.dataobj)
    cyst_ids = [
        int(key) for key, name in class_map["kidney_cysts"].items() if CYST_NAME.search(str(name))
    ]
    return np.isin(labels, cyst_ids)


# --------------------------------------------------------------------------------------
# AbdomenAtlasNet (nnU-Net v2)
# --------------------------------------------------------------------------------------


def discover_nnunet_model(root: Path) -> tuple[Path, list[str]] | None:
    """Find an nnU-Net v2 trained-model folder (dataset.json + plans.json + fold_*)."""

    if not root.exists():
        return None
    candidates = []
    for dataset_json in root.rglob("dataset.json"):
        folder = dataset_json.parent
        if not (folder / "plans.json").is_file():
            continue
        folds = sorted(
            path.name.removeprefix("fold_")
            for path in folder.glob("fold_*")
            if any((path / name).is_file() for name in CHECKPOINT_NAMES)
        )
        if folds:
            candidates.append((folder, folds))
    if not candidates:
        return None
    # Prefer a 3d_fullres configuration when several are shipped.
    candidates.sort(key=lambda item: ("3d_fullres" not in item[0].name, str(item[0])))
    return candidates[0]


def nnunet_label_ids(dataset: dict[str, Any], organ: re.Pattern, kind: re.Pattern) -> set[int]:
    """IDs of labels whose name matches both patterns. Handles region-based (list) labels."""

    ids: set[int] = set()
    for name, value in dataset.get("labels", {}).items():
        if not (organ.search(name) and kind.search(name)):
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        ids.update(int(item) for item in values if int(item) != 0)
    return ids


def run_atlasnet_tumors(
    ct_path: Path, work_dir: Path, settings: PipelineSettings, log
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Returns (kidney tumour mask, kidney cyst mask, provenance) or None if not installed."""

    found = discover_nnunet_model(settings.atlasnet_tumor_dir)
    if found is None:
        log(f"No AbdomenAtlasNet tumour model found under {settings.atlasnet_tumor_dir}.")
        return None
    if not _available("nnunetv2"):
        log("nnunetv2 is not installed; tumour model skipped.")
        return None
    import nibabel as nib
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    model_folder, folds = found
    dataset = json.loads((model_folder / "dataset.json").read_text())
    tumor_ids = nnunet_label_ids(dataset, TUMOR_NAME, LESION_NAME)
    cyst_ids = nnunet_label_ids(dataset, TUMOR_NAME, CYST_NAME)
    if not tumor_ids:
        raise SegmentationError(
            f"No kidney tumour/lesion label in {model_folder / 'dataset.json'}: "
            f"{list(dataset.get('labels', {}))}"
        )
    use_folds = folds if settings.atlasnet_folds == "all" else settings.atlasnet_folds.split(",")
    use_folds = [int(fold) if fold.isdigit() else fold for fold in use_folds]
    device_name = torch_device(settings.device)
    device = torch.device({"gpu": "cuda"}.get(device_name, device_name))
    log(f"AbdomenAtlasNet-Tumors on {device_name}, folds {use_folds} …")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=device_name == "gpu",
        perform_everything_on_device=device_name != "cpu",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    checkpoint = (
        "checkpoint_final.pth"
        if (model_folder / f"fold_{use_folds[0]}" / "checkpoint_final.pth").is_file()
        else "checkpoint_best.pth"
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder), use_folds=tuple(use_folds), checkpoint_name=checkpoint
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    output_base = work_dir / "atlasnet_tumors"
    predictor.predict_from_files(
        [[str(ct_path)]],
        [str(output_base)],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )
    output = next(work_dir.glob("atlasnet_tumors.nii*"), None)
    if output is None:
        raise SegmentationError("nnU-Net produced no output file.")
    labels = np.asanyarray(nib.load(str(output)).dataobj)
    output.unlink(missing_ok=True)
    return (
        np.isin(labels, sorted(tumor_ids)),
        np.isin(labels, sorted(cyst_ids)) if cyst_ids else np.zeros(labels.shape, bool),
        {
            "model": "AbdomenAtlasNet-Tumors (nnU-Net v2)",
            "folder": model_folder.name,
            "folds": use_folds,
            "checkpoint": checkpoint,
            "tumor_label_ids": sorted(tumor_ids),
            "licence": "CC BY-NC 4.0 (non-commercial research)",
            "caveat": "Trained on AbdomenAtlas 3.0; not validated on this institution's scans.",
        },
    )


# --------------------------------------------------------------------------------------
# Collecting system from the excretory phase
# --------------------------------------------------------------------------------------


def collecting_system_from_excretory(
    excretory_hu: np.ndarray,
    renal_mask: np.ndarray,
    spacing,
    threshold_hu: float = 400.0,
    search_mm: float = 10.0,
    min_inside_fraction: float = 0.3,
    min_volume_ml: float = 0.1,
) -> np.ndarray:
    """Contrast-filled urine within/near the kidneys.

    Bright voxels within ``search_mm`` of the kidney are grouped into components; a component
    is kept when at least ``min_inside_fraction`` of it lies inside the closed renal envelope
    (sinus and calyces), which rejects ribs, spine and bowel contrast that merely touch the
    search region.
    """

    from ..planning import _bounds, _closing_mm, _edt

    spacing = tuple(float(value) for value in spacing)
    result = np.zeros(renal_mask.shape, dtype=bool)
    bounds = _bounds(renal_mask)
    if bounds is None:
        return result
    pad = [int(np.ceil((search_mm + 20.0) / value)) for value in spacing]
    crop = tuple(
        slice(max(0, low - pad[axis]), min(renal_mask.shape[axis], high + pad[axis]))
        for axis, (low, high) in enumerate(bounds)
    )
    renal = renal_mask[crop]
    image = excretory_hu[crop]
    near = _edt(~renal, spacing) <= search_mm
    envelope = ndi.binary_fill_holes(_closing_mm(renal, spacing, 15.0))
    bright = (image >= threshold_hu) & near
    components, count = ndi.label(bright, structure=ndi.generate_binary_structure(3, 2))
    if count == 0:
        return result
    index = np.arange(1, count + 1)
    sizes = ndi.sum_labels(np.ones_like(components), components, index)
    inside = ndi.sum_labels(envelope.astype(np.int32), components, index)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    inside_fraction = inside / np.maximum(sizes, 1)
    keep = index[(inside_fraction >= min_inside_fraction) & (sizes * voxel_ml >= min_volume_ml)]
    result[crop] = np.isin(components, keep)
    return result


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return "unknown"
