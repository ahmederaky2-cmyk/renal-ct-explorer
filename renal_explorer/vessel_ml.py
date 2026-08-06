from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
from scipy import ndimage as ndi

from .labels import CanonicalLabel


class VesselModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class VesselModelManifest:
    schema_version: int
    name: str
    version: str
    dataset: str
    dataset_url: str
    research_only: bool
    validated: bool
    validation_fold: int
    training_cases: int
    target_spacing_mm: tuple[float, float, float]
    patch_size: tuple[int, int, int]
    overlap: float
    intensity_clip_hu: tuple[float, float]
    canonical_labels: tuple[int, ...]
    minimum_component_mm3: float
    artifact_sha256: str
    validation_metrics: dict[str, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VesselModelManifest:
        required = {
            "schema_version",
            "name",
            "version",
            "dataset",
            "dataset_url",
            "research_only",
            "validated",
            "validation_fold",
            "training_cases",
            "target_spacing_mm",
            "patch_size",
            "overlap",
            "intensity_clip_hu",
            "canonical_labels",
            "minimum_component_mm3",
            "artifact_sha256",
            "validation_metrics",
        }
        missing = required.difference(data)
        if missing:
            raise VesselModelError(
                f"Model manifest is missing: {', '.join(sorted(missing))}"
            )
        manifest = cls(
            schema_version=int(data["schema_version"]),
            name=str(data["name"]),
            version=str(data["version"]),
            dataset=str(data["dataset"]),
            dataset_url=str(data["dataset_url"]),
            research_only=bool(data["research_only"]),
            validated=bool(data["validated"]),
            validation_fold=int(data["validation_fold"]),
            training_cases=int(data["training_cases"]),
            target_spacing_mm=tuple(float(value) for value in data["target_spacing_mm"]),
            patch_size=tuple(int(value) for value in data["patch_size"]),
            overlap=float(data["overlap"]),
            intensity_clip_hu=tuple(float(value) for value in data["intensity_clip_hu"]),
            canonical_labels=tuple(int(value) for value in data["canonical_labels"]),
            minimum_component_mm3=float(data["minimum_component_mm3"]),
            artifact_sha256=str(data["artifact_sha256"]).lower(),
            validation_metrics={
                str(key): float(value)
                for key, value in dict(data["validation_metrics"]).items()
            },
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.schema_version != 1:
            raise VesselModelError(
                f"Unsupported model manifest schema: {self.schema_version}"
            )
        if len(self.target_spacing_mm) != 3 or any(
            value <= 0 for value in self.target_spacing_mm
        ):
            raise VesselModelError("Target spacing must contain three positive values.")
        if len(self.patch_size) != 3 or any(value < 16 for value in self.patch_size):
            raise VesselModelError("Patch size must contain three values of at least 16.")
        if not 0.0 <= self.overlap < 1.0:
            raise VesselModelError("Sliding-window overlap must be in [0, 1).")
        if len(self.intensity_clip_hu) != 2 or (
            self.intensity_clip_hu[0] >= self.intensity_clip_hu[1]
        ):
            raise VesselModelError("HU clipping limits are invalid.")
        if int(CanonicalLabel.RENAL_ARTERY) not in self.canonical_labels or int(
            CanonicalLabel.RENAL_VEIN
        ) not in self.canonical_labels:
            raise VesselModelError("The model must include renal artery and vein channels.")
        if len(self.artifact_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.artifact_sha256
        ):
            raise VesselModelError("The model artifact needs a valid SHA-256 digest.")


@dataclass(frozen=True)
class VesselModelBundle:
    model_path: Path
    manifest_path: Path
    manifest: VesselModelManifest


@dataclass
class VesselPrediction:
    segmentation: np.ndarray
    confidence: np.ndarray
    provenance: dict[str, Any]


def read_manifest(path: Path) -> VesselModelManifest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VesselModelError(f"Could not read model manifest: {exc}") from exc
    if not isinstance(data, dict):
        raise VesselModelError("Model manifest must contain a JSON object.")
    return VesselModelManifest.from_dict(data)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def discover_model_bundle(model_dir: Path | None = None) -> VesselModelBundle | None:
    configured = os.environ.get("RENAL_VESSEL_MODEL_DIR")
    root = model_dir or (Path(configured).expanduser() if configured else None)
    if root is None:
        return None
    manifest_path = root / "manifest.json"
    model_path = root / "model.onnx"
    if not manifest_path.is_file() or not model_path.is_file():
        return None
    manifest = read_manifest(manifest_path)
    if not manifest.validated:
        raise VesselModelError(
            "The configured vessel model has not passed its validation gate."
        )
    actual_digest = sha256_file(model_path)
    if actual_digest != manifest.artifact_sha256:
        raise VesselModelError("The vessel model SHA-256 does not match its manifest.")
    return VesselModelBundle(model_path, manifest_path, manifest)


def _download_file(url: str, destination: Path, maximum_bytes: int) -> None:
    if urlparse(url).scheme != "https":
        raise VesselModelError("Remote model assets must use HTTPS.")
    request = Request(url, headers={"User-Agent": "renal-ct-explorer/1.0"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    downloaded = 0
    try:
        with urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
            declared_size = response.headers.get("Content-Length")
            if declared_size and int(declared_size) > maximum_bytes:
                raise VesselModelError("Remote model asset exceeds the size limit.")
            while block := response.read(1024 * 1024):
                downloaded += len(block)
                if downloaded > maximum_bytes:
                    raise VesselModelError("Remote model asset exceeds the size limit.")
                handle.write(block)
        temporary.replace(destination)
    except (OSError, ValueError, VesselModelError) as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, VesselModelError):
            raise
        raise VesselModelError(f"Could not download the vessel model: {exc}") from exc


def resolve_model_bundle(
    model_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> VesselModelBundle | None:
    local_bundle = discover_model_bundle(model_dir)
    if local_bundle is not None:
        return local_bundle

    manifest_url = os.environ.get("RENAL_VESSEL_MANIFEST_URL", "")
    model_url = os.environ.get("RENAL_VESSEL_MODEL_URL", "")
    if not manifest_url and not model_url:
        return None
    if not manifest_url or not model_url:
        raise VesselModelError(
            "Both RENAL_VESSEL_MANIFEST_URL and RENAL_VESSEL_MODEL_URL are required."
        )

    root = cache_dir or Path("/tmp/renal-ct-explorer-vessel-model")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    model_path = root / "model.onnx"
    _download_file(manifest_url, manifest_path, maximum_bytes=1024 * 1024)
    manifest = read_manifest(manifest_path)
    if not manifest.validated:
        raise VesselModelError("The remote vessel model has not passed validation.")
    if not model_path.is_file() or sha256_file(model_path) != manifest.artifact_sha256:
        _download_file(model_url, model_path, maximum_bytes=200 * 1024 * 1024)
    if sha256_file(model_path) != manifest.artifact_sha256:
        model_path.unlink(missing_ok=True)
        raise VesselModelError("The downloaded vessel model failed SHA-256 verification.")
    return VesselModelBundle(model_path, manifest_path, manifest)


def model_status(model_dir: Path | None = None) -> dict[str, Any]:
    try:
        bundle = resolve_model_bundle(model_dir)
    except VesselModelError as exc:
        return {"available": False, "message": str(exc), "manifest": None}
    if bundle is None:
        return {
            "available": False,
            "message": (
                "No validated KiPA22 ONNX model is installed. The image-filter vessel "
                "layer is not machine learning."
            ),
            "manifest": None,
        }
    return {
        "available": True,
        "message": f"{bundle.manifest.name} {bundle.manifest.version}",
        "manifest": bundle.manifest,
    }


def _crop_renal_region(
    shape: tuple[int, int, int],
    segmentation: np.ndarray | None,
    spacing: np.ndarray,
    margin_mm: float = 35.0,
) -> tuple[slice, slice, slice]:
    if segmentation is None:
        return tuple(slice(0, size) for size in shape)
    renal = np.isin(
        segmentation,
        (
            int(CanonicalLabel.KIDNEY),
            int(CanonicalLabel.TUMOR),
            int(CanonicalLabel.CYST),
        ),
    )
    coordinates = np.argwhere(renal)
    if coordinates.size == 0:
        return tuple(slice(0, size) for size in shape)
    margin = np.ceil(margin_mm / spacing).astype(int)
    minimum = np.maximum(coordinates.min(axis=0) - margin, 0)
    maximum = np.minimum(coordinates.max(axis=0) + margin + 1, np.asarray(shape))
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(minimum, maximum))


def _resize_to_shape(array: np.ndarray, shape: tuple[int, int, int], order: int) -> np.ndarray:
    factors = tuple(target / current for target, current in zip(shape, array.shape))
    resized = ndi.zoom(array, zoom=factors, order=order, prefilter=order > 1)
    output = np.zeros(shape, dtype=resized.dtype)
    shared = tuple(slice(0, min(target, actual)) for target, actual in zip(shape, resized.shape))
    output[shared] = resized[shared]
    return output


def _window_starts(size: int, patch: int, overlap: float) -> list[int]:
    if size <= patch:
        return [0]
    stride = max(1, int(round(patch * (1.0 - overlap))))
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _importance_map(patch_size: tuple[int, int, int]) -> np.ndarray:
    axes = []
    for size in patch_size:
        center = (size - 1) / 2.0
        sigma = max(size / 8.0, 1.0)
        coordinate = np.arange(size, dtype=np.float32)
        axes.append(np.exp(-0.5 * ((coordinate - center) / sigma) ** 2))
    importance = (
        axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    )
    return np.maximum(importance, 1e-3).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=0, keepdims=True)


def _sliding_window_probabilities(
    session: Any,
    volume: np.ndarray,
    patch_size: tuple[int, int, int],
    overlap: float,
    class_count: int,
) -> np.ndarray:
    pad_width = [
        (0, max(0, patch - size)) for size, patch in zip(volume.shape, patch_size)
    ]
    padded = np.pad(volume, pad_width, mode="constant", constant_values=0.0)
    probability_sum = np.zeros((class_count, *padded.shape), dtype=np.float32)
    weight_sum = np.zeros(padded.shape, dtype=np.float32)
    importance = _importance_map(patch_size)
    starts = [
        _window_starts(size, patch, overlap)
        for size, patch in zip(padded.shape, patch_size)
    ]
    input_name = session.get_inputs()[0].name
    for first in starts[0]:
        for second in starts[1]:
            for third in starts[2]:
                window = (
                    slice(first, first + patch_size[0]),
                    slice(second, second + patch_size[1]),
                    slice(third, third + patch_size[2]),
                )
                tensor = padded[window][None, None].astype(np.float32, copy=False)
                output = np.asarray(session.run(None, {input_name: tensor})[0])
                if output.ndim != 5 or output.shape[1] != class_count:
                    raise VesselModelError(
                        f"Unexpected ONNX output shape: {tuple(output.shape)}"
                    )
                probabilities = _softmax(output[0])
                probability_sum[(slice(None), *window)] += probabilities * importance
                weight_sum[window] += importance
    probability_sum /= np.maximum(weight_sum[None], 1e-7)
    original = tuple(slice(0, size) for size in volume.shape)
    return probability_sum[(slice(None), *original)]


def _remove_small_vessel_components(
    prediction: np.ndarray,
    spacing: tuple[float, float, float],
    minimum_component_mm3: float,
) -> np.ndarray:
    cleaned = prediction.copy()
    voxel_volume = float(np.prod(spacing))
    minimum_voxels = max(1, int(np.ceil(minimum_component_mm3 / voxel_volume)))
    for label_value in (
        int(CanonicalLabel.RENAL_ARTERY),
        int(CanonicalLabel.RENAL_VEIN),
    ):
        labels, count = ndi.label(cleaned == label_value)
        if count == 0:
            continue
        component_sizes = np.bincount(labels.ravel())
        remove_ids = np.flatnonzero(component_sizes < minimum_voxels)
        remove_ids = remove_ids[remove_ids != 0]
        cleaned[np.isin(labels, remove_ids)] = int(CanonicalLabel.BACKGROUND)
    return cleaned


@lru_cache(maxsize=1)
def _onnx_session(model_path: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise VesselModelError(
            "ONNX Runtime is required for trained vessel inference."
        ) from exc
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def predict_vessels(
    image: np.ndarray,
    spacing: tuple[float, float, float],
    segmentation: np.ndarray | None,
    bundle: VesselModelBundle,
    session: Any | None = None,
) -> VesselPrediction:
    """Run a validated KiPA22 model and return only artery/vein predictions."""

    if image.ndim != 3:
        raise VesselModelError(f"Expected a 3D CT image, got {image.shape}.")
    manifest = bundle.manifest
    spacing_array = np.asarray(spacing, dtype=float)
    crop = _crop_renal_region(image.shape, segmentation, spacing_array)
    image_crop = image[crop]
    output_shape = tuple(
        max(2, int(round(size * source / target)))
        for size, source, target in zip(
            image_crop.shape,
            spacing,
            manifest.target_spacing_mm,
        )
    )
    resampled = _resize_to_shape(image_crop, output_shape, order=1).astype(np.float32)
    low, high = manifest.intensity_clip_hu
    resampled = np.clip(resampled, low, high)
    resampled = (resampled - low) / (high - low)
    active_session = session or _onnx_session(str(bundle.model_path))
    probabilities = _sliding_window_probabilities(
        active_session,
        resampled,
        manifest.patch_size,
        manifest.overlap,
        len(manifest.canonical_labels),
    )
    channel_prediction = np.argmax(probabilities, axis=0)
    lookup = np.asarray(manifest.canonical_labels, dtype=np.uint8)
    canonical = lookup[channel_prediction]
    confidence = np.max(probabilities, axis=0).astype(np.float32)
    vessel_only = np.where(
        np.isin(
            canonical,
            (
                int(CanonicalLabel.RENAL_ARTERY),
                int(CanonicalLabel.RENAL_VEIN),
            ),
        ),
        canonical,
        int(CanonicalLabel.BACKGROUND),
    ).astype(np.uint8)
    vessel_crop = _resize_to_shape(vessel_only, image_crop.shape, order=0)
    confidence_crop = _resize_to_shape(confidence, image_crop.shape, order=1)
    vessel_crop = _remove_small_vessel_components(
        vessel_crop,
        spacing,
        manifest.minimum_component_mm3,
    )
    full_prediction = np.zeros(image.shape, dtype=np.uint8)
    full_confidence = np.zeros(image.shape, dtype=np.float32)
    full_prediction[crop] = vessel_crop
    full_confidence[crop] = confidence_crop
    vessel_voxels = full_prediction > 0
    return VesselPrediction(
        segmentation=full_prediction,
        confidence=full_confidence,
        provenance={
            "model": manifest.name,
            "version": manifest.version,
            "dataset": manifest.dataset,
            "validation_fold": manifest.validation_fold,
            "training_cases": manifest.training_cases,
            "validation_metrics": manifest.validation_metrics,
            "mean_vessel_confidence": (
                float(np.mean(full_confidence[vessel_voxels]))
                if np.any(vessel_voxels)
                else 0.0
            ),
            "artery_voxels": int(
                np.count_nonzero(
                    full_prediction == int(CanonicalLabel.RENAL_ARTERY)
                )
            ),
            "vein_voxels": int(
                np.count_nonzero(full_prediction == int(CanonicalLabel.RENAL_VEIN))
            ),
            "research_only": manifest.research_only,
        },
    )
