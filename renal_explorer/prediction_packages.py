from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from zipfile import BadZipFile, ZipFile

import nibabel as nib
import numpy as np

from .labels import CanonicalLabel

PACKAGE_SCHEMA_VERSION = 1
PACKAGE_FILES = {"vessels.nii.gz", "provenance.json"}
MAX_PACKAGE_BYTES = 200 * 1024 * 1024
MAX_NIFTI_BYTES = 1024 * 1024 * 1024
MAX_PROVENANCE_BYTES = 64 * 1024


class PredictionPackageError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImportedVesselPrediction:
    segmentation: np.ndarray
    provenance: dict[str, Any]
    package_sha256: str


def _bounded_read(handle, maximum_bytes: int) -> bytes:
    output = BytesIO()
    total = 0
    while block := handle.read(1024 * 1024):
        total += len(block)
        if total > maximum_bytes:
            raise PredictionPackageError("Prediction package content exceeds its size limit.")
        output.write(block)
    return output.getvalue()


def _read_gzip_nifti(payload: bytes) -> nib.Nifti1Image:
    try:
        with gzip.GzipFile(fileobj=BytesIO(payload), mode="rb") as handle:
            uncompressed = _bounded_read(handle, MAX_NIFTI_BYTES)
        image = nib.Nifti1Image.from_bytes(uncompressed)
    except (OSError, ValueError) as exc:
        raise PredictionPackageError(f"The vessel NIfTI file is unreadable: {exc}") from exc
    return nib.as_closest_canonical(image)


def _validate_provenance(
    provenance: dict[str, Any],
    expected_case_id: str,
    expected_shape: tuple[int, int, int],
    expected_affine: np.ndarray,
    expected_image_sha256: str,
) -> None:
    required = {
        "schema_version",
        "case_id",
        "source_image_sha256",
        "source_image_shape",
        "source_image_affine",
        "model",
        "research_only",
    }
    missing = required.difference(provenance)
    if missing:
        raise PredictionPackageError(
            f"Prediction provenance is missing: {', '.join(sorted(missing))}"
        )
    if int(provenance["schema_version"]) != PACKAGE_SCHEMA_VERSION:
        raise PredictionPackageError("Unsupported prediction package schema.")
    if str(provenance["case_id"]) != expected_case_id:
        raise PredictionPackageError(
            "The prediction belongs to a different case. Re-run Colab with the selected case ID."
        )
    if str(provenance["source_image_sha256"]).lower() != expected_image_sha256.lower():
        raise PredictionPackageError(
            "The prediction was generated from a different CT file (SHA-256 mismatch)."
        )
    recorded_shape = tuple(int(value) for value in provenance["source_image_shape"])
    if recorded_shape != expected_shape:
        raise PredictionPackageError("The recorded source-image dimensions do not match this CT.")
    recorded_affine = np.asarray(provenance["source_image_affine"], dtype=float)
    if recorded_affine.shape != (4, 4) or not np.allclose(
        recorded_affine, expected_affine, atol=1e-3
    ):
        raise PredictionPackageError("The recorded source-image geometry does not match this CT.")
    model = provenance["model"]
    if not isinstance(model, dict) or model.get("name") != "BA-Net":
        raise PredictionPackageError("Only BA-Net Colab prediction packages are accepted here.")
    try:
        fold = int(model["fold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PredictionPackageError("BA-Net provenance needs a valid checkpoint fold.") from exc
    if fold not in range(4):
        raise PredictionPackageError("BA-Net checkpoint fold must be between 0 and 3.")
    if model.get("weights_doi") != "10.5281/zenodo.7030423":
        raise PredictionPackageError("BA-Net checkpoint DOI does not match the pinned release.")
    if provenance["research_only"] is not True:
        raise PredictionPackageError("The prediction must be marked research-only.")


def load_banet_prediction_package(
    payload: bytes,
    expected_case_id: str,
    expected_shape: tuple[int, int, int],
    expected_affine: np.ndarray,
    expected_image_sha256: str,
) -> ImportedVesselPrediction:
    """Validate and map a Colab BA-Net package into canonical vessel labels."""

    if not payload or len(payload) > MAX_PACKAGE_BYTES:
        raise PredictionPackageError("Prediction package is empty or exceeds 200 MB.")
    try:
        with ZipFile(BytesIO(payload)) as archive:
            file_items = [item for item in archive.infolist() if not item.is_dir()]
            members = {item.filename: item for item in file_items}
            if len(file_items) != len(PACKAGE_FILES):
                raise PredictionPackageError(
                    "Prediction ZIP contains duplicate or unexpected file entries."
                )
            if set(members) != PACKAGE_FILES:
                raise PredictionPackageError(
                    "Prediction ZIP must contain only vessels.nii.gz and provenance.json."
                )
            if members["provenance.json"].file_size > MAX_PROVENANCE_BYTES:
                raise PredictionPackageError("Prediction provenance is unexpectedly large.")
            if members["vessels.nii.gz"].file_size > MAX_PACKAGE_BYTES:
                raise PredictionPackageError("Compressed vessel NIfTI exceeds 200 MB.")
            for item in members.values():
                if item.flag_bits & 0x1:
                    raise PredictionPackageError("Encrypted prediction packages are not supported.")
            with archive.open("provenance.json") as handle:
                provenance_payload = _bounded_read(handle, MAX_PROVENANCE_BYTES)
            with archive.open("vessels.nii.gz") as handle:
                nifti_payload = _bounded_read(handle, MAX_PACKAGE_BYTES)
    except BadZipFile as exc:
        raise PredictionPackageError("The uploaded prediction is not a valid ZIP package.") from exc

    try:
        provenance = json.loads(provenance_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictionPackageError("Prediction provenance is not valid JSON.") from exc
    if not isinstance(provenance, dict):
        raise PredictionPackageError("Prediction provenance must contain a JSON object.")
    _validate_provenance(
        provenance,
        expected_case_id,
        expected_shape,
        np.asarray(expected_affine, dtype=float),
        expected_image_sha256,
    )

    prediction_nii = _read_gzip_nifti(nifti_payload)
    if prediction_nii.shape != expected_shape:
        raise PredictionPackageError(
            f"Prediction dimensions differ from the CT: {prediction_nii.shape} vs {expected_shape}."
        )
    if not np.allclose(prediction_nii.affine, expected_affine, atol=1e-3):
        raise PredictionPackageError("Prediction and CT affine matrices do not match.")
    source = np.asanyarray(prediction_nii.dataobj)
    if not np.issubdtype(source.dtype, np.integer):
        # Float storage: check integrality slab by slab to avoid full float64 copies.
        for start in range(0, source.shape[2], 16):
            slab = source[:, :, start : start + 16]
            if not np.all(np.isfinite(slab)) or not np.array_equal(slab, np.rint(slab)):
                raise PredictionPackageError("Prediction labels must be finite integer values.")
    # BA-Net labels are 0-4; a range check avoids sorting the volume (np.unique).
    if source.size and (source.min() < 0 or source.max() > 4):
        found = np.unique(source[(source < 0) | (source > 4)])
        raise PredictionPackageError(
            f"Prediction contains unexpected BA-Net labels: {sorted(int(v) for v in found)}."
        )
    source = source.astype(np.uint8, copy=False)

    canonical = np.zeros(expected_shape, dtype=np.uint8)
    canonical[source == 1] = int(CanonicalLabel.RENAL_VEIN)
    canonical[source == 3] = int(CanonicalLabel.RENAL_ARTERY)
    provenance = {
        **provenance,
        "artery_voxels": int(np.count_nonzero(canonical == CanonicalLabel.RENAL_ARTERY)),
        "vein_voxels": int(np.count_nonzero(canonical == CanonicalLabel.RENAL_VEIN)),
    }
    return ImportedVesselPrediction(
        segmentation=canonical,
        provenance=provenance,
        package_sha256=hashlib.sha256(payload).hexdigest(),
    )
