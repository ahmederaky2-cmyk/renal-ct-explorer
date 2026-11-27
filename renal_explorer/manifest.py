"""Machine-readable reconstruction manifest for every export.

It records what the model was built from (content hashes, geometry), how (algorithm and
parameters, software version), how good the geometry is (mesh QA) and who has reviewed it.
Values are filled from the case; nothing here is a placeholder.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_SCHEMA = "renal-ct-explorer/reconstruction-manifest"
MANIFEST_SCHEMA_VERSION = 1

MEASUREMENT_DEFINITIONS = {
    "volumes": "Voxel count x physical voxel volume |det(affine)|, from the label masks.",
    "tumor_max_diameter_mm": (
        "Maximum Euclidean distance between tumour surface voxels (convex hull), in mm; "
        "not an axis-aligned bounding-box size."
    ),
    "distances_gap_mm": (
        "Minimum surface-to-surface gap between the tumour mask and each structure mask, "
        "in mm (voxel based). Not a centreline distance."
    ),
    "nephrometry": (
        "Automated approximations of R.E.N.A.L. and PADUA; categorical items need clinician "
        "verification."
    ),
    "meshes": (
        "Display only. Measurements never come from meshes. Exported STL files are the "
        "unsmoothed reference surfaces."
    ),
}


@lru_cache(maxsize=1)
def software_version() -> dict[str, str]:
    """App version: commit from the environment (deployments) or the local git checkout."""

    commit = os.environ.get("RENAL_APP_COMMIT") or os.environ.get("GITHUB_SHA") or ""
    if not commit:
        root = Path(__file__).resolve().parents[1]
        try:
            commit = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            commit = "unknown"
    return {"name": "renal-ct-explorer", "commit": commit}


def segmentation_source(case, imported_prediction=None, patient_provenance=None) -> dict:
    """Provenance of the labels: dataset annotation, AI prediction, or both."""

    metadata = getattr(case, "metadata", {}) or {}
    if patient_provenance is not None:
        source: dict[str, Any] = {
            "type": "ai_prediction",
            "detail": "Automatic segmentation on the private workstation",
            "models": {
                key: patient_provenance.get(key)
                for key in ("totalsegmentator", "tumor_model", "kidney_cysts")
                if patient_provenance.get(key)
            },
        }
    elif getattr(case, "segmentation", None) is None:
        source = {"type": "none"}
    else:
        source = {
            "type": "dataset_reference_annotation",
            "dataset": metadata.get("dataset") or metadata.get("source"),
        }
    if imported_prediction is not None:
        provenance = getattr(imported_prediction, "provenance", {}) or {}
        source["vessels"] = {
            "type": "ai_prediction",
            "model": provenance.get("model"),
            "package_sha256": getattr(imported_prediction, "package_sha256", None),
        }
    corrections = (patient_provenance or {}).get("manual_corrections") or []
    source["clinician_corrected"] = bool(corrections)
    if corrections:
        source["corrections"] = corrections
    return source


def build_manifest(
    *,
    case_id: str,
    dataset: str,
    ct_sha256: str | None,
    segmentation_sha256: str | None,
    image_shape: tuple[int, ...],
    spacing: tuple[float, ...] | list[float],
    affine: np.ndarray,
    resolution: dict[str, Any] | None,
    segmentation: dict[str, Any],
    reconstruction: dict[str, Any],
    mesh_quality: dict[str, dict[str, Any]],
    qa: dict[str, Any],
    review: dict[str, Any] | None,
    state: str,
    limitations: list[str] | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    resolution = resolution or {}
    limitations = list(limitations or [])
    if resolution.get("reduced"):
        limitations.append("Built on a reduced demo grid; geometry and measurements approximate.")
    if review is None or review.get("status") in (None, "unreviewed"):
        limitations.append("Not reviewed by a clinician.")
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": (created_at or datetime.now(timezone.utc)).isoformat(),
        "research_use_only": True,
        "case_id": case_id,
        "dataset": dataset,
        "source": {
            "ct_sha256": ct_sha256,
            "segmentation_sha256": segmentation_sha256,
            "image_shape": [int(value) for value in image_shape],
            "voxel_spacing_mm": [float(value) for value in spacing],
            "coordinate_system": "RAS (canonical, NIfTI); DICOM LPS is mirrored in x and y",
            "affine_matrix": np.asarray(affine, dtype=float).round(6).tolist(),
            "reduced_resolution": bool(resolution.get("reduced")),
            "full_resolution_spacing_mm": resolution.get("full_spacing_mm"),
        },
        "segmentation": segmentation,
        "reconstruction": {"software": software_version(), **reconstruction},
        "mesh_quality": mesh_quality,
        "measurement_definitions": MEASUREMENT_DEFINITIONS,
        "review": {
            "state": state,
            "geometry_qa_status": qa.get("status"),
            "geometry_qa_checks": qa.get("checks", []),
            "geometry_qa_criteria": qa.get("criteria"),
            "clinician_review_status": (review or {}).get("status", "unreviewed"),
            "reviewer": (review or {}).get("reviewer") or None,
            "reviewed_at": (review or {}).get("updated_at"),
            "checks": (review or {}).get("checks"),
            "history": (review or {}).get("history", []),
        },
        "limitations": limitations,
    }
