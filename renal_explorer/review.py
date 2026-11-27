"""Model review states.

Two independent tracks, as required for a research planning tool:

* **Automated QA** (``pending`` / ``passed`` / ``failed``) is computed from geometry and
  integrity checks. Passing it says nothing about whether the anatomy is correct.
* **Clinician review** (``unreviewed`` / ``reviewed`` / ``accepted`` / ``rejected``) is
  recorded by a named person, with an append-only history.

"Accepted for research analysis" needs both: automated QA passed *and* a clinician review
with every anatomical check confirmed. A review is bound to the exact CT, labels and grid it
was made on (``review_case_key``), so changed data never inherits an old review.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

QA_PENDING, QA_PASSED, QA_FAILED = "pending", "passed", "failed"
UNREVIEWED, REVIEWED, ACCEPTED, REJECTED = "unreviewed", "reviewed", "accepted", "rejected"
CLINICIAN_STATES = (UNREVIEWED, REVIEWED, ACCEPTED, REJECTED)

STATE_LABELS = {
    "generated": "Generated: not anatomically verified",
    QA_PASSED: "Automated QA passed: geometry checks only",
    REVIEWED: "Clinician reviewed",
    ACCEPTED: "Accepted for research analysis",
    REJECTED: "Rejected by clinician",
    QA_FAILED: "Automated QA failed",
}

REVIEW_CHECKS = {
    "kidney_contour": "Kidney contour matches the CT",
    "tumour_boundaries": "Tumour boundaries, including the deep margin, match the CT",
    "vessels": "Vessels checked, or marked not adequately visualised",
    "collecting_system": "Collecting system checked, or marked not adequately visualised",
    "no_discrepancy": "No major unexpected anatomical discrepancy",
}

# Engineering criteria for the automated geometry checks (not guideline thresholds).
QA_MAX_VOLUME_ERROR_PCT = 5.0
QA_SURFACE_TO_MASK_FACTOR = 1.0  # p95 departure from the mask <= coarsest voxel


class ReviewError(ValueError):
    """An invalid review transition."""


def review_case_key(*parts: Any) -> str:
    """Stable key for the exact data a review applies to (hashes, grid, prediction)."""

    text = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def automated_qa(
    meshes: dict[str, dict[str, Any] | None],
    spacing: tuple[float, float, float] | list[float],
    reduced_resolution: bool = False,
) -> dict[str, Any]:
    """Geometry and integrity checks on the reference meshes of one plan."""

    checks: list[dict[str, Any]] = []
    surface_limit = QA_SURFACE_TO_MASK_FACTOR * float(max(spacing))
    for name, mesh in meshes.items():
        if mesh is None or len(mesh.get("faces", [])) == 0:
            checks.append(
                {
                    "structure": name,
                    "check": "mesh present",
                    "passed": False,
                    "detail": "empty mesh",
                }
            )
            continue
        quality = mesh.get("quality", {})
        vertices = mesh["vertices"]
        finite = bool(np.all(np.isfinite(vertices)))
        checks.append(
            {
                "structure": name,
                "check": "finite coordinates",
                "passed": finite,
                "detail": "" if finite else "NaN or infinite vertex",
            }
        )
        watertight = bool(quality.get("watertight", True))
        checks.append(
            {
                "structure": name,
                "check": "closed surface",
                "passed": watertight,
                "detail": "" if watertight else "surface has holes",
            }
        )
        error = float(quality.get("volume_error_pct", 0.0))
        checks.append(
            {
                "structure": name,
                "check": f"mesh vs mask volume within {QA_MAX_VOLUME_ERROR_PCT:.0f} %",
                "passed": error <= QA_MAX_VOLUME_ERROR_PCT,
                "detail": f"{error:.2f} %",
            }
        )
        if quality.get("surface_to_mask_p95_mm") is not None:
            departure = float(quality["surface_to_mask_p95_mm"])
            checks.append(
                {
                    "structure": name,
                    "check": f"surface within {surface_limit:.1f} mm of mask (p95)",
                    "passed": departure <= surface_limit,
                    "detail": f"{departure:.2f} mm",
                }
            )
    if not checks:
        status = QA_PENDING
    else:
        status = QA_PASSED if all(item["passed"] for item in checks) else QA_FAILED
    warnings = []
    if reduced_resolution:
        warnings.append("Demo resolution: geometry is approximate.")
    return {
        "status": status,
        "checks": checks,
        "warnings": warnings,
        "criteria": "engineering criteria, not guideline thresholds",
    }


@dataclass
class ReviewRecord:
    case_key: str
    case_label: str = ""
    status: str = UNREVIEWED
    checks: dict[str, bool] = field(default_factory=lambda: dict.fromkeys(REVIEW_CHECKS, False))
    reviewer: str = ""
    notes: str = ""
    limitations: list[str] = field(default_factory=list)
    updated_at: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReviewRecord:
        known = {key: payload[key] for key in cls.__dataclass_fields__ if key in payload}
        record = cls(**known)
        record.checks = {key: bool(record.checks.get(key, False)) for key in REVIEW_CHECKS}
        if record.status not in CLINICIAN_STATES:
            raise ReviewError(f"Unknown review status {record.status!r}")
        return record


def overall_state(qa_status: str, record: ReviewRecord | None) -> str:
    """The single state shown on screen and in exports."""

    status = record.status if record else UNREVIEWED
    if status == REJECTED:
        return REJECTED
    if status == ACCEPTED and qa_status == QA_PASSED:
        return ACCEPTED
    if status in (REVIEWED, ACCEPTED):
        return REVIEWED
    if qa_status == QA_FAILED:
        return QA_FAILED
    if qa_status == QA_PASSED:
        return QA_PASSED
    return "generated"


def transition(
    record: ReviewRecord,
    new_status: str,
    *,
    reviewer: str,
    qa_status: str,
    checks: dict[str, bool] | None = None,
    notes: str = "",
    limitations: list[str] | None = None,
    now: float | None = None,
) -> ReviewRecord:
    """Apply one clinician action, enforcing the rules, and log it."""

    reviewer = reviewer.strip()
    if new_status not in CLINICIAN_STATES:
        raise ReviewError(f"Unknown review status {new_status!r}")
    if not reviewer:
        raise ReviewError("Enter the reviewer's name.")
    checks = {key: bool((checks or record.checks).get(key, False)) for key in REVIEW_CHECKS}
    if new_status == REVIEWED and not any(checks.values()):
        raise ReviewError("Confirm at least one anatomical check before marking reviewed.")
    if new_status == ACCEPTED:
        if record.status != REVIEWED:
            raise ReviewError("Mark the model reviewed before accepting it.")
        if qa_status != QA_PASSED:
            raise ReviewError("Automated QA must pass before acceptance.")
        missing = [REVIEW_CHECKS[key] for key, value in checks.items() if not value]
        if missing:
            raise ReviewError("Not yet confirmed: " + "; ".join(missing))
    if new_status in (REJECTED, UNREVIEWED) and not notes.strip():
        raise ReviewError("Add a note explaining the rejection or reopening.")

    timestamp = time.time() if now is None else now
    record.history.append(
        {
            "time": timestamp,
            "reviewer": reviewer,
            "from": record.status,
            "to": new_status,
            "qa_status": qa_status,
            "checks": checks,
            "notes": notes.strip(),
        }
    )
    record.status = new_status
    record.checks = checks
    record.reviewer = reviewer
    record.notes = notes.strip()
    if limitations is not None:
        record.limitations = [item.strip() for item in limitations if item.strip()]
    record.updated_at = timestamp
    return record


class ReviewStore:
    """One JSON file per case key. ``root=None`` keeps reviews in memory only."""

    def __init__(self, root: str | os.PathLike | None) -> None:
        self.root = Path(root).expanduser() if root else None
        self._memory: dict[str, dict[str, Any]] = {}
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self.root, 0o700)  # reviews name people and cases
            except OSError:
                pass

    @property
    def persistent(self) -> bool:
        return self.root is not None

    def _path(self, case_key: str) -> Path:
        if not case_key.isalnum():
            raise ReviewError("Invalid case key.")
        assert self.root is not None
        return self.root / f"{case_key}.json"

    def load(self, case_key: str, case_label: str = "") -> ReviewRecord:
        payload = None
        if self.root is None:
            payload = self._memory.get(case_key)
        elif self._path(case_key).exists():
            payload = json.loads(self._path(case_key).read_text())
        if payload is None:
            return ReviewRecord(case_key=case_key, case_label=case_label)
        return ReviewRecord.from_dict(payload)

    def save(self, record: ReviewRecord) -> None:
        payload = record.to_dict()
        if self.root is None:
            self._memory[record.case_key] = json.loads(json.dumps(payload))
            return
        path = self._path(record.case_key)
        handle, temporary = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(handle, "w") as stream:
            json.dump(payload, stream, indent=2)
        os.replace(temporary, path)
