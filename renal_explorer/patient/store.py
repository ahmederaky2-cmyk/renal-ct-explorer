"""On-disk store for patient cases processed on a private machine.

Layout (one folder per pseudonymous case)::

    <root>/
      PN-3f9a1c2e/
        status.json            # job state, step, timestamps, errors
        request.json           # chosen series/phases and options (no patient identifiers)
        imaging.nii.gz         # main (nephrographic/venous) phase, the planning grid
        segmentation.nii.gz    # canonical labels on the imaging grid
        phases/arterial.nii.gz # other phases rigidly registered to the imaging grid
        provenance.json        # models, versions, steps, warnings
        history/               # earlier segmentations kept when a correction is uploaded
        worker.log

No DICOM headers are kept. The raw upload is deleted as soon as the NIfTI conversion succeeds
(or fails). Case folders older than the retention period are purged.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(os.environ.get("RENAL_PATIENT_ROOT", "~/RenalPlanner/cases")).expanduser()
DEFAULT_RETENTION_DAYS = float(os.environ.get("RENAL_RETENTION_DAYS", "30"))

STATES = ("uploaded", "queued", "running", "done", "failed")


def _now() -> float:
    return time.time()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(temporary, path)


@dataclass(frozen=True)
class PatientCase:
    case_id: str
    path: Path

    @property
    def image_path(self) -> Path:
        return self.path / "imaging.nii.gz"

    @property
    def segmentation_path(self) -> Path:
        return self.path / "segmentation.nii.gz"

    @property
    def phases_dir(self) -> Path:
        return self.path / "phases"

    @property
    def raw_dir(self) -> Path:
        return self.path / "raw"

    @property
    def log_path(self) -> Path:
        return self.path / "worker.log"

    def status(self) -> dict[str, Any]:
        try:
            return json.loads((self.path / "status.json").read_text())
        except (OSError, json.JSONDecodeError):
            return {"state": "failed", "error": "status unreadable"}

    def update_status(self, **changes: Any) -> dict[str, Any]:
        status = self.status() if (self.path / "status.json").exists() else {}
        status.update(changes)
        status["updated"] = _now()
        _atomic_write_json(self.path / "status.json", status)
        return status

    def request(self) -> dict[str, Any]:
        try:
            return json.loads((self.path / "request.json").read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def write_request(self, request: dict[str, Any]) -> None:
        _atomic_write_json(self.path / "request.json", request)

    def provenance(self) -> dict[str, Any]:
        try:
            return json.loads((self.path / "provenance.json").read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def write_provenance(self, provenance: dict[str, Any]) -> None:
        _atomic_write_json(self.path / "provenance.json", provenance)

    def is_ready(self) -> bool:
        return (
            self.status().get("state") == "done"
            and self.image_path.is_file()
            and self.segmentation_path.is_file()
        )

    def discard_raw(self) -> None:
        shutil.rmtree(self.raw_dir, ignore_errors=True)


class PatientStore:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root).expanduser() if root else DEFAULT_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass

    def create_case(self, alias: str = "") -> PatientCase:
        for _attempt in range(10):
            case_id = f"PN-{secrets.token_hex(4)}"
            path = self.root / case_id
            try:
                path.mkdir()
            except FileExistsError:
                continue
            case = PatientCase(case_id, path)
            case.update_status(
                state="uploaded",
                created=_now(),
                alias=_clean_alias(alias),
                step="Waiting for series selection",
            )
            return case
        raise RuntimeError("Could not allocate a case identifier.")

    def get(self, case_id: str) -> PatientCase:
        if not case_id.startswith("PN-") or "/" in case_id or ".." in case_id:
            raise KeyError(case_id)
        path = self.root / case_id
        if not path.is_dir():
            raise KeyError(case_id)
        return PatientCase(case_id, path)

    def cases(self) -> list[PatientCase]:
        items = [
            PatientCase(path.name, path)
            for path in self.root.iterdir()
            if path.is_dir() and path.name.startswith("PN-")
        ]
        return sorted(items, key=lambda case: case.status().get("created", 0), reverse=True)

    def ready_cases(self) -> list[PatientCase]:
        return [case for case in self.cases() if case.is_ready()]

    def delete(self, case_id: str) -> None:
        shutil.rmtree(self.get(case_id).path, ignore_errors=True)

    def purge_expired(self, retention_days: float | None = None) -> list[str]:
        days = DEFAULT_RETENTION_DAYS if retention_days is None else retention_days
        if days <= 0:
            return []
        cutoff = _now() - days * 86_400
        removed = []
        for case in self.cases():
            if case.status().get("state") == "running":
                continue
            if case.status().get("created", _now()) < cutoff:
                shutil.rmtree(case.path, ignore_errors=True)
                removed.append(case.case_id)
        return removed


def _clean_alias(alias: str) -> str:
    """A short free-text label. Kept deliberately small; users are told not to enter names."""

    allowed = "".join(
        character for character in alias if character.isalnum() or character in " -_#"
    )
    return allowed.strip()[:24]
