from __future__ import annotations

import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .models import CaseFiles


class RemoteDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class DemoCase:
    case_id: str
    description: str


# A deliberately small catalogue for free hosting. More cases can be added after checking
# memory, transfer, and licensing impact on the deployed application.
_EXAMPLES = (
    DemoCase("case_00015", "Compact KiTS23 example"),
    DemoCase("case_00000", "KiTS23 example 1"),
    DemoCase("case_00050", "KiTS23 example 2"),
)


def _scoring_cases() -> tuple[DemoCase, ...]:
    """The blind R.E.N.A.L. scoring set (validation/scoring_cases.txt)."""

    path = Path(__file__).resolve().parents[1] / "validation" / "scoring_cases.txt"
    try:
        identifiers = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    except OSError:
        return ()
    known = {item.case_id for item in _EXAMPLES}
    return tuple(
        DemoCase(case_id, "Scoring set") for case_id in identifiers if case_id not in known
    )


DEMO_CASES = _EXAMPLES + _scoring_cases()

HF_IMAGE_BASE = (
    "https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images"
)
GITHUB_SEGMENTATION_BASE = (
    "https://raw.githubusercontent.com/neheller/kits23/main/dataset"
)
MAX_DOWNLOAD_BYTES = 600 * 1024 * 1024


def demo_case_ids() -> tuple[str, ...]:
    return tuple(item.case_id for item in DEMO_CASES)


def demo_case_label(case_id: str) -> str:
    for item in DEMO_CASES:
        if item.case_id == case_id:
            return f"{item.case_id} — {item.description}"
    return case_id


def remote_urls(case_id: str) -> tuple[str, str]:
    if case_id not in demo_case_ids():
        raise RemoteDataError(f"Case {case_id} is not in the public demo catalogue.")
    return (
        f"{HF_IMAGE_BASE}/{case_id}.nii.gz",
        f"{GITHUB_SEGMENTATION_BASE}/{case_id}/segmentation.nii.gz",
    )


KITS23_CASE_ID = re.compile(r"^case_\d{5}$")


def kits23_case_urls(case_id: str) -> tuple[str, str]:
    """Official image/segmentation URLs for any KiTS23 case (validation and scoring use)."""

    if not KITS23_CASE_ID.match(case_id):
        raise RemoteDataError(f"Invalid KiTS23 case ID {case_id!r}.")
    return (
        f"{HF_IMAGE_BASE}/{case_id}.nii.gz",
        f"{GITHUB_SEGMENTATION_BASE}/{case_id}/segmentation.nii.gz",
    )


def download_kits23_case(case_id: str, cache_root: Path) -> CaseFiles:
    image_url, segmentation_url = kits23_case_urls(case_id)
    case_dir = cache_root / "kits23" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    image_path = case_dir / "imaging.nii.gz"
    segmentation_path = case_dir / "segmentation.nii.gz"
    _download_if_needed(segmentation_url, segmentation_path)
    _download_if_needed(image_url, image_path)
    return CaseFiles(case_id, image_path, segmentation_path)


def ensure_demo_case(case_id: str, cache_root: Path) -> CaseFiles:
    """Download one official KiTS23 case into an ephemeral server cache."""

    image_url, segmentation_url = remote_urls(case_id)
    case_dir = cache_root / "kits23" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    image_path = case_dir / "imaging.nii.gz"
    segmentation_path = case_dir / "segmentation.nii.gz"

    _download_if_needed(image_url, image_path)
    _download_if_needed(segmentation_url, segmentation_path)
    return CaseFiles(case_id, image_path, segmentation_path)


def _download_if_needed(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 1024:
        return

    partial = destination.with_name(f".{destination.name}.partial")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "renal-ct-research-explorer/1.0"},
    )
    downloaded = 0
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
                raise RemoteDataError(
                    f"Remote file exceeds the {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB limit."
                )
            with partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    downloaded += len(chunk)
                    if downloaded > MAX_DOWNLOAD_BYTES:
                        raise RemoteDataError("Remote file exceeded the safe download limit.")
                    output.write(chunk)
        os.replace(partial, destination)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        if isinstance(exc, RemoteDataError):
            raise
        raise RemoteDataError(f"Could not download {destination.name}: {exc}") from exc
