"""DICOM upload handling: safe extraction, series discovery, phase guess, NIfTI conversion.

SimpleITK is imported lazily so the public demo (which never sees DICOM) does not need it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_UNCOMPRESSED_BYTES = 6 * 1024**3
MAX_FILES = 20_000
MIN_SLICES = 20
# Kept inside raw/ only, so they are deleted together with the DICOM upload.
PRIVATE_INDEX = ".series_index.json"
SOURCE_RECORD = ".source.json"

PHASES = ("nephrographic", "arterial", "excretory", "non-contrast", "other")
MAIN_PHASE_PREFERENCE = ("nephrographic", "arterial", "excretory", "non-contrast", "other")

# English and German series-description keywords (lower-case, accents folded).
PHASE_KEYWORDS = {
    "excretory": (
        "excret", "delay", "urogra", "late", "spat", "spaet", "ausscheid", "pyelo", "10 min",
        "5 min", "10min", "5min",
    ),
    "arterial": ("arter", "angio", "cta", "art.", " art ", "early", "fruh", "frueh"),
    "nephrographic": (
        "nephro", "venous", "venos", "venoes", "portal", "pv", "parenchym", "70s", "80s",
        "90s", "100s", "100 s",
    ),
    "non-contrast": ("native", "nativ", "non-con", "noncon", "plain", "w/o", "ohne km", "nc "),
}


class DicomIntakeError(RuntimeError):
    pass


@dataclass
class SeriesInfo:
    series_uid: str
    description: str
    slices: int
    slice_thickness_mm: float | None
    series_time: str
    phase_guess: str
    files: list[str]


def write_series_index(raw_dir: Path, series: list[SeriesInfo]) -> list[dict[str, Any]]:
    """Store UIDs and descriptions privately (in raw/); return opaque public records.

    Series descriptions can contain free text and UIDs are linkable identifiers, so neither
    is ever written outside raw/, which is deleted after conversion.
    """

    index: dict[str, dict[str, str]] = {}
    public: list[dict[str, Any]] = []
    for number, item in enumerate(series, start=1):
        handle = f"S{number}"
        index[handle] = {"series_uid": item.series_uid, "description": item.description}
        public.append(
            {
                "handle": handle,
                "slices": item.slices,
                "slice_thickness_mm": item.slice_thickness_mm,
                "phase_guess": item.phase_guess,
            }
        )
    (raw_dir / PRIVATE_INDEX).write_text(json.dumps(index))
    return public


def read_series_index(raw_dir: Path) -> dict[str, dict[str, str]]:
    try:
        return json.loads((raw_dir / PRIVATE_INDEX).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def source_fingerprint(source: Path) -> str:
    """Identity of an inbox item: inode, size and mtime of it and of every file inside.

    A different study later saved under the same name gets a different fingerprint.
    """

    digest = hashlib.sha256()
    entries = [source] if source.is_file() else [source, *sorted(source.rglob("*"))]
    for entry in entries:
        info = entry.lstat()
        relative = "." if entry == source else str(entry.relative_to(source))
        digest.update(f"{relative}|{info.st_ino}|{info.st_size}|{info.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def record_inbox_source(raw_dir: Path, source: Path, fingerprint: str | None = None) -> None:
    """Remember which inbox item may be deleted after conversion.

    Pass the fingerprint taken *before* copying; the caller checks it again after the copy.
    """

    record = {"path": str(source), "fingerprint": fingerprint or source_fingerprint(source)}
    (raw_dir / SOURCE_RECORD).write_text(json.dumps(record))


def consume_inbox_source(raw_dir: Path, inbox_root: Path) -> bool:
    """Delete the original inbox study once its conversion has succeeded.

    Only a direct child of the inbox folder is ever deleted, and only if it is still the
    very item that was imported (same fingerprint).
    """

    try:
        record = json.loads((raw_dir / SOURCE_RECORD).read_text())
    except (OSError, json.JSONDecodeError):
        return False
    source = Path(record.get("path", ""))
    try:
        if source.resolve().parent != inbox_root.resolve() or source.is_symlink():
            return False
        if not source.exists() or source_fingerprint(source) != record.get("fingerprint"):
            return False
    except OSError:
        return False
    if source.is_dir():
        shutil.rmtree(source)
    elif source.is_file():
        source.unlink()
    else:
        return False
    return True


def _fold(text: str) -> str:
    text = text.lower()
    for source, target in (("ä", "a"), ("ö", "o"), ("ü", "u"), ("ß", "ss")):
        text = text.replace(source, target)
    return f" {text} "


def guess_phase(description: str) -> str:
    folded = _fold(description)
    for phase in ("excretory", "arterial", "non-contrast", "nephrographic"):
        if any(keyword in folded for keyword in PHASE_KEYWORDS[phase]):
            return phase
    return "other"


def safe_extract_zip(archive_path: Path, destination: Path) -> int:
    """Extract a ZIP while blocking path traversal, links and zip bombs."""

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    total = 0
    count = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if len(members) > MAX_FILES:
            raise DicomIntakeError(f"Too many files in the upload (>{MAX_FILES}).")
        for member in members:
            total += member.file_size
            if total > MAX_UNCOMPRESSED_BYTES:
                raise DicomIntakeError("The upload is too large once uncompressed.")
            mode = (member.external_attr >> 16) & 0o170000
            if mode == 0o120000:  # symbolic link
                continue
            name = member.filename.replace("\\", "/")
            if name.startswith("/") or ".." in Path(name).parts:
                continue
            if Path(name).name.startswith(".") or "__MACOSX" in name:
                continue
            target = (destination / f"{count:06d}_{Path(name).name}").resolve()
            if destination not in target.parents:
                continue
            with archive.open(member) as source, open(target, "wb") as sink:
                while chunk := source.read(1024 * 1024):
                    sink.write(chunk)
            count += 1
    if count == 0:
        raise DicomIntakeError("The ZIP did not contain any files.")
    return count


def _sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - depends on the patient install
        raise DicomIntakeError(
            "SimpleITK is required for DICOM import. Install requirements-patient.txt."
        ) from exc
    return sitk


def list_series(folder: Path) -> list[SeriesInfo]:
    sitk = _sitk()
    reader = sitk.ImageFileReader()
    series: list[SeriesInfo] = []
    directories = {path.parent for path in folder.rglob("*") if path.is_file()}
    for directory in sorted(directories):
        for uid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or ():
            files = list(
                sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), uid)
            )
            if len(files) < MIN_SLICES:
                continue
            reader.SetFileName(files[len(files) // 2])
            reader.LoadPrivateTagsOff()
            try:
                reader.ReadImageInformation()
            except RuntimeError:
                continue

            def tag(key: str, default: str = "") -> str:
                return reader.GetMetaData(key).strip() if reader.HasMetaDataKey(key) else default

            modality = tag("0008|0060")
            image_type = tag("0008|0008").upper()
            if modality and modality != "CT":
                continue
            if "LOCALIZER" in image_type or "SCOUT" in image_type:
                continue
            description = re.sub(r"\s+", " ", tag("0008|103e") or tag("0018|1030") or "")
            try:
                thickness = float(tag("0018|0050")) if tag("0018|0050") else None
            except ValueError:
                thickness = None
            series.append(
                SeriesInfo(
                    series_uid=uid,
                    description=description[:80],
                    slices=len(files),
                    slice_thickness_mm=thickness,
                    series_time=tag("0008|0031"),
                    phase_guess=guess_phase(description),
                    files=files,
                )
            )
    series.sort(key=lambda item: (item.series_time, item.description))
    return series


def default_assignment(series: list[SeriesInfo]) -> dict[str, str]:
    """Pick at most one series per phase, preferring thin slices and more coverage."""

    chosen: dict[str, SeriesInfo] = {}
    for item in series:
        phase = item.phase_guess
        if phase == "other":
            continue
        current = chosen.get(phase)
        key = (-(item.slice_thickness_mm or 99.0), item.slices)
        if current is None or key > (-(current.slice_thickness_mm or 99.0), current.slices):
            chosen[phase] = item
    return {phase: item.series_uid for phase, item in chosen.items()}


def main_phase(assignment: dict[str, str]) -> str | None:
    for phase in MAIN_PHASE_PREFERENCE:
        if phase in assignment:
            return phase
    return None


def convert_series(files: list[str], output_path: Path) -> dict[str, Any]:
    """Write one series as NIfTI (HU), dropping every DICOM header field."""

    sitk = _sitk()
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    reader.MetaDataDictionaryArrayUpdateOff()
    reader.LoadPrivateTagsOff()
    image = reader.Execute()
    if image.GetDimension() != 3:
        raise DicomIntakeError("Only 3D CT series are supported.")
    image = sitk.Cast(image, sitk.sitkInt16)
    for key in list(image.GetMetaDataKeys()):
        image.EraseMetaData(key)
    sitk.WriteImage(image, str(output_path), useCompression=True)
    return {
        "size": list(image.GetSize()),
        "spacing_mm": [float(value) for value in image.GetSpacing()],
    }
