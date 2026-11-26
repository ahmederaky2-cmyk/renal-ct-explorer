from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import nibabel as nib
import numpy as np
import requests

BANET_REPOSITORY = "https://github.com/ShishuaiHu/BA-Net.git"
BANET_BRANCH = "kipa22"
WEIGHTS_DOI = "10.5281/zenodo.7030423"
WEIGHTS_URL = "https://zenodo.org/records/7030423/files/kipa22.zip?download=1"
EXPECTED_ARCHIVE_BYTES = 3_708_026_487
EXPECTED_ARCHIVE_MD5 = "fb22a9730070fdf45ed3923b1d82b4d2"
PLANS_IDENTIFIER = "nnUNetPlans_FabiansResUNet_v2.1"
TRAINER = "BANetV2Trainer_1000Epoch"
TASK_NAME = "Task1001_KiPA22"
MAX_MEMBER_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class RemoteZipMember:
    name: str
    crc32: int
    compressed_size: int
    uncompressed_size: int
    compression: int
    local_header_offset: int


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _range_response(url: str, start: int, end: int, *, stream: bool = False):
    response = requests.get(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "renal-ct-explorer/1.0"},
        timeout=(20, 300),
        stream=stream,
    )
    if response.status_code != 206:
        response.close()
        raise RuntimeError(
            "The model host did not honor a byte-range request. Stopping to avoid "
            "downloading the full 3.7 GB ensemble."
        )
    return response


def _remote_size(url: str) -> int:
    with _range_response(url, 0, 0) as response:
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes 0-0/(\d+)", content_range)
        if not match:
            raise RuntimeError("Could not determine the published model archive size.")
        return int(match.group(1))


def read_remote_zip_index(url: str = WEIGHTS_URL) -> dict[str, RemoteZipMember]:
    """Read only the ZIP index at the end of the remote 3.7 GB archive."""

    archive_size = _remote_size(url)
    if archive_size != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError(
            f"Published archive size changed: {archive_size:,} vs "
            f"{EXPECTED_ARCHIVE_BYTES:,} expected. Review the Zenodo record before continuing."
        )
    tail_start = max(0, archive_size - 65_557)
    with _range_response(url, tail_start, archive_size - 1) as response:
        tail = response.content
    eocd_position = tail.rfind(b"PK\x05\x06")
    if eocd_position < 0:
        raise RuntimeError("The published model archive has no readable ZIP index.")
    eocd = struct.unpack_from("<4s4H2LH", tail, eocd_position)
    entry_count, central_size, central_offset = eocd[4], eocd[5], eocd[6]
    local_central_start = central_offset - tail_start
    if local_central_start >= 0 and local_central_start + central_size <= len(tail):
        central = tail[local_central_start : local_central_start + central_size]
    else:
        with _range_response(
            url,
            central_offset,
            central_offset + central_size - 1,
        ) as response:
            central = response.content

    members: dict[str, RemoteZipMember] = {}
    position = 0
    for _index in range(entry_count):
        if central[position : position + 4] != b"PK\x01\x02":
            raise RuntimeError("The published model archive index is malformed.")
        header = struct.unpack_from("<4s6H3L5H2L", central, position)
        name_length, extra_length, comment_length = header[10], header[11], header[12]
        name_start = position + 46
        name = central[name_start : name_start + name_length].decode("utf-8")
        members[name] = RemoteZipMember(
            name=name,
            crc32=header[7],
            compressed_size=header[8],
            uncompressed_size=header[9],
            compression=header[4],
            local_header_offset=header[16],
        )
        position += 46 + name_length + extra_length + comment_length
    return members


def _copy_member_stream(
    source: BinaryIO,
    destination: BinaryIO,
    compression: int,
    maximum_bytes: int,
) -> tuple[int, int]:
    decompressor = zlib.decompressobj(-15) if compression == ZIP_DEFLATED else None
    if compression not in (ZIP_STORED, ZIP_DEFLATED):
        raise RuntimeError(f"Unsupported ZIP compression method: {compression}.")
    output_size = 0
    crc = 0
    for compressed_block in source.iter_content(chunk_size=4 * 1024 * 1024):
        if not compressed_block:
            continue
        block = decompressor.decompress(compressed_block) if decompressor else compressed_block
        output_size += len(block)
        if output_size > maximum_bytes:
            raise RuntimeError("A model archive member exceeds the safety size limit.")
        destination.write(block)
        crc = zlib.crc32(block, crc)
    if decompressor:
        block = decompressor.flush()
        output_size += len(block)
        if output_size > maximum_bytes:
            raise RuntimeError("A model archive member exceeds the safety size limit.")
        destination.write(block)
        crc = zlib.crc32(block, crc)
    return output_size, crc & 0xFFFFFFFF


def download_remote_zip_member(
    member: RemoteZipMember,
    destination: Path,
    url: str = WEIGHTS_URL,
) -> None:
    if member.uncompressed_size > MAX_MEMBER_BYTES:
        raise RuntimeError(f"Refusing unexpectedly large model member: {member.name}.")
    with _range_response(
        url,
        member.local_header_offset,
        member.local_header_offset + 29,
    ) as response:
        local_header = response.content
    if local_header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Invalid local ZIP header for {member.name}.")
    name_length, extra_length = struct.unpack_from("<HH", local_header, 26)
    data_start = member.local_header_offset + 30 + name_length + extra_length
    data_end = data_start + member.compressed_size - 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with _range_response(url, data_start, data_end, stream=True) as response:
            with temporary.open("wb") as handle:
                output_size, crc = _copy_member_stream(
                    response,
                    handle,
                    member.compression,
                    MAX_MEMBER_BYTES,
                )
        if output_size != member.uncompressed_size or crc != member.crc32:
            raise RuntimeError(f"CRC/size verification failed for {member.name}.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def install_single_banet_fold(
    results_root: Path,
    fold: int = 0,
    url: str = WEIGHTS_URL,
) -> tuple[Path, dict[str, str]]:
    if fold not in range(4):
        raise ValueError("The published BA-Net archive contains folds 0 through 3.")
    model_root = (
        results_root
        / "nnUNet"
        / "3d_fullres"
        / TASK_NAME
        / f"{TRAINER}__{PLANS_IDENTIFIER}"
    )
    requested = {
        "BA-Net/plans.pkl": model_root / "plans.pkl",
        f"BA-Net/fold_{fold}/model_final_checkpoint.model": (
            model_root / f"fold_{fold}" / "model_final_checkpoint.model"
        ),
        f"BA-Net/fold_{fold}/model_final_checkpoint.model.pkl": (
            model_root / f"fold_{fold}" / "model_final_checkpoint.model.pkl"
        ),
    }
    index = read_remote_zip_index(url)
    integrity: dict[str, str] = {}
    for member_name, destination in requested.items():
        member = index.get(member_name)
        if member is None:
            raise RuntimeError(f"Published archive is missing {member_name}.")
        if not destination.is_file() or destination.stat().st_size != member.uncompressed_size:
            print(f"Downloading {member_name} ({member.compressed_size / 1024**2:.1f} MiB)…")
            download_remote_zip_member(member, destination, url)
        integrity[member_name] = f"crc32:{member.crc32:08x}"
    return model_root, integrity


def patch_banet_for_current_colab(repository: Path) -> None:
    """Apply narrowly scoped compatibility fixes to the temporary official checkout."""

    replacements = {
        repository / "nnUNet/nnunet/training/network_training/network_trainer.py": [
            (
                "torch.load(fname, map_location=torch.device('cpu'))",
                "torch.load(fname, map_location=torch.device('cpu'), weights_only=False)",
            )
        ],
        repository / "nnUNet/nnunet/training/model_restore.py": [
            (
                "torch.load(i, map_location=torch.device('cpu'))",
                "torch.load(i, map_location=torch.device('cpu'), weights_only=False)",
            )
        ],
        repository / "nnUNet/nnunet/training/loss_functions/boundary_loss.py": [
            ("astype(np.bool)", "astype(np.bool_)")
        ],
    }
    for path, changes in replacements.items():
        text = path.read_text(encoding="utf-8")
        for old, new in changes:
            if new not in text:
                if old not in text:
                    raise RuntimeError(f"Expected compatibility target not found in {path}.")
                text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")


def install_banet_dependencies(repository: Path) -> None:
    packages = [
        "batchgenerators>=0.23,<0.26",
        "dicom2nifti",
        "graphviz",
        "hiddenlayer",
        "medpy",
        "nibabel",
        "scikit-image",
        "scikit-learn",
        "SimpleITK",
        "tifffile",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "-e",
            str(repository / "nnUNet"),
        ],
        check=True,
    )


def build_prediction_package(
    source_image: Path,
    prediction: Path,
    output: Path,
    case_id: str,
    fold: int,
    banet_commit: str,
    member_integrity: dict[str, str],
    disable_tta: bool,
) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", case_id):
        raise ValueError("CASE_ID may contain only letters, numbers, dot, dash, and underscore.")
    source_nii = nib.as_closest_canonical(nib.load(str(source_image)))
    prediction_nii = nib.as_closest_canonical(nib.load(str(prediction)))
    if prediction_nii.shape != source_nii.shape or not np.allclose(
        prediction_nii.affine, source_nii.affine, atol=1e-3
    ):
        raise RuntimeError("BA-Net output geometry does not match the uploaded CT.")
    labels = np.asanyarray(prediction_nii.dataobj)
    if not np.all(np.isfinite(labels)) or not np.allclose(labels, np.rint(labels)):
        raise RuntimeError("BA-Net output does not contain finite integer labels.")
    unique_labels = {int(value) for value in np.unique(labels)}
    if not unique_labels.issubset({0, 1, 2, 3, 4}):
        raise RuntimeError(f"Unexpected BA-Net output labels: {sorted(unique_labels)}.")

    provenance = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "source_image_sha256": sha256_file(source_image),
        "source_image_shape": [int(value) for value in source_nii.shape],
        "source_image_affine": np.asarray(source_nii.affine, dtype=float).tolist(),
        "source_labels": sorted(unique_labels),
        "prediction_labels": {
            "1": "renal vein",
            "2": "kidney",
            "3": "renal artery",
            "4": "kidney tumor",
        },
        "model": {
            "name": "BA-Net",
            "trainer": TRAINER,
            "fold": fold,
            "source_repository": BANET_REPOSITORY,
            "source_branch": BANET_BRANCH,
            "source_commit": banet_commit,
            "weights_doi": WEIGHTS_DOI,
            "weights_url": WEIGHTS_URL,
            "weights_archive_bytes": EXPECTED_ARCHIVE_BYTES,
            "weights_archive_md5_published": EXPECTED_ARCHIVE_MD5,
            "downloaded_member_integrity": member_integrity,
            "test_time_augmentation": not disable_tta,
        },
        "training_domain": "cropped KiPA22 computed tomography angiography (CTA)",
        "validation_status": (
            "Published challenge checkpoint; not validated by this notebook on the current CT."
        ),
        "research_only": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Store labels as uint8 (BA-Net may write float32): 4x smaller and cheap to import.
    native_nii = nib.load(str(prediction))
    compact = nib.Nifti1Image(
        np.rint(np.asanyarray(native_nii.dataobj)).astype(np.uint8), native_nii.affine
    )
    compact.set_data_dtype(np.uint8)
    staging = tempfile.TemporaryDirectory(prefix="banet-package-")
    compact_path = Path(staging.name) / "vessels.nii.gz"  # intermediate, never left behind
    nib.save(compact, str(compact_path))
    with staging, ZipFile(output, "w") as archive:
        archive.write(compact_path, "vessels.nii.gz", compress_type=ZIP_STORED)
        archive.writestr(
            "provenance.json",
            json.dumps(provenance, indent=2),
            compress_type=ZIP_DEFLATED,
        )
    return provenance
