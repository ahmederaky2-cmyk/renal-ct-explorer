from __future__ import annotations

import io
import json
import os
import zipfile

import numpy as np
import pytest

from renal_explorer.exports import stl_zip
from renal_explorer.manifest import build_manifest, segmentation_source
from renal_explorer.review import (
    ACCEPTED,
    QA_FAILED,
    QA_PASSED,
    QA_PENDING,
    REJECTED,
    REVIEW_CHECKS,
    REVIEWED,
    UNREVIEWED,
    ReviewError,
    ReviewRecord,
    ReviewStore,
    automated_qa,
    overall_state,
    review_case_key,
    transition,
)
from renal_explorer.vessel_ml import sha256_file


def _cube_mesh(**quality):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    base = {"watertight": True, "volume_error_pct": 0.5, "surface_to_mask_p95_mm": 0.4}
    base.update(quality)
    return {"vertices": vertices, "faces": faces, "quality": base}


ALL_CHECKS = dict.fromkeys(REVIEW_CHECKS, True)


def test_qa_is_pending_without_meshes_and_fails_on_bad_geometry() -> None:
    spacing = (1.0, 1.0, 3.0)
    assert automated_qa({}, spacing)["status"] == QA_PENDING
    assert automated_qa({"Kidney": _cube_mesh()}, spacing)["status"] == QA_PASSED
    assert (
        automated_qa({"Kidney": _cube_mesh(volume_error_pct=9.0)}, spacing)["status"] == QA_FAILED
    )
    assert (
        automated_qa({"Kidney": _cube_mesh(surface_to_mask_p95_mm=3.5)}, spacing)["status"]
        == QA_FAILED
    )
    broken = _cube_mesh()
    broken["vertices"][0, 0] = np.nan
    assert automated_qa({"Kidney": broken}, spacing)["status"] == QA_FAILED
    assert automated_qa({"Kidney": None}, spacing)["status"] == QA_FAILED


def test_acceptance_needs_review_all_checks_and_passed_qa() -> None:
    record = ReviewRecord(case_key="abc")
    with pytest.raises(ReviewError):
        transition(record, ACCEPTED, reviewer="Dr A", qa_status=QA_PASSED, checks=ALL_CHECKS)
    with pytest.raises(ReviewError):
        transition(record, REVIEWED, reviewer="  ", qa_status=QA_PASSED, checks=ALL_CHECKS)
    transition(record, REVIEWED, reviewer="Dr A", qa_status=QA_FAILED, checks=ALL_CHECKS)
    assert overall_state(QA_FAILED, record) == REVIEWED
    with pytest.raises(ReviewError, match="QA"):
        transition(record, ACCEPTED, reviewer="Dr A", qa_status=QA_FAILED, checks=ALL_CHECKS)
    partial = {**ALL_CHECKS, "vessels": False}
    with pytest.raises(ReviewError, match="Not yet confirmed"):
        transition(record, ACCEPTED, reviewer="Dr A", qa_status=QA_PASSED, checks=partial)
    transition(record, ACCEPTED, reviewer="Dr A", qa_status=QA_PASSED, checks=ALL_CHECKS)
    assert overall_state(QA_PASSED, record) == ACCEPTED
    # Acceptance does not survive a later QA failure.
    assert overall_state(QA_FAILED, record) == REVIEWED
    assert [item["to"] for item in record.history] == [REVIEWED, ACCEPTED]


def test_reject_and_reopen_need_a_note() -> None:
    record = ReviewRecord(case_key="abc")
    with pytest.raises(ReviewError):
        transition(record, REJECTED, reviewer="Dr A", qa_status=QA_PASSED)
    transition(record, REJECTED, reviewer="Dr A", qa_status=QA_PASSED, notes="tumour missed")
    assert overall_state(QA_PASSED, record) == REJECTED
    transition(record, UNREVIEWED, reviewer="Dr B", qa_status=QA_PASSED, notes="relabelled")
    assert overall_state(QA_PASSED, record) == QA_PASSED
    assert overall_state(QA_PENDING, ReviewRecord(case_key="x")) == "generated"


def test_store_persists_and_reviews_are_bound_to_content(tmp_path) -> None:
    store = ReviewStore(tmp_path / "reviews")
    key = review_case_key("ct-hash", "seg-hash")
    record = store.load(key, "case 1")
    transition(record, REVIEWED, reviewer="Dr A", qa_status=QA_PASSED, checks=ALL_CHECKS)
    store.save(record)
    again = ReviewStore(tmp_path / "reviews").load(key)
    assert again.status == REVIEWED and again.reviewer == "Dr A"
    assert len(again.history) == 1
    changed = store.load(review_case_key("ct-hash", "edited-seg-hash"))
    assert changed.status == UNREVIEWED
    with pytest.raises(ReviewError):
        store.load("../escape")
    memory = ReviewStore(None)
    memory.save(record)
    assert memory.load(key).status == REVIEWED and not memory.persistent


def test_content_hash_detects_replacement_with_restored_timestamp(tmp_path) -> None:
    path = tmp_path / "ct.nii.gz"
    path.write_bytes(b"a" * 1024)
    first = sha256_file(path)
    stat = path.stat()
    path.write_bytes(b"b" * 1024)  # same size
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    assert path.stat().st_mtime_ns == stat.st_mtime_ns
    assert sha256_file(path) != first


class _Case:
    case_id = "case_00001"
    segmentation = np.zeros((2, 2, 2), np.uint8)
    metadata = {"dataset": "KiTS23"}


def _manifest(review=None, resolution=None):
    mesh = _cube_mesh()
    qa = automated_qa({"Tumor": mesh}, (1.0, 1.0, 1.0))
    return build_manifest(
        case_id="case_00001",
        dataset="KiTS23",
        ct_sha256="c" * 64,
        segmentation_sha256="s" * 64,
        image_shape=(2, 2, 2),
        spacing=(1.0, 1.0, 1.0),
        affine=np.eye(4),
        resolution=resolution,
        segmentation=segmentation_source(_Case()),
        reconstruction={"algorithm": "test"},
        mesh_quality={"Tumor": mesh["quality"]},
        qa=qa,
        review=review,
        state=overall_state(qa["status"], None),
    )


def test_manifest_records_sources_review_and_limitations() -> None:
    manifest = _manifest(resolution={"reduced": True, "full_spacing_mm": [0.8, 0.8, 3.0]})
    assert manifest["source"]["ct_sha256"] == "c" * 64
    assert manifest["segmentation"]["type"] == "dataset_reference_annotation"
    assert manifest["review"]["geometry_qa_status"] == QA_PASSED
    assert manifest["review"]["clinician_review_status"] == "unreviewed"
    assert any("reduced demo grid" in item for item in manifest["limitations"])
    assert any("Not reviewed" in item for item in manifest["limitations"])
    assert manifest["reconstruction"]["software"]["commit"]
    json.dumps(manifest)

    corrected = segmentation_source(
        _Case(),
        patient_provenance={"totalsegmentator": {"version": "2"}, "manual_corrections": [{}]},
    )
    assert corrected["type"] == "ai_prediction" and corrected["clinician_corrected"]


def test_stl_zip_and_pdf_carry_manifest_and_status() -> None:
    manifest = _manifest()
    payload = stl_zip({"Tumor": _cube_mesh()}, manifest=manifest)
    archive = zipfile.ZipFile(io.BytesIO(payload))
    assert json.loads(archive.read("manifest.json"))["case_id"] == "case_00001"
    assert "mesh_quality.json" in archive.namelist()
