from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from renal_explorer.labels import CanonicalLabel
from renal_explorer.patient.dicom_intake import (
    PRIVATE_INDEX,
    DicomIntakeError,
    SeriesInfo,
    consume_inbox_source,
    default_assignment,
    guess_phase,
    main_phase,
    read_series_index,
    record_inbox_source,
    safe_extract_zip,
    write_series_index,
)
from renal_explorer.patient.fusion import fuse
from renal_explorer.patient.segmenters import (
    LESION_NAME,
    TUMOR_NAME,
    collecting_system_from_excretory,
    discover_nnunet_model,
    map_totalseg_labels,
    nnunet_label_ids,
)
from renal_explorer.patient.store import PatientStore
from renal_explorer.patient.worker import reconcile


@pytest.mark.parametrize(
    ("description", "phase"),
    [
        ("Abdomen arteriell 1.0 Br40", "arterial"),
        ("CTA Nierenarterien", "arterial"),
        ("Abdomen portalvenös 1.0", "nephrographic"),
        ("Nephrographic 100s", "nephrographic"),
        ("Spätphase 10 min", "excretory"),
        ("Urographie Ausscheidungsphase", "excretory"),
        ("Delayed excretory", "excretory"),
        ("Nativ 5.0", "non-contrast"),
        ("Topogram", "other"),
    ],
)
def test_guess_phase_english_and_german(description: str, phase: str) -> None:
    assert guess_phase(description) == phase


def _series(uid, phase, thickness, slices):
    return SeriesInfo(uid, phase, slices, thickness, "120000", phase, [])


def test_default_assignment_prefers_thin_slices_and_main_phase() -> None:
    series = [
        _series("a5", "arterial", 5.0, 80),
        _series("a1", "arterial", 1.0, 400),
        _series("n1", "nephrographic", 1.0, 400),
        _series("o", "other", 1.0, 400),
    ]
    assignment = default_assignment(series)
    assert assignment == {"arterial": "a1", "nephrographic": "n1"}
    assert main_phase(assignment) == "nephrographic"
    assert main_phase({"arterial": "a1"}) == "arterial"


def test_safe_extract_blocks_traversal(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("study/IM0001", b"dicom")
        archive.writestr("../../escape.txt", b"bad")
        archive.writestr("/abs.txt", b"bad")
        archive.writestr("__MACOSX/._IM0001", b"junk")
    archive_path = tmp_path / "upload.zip"
    archive_path.write_bytes(buffer.getvalue())
    destination = tmp_path / "raw"
    assert safe_extract_zip(archive_path, destination) == 1
    assert not (tmp_path / "escape.txt").exists()
    assert [path.name for path in destination.iterdir()] == ["000000_IM0001"]

    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w"):
        pass
    with pytest.raises(DicomIntakeError):
        safe_extract_zip(empty, tmp_path / "raw2")


def test_store_lifecycle_and_retention(tmp_path: Path) -> None:
    store = PatientStore(tmp_path)
    case = store.create_case("Fall 7 / <script>")
    assert case.case_id.startswith("PN-") and len(case.case_id) == 11
    assert case.status()["state"] == "uploaded"
    assert case.status()["alias"] == "Fall 7  script"
    assert not case.is_ready()
    case.write_request({"assignment": {"nephrographic": "x"}})
    assert case.request()["assignment"] == {"nephrographic": "x"}
    with pytest.raises(KeyError):
        store.get("../etc")

    case.update_status(state="queued", pid=None)
    reconcile(store)
    assert case.status()["state"] == "queued"  # just launched: grace period, no PID yet
    case.update_status(state="running", pid=999_999_999)
    case.raw_dir.mkdir()
    (case.raw_dir / "IM0001").write_bytes(b"dicom")
    reconcile(store)
    assert case.status()["state"] == "failed"
    assert not case.raw_dir.exists()  # regression (review): dead worker kept raw input

    case.update_status(created=0)
    assert store.purge_expired(retention_days=1) == [case.case_id]
    assert store.cases() == []


def test_map_totalseg_labels_merges_sides() -> None:
    class_map = {1: "spleen", 2: "kidney_right", 3: "kidney_left", 52: "aorta", 99: "brain"}
    labels = np.array([[[0, 1, 2, 3, 52, 99]]])
    masks = map_totalseg_labels(labels, class_map)
    assert set(masks) == {"SPLEEN", "KIDNEY", "AORTA"}
    assert masks["KIDNEY"].sum() == 2


def test_nnunet_label_ids_region_based_and_discovery(tmp_path: Path) -> None:
    dataset = {
        "labels": {
            "background": 0,
            "liver_lesion": [5, 6],
            "kidney_lesion": [7, 8, 9],
            "kidney_cyst": 10,
            "pancreatic_lesion": [10, 11, 12, 13],
        }
    }
    assert nnunet_label_ids(dataset, TUMOR_NAME, LESION_NAME) == {7, 8, 9}

    model = tmp_path / "AbdomenAtlasNetTumors" / "Dataset1__nnUNetTrainer__nnUNetPlans__3d_fullres"
    (model / "fold_0").mkdir(parents=True)
    (model / "fold_0" / "checkpoint_final.pth").write_bytes(b"x")
    (model / "dataset.json").write_text(json.dumps(dataset))
    (model / "plans.json").write_text("{}")
    found = discover_nnunet_model(tmp_path)
    assert found is not None and found[0] == model and found[1] == ["0"]
    assert discover_nnunet_model(tmp_path / "missing") is None


def _kidney_phantom():
    shape = (80, 80, 60)
    x, y, z = np.meshgrid(*[np.arange(size, dtype=float) for size in shape], indexing="ij")
    kidney = ((x - 40) / 20) ** 2 + ((y - 40) / 15) ** 2 + ((z - 30) / 25) ** 2 <= 1
    sinus = ((x - 46) / 7) ** 2 + ((y - 40) / 5) ** 2 + ((z - 30) / 12) ** 2 <= 1
    channel = (x >= 46) & (np.abs(y - 40) <= 5) & (np.abs(z - 30) <= 8)
    kidney &= ~(sinus | channel)
    return shape, kidney, sinus


def test_collecting_system_keeps_sinus_contrast_rejects_rib() -> None:
    shape, kidney, sinus = _kidney_phantom()
    excretory = np.full(shape, 30.0, dtype=np.float32)
    excretory[kidney] = 180.0
    excretory[sinus] = 900.0  # contrast-filled pelvis/calyces
    rib = np.zeros(shape, bool)
    rib[14:18, 20:60, 50:53] = True  # bright bone grazing the upper pole
    excretory[rib] = 1200.0
    collecting = collecting_system_from_excretory(excretory, kidney, (1.0, 1.0, 1.0))
    assert collecting[sinus].mean() > 0.9
    assert not collecting[rib].any()


def test_fusion_priority_and_restriction() -> None:
    shape, kidney, _sinus = _kidney_phantom()
    tumor = np.zeros(shape, bool)
    tumor[18:26, 36:44, 26:34] = True  # touches the lateral kidney edge
    far_tumor = np.zeros(shape, bool)
    far_tumor[70:75, 5:10, 5:10] = True  # false positive far away
    liver = np.zeros(shape, bool)
    liver[:, :, 50:] = True
    labels, counts, notes = fuse(
        {
            CanonicalLabel.KIDNEY: kidney,
            CanonicalLabel.TUMOR: tumor | far_tumor,
            CanonicalLabel.LIVER: liver,
        },
        (1.0, 1.0, 1.0),
    )
    assert labels.dtype == np.uint8
    assert (labels[tumor] == int(CanonicalLabel.TUMOR)).all()
    assert not (labels[far_tumor] == int(CanonicalLabel.TUMOR)).any()
    assert (labels[kidney & ~tumor & ~liver] == int(CanonicalLabel.KIDNEY)).all()
    assert counts["LIVER"] > 0
    assert any("tumor" in note for note in notes)


def test_ras_round_trip_matches_nibabel_canonical() -> None:
    nib = pytest.importorskip("nibabel")
    if not hasattr(nib, "Nifti1Image"):
        pytest.skip("nibabel stub")
    from renal_explorer.patient.pipeline import from_ras, to_ras

    affine = np.array(
        [[0.0, 0.0, -3.0, 10.0], [-0.8, 0.0, 0.0, 5.0], [0.0, 0.7, 0.0, -2.0], [0, 0, 0, 1]]
    )
    data = np.random.default_rng(0).integers(0, 9, size=(6, 7, 8)).astype(np.uint8)
    canonical = nib.as_closest_canonical(nib.Nifti1Image(data, affine))
    ras, spacing = to_ras(data, affine)
    assert np.array_equal(ras, np.asanyarray(canonical.dataobj))
    assert np.allclose(spacing, canonical.header.get_zooms()[:3])
    assert np.array_equal(from_ras(ras, affine), data)


def test_pipeline_end_to_end_with_fake_models(tmp_path: Path, monkeypatch) -> None:
    nib = pytest.importorskip("nibabel")
    if not hasattr(nib, "Nifti1Image"):
        pytest.skip("nibabel stub")
    from renal_explorer.patient import pipeline
    from renal_explorer.patient.dicom_intake import SeriesInfo

    shape, kidney, sinus = _kidney_phantom()
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    ct = np.full(shape, -100, np.int16)
    ct[kidney] = 160
    tumor = np.zeros(shape, bool)
    tumor[18:26, 36:44, 26:34] = True
    excretory = ct.copy()
    excretory[sinus] = 900

    store = PatientStore(tmp_path)
    case = store.create_case("e2e")
    case.raw_dir.mkdir()
    (case.raw_dir / "x.dcm").write_bytes(b"0")
    case.write_request(
        {"assignment": {"nephrographic": "n", "excretory": "e"}, "main_phase": "nephrographic"}
    )
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    study = inbox / "study"
    study.mkdir()
    record_inbox_source(case.raw_dir, study)
    monkeypatch.setenv("RENAL_INBOX", str(inbox))

    volumes = {"n": ct, "e": excretory}
    monkeypatch.setattr(
        pipeline,
        "list_series",
        lambda folder: [SeriesInfo(uid, uid, 60, 1.0, "", "", [uid]) for uid in volumes],
    )

    def fake_convert(files, output_path):
        nib.save(nib.Nifti1Image(volumes[files[0]], affine), str(output_path))
        return {"size": list(shape), "spacing_mm": [1.0, 1.0, 1.0]}

    def fake_register(fixed, moving, output, roi, roi_affine):
        output.write_bytes(Path(moving).read_bytes())
        return {"method": "identity (test)"}

    totalseg = np.zeros(shape, np.uint16)
    totalseg[kidney] = 3  # kidney_left in the fake class map
    monkeypatch.setattr(pipeline, "convert_series", fake_convert)
    monkeypatch.setattr(pipeline, "register_phase", fake_register)
    monkeypatch.setattr(
        pipeline, "run_totalsegmentator", lambda *a: (totalseg, {"backend": "fake"})
    )
    monkeypatch.setattr(pipeline, "totalseg_class_map", lambda: {3: "kidney_left"})
    monkeypatch.setattr(pipeline, "run_kidney_cysts", lambda *a: None)
    monkeypatch.setattr(
        pipeline,
        "run_atlasnet_tumors",
        lambda *a: (tumor, np.zeros(shape, bool), {"model": "fake"}),
    )

    pipeline.run_and_record(case)
    status = case.status()
    assert status["state"] == "done", status
    assert not study.exists()  # consumed only after a successful conversion
    assert not case.raw_dir.exists()
    labels = np.asanyarray(nib.load(str(case.segmentation_path)).dataobj)
    assert (labels[tumor] == int(CanonicalLabel.TUMOR)).all()
    assert (labels[sinus] == int(CanonicalLabel.COLLECTING_SYSTEM)).mean() > 0.9
    provenance = case.provenance()
    assert provenance["collecting_system"]["voxels"] > 0
    assert any("arterial" in warning.lower() for warning in provenance["warnings"])
    assert store.ready_cases() == [case]


def test_series_identifiers_stay_in_raw_only(tmp_path: Path) -> None:
    """Regression (review): descriptions/UIDs must not be persisted outside raw/."""

    store = PatientStore(tmp_path / "store")
    case = store.create_case()
    case.raw_dir.mkdir()
    marker = "MUSTERMANN^MAX arteriell"
    series = [SeriesInfo("1.2.840.99.1", marker, 300, 1.0, "101010", "arterial", ["x"])]
    public = write_series_index(case.raw_dir, series)
    case.write_request({"series": public})
    stored = (case.path / "request.json").read_text()
    assert marker not in stored and "1.2.840" not in stored
    assert public == [
        {"handle": "S1", "slices": 300, "slice_thickness_mm": 1.0, "phase_guess": "arterial"}
    ]
    assert read_series_index(case.raw_dir)["S1"]["description"] == marker
    assert (case.raw_dir / PRIVATE_INDEX).exists()
    case.discard_raw()
    assert not any(marker in path.read_text() for path in case.path.rglob("*") if path.is_file())


def test_inbox_source_deleted_only_when_consumed_and_inside_inbox(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    study = inbox / "study.zip"
    study.write_bytes(b"zip")
    outside = tmp_path / "elsewhere.zip"
    outside.write_bytes(b"zip")
    raw = tmp_path / "raw"
    raw.mkdir()

    assert consume_inbox_source(raw, inbox) is False  # nothing recorded: nothing deleted
    record_inbox_source(raw, outside)
    assert consume_inbox_source(raw, inbox) is False and outside.exists()
    record_inbox_source(raw, study)
    assert consume_inbox_source(raw, inbox) is True and not study.exists()


def test_failed_conversion_keeps_inbox_study(tmp_path: Path, monkeypatch) -> None:
    """Regression (review): an import/conversion failure must not delete the original."""

    from renal_explorer.patient import pipeline
    from renal_explorer.patient.dicom_intake import SeriesInfo as Info

    store = PatientStore(tmp_path / "store")
    case = store.create_case()
    case.raw_dir.mkdir()
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    study = inbox / "study.zip"
    study.write_bytes(b"zip")
    record_inbox_source(case.raw_dir, study)
    case.write_request({"assignment": {"nephrographic": "n"}, "main_phase": "nephrographic"})
    monkeypatch.setenv("RENAL_INBOX", str(inbox))
    monkeypatch.setattr(
        pipeline, "list_series", lambda folder: [Info("n", "", 60, 1.0, "", "", [])]
    )

    def broken(files, output_path):
        raise RuntimeError("corrupt series")

    monkeypatch.setattr(pipeline, "convert_series", broken)
    pipeline.run_and_record(case)
    assert case.status()["state"] == "failed"
    assert study.exists()
    assert not case.raw_dir.exists()


def test_inbox_item_replaced_under_same_name_is_not_deleted(tmp_path: Path) -> None:
    """Regression (re-review): a new study reusing the imported pathname was deleted."""

    import os

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    study = inbox / "study.zip"
    study.write_bytes(b"first study")
    record_inbox_source(raw, study)
    study.unlink()
    study.write_bytes(b"second, different study")
    os.utime(study, ns=(1, 1))
    assert consume_inbox_source(raw, inbox) is False
    assert study.read_bytes() == b"second, different study"

    folder = inbox / "folder_study"
    folder.mkdir()
    (folder / "IM1").write_bytes(b"a")
    record_inbox_source(raw, folder)
    (folder / "IM2").write_bytes(b"added later")
    assert consume_inbox_source(raw, inbox) is False and folder.exists()

    untouched = inbox / "untouched_study"
    untouched.mkdir()
    (untouched / "IM1").write_bytes(b"a")
    record_inbox_source(raw, untouched)
    assert consume_inbox_source(raw, inbox) is True and not untouched.exists()


def test_inbox_change_during_import_aborts_without_deleting(tmp_path: Path, monkeypatch) -> None:
    """Regression (third review): study replaced while being copied."""

    import renal_explorer.patient_ui as patient_ui

    monkeypatch.setenv("RENAL_PATIENT_ROOT", str(tmp_path / "store"))
    store = PatientStore(tmp_path / "store")
    case = store.create_case()
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    study = inbox / "study"
    study.mkdir()
    (study / "IM1").write_bytes(b"study A")
    real_copytree = patient_ui.shutil.copytree

    def copy_then_replace(source, target, **kwargs):
        result = real_copytree(source, target, **kwargs)
        (study / "IM1").write_bytes(b"study B, different and longer")
        return result

    monkeypatch.setattr(patient_ui.shutil, "copytree", copy_then_replace)
    assert patient_ui._ingest(case, study, study) is False
    assert case.status()["state"] == "failed"
    assert study.exists() and not case.raw_dir.exists()
