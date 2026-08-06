from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

from renal_explorer.analysis import analyze_case, region_statistics
from renal_explorer.datasets import KiPA22Adapter, KiTS23Adapter, remap_labels
from renal_explorer.labels import CanonicalLabel
from renal_explorer.models import VolumeCase
from renal_explorer.prediction_packages import (
    PredictionPackageError,
    load_banet_prediction_package,
)
from renal_explorer.remote_data import RemoteDataError, ensure_demo_case, remote_urls
from renal_explorer.rendering import render_view
from renal_explorer.three_d import candidate_vessel_mesh, surface_mesh
from renal_explorer.vessel_ml import (
    VesselModelBundle,
    VesselModelManifest,
    discover_model_bundle,
    predict_vessels,
    resolve_model_bundle,
    sha256_file,
)


def synthetic_case(segmentation: np.ndarray) -> VolumeCase:
    image = np.arange(segmentation.size, dtype=np.float32).reshape(segmentation.shape)
    return VolumeCase(
        dataset_key="test",
        case_id="case_test",
        image=image,
        segmentation=segmentation,
        spacing=(2.0, 1.0, 3.0),
        affine=np.diag([2.0, 1.0, 3.0, 1.0]),
    )


def test_kits_label_mapping() -> None:
    source = np.array([0, 1, 2, 3, 9], dtype=np.int16)
    mapped = remap_labels(source, KiTS23Adapter.label_mapping)
    assert mapped.tolist() == [0, 1, 2, 3, 0]


def test_kipa_label_mapping() -> None:
    source = np.array([0, 1, 2, 3, 4], dtype=np.int16)
    mapped = remap_labels(source, KiPA22Adapter.label_mapping)
    assert mapped.tolist() == [0, 5, 1, 4, 2]


def test_region_volume_uses_physical_spacing() -> None:
    image = np.ones((3, 3, 3), dtype=np.float32) * 40
    mask = np.zeros_like(image, dtype=bool)
    mask[0:2, 0:2, 0:2] = True
    stats = region_statistics(image, mask, (2.0, 1.0, 3.0))
    assert stats["voxels"] == 8
    assert stats["volume_ml"] == 0.048
    assert stats["mean_hu"] == 40.0


def test_components_get_laterality_and_physical_distance() -> None:
    segmentation = np.zeros((12, 8, 6), dtype=np.uint8)
    segmentation[1:4, 2:6, 1:5] = int(CanonicalLabel.KIDNEY)
    segmentation[8:11, 2:6, 1:5] = int(CanonicalLabel.KIDNEY)
    segmentation[9:11, 3:5, 2:4] = int(CanonicalLabel.TUMOR)
    result = analyze_case(synthetic_case(segmentation))
    sides = {component["laterality"] for component in result["kidney_components"]}
    assert sides == {"Left", "Right"}
    assert result["tumor_components"][0]["nearest_kidney"] == "Right"
    assert result["tumor_components"][0]["kidney_centroid_distance_mm"] >= 0


def test_rendering_handles_last_slice_and_anisotropic_spacing() -> None:
    segmentation = np.zeros((10, 9, 8), dtype=np.uint8)
    segmentation[2:8, 2:7, 2:6] = int(CanonicalLabel.KIDNEY)
    case = synthetic_case(segmentation)
    indices = {"Axial": 7, "Coronal": 8, "Sagittal": 9}
    view = render_view(
        image=case.image,
        segmentation=case.segmentation,
        spacing=case.spacing,
        orientation="Axial",
        index=7,
        level=40,
        width=400,
        visible_labels=[CanonicalLabel.KIDNEY],
        overlay_mode="Fill",
        opacity=0.4,
        linked_indices=indices,
    )
    assert view.ndim == 3
    assert view.shape[2] == 3
    assert view.dtype == np.uint8


def test_rendering_keeps_prediction_as_a_separate_overlay_layer() -> None:
    image = np.zeros((8, 8, 8), dtype=np.float32)
    segmentation = np.zeros(image.shape, dtype=np.uint8)
    segmentation[1:7, 1:7, 1:7] = int(CanonicalLabel.KIDNEY)
    prediction = np.zeros(image.shape, dtype=np.uint8)
    prediction[3:5, 3:5, 3:5] = int(CanonicalLabel.RENAL_ARTERY)

    view = render_view(
        image=image,
        segmentation=segmentation,
        spacing=(1.0, 1.0, 1.0),
        orientation="Axial",
        index=4,
        level=40,
        width=400,
        visible_labels=[CanonicalLabel.KIDNEY, CanonicalLabel.RENAL_ARTERY],
        overlay_mode="Fill",
        opacity=1.0,
        supplemental_segmentation=prediction,
    )

    assert np.any(np.all(view == (45, 200, 90), axis=2))
    assert np.any(np.all(view == (255, 55, 55), axis=2))


def test_3d_surface_uses_physical_coordinates() -> None:
    segmentation = np.zeros((10, 9, 8), dtype=np.uint8)
    segmentation[2:6, 2:7, 2:6] = int(CanonicalLabel.KIDNEY)
    case = synthetic_case(segmentation)
    mesh = surface_mesh(
        case.segmentation,
        case.affine,
        case.spacing,
        int(CanonicalLabel.KIDNEY),
        quality="Balanced",
    )

    assert mesh is not None
    assert mesh["vertices"].shape[1] == 3
    assert mesh["faces"].shape[1] == 3
    assert mesh["faces"].shape[0] > 0
    assert np.all(np.isfinite(mesh["vertices"]))
    assert np.ptp(mesh["vertices"][:, 0]) > np.ptp(mesh["vertices"][:, 1])
    assert mesh["quality"]["watertight"] is True
    assert mesh["quality"]["degenerate_triangles"] == 0
    assert mesh["quality"]["volume_error_pct"] <= 6.0
    assert mesh["quality"]["target_spacing_mm"] == 0.75


def test_labeled_vessel_surface_includes_centerline() -> None:
    segmentation = np.zeros((24, 18, 16), dtype=np.uint8)
    segmentation[4:20, 8:11, 7:10] = int(CanonicalLabel.RENAL_ARTERY)
    case = synthetic_case(segmentation)
    mesh = surface_mesh(
        case.segmentation,
        case.affine,
        case.spacing,
        int(CanonicalLabel.RENAL_ARTERY),
        quality="Balanced",
    )

    assert mesh is not None
    assert mesh["centerline_points"] is not None
    assert mesh["centerline_points"].shape[1] == 3


def test_candidate_vessels_require_a_renal_roi() -> None:
    segmentation = np.zeros((12, 12, 12), dtype=np.uint8)
    case = synthetic_case(segmentation)
    mesh = candidate_vessel_mesh(
        case.image,
        case.segmentation,
        case.affine,
        case.spacing,
    )
    assert mesh is None


def _validated_manifest(digest: str) -> VesselModelManifest:
    return VesselModelManifest.from_dict(
        {
            "schema_version": 1,
            "name": "Test KiPA22 model",
            "version": "1.0.0",
            "dataset": "KiPA22 CTA",
            "dataset_url": "https://kipa22.grand-challenge.org/",
            "research_only": True,
            "validated": True,
            "validation_fold": 0,
            "training_cases": 56,
            "target_spacing_mm": [1.0, 1.0, 1.0],
            "patch_size": [16, 16, 16],
            "overlap": 0.5,
            "intensity_clip_hu": [-100.0, 1000.0],
            "canonical_labels": [0, 5, 1, 4, 2],
            "minimum_component_mm3": 1.0,
            "artifact_sha256": digest,
            "validation_metrics": {
                "renal_artery_dice": 0.75,
                "renal_vein_dice": 0.70,
                "mean_vessel_dice": 0.725,
            },
        }
    )


def test_ml_model_bundle_requires_validated_matching_artifact(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"validated model")
    manifest = _validated_manifest(sha256_file(model_path))
    manifest_data = {
        **manifest.__dict__,
        "target_spacing_mm": list(manifest.target_spacing_mm),
        "patch_size": list(manifest.patch_size),
        "intensity_clip_hu": list(manifest.intensity_clip_hu),
        "canonical_labels": list(manifest.canonical_labels),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest_data))

    bundle = discover_model_bundle(tmp_path)

    assert bundle is not None
    assert bundle.manifest.validated is True
    assert bundle.manifest.dataset == "KiPA22 CTA"


def test_ml_vessel_inference_maps_channels_to_artery_and_vein(tmp_path: Path) -> None:
    class Input:
        name = "ct"

    class FakeSession:
        def get_inputs(self):
            return [Input()]

        def run(self, _outputs, feed):
            volume = feed["ct"]
            logits = np.full((1, 5, *volume.shape[2:]), -5.0, dtype=np.float32)
            midpoint = volume.shape[2] // 2
            logits[:, 1, :midpoint] = 5.0
            logits[:, 3, midpoint:] = 5.0
            return [logits]

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"test")
    manifest_path = tmp_path / "manifest.json"
    manifest = _validated_manifest(sha256_file(model_path))
    bundle = VesselModelBundle(model_path, manifest_path, manifest)
    image = np.full((20, 20, 20), 200.0, dtype=np.float32)
    kidney = np.zeros(image.shape, dtype=np.uint8)
    kidney[4:16, 4:16, 4:16] = int(CanonicalLabel.KIDNEY)

    prediction = predict_vessels(
        image,
        (1.0, 1.0, 1.0),
        kidney,
        bundle,
        session=FakeSession(),
    )

    assert np.any(prediction.segmentation == int(CanonicalLabel.RENAL_ARTERY))
    assert np.any(prediction.segmentation == int(CanonicalLabel.RENAL_VEIN))
    assert prediction.provenance["dataset"] == "KiPA22 CTA"
    assert prediction.provenance["mean_vessel_confidence"] > 0.9


def test_remote_ml_bundle_download_is_hash_verified(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"portable validated model")
    manifest = _validated_manifest(sha256_file(source / "model.onnx"))
    manifest_data = {
        **manifest.__dict__,
        "target_spacing_mm": list(manifest.target_spacing_mm),
        "patch_size": list(manifest.patch_size),
        "intensity_clip_hu": list(manifest.intensity_clip_hu),
        "canonical_labels": list(manifest.canonical_labels),
    }
    (source / "manifest.json").write_text(json.dumps(manifest_data))
    monkeypatch.delenv("RENAL_VESSEL_MODEL_DIR", raising=False)
    monkeypatch.setenv("RENAL_VESSEL_MANIFEST_URL", "https://example.test/manifest")
    monkeypatch.setenv("RENAL_VESSEL_MODEL_URL", "https://example.test/model")

    def fake_download(url, destination, maximum_bytes):
        del maximum_bytes
        filename = "manifest.json" if url.endswith("manifest") else "model.onnx"
        shutil.copyfile(source / filename, destination)

    monkeypatch.setattr("renal_explorer.vessel_ml._download_file", fake_download)

    bundle = resolve_model_bundle(cache_dir=cache)

    assert bundle is not None
    assert bundle.model_path.read_bytes() == b"portable validated model"
    assert bundle.manifest.artifact_sha256 == sha256_file(bundle.model_path)


def test_case_directory_discovery(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_00001"
    case_dir.mkdir()
    (case_dir / "imaging.nii.gz").touch()
    cases = KiTS23Adapter().list_cases(tmp_path)
    assert [case.case_id for case in cases] == ["case_00001"]
    assert cases[0].segmentation_path is None


def test_remote_demo_download_is_cached(tmp_path: Path, monkeypatch) -> None:
    class FakeResponse:
        headers = {"Content-Length": "2048"}

        def __init__(self) -> None:
            self.remaining = 2048

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, _size: int) -> bytes:
            if self.remaining == 0:
                return b""
            self.remaining = 0
            return b"0" * 2048

    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    files = ensure_demo_case("case_00015", tmp_path)
    ensure_demo_case("case_00015", tmp_path)

    assert files.image_path.stat().st_size == 2048
    assert files.segmentation_path is not None
    assert files.segmentation_path.stat().st_size == 2048
    assert len(calls) == 2


def test_remote_demo_rejects_arbitrary_case() -> None:
    try:
        remote_urls("case_99999")
    except RemoteDataError:
        pass
    else:
        raise AssertionError("An arbitrary remote case should not be accepted")


def _banet_package(
    source: np.ndarray,
    affine: np.ndarray,
    case_id: str = "case_test",
    image_digest: str | None = None,
) -> tuple[bytes, str]:
    digest = image_digest or hashlib.sha256(b"source CT").hexdigest()
    prediction_bytes = gzip.compress(
        nib.Nifti1Image(source.astype(np.uint8), affine).to_bytes()
    )
    provenance = {
        "schema_version": 1,
        "case_id": case_id,
        "source_image_sha256": digest,
        "source_image_shape": list(source.shape),
        "source_image_affine": affine.tolist(),
        "model": {
            "name": "BA-Net",
            "fold": 0,
            "weights_doi": "10.5281/zenodo.7030423",
        },
        "research_only": True,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("vessels.nii.gz", prediction_bytes)
        archive.writestr("provenance.json", json.dumps(provenance))
    return buffer.getvalue(), digest


def test_banet_prediction_package_is_identity_checked_and_mapped() -> None:
    affine = np.diag([1.2, 1.1, 2.0, 1.0])
    source = np.zeros((8, 7, 6), dtype=np.uint8)
    source[1:3, 2:5, 2:4] = 1  # KiPA renal vein
    source[4:7, 2:4, 1:5] = 3  # KiPA renal artery
    payload, image_digest = _banet_package(source, affine)

    imported = load_banet_prediction_package(
        payload,
        "case_test",
        source.shape,
        affine,
        image_digest,
    )

    assert np.count_nonzero(imported.segmentation == CanonicalLabel.RENAL_VEIN) == 12
    assert np.count_nonzero(imported.segmentation == CanonicalLabel.RENAL_ARTERY) == 24
    assert imported.provenance["vein_voxels"] == 12
    assert imported.provenance["artery_voxels"] == 24
    assert len(imported.package_sha256) == 64


def test_banet_prediction_package_rejects_wrong_source_ct() -> None:
    affine = np.eye(4)
    source = np.zeros((4, 4, 4), dtype=np.uint8)
    payload, _digest = _banet_package(source, affine)

    try:
        load_banet_prediction_package(
            payload,
            "case_test",
            source.shape,
            affine,
            hashlib.sha256(b"another CT").hexdigest(),
        )
    except PredictionPackageError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("A prediction for another CT should be rejected")
