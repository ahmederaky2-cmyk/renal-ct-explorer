from pathlib import Path

import numpy as np

from renal_explorer.analysis import analyze_case, region_statistics
from renal_explorer.datasets import KiPA22Adapter, KiTS23Adapter, remap_labels
from renal_explorer.labels import CanonicalLabel
from renal_explorer.models import VolumeCase
from renal_explorer.remote_data import RemoteDataError, ensure_demo_case, remote_urls
from renal_explorer.rendering import render_view


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
