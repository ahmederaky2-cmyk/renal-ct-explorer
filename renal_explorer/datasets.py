from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from .labels import CanonicalLabel
from .models import CaseFiles, VolumeCase


class DatasetError(RuntimeError):
    pass


def remap_labels(source: np.ndarray, mapping: Mapping[int, CanonicalLabel]) -> np.ndarray:
    """Map a dataset mask into the shared label vocabulary."""

    result = np.zeros(source.shape, dtype=np.uint8)
    for source_label, canonical_label in mapping.items():
        if source_label == 0:
            continue
        result[source == source_label] = int(canonical_label)
    return result


@dataclass(frozen=True)
class DatasetInfo:
    key: str
    name: str
    source_url: str
    license_name: str
    description: str
    expected_layout: str


class DatasetAdapter:
    info: DatasetInfo
    label_mapping: Mapping[int, CanonicalLabel]

    def list_cases(self, root: Path) -> list[CaseFiles]:
        raise NotImplementedError

    def load_case(self, files: CaseFiles) -> VolumeCase:
        image_nii = nib.as_closest_canonical(nib.load(str(files.image_path)))
        image = image_nii.get_fdata(dtype=np.float32)
        if image.ndim != 3:
            raise DatasetError(f"Expected a 3D image, got shape {image.shape}.")
        if not np.all(np.isfinite(image)):
            raise DatasetError("The image contains NaN or infinite values.")

        segmentation = None
        source_labels: tuple[int, ...] = ()
        if files.segmentation_path is not None:
            seg_nii = nib.as_closest_canonical(nib.load(str(files.segmentation_path)))
            if seg_nii.shape != image_nii.shape:
                raise DatasetError(
                    f"Image and segmentation shapes differ: {image_nii.shape} vs {seg_nii.shape}."
                )
            if not np.allclose(image_nii.affine, seg_nii.affine, atol=1e-3):
                raise DatasetError("Image and segmentation affine matrices do not match.")
            source = np.asanyarray(seg_nii.dataobj).astype(np.int16, copy=False)
            source_labels = tuple(int(value) for value in np.unique(source))
            segmentation = remap_labels(source, self.label_mapping)

        spacing = tuple(float(value) for value in image_nii.header.get_zooms()[:3])
        return VolumeCase(
            dataset_key=self.info.key,
            case_id=files.case_id,
            image=image,
            segmentation=segmentation,
            spacing=spacing,
            affine=np.asarray(image_nii.affine, dtype=float),
            source_labels=source_labels,
            metadata={
                "image_file": files.image_path.name,
                "segmentation_file": (
                    files.segmentation_path.name if files.segmentation_path else None
                ),
            },
        )


class CaseDirectoryAdapter(DatasetAdapter):
    image_names: tuple[str, ...] = ("imaging.nii.gz", "ct.nii.gz")
    segmentation_names: tuple[str, ...] = (
        "segmentation.nii.gz",
        "combined_labels.nii.gz",
        "label.nii.gz",
    )

    def list_cases(self, root: Path) -> list[CaseFiles]:
        if not root.is_dir():
            return []
        cases: list[CaseFiles] = []
        for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            image_path = _first_existing(case_dir, self.image_names)
            if image_path is None:
                continue
            segmentation_path = _first_existing(case_dir, self.segmentation_names)
            cases.append(CaseFiles(case_dir.name, image_path, segmentation_path))
        return cases


class SplitDirectoryAdapter(DatasetAdapter):
    image_dir_names: tuple[str, ...] = ("imagesTr", "images")
    label_dir_names: tuple[str, ...] = ("labelsTr", "labels")
    strip_image_suffix = False

    def list_cases(self, root: Path) -> list[CaseFiles]:
        image_dir = _first_directory(root, self.image_dir_names)
        if image_dir is None:
            return []
        label_dir = _first_directory(root, self.label_dir_names)
        cases: list[CaseFiles] = []
        for image_path in sorted(image_dir.glob("*.nii.gz")):
            case_id = image_path.name.removesuffix(".nii.gz")
            label_stem = case_id.removesuffix("_0000") if self.strip_image_suffix else case_id
            segmentation_path = label_dir / f"{label_stem}.nii.gz" if label_dir else None
            if segmentation_path is not None and not segmentation_path.exists():
                segmentation_path = None
            cases.append(CaseFiles(case_id, image_path, segmentation_path))
        return cases


def _first_existing(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def _first_directory(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


class KiTS23Adapter(CaseDirectoryAdapter):
    info = DatasetInfo(
        key="kits23",
        name="KiTS23",
        source_url="https://github.com/neheller/kits23",
        license_name="CC BY-NC-SA 4.0",
        description="Kidney, tumor, and cyst CT segmentations.",
        expected_layout="case_xxxxx/{imaging.nii.gz, segmentation.nii.gz}",
    )
    label_mapping = {
        1: CanonicalLabel.KIDNEY,
        2: CanonicalLabel.TUMOR,
        3: CanonicalLabel.CYST,
    }


class KiPA22Adapter(CaseDirectoryAdapter):
    info = DatasetInfo(
        key="kipa22",
        name="KiPA22",
        source_url="https://kipa22.grand-challenge.org/",
        license_name="CC BY-NC-ND + participant agreement",
        description="CTA with kidney, tumor, renal artery, and renal vein labels.",
        expected_layout="case_xxxxx/{imaging.nii.gz, segmentation.nii.gz}",
    )
    label_mapping = {
        1: CanonicalLabel.RENAL_VEIN,
        2: CanonicalLabel.KIDNEY,
        3: CanonicalLabel.RENAL_ARTERY,
        4: CanonicalLabel.TUMOR,
    }


class C4KCKiTSAdapter(CaseDirectoryAdapter):
    info = DatasetInfo(
        key="c4kc_kits",
        name="C4KC-KiTS19 (converted NIfTI)",
        source_url="https://www.cancerimagingarchive.net/collection/c4kc-kits/",
        license_name="CC BY 3.0",
        description="DICOM-origin KiTS19 imaging, segmentations, and clinical data.",
        expected_layout=(
            "Convert each selected DICOM series and SEG to "
            "case_xxxxx/{imaging.nii.gz, segmentation.nii.gz}."
        ),
    )
    label_mapping = {1: CanonicalLabel.KIDNEY, 2: CanonicalLabel.TUMOR}


class AbdomenAtlasAdapter(CaseDirectoryAdapter):
    info = DatasetInfo(
        key="abdomen_atlas",
        name="AbdomenAtlas 1.x",
        source_url="https://github.com/MrGiovanni/AbdomenAtlas",
        license_name="Check release-specific terms",
        description="Large multi-center abdominal CT anatomy dataset.",
        expected_layout="case_folder/{ct.nii.gz, combined_labels.nii.gz}",
    )
    label_mapping = {
        1: CanonicalLabel.AORTA,
        3: CanonicalLabel.KIDNEY,
        4: CanonicalLabel.KIDNEY,
        7: CanonicalLabel.IVC,
    }


class AMOS22Adapter(SplitDirectoryAdapter):
    info = DatasetInfo(
        key="amos22",
        name="AMOS22",
        source_url="https://amos22.grand-challenge.org/",
        license_name="Challenge terms apply",
        description="Multi-center CT/MRI with 15 abdominal organ labels.",
        expected_layout="{imagesTr, labelsTr}/amos_xxxx.nii.gz",
    )
    label_mapping = {
        2: CanonicalLabel.KIDNEY,
        3: CanonicalLabel.KIDNEY,
        8: CanonicalLabel.AORTA,
        9: CanonicalLabel.IVC,
    }


class FLARE22Adapter(SplitDirectoryAdapter):
    strip_image_suffix = True
    info = DatasetInfo(
        key="flare22",
        name="FLARE22",
        source_url="https://flare22.grand-challenge.org/",
        license_name="Check release-specific terms",
        description="Diverse abdominal CT cohort for organ segmentation and generalization.",
        expected_layout="imagesTr/name_0000.nii.gz and labelsTr/name.nii.gz",
    )
    label_mapping = {
        2: CanonicalLabel.KIDNEY,
        5: CanonicalLabel.AORTA,
        6: CanonicalLabel.IVC,
        13: CanonicalLabel.KIDNEY,
    }


class PatientCasesAdapter(CaseDirectoryAdapter):
    """Cases produced by ``renal_explorer.patient`` (labels already canonical)."""

    info = DatasetInfo(
        key="patient",
        name="Patient cases (private)",
        source_url="https://github.com/ahmederaky2-cmyk/renal-ct-explorer",
        license_name="Institutional data — do not redistribute",
        description=(
            "Pseudonymised cases uploaded as DICOM and segmented automatically on this server. "
            "Automatic labels require review before use."
        ),
        expected_layout="PN-xxxxxxxx/imaging.nii.gz + segmentation.nii.gz",
    )
    label_mapping = {int(label): label for label in CanonicalLabel if int(label) != 0}

    def list_cases(self, root: Path) -> list[CaseFiles]:
        return [
            case
            for case in super().list_cases(root)
            if case.case_id.startswith("PN-") and case.segmentation_path is not None
        ]


DATASETS: dict[str, type[DatasetAdapter]] = {
    adapter.info.key: adapter
    for adapter in (
        KiTS23Adapter,
        KiPA22Adapter,
        C4KCKiTSAdapter,
        AbdomenAtlasAdapter,
        AMOS22Adapter,
        FLARE22Adapter,
        PatientCasesAdapter,
    )
}


def get_adapter(dataset_key: str) -> DatasetAdapter:
    try:
        return DATASETS[dataset_key]()
    except KeyError as exc:
        raise DatasetError(f"Unsupported dataset: {dataset_key}") from exc
