"""Export processed patient cases as an nnU-Net v2 raw dataset for training.

Example::

    python scripts/export_nnunet_dataset.py --dataset-id 501 \
        --labels KIDNEY TUMOR CYST COLLECTING_SYSTEM --corrected-only \
        --output ~/RenalPlanner/nnUNet_raw

Labels are renumbered 1..N in the order given. Only cases whose segmentation was manually
corrected are exported with ``--corrected-only`` (recommended: automatic labels must not
become training ground truth unreviewed).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renal_explorer.labels import CanonicalLabel  # noqa: E402
from renal_explorer.patient.store import PatientStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--name", default="RenalPlanner")
    parser.add_argument(
        "--labels", nargs="+", required=True, choices=[label.name for label in CanonicalLabel][1:]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=None)
    parser.add_argument("--corrected-only", action="store_true")
    parser.add_argument("--require-label", default=None, help="Skip cases without this label")
    args = parser.parse_args()

    import nibabel as nib
    import numpy as np

    store = PatientStore(args.store)
    folder = args.output.expanduser() / f"Dataset{args.dataset_id:03d}_{args.name}"
    images = folder / "imagesTr"
    labels_dir = folder / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    mapping = {CanonicalLabel[name]: index for index, name in enumerate(args.labels, start=1)}

    exported = 0
    for case in store.ready_cases():
        if args.corrected_only and not case.provenance().get("manual_corrections"):
            continue
        segmentation = nib.load(str(case.segmentation_path))
        source = np.asanyarray(segmentation.dataobj)
        if args.require_label and not np.any(source == int(CanonicalLabel[args.require_label])):
            continue
        target = np.zeros(source.shape, dtype=np.uint8)
        for label, index in mapping.items():
            target[source == int(label)] = index
        shutil.copy2(case.image_path, images / f"{case.case_id}_0000.nii.gz")
        nib.save(
            nib.Nifti1Image(target, segmentation.affine),
            str(labels_dir / f"{case.case_id}.nii.gz"),
        )
        exported += 1

    (folder / "dataset.json").write_text(
        json.dumps(
            {
                "channel_names": {"0": "CT"},
                "labels": {
                    "background": 0,
                    **{label.name.lower(): index for label, index in mapping.items()},
                },
                "numTraining": exported,
                "file_ending": ".nii.gz",
                "description": (
                    "Pseudonymised renal planner cases; research use under ethics approval."
                ),
            },
            indent=2,
        )
    )
    print(f"Exported {exported} cases to {folder}")
    return 0 if exported else 1


if __name__ == "__main__":
    raise SystemExit(main())
