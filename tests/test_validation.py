from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from renal_explorer.labels import CanonicalLabel
from renal_explorer.validation import (
    agreement,
    dice,
    evaluate_segmentation,
    lesion_detection,
    main_tumors_match,
    renal_complexity,
    weighted_kappa,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "validation" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_dice_and_detection() -> None:
    first = np.zeros((20, 20, 20), bool)
    second = np.zeros_like(first)
    assert dice(first, second) is None
    first[2:6, 2:6, 2:6] = True
    second[2:6, 2:6, 4:8] = True
    assert dice(first, second) == pytest.approx(0.5)

    truth = np.zeros((40, 40, 40), bool)
    truth[2:8, 2:8, 2:8] = True  # found
    truth[20:24, 20:24, 20:24] = True  # missed
    prediction = np.zeros_like(truth)
    prediction[3:5, 3:5, 3:5] = True
    prediction[30:36, 30:36, 30:36] = True  # false positive
    result = lesion_detection(truth, prediction, (1.0, 1.0, 1.0), min_volume_ml=0.01)
    assert (result["true_lesions"], result["found"], result["false_positive_lesions"]) == (2, 1, 1)


def test_evaluate_segmentation_uses_renal_mass() -> None:
    truth = np.zeros((30, 30, 30), np.uint8)
    truth[5:25, 5:25, 5:25] = int(CanonicalLabel.KIDNEY)
    truth[5:12, 5:12, 5:12] = int(CanonicalLabel.TUMOR)
    prediction = truth.copy()
    prediction[prediction == int(CanonicalLabel.TUMOR)] = int(CanonicalLabel.KIDNEY)
    metrics = evaluate_segmentation(truth, prediction, (1.0, 1.0, 1.0))
    assert metrics["renal_mass_dice"] == pytest.approx(1.0)
    assert metrics["tumor_dice"] == pytest.approx(0.0)
    assert metrics["found"] == 0 and metrics["missed_lesion_diameters_mm"]
    assert not main_tumors_match(truth, prediction)


def test_weighted_kappa_and_agreement() -> None:
    perfect = [1, 2, 3, 1, 2, 3]
    assert weighted_kappa(perfect, perfect, [1, 2, 3]) == pytest.approx(1.0)
    off_by_one = [2, 3, 3, 2, 3, 3]
    off_by_two = [3, 3, 1, 3, 1, 1]
    assert weighted_kappa(perfect, off_by_one, [1, 2, 3]) > weighted_kappa(
        perfect, off_by_two, [1, 2, 3]
    )
    result = agreement([6, 8, 10, None], [7, 8, 9, 5], list(range(4, 13)))
    assert result["n"] == 3
    assert result["exact_agreement"] == pytest.approx(1 / 3)
    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["mean_absolute_difference"] == pytest.approx(2 / 3)
    assert renal_complexity(6) == "Low" and renal_complexity(9) == "Intermediate"
    assert renal_complexity(10) == "High" and renal_complexity(None) is None


def test_compare_scores_reads_sheet_export(tmp_path: Path) -> None:
    compare_scores = _load("compare_scores")
    manual = tmp_path / "manual.csv"
    manual.write_text(
        "Case,R,E,N,A,L,h,Total,Notes\n"
        "case_00001,1,2,3,a,2,no,8,\n"
        "case_00005,'3,'1,'1,P,'3,yes,8,big\n"
        "case_00012,,,,,,,,\n"
    )
    planner = tmp_path / "results.csv"
    with open(planner, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", *[f"{p}_{i}" for p in ("expert", "auto") for i in "RENALh"],
                        "expert_renal_total", "auto_renal_total"],
        )
        writer.writeheader()
        writer.writerow(
            {"case": "case_00001", "expert_R": 1, "expert_E": 2, "expert_N": 3, "expert_A": "a",
             "expert_L": 2, "expert_h": "False", "expert_renal_total": 8,
             "auto_R": 1, "auto_E": 1, "auto_N": 3, "auto_A": "a", "auto_L": 2,
             "auto_h": "False", "auto_renal_total": 7}
        )
        writer.writerow(
            {"case": "case_00005", "expert_R": 3, "expert_E": 1, "expert_N": 2, "expert_A": "p",
             "expert_L": 3, "expert_h": "True", "expert_renal_total": 9,
             "auto_R": 3, "auto_E": 1, "auto_N": 1, "auto_A": "x", "auto_L": 3,
             "auto_h": "True", "auto_renal_total": 8}
        )
    scores = compare_scores.read_manual(manual)
    assert set(scores) == {"case_00001", "case_00005"}
    assert scores["case_00005"]["A"] == "p" and scores["case_00005"]["h"] is True
    results = {
        prefix: compare_scores.compare(scores, compare_scores.read_planner(planner), prefix)
        for prefix in ("expert", "auto")
    }
    assert results["expert"]["R"]["exact_agreement"] == 1.0
    assert results["expert"]["total"]["mean_difference"] == pytest.approx(0.5)
    assert results["auto"]["A"]["exact_agreement"] == 0.5
    assert "Cases scored by both: 2" in compare_scores.report(results)
    assert results["auto"]["large_disagreements"] == []


def test_run_validation_end_to_end_with_fake_model(tmp_path: Path, monkeypatch) -> None:
    nib = pytest.importorskip("nibabel")
    if not hasattr(nib, "Nifti1Image"):
        pytest.skip("nibabel stub")
    sys.path.insert(0, str(ROOT / "tests"))
    from test_planning import CENTER_MM, SPACING, phantom

    run_validation = _load("run_validation")
    root = tmp_path / "kits"
    for index, offset in enumerate((0.0, 3.0)):
        segmentation, image, affine = phantom(
            (CENTER_MM[0] - 23.0 + offset, CENTER_MM[1], CENTER_MM[2] + 32.0), 12.0
        )
        kits_labels = np.zeros_like(segmentation)
        kits_labels[segmentation == int(CanonicalLabel.KIDNEY)] = 1
        kits_labels[segmentation == int(CanonicalLabel.TUMOR)] = 2
        case_dir = root / f"case_{index:05d}"
        case_dir.mkdir(parents=True)
        nib.save(nib.Nifti1Image(image, affine), str(case_dir / "imaging.nii.gz"))
        nib.save(nib.Nifti1Image(kits_labels, affine), str(case_dir / "segmentation.nii.gz"))

    def fake_segment(image_path, prediction_path, work):
        truth_path = Path(image_path).parent / "segmentation.nii.gz"
        truth = np.asanyarray(nib.load(str(truth_path)).dataobj)
        prediction = np.zeros(truth.shape, np.uint8)
        prediction[truth == 1] = int(CanonicalLabel.KIDNEY)
        tumor = truth == 2
        prediction[tumor] = int(CanonicalLabel.TUMOR)
        prediction[np.roll(tumor, 2, axis=0) & ~tumor] = int(CanonicalLabel.TUMOR)
        nib.save(nib.Nifti1Image(prediction, np.diag([*SPACING, 1.0])), str(prediction_path))
        return {"warnings": []}

    monkeypatch.setattr(run_validation, "segment", fake_segment)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_validation", "--dataset", "kits23", "--root", str(root), "--out", str(out),
         "--work", str(tmp_path / "work")],
    )
    assert run_validation.main() == 0
    rows = run_validation.read_results(out / "results.csv")
    assert [row["case"] for row in rows] == ["case_00000", "case_00001"]
    assert all(not row.get("error") for row in rows)
    assert float(rows[0]["tumor_dice"]) > 0.8
    report = (out / "report.md").read_text()
    assert "Optimistic" in report and "renal_total" in report

    # A rerun skips finished cases (resume across sessions).
    assert run_validation.main() == 0
    assert len(run_validation.read_results(out / "results.csv")) == 2


def test_failed_cases_are_counted_excluded_and_retried(tmp_path: Path, monkeypatch) -> None:
    """Regression (review): an empty-message failure was reported as 'failed: 0'."""

    run_validation = _load("run_validation")
    attempts = {"count": 0}

    def flaky(args, case_id):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("")  # e.g. a multiprocessing crash with no message
        return {"case": case_id, "status": "ok", "dataset": "kits23", "tumor_dice": 0.9,
                "renal_mass_dice": 0.95, "true_lesions": 1, "found": 1,
                "false_positive_lesions": 0, "missed_lesion_diameters_mm": "[]",
                "main_tumor_matched": True}

    monkeypatch.setattr(run_validation, "evaluate_case", flaky)
    out = tmp_path / "out"
    argv = ["run_validation", "--dataset", "kits23", "--cases", "case_00001",
            "--out", str(out), "--work", str(tmp_path / "work")]
    monkeypatch.setattr(sys, "argv", argv)

    assert run_validation.main() == 1
    rows = run_validation.read_results(out / "results.csv")
    assert rows[0]["status"] == "failed" and rows[0]["error"].startswith("RuntimeError")
    summary = run_validation.summarise(rows)
    assert (summary["succeeded"], summary["failed"]) == (0, 1)
    assert "1 case(s) failed" in (out / "report.md").read_text()

    assert run_validation.main() == 0  # resume retries the failed case
    rows = run_validation.read_results(out / "results.csv")
    assert len(rows) == 1 and rows[0]["status"] == "ok"
    assert run_validation.summarise(rows)["tumor_dice"]["n"] == 1
