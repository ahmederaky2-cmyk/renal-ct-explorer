"""Validate the automatic pipeline and the planner on public data with expert labels.

For each case: run the automatic segmentation, compare it with the expert labels, then run
the planner twice (expert labels vs automatic labels) and record how much the surgical
numbers change. Results are appended per case, so an interrupted run (Kaggle/Colab session
limit) continues where it stopped.

Examples::

    # KiTS23 cases downloaded on the fly (seen by AbdomenAtlasNet during training!)
    python validation/run_validation.py --dataset kits23 \
        --cases-file validation/scoring_cases.txt --limit 20 \
        --work /kaggle/tmp/work --out /kaggle/working/validation

    # Held-out data you have locally, e.g. KiPA22 (tumour + kidney + vessel labels)
    python validation/run_validation.py --dataset kipa22 --root /kaggle/input/kipa22 \
        --out /kaggle/working/validation_kipa22
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renal_explorer.datasets import DATASETS, get_adapter  # noqa: E402
from renal_explorer.validation import (  # noqa: E402
    agreement,
    evaluate_segmentation,
    main_tumors_match,
    plan_row,
)

# Datasets that AbdomenAtlas 3.0 (and therefore AbdomenAtlasNet) drew on.
SEEN_BY_TUMOR_MODEL = {"kits23", "c4kc_kits", "abdomen_atlas", "flare22"}
SIZE_BINS = ((0, 20, "<2 cm"), (20, 40, "2–4 cm"), (40, 70, "4–7 cm"), (70, 1e9, ">7 cm"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", default="kits23", choices=sorted(DATASETS))
    parser.add_argument("--root", type=Path, help="Local dataset folder (non-download datasets)")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--cases-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--work", type=Path, default=Path("/tmp/renal_validation"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--margin-mm", type=float, default=3.0)
    parser.add_argument("--keep-downloads", action="store_true")
    parser.add_argument("--time-limit-hours", type=float, default=0.0)
    parser.add_argument(
        "--report-only", action="store_true", help="Only rebuild the report from results.csv"
    )
    return parser.parse_args()


def case_list(args: argparse.Namespace) -> list[str]:
    if args.cases:
        cases = list(args.cases)
    elif args.cases_file:
        cases = [line.strip() for line in args.cases_file.read_text().splitlines() if line.strip()]
    elif args.root:
        cases = [item.case_id for item in get_adapter(args.dataset).list_cases(args.root)]
    else:
        raise SystemExit("Give --cases, --cases-file or --root.")
    return cases[: args.limit] if args.limit else cases


def case_files(args: argparse.Namespace, case_id: str):
    if args.dataset == "kits23" and not args.root:
        from renal_explorer.remote_data import download_kits23_case

        return download_kits23_case(case_id, args.work)
    adapter = get_adapter(args.dataset)
    for item in adapter.list_cases(args.root):
        if item.case_id == case_id:
            return item
    raise FileNotFoundError(case_id)


def segment(image_path: Path, prediction_path: Path, work: Path) -> dict[str, Any]:
    import nibabel as nib

    from renal_explorer.patient.fusion import fuse
    from renal_explorer.patient.pipeline import segment_main_phase
    from renal_explorer.patient.segmenters import PipelineSettings

    provenance: dict[str, Any] = {"steps": [], "warnings": []}

    def log(message: str) -> None:
        print(f"    {message}", flush=True)

    def step(name: str) -> None:
        log(name)
        provenance["steps"].append(name)

    def warn(message: str) -> None:
        log(f"WARNING: {message}")
        provenance["warnings"].append(message)

    reference = nib.load(str(image_path))
    spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
    masks = segment_main_phase(image_path, work, PipelineSettings(), log, step, warn, provenance)
    labels, _counts, notes = fuse(masks, spacing)
    provenance["warnings"].extend(notes)
    nib.save(nib.Nifti1Image(labels, reference.affine), str(prediction_path))
    return provenance


def load_canonical_labels(path: Path, shape) -> np.ndarray:
    import nibabel as nib

    image = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asanyarray(image.dataobj).astype(np.uint8)
    if data.shape != tuple(shape):
        raise ValueError(f"Prediction shape {data.shape} differs from CT {shape}.")
    return data


def evaluate_case(args: argparse.Namespace, case_id: str) -> dict[str, Any]:
    from renal_explorer.planning import PlanningError, build_plan

    started = time.monotonic()
    files = case_files(args, case_id)
    adapter = get_adapter(args.dataset)
    case = adapter.load_case(files)
    if case.segmentation is None:
        raise ValueError("No expert segmentation.")

    predictions = args.out / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    prediction_path = predictions / f"{case_id}.nii.gz"
    warnings: list[str] = []
    if not prediction_path.is_file():
        provenance = segment(files.image_path, prediction_path, args.work / "nnunet")
        warnings = provenance["warnings"]
    prediction = load_canonical_labels(prediction_path, case.image.shape)

    if not prediction_path.is_file():
        raise RuntimeError("Segmentation produced no prediction file.")
    row: dict[str, Any] = {
        "case": case_id,
        "status": "ok",
        "dataset": args.dataset,
        "seen_by_tumor_model": args.dataset in SEEN_BY_TUMOR_MODEL,
        "slice_mm": round(case.spacing[2], 2),
    }
    metrics = evaluate_segmentation(case.segmentation, prediction, case.spacing)
    detection = metrics.pop("missed_lesion_diameters_mm")
    row.update(metrics)
    row["missed_lesion_diameters_mm"] = json.dumps(detection)
    row["main_tumor_matched"] = main_tumors_match(case.segmentation, prediction)

    for prefix, labels in (("expert", case.segmentation), ("auto", prediction)):
        try:
            plan = build_plan(labels, case.affine, case.spacing, case.image, 0, args.margin_mm)
            row.update(plan_row(plan["summary"], prefix))
        except PlanningError as exc:
            row[f"{prefix}_plan_error"] = str(exc)
    row["warnings"] = " | ".join(warnings)[:500]
    row["seconds"] = round(time.monotonic() - started, 1)

    if not args.keep_downloads and args.dataset == "kits23" and not args.root:
        shutil.rmtree(files.image_path.parent, ignore_errors=True)
    return row


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def append_result(path: Path, row: dict[str, Any]) -> None:
    """Add a row, replacing an earlier attempt at the same case (failed cases are retried)."""

    rows = [item for item in read_results(path) if item.get("case") != row.get("case")] + [row]
    fields: list[str] = []
    for item in rows:
        fields.extend(key for key in item if key not in fields)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(number) else number


def _ordinal(value: Any) -> Any:
    number = _number(value)
    if number is not None:
        return int(number)
    return value if value not in ("", None, "None") else None


def succeeded(row: dict[str, Any]) -> bool:
    return row.get("status") == "ok"


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if succeeded(row)]

    def stats(key: str) -> dict[str, Any]:
        values = [value for value in (_number(row.get(key)) for row in ok) if value is not None]
        if not values:
            return {"n": 0}
        return {
            "n": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
        }

    true_lesions = sum(int(_number(row.get("true_lesions")) or 0) for row in ok)
    found = sum(int(_number(row.get("found")) or 0) for row in ok)
    false_positives = sum(int(_number(row.get("false_positive_lesions")) or 0) for row in ok)
    missed = [
        size
        for row in ok
        for size in json.loads(row.get("missed_lesion_diameters_mm") or "[]")
    ]
    plan: dict[str, Any] = {}
    for item, categories in (
        ("R", [1, 2, 3]),
        ("E", [1, 2, 3]),
        ("N", [1, 2, 3]),
        ("L", [1, 2, 3]),
        ("A", ["a", "p", "x"]),
        ("renal_total", list(range(4, 13))),
    ):
        plan[item] = agreement(
            [_ordinal(row.get(f"expert_{item}")) for row in ok],
            [_ordinal(row.get(f"auto_{item}")) for row in ok],
            categories,
        )
    for item in ("diameter_mm", "nearness_mm", "exophytic_pct", "parenchyma_kept_pct"):
        plan[item] = agreement(
            [_number(row.get(f"expert_{item}")) for row in ok],
            [_number(row.get(f"auto_{item}")) for row in ok],
        )
    return {
        "cases": len(rows),
        "succeeded": len(ok),
        "failed": len(rows) - len(ok),
        "failed_cases": [row.get("case") for row in rows if not succeeded(row)],
        "n_definitive_fraction": (
            float(np.mean([str(row.get("auto_N_definitive")) == "True" for row in ok]))
            if ok
            else None
        ),
        "seen_by_tumor_model": any(str(row.get("seen_by_tumor_model")) == "True" for row in ok),
        "renal_mass_dice": stats("renal_mass_dice"),
        "tumor_dice": stats("tumor_dice"),
        "lesion_sensitivity": found / true_lesions if true_lesions else None,
        "true_lesions": true_lesions,
        "false_positive_lesions_per_case": false_positives / len(ok) if ok else None,
        "missed_lesion_diameters_mm": sorted(missed),
        "main_tumor_matched": float(
            np.mean([str(row.get("main_tumor_matched")) == "True" for row in ok])
        )
        if ok
        else None,
        "plan_agreement_expert_vs_auto": plan,
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(out: Path, summary: dict[str, Any]) -> Path:
    lines = [
        "# Validation report",
        "",
        f"Cases: {summary['cases']} (succeeded: {summary['succeeded']}, "
        f"failed: {summary['failed']})",
        "",
    ]
    if summary["failed"]:
        lines += [
            f"> **{summary['failed']} case(s) failed** and are excluded from every metric: "
            f"{', '.join(summary['failed_cases'])}. See the `error` column; rerun to retry.",
            "",
        ]
    if summary["seen_by_tumor_model"]:
        lines += [
            "> **Optimistic:** these cases come from a dataset AbdomenAtlasNet was trained on. "
            "Use held-out data (e.g. KiPA22 or your own) for an unbiased estimate.",
            "",
        ]
    lines += [
        "## Segmentation",
        "",
        "| Metric | n | Mean | Median | 10th pct |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, name in (("renal_mass_dice", "Renal mass Dice"), ("tumor_dice", "Tumour Dice")):
        item = summary[key]
        lines.append(
            f"| {name} | {item.get('n', 0)} | {_fmt(item.get('mean'))} | "
            f"{_fmt(item.get('median'))} | {_fmt(item.get('p10'))} |"
        )
    lines += [
        "",
        f"Lesion sensitivity: {_fmt(summary['lesion_sensitivity'])} of "
        f"{summary['true_lesions']} lesions · false-positive lesions per case: "
        f"{_fmt(summary['false_positive_lesions_per_case'])} · main tumour matched: "
        f"{_fmt(summary['main_tumor_matched'])}",
        "",
        f"Missed lesion sizes (equivalent diameter, mm): {summary['missed_lesion_diameters_mm']}",
        "",
        "## Does automatic segmentation change the plan?",
        "",
        "Planner on expert labels vs planner on automatic labels (same algorithm).",
        "",
        "| Item | n | Exact agreement | Weighted κ | Mean diff | Mean abs diff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, item in summary["plan_agreement_expert_vs_auto"].items():
        lines.append(
            f"| {key} | {item.get('n', 0)} | {_fmt(item.get('exact_agreement'))} | "
            f"{_fmt(item.get('kappa'))} | {_fmt(item.get('mean_difference'))} | "
            f"{_fmt(item.get('mean_absolute_difference'))} |"
        )
    lines += [
        "",
        "Exact agreement on continuous items (mm, %) is not meaningful; read the differences.",
        "N (and therefore the total) is an estimate unless a collecting-system label exists: "
        f"definitive in {_fmt(summary['n_definitive_fraction'])} of cases.",
        "Research software — not a clinical validation.",
    ]
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    results = args.out / "results.csv"

    if not args.report_only:
        done = {row["case"] for row in read_results(results) if succeeded(row)}
        started = time.monotonic()
        durations: list[float] = []
        for case_id in case_list(args):
            if case_id in done:
                continue
            if args.time_limit_hours and durations:
                budget = args.time_limit_hours * 3600
                if time.monotonic() - started + 1.5 * max(durations[-5:]) > budget:
                    print("Time budget reached; rerun the same command to continue.")
                    break
            print(f"{case_id} …", flush=True)
            case_start = time.monotonic()
            try:
                row = evaluate_case(args, case_id)
            except Exception as exc:  # noqa: BLE001 - keep going, record the failure
                message = f"{type(exc).__name__}: {exc}".strip()[:300]
                row = {
                    "case": case_id,
                    "status": "failed",
                    "dataset": args.dataset,
                    "error": message,
                }
                print(f"    failed: {message}", flush=True)
            append_result(results, row)
            durations.append(time.monotonic() - case_start)

    summary = summarise(read_results(results))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    report = write_report(args.out, summary)
    print(report.read_text())
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
