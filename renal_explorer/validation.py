"""Validation metrics: segmentation accuracy, plan agreement, and rater agreement.

Used by ``validation/run_validation.py`` (public datasets) and
``validation/compare_scores.py`` (your manual R.E.N.A.L. scores vs the planner).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .labels import CanonicalLabel

STRUCTURE = ndi.generate_binary_structure(3, 2)
RENAL_MASS = (CanonicalLabel.KIDNEY, CanonicalLabel.TUMOR, CanonicalLabel.CYST)


# --------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------


def dice(first: np.ndarray, second: np.ndarray) -> float | None:
    total = int(first.sum()) + int(second.sum())
    if total == 0:
        return None  # both empty: undefined, reported separately
    return 2.0 * int(np.logical_and(first, second).sum()) / total


def lesion_detection(
    truth: np.ndarray, prediction: np.ndarray, spacing, min_volume_ml: float = 0.1
) -> dict[str, Any]:
    """Lesion-wise detection: a true lesion counts as found if any predicted voxel overlaps."""

    voxel_ml = float(np.prod(spacing)) / 1000.0
    truth_labels, truth_count = ndi.label(truth, structure=STRUCTURE)
    prediction_labels, prediction_count = ndi.label(prediction, structure=STRUCTURE)
    lesions = []
    for index in range(1, truth_count + 1):
        component = truth_labels == index
        volume = float(component.sum()) * voxel_ml
        if volume < min_volume_ml:
            continue
        diameter = 2.0 * (3.0 * volume * 1000.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        lesions.append(
            {
                "volume_ml": volume,
                "equivalent_diameter_mm": diameter,
                "found": bool(np.any(prediction[component])),
            }
        )
    false_positives = 0
    for index in range(1, prediction_count + 1):
        component = prediction_labels == index
        if float(component.sum()) * voxel_ml >= min_volume_ml and not np.any(truth[component]):
            false_positives += 1
    return {
        "true_lesions": len(lesions),
        "found": sum(lesion["found"] for lesion in lesions),
        "false_positive_lesions": false_positives,
        "lesions": lesions,
    }


def evaluate_segmentation(
    truth: np.ndarray, prediction: np.ndarray, spacing
) -> dict[str, Any]:
    """Compare canonical labelmaps. Kidney is scored as the whole renal mass
    (kidney + tumour + cyst) because datasets disagree on whether tumour is 'kidney'."""

    voxel_ml = float(np.prod(spacing)) / 1000.0
    truth_renal = np.isin(truth, [int(label) for label in RENAL_MASS])
    prediction_renal = np.isin(prediction, [int(label) for label in RENAL_MASS])
    truth_tumor = truth == int(CanonicalLabel.TUMOR)
    prediction_tumor = prediction == int(CanonicalLabel.TUMOR)
    result: dict[str, Any] = {
        "renal_mass_dice": dice(truth_renal, prediction_renal),
        "tumor_dice": dice(truth_tumor, prediction_tumor),
        "tumor_volume_true_ml": float(truth_tumor.sum()) * voxel_ml,
        "tumor_volume_pred_ml": float(prediction_tumor.sum()) * voxel_ml,
    }
    for label, key in (
        (CanonicalLabel.CYST, "cyst_dice"),
        (CanonicalLabel.RENAL_ARTERY, "renal_artery_dice"),
        (CanonicalLabel.RENAL_VEIN, "renal_vein_dice"),
    ):
        truth_mask = truth == int(label)
        if truth_mask.any():
            result[key] = dice(truth_mask, prediction == int(label))
    detection = lesion_detection(truth_tumor, prediction_tumor, spacing)
    result.update({key: value for key, value in detection.items() if key != "lesions"})
    result["missed_lesion_diameters_mm"] = [
        round(lesion["equivalent_diameter_mm"], 1)
        for lesion in detection["lesions"]
        if not lesion["found"]
    ]
    return result


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask, structure=STRUCTURE)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum_labels(np.ones_like(labels), labels, index=np.arange(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def main_tumors_match(truth: np.ndarray, prediction: np.ndarray) -> bool:
    """True when the largest predicted tumour overlaps the largest true tumour."""

    first = largest_component(truth == int(CanonicalLabel.TUMOR))
    second = largest_component(prediction == int(CanonicalLabel.TUMOR))
    return bool(np.any(first & second))


# --------------------------------------------------------------------------------------
# Plan agreement (planner on expert labels vs planner on automatic labels)
# --------------------------------------------------------------------------------------


def plan_row(summary: dict[str, Any], prefix: str) -> dict[str, Any]:
    scores = summary["scores"]
    renal = scores["renal"]
    padua = scores["padua"]
    return {
        f"{prefix}_R": renal["R"],
        f"{prefix}_E": renal["E"],
        # N is usually an estimate (no reviewed sinus outline); the flag says which.
        f"{prefix}_N": renal["N"] if renal["N"] is not None else renal.get("N_estimate"),
        f"{prefix}_N_definitive": renal["N"] is not None,
        f"{prefix}_N_source": renal.get("N_source"),
        f"{prefix}_A": renal["A"],
        f"{prefix}_L": renal["L"],
        f"{prefix}_h": renal["hilar"],
        f"{prefix}_renal_total": renal.get("total_estimate"),
        f"{prefix}_renal_total_range": f"{renal['total_min']}-{renal['total_max']}",
        f"{prefix}_padua_min": padua["total_min"],
        f"{prefix}_diameter_mm": scores["max_diameter_mm"],
        f"{prefix}_exophytic_pct": 100.0 * scores["exophytic_fraction"],
        f"{prefix}_nearness_mm": renal.get("N_gap_mm"),
        f"{prefix}_parenchyma_kept_pct": summary["resection"]["preserved_ipsilateral_pct"],
    }


# --------------------------------------------------------------------------------------
# Rater agreement
# --------------------------------------------------------------------------------------


def weighted_kappa(
    first: Sequence[Any],
    second: Sequence[Any],
    categories: Sequence[Any],
    weights: str = "quadratic",
) -> float | None:
    """Cohen's kappa with linear/quadratic weights for ordinal scores (unweighted: 'none')."""

    pairs = [
        (a, b)
        for a, b in zip(first, second)  # noqa: B905
        if a in categories and b in categories
    ]
    if len(pairs) < 2:
        return None
    index = {category: position for position, category in enumerate(categories)}
    size = len(categories)
    observed = np.zeros((size, size))
    for a, b in pairs:
        observed[index[a], index[b]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    grid = np.subtract.outer(np.arange(size), np.arange(size))
    if weights == "quadratic":
        penalty = (grid / max(size - 1, 1)) ** 2
    elif weights == "linear":
        penalty = np.abs(grid) / max(size - 1, 1)
    else:
        penalty = (grid != 0).astype(float)
    denominator = float((penalty * expected).sum())
    if denominator == 0:
        return 1.0 if float((penalty * observed).sum()) == 0 else None
    return 1.0 - float((penalty * observed).sum()) / denominator


def agreement(
    first: Sequence[Any], second: Sequence[Any], categories: Sequence[Any] | None = None
) -> dict[str, Any]:
    pairs = [
        (a, b) for a, b in zip(first, second)  # noqa: B905
        if a is not None and b is not None and not _is_nan(a) and not _is_nan(b)
    ]
    result: dict[str, Any] = {"n": len(pairs)}
    if not pairs:
        return result
    result["exact_agreement"] = float(np.mean([a == b for a, b in pairs]))
    if categories is not None:
        ordinal = all(isinstance(category, (int, float)) for category in categories)
        result["kappa"] = weighted_kappa(
            [a for a, _ in pairs],
            [b for _, b in pairs],
            categories,
            "quadratic" if ordinal else "none",
        )
    if all(isinstance(value, (int, float)) for pair in pairs for value in pair):
        differences = np.array([float(b) - float(a) for a, b in pairs])
        result["mean_difference"] = float(differences.mean())
        result["mean_absolute_difference"] = float(np.abs(differences).mean())
        if len(differences) > 1:
            spread = 1.96 * float(differences.std(ddof=1))
            result["limits_of_agreement"] = [
                result["mean_difference"] - spread,
                result["mean_difference"] + spread,
            ]
    return result


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def renal_complexity(total: float | None) -> str | None:
    if total is None or _is_nan(total):
        return None
    return "Low" if total <= 6 else "Intermediate" if total <= 9 else "High"
