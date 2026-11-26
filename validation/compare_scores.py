"""Compare your manual R.E.N.A.L. scores with the planner.

Two comparisons, from the ``results.csv`` written by ``run_validation.py``:

* **you vs planner on expert labels**: is the scoring algorithm right?
* **you vs planner on automatic labels**: is the whole pipeline right?

The manual CSV needs a ``Case`` column and ``R E N A L h`` (``A`` = a/p/x,
``h`` = yes/no). It can be exported from the scoring sheet. Column names are not
case-sensitive, and empty rows are ignored.

    python validation/compare_scores.py --manual my_scores.csv \
        --planner validation_results/results.csv --out validation_results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from renal_explorer.validation import agreement, renal_complexity  # noqa: E402

ITEMS = {
    "R": [1, 2, 3],
    "E": [1, 2, 3],
    "N": [1, 2, 3],
    "L": [1, 2, 3],
    "A": ["a", "p", "x"],
}


def _norm(value: Any) -> Any:
    text = str(value or "").strip().lower().lstrip("'")
    if text in ("", "none", "nan", "—", "-"):
        return None
    if text in ("yes", "y", "true", "1h", "h", "ja", "j"):
        return True
    if text in ("no", "n", "false", "nein"):
        return False
    try:
        number = float(text)
        return int(number) if number.is_integer() else number
    except ValueError:
        return text


def read_manual(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    scores: dict[str, dict[str, Any]] = {}
    for row in rows:
        lowered = {key.strip().lower(): value for key, value in row.items() if key}
        case = str(lowered.get("case") or "").strip()
        if not case or all(_norm(lowered.get(item.lower())) is None for item in ITEMS):
            continue
        entry = {item: _norm(lowered.get(item.lower())) for item in ITEMS}
        entry["h"] = _norm(lowered.get("h"))
        if isinstance(entry["h"], str):
            entry["h"] = entry["h"] in ("h", "hilar")
        parts = [entry[item] for item in ("R", "E", "N", "L")]
        entry["total"] = sum(parts) if all(isinstance(part, int) for part in parts) else None
        scores[case] = entry
    return scores


def read_planner(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, newline="") as handle:
        return {row["case"]: row for row in csv.DictReader(handle) if not row.get("error")}


def compare(manual: dict[str, dict[str, Any]], planner: dict[str, dict[str, Any]], prefix: str):
    cases = sorted(set(manual) & set(planner))
    result: dict[str, Any] = {"cases": len(cases)}
    for item, categories in ITEMS.items():
        result[item] = agreement(
            [manual[case][item] for case in cases],
            [_norm(planner[case].get(f"{prefix}_{item}")) for case in cases],
            categories,
        )
    result["total"] = agreement(
        [manual[case]["total"] for case in cases],
        [_norm(planner[case].get(f"{prefix}_renal_total")) for case in cases],
        list(range(4, 13)),
    )
    result["complexity"] = agreement(
        [renal_complexity(manual[case]["total"]) for case in cases],
        [
            renal_complexity(_norm(planner[case].get(f"{prefix}_renal_total")))
            for case in cases
        ],
        ["Low", "Intermediate", "High"],
    )
    result["hilar"] = agreement(
        [manual[case]["h"] for case in cases],
        [_norm(planner[case].get(f"{prefix}_h")) for case in cases],
    )
    disagreements = []
    for case in cases:
        mine = manual[case]["total"]
        theirs = _norm(planner[case].get(f"{prefix}_renal_total"))
        if isinstance(mine, int) and isinstance(theirs, int) and abs(mine - theirs) >= 2:
            disagreements.append({"case": case, "manual": mine, "planner": theirs})
    result["large_disagreements"] = disagreements
    return result


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)


def report(results: dict[str, Any]) -> str:
    lines = ["# Manual vs planner R.E.N.A.L.", ""]
    for prefix, title in (
        ("expert", "Planner on expert labels (tests the scoring algorithm)"),
        ("auto", "Planner on automatic labels (tests the whole pipeline)"),
    ):
        result = results[prefix]
        lines += [
            f"## {title}",
            "",
            f"Cases scored by both: {result['cases']}",
            "",
            "| Item | n | Exact agreement | κ (quadratic for ordinal) "
            "| Mean diff (planner − you) |",
            "|---|---:|---:|---:|---:|",
        ]
        for item in (*ITEMS, "total", "complexity", "hilar"):
            data = result[item]
            lines.append(
                f"| {item} | {data.get('n', 0)} | {_fmt(data.get('exact_agreement'))} | "
                f"{_fmt(data.get('kappa'))} | {_fmt(data.get('mean_difference'))} |"
            )
        if result["large_disagreements"]:
            lines += ["", "Total differs by ≥ 2 points:", ""] + [
                f"- {item['case']}: you {item['manual']}, planner {item['planner']}"
                for item in result["large_disagreements"]
            ]
        lines.append("")
    lines.append(
        "One rater is not a reference standard; published inter-observer agreement for "
        "R.E.N.A.L. components is itself imperfect. Add a second rater when you can."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()
    manual = read_manual(args.manual)
    planner = read_planner(args.planner)
    results = {prefix: compare(manual, planner, prefix) for prefix in ("expert", "auto")}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manual_vs_planner.json").write_text(json.dumps(results, indent=2))
    text = report(results)
    (args.out / "manual_vs_planner.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
