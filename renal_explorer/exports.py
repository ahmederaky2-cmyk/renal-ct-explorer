"""Exports for the research planning workflow: STL surfaces and a PDF plan report."""

from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime, timezone
from typing import Any

import numpy as np

STL_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ]
)


def stl_bytes(vertices: np.ndarray, faces: np.ndarray, name: str = "surface") -> bytes:
    """Binary STL in RAS millimetres (the unit 3D-printing slicers expect)."""

    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    records = np.zeros(triangles.shape[0], dtype=STL_DTYPE)
    records["normal"] = normals
    records["vertices"] = triangles
    header = f"renal-ct-explorer {name} RAS mm research-only".encode("ascii", "replace")
    header = header[:80].ljust(80, b" ")
    return header + np.uint32(triangles.shape[0]).tobytes() + records.tobytes()


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_").lower() or "surface"


def stl_zip(
    meshes: dict[str, dict[str, Any]],
    readme: str = "",
    resolution: dict[str, Any] | None = None,
) -> bytes:
    """Zip of one STL per structure. ``meshes`` maps display name -> mesh dict."""

    if not readme:
        readme = (
            "Research-only surfaces exported from Renal CT Research Explorer.\n"
            "Coordinates are RAS millimetres; all files share one coordinate frame, so they\n"
            "line up when imported together. Not a medical device. Not for clinical use.\n"
        )
    if resolution and resolution.get("reduced"):
        readme += (
            "\nDEMO RESOLUTION: built from a reduced grid "
            f"({resolution.get('spacing_mm')} mm, full {resolution.get('full_spacing_mm')} mm). "
            "Shapes and sizes are approximate.\n"
        )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, mesh in meshes.items():
            if mesh is None or len(mesh.get("faces", [])) == 0:
                continue
            archive.writestr(
                f"{_safe_name(name)}.stl", stl_bytes(mesh["vertices"], mesh["faces"], name)
            )
        archive.writestr("README.txt", readme)
    return buffer.getvalue()


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}{suffix}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _l_range(renal: dict[str, Any]) -> str:
    low, high = renal.get("L_range") or (renal["L"], renal["L"])
    return str(renal["L"]) if low == high else f"{low}-{high} (est. {renal['L']})"


def _fmt_list(values) -> str:
    return " × ".join(f"{float(value):.2f}" for value in values) if values else "?"


def _n_range(renal: dict[str, Any]) -> str:
    low, high = renal.get("N_range") or (1, 3)
    estimate = renal.get("N_estimate")
    return f"{low}-{high}" + (f" (est. {estimate}, {renal.get('N_source')})" if estimate else "")


def plan_report_pdf(
    summary: dict[str, Any],
    case_label: str,
    images: list[tuple[str, np.ndarray]] | None = None,
) -> bytes:
    """A compact PDF of the plan summary. ``images`` holds (caption, RGB uint8 array)."""

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=f"Partial nephrectomy research plan – {case_label}",
    )
    table_style = TableStyle(
        [
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef9")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0b8c8")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
    )

    cell_style = styles["BodyText"].clone("cell", fontSize=9, leading=11)

    def table(rows, widths=None):
        rows = [
            [
                Paragraph(str(value), cell_style) if isinstance(value, str) else value
                for value in row
            ]
            for row in rows
        ]
        item = Table(rows, colWidths=widths, hAlign="LEFT")
        item.setStyle(table_style)
        return item

    scores = summary["scores"]
    renal = scores["renal"]
    padua = scores["padua"]
    resection = summary["resection"]
    story: list[Any] = [
        Paragraph("Partial nephrectomy research plan", styles["Title"]),
        Paragraph(
            f"{case_label} · {summary['side']} kidney · tumour {summary['tumor_index'] + 1} · "
            f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
            styles["Normal"],
        ),
        Spacer(1, 3 * mm),
        Paragraph(
            "<b>Research software. Not a medical device. Automated scores are unvalidated "
            "approximations and must be checked by a qualified clinician.</b>",
            styles["Normal"],
        ),
        *(
            [
                Spacer(1, 2 * mm),
                Paragraph(
                    "<b>DEMO RESOLUTION:</b> computed on a reduced grid "
                    f"({_fmt_list(summary['resolution'].get('spacing_mm'))} mm instead of "
                    f"{_fmt_list(summary['resolution'].get('full_spacing_mm'))} mm). "
                    "Volumes, distances and scores are approximate.",
                    styles["Normal"],
                ),
            ]
            if (summary.get("resolution") or {}).get("reduced")
            else []
        ),
        Spacer(1, 4 * mm),
        Paragraph("Nephrometry", styles["Heading2"]),
        table(
            [
                ["System", "Score", "Complexity / risk", "Components"],
                [
                    "R.E.N.A.L.",
                    renal["score"],
                    renal["complexity"],
                    f"R{renal['R']} E{renal['E']} "
                    f"N{_fmt(renal['N']) if renal['N'] is not None else _n_range(renal)} "
                    f"A:{renal['A']} "
                    f"L{_l_range(renal)} h:{_fmt(renal['hilar'])}",
                ],
                [
                    "PADUA",
                    padua["score"],
                    padua["risk"],
                    ", ".join(f"{key} {_fmt(value)}" for key, value in padua["items"].items()),
                ],
            ],
            [26 * mm, 22 * mm, 34 * mm, 96 * mm],
        ),
        Spacer(1, 2 * mm),
        table(
            [
                ["Max diameter", "Exophytic", "Location", "Contact surface"],
                [
                    _fmt(scores["max_diameter_mm"], 1, " mm"),
                    _fmt(100 * scores["exophytic_fraction"], 0, " %"),
                    scores["location"],
                    f"{_fmt(summary['contact_surface']['contact_cm2'])} cm<super>2</super> of "
                    f"{_fmt(summary['contact_surface']['tumor_surface_cm2'])} cm<super>2</super>",
                ],
            ]
        ),
    ]

    if summary["distances"]:
        story += [
            Paragraph("Distances from tumour surface", styles["Heading2"]),
            table(
                [["Structure", "Gap", "Source"]]
                + [
                    [name, _fmt(item["gap_mm"], 1, " mm"), item["source"]]
                    for name, item in summary["distances"].items()
                ]
            ),
        ]

    story += [
        Paragraph("Resection simulation", styles["Heading2"]),
        table(
            [
                ["Technique", "Tumour", "Parenchyma removed", "Ipsilateral kept", "Total kept"],
                [
                    resection["technique"],
                    _fmt(resection["tumor_ml"], 1, " mL"),
                    _fmt(resection["removed_parenchyma_ml"], 1, " mL"),
                    _fmt(resection["preserved_ipsilateral_pct"], 1, " %"),
                    _fmt(resection["preserved_total_pct"], 1, " %"),
                ],
            ]
        ),
    ]
    for warning in resection["warnings"]:
        story.append(
            Paragraph(
                f"Warning – {warning['structure']}: {_fmt(warning['gap_mm'])} mm from tumour "
                f"({warning['level']}).",
                styles["Normal"],
            )
        )

    territories = summary.get("territories")
    if territories:
        story += [
            Paragraph("Arterial territories (approximation)", styles["Heading2"]),
            table(
                [["Branch", "Region", "Parenchyma", "Tumour supplied", "Feeds tumour"]]
                + [
                    [
                        branch["name"],
                        branch["region"],
                        f"{_fmt(branch['parenchyma_ml'])} mL "
                        f"({_fmt(branch['parenchyma_pct'], 0)} %)",
                        _fmt(branch["tumor_pct"], 0, " %"),
                        _fmt(branch["feeds_tumor"]),
                    ]
                    for branch in territories["branches"]
                ]
            ),
            Paragraph(
                (
                    f"Renal arteries reaching kidney: {territories['renal_artery_count']}. "
                    f"Selective clamp of feeding branches: ischaemic parenchyma roughly "
                    f"{_fmt(territories['selective_clamp_ischemic_ml'])} mL "
                    f"({_fmt(territories['selective_clamp_ischemic_pct'], 0)} %) vs main artery "
                    f"{_fmt(territories['main_clamp_ischemic_ml'])} mL. "
                    if territories.get("ischemia_estimate_reliable")
                    else "Ischaemic-volume estimate withheld: "
                    + " ".join(territories.get("warnings", []))
                    + " "
                )
                + f"{territories['method']}.",
                styles["Normal"],
            ),
        ]

    for caption, rgb in images or []:
        array = np.asarray(rgb, dtype=np.uint8)
        from PIL import Image as PILImage

        png = io.BytesIO()
        PILImage.fromarray(array).save(png, format="PNG")
        png.seek(0)
        width_px, height_px = array.shape[1], array.shape[0]
        display_width = 120 * mm
        story += [
            Spacer(1, 3 * mm),
            Image(
                png,
                width=display_width,
                height=display_width * height_px / max(width_px, 1),
            ),
            Paragraph(caption, styles["Italic"]),
        ]

    notes = list(scores.get("notes", [])) + [f"Renal sinus: {summary['sinus_source']}."]
    story += [Paragraph("Method notes", styles["Heading2"])] + [
        Paragraph(f"• {note}", styles["Normal"]) for note in notes
    ]
    document.build(story)
    return buffer.getvalue()
