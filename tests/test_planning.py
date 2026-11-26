from __future__ import annotations

import numpy as np
import pytest

from renal_explorer.labels import CanonicalLabel
from renal_explorer.planning import (
    PlanningError,
    _closing_mm,
    _gap_mm,
    build_plan,
    resect_with_margin,
    tumor_components,
)

SPACING = (1.0, 1.0, 2.0)
SHAPE = (160, 120, 80)
CENTER_MM = np.array([40.0, 60.0, 80.0])  # left kidney (low x) in a 160 mm wide volume


def _grid_mm() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = [np.arange(size) * spacing for size, spacing in zip(SHAPE, SPACING)]  # noqa: B905
    return np.meshgrid(*axes, indexing="ij")


def _ellipsoid(center, semi) -> np.ndarray:
    x, y, z = _grid_mm()
    return (
        ((x - center[0]) / semi[0]) ** 2
        + ((y - center[1]) / semi[1]) ** 2
        + ((z - center[2]) / semi[2]) ** 2
    ) <= 1.0


def _tube(start, end, radius) -> np.ndarray:
    x, y, z = _grid_mm()
    points = np.stack((x, y, z), axis=-1)
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    direction = end - start
    t = np.clip(((points - start) @ direction) / float(direction @ direction), 0.0, 1.0)
    nearest = start + t[..., None] * direction
    return np.linalg.norm(points - nearest, axis=-1) <= radius


def phantom(tumor_center, tumor_radius, with_artery=False):
    """Bean-shaped left kidney with a renal sinus opening medially (+x)."""

    outer = _ellipsoid(CENTER_MM, (25.0, 20.0, 50.0))
    sinus_center = CENTER_MM + [8.0, 0.0, 0.0]
    sinus = _ellipsoid(sinus_center, (12.0, 8.0, 25.0))
    x, y, z = _grid_mm()
    channel = (
        (x >= sinus_center[0])
        & (np.abs(y - CENTER_MM[1]) <= 8.0)
        & (np.abs(z - CENTER_MM[2]) <= 20.0)
    )
    kidney = outer & ~(sinus | channel)

    segmentation = np.zeros(SHAPE, dtype=np.uint8)
    image = np.full(SHAPE, -100.0, dtype=np.float32)
    segmentation[kidney] = int(CanonicalLabel.KIDNEY)
    image[kidney] = 150.0

    tumor = _ellipsoid(tumor_center, (tumor_radius,) * 3)
    segmentation[tumor] = int(CanonicalLabel.TUMOR)
    image[tumor] = 60.0

    # A contralateral kidney so split-volume figures are defined.
    contralateral = _ellipsoid((120.0, 60.0, 80.0), (20.0, 15.0, 40.0))
    segmentation[contralateral] = int(CanonicalLabel.KIDNEY)
    image[contralateral] = 150.0

    if with_artery:
        hilum = (CENTER_MM[0] + 22.0, CENTER_MM[1], CENTER_MM[2])
        trunk = _tube((95.0, 60.0, 80.0), hilum, 2.0)
        upper = _tube(hilum, (CENTER_MM[0] - 10.0, 60.0, CENTER_MM[2] + 32.0), 1.5)
        lower = _tube(hilum, (CENTER_MM[0] - 10.0, 60.0, CENTER_MM[2] - 32.0), 1.5)
        artery = (trunk | upper | lower) & (segmentation != int(CanonicalLabel.TUMOR))
        segmentation[artery] = int(CanonicalLabel.RENAL_ARTERY)
        image[artery] = 300.0

    affine = np.diag([*SPACING, 1.0])
    return segmentation, image, affine


def test_gap_is_zero_for_face_neighbours_on_anisotropic_grid() -> None:
    assert _gap_mm((0, 0, 0), (0, 0, 1), (0.8, 0.8, 3.0)) == pytest.approx(0.0)
    assert _gap_mm((0, 0, 0), (0, 0, 3), (0.8, 0.8, 3.0)) == pytest.approx(6.0)


def test_closing_fills_narrow_notch_and_keeps_mask() -> None:
    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[5:35, 5:35, 5:35] = True
    mask[18:22, 5:20, 5:35] = False  # 4 mm wide notch
    closed = _closing_mm(mask, (1.0, 1.0, 1.0), 5.0)
    assert closed[mask].all()
    assert closed[20, 10, 20]


def test_exophytic_upper_pole_tumor_scores() -> None:
    # Lateral upper-pole surface point, tumour centre pushed 3 mm outward.
    tumor_center = (CENTER_MM[0] - 23.0, CENTER_MM[1], CENTER_MM[2] + 32.0)
    segmentation, image, affine = phantom(tumor_center, 12.0)
    plan = build_plan(segmentation, affine, SPACING, image, margin_mm=5.0)
    summary = plan["summary"]
    renal = summary["scores"]["renal"]

    assert summary["side"] == "Left"
    assert 22.0 <= summary["scores"]["max_diameter_mm"] <= 28.0
    assert renal["R"] == 1
    assert renal["E"] == 1
    assert renal["N"] is None and renal["N_range"] == [1, 3]  # no reviewed sinus outline
    assert renal["N_estimate"] == 1 and renal["N_source"] == "Renal sinus (proxy)"
    assert renal["A"] == "x"
    assert renal["L"] in (1, 2)
    assert summary["scores"]["location"] == "Upper pole"
    assert renal["complexity"] == "Low"
    assert "Renal sinus" in summary["distances"]

    contact = summary["contact_surface"]
    assert 0.15 < contact["contact_fraction"] < 0.7

    resection = summary["resection"]
    assert resection["removed_parenchyma_ml"] > 0
    assert 80.0 < resection["preserved_ipsilateral_pct"] < 100.0
    assert resection["contralateral_parenchyma_ml"] > 30.0
    assert resection["ipsilateral_share_after_pct"] < resection["ipsilateral_share_before_pct"]


def test_central_endophytic_tumor_is_high_complexity() -> None:
    tumor_center = (CENTER_MM[0] - 14.0, CENTER_MM[1], CENTER_MM[2])
    segmentation, image, affine = phantom(tumor_center, 7.0)
    plan = build_plan(segmentation, affine, SPACING, image, margin_mm=0.0)
    renal = plan["summary"]["scores"]["renal"]
    assert renal["E"] == 3
    assert renal["N_estimate"] == 3
    assert renal["L"] == 3
    assert renal["total_estimate"] >= 10
    assert renal["total_max"] >= 10 and "est." in renal["score"]
    assert plan["summary"]["resection"]["removed_parenchyma_ml"] == 0.0
    assert plan["summary"]["scores"]["padua"]["total_max"] > plan["summary"]["scores"][
        "padua"
    ]["total_min"]  # collecting system item unknown without its label


def test_enucleation_margin_monotonic() -> None:
    tumor_center = (CENTER_MM[0] - 23.0, CENTER_MM[1], CENTER_MM[2] + 32.0)
    segmentation, image, affine = phantom(tumor_center, 12.0)
    plan = build_plan(segmentation, affine, SPACING, image, margin_mm=3.0)
    zero = resect_with_margin(plan, 0.0).sum()
    three = resect_with_margin(plan, 3.0).sum()
    seven = resect_with_margin(plan, 7.0).sum()
    assert zero == 0 < three < seven
    assert three == plan["display"]["resection"].sum()


def test_arterial_territories_identify_feeding_branch() -> None:
    tumor_center = (CENTER_MM[0] - 23.0, CENTER_MM[1], CENTER_MM[2] + 32.0)
    segmentation, image, affine = phantom(tumor_center, 12.0, with_artery=True)
    plan = build_plan(segmentation, affine, SPACING, image, margin_mm=3.0, branch_depth=1)
    territories = plan["summary"]["territories"]
    assert territories is not None
    assert territories["renal_artery_count"] == 1
    assert len(territories["branches"]) == 2
    feeding = [branch for branch in territories["branches"] if branch["feeds_tumor"]]
    assert len(feeding) == 1
    assert feeding[0]["region"].startswith("upper")
    assert feeding[0]["tumor_pct"] > 90
    assert 30 < territories["selective_clamp_ischemic_pct"] < 70
    assert plan["display"]["territory_map"].shape == plan["display"]["kidney"].shape


def test_tumor_components_sorted_and_missing_tumor_rejected() -> None:
    segmentation = np.zeros((30, 30, 30), dtype=np.uint8)
    segmentation[2:6, 2:6, 2:6] = int(CanonicalLabel.TUMOR)
    segmentation[10:20, 10:20, 10:20] = int(CanonicalLabel.TUMOR)
    items = tumor_components(segmentation, (1.0, 1.0, 1.0))
    assert [round(item["volume_ml"], 3) for item in items] == [1.0, 0.064]

    kidney_only = np.zeros((30, 30, 30), dtype=np.uint8)
    kidney_only[5:25, 5:25, 5:25] = int(CanonicalLabel.KIDNEY)
    with pytest.raises(PlanningError):
        build_plan(kidney_only, np.eye(4), (1.0, 1.0, 1.0))


def test_hilar_suffix_only_for_main_vessel_contact() -> None:
    hilar_center = (CENTER_MM[0] + 16.0, CENTER_MM[1], CENTER_MM[2] + 8.0)
    segmentation, image, affine = phantom(hilar_center, 6.0, with_artery=True)
    renal = build_plan(segmentation, affine, SPACING, image)["summary"]["scores"]["renal"]
    assert renal["hilar"] is True
    assert renal["score"].split(" ")[0].endswith("h")

    lateral_center = (CENTER_MM[0] - 23.0, CENTER_MM[1], CENTER_MM[2] + 32.0)
    segmentation, image, affine = phantom(lateral_center, 12.0, with_artery=True)
    renal = build_plan(segmentation, affine, SPACING, image)["summary"]["scores"]["renal"]
    assert renal["hilar"] is False
    assert renal["N_estimate"] == 1


def test_exports_stl_zip_and_pdf() -> None:
    import io
    import zipfile

    from renal_explorer.exports import plan_report_pdf, stl_bytes, stl_zip

    vertices = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    payload = stl_bytes(vertices, faces, "tetra")
    assert len(payload) == 84 + 50 * 4
    assert int(np.frombuffer(payload[80:84], dtype="<u4")[0]) == 4

    surfaces = {"Tumor": {"vertices": vertices, "faces": faces}}
    archive = zipfile.ZipFile(io.BytesIO(stl_zip(surfaces)))
    assert sorted(archive.namelist()) == ["README.txt", "tumor.stl"]

    tumor_center = (CENTER_MM[0] - 23.0, CENTER_MM[1], CENTER_MM[2] + 32.0)
    segmentation, image, affine = phantom(tumor_center, 12.0, with_artery=True)
    summary = build_plan(segmentation, affine, SPACING, image)["summary"]
    pdf = plan_report_pdf(
        summary, "phantom", images=[("Axial", np.zeros((40, 60, 3), dtype=np.uint8))]
    )
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 2000


def test_assemble_does_not_mutate_cached_geometry() -> None:
    from renal_explorer.planning import assemble_plan, plan_geometry, tumor_center_voxel

    tumor_center = (CENTER_MM[0] - 23.0, CENTER_MM[1], CENTER_MM[2] + 32.0)
    segmentation, image, affine = phantom(tumor_center, 12.0, with_artery=True)
    geometry = plan_geometry(segmentation, affine, SPACING, image)
    first = assemble_plan(geometry, 2.0)
    second = assemble_plan(geometry, 6.0)
    assert "map" in geometry["territories"]
    assert first["summary"]["territories"] is not None and "map" not in first["summary"][
        "territories"
    ]
    assert (
        second["summary"]["resection"]["removed_parenchyma_ml"]
        > first["summary"]["resection"]["removed_parenchyma_ml"]
    )
    center = tumor_center_voxel(first)
    expected = np.rint(np.asarray(tumor_center) / np.asarray(SPACING))
    assert np.all(np.abs(np.asarray(center) - expected) <= 1)


def test_side_survives_cropping_and_uses_anatomy() -> None:
    """Regression (review): cropping must not flip a left kidney to 'Right'."""

    from renal_explorer.analysis import analyze_case
    from renal_explorer.models import VolumeCase

    tumor_center = (CENTER_MM[0] - 23.0, CENTER_MM[1], CENTER_MM[2] + 32.0)
    segmentation, image, affine = phantom(tumor_center, 12.0)
    assert build_plan(segmentation, affine, SPACING, image)["summary"]["side"] == "Left"

    # Crop away the low-x half: the left kidney now sits right of the image centre.
    cropped = segmentation[12:, :, :]
    cropped_affine = affine.copy()
    cropped_affine[0, 3] = 12 * SPACING[0]
    plan = build_plan(cropped, cropped_affine, SPACING, image[12:])
    assert plan["summary"]["side"] == "Left"

    # Only one kidney and no great vessels: refuse to guess.
    single = segmentation.copy()
    single[100:, :, :] = 0  # remove the contralateral kidney
    assert build_plan(single, affine, SPACING, image)["summary"]["side"] == "Unknown"

    # One kidney plus an aorta at the midline: the aorta decides.
    single[78:84, 66:72, 20:70] = int(CanonicalLabel.AORTA)
    assert build_plan(single, affine, SPACING, image)["summary"]["side"] == "Left"

    case = VolumeCase("t", "c", image[12:], cropped, SPACING, cropped_affine)
    analysis = analyze_case(case)
    assert {kidney["laterality"] for kidney in analysis["kidney_components"]} == {"Left", "Right"}
    assert analysis["tumor_components"][0]["laterality"] == "Left"


def test_n_uses_nearest_surface_and_ranges_without_reviewed_sinus() -> None:
    """Regression (review): collecting system 9 mm + sinus 2 mm must not give N1."""

    from renal_explorer.planning import extract_renal_unit, nephrometry

    tumor_center = (CENTER_MM[0] - 14.0, CENTER_MM[1], CENTER_MM[2])
    segmentation, image, affine = phantom(tumor_center, 7.0)
    unit = extract_renal_unit(segmentation, affine, SPACING, image)

    renal = nephrometry(
        unit, {"Collecting system": {"gap_mm": 9.0}, "Renal sinus": {"gap_mm": 2.0}}
    )["renal"]
    assert renal["N"] is None
    assert renal["N_range"] == [1, 3]
    assert renal["N_estimate"] == 3 and renal["N_source"] == "Renal sinus (proxy)"

    renal = nephrometry(
        unit, {"Collecting system": {"gap_mm": 2.0}, "Renal sinus": {"gap_mm": 6.0}}
    )["renal"]
    assert renal["N"] == 3 and renal["N_source"] == "Collecting system"

    renal = nephrometry(unit, {"Collecting system": {"gap_mm": 5.0}})["renal"]
    assert renal["N"] is None and renal["N_range"] == [2, 3]


def test_sinus_proxy_ignores_low_density_parenchyma() -> None:
    """Regression (review): unenhanced medulla inside the kidney label is not sinus."""

    from renal_explorer.planning import extract_renal_unit

    tumor_center = (CENTER_MM[0] - 23.0, CENTER_MM[1], CENTER_MM[2] + 32.0)
    segmentation, image, affine = phantom(tumor_center, 12.0)
    # Mimic a corticomedullary phase: a dark (30 HU) medulla band inside the kidney label.
    medulla = _ellipsoid(CENTER_MM - [8.0, 0.0, 0.0], (8.0, 10.0, 35.0))
    medulla &= segmentation == int(CanonicalLabel.KIDNEY)
    image[medulla] = 30.0
    unit = extract_renal_unit(segmentation, affine, SPACING, image)
    assert unit.sinus is not None
    assert not np.any(unit.sinus & unit.kidney & (unit.image > -30))


def test_intrarenal_fragment_is_not_an_accessory_artery() -> None:
    """Regression (review): a disconnected intrarenal fragment got its own 24 % territory."""

    tumor_center = (CENTER_MM[0] - 23.0, CENTER_MM[1], CENTER_MM[2] + 32.0)
    segmentation, image, affine = phantom(tumor_center, 12.0, with_artery=True)
    fragment = _tube(
        (CENTER_MM[0] - 14.0, CENTER_MM[1] + 12.0, CENTER_MM[2] - 12.0),
        (CENTER_MM[0] - 16.0, CENTER_MM[1] + 12.0, CENTER_MM[2] - 34.0),
        1.5,
    )
    segmentation[fragment & (segmentation == int(CanonicalLabel.KIDNEY))] = int(
        CanonicalLabel.RENAL_ARTERY
    )
    territories = build_plan(segmentation, affine, SPACING, image)["summary"]["territories"]
    assert territories["ignored_fragments"] == 1
    assert territories["renal_artery_count"] == 1
    assert not any(branch["accessory"] for branch in territories["branches"])
    assert len(territories["branches"]) == 2
    assert territories["ischemia_estimate_reliable"] is False
    assert territories["warnings"]


def test_imported_vessels_may_replace_kidney_but_not_tumour() -> None:
    from types import SimpleNamespace

    from renal_explorer.plan_ui import planning_segmentation

    labels = np.zeros((4, 4, 4), np.uint8)
    labels[0] = int(CanonicalLabel.KIDNEY)
    labels[1] = int(CanonicalLabel.TUMOR)
    prediction = np.full((4, 4, 4), int(CanonicalLabel.RENAL_ARTERY), np.uint8)
    case = SimpleNamespace(segmentation=labels)
    merged = planning_segmentation(case, SimpleNamespace(segmentation=prediction))
    assert (merged[0] == int(CanonicalLabel.RENAL_ARTERY)).all()  # was discarded before
    assert (merged[1] == int(CanonicalLabel.TUMOR)).all()
    assert (merged[2] == int(CanonicalLabel.RENAL_ARTERY)).all()
