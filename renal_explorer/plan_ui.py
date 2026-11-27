"""Streamlit "Surgical plan" tab. Research-only partial nephrectomy planning."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .exports import plan_report_pdf, stl_zip
from .labels import LABEL_COLORS, CanonicalLabel
from .manifest import build_manifest, segmentation_source
from .planning import PlanningError, assemble_plan, plan_geometry
from .rendering import render_view
from .review import (
    ACCEPTED,
    QA_PASSED,
    REJECTED,
    REVIEW_CHECKS,
    REVIEWED,
    STATE_LABELS,
    UNREVIEWED,
    ReviewError,
    ReviewStore,
    automated_qa,
    overall_state,
    review_case_key,
    transition,
)
from .three_d import REFERENCE_QUALITY, surface_mesh

BRANCH_DEPTHS = {
    1: "Divisions (1st significant bifurcation)",
    2: "Segmental (2nd significant bifurcation)",
    3: "Sub-segmental (3rd)",
}
PLAN_STYLE = {
    "Kidney": ((45, 200, 90), 0.30),
    "Tumor": ((255, 65, 65), 0.95),
    "Resection volume": ((255, 150, 40), 0.35),
    "Renal sinus (proxy)": ((240, 220, 140), 0.25),
    "Renal artery": (LABEL_COLORS[CanonicalLabel.RENAL_ARTERY], 0.95),
    "Renal vein": (LABEL_COLORS[CanonicalLabel.RENAL_VEIN], 0.85),
    "Collecting system": (LABEL_COLORS[CanonicalLabel.COLLECTING_SYSTEM], 0.85),
}


@st.cache_resource(show_spinner=False, max_entries=2)
def cached_plan_geometry(
    dataset_key: str,
    case_id: str,
    segmentation_token: str,
    tumor_index: int,
    branch_depth: int,
    _segmentation: np.ndarray,
    _affine: np.ndarray,
    _spacing: tuple[float, float, float],
    _image: np.ndarray,
):
    del dataset_key, case_id, segmentation_token
    return plan_geometry(_segmentation, _affine, _spacing, _image, tumor_index, branch_depth)


@st.cache_data(show_spinner=False, max_entries=24)
def cached_plan_mesh(
    plan_key: str,
    structure: str,
    quality: str,
    _mask: np.ndarray,
    _affine: np.ndarray,
    _spacing: tuple[float, float, float],
):
    del plan_key, structure
    if _mask is None or not np.any(_mask):
        return None
    return surface_mesh(_mask.astype(np.uint8), _affine, _spacing, 1, quality)


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def planning_segmentation(case, imported_prediction) -> np.ndarray:
    """Dataset labels, with imported vessel predictions filled into unlabeled voxels."""

    if imported_prediction is None:
        return case.segmentation
    merged = case.segmentation.copy()
    prediction = imported_prediction.segmentation
    # Predicted vessels may replace background or kidney (the sinus/hilum often carries a
    # kidney label) but never tumour, cyst or other structures.
    fill = (prediction > 0) & np.isin(merged, (0, int(CanonicalLabel.KIDNEY)))
    merged[fill] = prediction[fill]
    return merged


def _rgb(color) -> str:
    red, green, blue = color
    return f"rgb({red},{green},{blue})"


def territory_vertex_colors(
    vertices: np.ndarray,
    territory_map: np.ndarray,
    affine: np.ndarray,
    branches: list[dict[str, Any]],
    default_color,
) -> list[str]:
    inverse = np.linalg.inv(affine)
    voxels = vertices @ inverse[:3, :3].T + inverse[:3, 3]
    index = np.clip(np.rint(voxels).astype(int), 0, np.asarray(territory_map.shape) - 1)
    ids = territory_map[index[:, 0], index[:, 1], index[:, 2]]
    lookup = np.array(
        [_rgb(default_color)] * (int(territory_map.max()) + 1), dtype=object
    )
    for branch in branches:
        if branch["id"] < lookup.size:
            lookup[branch["id"]] = _rgb(branch["color"])
    return lookup[ids].tolist()


def build_plan_figure(
    meshes: dict[str, dict[str, Any]],
    plan: dict[str, Any],
    show_territories: bool,
    show_distances: bool,
) -> go.Figure:
    figure = go.Figure()
    summary = plan["summary"]
    display = plan["display"]
    territories = summary["territories"]
    for name, mesh in meshes.items():
        if mesh is None:
            continue
        color, opacity = PLAN_STYLE[name]
        vertices, faces = mesh["vertices"], mesh["faces"]
        trace = {
            "x": vertices[:, 0],
            "y": vertices[:, 1],
            "z": vertices[:, 2],
            "i": faces[:, 0],
            "j": faces[:, 1],
            "k": faces[:, 2],
            "name": name,
            "opacity": opacity,
            "flatshading": False,
            "lighting": {"ambient": 0.55, "diffuse": 0.75, "specular": 0.25},
            "hovertemplate": f"{name}<extra></extra>",
            "showlegend": True,
        }
        if (
            name == "Kidney"
            and show_territories
            and territories
            and display["territory_map"] is not None
        ):
            trace["vertexcolor"] = territory_vertex_colors(
                vertices,
                display["territory_map"],
                display["affine"],
                territories["branches"],
                color,
            )
            trace["opacity"] = 0.55
            trace["name"] = "Kidney (arterial territories)"
        else:
            trace["color"] = _rgb(color)
        figure.add_trace(go.Mesh3d(**trace))

    if show_territories and territories and display["centerlines"]:
        colors = {branch["id"]: branch["color"] for branch in territories["branches"]}
        names = {branch["id"]: branch["name"] for branch in territories["branches"]}
        for lineage_id, points in display["centerlines"].items():
            figure.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="lines",
                    line={"color": _rgb(colors.get(lineage_id, (255, 255, 255))), "width": 6},
                    name=names.get(lineage_id, "Proximal artery"),
                    hoverinfo="name",
                    showlegend=lineage_id in names,
                )
            )

    if show_distances:
        for name, points in display["distance_lines"].items():
            gap = summary["distances"][name]["gap_mm"]
            middle = points.mean(axis=0)
            figure.add_trace(
                go.Scatter3d(
                    x=[points[0, 0], middle[0], points[1, 0]],
                    y=[points[0, 1], middle[1], points[1, 1]],
                    z=[points[0, 2], middle[2], points[1, 2]],
                    mode="lines+markers+text",
                    text=["", f"{name} {gap:.1f} mm", ""],
                    textfont={"size": 11, "color": "white"},
                    marker={"size": 3, "color": "white"},
                    line={"color": "white", "width": 3, "dash": "dash"},
                    name=f"Distance: {name}",
                    showlegend=False,
                    hoverinfo="text",
                )
            )

    figure.update_layout(
        height=680,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        scene={
            "aspectmode": "data",
            "xaxis_title": "Right ↔ Left (mm)",
            "yaxis_title": "Anterior ↔ Posterior (mm)",
            "zaxis_title": "Inferior ↔ Superior (mm)",
            "camera": {"eye": {"x": 1.4, "y": 1.4, "z": 0.9}},
        },
        uirevision="renal-plan",
    )
    return figure


def _l_text(renal: dict[str, Any]) -> str:
    low, high = renal.get("L_range") or (renal["L"], renal["L"])
    return str(renal["L"]) if low == high else f"{low}–{high} (est. {renal['L']})"


def _score_tables(summary: dict[str, Any]) -> None:
    scores = summary["scores"]
    renal = scores["renal"]
    padua = scores["padua"]
    renal_n = summary["scores"]["renal"]
    nearness_text = (
        f"{renal_n['N_gap_mm']:.1f} mm to {renal_n['N_source']}"
        if renal_n.get("N_gap_mm") is not None
        else "not assessable"
    )
    if renal_n["N"] is None:
        low, high = renal_n["N_range"]
        estimate = renal_n.get("N_estimate")
        n_text = f"{low}–{high}" + (
            f" (proxy suggests {estimate})" if estimate is not None else ""
        )
    else:
        n_text = str(renal_n["N"])
    left, right = st.columns(2)
    with left:
        st.markdown("**R.E.N.A.L. components**")
        st.dataframe(
            pd.DataFrame(
                [
                    ("R – max diameter", f"{scores['max_diameter_mm']:.1f} mm", renal["R"]),
                    (
                        "E – exophytic",
                        f"{100 * scores['exophytic_fraction']:.0f} % outside contour",
                        renal["E"],
                    ),
                    (
                        "N – nearness",
                        nearness_text,
                        n_text,
                    ),
                    (
                        "A – face",
                        f"{100 * scores['anterior_fraction']:.0f} % anterior",
                        renal["A"],
                    ),
                    (
                        "L – polar lines",
                        f"{100 * scores['between_polar_lines_fraction']:.0f} % between lines"
                        + (", crosses midline" if scores["crosses_midline"] else ""),
                        _l_text(renal),
                    ),
                    (
                        "h – hilar",
                        "touches main artery/vein" if renal["hilar"] else "—",
                        "h" if renal["hilar"] else ("?" if renal["hilar"] is None else "—"),
                    ),
                ],
                columns=["Item", "Measurement", "Points"],
            ).astype(str),
            hide_index=True,
            width="stretch",
        )
    with right:
        st.markdown("**PADUA items**")
        st.dataframe(
            pd.DataFrame(
                [
                    (
                        name,
                        (
                            f"? (est. {padua['renal_sinus_estimate']})"
                            if name == "Renal sinus" and padua.get("renal_sinus_estimate")
                            else "?"
                        )
                        if value is None
                        else value,
                    )
                    for name, value in padua["items"].items()
                ]
                + [("Face", padua["face"])],
                columns=["Item", "Points"],
            ).astype(str),
            hide_index=True,
            width="stretch",
        )


def render_plan_tab(
    *,
    adapter,
    dataset_key: str,
    case,
    analysis: dict[str, Any],
    imported_prediction,
    segmentation_token: str,
    ct_digest: str | None = None,
    segmentation_digest: str | None = None,
    patient_provenance: dict | None = None,
) -> None:
    st.caption(
        "Research prototype. Scores are automated approximations of R.E.N.A.L. and PADUA "
        "computed from the labelmaps; they are not validated and must be checked by a "
        "qualified clinician."
    )
    tumors = analysis.get("tumor_components", [])
    if case.segmentation is None or not tumors:
        st.info("A tumour label is required to plan a partial nephrectomy for this case.")
        return

    planning_labels = planning_segmentation(case, imported_prediction)
    has_artery = bool(np.any(planning_labels == int(CanonicalLabel.RENAL_ARTERY)))

    controls = st.columns(3)
    with controls[0]:
        tumor_index = st.selectbox(
            "Tumour",
            list(range(len(tumors))),
            format_func=lambda index: (
                f"Tumour {index + 1} · {tumors[index].get('laterality', '')} · "
                f"{tumors[index]['volume_ml']:.1f} mL"
            ),
        )
    with controls[1]:
        margin_mm = st.slider(
            "Resection margin (mm)",
            0,
            10,
            3,
            help="0 = enucleation. Healthy parenchyma within this distance of the tumour is "
            "counted as resected.",
        )
    with controls[2]:
        branch_depth = st.selectbox(
            "Arterial branch level",
            list(BRANCH_DEPTHS),
            format_func=BRANCH_DEPTHS.get,
            disabled=not has_artery,
            help="Needs renal artery labels (KiPA22) or an imported BA-Net prediction.",
        )

    plan_key = f"{dataset_key}_{case.case_id}_{segmentation_token}_{tumor_index}_{branch_depth}"
    if st.button("Compute surgical plan", type="primary", width="stretch", key=f"plan_{plan_key}"):
        st.session_state[f"plan_ready_{plan_key}"] = True
    if not st.session_state.get(f"plan_ready_{plan_key}", False):
        st.info(
            "Computes nephrometry, distances, contact surface, a resection simulation and — when "
            "renal arteries are available — approximate arterial territories."
        )
        return

    try:
        with st.spinner("Computing plan geometry on the server…"):
            geometry = cached_plan_geometry(
                dataset_key,
                case.case_id,
                segmentation_token,
                int(tumor_index),
                int(branch_depth),
                planning_labels,
                case.affine,
                case.spacing,
                case.image,
            )
    except PlanningError as exc:
        st.error(f"Plan could not be computed: {exc}")
        return
    plan = assemble_plan(geometry, float(margin_mm))
    resolution = case.metadata.get("resolution") or {"reduced": False}
    plan["summary"]["resolution"] = {
        "reduced": bool(resolution.get("reduced")),
        "spacing_mm": [float(value) for value in case.spacing],
        "full_spacing_mm": resolution.get("full_spacing_mm"),
        "strides": resolution.get("strides", [1, 1, 1]),
    }
    if plan["summary"]["resolution"]["reduced"]:
        st.warning(
            "Demo resolution: this plan is computed on a reduced grid "
            f"({', '.join(f'{value:.1f}' for value in case.spacing)} mm). Exports are marked "
            "as approximate."
        )
    summary = plan["summary"]
    scores = summary["scores"]
    resection = summary["resection"]

    cards = st.columns(4)
    cards[0].metric(
        "R.E.N.A.L.", scores["renal"]["score"], scores["renal"]["complexity"], delta_color="off"
    )
    cards[1].metric("PADUA", scores["padua"]["score"], scores["padua"]["risk"], delta_color="off")
    cards[2].metric(
        "Tumour", f"{scores['max_diameter_mm'] / 10:.1f} cm", scores["location"], delta_color="off"
    )
    cards[3].metric(
        "Kidney tissue kept",
        f"{resection['preserved_ipsilateral_pct']:.1f} %",
        f"{summary['side']} kidney · {resection['technique'].lower()}",
        delta_color="off",
    )

    for warning in resection["warnings"]:
        st.warning(
            f"{warning['structure']} is {warning['gap_mm']:.1f} mm from the tumour "
            f"({warning['level']})."
        )
    for name in resection.get("indeterminate", []):
        st.info(
            f"{name} relationship indeterminate: no reviewed outline. Check it on the CT; "
            "the experimental proxy distance is listed under the tables."
        )

    _score_tables(summary)

    left, right = st.columns(2)
    with left:
        st.markdown("**Distances from tumour surface**")
        if summary["distances"]:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Structure": name,
                            "Gap (mm)": round(item["gap_mm"], 1),
                            "Source": (
                                "experimental proxy (unreviewed)"
                                if item["source"] == "proxy"
                                else item["source"]
                            ),
                        }
                        for name, item in summary["distances"].items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        st.caption("Minimum gap between voxel-box surfaces of the two label masks.")
        contact = summary["contact_surface"]
        st.caption(
            f"Tumour–parenchyma contact surface: {contact['contact_cm2']:.1f} cm² of "
            f"{contact['tumor_surface_cm2']:.1f} cm² tumour surface."
        )
    with right:
        st.markdown("**Resection simulation**")
        rows = [
            ("Technique", resection["technique"]),
            ("Tumour volume", f"{resection['tumor_ml']:.1f} mL"),
            ("Kidney-labelled tissue removed", f"{resection['removed_parenchyma_ml']:.1f} mL"),
            ("Ipsilateral kidney tissue kept", f"{resection['preserved_ipsilateral_pct']:.1f} %"),
            (
                "Total kidney tissue kept",
                f"{resection['preserved_total_pct']:.1f} %"
                if resection["preserved_total_pct"] is not None
                else f"not shown: opposite kidney {resection['contralateral_status']}",
            ),
        ]
        if resection["ipsilateral_share_before_pct"] is not None:
            rows.append(
                (
                    "Ipsilateral volume share",
                    f"{resection['ipsilateral_share_before_pct']:.0f} % → "
                    f"{resection['ipsilateral_share_after_pct']:.0f} %",
                )
            )
        st.dataframe(
            pd.DataFrame(rows, columns=["Item", "Value"]), hide_index=True, width="stretch"
        )
        st.caption("Volume-based; not a prediction of postoperative eGFR.")

    territories = summary["territories"]
    if territories:
        st.markdown("**Arterial territories (approximation)**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Branch": branch["name"],
                        "Region": branch["region"],
                        "Parenchyma (mL)": round(branch["parenchyma_ml"], 1),
                        "Kidney (%)": round(branch["parenchyma_pct"], 0),
                        "Tumour supplied (%)": round(branch["tumor_pct"], 0),
                        "Feeds tumour": "Yes" if branch["feeds_tumor"] else "",
                    }
                    for branch in territories["branches"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        for warning in territories.get("warnings", []):
            st.warning(warning)
        if territories.get("ischemia_estimate_reliable"):
            st.info(
                f"{territories['renal_artery_count']} renal arter"
                f"{'y' if territories['renal_artery_count'] == 1 else 'ies'} reach the kidney. "
                f"Selective clamping of {', '.join(territories['feeding_branches']) or '—'} "
                f"would leave roughly {territories['selective_clamp_ischemic_pct']:.0f} % of "
                "parenchyma ischaemic versus 100 % with main-artery clamping (approximation)."
            )
        else:
            st.caption("Ischaemic-volume estimate withheld: the arterial tree looks incomplete.")
        st.caption(territories["method"] + ".")
    elif not has_artery:
        st.caption(
            "Arterial territories need renal artery labels: use a KiPA22 case or import a "
            "BA-Net prediction from the sidebar."
        )

    for note in scores["notes"] + [f"Renal sinus: {summary['sinus_source']}."]:
        st.caption(f"• {note}")

    st.markdown("**3D plan**")
    view_columns = st.columns(4)
    with view_columns[0]:
        quality = st.selectbox("Plan detail", ("Fast", "Balanced", "Detailed"), index=1)
    with view_columns[1]:
        show_territories = st.checkbox(
            "Colour territories", value=bool(territories), disabled=not territories
        )
    with view_columns[2]:
        show_distances = st.checkbox("Distance lines", value=True)
    with view_columns[3]:
        show_sinus = st.checkbox("Renal sinus proxy", value=False)

    display = plan["display"]
    build_key = f"plan3d_{plan_key}_{quality}"
    if st.button("Build 3D plan", key=f"button_{build_key}", width="stretch"):
        st.session_state[build_key] = True
    meshes: dict[str, dict[str, Any] | None] = {}
    export_sources: dict[str, tuple[str, np.ndarray]] = {}
    if st.session_state.get(build_key, False):
        sources = {
            "Kidney": display["kidney"],
            "Tumor": display["tumor"],
            "Resection volume": display["resection"],
            **display["structures"],
        }
        if show_sinus and display["sinus"] is not None:
            sources["Renal sinus (proxy)"] = display["sinus"]
        with st.spinner("Building plan surfaces…"):
            for name, mask in sources.items():
                key = f"{plan_key}_{margin_mm}" if name == "Resection volume" else plan_key
                meshes[name] = cached_plan_mesh(
                    key, name, quality, mask, display["affine"], display["spacing"]
                )
                export_sources[name] = (key, mask)
        st.plotly_chart(
            build_plan_figure(meshes, plan, show_territories, show_distances),
            config={"displaylogo": False, "scrollZoom": True},
        )

    # Reference (unsmoothed) surfaces: the authoritative geometry for QA and exports.
    reference_meshes = {
        name: cached_plan_mesh(
            key, name, REFERENCE_QUALITY, mask, display["affine"], display["spacing"]
        )
        for name, (key, mask) in export_sources.items()
    }
    case_label = f"{adapter.info.name} {case.case_id}"
    qa = automated_qa(
        reference_meshes, display["spacing"], bool(summary["resolution"]["reduced"])
    )
    review_key = review_case_key(ct_digest, segmentation_token)
    record = _render_review_panel(review_key, case_label, qa, patient_provenance is not None)
    state = overall_state(qa["status"], record)
    summary["model_review"] = {
        "state": state,
        "state_label": STATE_LABELS[state],
        "qa_status": qa["status"],
        "clinician_status": record.status,
        "reviewer": record.reviewer or None,
        "ct_sha256": ct_digest,
    }
    manifest = build_manifest(
        case_id=case.case_id,
        dataset=adapter.info.name,
        ct_sha256=ct_digest,
        segmentation_sha256=segmentation_digest,
        image_shape=case.image.shape,
        spacing=case.spacing,
        affine=case.affine,
        resolution=case.metadata.get("resolution"),
        segmentation=segmentation_source(case, imported_prediction, patient_provenance),
        reconstruction={
            "algorithm": "Label masks -> signed distance -> Lewiner marching cubes",
            "exported_surfaces": "reference (native planning grid, no smoothing/decimation)",
            "planning_grid_spacing_mm": [float(value) for value in display["spacing"]],
            "display_detail": quality,
            "parameters": {
                "tumor_index": int(tumor_index),
                "margin_mm": float(margin_mm),
                "branch_depth": int(branch_depth),
            },
        },
        mesh_quality={
            name: mesh["quality"] for name, mesh in reference_meshes.items() if mesh is not None
        },
        qa=qa,
        review=record.to_dict(),
        state=state,
        limitations=record.limitations,
    )
    summary["manifest"] = manifest

    downloads = st.columns(3)
    # Slice from the tumour's world position mapped onto the case's current grid.
    inverse = np.linalg.inv(case.affine)
    center_world = np.append(np.asarray(summary["tumor_center_mm"], dtype=float), 1.0)
    center = np.rint((inverse @ center_world)[:3]).astype(int)
    reduced_suffix = "_demo-resolution" if summary["resolution"]["reduced"] else ""
    axial = render_view(
        image=case.image,
        segmentation=case.segmentation,
        spacing=case.spacing,
        orientation="Axial",
        index=int(np.clip(center[2], 0, case.image.shape[2] - 1)),
        level=60,
        width=360,
        visible_labels=[CanonicalLabel.KIDNEY, CanonicalLabel.TUMOR, CanonicalLabel.CYST],
        overlay_mode="Outline",
        opacity=0.4,
    )
    with downloads[0]:
        st.download_button(
            "Plan report (PDF)",
            data=plan_report_pdf(
                summary, case_label, images=[("Axial slice through tumour centre", axial)]
            ),
            file_name=(
                f"{dataset_key}_{case.case_id}_tumor{tumor_index + 1}_plan{reduced_suffix}.pdf"
            ),
            mime="application/pdf",
            width="stretch",
        )
    with downloads[1]:
        st.download_button(
            "Plan data (JSON)",
            data=json.dumps(summary, indent=2, default=_json_default),
            file_name=(
                f"{dataset_key}_{case.case_id}_tumor{tumor_index + 1}_plan{reduced_suffix}.json"
            ),
            mime="application/json",
            width="stretch",
        )
    with downloads[2]:
        # Exports use the unsmoothed reference surfaces, never the display previews.
        available = {name: mesh for name, mesh in reference_meshes.items() if mesh is not None}
        st.download_button(
            "3D surfaces (STL zip)",
            data=(
                stl_zip(available, resolution=summary["resolution"], manifest=manifest)
                if available
                else b""
            ),
            file_name=(
                f"{dataset_key}_{case.case_id}_tumor{tumor_index + 1}"
                f"_surfaces{reduced_suffix}.zip"
            ),
            mime="application/zip",
            width="stretch",
            disabled=not available,
            help=(
                "Build the 3D plan first. Unsmoothed reference surfaces from the voxel "
                "labels, in shared RAS millimetre coordinates."
            ),
        )


def _review_store(patient_mode: bool) -> ReviewStore:
    """Reviews persist on disk on private installs; on the public demo, per session only."""

    configured = os.environ.get("RENAL_REVIEW_DIR")
    if configured:
        return ReviewStore(configured)
    if patient_mode:
        root = os.environ.get("RENAL_PATIENT_ROOT", "~/RenalPlanner/cases")
        return ReviewStore(Path(root).expanduser().parent / "reviews")
    if "review_store" not in st.session_state:
        st.session_state["review_store"] = ReviewStore(None)
    return st.session_state["review_store"]


def _render_review_panel(review_key: str, case_label: str, qa: dict, patient_mode: bool):
    store = _review_store(patient_mode)
    record = store.load(review_key, case_label)
    state = overall_state(qa["status"], record)
    st.markdown("**Model review**")
    message = f"{STATE_LABELS[state]}"
    if state == ACCEPTED:
        st.success(message)
    elif state in (REJECTED, "failed"):
        st.error(message)
    elif state == REVIEWED:
        st.info(message)
    else:
        st.warning(message)
    qa_label = {"pending": "not run: build the 3D plan", "passed": "passed", "failed": "failed"}
    with st.expander(f"Automated geometry QA: {qa_label[qa['status']]}", expanded=False):
        st.caption(
            "Technical checks on the reference surfaces only. Passing them does not mean the "
            "anatomy is correct. Thresholds are engineering criteria."
        )
        if qa["checks"]:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Structure": item["structure"],
                            "Check": item["check"],
                            "Result": "pass" if item["passed"] else "FAIL",
                            "Value": item["detail"],
                        }
                        for item in qa["checks"]
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        for warning in qa["warnings"]:
            st.caption(warning)
    with st.expander("Clinician review", expanded=record.status == UNREVIEWED):
        if not store.persistent:
            st.caption("Online demo: reviews are kept for this browser session only.")
        reviewer = st.text_input("Reviewer", value=record.reviewer, key=f"rv_name_{review_key}")
        checks = {
            key: st.checkbox(
                label, value=record.checks.get(key, False), key=f"rv_{key}_{review_key}"
            )
            for key, label in REVIEW_CHECKS.items()
        }
        notes = st.text_area("Notes", value=record.notes, key=f"rv_notes_{review_key}")
        limitations = st.text_area(
            "Limitations (one per line)",
            value="\n".join(record.limitations),
            key=f"rv_lim_{review_key}",
        )
        actions = st.columns(4)
        requested = None
        if actions[0].button("Mark reviewed", key=f"rv_do_rev_{review_key}", width="stretch"):
            requested = REVIEWED
        if actions[1].button(
            "Accept for research",
            key=f"rv_do_acc_{review_key}",
            width="stretch",
            disabled=record.status != REVIEWED or qa["status"] != QA_PASSED,
        ):
            requested = ACCEPTED
        if actions[2].button("Reject", key=f"rv_do_rej_{review_key}", width="stretch"):
            requested = REJECTED
        if actions[3].button(
            "Reopen",
            key=f"rv_do_reo_{review_key}",
            width="stretch",
            disabled=record.status == UNREVIEWED,
        ):
            requested = UNREVIEWED
        if requested:
            try:
                transition(
                    record,
                    requested,
                    reviewer=reviewer,
                    qa_status=qa["status"],
                    checks=checks,
                    notes=notes,
                    limitations=limitations.splitlines(),
                )
                store.save(record)
                st.rerun()
            except ReviewError as exc:
                st.error(str(exc))
        if record.history:
            st.caption(
                "History: "
                + " · ".join(
                    f"{item['from']}→{item['to']} by {item['reviewer']}"
                    for item in record.history[-5:]
                )
            )
    return record
