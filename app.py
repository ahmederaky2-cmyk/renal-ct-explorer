from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Renal CT Research Explorer",
    page_icon="🩻",
    layout="wide",
)

from renal_explorer.analysis import analyze_case, histogram_table  # noqa: E402
from renal_explorer.datasets import DATASETS, DatasetError, get_adapter  # noqa: E402
from renal_explorer.labels import LABEL_COLORS, LABEL_NAMES, CanonicalLabel  # noqa: E402
from renal_explorer.models import CaseFiles  # noqa: E402
from renal_explorer.remote_data import (  # noqa: E402
    RemoteDataError,
    demo_case_ids,
    demo_case_label,
    ensure_demo_case,
)
from renal_explorer.rendering import (  # noqa: E402
    VIEW_AXES,
    legend_html,
    render_view,
)
from renal_explorer.three_d import (  # noqa: E402
    candidate_vessel_mesh,
    surface_mesh,
)
from renal_explorer.vessel_ml import (  # noqa: E402
    VesselModelError,
    predict_vessels,
    resolve_model_bundle,
)

WINDOW_PRESETS = {
    "Abdomen": (60, 360),
    "Soft tissue": (40, 400),
    "Angio": (300, 600),
    "Liver": (30, 150),
    "Bone": (400, 1800),
    "Lung": (-600, 1500),
}

SURFACE_QUALITIES = ("Fast", "Balanced", "Detailed")
LABEL_OPACITY = {
    CanonicalLabel.KIDNEY: 0.32,
    CanonicalLabel.TUMOR: 0.95,
    CanonicalLabel.CYST: 0.72,
    CanonicalLabel.RENAL_ARTERY: 0.95,
    CanonicalLabel.RENAL_VEIN: 0.90,
    CanonicalLabel.AORTA: 0.65,
    CanonicalLabel.IVC: 0.65,
}


st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; max-width: 1600px;}
      div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: .55rem;
        padding: .65rem .8rem;
      }
      .dataset-note {
        border-left: 4px solid #4f8cff;
        padding: .65rem .9rem;
        background: rgba(79,140,255,.08);
        border-radius: .25rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60, show_spinner=False)
def discover_cases(dataset_key: str, root: str) -> list[tuple[str, str, str]]:
    adapter = get_adapter(dataset_key)
    return [
        (
            files.case_id,
            str(files.image_path),
            str(files.segmentation_path) if files.segmentation_path else "",
        )
        for files in adapter.list_cases(Path(root).expanduser())
    ]


@st.cache_resource(show_spinner=False, max_entries=1)
def load_case_bundle(
    dataset_key: str,
    case_id: str,
    image_path: str,
    segmentation_path: str,
    image_signature: tuple[int, int],
    segmentation_signature: tuple[int, int] | None,
):
    del image_signature, segmentation_signature
    adapter = get_adapter(dataset_key)
    files = CaseFiles(
        case_id=case_id,
        image_path=Path(image_path),
        segmentation_path=Path(segmentation_path) if segmentation_path else None,
    )
    case = adapter.load_case(files)
    return case, analyze_case(case)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_label_surface(
    dataset_key: str,
    case_id: str,
    segmentation_signature: tuple[int, int] | None,
    label_value: int,
    quality: str,
    _segmentation: np.ndarray,
    _affine: np.ndarray,
    _spacing: tuple[float, float, float],
):
    del dataset_key, case_id, segmentation_signature
    return surface_mesh(_segmentation, _affine, _spacing, label_value, quality)


@st.cache_data(show_spinner=False, max_entries=3)
def cached_candidate_vessels(
    dataset_key: str,
    case_id: str,
    image_signature: tuple[int, int],
    segmentation_signature: tuple[int, int] | None,
    sensitivity: str,
    detail: str,
    _image: np.ndarray,
    _segmentation: np.ndarray,
    _affine: np.ndarray,
    _spacing: tuple[float, float, float],
):
    del dataset_key, case_id, image_signature, segmentation_signature
    return candidate_vessel_mesh(
        _image,
        _segmentation,
        _affine,
        _spacing,
        sensitivity=sensitivity,
        detail=detail,
    )


@st.cache_resource(show_spinner=False, max_entries=1)
def cached_vessel_model_bundle(
    configured_directory: str,
    manifest_url: str,
    model_url: str,
):
    del manifest_url, model_url
    model_directory = Path(configured_directory).expanduser() if configured_directory else None
    return resolve_model_bundle(model_directory)


@st.cache_data(show_spinner=False, max_entries=2)
def cached_ml_vessels(
    dataset_key: str,
    case_id: str,
    image_signature: tuple[int, int],
    segmentation_signature: tuple[int, int] | None,
    model_sha256: str,
    _image: np.ndarray,
    _segmentation: np.ndarray | None,
    _spacing: tuple[float, float, float],
    _bundle,
):
    del (
        dataset_key,
        case_id,
        image_signature,
        segmentation_signature,
        model_sha256,
    )
    return predict_vessels(
        _image,
        _spacing,
        _segmentation,
        _bundle,
    )


def build_3d_figure(
    label_meshes: list[tuple[CanonicalLabel, dict]],
    vessel_candidates: dict | None,
) -> go.Figure:
    figure = go.Figure()

    def add_centerline(mesh: dict, name: str, color: str) -> None:
        points = mesh.get("centerline_points")
        if points is None or len(points) == 0:
            return
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="lines",
                name=f"{name} centerline",
                line={"color": color, "width": 4},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for label, mesh in label_meshes:
        vertices = mesh["vertices"]
        faces = mesh["faces"]
        red, green, blue = LABEL_COLORS[label]
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                name=LABEL_NAMES[label],
                color=f"rgb({red},{green},{blue})",
                opacity=LABEL_OPACITY[label],
                flatshading=False,
                lighting={"ambient": 0.55, "diffuse": 0.75, "specular": 0.25},
                hovertemplate=f"{LABEL_NAMES[label]}<extra></extra>",
            )
        )
        add_centerline(mesh, LABEL_NAMES[label], "rgba(255,255,255,0.88)")

    if vessel_candidates is not None:
        vertices = vessel_candidates["vertices"]
        faces = vessel_candidates["faces"]
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                name="Candidate vessels (experimental)",
                color="rgb(0,225,255)",
                opacity=0.88,
                flatshading=False,
                lighting={"ambient": 0.45, "diffuse": 0.85, "specular": 0.5},
                hovertemplate="Candidate vessel<extra></extra>",
            )
        )
        add_centerline(
            vessel_candidates,
            "Candidate vessel",
            "rgba(255,255,255,0.82)",
        )

    figure.update_layout(
        height=720,
        margin={"l": 0, "r": 0, "t": 35, "b": 0},
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
        scene={
            "aspectmode": "data",
            "xaxis_title": "Right ↔ Left (mm)",
            "yaxis_title": "Anterior ↔ Posterior (mm)",
            "zaxis_title": "Inferior ↔ Superior (mm)",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.1}},
        },
        uirevision="renal-3d",
    )
    return figure


def file_signature(path: str) -> tuple[int, int] | None:
    if not path:
        return None
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


def root_default(dataset_key: str) -> str:
    base = Path(os.environ.get("RENAL_DATA_ROOT", "/data"))
    return str(base / dataset_key)


def component_dataframe(items: list[dict]) -> pd.DataFrame:
    rows = []
    for item in items:
        rows.append(
            {
                "Side": item.get("laterality", ""),
                "Volume (mL)": round(item.get("volume_ml", 0.0), 2),
                "Equivalent diameter (mm)": round(
                    item.get("equivalent_diameter_mm", 0.0), 1
                ),
                "Mean HU": round(item.get("mean_hu", 0.0), 1),
                "HU IQR": round(item.get("iqr_hu", 0.0), 1),
                "Nearest kidney": item.get("nearest_kidney", ""),
                "Centroid distance (mm)": (
                    round(item["kidney_centroid_distance_mm"], 1)
                    if "kidney_centroid_distance_mm" in item
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def render_missing_dataset(adapter, dataset_root: str) -> None:
    st.title("🩻 Renal CT Research Explorer")
    st.warning(f"No readable cases were found in `{dataset_root}`.")
    st.markdown(
        f"""
        <div class="dataset-note">
          <b>{adapter.info.name}</b><br>
          {adapter.info.description}<br><br>
          Expected layout: <code>{adapter.info.expected_layout}</code><br>
          Source: <a href="{adapter.info.source_url}">{adapter.info.source_url}</a><br>
          License: {adapter.info.license_name}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if adapter.info.key == "kits23":
        st.code(
            "git clone https://github.com/neheller/kits23\n"
            "cd kits23 && pip install -e .\n"
            "kits23_download_data",
            language="bash",
        )
    st.info(
        "For an online deployment, place the dataset on the server's persistent "
        "data volume and set RENAL_DATA_ROOT to that volume. Users do not need a local copy."
    )


def main() -> None:
    dataset_keys = list(DATASETS)
    missing_server_dataset = False
    with st.sidebar:
        st.title("Dataset")
        access_mode = st.radio(
            "Data access",
            ("Online demo", "Server datasets"),
            index=1 if "RENAL_DATA_ROOT" in os.environ else 0,
            help="Online demo retrieves only the selected public KiTS23 case.",
        )

        if access_mode == "Online demo":
            dataset_key = "kits23"
            adapter = get_adapter(dataset_key)
            case_id = st.selectbox(
                "Public case",
                demo_case_ids(),
                format_func=demo_case_label,
            )
            dataset_root = str(
                Path(os.environ.get("RENAL_DEMO_CACHE", "/tmp/renal-ct-explorer-demo"))
                / dataset_key
            )
            try:
                with st.spinner("Retrieving the selected public case…"):
                    files = ensure_demo_case(
                        case_id,
                        Path(os.environ.get("RENAL_DEMO_CACHE", "/tmp/renal-ct-explorer-demo")),
                    )
                image_path = str(files.image_path)
                segmentation_path = str(files.segmentation_path or "")
            except RemoteDataError as exc:
                st.error(str(exc))
                st.stop()
            st.caption("Downloaded temporarily on the hosted server")
        else:
            dataset_key = st.selectbox(
                "Source",
                dataset_keys,
                format_func=lambda key: DATASETS[key].info.name,
            )
            adapter = get_adapter(dataset_key)
            dataset_root = st.text_input(
                "Server dataset folder",
                value=root_default(dataset_key),
                key=f"root_{dataset_key}",
            )
            cases = discover_cases(dataset_key, dataset_root)
            if not cases:
                missing_server_dataset = True
                case_id = image_path = segmentation_path = ""
            else:
                case_lookup = {
                    found_id: (image, segmentation)
                    for found_id, image, segmentation in cases
                }
                case_id = st.selectbox("Case", list(case_lookup))
                image_path, segmentation_path = case_lookup[case_id]

        st.caption(f"License: {adapter.info.license_name}")
        st.markdown(f"[Official source]({adapter.info.source_url})")

    if missing_server_dataset:
        render_missing_dataset(adapter, dataset_root)
        st.stop()

    try:
        with st.spinner(f"Loading {case_id} on the server…"):
            case, analysis = load_case_bundle(
                dataset_key,
                case_id,
                image_path,
                segmentation_path,
                file_signature(image_path),
                file_signature(segmentation_path),
            )
    except (DatasetError, OSError, ValueError) as exc:
        st.error(f"The case could not be loaded: {exc}")
        st.stop()

    available_labels: list[CanonicalLabel] = []
    if case.segmentation is not None:
        present = set(int(value) for value in np.unique(case.segmentation))
        available_labels = [
            label
            for label in CanonicalLabel
            if label != CanonicalLabel.BACKGROUND and int(label) in present
        ]

    with st.sidebar:
        st.divider()
        with st.expander("Display", expanded=True):
            preset_name = st.selectbox("CT window", list(WINDOW_PRESETS))
            preset_level, preset_width = WINDOW_PRESETS[preset_name]
            level = st.slider(
                "Window level (HU)",
                -1000,
                1000,
                preset_level,
                key=f"level_{preset_name}",
            )
            width = st.slider(
                "Window width (HU)",
                1,
                3000,
                preset_width,
                key=f"width_{preset_name}",
            )
            overlay_mode = st.selectbox("Overlay", ["Fill", "Outline", "Off"])
            opacity = st.slider(
                "Overlay opacity",
                0.0,
                1.0,
                0.4,
                disabled=overlay_mode != "Fill",
            )
            selected_names = st.multiselect(
                "Structures",
                [LABEL_NAMES[label] for label in available_labels],
                default=[LABEL_NAMES[label] for label in available_labels],
            )
            visible_labels = [
                label for label in available_labels if LABEL_NAMES[label] in selected_names
            ]

        with st.expander("Navigation", expanded=True):
            link_views = st.checkbox("Link views", value=True)
            show_crosshairs = st.checkbox("Show linked crosshairs", value=True)
            if link_views:
                slice_percentage = st.slider("Position (%)", 0, 100, 50)
                indices = {
                    orientation: int(
                        round(
                            (case.image.shape[axis] - 1)
                            * (slice_percentage / 100.0)
                        )
                    )
                    for orientation, axis in VIEW_AXES.items()
                }
            else:
                indices = {
                    orientation: st.slider(
                        f"{orientation} slice",
                        0,
                        case.image.shape[axis] - 1,
                        case.image.shape[axis] // 2,
                        key=f"slice_{dataset_key}_{case_id}_{orientation}",
                    )
                    for orientation, axis in VIEW_AXES.items()
                }

        if st.button("Clear memory cache", width="stretch"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    st.title("🩻 Renal CT Research Explorer")
    st.caption(
        f"{adapter.info.name} · {case.case_id} · "
        f"{case.image.shape[0]} × {case.image.shape[1]} × {case.image.shape[2]} voxels · "
        f"spacing {case.spacing[0]:.2f} × {case.spacing[1]:.2f} × {case.spacing[2]:.2f} mm"
    )

    regions = analysis["regions"]
    kidney = regions.get("Kidney", {})
    tumor = regions.get("Tumor", {})
    cyst = regions.get("Cyst", {})
    artery = regions.get("Renal artery", {})

    metric_columns = st.columns(4)
    metrics = (
        ("Kidney volume", f"{kidney.get('volume_ml', 0):.1f} mL"),
        ("Tumor volume", f"{tumor.get('volume_ml', 0):.1f} mL"),
        ("Cyst volume", f"{cyst.get('volume_ml', 0):.1f} mL"),
        ("Renal artery volume", f"{artery.get('volume_ml', 0):.1f} mL"),
    )
    for column, (label, value) in zip(metric_columns, metrics):  # noqa: B905
        column.metric(label, value)

    st.subheader("Multiplanar views")
    view_columns = st.columns(3)
    for column, orientation in zip(  # noqa: B905
        view_columns, ("Axial", "Coronal", "Sagittal")
    ):
        axis = VIEW_AXES[orientation]
        index = indices[orientation]
        with column:
            st.markdown(f"**{orientation}**")
            view = render_view(
                image=case.image,
                segmentation=case.segmentation,
                spacing=case.spacing,
                orientation=orientation,
                index=index,
                level=level,
                width=width,
                visible_labels=visible_labels,
                overlay_mode=overlay_mode,
                opacity=opacity,
                linked_indices=indices if link_views and show_crosshairs else None,
            )
            st.image(view, width="stretch")
            st.caption(
                f"Slice {index + 1}/{case.image.shape[axis]} · "
                f"voxel-axis position {index * case.spacing[axis]:.1f} mm"
            )
    if visible_labels and overlay_mode != "Off":
        st.markdown(legend_html(visible_labels), unsafe_allow_html=True)

    tab_measurements, tab_3d, tab_histogram, tab_export, tab_source = st.tabs(
        ["Measurements", "Interactive 3D", "HU histogram", "Export", "Source"]
    )
    with tab_measurements:
        if not regions:
            st.info("This case has no mapped segmentation file.")
        else:
            region_rows = []
            for name, stats in regions.items():
                region_rows.append(
                    {
                        "Structure": name,
                        "Volume (mL)": round(stats["volume_ml"], 2),
                        "Mean HU": round(stats["mean_hu"], 1),
                        "Median HU": round(stats["median_hu"], 1),
                        "HU IQR": round(stats["iqr_hu"], 1),
                        "Equivalent diameter (mm)": round(
                            stats["equivalent_diameter_mm"], 1
                        ),
                    }
                )
            st.dataframe(pd.DataFrame(region_rows), hide_index=True, width="stretch")

            if analysis["kidney_components"]:
                st.markdown("**Kidney components**")
                st.dataframe(
                    component_dataframe(analysis["kidney_components"]),
                    hide_index=True,
                    width="stretch",
                )
            if analysis["tumor_components"]:
                st.markdown("**Tumor components**")
                st.dataframe(
                    component_dataframe(analysis["tumor_components"]),
                    hide_index=True,
                    width="stretch",
                )
        st.caption(
            "Laterality is derived from connected components after canonical RAS orientation. "
            "Measurements are research outputs and are not clinically validated."
        )

    with tab_3d:
        if case.segmentation is None:
            st.info("A segmentation is required to build the 3D surface model.")
        else:
            configured_model_directory = os.environ.get("RENAL_VESSEL_MODEL_DIR", "")
            configured_manifest_url = os.environ.get("RENAL_VESSEL_MANIFEST_URL", "")
            configured_model_url = os.environ.get("RENAL_VESSEL_MODEL_URL", "")
            model_error = None
            try:
                vessel_model_bundle = cached_vessel_model_bundle(
                    configured_model_directory,
                    configured_manifest_url,
                    configured_model_url,
                )
            except VesselModelError as exc:
                vessel_model_bundle = None
                model_error = str(exc)

            control_columns = st.columns(4)
            with control_columns[0]:
                detail = st.selectbox(
                    "3D detail",
                    SURFACE_QUALITIES,
                    index=1,
                    help="Balanced is recommended for free hosting.",
                )
            has_renal_vessel_labels = any(
                label in available_labels
                for label in (CanonicalLabel.RENAL_ARTERY, CanonicalLabel.RENAL_VEIN)
            )
            with control_columns[1]:
                include_generated_vessels = st.checkbox(
                    "Add generated vessels",
                    value=not has_renal_vessel_labels,
                    disabled=has_renal_vessel_labels,
                    help=(
                        "Uses a validated KiPA22 model when installed, or an explicitly "
                        "experimental image filter as a fallback."
                    ),
                )
            with control_columns[2]:
                vessel_methods = []
                if vessel_model_bundle is not None:
                    vessel_methods.append("KiPA22 3D U-Net (ML)")
                vessel_methods.append("Image filter (not ML)")
                vessel_method = st.selectbox(
                    "Generation method",
                    vessel_methods,
                    disabled=(
                        has_renal_vessel_labels or not include_generated_vessels
                    ),
                )
            with control_columns[3]:
                vessel_sensitivity = st.selectbox(
                    "Filter sensitivity",
                    ("Strict", "Balanced", "Sensitive"),
                    index=1,
                    disabled=(
                        not include_generated_vessels
                        or vessel_method == "KiPA22 3D U-Net (ML)"
                    ),
                )

            if not has_renal_vessel_labels and vessel_model_bundle is None:
                reason = model_error or (
                    "No validated KiPA22 ONNX model is installed on this server."
                )
                st.info(
                    f"Real vessel ML is currently unavailable: {reason} The selectable "
                    "image-filter fallback is not presented as machine learning."
                )

            label_key = "-".join(str(int(label)) for label in visible_labels) or "none"
            build_key = (
                f"three_d_ready_{dataset_key}_{case_id}_{detail}_{label_key}_"
                f"{include_generated_vessels}_{vessel_method}_{vessel_sensitivity}"
            )
            if st.button(
                "Build interactive 3D",
                key=f"build_{build_key}",
                type="primary",
                width="stretch",
            ):
                st.session_state[build_key] = True

            if not st.session_state.get(build_key, False):
                st.info(
                    "Build the rotatable physical-coordinate model when needed. "
                    "It is generated on the server and cached for later visitors."
                )
            else:
                with st.spinner("Building 3D surfaces and vessel layers…"):
                    label_meshes = []
                    for label in visible_labels:
                        mesh = cached_label_surface(
                            dataset_key,
                            case_id,
                            file_signature(segmentation_path),
                            int(label),
                            detail,
                            case.segmentation,
                            case.affine,
                            case.spacing,
                        )
                        if mesh is not None:
                            label_meshes.append((label, mesh))

                    vessel_candidates = None
                    ml_prediction = None
                    if (
                        include_generated_vessels
                        and vessel_method == "KiPA22 3D U-Net (ML)"
                        and vessel_model_bundle is not None
                    ):
                        ml_prediction = cached_ml_vessels(
                            dataset_key,
                            case_id,
                            file_signature(image_path),
                            file_signature(segmentation_path),
                            vessel_model_bundle.manifest.artifact_sha256,
                            case.image,
                            case.segmentation,
                            case.spacing,
                            vessel_model_bundle,
                        )
                        for vessel_label in (
                            CanonicalLabel.RENAL_ARTERY,
                            CanonicalLabel.RENAL_VEIN,
                        ):
                            vessel_mesh = surface_mesh(
                                ml_prediction.segmentation,
                                case.affine,
                                case.spacing,
                                int(vessel_label),
                                detail,
                            )
                            if vessel_mesh is not None:
                                label_meshes.append((vessel_label, vessel_mesh))
                    elif include_generated_vessels:
                        vessel_candidates = cached_candidate_vessels(
                            dataset_key,
                            case_id,
                            file_signature(image_path),
                            file_signature(segmentation_path),
                            vessel_sensitivity,
                            detail,
                            case.image,
                            case.segmentation,
                            case.affine,
                            case.spacing,
                        )

                if not label_meshes and vessel_candidates is None:
                    st.warning("No surface could be generated with the current selections.")
                else:
                    figure = build_3d_figure(label_meshes, vessel_candidates)
                    st.plotly_chart(
                        figure,
                        config={"displaylogo": False, "scrollZoom": True},
                    )
                    triangle_count = sum(
                        int(mesh["faces"].shape[0]) for _label, mesh in label_meshes
                    )
                    if vessel_candidates is not None:
                        triangle_count += int(vessel_candidates["faces"].shape[0])
                    st.caption(f"Rendered {triangle_count:,} surface triangles in RAS millimetres.")

                    quality_rows = []
                    for label, mesh in label_meshes:
                        quality = mesh["quality"]
                        quality_rows.append(
                            {
                                "Structure": LABEL_NAMES[label],
                                "Triangles": quality["triangles"],
                                "Volume error": f"{quality['volume_error_pct']:.2f}%",
                                "Surface area": f"{quality['surface_area_cm2']:.1f} cm²",
                                "Smooth p95": (
                                    f"{quality['smoothing_displacement_p95_mm']:.2f} mm"
                                ),
                                "Mesh spacing": f"{quality['target_spacing_mm']:.2f} mm",
                                "Watertight": "Yes" if quality["watertight"] else "No",
                                "Components": "—",
                            }
                        )
                    if vessel_candidates is not None:
                        quality = vessel_candidates["quality"]
                        quality_rows.append(
                            {
                                "Structure": "Candidate vessels",
                                "Triangles": quality["triangles"],
                                "Volume error": f"{quality['volume_error_pct']:.2f}%",
                                "Surface area": f"{quality['surface_area_cm2']:.1f} cm²",
                                "Smooth p95": (
                                    f"{quality['smoothing_displacement_p95_mm']:.2f} mm"
                                ),
                                "Mesh spacing": f"{quality['target_spacing_mm']:.2f} mm",
                                "Watertight": "Yes" if quality["watertight"] else "No",
                                "Components": str(quality["candidate_components"]),
                            }
                        )
                    with st.expander("3D quality report", expanded=True):
                        st.dataframe(
                            pd.DataFrame(quality_rows),
                            hide_index=True,
                            width="stretch",
                        )
                        st.caption(
                            "Volume is checked against the source labelmap; smoothing is "
                            "limited by physical displacement and volume error. Watertight "
                            "requires zero boundary, non-manifold, and degenerate edges."
                        )
                    if ml_prediction is not None:
                        provenance = ml_prediction.provenance
                        st.success(
                            f"ML inference: {provenance['model']} "
                            f"v{provenance['version']} · trained on "
                            f"{provenance['dataset']} · mean predicted-vessel confidence "
                            f"{provenance['mean_vessel_confidence']:.1%}."
                        )
                        st.caption(
                            "This is a trained model output, but remains research-only and "
                            "must be externally validated for the scanner, contrast phase, "
                            "and patient population where it is used."
                        )

            if has_renal_vessel_labels:
                st.success(
                    "This case contains mapped renal artery/vein annotations; they are "
                    "rendered directly when selected in Structures."
                )
            elif vessel_method == "KiPA22 3D U-Net (ML)":
                st.warning(
                    "The ML model is trained on cropped KiPA22 CTA. Predictions on KiTS23 "
                    "or another contrast phase are out-of-domain and are not ground truth."
                )
            else:
                st.warning(
                    "This dataset has no renal vessel labels. Cyan structures are "
                    "Frangi/Sato connectivity-filtered candidates only: they are not "
                    "validated and cannot distinguish "
                    "arteries from veins. Use KiPA22 for labeled renal artery/vein surfaces."
                )

    with tab_histogram:
        histogram = pd.DataFrame(histogram_table(case)).set_index("HU")
        st.line_chart(histogram)
        st.caption(
            "The vertical axis is log(1 + voxel count); HU values are limited to −1000…1000."
        )

    with tab_export:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "case": case.json_metadata(),
            "analysis": analysis,
            "research_use_only": True,
        }
        st.download_button(
            "Download JSON report",
            data=json.dumps(report, indent=2),
            file_name=f"{dataset_key}_{case_id}_report.json",
            mime="application/json",
            width="stretch",
        )
        if regions:
            csv_rows = [dict(Structure=name, **stats) for name, stats in regions.items()]
            st.download_button(
                "Download region statistics CSV",
                data=pd.DataFrame(csv_rows).to_csv(index=False),
                file_name=f"{dataset_key}_{case_id}_statistics.csv",
                mime="text/csv",
                width="stretch",
            )

    with tab_source:
        st.markdown(f"**Dataset:** {adapter.info.name}")
        st.write(adapter.info.description)
        st.markdown(f"**License:** {adapter.info.license_name}")
        st.markdown(f"[Official dataset source]({adapter.info.source_url})")
        st.code(adapter.info.expected_layout, language="text")
        if case.source_labels:
            st.write("Source label values found:", list(case.source_labels))

    st.divider()
    st.caption("Research software · Not a medical device · Not for diagnostic use")


if __name__ == "__main__":
    main()
