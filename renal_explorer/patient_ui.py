"""Streamlit pieces for the private patient workflow (enabled with RENAL_PATIENT_MODE=1)."""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .labels import LABEL_COLORS, LABEL_NAMES, CanonicalLabel

PATIENT_MODE = os.environ.get("RENAL_PATIENT_MODE", "0") == "1"
INBOX = Path(os.environ.get("RENAL_INBOX", "~/RenalPlanner/inbox")).expanduser()
PHASE_LABELS = {
    "nephrographic": "Nephrographic / portal venous (planning grid)",
    "arterial": "Arterial / CTA",
    "excretory": "Excretory / delayed",
    "non-contrast": "Non-contrast",
}
STATE_ICONS = {
    "uploaded": "📝",
    "queued": "⏳",
    "running": "⚙️",
    "done": "✅",
    "failed": "❌",
}


# --------------------------------------------------------------------------------------
# Access control
# --------------------------------------------------------------------------------------


def require_password() -> None:
    """Simple shared-password gate (defence in depth behind Cloudflare Access/Tailscale).

    Set RENAL_APP_PASSWORD_SHA256 (preferred) or RENAL_APP_PASSWORD. No-op when neither is set.
    """

    expected_hash = os.environ.get("RENAL_APP_PASSWORD_SHA256", "").strip().lower()
    plain = os.environ.get("RENAL_APP_PASSWORD", "")
    if not expected_hash and plain:
        expected_hash = hashlib.sha256(plain.encode()).hexdigest()
    if not expected_hash or st.session_state.get("authenticated"):
        return
    st.title("🩻 Renal planner")
    entered = st.text_input("Password", type="password")
    if entered:
        digest = hashlib.sha256(entered.encode()).hexdigest()
        if hmac.compare_digest(digest, expected_hash):
            st.session_state["authenticated"] = True
            st.rerun()
        time.sleep(1.0)
        st.error("Wrong password.")
    st.stop()


# --------------------------------------------------------------------------------------
# Case selection (sidebar)
# --------------------------------------------------------------------------------------


def _store():
    from .patient.store import PatientStore

    return PatientStore()


def _case_label(case) -> str:
    status = case.status()
    created = time.strftime("%d.%m %H:%M", time.localtime(status.get("created", 0)))
    alias = f" · {status['alias']}" if status.get("alias") else ""
    return f"{case.case_id}{alias} · {created}"


def select_patient_case() -> tuple[str, str, str, str] | None:
    """Sidebar case picker. Returns (case_id, image, segmentation, root) or None."""

    from .patient.worker import reconcile

    store = _store()
    store.purge_expired()
    reconcile(store)
    ready = store.ready_cases()
    if not ready:
        st.info("No processed patient cases yet.")
        return None
    case = st.selectbox("Patient case", ready, format_func=_case_label)
    return case.case_id, str(case.image_path), str(case.segmentation_path), str(store.root)


# --------------------------------------------------------------------------------------
# Upload, series selection and job list (main area)
# --------------------------------------------------------------------------------------


def _ingest(case, source: Path, inbox_source: Path | None = None) -> bool:
    """Copy/extract a study into the case. True when usable CT series were found.

    An inbox original is never deleted here: the worker deletes it only after the
    DICOM-to-NIfTI conversion has succeeded.
    """

    from .patient.dicom_intake import (
        list_series,
        record_inbox_source,
        safe_extract_zip,
        write_series_index,
    )

    if source.is_dir():
        shutil.copytree(source, case.raw_dir, dirs_exist_ok=True)
    else:
        safe_extract_zip(source, case.raw_dir)
    series = list_series(case.raw_dir)
    if not series:
        case.discard_raw()
        case.update_status(state="failed", error="No CT series with ≥20 slices found.")
        return False
    if inbox_source is not None:
        record_inbox_source(case.raw_dir, inbox_source)
    case.write_request({"series": write_series_index(case.raw_dir, series)})
    case.update_status(step="Choose series and phases")
    return True


def _upload_panel() -> None:
    store = _store()
    st.markdown("**New case**")
    alias = st.text_input(
        "Short label (optional — never a patient name or ID)",
        max_chars=24,
        key="patient_alias",
    )
    upload_tab, inbox_tab = st.tabs(
        ["Upload ZIP (≤ 100 MB through Cloudflare)", "Import from inbox"]
    )
    with upload_tab:
        upload = st.file_uploader("DICOM study as .zip", type=("zip",), key="dicom_zip")
        if upload is not None and st.button("Import upload", type="primary"):
            case = store.create_case(alias)
            archive = case.path / "upload.zip"
            try:
                with open(archive, "wb") as handle:
                    shutil.copyfileobj(io.BytesIO(upload.getvalue()), handle)
                with st.spinner("Reading DICOM series…"):
                    _ingest(case, archive)
            except Exception as exc:  # noqa: BLE001
                case.discard_raw()
                case.update_status(state="failed", error=str(exc)[:300])
                st.error(f"Import failed: {exc}")
            finally:
                archive.unlink(missing_ok=True)
            st.rerun()
    with inbox_tab:
        INBOX.mkdir(parents=True, exist_ok=True)
        items = sorted(
            path
            for path in INBOX.iterdir()
            if not path.name.startswith(".") and (path.is_dir() or path.suffix.lower() == ".zip")
        )
        st.caption(
            f"Drop a DICOM ZIP or folder into `{INBOX}` on the planning Mac (PACS export, USB, "
            "network share). No size limit."
        )
        if not items:
            st.info("The inbox is empty.")
        else:
            item = st.selectbox("Inbox item", items, format_func=lambda path: path.name)
            remove = st.checkbox(
                "Delete from inbox once converted successfully", value=True
            )
            if st.button("Import from inbox", type="primary"):
                case = store.create_case(alias)
                try:
                    with st.spinner("Reading DICOM series…"):
                        if not _ingest(case, item, item if remove else None):
                            st.error("No usable CT series found; the inbox item was kept.")
                except Exception as exc:  # noqa: BLE001
                    case.discard_raw()
                    case.update_status(state="failed", error=str(exc)[:300])
                    st.error(f"Import failed: {exc}")
                st.rerun()


def _series_selection(case) -> None:
    from .patient.dicom_intake import SeriesInfo, default_assignment, read_series_index
    from .patient.worker import launch

    request = case.request()
    series = request.get("series", [])
    # Descriptions are read from raw/ (deleted after conversion), never from request.json.
    private = read_series_index(case.raw_dir)
    st.markdown(f"**{_case_label(case)} — choose series**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Series": item["handle"],
                    "Description": private.get(item["handle"], {}).get("description", ""),
                    "Slices": item["slices"],
                    "Thickness (mm)": item["slice_thickness_mm"],
                    "Guessed phase": item["phase_guess"],
                }
                for item in series
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    defaults = default_assignment(
        [
            SeriesInfo(
                series_uid=item["handle"],
                description="",
                slices=item["slices"],
                slice_thickness_mm=item["slice_thickness_mm"],
                series_time="",
                phase_guess=item["phase_guess"],
                files=[],
            )
            for item in series
        ]
    )
    by_handle = {item["handle"]: item for item in series}
    options = [""] + [item["handle"] for item in series]

    def describe(handle: str) -> str:
        if not handle:
            return "— not available —"
        item = by_handle[handle]
        description = private.get(handle, {}).get("description") or "unnamed"
        return (
            f"{handle} · {description} · {item['slices']} sl · "
            f"{item['slice_thickness_mm']} mm"
        )

    assignment: dict[str, str] = {}
    columns = st.columns(len(PHASE_LABELS))
    for column, (phase, label) in zip(columns, PHASE_LABELS.items()):  # noqa: B905
        with column:
            choice = st.selectbox(
                label,
                options,
                index=options.index(defaults[phase]) if phase in defaults else 0,
                format_func=describe,
                key=f"phase_{case.case_id}_{phase}",
            )
            if choice:
                assignment[phase] = choice
    planning_phase = next(
        (phase for phase in PHASE_LABELS if phase in assignment),
        None,
    )
    st.caption(
        "The nephrographic phase is the planning grid; arterial feeds the vessel model, excretory "
        "the collecting system. Other phases are registered onto the planning grid."
    )
    start, discard = st.columns(2)
    if start.button(
        "Start automatic segmentation",
        type="primary",
        disabled=planning_phase is None,
        key=f"start_{case.case_id}",
        width="stretch",
    ):
        case.write_request(
            {**request, "assignment": assignment, "main_phase": planning_phase}
        )
        launch(case)
        st.rerun()
    if discard.button("Discard case", key=f"discard_{case.case_id}", width="stretch"):
        _store().delete(case.case_id)
        st.rerun()


@st.fragment(run_every=5)
def _job_table() -> None:
    store = _store()
    cases = store.cases()
    if not cases:
        st.caption("No cases yet.")
        return
    rows = []
    for case in cases:
        status = case.status()
        rows.append(
            {
                "": STATE_ICONS.get(status.get("state"), "?"),
                "Case": _case_label(case),
                "State": status.get("state"),
                "Step": status.get("step", ""),
                "Warnings": status.get("warnings", ""),
                "Error": status.get("error") or "",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_patient_workspace(expanded: bool) -> None:
    store = _store()
    with st.expander("Patient cases — upload, processing queue", expanded=expanded):
        st.caption(
            "Private research workflow. Raw DICOM is deleted after conversion; cases are "
            "pseudonymised and purged after the retention period. Automatic labels must be "
            "reviewed before any use."
        )
        pending = [case for case in store.cases() if case.status().get("state") == "uploaded"]
        for case in pending:
            if case.request().get("series"):
                _series_selection(case)
                st.divider()
        _upload_panel()
        st.divider()
        st.markdown("**Processing queue**")
        _job_table()
        failed = [case for case in store.cases() if case.status().get("state") == "failed"]
        if failed:
            target = st.selectbox(
                "Inspect failed case", failed, format_func=_case_label, key="failed_case"
            )
            log = target.log_path.read_text()[-4000:] if target.log_path.exists() else ""
            st.code(log or "(no log)", language="text")
            if st.button("Delete failed case"):
                store.delete(target.case_id)
                st.rerun()


# --------------------------------------------------------------------------------------
# Review & correction round trip
# --------------------------------------------------------------------------------------


def slicer_color_table() -> str:
    lines = [
        "# 3D Slicer color table for renal-ct-explorer canonical labels",
        "0 Background 0 0 0 0",
    ]
    for label in CanonicalLabel:
        if label == CanonicalLabel.BACKGROUND:
            continue
        red, green, blue = LABEL_COLORS[label]
        name = LABEL_NAMES[label].replace(" ", "_")
        lines.append(f"{int(label)} {name} {red} {green} {blue} 255")
    return "\n".join(lines) + "\n"


def validate_correction(payload: bytes, image_path: Path) -> Any:
    import gzip

    import nibabel as nib
    import numpy as np

    raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    corrected = nib.Nifti1Image.from_bytes(raw)
    reference = nib.load(str(image_path))
    if corrected.shape != reference.shape:
        raise ValueError(f"Shape {corrected.shape} differs from the CT {reference.shape}.")
    if not np.allclose(corrected.affine, reference.affine, atol=1e-2):
        raise ValueError("Orientation/position differs from the CT; export on the CT's geometry.")
    data = np.asanyarray(corrected.dataobj)
    values = set(int(value) for value in np.unique(data))
    allowed = {int(label) for label in CanonicalLabel}
    if not values <= allowed:
        raise ValueError(f"Unknown label values {sorted(values - allowed)}; use the colour table.")
    return nib.Nifti1Image(data.astype(np.uint8), reference.affine), values


def render_correction_panel(case_id: str) -> None:
    store = _store()
    try:
        case = store.get(case_id)
    except KeyError:
        return
    provenance = case.provenance()
    with st.expander("Review & correct labels", expanded=False):
        for warning in provenance.get("warnings", []):
            st.warning(warning)
        st.download_button(
            "Download segmentation (NIfTI)",
            data=case.segmentation_path.read_bytes(),
            file_name=f"{case.case_id}_segmentation.nii.gz",
            width="stretch",
        )
        st.download_button(
            "Download CT (NIfTI)",
            data=case.image_path.read_bytes(),
            file_name=f"{case.case_id}_imaging.nii.gz",
            width="stretch",
        )
        st.download_button(
            "3D Slicer colour table",
            data=slicer_color_table(),
            file_name="renal_planner_labels.ctbl",
            width="stretch",
        )
        st.caption(
            "Open both NIfTI files in 3D Slicer, load the colour table, edit in Segment Editor, "
            "export as a labelmap on the CT geometry and upload it here."
        )
        upload = st.file_uploader("Corrected segmentation (.nii.gz)", type=("gz", "nii"))
        if upload is not None and st.button("Replace segmentation", type="primary"):
            import nibabel as nib

            try:
                image, values = validate_correction(upload.getvalue(), case.image_path)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Rejected: {exc}")
                return
            history = case.path / "history"
            history.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            shutil.copy2(case.segmentation_path, history / f"segmentation_{stamp}.nii.gz")
            nib.save(image, str(case.segmentation_path))
            provenance.setdefault("manual_corrections", []).append(
                {"time": stamp, "labels": sorted(values)}
            )
            case.write_provenance(provenance)
            st.success("Segmentation replaced. The case reloads with your corrections.")
            st.rerun()
        with st.popover("Delete this case"):
            st.write("Removes the CT, labels and history from this Mac.")
            if st.button("Delete permanently", key=f"delete_{case.case_id}"):
                store.delete(case.case_id)
                st.rerun()
