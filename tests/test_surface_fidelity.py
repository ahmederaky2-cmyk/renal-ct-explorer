from __future__ import annotations

import numpy as np
from scipy import sparse

from renal_explorer import three_d


def _taubin(vertices, faces, factor):
    """Small numpy stand-in for VTK windowed-sinc smoothing (CI without VTK)."""
    n = vertices.shape[0]
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    adjacency = sparse.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    adjacency = ((adjacency + adjacency.T) > 0).astype(float)
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1.0)
    average = sparse.diags(1.0 / degree) @ adjacency
    smoothed = vertices.astype(float).copy()
    for _ in range(int(20 + factor * 40)):
        smoothed += 0.5 * (average @ smoothed - smoothed)
        smoothed += -0.53 * (average @ smoothed - smoothed)
    return smoothed


def _thick_slice_kidney():
    spacing = (1.56, 1.56, 3.0)  # KiTS-like 0.78 mm in-plane, reduced for the demo
    shape = (70, 60, 50)
    x, y, z = np.meshgrid(*(np.arange(n) * s for n, s in zip(shape, spacing)), indexing="ij")
    kidney = ((x - 55) / 30) ** 2 + ((y - 47) / 20) ** 2 + ((z - 75) / 55) ** 2 <= 1.0
    affine = np.diag((*spacing, 1.0))
    return kidney.astype(np.uint8), affine, spacing


def test_reference_surface_is_unsmoothed_and_on_the_mask(monkeypatch) -> None:
    monkeypatch.setattr(three_d, "_windowed_sinc_smooth", _taubin)
    monkeypatch.setattr(three_d, "_decimate", lambda v, f, n: (v, f))
    mask, affine, spacing = _thick_slice_kidney()

    reference = three_d.surface_mesh(mask, affine, spacing, 1, quality="Reference")
    display = three_d.surface_mesh(mask, affine, spacing, 1, quality="Balanced")

    assert reference["quality"]["role"] == "reference"
    assert reference["quality"]["smoothing_displacement_p95_mm"] == 0.0
    assert reference["quality"]["surface_to_mask_p95_mm"] <= 0.5 * max(spacing)
    assert display["quality"]["role"] == "display"
    assert display["quality"]["volume_error_pct"] < 3.5
    # The preview may round off slice terraces, but never by more than about one slice.
    assert display["quality"]["surface_to_mask_p95_mm"] <= max(spacing)
