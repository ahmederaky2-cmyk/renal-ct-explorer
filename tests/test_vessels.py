from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from renal_explorer.three_d import bridge_vessel_gaps, detect_great_vessels

SPACING = np.asarray((1.5, 1.5, 3.0))
ISO = np.full(3, 1.5)


def _grid(shape):
    x, y, z = np.meshgrid(*(np.arange(n) * s for n, s in zip(shape, SPACING)), indexing="ij")
    return x, y, z


def _phantom():
    shape = (120, 100, 50)
    x, y, _z = _grid(shape)
    image = np.full(shape, 40.0, dtype=np.float32)
    aorta = (x - 90.0) ** 2 + (y - 60.0) ** 2 <= 10.0**2
    image[aorta] = 260.0
    spine_radius = np.sqrt((x - 90.0) ** 2 + (y - 110.0) ** 2)
    image[spine_radius <= 20.0] = 250.0
    image[(spine_radius > 16.0) & (spine_radius <= 20.0)] = 950.0
    liver = (x < 40.0) & (y < 90.0)
    image[liver] = 120.0
    kidney = ((x - 150.0) / 25.0) ** 2 + ((y - 70.0) / 15.0) ** 2 <= 1.0
    kidney &= (_z := _grid(shape)[2]) > 30.0
    kidney &= _z < 120.0
    image[kidney] = 160.0
    return image, aorta, kidney, liver


def test_detects_the_aorta_but_not_spine_or_liver() -> None:
    image, aorta, kidney, liver = _phantom()
    found = detect_great_vessels(image, kidney, SPACING, midline_index=60.0)
    assert np.count_nonzero(found & aorta) > 0.8 * np.count_nonzero(aorta)
    assert not np.any(found & liver)
    assert np.count_nonzero(found & ~aorta) < 0.1 * np.count_nonzero(aorta)


def test_bridges_a_short_gap_in_enhancing_tissue() -> None:
    shape = (80, 30, 30)
    image = np.full(shape, 150.0, dtype=np.float32)
    tube = np.zeros(shape, dtype=bool)
    tube[5:35, 14:17, 14:17] = True
    tube[38:75, 14:17, 14:17] = True  # centres 6 mm apart
    bridged, count = bridge_vessel_gaps(tube, image, np.zeros(shape, bool), ISO)
    assert count == 1
    assert ndi.label(bridged, structure=np.ones((3, 3, 3)))[1] == 1


def test_does_not_bridge_across_fat_or_long_gaps() -> None:
    shape = (80, 30, 30)
    image = np.full(shape, 150.0, dtype=np.float32)
    image[35:38] = -100.0
    tube = np.zeros(shape, dtype=bool)
    tube[5:35, 14:17, 14:17] = True
    tube[38:75, 14:17, 14:17] = True
    _bridged, fat_count = bridge_vessel_gaps(tube, image, np.zeros(shape, bool), ISO)
    assert fat_count == 0

    far = np.zeros(shape, dtype=bool)
    far[5:25, 14:17, 14:17] = True
    far[40:75, 14:17, 14:17] = True  # 15 voxels = 22.5 mm
    _bridged, far_count = bridge_vessel_gaps(far, np.full(shape, 150.0), np.zeros(shape, bool), ISO)
    assert far_count == 0


def test_bridges_a_branch_to_an_anchor_without_painting_the_anchor() -> None:
    shape = (60, 40, 30)
    image = np.full(shape, 150.0, dtype=np.float32)
    anchor = np.zeros(shape, dtype=bool)
    anchor[0:12, :, :] = True
    branch = np.zeros(shape, dtype=bool)
    branch[15:55, 19:22, 14:17] = True
    bridged, count = bridge_vessel_gaps(branch, image, anchor, ISO)
    assert count == 1
    assert not np.any(bridged & anchor)
    assert ndi.label(bridged | anchor, structure=np.ones((3, 3, 3)))[1] == 1


def test_candidate_layer_reaches_the_aorta(monkeypatch) -> None:
    from renal_explorer import three_d
    from renal_explorer.labels import CanonicalLabel

    monkeypatch.setattr(three_d, "_windowed_sinc_smooth", lambda vertices, faces, factor: vertices)
    shape = (110, 70, 40)
    spacing = (1.5, 1.5, 3.0)
    x, y, z = np.meshgrid(*(np.arange(n) * s for n, s in zip(shape, spacing)), indexing="ij")
    image = np.full(shape, -80.0, dtype=np.float32)
    image[(x - 40.0) ** 2 + (y - 50.0) ** 2 <= 11.0**2] = 260.0  # aorta near the midline
    kidney = ((x - 125.0) / 22.0) ** 2 + ((y - 50.0) / 14.0) ** 2 + ((z - 60.0) / 40.0) ** 2 <= 1
    image[kidney] = 170.0
    artery = ((y - 50.0) ** 2 + (z - 60.0) ** 2 <= 2.5**2) & (x > 48.0) & (x < 108.0)
    image[artery] = 280.0
    segmentation = np.zeros(shape, dtype=np.uint8)
    segmentation[kidney] = int(CanonicalLabel.KIDNEY)
    affine = np.diag((*spacing, 1.0))

    mesh = three_d.candidate_vessel_mesh(image, segmentation, affine, spacing)

    assert mesh is not None
    assert mesh["quality"]["great_vessels"] == "detected"
    vertices = mesh["vertices"]
    assert vertices[:, 0].min() < 35.0  # reaches the far wall of the aorta
    assert mesh["quality"]["largest_component_fraction"] > 0.9
