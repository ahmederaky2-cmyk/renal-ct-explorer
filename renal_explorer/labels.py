from __future__ import annotations

from enum import IntEnum


class CanonicalLabel(IntEnum):
    """Labels shared by every supported dataset.

    Dataset-specific left/right kidney labels are intentionally merged here.
    Laterality is calculated from connected components in canonical RAS space.
    """

    BACKGROUND = 0
    KIDNEY = 1
    TUMOR = 2
    CYST = 3
    RENAL_ARTERY = 4
    RENAL_VEIN = 5
    AORTA = 6
    IVC = 7
    COLLECTING_SYSTEM = 8


LABEL_NAMES = {
    CanonicalLabel.KIDNEY: "Kidney",
    CanonicalLabel.TUMOR: "Tumor",
    CanonicalLabel.CYST: "Cyst",
    CanonicalLabel.RENAL_ARTERY: "Renal artery",
    CanonicalLabel.RENAL_VEIN: "Renal vein",
    CanonicalLabel.AORTA: "Aorta",
    CanonicalLabel.IVC: "Inferior vena cava",
    CanonicalLabel.COLLECTING_SYSTEM: "Collecting system",
}


LABEL_COLORS = {
    CanonicalLabel.KIDNEY: (45, 200, 90),
    CanonicalLabel.TUMOR: (255, 65, 65),
    CanonicalLabel.CYST: (255, 196, 45),
    CanonicalLabel.RENAL_ARTERY: (255, 55, 55),
    CanonicalLabel.RENAL_VEIN: (75, 115, 255),
    CanonicalLabel.AORTA: (225, 55, 95),
    CanonicalLabel.IVC: (85, 165, 255),
    CanonicalLabel.COLLECTING_SYSTEM: (250, 235, 120),
}
