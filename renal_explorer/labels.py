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
    # Context organs for approach planning (from TotalSegmentator on patient cases).
    LIVER = 9
    SPLEEN = 10
    PANCREAS = 11
    DUODENUM = 12
    COLON = 13
    ADRENAL = 14
    PSOAS = 15


LABEL_NAMES = {
    CanonicalLabel.KIDNEY: "Kidney",
    CanonicalLabel.TUMOR: "Tumor",
    CanonicalLabel.CYST: "Cyst",
    CanonicalLabel.RENAL_ARTERY: "Renal artery",
    CanonicalLabel.RENAL_VEIN: "Renal vein",
    CanonicalLabel.AORTA: "Aorta",
    CanonicalLabel.IVC: "Inferior vena cava",
    CanonicalLabel.COLLECTING_SYSTEM: "Collecting system",
    CanonicalLabel.LIVER: "Liver",
    CanonicalLabel.SPLEEN: "Spleen",
    CanonicalLabel.PANCREAS: "Pancreas",
    CanonicalLabel.DUODENUM: "Duodenum",
    CanonicalLabel.COLON: "Colon",
    CanonicalLabel.ADRENAL: "Adrenal gland",
    CanonicalLabel.PSOAS: "Psoas",
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
    CanonicalLabel.LIVER: (150, 75, 50),
    CanonicalLabel.SPLEEN: (160, 90, 200),
    CanonicalLabel.PANCREAS: (240, 200, 160),
    CanonicalLabel.DUODENUM: (255, 170, 200),
    CanonicalLabel.COLON: (200, 160, 90),
    CanonicalLabel.ADRENAL: (255, 120, 0),
    CanonicalLabel.PSOAS: (190, 70, 70),
}


RENAL_LABELS = frozenset(
    {
        CanonicalLabel.KIDNEY,
        CanonicalLabel.TUMOR,
        CanonicalLabel.CYST,
        CanonicalLabel.RENAL_ARTERY,
        CanonicalLabel.RENAL_VEIN,
        CanonicalLabel.AORTA,
        CanonicalLabel.IVC,
        CanonicalLabel.COLLECTING_SYSTEM,
    }
)
"""Structures shown by default; context organs are opt-in to keep 3D views light."""
