from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CaseFiles:
    case_id: str
    image_path: Path
    segmentation_path: Path | None = None


@dataclass
class VolumeCase:
    dataset_key: str
    case_id: str
    image: np.ndarray
    segmentation: np.ndarray | None
    spacing: tuple[float, float, float]
    affine: np.ndarray
    source_labels: tuple[int, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def json_metadata(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_key,
            "case_id": self.case_id,
            "shape": [int(value) for value in self.image.shape],
            "spacing_mm": [float(value) for value in self.spacing],
            "affine": self.affine.tolist(),
            "source_labels": list(self.source_labels),
            **self.metadata,
        }
