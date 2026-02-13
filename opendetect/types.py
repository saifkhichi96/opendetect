from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

Detections = dict[str, np.ndarray]


@dataclass(frozen=True)
class LoadedModelInfo:
    model_id: str | None
    implementation: str
    input_size: tuple[int, int]
    model_path: Path
    source: str
