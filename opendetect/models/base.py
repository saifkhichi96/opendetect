from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

Detections = dict[str, np.ndarray]


class DetectorModel(Protocol):
    model_name: str
    model_path: Path
    input_size: tuple[int, int]

    @classmethod
    def default_class_ids(cls) -> list[int] | None: ...

    @classmethod
    def default_input_size(cls) -> tuple[int, int]: ...

    def preprocess_rgb_frame(
        self, image_rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def forward(self, input_tensor: np.ndarray) -> tuple[np.ndarray, ...]: ...

    def postprocess(
        self,
        outputs: tuple[np.ndarray, ...],
        target_sizes: np.ndarray,
    ) -> list[Detections]: ...

    def predict_rgb_frame(self, image_rgb: np.ndarray) -> Detections: ...

    def draw_detections_on_bgr_frame(
        self, frame_bgr: np.ndarray, detections: Detections
    ) -> np.ndarray: ...
