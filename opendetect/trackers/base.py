"""The common per-stream interface for bounding-box identity trackers."""

from typing import Protocol

import numpy as np

# Track ID, xyxy box in input coordinates, detection confidence.
Track = tuple[int, np.ndarray, float]


class Tracker(Protocol):
    """Associate detections across consecutive frames of one stream."""

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[Track]:
        """Consume one frame and return tracks observed in that frame."""
        ...

    def reset(self) -> None:
        """Discard history and restart identity assignment for a new stream."""
        ...
