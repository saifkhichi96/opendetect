"""Greedy IoU association with a smoothed box-center velocity."""

from dataclasses import dataclass

import numpy as np

from ._geometry import iou_matrix
from .base import Track


@dataclass
class _Track:
    id: int
    box: np.ndarray
    velocity: np.ndarray | None = None
    missed: int = 0

    def predict(self):
        if self.velocity is None:
            return self.box
        return self.box + np.tile(self.velocity, 2)


class GreedyIoUTracker:
    """Associate one stream of xyxy boxes using the best available overlap first."""

    def __init__(self, iou_thr=0.3, max_missed=10, min_area=1000, velocity_alpha=0.8):
        self.iou_thr = iou_thr
        self.max_missed = max_missed
        self.min_area = min_area
        self.velocity_alpha = velocity_alpha
        self.tracks: list[_Track] = []
        self.next_id = 0

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 0

    def has_tracks(self) -> bool:
        return bool(self.tracks)

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[Track]:
        return [track for track, _ in self.update_with_indices(boxes, scores)]

    def update_with_indices(
        self, boxes: np.ndarray, scores: np.ndarray
    ) -> list[tuple[Track, int]]:
        """Return tracks seen in this frame alongside their source detection rows."""
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.ndim == 1 and boxes.size == 0:
            boxes = boxes.reshape(0, 4)
        scores = np.asarray(scores).reshape(-1)
        if boxes.shape != (len(scores), 4):
            raise ValueError("Expected one xyxy box per detection score")
        sizes = boxes[:, 2:] - boxes[:, :2]
        rows = np.flatnonzero(
            np.isfinite(boxes).all(axis=1)
            & (sizes > 0).all(axis=1)
            & (sizes.prod(axis=1) >= self.min_area)
        )
        matches = {}
        used_tracks = set()
        if self.tracks and len(rows):
            predicted = np.asarray([track.predict() for track in self.tracks])
            overlaps = iou_matrix(predicted, boxes[rows])
            track_rows, detection_rows = np.nonzero(overlaps >= self.iou_thr)
            order = np.argsort(-overlaps[track_rows, detection_rows], kind="stable")
            for pair in order:
                track_index = int(track_rows[pair])
                row = int(rows[detection_rows[pair]])
                if track_index in used_tracks or row in matches:
                    continue
                track = self.tracks[track_index]
                box = boxes[row]
                velocity = ((box[:2] - track.box[:2]) + (box[2:] - track.box[2:])) / 2
                if track.velocity is not None:
                    alpha = self.velocity_alpha
                    velocity = alpha * track.velocity + (1 - alpha) * velocity
                track.box = box.copy()
                track.velocity = velocity
                track.missed = 0
                matches[row] = track.id
                used_tracks.add(track_index)

        for index, track in enumerate(self.tracks):
            if index not in used_tracks:
                track.missed += 1
        self.tracks = [
            track for track in self.tracks if track.missed <= self.max_missed
        ]
        for row in map(int, rows):
            if row not in matches:
                self.tracks.append(_Track(self.next_id, boxes[row].copy()))
                matches[row] = self.next_id
                self.next_id += 1
        return [
            ((matches[row], boxes[row].copy(), float(scores[row])), row)
            for row in sorted(matches)
        ]
