"""ByteTrack association (motion only): a Kalman filter per track, two-stage IoU matching.

Adapted from the TRACX mesh service's port of NVIDIA GEM-X (Apache-2.0),
based on ByteTrack by Yifu Zhang et al. (MIT). See LICENSE for attribution
and third-party terms.
Track ids are per tracker instance (safe with concurrent jobs).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._geometry import iou_matrix
from .base import Track


class _Track:
    """Kalman filter over [cx, cy, aspect, h] with constant velocity."""

    F = np.eye(8)
    F[:4, 4:] = np.eye(4)
    H = np.eye(4, 8)
    Q = np.diag([1.0] * 4 + [0.01] * 4)
    R = np.diag([1.0, 1.0, 10.0, 1.0])

    def __init__(self, track_id: int, box: np.ndarray, score: float):
        self.id = track_id
        self.x = np.concatenate([_to_xyah(box), np.zeros(4)])
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100.0
        self.time_since_update = 0
        self.score = score

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1

    def update(self, box: np.ndarray, score: float) -> None:
        y = _to_xyah(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.score = score

    def box(self) -> np.ndarray:
        cx, cy, a, h = self.x[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _to_xyah(box: np.ndarray) -> np.ndarray:
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, w / (h + 1e-7), h])


class ByteTracker:
    """High-score detections start and continue tracks; low-score ones only continue them."""

    def __init__(
        self,
        high_thr: float = 0.5,
        low_thr: float = 0.1,
        match_thr: float = 0.3,
        max_lost: int = 30,
    ):
        self.high_thr = high_thr
        self.low_thr = low_thr
        self.match_thr = match_thr
        self.max_lost = max_lost
        self.tracks: list[_Track] = []
        self._next_id = 0

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 0

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[Track]:
        """One frame of detections -> (track id, box xyxy, score) of the tracks seen in it."""
        return [track for track, _ in self.update_with_indices(boxes, scores)]

    def update_with_indices(
        self, boxes: np.ndarray, scores: np.ndarray
    ) -> list[tuple[Track, int]]:
        """Return visible tracks with their corresponding input detection indices."""
        boxes = np.asarray(boxes, dtype=np.float64)
        if boxes.ndim == 1 and boxes.size == 0:
            boxes = boxes.reshape(0, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if boxes.shape != (len(scores), 4):
            raise ValueError("Expected one xyxy box per detection score")
        valid = (
            np.isfinite(boxes).all(axis=1)
            & np.isfinite(scores)
            & (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
        )
        for t in self.tracks:
            t.predict()
        high = np.flatnonzero(valid & (scores >= self.high_thr))
        low = np.flatnonzero(valid & (scores < self.high_thr) & (scores > self.low_thr))
        hb, hs, lb, ls = boxes[high], scores[high], boxes[low], scores[low]
        matched = {}

        pairs, open_tracks, open_dets = self._match(self.tracks, hb)
        for ti, di in pairs:
            track = self.tracks[ti]
            track.update(hb[di], hs[di])
            matched[track.id] = int(high[di])
        remaining = [self.tracks[i] for i in open_tracks]
        if remaining and len(lb):
            pairs2, _, _ = self._match(remaining, lb)
            for ti, di in pairs2:
                track = remaining[ti]
                track.update(lb[di], ls[di])
                matched[track.id] = int(low[di])
        for di in open_dets:
            self.tracks.append(_Track(self._next_id, hb[di], hs[di]))
            matched[self._next_id] = int(high[di])
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost]
        return [
            ((t.id, t.box(), float(t.score)), matched[t.id])
            for t in self.tracks
            if t.time_since_update == 0
        ]

    def _match(
        self, tracks: list[_Track], boxes: np.ndarray
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not len(boxes):
            return [], list(range(len(tracks))), list(range(len(boxes)))
        iou = iou_matrix(np.array([t.box() for t in tracks]), boxes)
        rows, cols = linear_sum_assignment(1.0 - iou)
        pairs = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= self.match_thr]
        used_t, used_d = {r for r, _ in pairs}, {c for _, c in pairs}
        return (
            pairs,
            [i for i in range(len(tracks)) if i not in used_t],
            [i for i in range(len(boxes)) if i not in used_d],
        )
