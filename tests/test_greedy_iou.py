import unittest

import numpy as np

from opendetect.trackers import GreedyIoUTracker


class GreedyIoUTests(unittest.TestCase):
    def test_tracks_follow_detection_rows_and_expire(self):
        tracker = GreedyIoUTracker(min_area=20, max_missed=1)
        boxes = np.array([[0, 0, 60, 60], [100, 0, 155, 55]])
        self.assertEqual(
            [track[0] for track in tracker.update(boxes, [0.8, 0.9])], [0, 1]
        )
        moved = np.vstack([[1, 1, 2, 2], boxes[::-1] + 5])
        matches = tracker.update_with_indices(moved, [0.9, 0.2, 0.9])
        self.assertEqual([(track[0], row) for track, row in matches], [(1, 1), (0, 2)])
        self.assertEqual(tracker.update([], []), [])
        self.assertTrue(tracker.has_tracks())
        tracker.update([], [])
        self.assertFalse(tracker.has_tracks())
        tracker.reset()
        self.assertEqual(tracker.update(boxes[:1], [0.8])[0][0], 0)


if __name__ == "__main__":
    unittest.main()
