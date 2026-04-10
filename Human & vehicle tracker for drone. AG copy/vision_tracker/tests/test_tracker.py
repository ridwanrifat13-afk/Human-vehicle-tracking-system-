import unittest
from tracking.bytetrack_tracker import ByteTrackTracker


class TestByteTrackTracker(unittest.TestCase):
    def test_update_creates_tracks(self):
        tracker = ByteTrackTracker()
        detections = [
            {"bbox": [0, 0, 10, 10], "class_name": "person", "confidence": 0.9},
        ]
        tracks = tracker.update(detections)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["id"], 1)
        self.assertEqual(tracks[0]["class_name"], "person")

    def test_unmatched_tracks_are_retained(self):
        tracker = ByteTrackTracker(max_lost=1)
        detections = [
            {"bbox": [0, 0, 10, 10], "class_name": "person", "confidence": 0.9},
        ]
        tracker.update(detections)
        tracks = tracker.update([])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["state"], "tentative")

    def test_confirmed_track_state(self):
        tracker = ByteTrackTracker()
        detections = [
            {"bbox": [0, 0, 10, 10], "class_name": "person", "confidence": 0.9},
        ]
        tracker.update(detections)
        tracks = tracker.update(detections)
        self.assertEqual(tracks[0]["state"], "confirmed")


if __name__ == "__main__":
    unittest.main()
