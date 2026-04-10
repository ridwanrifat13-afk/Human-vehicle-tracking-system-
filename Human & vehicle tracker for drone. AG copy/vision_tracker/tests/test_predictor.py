import unittest
from prediction.kalman_filter import KalmanFilterPredictor


class TestKalmanFilterPredictor(unittest.TestCase):
    def test_update_generates_prediction(self):
        predictor = KalmanFilterPredictor()
        tracks = [
            {"id": 1, "bbox": [0, 0, 10, 10], "class_name": "person", "confidence": 0.8},
        ]
        predictions = predictor.update(tracks, dt=1 / 30.0)
        self.assertEqual(len(predictions), 1)
        self.assertIn("prediction", predictions[0])
        self.assertIn("velocity", predictions[0])

    def test_prediction_updates_same_track_id(self):
        predictor = KalmanFilterPredictor()
        tracks = [
            {"id": 1, "bbox": [0, 0, 10, 10], "class_name": "person", "confidence": 0.8},
        ]
        predictions1 = predictor.update(tracks, dt=1 / 30.0)
        predictions2 = predictor.update(tracks, dt=1 / 30.0)
        self.assertEqual(predictions1[0]["id"], predictions2[0]["id"])


if __name__ == "__main__":
    unittest.main()
