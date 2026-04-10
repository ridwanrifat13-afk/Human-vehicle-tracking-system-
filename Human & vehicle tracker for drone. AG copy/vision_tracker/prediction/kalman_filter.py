import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanFilterPredictor:
    def __init__(self):
        self.filters = {}

    def _create_filter(self, center, dt):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([center[0], center[1], 0.0, 0.0], dtype=float)
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)
        kf.P *= 1000.0
        kf.R = np.eye(2) * 10.0
        kf.Q = np.eye(4) * 0.01
        return kf

    def update(self, tracks, dt=1 / 30.0):
        predictions = []
        active_ids = set()

        for track in tracks:
            track_id = track["id"]
            x1, y1, x2, y2 = track["bbox"]
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            if track_id not in self.filters:
                self.filters[track_id] = self._create_filter(center, dt)

            kf = self.filters[track_id]
            kf.F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=float)
            kf.predict()
            kf.update(np.array(center, dtype=float))

            predicted_center = (float(kf.x[0]), float(kf.x[1]))
            velocity = (float(kf.x[2]), float(kf.x[3]))

            predictions.append({
                **track,
                "prediction": predicted_center,
                "velocity": velocity,
            })
            active_ids.add(track_id)

        self.filters = {tid: kf for tid, kf in self.filters.items() if tid in active_ids}
        return predictions
