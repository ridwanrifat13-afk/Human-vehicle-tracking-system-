import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanFilterPredictor:
    """Kalman filter wrapper for predicting and smoothing object tracks with dynamic dt."""

    def __init__(self):
        self.filters = {}

    def _create_filter(self, center, dt):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        # State: [x, y, vx, vy]
        kf.x = np.array([center[0], center[1], 0.0, 0.0], dtype=float)
        
        # State transition matrix
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        
        # Measurement matrix (we only measure x and y)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)
        
        kf.P *= 1000.0  # Initial uncertainty
        kf.R = np.eye(2) * 1.0  # Measurement noise (lower = trust sensors more)
        kf.Q = np.array([
            [0.1, 0, 0.1, 0],
            [0, 0.1, 0, 0.1],
            [0.1, 0, 1.0, 0],
            [0, 0.1, 0, 1.0],
        ]) * 0.01  # Process noise
        
        return kf

    def predict(self, track_id, dt):
        """Project the state forward by dt without an update."""
        if track_id not in self.filters:
            return None
        
        kf = self.filters[track_id]
        kf.F[0, 2] = dt
        kf.F[1, 3] = dt
        kf.predict()
        
        return {
            "prediction": (float(kf.x[0]), float(kf.x[1])),
            "velocity": (float(kf.x[2]), float(kf.x[3]))
        }

    def update(self, tracks, dt=1 / 30.0):
        """Perform a full predict and update cycle for a list of tracks."""
        predictions = []
        active_ids = set()

        for track in tracks:
            track_id = track["id"]
            x1, y1, x2, y2 = track["bbox"]
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            if track_id not in self.filters:
                self.filters[track_id] = self._create_filter(center, dt)

            kf = self.filters[track_id]
            kf.F[0, 2] = dt
            kf.F[1, 3] = dt
            
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

        # Cleanup inactive filters
        self.filters = {tid: kf for tid, kf in self.filters.items() if tid in active_ids}
        return predictions
