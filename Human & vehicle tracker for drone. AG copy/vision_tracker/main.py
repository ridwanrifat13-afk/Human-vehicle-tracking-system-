
import logging
import time

from config.settings import VIDEO_SOURCE, FRAME_HEIGHT, FRAME_RATE, FRAME_WIDTH, WINDOW_TITLE
from control.null_controller import NullController
from detection.yolo_detector import YOLODetector
from prediction.kalman_filter import KalmanFilterPredictor
from sources.webcam_source import WebcamSource
from tracking.bytetrack_tracker import ByteTrackTracker
from utils.logger import setup_logging
from visualization.ui import TrackerUI

logger = setup_logging()


def main():
    video_source = WebcamSource(source=0, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=FRAME_RATE)
    tracker = ByteTrackTracker()
    predictor = KalmanFilterPredictor()
    controller = NullController()
    ui = TrackerUI(width=FRAME_WIDTH, height=FRAME_HEIGHT, title=WINDOW_TITLE)

    try:
        detector = YOLODetector()
        logger.info("YOLO detector initialized")
        video_source.start()
        logger.info("Webcam source started")

        predictions = []
        last_time = time.time()

        ui.update_camera_status(
            camera_name=f"Camera {VIDEO_SOURCE}",
            connected=False,
            resolution=f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            fps=0.0,
            neural_link="Initialising",
            detection_engine="YOLOv8n",
        )
        ui.update_drone_status(altitude="N/A", longitude="N/A", battery="N/A", connection="Disconnected")
        ui.update_system_status("Ready", "Awaiting frames")

        while not ui.closed:
            frame = video_source.read_frame()
            if frame is None:
                logger.warning("No frame received from webcam; exiting loop")
                break

            start_time = time.time()
            if ui.paused:
                ui.update_system_status("Paused", "Tracking suspended")
                ui.render(frame, predictions, 0.0)
                last_time = time.time()
                continue

            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame_rate=FRAME_RATE)
            dt = max(1.0 / FRAME_RATE, time.time() - last_time)
            predictions = predictor.update(tracks, dt)

            for track in predictions:
                controller.send_target_position(*track["prediction"])

            fps = 1.0 / max(1e-6, time.time() - start_time)
            ui.update_camera_status(
                camera_name=f"Camera {VIDEO_SOURCE}",
                connected=True,
                resolution=f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
                fps=fps,
                neural_link="Active",
                detection_engine="YOLOv8n",
            )
            ui.update_drone_status(altitude="N/A", longitude="N/A", battery="N/A", connection="Disconnected")
            ui.update_system_status("Tracking", "Operational")
            ui.render(frame, predictions, fps)
            last_time = time.time()

    except Exception:
        logger.exception("Tracker error")
    finally:
        video_source.stop()
        if not ui.closed:
            ui.close()


if __name__ == "__main__":
    main()
