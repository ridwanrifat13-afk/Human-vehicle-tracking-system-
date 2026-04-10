import logging
import time

from config.settings import FRAME_HEIGHT, FRAME_RATE, FRAME_WIDTH, WINDOW_TITLE
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

        last_time = time.time()

        while not ui.closed:
            frame = video_source.read_frame()
            if frame is None:
                logger.warning("No frame received from webcam; exiting loop")
                break

            start_time = time.time()
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame_rate=FRAME_RATE)
            dt = max(1.0 / FRAME_RATE, time.time() - last_time)
            predictions = predictor.update(tracks, dt)

            for track in predictions:
                controller.send_target_position(*track["prediction"])

            fps = 1.0 / max(1e-6, time.time() - start_time)
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
