import logging
import time

from config.settings import VIDEO_SOURCE, FRAME_HEIGHT, FRAME_RATE, FRAME_WIDTH, WINDOW_TITLE, MODEL_NAME, DEVICE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
from control.null_controller import NullController
from detection.async_detector import AsyncDetector
from prediction.kalman_filter import KalmanFilterPredictor
from sources.webcam_source import WebcamSource
from tracking.bytetrack_tracker import ByteTrackTracker
from utils.logger import setup_logging
from visualization.ui import TrackerUI

logger = setup_logging()


def main():
    # Initialize components
    video_source = WebcamSource(source=VIDEO_SOURCE, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=FRAME_RATE)
    tracker = ByteTrackTracker()
    predictor = KalmanFilterPredictor()
    controller = NullController()
    ui = TrackerUI(width=FRAME_WIDTH, height=FRAME_HEIGHT, title=WINDOW_TITLE)

    try:
        # ASYNC DETECTOR: Runs inference in background
        detector = AsyncDetector(
            model_path=MODEL_NAME, 
            device=DEVICE, 
            confidence=CONFIDENCE_THRESHOLD, 
            iou=IOU_THRESHOLD
        )
        logger.info(f"Async YOLO detector initialized on {DEVICE}")
        
        video_source.start()
        logger.info("Threaded webcam source started")

        predictions = []
        last_render_time = time.time()
        last_detection_time = time.time()

        # Initial UI Setup
        ui.update_camera_status(
            camera_name=f"CAM_{VIDEO_SOURCE:02d}",
            connected=True,
            resolution=f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            fps=0.0,
            neural_link="SYNCING",
            detection_engine="YOLOv8n-ASYNC",
        )
        ui.update_drone_status(connection="OFFLINE")
        ui.update_system_status("BOOTING", "READY")

        while not ui.closed:
            start_time = time.time()
            frame = video_source.read_frame()
            
            if frame is None:
                # Still waiting for first frame
                time.sleep(0.01)
                ui.root.update()
                continue

            dt_render = time.time() - last_render_time
            last_render_time = time.time()

            if ui.paused:
                ui.update_system_status("MANUAL_PAUSE", "STDBY")
                ui.render(frame, predictions, 0.0)
                continue

            # 1. Trigger Async Detection (Submits frame if detector is idle)
            detector.detect_async(frame)

            # 2. Check for new detection results
            new_detections = detector.get_detections()
            
            if new_detections is not None:
                # We have fresh data! Update tracker and Kalman
                dt_detect = time.time() - last_detection_time
                last_detection_time = time.time()
                
                tracks = tracker.update(new_detections, frame_rate=1.0/max(1e-6, dt_detect))
                predictions = predictor.update(tracks, dt=dt_detect)
            else:
                # No new detections. Interpolate existing tracks using Kalman prediction
                # This keeps the HUD smooth even if detection is slow
                interpolated_predictions = []
                for track_id, kf in predictor.filters.items():
                    # Find base track info from last update
                    base_track = next((t for t in predictions if t["id"] == track_id), None)
                    if base_track:
                        pred_data = predictor.predict(track_id, dt_render)
                        if pred_data:
                            # Project the bounding box forward (simplified: shift by velocity)
                            vx, vy = pred_data["velocity"]
                            x1, y1, x2, y2 = base_track["bbox"]
                            shifted_bbox = [
                                x1 + vx * dt_render, 
                                y1 + vy * dt_render, 
                                x2 + vx * dt_render, 
                                y2 + vy * dt_render
                            ]
                            interpolated_predictions.append({
                                **base_track,
                                "bbox": shifted_bbox,
                                "prediction": pred_data["prediction"],
                                "velocity": pred_data["velocity"]
                            })
                predictions = interpolated_predictions

            # 3. Control Commands (Send most recent predicted target)
            for track in predictions:
                if track["id"] in ui.selected_track_ids:
                    controller.send_target_position(*track["prediction"])

            # 4. Telemetry Update
            fps = 1.0 / max(1e-6, time.time() - start_time)
            ui.update_camera_status(
                camera_name=f"CAM_{VIDEO_SOURCE:02d}",
                connected=True,
                resolution=f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
                fps=fps,
                neural_link="UP-LINK ACTIVE",
                detection_engine="YOLOv8n-ASYNC",
            )
            ui.update_system_status("NAV_TRACKING", "OP_STABLE")
            
            # 5. Render tactical feed
            ui.render(frame, predictions, fps)

    except Exception:
        logger.exception("FATAL_SYSTEM_ERROR")
    finally:
        video_source.stop()
        if 'detector' in locals():
            detector.stop()
        if not ui.closed:
            ui.close()


if __name__ == "__main__":
    main()
