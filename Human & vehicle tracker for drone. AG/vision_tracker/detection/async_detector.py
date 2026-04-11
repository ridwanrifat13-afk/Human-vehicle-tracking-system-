import threading
import queue
import time
from .yolo_detector import YOLODetector


class AsyncDetector:
    """Asynchronous wrapper for YOLO detector to prevent blocking the main loop."""

    def __init__(self, model_path="yolov8n.pt", device="cpu", confidence=0.25, iou=0.45):
        self.detector = YOLODetector(model_path, device, confidence, iou)
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        """Background worker thread for inference."""
        while self.running:
            try:
                # Wait for a frame with a timeout to check running status
                frame = self.input_queue.get(timeout=1.0)
                if frame is None:
                    continue
                
                detections = self.detector.detect(frame)
                
                # Push detections to output queue, clearing old ones if necessary
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.output_queue.put(detections)
                
            except queue.Empty:
                continue
            except Exception:
                # Log or handle detection errors here if needed
                continue

    def detect_async(self, frame):
        """Submit a frame for detection if the detector is ready."""
        if self.input_queue.empty():
            try:
                self.input_queue.put_nowait(frame)
                return True
            except queue.Full:
                return False
        return False

    def get_detections(self):
        """Retrieve the latest detections if available."""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
