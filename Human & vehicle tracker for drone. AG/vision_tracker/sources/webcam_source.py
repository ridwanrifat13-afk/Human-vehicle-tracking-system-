import cv2
import threading
import time
from .video_source import VideoSource


class WebcamSource(VideoSource):
    """Threaded webcam video source implementation for low-latency acquisition."""

    def __init__(self, source=0, width=1280, height=720, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.capture = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        self.capture = cv2.VideoCapture(self.source, cv2.CAP_ANY)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(f"Unable to open webcam source: {self.source}")

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Internal thread loop to continuously read frames."""
        while self.running:
            if self.capture is not None:
                success, frame = self.capture.read()
                if success:
                    with self.lock:
                        self.frame = frame
                else:
                    time.sleep(0.01)
            else:
                break

    def read_frame(self):
        """Retrieve the latest frame captured by the thread."""
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
