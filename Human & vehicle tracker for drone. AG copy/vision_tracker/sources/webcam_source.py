import cv2
from .video_source import VideoSource


class WebcamSource(VideoSource):
    """Webcam video source implementation using OpenCV."""

    def __init__(self, source=0, width=1280, height=720, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.capture = None

    def start(self):
        self.capture = cv2.VideoCapture(self.source, cv2.CAP_ANY)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(f"Unable to open webcam source: {self.source}")

    def read_frame(self):
        if self.capture is None or not self.capture.isOpened():
            return None
        success, frame = self.capture.read()
        return frame if success else None

    def stop(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
