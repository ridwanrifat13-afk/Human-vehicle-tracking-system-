import os

try:
    import torch
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
except ImportError:
    DEVICE = os.environ.get("VISION_TRACKER_DEVICE", "cpu")

VIDEO_SOURCE = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_RATE = 30

DETECTION_CLASSES = [
    "person",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "tank",
]

MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
TRACK_IOU_THRESHOLD = 0.3
MAX_LOST_FRAMES = 30
TRACK_CONFIRMATION_FRAMES = 2

# Visualization and UI
WINDOW_TITLE = "Vision Tracker"
