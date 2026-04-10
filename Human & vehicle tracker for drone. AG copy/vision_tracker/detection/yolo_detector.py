from ultralytics import YOLO
from config.settings import MODEL_NAME, DEVICE, DETECTION_CLASSES, CONFIDENCE_THRESHOLD, IOU_THRESHOLD

COCO_CLASS_NAME_TO_ID = {
    "person": 0,
    "car": 2,
    "bus": 5,
    "truck": 7,
    "motorcycle": 3,
}

SUPPORTED_CLASS_IDS = [COCO_CLASS_NAME_TO_ID[name] for name in DETECTION_CLASSES if name in COCO_CLASS_NAME_TO_ID]


class YOLODetector:
    def __init__(self, model_path=MODEL_NAME, device=DEVICE, confidence=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD):
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence
        self.iou = iou

    def detect(self, frame):
        detections = []
        results = self.model(frame, device=self.device, conf=self.confidence, iou=self.iou, classes=SUPPORTED_CLASS_IDS)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in SUPPORTED_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(box.conf[0]),
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                })
        return detections
