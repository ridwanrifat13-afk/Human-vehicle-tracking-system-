import numpy as np
from config.settings import TRACK_IOU_THRESHOLD, MAX_LOST_FRAMES, TRACK_CONFIRMATION_FRAMES


def iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


class Track:
    def __init__(self, track_id, bbox, class_name, confidence):
        self.track_id = track_id
        self.bbox = bbox
        self.class_name = class_name
        self.confidence = confidence
        self.age = 1
        self.consecutive_invisible = 0
        self.state = "tentative"

    def update(self, det):
        self.bbox = det["bbox"]
        self.class_name = det["class_name"]
        self.confidence = det["confidence"]
        self.consecutive_invisible = 0
        self.age += 1
        if self.state == "tentative" and self.age >= TRACK_CONFIRMATION_FRAMES:
            self.state = "confirmed"

    def mark_lost(self):
        self.consecutive_invisible += 1
        self.age += 1

    def is_active(self):
        return self.consecutive_invisible <= MAX_LOST_FRAMES


class ByteTrackTracker:
    def __init__(self, iou_threshold=TRACK_IOU_THRESHOLD, max_lost=MAX_LOST_FRAMES):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks = {}
        self.next_id = 1

    def _match_detections(self, detections):
        track_list = list(self.tracks.values())
        if not track_list or not detections:
            return [], list(range(len(track_list))), list(range(len(detections)))

        iou_pairs = []
        for track_index, track in enumerate(track_list):
            for det_index, det in enumerate(detections):
                score = iou(track.bbox, det["bbox"])
                iou_pairs.append((score, track_index, det_index))

        iou_pairs.sort(key=lambda item: item[0], reverse=True)
        matched_tracks = set()
        matched_detections = set()
        matches = []

        for score, track_index, det_index in iou_pairs:
            if score < self.iou_threshold:
                break
            if track_index in matched_tracks or det_index in matched_detections:
                continue
            matched_tracks.add(track_index)
            matched_detections.add(det_index)
            matches.append((track_index, det_index))

        unmatched_tracks = [i for i in range(len(track_list)) if i not in matched_tracks]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]
        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections, frame_rate=30):
        updated_tracks = {}
        track_list = list(self.tracks.values())
        matches, unmatched_tracks, unmatched_detections = self._match_detections(detections)

        for track_index, det_index in matches:
            track = track_list[track_index]
            det = detections[det_index]
            track.update(det)
            updated_tracks[track.track_id] = track

        for track_index in unmatched_tracks:
            track = track_list[track_index]
            track.mark_lost()
            if track.is_active():
                updated_tracks[track.track_id] = track

        for det_index in unmatched_detections:
            det = detections[det_index]
            track = Track(self.next_id, det["bbox"], det["class_name"], det["confidence"])
            updated_tracks[self.next_id] = track
            self.next_id += 1

        self.tracks = updated_tracks
        return [
            {
                "id": track.track_id,
                "bbox": track.bbox,
                "class_name": track.class_name,
                "confidence": track.confidence,
                "state": track.state,
            }
            for track in self.tracks.values()
        ]
