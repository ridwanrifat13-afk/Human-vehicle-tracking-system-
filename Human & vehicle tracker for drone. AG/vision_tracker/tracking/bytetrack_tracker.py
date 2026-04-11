import numpy as np
from scipy.optimize import linear_sum_assignment
from config.settings import TRACK_IOU_THRESHOLD, MAX_LOST_FRAMES, TRACK_CONFIRMATION_FRAMES


def calculate_ious(bboxes1, bboxes2):
    """Vectorized IOU calculation for two sets of bounding boxes."""
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.zeros((len(bboxes1), len(bboxes2)))

    bboxes1 = np.array(bboxes1)
    bboxes2 = np.array(bboxes2)

    x11, y11, x12, y12 = bboxes1[:, 0], bboxes1[:, 1], bboxes1[:, 2], bboxes1[:, 3]
    x21, y21, x22, y22 = bboxes2[:, 0], bboxes2[:, 1], bboxes2[:, 2], bboxes2[:, 3]

    yi1 = np.maximum(y11[:, None], y21)
    xi1 = np.maximum(x11[:, None], x21)
    yi2 = np.minimum(y12[:, None], y22)
    xi2 = np.minimum(x12[:, None], x22)
    
    inter_area = np.maximum(0, yi2 - yi1) * np.maximum(0, xi2 - xi1)
    
    area1 = (y12 - y11) * (x12 - x11)
    area2 = (y22 - y21) * (x22 - x21)
    union_area = area1[:, None] + area2 - inter_area
    
    return inter_area / np.maximum(union_area, 1e-6)


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

    def _assign_labels(self, track_indices, detection_indices, iou_matrix, threshold):
        """Perform optimal assignment using the Hungarian Algorithm."""
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return [], track_indices, detection_indices

        # Hungarian algorithm minimizes cost, so we use 1 - IOU
        cost_matrix = 1 - iou_matrix[track_indices][:, detection_indices]
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        matched_tracks = set()
        matched_detections = set()

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[track_indices[r], detection_indices[c]] >= threshold:
                matches.append((track_indices[r], detection_indices[c]))
                matched_tracks.add(track_indices[r])
                matched_detections.add(detection_indices[c])

        unmatched_tracks = [i for i in track_indices if i not in matched_tracks]
        unmatched_detections = [i for i in detection_indices if i not in matched_detections]

        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections, frame_rate=30):
        # 1. Split detections based on confidence (ByteTrack principle)
        high_score_dets = []
        low_score_dets = []
        
        for i, det in enumerate(detections):
            if det["confidence"] >= 0.6: # High confidence threshold
                high_score_dets.append(i)
            else:
                low_score_dets.append(i)

        current_tracks = list(self.tracks.values())
        track_bboxes = [t.bbox for t in current_tracks]
        det_bboxes = [d["bbox"] for d in detections]
        
        iou_matrix = calculate_ious(track_bboxes, det_bboxes)
        
        # 2. First association: high score detections
        matches1, unmatched_tracks1, unmatched_detections1 = self._assign_labels(
            list(range(len(current_tracks))), 
            high_score_dets, 
            iou_matrix, 
            self.iou_threshold
        )

        # 3. Second association: remaining tracks with low score detections
        # Only try to match confirmed tracks that were not matched in first stage
        remaining_track_indices = [i for i in unmatched_tracks1 if current_tracks[i].state == "confirmed"]
        matches2, unmatched_tracks2, unmatched_detections2 = self._assign_labels(
            remaining_track_indices,
            low_score_dets,
            iou_matrix,
            0.5 # Stricter IOU for low-confidence to avoid false positives
        )

        # 4. Final Processing
        updated_tracks = {}
        all_matches = matches1 + matches2
        matched_track_indices = {m[0] for m in all_matches}
        
        # Update matched tracks
        for t_idx, d_idx in all_matches:
            track = current_tracks[t_idx]
            track.update(detections[d_idx])
            updated_tracks[track.track_id] = track

        # Handle unmatched tracks (mark lost)
        unmatched_final_tracks = [i for i in range(len(current_tracks)) if i not in matched_track_indices]
        for t_idx in unmatched_final_tracks:
            track = current_tracks[t_idx]
            track.mark_lost()
            if track.is_active():
                updated_tracks[track.track_id] = track

        # Initialize new tracks from unmatched high-score detections
        for d_idx in unmatched_detections1:
            det = detections[d_idx]
            new_track = Track(self.next_id, det["bbox"], det["class_name"], det["confidence"])
            updated_tracks[self.next_id] = new_track
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
