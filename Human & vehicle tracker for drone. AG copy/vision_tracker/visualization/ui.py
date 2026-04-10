import tkinter as tk
from PIL import Image, ImageTk
import cv2


class TrackerUI:
    def __init__(self, width=1280, height=720, title="Vision Tracker"):
        self.width = width
        self.height = height
        self.root = tk.Tk()
        self.root.title(title)
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
        self.canvas.pack()
        self.info_label = tk.Label(self.root, text="Initializing tracker...", font=("Arial", 12))
        self.info_label.pack(fill=tk.X)
        self.photo = None
        self.closed = False
        self.selected_track_ids = set()
        self.box_regions = []
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.canvas.bind("<Button-1>", self._on_mouse_click)
        self.root.bind("<Key-c>", self._clear_selection)

    def close(self):
        self.closed = True
        self.root.quit()

    def _on_mouse_click(self, event):
        for region in self.box_regions:
            x1, y1, x2, y2 = region["bbox"]
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                track_id = region["id"]
                if track_id in self.selected_track_ids:
                    self.selected_track_ids.remove(track_id)
                else:
                    self.selected_track_ids.add(track_id)
                return

    def _clear_selection(self, event=None):
        self.selected_track_ids.clear()

    def render(self, frame, tracks, fps):
        overlay = frame.copy()
        self.box_regions = []

        if self.selected_track_ids:
            display_tracks = [t for t in tracks if t["id"] in self.selected_track_ids]
        else:
            display_tracks = tracks

        for track in display_tracks:
            x1, y1, x2, y2 = map(int, track["bbox"])
            selected = track["id"] in self.selected_track_ids
            color = (0, 165, 255) if selected else (0, 255, 0)
            label = f"{track['class_name']} | ID {track['id']}"
            if "state" in track:
                label += f" ({track['state']})"
            confidence_text = f"{track['confidence']:.2f}"

            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_y1 = max(0, y1 - text_size[1] - 8)
            label_y2 = y1
            border_color = (0, 215, 255) if selected else color
            bg_color = (255, 255, 255) if selected else color
            text_color = (0, 0, 0) if selected else (0, 0, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(overlay, (cx, cy), 5, color, -1)
            cv2.rectangle(overlay, (x1, label_y1), (x1 + text_size[0] + 10, label_y2), bg_color, -1)
            cv2.rectangle(overlay, (x1, label_y1), (x1 + text_size[0] + 10, label_y2), border_color, 1)
            cv2.putText(overlay, label, (x1 + 5, label_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.putText(overlay, confidence_text, (x1 + 5, label_y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            px, py = map(int, track["prediction"])
            cv2.circle(overlay, (px, py), 4, (255, 0, 0), -1)
            cv2.putText(overlay, f"P({px},{py})", (px + 6, py + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            if "velocity" in track:
                vx, vy = track["velocity"]
                cv2.putText(overlay, f"v={vx:.1f},{vy:.1f}", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            self.box_regions.append({
                "id": track["id"],
                "bbox": (x1, y1, x2, y2),
            })

        # Display status and coordinate information in the top-right corner.
        status_lines = [f"FPS: {fps:.1f}", f"Tracks: {len(tracks)}"]
        if self.selected_track_ids:
            selected_info = [f"Selected: {sorted(self.selected_track_ids)}"]
            coords = []
            for track in display_tracks:
                x1, y1, x2, y2 = map(int, track["bbox"])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                coords.append(f"ID {track['id']}: ({cx}, {cy})")
            status_lines += selected_info + coords
        else:
            status_lines.append("Click box to select target(s)")
            status_lines.append("Press C to clear selection")

        for index, line in enumerate(status_lines):
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            x = self.width - text_size[0] - 10
            y = 20 + index * 18
            cv2.putText(overlay, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        self.info_label.config(text=f"Rendered {len(display_tracks)} target(s)")

        image = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.root.update_idletasks()
        self.root.update()
