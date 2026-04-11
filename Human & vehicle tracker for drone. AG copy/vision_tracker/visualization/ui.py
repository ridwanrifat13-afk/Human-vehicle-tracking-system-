
import tkinter as tk
from PIL import Image, ImageTk
import cv2


class TrackerUI:
    def __init__(self, width=1280, height=720, title="Vision Tracker"):
        self.width = width
        self.height = height
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg="#121212")
        self.root.resizable(False, False)

        self.bg_color = "#121212"
        self.panel_bg = "#1E1E1E"
        self.sidebar_bg = "#171717"
        self.text_color = "#ECECEC"
        self.ok_color = "#4FD1C5"
        self.warn_color = "#F687B3"
        self.highlight_color = "#F6AD55"
        self.photo = None
        self.video_image = None
        self.closed = False
        self.paused = False
        self.selected_track_ids = set()
        self.box_regions = []
        self.last_tracks = []

        self._create_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.canvas.bind("<Button-1>", self._on_mouse_click)
        self.root.bind("<Key-c>", self._clear_selection)

    def _create_layout(self):
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        video_panel = tk.Frame(main_frame, bg=self.bg_color)
        video_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(main_frame, width=360, bg=self.sidebar_bg)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        title_label = tk.Label(
            video_panel,
            text="Live Tracking Feed",
            bg=self.bg_color,
            fg=self.text_color,
            font=("Segoe UI", 16, "bold"),
        )
        title_label.pack(anchor=tk.NW, pady=(0, 8))

        self.canvas = tk.Canvas(
            video_panel,
            width=self.width,
            height=self.height,
            bg="#000000",
            highlightthickness=0,
        )
        self.canvas.pack()

        self.status_header = tk.Label(
            video_panel,
            text="Click targets to select, press C to clear",
            bg=self.bg_color,
            fg="#A0A0A0",
            font=("Segoe UI", 11),
        )
        self.status_header.pack(anchor=tk.W, pady=(8, 0))

        self._create_sidebar(sidebar)

    def _create_sidebar(self, parent):
        label = tk.Label(
            parent,
            text="System Dashboard",
            bg=self.sidebar_bg,
            fg=self.text_color,
            font=("Segoe UI", 14, "bold"),
        )
        label.pack(anchor=tk.NW, padx=16, pady=(16, 8))

        self.camera_frame = self._create_info_block(parent, "Camera Module")
        self.drone_frame = self._create_info_block(parent, "Drone Telemetry")
        self.system_frame = self._create_info_block(parent, "System Controls")
        self.selection_frame = self._create_info_block(parent, "Selected Target")
        self.log_frame = self._create_info_block(parent, "Event Log", height=170)

        self._build_camera_status()
        self._build_drone_status()
        self._build_system_widgets()
        self._build_selection_summary()
        self._build_log_window()

    def _create_info_block(self, parent, title, height=None):
        block = tk.Frame(parent, bg=self.panel_bg)
        block.pack(fill=tk.X, padx=16, pady=(0, 12))

        title_label = tk.Label(
            block,
            text=title,
            bg=self.panel_bg,
            fg=self.text_color,
            font=("Segoe UI", 12, "bold"),
            anchor=tk.W,
        )
        title_label.pack(fill=tk.X, pady=(12, 6), padx=12)
        if height:
            block.configure(height=height)
        return block

    def _build_camera_status(self):
        self.camera_values = self._build_status_rows(
            self.camera_frame,
            [
                ("Module:", "N/A"),
                ("Connection:", "Disconnected"),
                ("Resolution:", "N/A"),
                ("FPS:", "0.0"),
                ("Neural Link:", "Offline"),
                ("Detection Engine:", "N/A"),
            ],
        )

    def _build_drone_status(self):
        self.drone_values = self._build_status_rows(
            self.drone_frame,
            [
                ("Altitude:", "N/A"),
                ("Longitude:", "N/A"),
                ("Battery:", "N/A"),
                ("Connection:", "Disconnected"),
            ],
        )

    def _build_system_widgets(self):
        self.system_values = self._build_status_rows(
            self.system_frame,
            [
                ("Mode:", "Initializing"),
                ("State:", "Ready"),
            ],
        )

        button_frame = tk.Frame(self.system_frame, bg=self.panel_bg)
        button_frame.pack(fill=tk.X, padx=12, pady=(8, 12))

        self.pause_button = tk.Button(
            button_frame,
            text="Pause",
            command=self._toggle_pause,
            bg="#2D3748",
            fg="#000000",
            activebackground="#4A5568",
            activeforeground="#000000",
            relief=tk.FLAT,
            padx=10,
            pady=6,
            font=("Segoe UI", 10, "bold"),
            width=12,
        )
        self.pause_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))

        stop_button = tk.Button(
            button_frame,
            text="Stop",
            command=self._stop,
            bg="#C53030",
            fg="#000000",
            activebackground="#9B2C2C",
            activeforeground="#000000",
            relief=tk.FLAT,
            padx=10,
            pady=6,
            font=("Segoe UI", 10, "bold"),
            width=12,
        )
        stop_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))

        clear_button = tk.Button(
            self.system_frame,
            text="Clear Selection",
            command=self._clear_selection,
            bg="#2D3748",
            fg="#000000",
            activebackground="#4A5568",
            activeforeground="#000000",
            relief=tk.FLAT,
            padx=10,
            pady=6,
            font=("Segoe UI", 10, "bold"),
            width=18,
        )
        clear_button.pack(fill=tk.X, padx=12, pady=(0, 12))

    def _build_selection_summary(self):
        self.selected_label = tk.Label(
            self.selection_frame,
            text="No target selected",
            bg=self.panel_bg,
            fg=self.text_color,
            font=("Segoe UI", 11),
            justify=tk.LEFT,
            wraplength=320,
        )
        self.selected_label.pack(fill=tk.X, padx=12, pady=(6, 12))

    def _build_log_window(self):
        self.log_text = tk.Text(
            self.log_frame,
            bg="#101010",
            fg=self.text_color,
            insertbackground=self.text_color,
            height=10,
            wrap=tk.WORD,
            bd=0,
            highlightthickness=0,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(6, 12))
        self.log_text.configure(state=tk.DISABLED)

    def _build_status_rows(self, parent, rows):
        values = {}
        for label_text, value_text in rows:
            row = tk.Frame(parent, bg=self.panel_bg)
            row.pack(fill=tk.X, padx=12, pady=2)
            key_label = tk.Label(
                row,
                text=label_text,
                bg=self.panel_bg,
                fg="#A0A0A0",
                font=("Segoe UI", 10),
                anchor=tk.W,
                width=14,
            )
            key_label.pack(side=tk.LEFT)
            value_label = tk.Label(
                row,
                text=value_text,
                bg=self.panel_bg,
                fg=self.text_color,
                font=("Segoe UI", 10, "bold"),
                anchor=tk.W,
            )
            value_label.pack(side=tk.LEFT, fill=tk.X)
            values[label_text.strip(":")] = value_label
        return values

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
        self.selected_label.config(text="No target selected")
        self.append_log("Selection cleared.")

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")
        self.update_system_status("Paused" if self.paused else "Tracking")
        self.append_log("Tracking paused." if self.paused else "Tracking resumed.")

    def _stop(self):
        self.append_log("Stopping system...")
        self.close()

    def update_camera_status(self, camera_name="Camera", connected=False, resolution="N/A", fps=0.0, neural_link="Offline", detection_engine="N/A"):
        self.camera_values["Module"].config(text=camera_name)
        self.camera_values["Connection"].config(
            text="Connected" if connected else "Disconnected",
            fg=self.ok_color if connected else self.warn_color,
        )
        self.camera_values["Resolution"].config(text=resolution)
        self.camera_values["FPS"].config(text=f"{fps:.1f}" if isinstance(fps, float) else str(fps))
        self.camera_values["Neural Link"].config(text=neural_link)
        self.camera_values["Detection Engine"].config(text=detection_engine)

    def update_drone_status(self, altitude="N/A", longitude="N/A", battery="N/A", connection="Disconnected"):
        self.drone_values["Altitude"].config(text=altitude)
        self.drone_values["Longitude"].config(text=longitude)
        self.drone_values["Battery"].config(text=battery)
        self.drone_values["Connection"].config(
            text=connection,
            fg=self.ok_color if connection.lower() == "connected" else self.warn_color,
        )

    def update_system_status(self, mode="Ready", state="Operational"):
        self.system_values["Mode"].config(text=mode)
        self.system_values["State"].config(text=state)

    def append_log(self, message):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def render(self, frame, tracks, fps):
        if frame is None:
            return

        if tracks:
            self.last_tracks = tracks

        display_tracks = tracks if tracks else self.last_tracks
        overlay = frame.copy()
        self.box_regions = []

        for track in display_tracks:
            x1, y1, x2, y2 = map(int, track["bbox"])
            selected = track["id"] in self.selected_track_ids
            border_color = self.highlight_color if selected else "#4FD1C5"
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self._hex_to_bgr(border_color), 3 if selected else 2)
            label = f"{track['class_name']} ID {track['id']}"
            if "state" in track:
                label += f" ({track['state']})"
            text_color = (255, 255, 255)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                overlay,
                (x1, y1 - label_size[1] - 14),
                (x1 + label_size[0] + 12, y1),
                self._hex_to_bgr("#101010"),
                -1,
            )
            cv2.putText(
                overlay,
                label,
                (x1 + 6, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                cv2.LINE_AA,
            )

            if "confidence" in track:
                confidence_text = f"{track['confidence']:.2f}"
                cv2.putText(
                    overlay,
                    confidence_text,
                    (x1 + 6, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    self._hex_to_bgr(border_color),
                    1,
                    cv2.LINE_AA,
                )

            if "prediction" in track:
                px, py = map(int, track["prediction"])
                cv2.circle(overlay, (px, py), 5, (255, 128, 0), -1)
                cv2.putText(
                    overlay,
                    f"P({px},{py})",
                    (px + 8, py + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            self.box_regions.append({
                "id": track["id"],
                "bbox": (x1, y1, x2, y2),
            })

        if self.paused:
            cv2.putText(
                overlay,
                "PAUSED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        selected_text = "No target selected"
        if self.selected_track_ids:
            selected_info = []
            for track in display_tracks:
                if track["id"] in self.selected_track_ids:
                    selected_info.append(f"ID {track['id']} ({track['class_name']})")
            selected_text = "Selected: " + ", ".join(selected_info)
        self.selected_label.config(text=selected_text)

        self.update_camera_status(
            camera_name=self.camera_values["Module"].cget("text"),
            connected=self.camera_values["Connection"].cget("text") == "Connected",
            resolution=self.camera_values["Resolution"].cget("text"),
            fps=fps,
            neural_link=self.camera_values["Neural Link"].cget("text"),
            detection_engine=self.camera_values["Detection Engine"].cget("text"),
        )

        image = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        self.photo = ImageTk.PhotoImage(image=image)
        if self.video_image is None:
            self.video_image = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        else:
            self.canvas.itemconfig(self.video_image, image=self.photo)

        self.root.update_idletasks()
        self.root.update()

    def _hex_to_bgr(self, hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (4, 2, 0))
