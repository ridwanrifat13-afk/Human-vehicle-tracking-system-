import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np


class TrackerUI:
    """Refined Drone-Style HUD UI with High-DPI support and telemetry overlays."""

    def __init__(self, width=1280, height=720, title="Vision Tracker Pro"):
        self.width = width
        self.height = height
        
        # High-DPI Awareness for macOS/Windows
        self.root = tk.Tk()
        try:
            self.root.tk.call('tk', 'scaling', 2.0)
        except Exception:
            pass
            
        self.root.title(title)
        self.root.configure(bg="#0A0A0B")
        self.root.resizable(False, False)

        # Premium Color Palette
        self.bg_color = "#0A0A0B"
        self.panel_bg = "#121214"
        self.sidebar_bg = "#0D0D0F"
        self.text_color = "#F0F0F0"
        self.ok_color = "#00F5FF"  # Cyan
        self.warn_color = "#FF3131" # Red
        self.highlight_color = "#FFD700" # Gold
        
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
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Video Section
        video_panel = tk.Frame(main_frame, bg=self.bg_color)
        video_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Sidebar Section
        sidebar = tk.Frame(main_frame, width=340, bg=self.sidebar_bg)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))

        # Title with a modern touch
        title_container = tk.Frame(video_panel, bg=self.bg_color)
        title_container.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            title_container,
            text="MISSION CONTROL",
            bg=self.bg_color,
            fg=self.ok_color,
            font=("Inter", 10, "bold"),
        ).pack(side=tk.LEFT)
        
        tk.Label(
            title_container,
            text=" | LIVE_TACTICAL_FEED",
            bg=self.bg_color,
            fg="#505050",
            font=("Inter", 10),
        ).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(
            video_panel,
            width=self.width,
            height=self.height,
            bg="#000000",
            highlightthickness=1,
            highlightbackground="#222222",
        )
        self.canvas.pack()

        self.status_header = tk.Label(
            video_panel,
            text="[M1] SELECT_TARGET | [C] CLEAR_RESET",
            bg=self.bg_color,
            fg="#606060",
            font=("Courier New", 9),
        )
        self.status_header.pack(anchor=tk.W, pady=(10, 0))

        self._create_sidebar(sidebar)

    def _create_sidebar(self, parent):
        padding_frame = tk.Frame(parent, bg=self.sidebar_bg)
        padding_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        tk.Label(
            padding_frame,
            text="SYSTEM TELEMETRY",
            bg=self.sidebar_bg,
            fg=self.text_color,
            font=("Inter", 12, "bold"),
        ).pack(anchor=tk.NW, pady=(0, 16))

        self.camera_frame = self._create_info_block(padding_frame, "VISION_ENGINE")
        self.drone_frame = self._create_info_block(padding_frame, "FLIGHT_STATUS")
        self.system_frame = self._create_info_block(padding_frame, "OPERATIONS")
        self.selection_frame = self._create_info_block(padding_frame, "TARGET_LOCK")
        self.coords_frame = self._create_info_block(padding_frame, "TARGET_COORDINATES")
        self.log_frame = self._create_info_block(padding_frame, "TERMINAL_OUTPUT", height=120)

        self._build_camera_status()
        self._build_drone_status()
        self._build_system_widgets()
        self._build_selection_summary()
        self._build_coords_summary()
        self._build_log_window()

    def _create_info_block(self, parent, title, height=None):
        block = tk.Frame(parent, bg=self.panel_bg, padx=12, pady=12)
        block.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            block,
            text=title,
            bg=self.panel_bg,
            fg="#808080",
            font=("Inter", 9, "bold"),
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 6))
        
        if height:
            block.configure(height=height)
            block.pack_propagate(False)
        return block

    def _build_camera_status(self):
        self.camera_values = self._build_status_rows(
            self.camera_frame,
            [
                ("ID:", "CAM_01"),
                ("LINK:", "OFFLINE"),
                ("RES:", "0x0"),
                ("FPS:", "0.0"),
                ("CORE:", "YOLOv8n"),
            ],
        )

    def _build_drone_status(self):
        self.drone_values = self._build_status_rows(
            self.drone_frame,
            [
                ("ALT:", "--- M"),
                ("SPD:", "--- KPH"),
                ("BAT:", "--- %"),
                ("COM:", "PENDING"),
            ],
        )

    def _build_system_widgets(self):
        self.system_values = self._build_status_rows(
            self.system_frame,
            [
                ("MODE:", "IDLE"),
                ("STAT:", "WAITING"),
            ],
        )

        btn_container = tk.Frame(self.system_frame, bg=self.panel_bg)
        btn_container.pack(fill=tk.X, pady=(10, 0))

        self.pause_button = tk.Button(
            btn_container,
            text="PAUSE_FEED",
            command=self._toggle_pause,
            bg="#A0A0A0",
            fg="#000000",
            activebackground="#CCCCCC",
            relief=tk.FLAT,
            font=("Inter", 8, "bold"),
            height=2,
        )
        self.pause_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        tk.Button(
            btn_container,
            text="ABORT",
            command=self._stop,
            bg="#3D1A1A",
            fg="#FF5050",
            activebackground="#5D1A1A",
            relief=tk.FLAT,
            font=("Inter", 8, "bold"),
            height=2,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

    def _build_selection_summary(self):
        self.selected_label = tk.Label(
            self.selection_frame,
            text="AWAITING_LOCK",
            bg=self.panel_bg,
            fg="#A0A0A0",
            font=("Courier New", 10),
            justify=tk.LEFT,
            wraplength=300,
        )
        self.selected_label.pack(fill=tk.X, pady=(2, 0))

    def _build_coords_summary(self):
        self.coords_label = tk.Label(
            self.coords_frame,
            text="---",
            bg=self.panel_bg,
            fg=self.ok_color,
            font=("Courier New", 9),
            justify=tk.LEFT,
            wraplength=300,
        )
        self.coords_label.pack(fill=tk.X, pady=(2, 0))

    def _build_log_window(self):
        self.log_text = tk.Text(
            self.log_frame,
            bg="#0F0F10",
            fg="#00F5FF",
            insertbackground=self.text_color,
            font=("Courier New", 8),
            wrap=tk.WORD,
            bd=0,
            highlightthickness=0,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

    def _build_status_rows(self, parent, rows):
        values = {}
        for label_text, value_text in rows:
            row = tk.Frame(parent, bg=self.panel_bg)
            row.pack(fill=tk.X, pady=1)
            tk.Label(
                row,
                text=label_text,
                bg=self.panel_bg,
                fg="#505050",
                font=("Courier New", 9, "bold"),
                anchor=tk.W,
                width=8,
            ).pack(side=tk.LEFT)
            v_label = tk.Label(
                row,
                text=value_text,
                bg=self.panel_bg,
                fg=self.text_color,
                font=("Courier New", 9, "bold"),
                anchor=tk.W,
            )
            v_label.pack(side=tk.LEFT, fill=tk.X)
            values[label_text.strip(":")] = v_label
        return values

    def close(self):
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def _on_mouse_click(self, event):
        for region in self.box_regions:
            x1, y1, x2, y2 = region["bbox"]
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                track_id = region["id"]
                label = region["label"]
                if track_id in self.selected_track_ids:
                    self.selected_track_ids.remove(track_id)
                    self.append_log(f"LOCK_RELEASED: {label}")
                else:
                    self.selected_track_ids.add(track_id)
                    self.append_log(f"LOCK_ACQUIRED: {label}")
                return

    def _clear_selection(self, event=None):
        self.selected_track_ids.clear()
        self.selected_label.config(text="AWAITING_LOCK")
        self.coords_label.config(text="---")
        self.append_log("CMD: RESET_SELECTION")

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.config(text="RESUME" if self.paused else "PAUSE")
        self.append_log("SYS: FEED_" + ("PAUSED" if self.paused else "RESUMED"))

    def _stop(self):
        self.append_log("SYS: ABORT_SEQUENCE")
        self.close()

    def update_camera_status(self, camera_name="CAM_01", connected=False, resolution="0x0", fps=0.0, neural_link="OFFLINE", detection_engine="N/A"):
        self.camera_values["ID"].config(text=camera_name)
        self.camera_values["LINK"].config(
            text="ACTIVE" if connected else "LOST",
            fg=self.ok_color if connected else self.warn_color,
        )
        self.camera_values["RES"].config(text=resolution)
        self.camera_values["FPS"].config(text=f"{fps:.1f}" if isinstance(fps, (float, int)) else str(fps))
        self.camera_values["CORE"].config(text=detection_engine)

    def update_drone_status(self, altitude="--- M", speed="--- KPH", battery="--- %", connection="OFFLINE"):
        self.drone_values["ALT"].config(text=altitude)
        self.drone_values["SPD"].config(text=speed)
        self.drone_values["BAT"].config(text=battery)
        self.drone_values["COM"].config(
            text=connection,
            fg=self.ok_color if connection.lower() == "online" else self.warn_color,
        )

    def update_system_status(self, mode="IDLE", state="WAITING"):
        self.system_values["MODE"].config(text=mode)
        self.system_values["STAT"].config(text=state)

    def append_log(self, message):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"> {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def render(self, frame, tracks, fps):
        if frame is None:
            return

        if tracks:
            self.last_tracks = tracks

        overlay = frame.copy()
        self.box_regions = []
        selected_coords = []

        # HUD Decorative Corner Elements (Static)
        h, w = overlay.shape[:2]
        thickness = 1
        cv2.line(overlay, (20, 20), (60, 20), self._hex_to_bgr(self.ok_color), thickness)
        cv2.line(overlay, (20, 20), (20, 60), self._hex_to_bgr(self.ok_color), thickness)
        
        for track in (tracks if tracks else self.last_tracks):
            x1, y1, x2, y2 = map(int, track["bbox"])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            selected = track["id"] in self.selected_track_ids
            color = self.highlight_color if selected else self.ok_color
            bgr_color = self._hex_to_bgr(color)
            line_w = 4 if selected else 2

            # Solid Bounding Box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr_color, line_w)

            # Labels - format: CLASS_ID (e.g., PERSON_01)
            label = f"{track['class_name'].upper()}_{track['id']:02d}"
            cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr_color, 1, cv2.LINE_AA)

            if selected:
                selected_coords.append(f"{label}: ({cx}, {cy})")

            # Telemetry (Prediction & Velocity)
            if "velocity" in track:
                vx, vy = track["velocity"]
                v_mag = np.sqrt(vx**2 + vy**2)
                cv2.putText(overlay, f"V_{v_mag:.1f}PX/F", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, bgr_color, 1, cv2.LINE_AA)

            if "prediction" in track:
                px, py = map(int, track["prediction"])
                cv2.drawMarker(overlay, (px, py), bgr_color, cv2.MARKER_CROSS, 8, 1)

            # Map the region for interaction
            self.box_regions.append({"id": track["id"], "bbox": (x1, y1, x2, y2), "label": label})

        # Crosshair in center
        center_x, center_y = w // 2, h // 2
        cv2.line(overlay, (center_x - 10, center_y), (center_x + 10, center_y), (255, 255, 255), 1)
        cv2.line(overlay, (center_x, center_y - 10), (center_x, center_y + 10), (255, 255, 255), 1)

        # Update Sidebar
        selected_text = "AWAITING_LOCK"
        if self.selected_track_ids:
            sel_list = []
            for track in (tracks if tracks else self.last_tracks):
                if track["id"] in self.selected_track_ids:
                    sel_list.append(f"{track['class_name'].upper()}_{track['id']:02d}")
            selected_text = "LOCKED: " + ", ".join(sel_list)
        
        self.selected_label.config(text=selected_text)
        
        if selected_coords:
            self.coords_label.config(text="\n".join(selected_coords))
        else:
            self.coords_label.config(text="---")

        # Render to Tkinter
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
