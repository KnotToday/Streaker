# ------------------------------------------------------------------------------
# Script Name:     StreakerPlayBack_V2.py
#
# Description:     Plays back stacked PNG frames with overlay showing the number
#                  of detected objects per frame. Called automatically after
#                  Full_Detect_counts.py finishes processing.
# ------------------------------------------------------------------------------

import os
import cv2
import tkinter as tk
from tkinter import Toplevel, Button, Scale, Label, Frame
from PIL import Image, ImageTk


class DetectionVideoPlayer:
    def __init__(self, root, frame_paths, fps=10, title="Streaker Playback", autoplay=False, object_counts=None):
        self.root = root
        self.fps = fps
        self.frame_paths = frame_paths
        self.frames = [cv2.imread(p) for p in frame_paths if os.path.exists(p)]
        self.frame_index = 0
        self.paused = not autoplay
        self.loop = True

        self.source_folder = os.path.dirname(frame_paths[0]) if frame_paths else ""
        # Accept object counts or default to 0s
        self.object_counts = object_counts if object_counts else [0] * len(self.frames)

        self.window = Toplevel(root)
        self.window.title(f"{title} - {os.path.basename(__file__)}")
        self.label = Label(self.window)
        self.label.pack()

        controls = Frame(self.window)
        controls.pack(pady=5)

        Button(controls, text="Play/Pause", command=self.toggle_pause).pack(side=tk.LEFT)
        Button(controls, text="Step Forward", command=self.step_forward).pack(side=tk.LEFT)
        Button(controls, text="Step Back", command=self.step_back).pack(side=tk.LEFT)
        Button(controls, text="Loop On/Off", command=self.toggle_loop).pack(side=tk.LEFT)

        self.speed_scale = Scale(controls, from_=0.25, to=4.0, resolution=0.25,
                                 orient=tk.HORIZONTAL, label="Speed")
        self.speed_scale.set(1.0)
        self.speed_scale.pack(side=tk.LEFT)

        self.display_frame()
        self.play_loop()

    def toggle_pause(self):
        self.paused = not self.paused

    def toggle_loop(self):
        self.loop = not self.loop

    def step_forward(self):
        self.paused = True
        self.frame_index = (self.frame_index + 1) % len(self.frames)
        self.display_frame()

    def step_back(self):
        self.paused = True
        self.frame_index = (self.frame_index - 1) % len(self.frames)
        self.display_frame()

    def display_frame(self):
        frame = self.frames[self.frame_index].copy()

        # Draw object count overlay
        count = self.object_counts[self.frame_index] if self.frame_index < len(self.object_counts) else 0

        overlay_text = f"Objects: {count}"
        cv2.putText(frame, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.label.config(image=image)
        self.label.image = image

    def play_loop(self):
        if not self.paused:
            self.display_frame()
            self.frame_index += 1
            if self.frame_index >= len(self.frames):
                self.frame_index = 0 if self.loop else len(self.frames) - 1
        delay = int(1000 / (self.fps * self.speed_scale.get()))
        self.window.after(delay, self.play_loop)


# --- Standalone test usage ---
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    test_folder = "T:/Dahua_MKV_Streaker/Stacks/2025-06-15_00-00-00"
    test_frames = [os.path.join(test_folder, f) for f in sorted(os.listdir(test_folder)) if f.endswith(".png")][:50]
    dummy_counts = [3 if i % 10 == 0 else 1 for i in range(len(test_frames))]  # Example dummy data
    DetectionVideoPlayer(root, test_frames, object_counts=dummy_counts)
    root.mainloop()
