# ------------------------------------------------------------------------------
# Script Name:     StreakerPlayBack_V2.py
# Description:     Playback viewer for detection results with lazy frame loading,
#                  object count overlays, and frame navigation controls.
#                  Supports diff mask overlay from detection process.
# ------------------------------------------------------------------------------

import os
import cv2
import tkinter as tk
from tkinter import Toplevel
from PIL import Image, ImageTk
import numpy as np

class DetectionVideoPlayer:
    def __init__(self, master, frame_paths, object_counts=None, diff_masks=None, autoplay=False):
        self.master = master
        self.frame_paths = frame_paths
        self.object_counts = object_counts if object_counts else [0] * len(frame_paths)
        self.diff_masks = diff_masks if diff_masks else [None] * len(frame_paths)
        self.frame_index = 0
        self.frame_count = len(frame_paths)
        self.autoplay = autoplay
        self.paused = not autoplay
        self.playback_speed = 100  # milliseconds per frame
        self.play_loop_id = None

        self.window = Toplevel(master)
        self.window.title(f"Detection Viewer - {os.path.basename(__file__)}")

        self.canvas = tk.Canvas(self.window, width=800, height=600)
        self.canvas.pack(fill="both", expand=True)
        self.window.bind("<Right>", lambda e: self.step_frame(pause_after=True))
        self.window.bind("<Left>", lambda e: self.step_frame(reverse=True, pause_after=True))
        self.window.bind("<space>", lambda e: self.toggle_play())
        self.window.bind("<Up>", lambda e: self.change_speed(faster=True))
        self.window.bind("<Down>", lambda e: self.change_speed(faster=False))

        self.current_image = None
        self.aspect_ratio = None

        self.window.bind("<Configure>", self.on_resize)
        self.display_width = 800
        self.display_height = 600

        self.update_frame()

        if self.autoplay:
            self.play_loop()

    def on_resize(self, event):
        self.display_width = event.width
        self.display_height = event.height
        self.update_frame()

    def toggle_play(self):
        self.paused = not self.paused
        if self.paused:
            if self.play_loop_id:
                self.window.after_cancel(self.play_loop_id)
                self.play_loop_id = None
        else:
            self.play_loop()

    def change_speed(self, faster=True):
        if faster:
            self.playback_speed = max(10, self.playback_speed - 10)
        else:
            self.playback_speed += 10
        print(f"[DEBUG] Playback speed set to {self.playback_speed} ms per frame")

    def play_loop(self):
        if not self.paused:
            self.step_frame()
            self.play_loop_id = self.window.after(self.playback_speed, self.play_loop)

    def step_frame(self, reverse=False, pause_after=False):
        if self.frame_count == 0:
            return
        self.frame_index = (self.frame_index - 1 if reverse else self.frame_index + 1) % self.frame_count
        self.update_frame()
        if pause_after:
            self.paused = True
            if self.play_loop_id:
                self.window.after_cancel(self.play_loop_id)
                self.play_loop_id = None

    def update_frame(self):
        if not self.frame_paths:
            return

        path = self.frame_paths[self.frame_index]
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            return

        object_count = self.object_counts[self.frame_index] if self.object_counts else 0
        diff_mask_path = self.diff_masks[self.frame_index] if self.diff_masks else None

        if diff_mask_path and os.path.exists(diff_mask_path):
            diff = cv2.imread(diff_mask_path, cv2.IMREAD_GRAYSCALE)
            if diff is not None:
                blue_overlay = np.zeros_like(frame)
                blue_overlay[:, :, 0] = diff  # Blue channel
                frame = cv2.addWeighted(frame, 1.0, blue_overlay, 0.6, 0)

        if object_count > 0:
            text = f"Objects: {object_count}"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 0, 0), 4)

        text2 = f"Frame {self.frame_index+1}/{self.frame_count}"
        cv2.putText(frame, text2, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)

        height, width = frame.shape[:2]
        self.aspect_ratio = width / height

        canvas_ratio = self.display_width / self.display_height
        if canvas_ratio > self.aspect_ratio:
            new_height = self.display_height
            new_width = int(new_height * self.aspect_ratio)
        else:
            new_width = self.display_width
            new_height = int(new_width / self.aspect_ratio)

        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        self.current_image = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image((self.display_width - new_width) // 2, (self.display_height - new_height) // 2, anchor="nw", image=self.current_image)
