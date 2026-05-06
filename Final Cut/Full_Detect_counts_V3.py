# ------------------------------------------------------------------------------
# Script Name:     Full_Detect_counts.py
# Description:     Detection module launched from overlay viewer. Uses temporal
#                  differencing and mask to detect motion in stacked frames,
#                  displays results, and launches playback viewer with overlays.
# ------------------------------------------------------------------------------

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button, ttk
from glob import glob
from collections import defaultdict
from datetime import timedelta
import time
import traceback

from StreakerPlayBack_V2 import DetectionVideoPlayer

# --- CONFIGURATION ---
DIFF_THRESHOLD = 25
STACK_FOLDER = ""
MASK_PATH = ""
FRAME_RATE = 10

def load_mask(path):
    if os.path.exists(path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    return None

def detect_events(diff_img, min_area=40):
    contours, _ = cv2.findContours(diff_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    return detections

class DetectionGUI:
    def __init__(self, master):
        self.master = master
        master.title(f"Detection - {os.path.basename(__file__)}")

        style = ttk.Style(master)
        style.theme_use("default")

        self.label = Label(master, text=f"Running detection with threshold: {DIFF_THRESHOLD}")
        self.label.pack(pady=5)

        self.status_label = Label(master, text="Idle", fg="blue")
        self.status_label.pack(pady=5)

        self.progress = ttk.Progressbar(master, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=5)

        self.eta_label = Label(master, text="")
        self.eta_label.pack(pady=5)

        self.run_button = Button(master, text="Run Detection", command=self.run_detection)
        self.run_button.pack(pady=5)

    def run_detection(self):
        try:
            print("[DEBUG] Run detection started")
            self.status_label.config(text="Processing...", fg="orange")
            self.master.update_idletasks()

            mask = load_mask(MASK_PATH)
            detection_log = defaultdict(list)

            frame_paths = sorted(glob(os.path.join(STACK_FOLDER, "*.png")))
            print(f"[DEBUG] Found {len(frame_paths)} frame(s) in: {STACK_FOLDER}")
            self.progress["maximum"] = len(frame_paths)

            diff_overlay_paths = []
            object_counts = []
            previous_frame = None
            start_time = time.time()

            for i, img_path in enumerate(frame_paths):
                print(f"[DEBUG] Processing frame {i+1}: {img_path}")
                try:
                    frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if frame is None:
                        print(f"[WARNING] Skipping unreadable frame: {img_path}")
                        diff_overlay_paths.append(None)
                        object_counts.append(0)
                        continue

                    if mask is not None:
                        frame = cv2.bitwise_and(frame, frame, mask=mask)

                    if previous_frame is not None:
                        diff = cv2.absdiff(frame, previous_frame)
                        diff = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
                        detections = detect_events(diff)
                        count = len(detections)
                        object_counts.append(count)

                        if count > 0:
                            overlay_path = os.path.splitext(img_path)[0] + "_diff.png"
                            cv2.imwrite(overlay_path, diff)
                            diff_overlay_paths.append(overlay_path)
                            print(f"[DETECTED] Frame {i:04d} - Object Count: {count}")
                        else:
                            diff_overlay_paths.append(None)
                    else:
                        object_counts.append(0)
                        diff_overlay_paths.append(None)

                    previous_frame = frame

                    elapsed = time.time() - start_time
                    estimated_total = elapsed / (i + 1) * len(frame_paths)
                    remaining = estimated_total - elapsed
                    self.eta_label.config(text=f"Time remaining: {timedelta(seconds=int(remaining))}")

                    self.progress["value"] = i + 1
                    self.status_label.config(text=f"Processing frame {i+1} of {len(frame_paths)}")
                    self.master.update_idletasks()

                except Exception as frame_exc:
                    print(f"[ERROR] Exception during frame {i}: {frame_exc}")
                    object_counts.append(0)
                    diff_overlay_paths.append(None)

            summary = f"[SUMMARY] {sum(c > 0 for c in object_counts)} frames with detections | Total objects detected: {sum(object_counts)}"
            print(summary)
            self.label.config(text="Detection complete.")
            self.status_label.config(text="Done", fg="green")
            self.eta_label.config(text="")

            DetectionVideoPlayer(self.master, frame_paths, object_counts=object_counts, diff_masks=diff_overlay_paths, autoplay=True)

        except Exception:
            traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        try:
            DIFF_THRESHOLD = int(sys.argv[1])
            STACK_FOLDER = sys.argv[2]
            MASK_PATH = sys.argv[3]
        except ValueError:
            print("[ERROR] Invalid arguments. Usage: python Full_Detect_counts.py <diff> <stack_folder> <mask_path>")
            sys.exit(1)
    else:
        print("[ERROR] Missing arguments. Usage: python Full_Detect_counts.py <diff> <stack_folder> <mask_path>")
        sys.exit(1)

    root = tk.Tk()
    gui = DetectionGUI(root)
    root.mainloop()
