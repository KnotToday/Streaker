# flake8: noqa
# pylint: skip-file

import os
import time
import cv2
import subprocess
import numpy as np
from tkinter import Tk, Label, Button, filedialog, Text, Scrollbar, END, StringVar
from threading import Thread, Event
from datetime import datetime


class DetectStreaksGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Streak Detector Monitor")
        self.monitoring = False
        self.stop_event = Event()
        self.base_dir = StringVar(value="")

        Label(root, text="Top-Level Recordings Folder:").pack(pady=5)
        Button(root, text="Select Folder", command=self.select_folder).pack()

        self.status_label = Label(root, text="No folder selected.")
        self.status_label.pack(pady=5)

        self.toggle_button = Button(
            root, text="Start Watching", command=self.toggle_monitoring
        )
        self.toggle_button.pack(pady=10)

        self.log = Text(root, height=15, width=80)
        self.log.pack(side="left", padx=(10, 0), pady=5)

        scrollbar = Scrollbar(root, command=self.log.yview)
        scrollbar.pack(side="right", fill="y")
        self.log.config(yscrollcommand=scrollbar.set)

    def wait_until_stable(self, folder_path):
        self.log_message(f"[WAIT] Checking stability for: {os.path.basename(folder_path)}")
        last_size = -1
        stable_count = 0
        while stable_count < 3 and not self.stop_event.is_set():
            try:
                current_size = sum(os.path.getsize(os.path.join(folder_path, f)) for f in os.listdir(folder_path))
                if current_size == last_size:
                    stable_count += 1
                else:
                    stable_count = 0
                last_size = current_size
            except Exception as e:  # TODO: Use specific exception
                self.log_message(f"[ERROR] Could not stat folder: {e}")
                return
            time.sleep(1)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.base_dir.set(folder)
            self.status_label.config(text=f"Monitoring latest folder in: {folder}")

    def toggle_monitoring(self):
        if not self.monitoring:
            self.start_monitoring()
        else:
            self.stop_event.set()
            self.toggle_button.config(text="Start Watching")
            self.log_message("Stopped monitoring.")
            self.monitoring = False

    def start_monitoring(self):
        if not self.base_dir.get():
            self.log_message("No folder selected.")
            return
        self.stop_event.clear()
        self.monitoring = True
        self.toggle_button.config(text="Stop Watching")
        self.log_message("Started monitoring for new stacked folders...")
        Thread(target=self.monitor_loop, daemon=True).start()

    def get_latest_dated_subfolder(self):
        try:
            self.log_message(f"[DEBUG] Looking for date folders in: {self.base_dir.get()}")
            self.log_message(f"[DEBUG] Found: {os.listdir(self.base_dir.get())}")
            dated_folders = [d for d in os.listdir(self.base_dir.get()) if os.path.isdir(os.path.join(self.base_dir.get(), d)) and self.is_valid_date_folder(d)]
            if not dated_folders:
                self.log_message("[DEBUG] No valid date folders found.")
                return None
            latest = max(dated_folders)
            self.log_message(f"[DEBUG] Latest dated folder: {latest}")
            return latest
        except Exception as e:  # TODO: Use specific exception
            self.log_message(f"[ERROR] Failed to list subfolders: {e}")
            return None

    @staticmethod
    def is_valid_date_folder(folder_name):
        try:
            datetime.strptime(folder_name, "%m-%d-%Y")
            return True
        except ValueError:
            return False

    def monitor_loop(self):
        seen = set()
        while not self.stop_event.is_set():
            latest_folder = self.get_latest_dated_subfolder()
            if not latest_folder:
                time.sleep(5)
                continue
            full_path = os.path.join(self.base_dir.get(), latest_folder)
            self.log_message(f"[DEBUG] Scanning: {full_path}")
            self.log_message(f"[DEBUG] Contents: {os.listdir(full_path)}")
            for fname in os.listdir(full_path):
                if fname.startswith("Stack_") and fname not in seen:
                    self.log_message(f"[DEBUG] Checking folder: {fname}")
                    seen.add(fname)
                    stack_path = os.path.join(full_path, fname)
                    frames_path = os.path.join(full_path, "Frames_" + fname[6:])
                    if os.path.exists(frames_path):
                        self.log_message(f"[NEW] Detected: {fname}")
                        self.wait_until_stable(stack_path)
                        self.detect_streaks(stack_path, frames_path)
            time.sleep(5)

    def detect_streaks(self, stack_folder, frames_folder):
        top_folder = os.path.abspath(os.path.join(stack_folder, os.pardir))
        mask_path = os.path.join(top_folder, "mask.png")
        mask = None
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                self.log_message("[MASK] Loaded mask from top folder")
            else:
                self.log_message("[MASK] Failed to load mask")

        stack_files = sorted([os.path.join(stack_folder, f) for f in os.listdir(stack_folder) if f.endswith(".png")])
        if len(stack_files) < 2:
            self.log_message(f"[SKIP] Not enough stacks in {stack_folder}")
            return

        detection_folder = os.path.join(os.path.dirname(stack_folder), "Detections", "Clips")
        os.makedirs(detection_folder, exist_ok=True)
        pattern = os.path.join(frames_folder, "%05d.png")

        for i in range(1, len(stack_files)):
            prev = cv2.imread(stack_files[i - 1], cv2.IMREAD_GRAYSCALE)
            curr = cv2.imread(stack_files[i], cv2.IMREAD_GRAYSCALE)
            if prev is None or curr is None:
                continue
            if mask is not None and prev.shape == mask.shape:
                prev = cv2.bitwise_and(prev, prev, mask=mask)
                curr = cv2.bitwise_and(curr, curr, mask=mask)

            diff = cv2.absdiff(curr, prev)
            blur = cv2.GaussianBlur(diff, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=5)
            self.log_message(f"[DEBUG] Diff max: {np.max(diff)}")
            self.log_message(f"[DEBUG] Lines found: {len(lines) if lines is not None else 0}")

            if lines is not None:
                start = (i - 1) * 20 + 1
                end = i * 20
                mp4_out = os.path.join(detection_folder, f"clip_{start:05d}_{end:05d}.mp4")
                gif_out = os.path.join(detection_folder, f"clip_{start:05d}_{end:05d}.gif")

                self.log_message(f"[STREAK] Detected in frames {start}-{end}")

                subprocess.run(["ffmpeg", "-y", "-framerate", "20", "-start_number", str(start), "-i", pattern, "-vframes", str(end - start + 1), "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4_out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                subprocess.run(["ffmpeg", "-y", "-framerate", "20", "-start_number", str(start), "-i", pattern, "-vframes", str(end - start + 1), gif_out], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                self.log_message(f"[SAVED] {os.path.basename(gif_out)}")

    def log_message(self, message):
        timestamp = time.strftime("[%H:%M:%S]")
        self.log.insert(END, f"{timestamp} {message}\n")
        self.log.see(END)


if __name__ == "__main__":
    root = Tk()
    app = DetectStreaksGUI(root)
    root.mainloop()
