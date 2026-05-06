import os
import sys
import time
import cv2
import numpy as np
import subprocess
from tkinter import Tk, Label, Button, filedialog, Text, Scrollbar, END, StringVar, messagebox, ttk, BooleanVar, Checkbutton
from threading import Thread, Event
from datetime import datetime

class StackMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Frame Stacker Monitor")
        self.monitoring = False
        self.stop_event = Event()
        self.base_dir = StringVar(value="")
        self.selected_folders = []
        self.current_folder_index = 0
        self.auto_launch = BooleanVar(value=False)
        self.folder_stacked_any = False

        Label(root, text="Top-Level or Dated Recordings Folder:").pack(pady=5)
        Button(root, text="Select Top-Level Folder", command=self.select_top_folder).pack()
        Button(root, text="Select Multiple Folders", command=self.select_multiple_folders).pack(pady=(0, 5))

        self.status_label = Label(root, text="No folder(s) selected.")
        self.status_label.pack(pady=5)

        Checkbutton(root, text="Auto-launch detection tool after stacking", variable=self.auto_launch).pack(pady=(0, 5))

        self.toggle_button = Button(root, text="Start Watching", command=self.toggle_monitoring)
        self.toggle_button.pack(pady=10)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=(0, 10))

        self.log = Text(root, height=15, width=80)
        self.log.pack(side="left", padx=(10, 0), pady=5)

        scrollbar = Scrollbar(root, command=self.log.yview)
        scrollbar.pack(side="right", fill="y")
        self.log.config(yscrollcommand=scrollbar.set)

    def select_top_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            if self.is_valid_date_folder(os.path.basename(folder)):
                self.selected_folders = [os.path.basename(folder)]
                self.base_dir.set(os.path.dirname(folder))
                self.status_label.config(text=f"Monitoring dated folder: {folder}")
            else:
                self.selected_folders = self.get_all_dated_subfolders(folder)
                self.base_dir.set(folder)
                if not self.selected_folders:
                    self.log_message("No valid dated subfolders found.")
                    return
                self.status_label.config(text=f"Monitoring all folders in: {folder}")
            self.current_folder_index = 0

    def select_multiple_folders(self):
        folder = filedialog.askdirectory(title="Select a folder containing multiple dated folders")
        if folder:
            parent = os.path.dirname(folder)
            base = os.path.basename(folder)
            if self.is_valid_date_folder(base):
                self.selected_folders = [base]
                self.base_dir.set(parent)
                self.status_label.config(text=f"Selected dated folder: {base}")
            else:
                all_folders = self.get_all_dated_subfolders(folder)
                self.selected_folders = all_folders
                self.base_dir.set(folder)
                self.status_label.config(text=f"Selected multiple folders in: {folder}")
            self.current_folder_index = 0

    def toggle_monitoring(self):
        if not self.monitoring:
            self.start_monitoring()
        else:
            self.stop_event.set()
            self.toggle_button.config(text="Start Watching")
            self.log_message("Stopped monitoring.")
            self.monitoring = False

    def start_monitoring(self):
        if not self.selected_folders:
            self.log_message("No valid folders selected.")
            return
        self.stop_event.clear()
        self.monitoring = True
        self.toggle_button.config(text="Stop Watching")
        self.log_message("Started monitoring folders for Frame_ folders...")
        Thread(target=self.monitor_loop, daemon=True).start()

    def get_all_dated_subfolders(self, base_path):
        folders = [f for f in os.listdir(base_path) if self.is_valid_date_folder(f)]
        return sorted(folders, key=lambda d: datetime.strptime(d, "%m-%d-%Y"))

    @staticmethod
    def is_valid_date_folder(folder_name):
        try:
            datetime.strptime(folder_name, "%m-%d-%Y")
            return True
        except ValueError:
            return False

    def monitor_loop(self):
        while not self.stop_event.is_set() and self.current_folder_index < len(self.selected_folders):
            folder = self.selected_folders[self.current_folder_index]
            full_path = os.path.join(self.base_dir.get(), folder)
            seen = set()
            self.folder_stacked_any = False

            self.log_message(f"[INFO] Processing folder: {folder}")
            frames_folders = [f for f in os.listdir(full_path) if f.startswith("Frames_")]
            unprocessed = [f for f in frames_folders if not os.path.exists(os.path.join(full_path, f.replace("Frames_", "Stack_")))]

            if not unprocessed:
                self.log_message(f"[INFO] Skipping {folder}, all Frames folders already stacked.")
                self.current_folder_index += 1
                continue

            self.progress["maximum"] = len(unprocessed)
            self.progress["value"] = 0

            for index, fname in enumerate(unprocessed):
                if self.stop_event.is_set():
                    return
                frames_path = os.path.join(full_path, fname)
                self.log_message(f"[NEW] Detected: {fname}")
                if self.stack_frames(frames_path):
                    self.folder_stacked_any = True
                self.progress["value"] = index + 1
                self.root.update_idletasks()

            if self.folder_stacked_any and self.auto_launch.get():
                try:
                    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Full_Detect_counts.py")
                    subprocess.Popen([sys.executable, script_path])
                    self.log_message(f"[INFO] Launched detection tool.")
                except Exception as e:
                    self.log_message(f"[ERROR] Could not launch detection tool: {e}")

            if not self.stop_event.is_set():
                self.ask_to_continue()

        self.log_message("[DONE] Finished all folders.")
        self.toggle_button.config(text="Start Watching")
        self.monitoring = False

    def ask_to_continue(self):
        if self.current_folder_index + 1 < len(self.selected_folders):
            next_folder = self.selected_folders[self.current_folder_index + 1]
            answer = messagebox.askyesno("Continue?", f"Continue to next folder: {next_folder}?")
            if answer:
                self.current_folder_index += 1
            else:
                self.stop_event.set()
        else:
            self.log_message("[INFO] No more folders to process.")

    def stack_frames(self, frames_folder):
        try:
            frame_files = sorted(
                [os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith(".png")]
            )
            if len(frame_files) < 20:
                self.log_message(f"[SKIP] Not enough frames in {frames_folder}")
                return False

            stack_folder = os.path.join(
                os.path.dirname(frames_folder),
                f"Stack_{os.path.basename(frames_folder)[7:]}"
            )
            if os.path.exists(stack_folder):
                self.log_message(f"[SKIP] Already stacked: {stack_folder}")
                return False

            os.makedirs(stack_folder, exist_ok=True)
            self.log_message(f"[STACK] {os.path.basename(frames_folder)} → {os.path.basename(stack_folder)}")

            for i in range(0, len(frame_files), 20):
                group = frame_files[i:i + 20]
                if len(group) < 20:
                    continue
                imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in group]
                avg = np.mean(imgs, axis=0).astype(np.uint8)
                out_file = os.path.join(stack_folder, f"{i//20 + 1:05d}.png")
                cv2.imwrite(out_file, avg)

            self.log_message(f"[DONE] Stacked to {stack_folder}")
            return True

        except Exception as e:
            self.log_message(f"[ERROR] Exception during stacking: {e}")
            return False

    def log_message(self, message):
        timestamp = time.strftime("[%H:%M:%S]")
        self.log.insert(END, f"{timestamp} {message}\n")
        self.log.see(END)

if __name__ == "__main__":
    try:
        root = Tk()
        app = StackMonitorGUI(root)
        root.mainloop()
    except Exception as e:
        print("[FATAL ERROR]", e)
        import traceback
        traceback.print_exc()
