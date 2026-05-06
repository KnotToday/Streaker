import os
import time
import subprocess
from tkinter import Tk, Label, Button, filedialog, Text, Scrollbar, END, StringVar
from threading import Thread, Event
from datetime import datetime


class ExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MKV Frame Extractor")
        self.monitoring = False
        self.stop_event = Event()
        self.recordings_dir = StringVar(value="")

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

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.recordings_dir.set(folder)
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
        if not self.recordings_dir.get():
            self.log_message("No folder selected.")
            return
        self.stop_event.clear()
        self.monitoring = True
        self.toggle_button.config(text="Stop Watching")
        self.log_message("Started monitoring for new MKV files...")
        Thread(target=self.monitor_loop, daemon=True).start()

    def get_latest_dated_subfolder(self):
        try:
            self.log_message(
                f"[DEBUG] Looking for date folders in: {self.recordings_dir.get()}"
            )
            self.log_message(f"[DEBUG] Found: {os.listdir(self.recordings_dir.get())}")

            dated_folders = [
                d
                for d in os.listdir(self.recordings_dir.get())
                if os.path.isdir(os.path.join(self.recordings_dir.get(), d))
                and self.is_valid_date_folder(d)
            ]
            if not dated_folders:
                return None
            latest = max(dated_folders)
            self.log_message(f"[DEBUG] Latest dated folder: {latest}")
            return latest

        except Exception as e:
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

            full_path = os.path.join(self.recordings_dir.get(), latest_folder)
            for fname in os.listdir(full_path):
                self.log_message(f"[ERROR]" "Folder does not exist: {full_path}")
                time.sleep(5)
                continue
                self.log_message(f"[DEBUG] Checking Seen: {seen}")
                if fname.endswith(".mkv") and fname not in seen:
                    self.log_message(f"[DEBUG] Checking file: {fname}")
                    mkv_path = os.path.join(full_path, fname)
                    self.log_message(f"[NEW] Detected: {fname}")
                    if self.wait_until_stable(mkv_path):
                        seen.add(fname)
                        self.extract_frames(mkv_path)

                    self.log_message(f"[NEW] Detected: {fname}")
                    self.wait_until_stable(mkv_path)
                    self.extract_frames(mkv_path)
            self.log_message(f"[DEBUG] Scanning: {full_path}")
            self.log_message(f"[DEBUG] Contents: {os.listdir(full_path)}")
            time.sleep(5)

    def wait_until_stable(self, filepath):
        self.log_message(f"[WAIT] Checking stability for: {os.path.basename(filepath)}")
        last_size = -1
        stable_count = 0
        while stable_count < 3 and not self.stop_event.is_set():
            try:
                current_size = os.path.getsize(filepath)
                if current_size == last_size:
                    stable_count += 1
                else:
                    stable_count = 0
                last_size = current_size
            except Exception as e:
                self.log_message(f"[ERROR] Could not stat file: {e}")
                return False
            time.sleep(1)
            return True
    def extract_frames(self, mkv_path):
        try:
            self.log_message(f"[DEBUG] Starting extract_frames for {mkv_path}")
            base_dir = os.path.dirname(mkv_path)
            video_stem = os.path.splitext(os.path.basename(mkv_path))[0]
            output_folder = os.path.join(base_dir, f"Frames_{video_stem}")
            self.log_message(f"[DEBUG] Output folder will be: {output_folder}")

            if os.path.exists(output_folder):
                self.log_message(f"[SKIP] Already processed: {output_folder}")
                return

            os.makedirs(output_folder, exist_ok=True)
            output_pattern = os.path.join(output_folder, "%05d.png")
            self.log_message(f"[EXTRACT] {video_stem} → {os.path.basename(output_folder)}")
            self.log_message(f"[DEBUG] Running ffmpeg with: -i \"{mkv_path}\" -vf fps=20 \"{output_pattern}\"")

            result = subprocess.run(
                ["ffmpeg", "-i", mkv_path, "-vf", "fps=20", output_pattern],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if result.returncode == 0:
                self.log_message(f"[DONE] Extracted to {output_folder}")
            else:
                self.log_message(f"[ERROR] FFmpeg failed with return code {result.returncode}")
                self.log_message(result.stderr.decode())

        except Exception as e:
            self.log_message(f"[ERROR] Exception in extract_frames: {e}")


    def log_message(self, message):
        timestamp = time.strftime("[%H:%M:%S]")
        self.log.insert(END, f"{timestamp} {message}\n")
        self.log.see(END)


if __name__ == "__main__":
    root = Tk()
    app = ExtractorGUI(root)
    root.mainloop()
