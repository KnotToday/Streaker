import os  # file and directory operations
import time  # timing for delays
import cv2  # OpenCV for image handling
import numpy as np  # numerical operations
import json  # for method config
from tkinter import (
    Tk,
    Label,
    Button,
    filedialog,
    Text,
    Scrollbar,
    END,
    StringVar,
    IntVar,
    OptionMenu,
    Checkbutton,
    Canvas,
    NW,
    Toplevel,
    Scale,
    HORIZONTAL,
)  # GUI components
from tkinter import ttk  # ttk for better Combobox dropdown
from datetime import datetime  # for date labels
from PIL import Image, ImageTk  # for preview rendering
from threading import Thread, Event  # to manage background tasks
import subprocess  # for triggering detection process
import shutil  # for faster file copying


class Stack_Builder_Tester_GUI:  # main GUI class for generating stack sweeps and detection clips
    def __init__(self, root):
        self.root = root
        self.root.title("Detection Test Builder")
        self.stack_group_size = IntVar(value=20)  # default number of frames per stack
        self.frames_folder = StringVar(value="")  # folder containing input frame PNGs
        self.start_filename = StringVar(value="")  # filename to start processing from
        self.stop_event = Event()  # event flag to stop background thread gracefully

        Label(root, text="Select Frames Folder:").pack(
            pady=5
        )  # prompt to choose input folder
        Button(root, text="Choose Folder", command=self.select_folder).pack()

        self.status_label = Label(
            root, text="No folder selected."
        )  # dynamic label for folder path
        self.status_label.pack(pady=5)

        self.start_frame_label = Label(
            root, text="Start from Frame Filename:"
        )  # prompt to select start frame
        self.start_frame_label.pack()
        self.filename_combobox = ttk.Combobox(  # dropdown for available .png filenames
            root, textvariable=self.start_filename, state="readonly", height=20
        )
        self.filename_combobox.pack()

        Label(root, text="Frames per Stack:").pack()  # slider label
        Scale(  # user sets how many frames are used per stack
            root, from_=2, to=40, orient=HORIZONTAL, variable=self.stack_group_size
        ).pack()

        Label(root, text="Select Stacking Methods:").pack()  # label for checkboxes
        self.method_vars = {
            "max": IntVar(value=1),
            "mean": IntVar(value=0),
            "median": IntVar(value=0),
            "min": IntVar(value=0),
        }
        for method, var in self.method_vars.items():
            Checkbutton(root, text=method, variable=var).pack(
                anchor="w"
            )  # user can select stacking method(s)

        Button(
            root, text="Generate Sweep Samples", command=self.start_stack_thread
        ).pack(pady=10)
        Button(
            root,
            text="Create Detection Clip (1 min)",
            command=self.generate_detection_clip,
        ).pack(pady=2)
        Button(root, text="Force Quit", command=self.force_quit).pack(pady=2)

        self.log = Text(root, height=15, width=80)  # scrolling log window for feedback
        self.log.pack(side="left", padx=(10, 0), pady=5)

        scrollbar = Scrollbar(
            root, command=self.log.yview
        )  # vertical scrollbar for log
        scrollbar.pack(side="right", fill="y")
        self.log.config(yscrollcommand=scrollbar.set)

    def force_quit(self):
        self.stop_event.set()
        self.log_message("[FORCE] User triggered quit.")

    def start_stack_thread(self):
        self.stop_event.clear()  # reset stop flag before background task
        Thread(target=self.generate_stack_sweep, daemon=True).start()

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Folder Containing Frame PNGs")
        if folder:
            self.frames_folder.set(folder)
            self.status_label.config(text=f"Selected: {folder}")
            self.update_filename_combobox()

    def update_filename_combobox(self):
        folder = self.frames_folder.get()  # get selected input folder
        pngs = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
        if pngs:
            self.filename_combobox["values"] = pngs
            self.start_filename.set(pngs[0])

    def generate_stack_sweep(
        self,
    ):  # builds stacked images using sweep of frame counts and methods
        folder = self.frames_folder.get()
        if not folder:
            self.log_message("[ERROR] No folder selected.")
            return

        base_folder = os.path.dirname(
            folder
        )  # base directory of selected frames  # parent directory of frame folder
        stem = os.path.basename(
            folder
        )  # folder name used as identifier  # folder name to use as base label
        frame_files = sorted(
            [f for f in os.listdir(folder) if f.endswith(".png")]
        )  # all .png frame names sorted

        if not frame_files:
            self.log_message("[ERROR] No PNG files found in folder.")
            return

        try:
            start_index = frame_files.index(
                self.start_filename.get()
            )  # find index to begin stack from
        except ValueError:
            self.log_message("[ERROR] Starting filename not found in folder.")
            return

        full_paths = [
            os.path.join(folder, f) for f in frame_files
        ]  # full paths to all .png files

        selected_methods = [  # get stacking methods selected by user
            method for method, var in self.method_vars.items() if var.get() == 1
        ]

        self.log_message(f"[INFO] Generating sweep samples from: {folder}")
        self.log_message(f"[INFO] Methods: {', '.join(selected_methods)}")

        sample_folder = os.path.join(
            base_folder, f"Stack_Sweep_Samples_{stem}"
        )  # output folder for sweep results
        os.makedirs(sample_folder, exist_ok=True)

        try:
            for size in range(2, 41):  # loop over group sizes from 2 to 40
                if self.stop_event.is_set():
                    self.log_message("[STOPPED] Sweep aborted.")
                    return
                group = full_paths[
                    start_index : start_index + size
                ]  # select N-frame group
                if len(group) < size:
                    self.log_message(
                        f"[SKIP] Not enough frames for group size {size} from index {start_index}."
                    )
                    continue
                imgs = [
                    cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in group
                ]  # load each image into memory  # load frames in grayscale
                if any(img is None for img in imgs):
                    self.log_message(
                        f"[ERROR] Failed to load one or more images for size {size}."
                    )
                    continue

                for method in selected_methods:
                    if method == "mean":
                        out = np.mean(imgs, axis=0).astype(np.uint8)  # mean stack
                    elif method == "max":
                        out = np.max(imgs, axis=0).astype(
                            np.uint8
                        )  # max stack across image group  # max stack
                    elif method == "median":
                        out = np.median(imgs, axis=0).astype(np.uint8)  # median stack
                    elif method == "min":
                        out = np.min(imgs, axis=0).astype(np.uint8)  # min stack
                    else:
                        continue
                    out_path = os.path.join(sample_folder, f"{method}_{size}.png")
                    cv2.imwrite(out_path, out)

        except Exception as e:
            self.log_message(f"[ERROR] Failed sweep: {e}")

        self.log_message(f"[DONE] Saved sweep samples to: {sample_folder}")

    def generate_detection_clip(
        self,
    ):  # creates a detection test clip from selected frames
        folder = self.frames_folder.get()
        if not folder:
            self.log_message("[ERROR] No frames folder selected.")
            return

        start_name = self.start_filename.get()  # get starting filename
        try:
            start_number = int(
                os.path.splitext(start_name)[0]
            )  # extract numeric start frame from filename
        except ValueError:
            self.log_message("[ERROR] Invalid start frame name format.")
            return

        base_folder = os.path.dirname(folder)
        stem = os.path.basename(folder)

        sweep_folder = os.path.join(base_folder, f"Stack_Sweep_Samples_{stem}")
        os.makedirs(sweep_folder, exist_ok=True)

        timestamp = datetime.now().strftime(
            "%m-%d-%Y_%H-%M-%S"
        )  # unique timestamp for output folder
        output_dir = os.path.join(
            sweep_folder, f"Detection_Clip_{stem}_start_{start_number}_{timestamp}"
        )
        frames_output = os.path.join(
            output_dir, "Frames"
        )  # subfolder to hold copied frames
        stack_output = os.path.join(
            output_dir, "Stacked"
        )  # subfolder to hold stacked images
        os.makedirs(frames_output, exist_ok=True)
        os.makedirs(stack_output, exist_ok=True)

        self.log_message("[CLIP] Copying frames and generating stack...")

        frame_paths = []
        for i in range(
            start_number, start_number + 1200
        ):  # select 1-minute worth of frames
            src = os.path.join(folder, f"{i:05d}.png")
            dst = os.path.join(frames_output, f"{i:05d}.png")
            if os.path.exists(src):
                frame_paths.append(dst)
                shutil.copy2(src, dst)  # fast file copy instead of decoding/encoding

        stack_method = "max"
        stack_group_size = (
            self.stack_group_size.get()
        )  # get frames-per-stack from slider
        stacked_frames = []
        for i in range(
            0, len(frame_paths), stack_group_size
        ):  # process each stack group
            group = frame_paths[i : i + stack_group_size]
            if len(group) < stack_group_size:
                continue
            imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in group]
            if any(img is None for img in imgs):
                continue
            if stack_method == "max":
                out = np.max(imgs, axis=0).astype(np.uint8)
            else:
                continue
            cv2.imwrite(
                os.path.join(stack_output, f"{i//stack_group_size+1:04d}.png"),
                out,
            )

        metadata = {
            "start_frame": start_number,
            "stack_method": stack_method,
            "stack_group_size": stack_group_size,
            "frame_count": len(frame_paths),
        }
        with open(
            os.path.join(output_dir, "clip_info.txt"), "w"
        ) as f:  # save metadata about this run
            json.dump(metadata, f, indent=2)

        self.log_message("[CLIP] Creating video clip...")

        cmd = [  # ffmpeg command to encode copied frames into a .mp4
            "ffmpeg",
            "-y",
            "-framerate",
            "20",
            "-start_number",
            str(start_number),
            "-i",
            os.path.join(frames_output, "%05d.png"),
            "-vframes",
            str(1200),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            os.path.join(output_dir, f"detection_clip_start_{start_number}.mp4"),
        ]
        try:
            subprocess.run(cmd, check=True)  # run ffmpeg to make clip
            self.log_message("[CLIP] Detection clip saved. Launching detection test...")
            try:
                subprocess.Popen(
                    ["python", "detect_sample_viewer.py", output_dir]
                )  # launch viewer GUI
            except FileNotFoundError:
                self.log_message(
                    "[ERROR] detect_sample_viewer.py not found. Skipping launch."
                )
        except Exception as e:
            self.log_message(f"[ERROR] Failed to generate detection clip: {e}")

    def log_message(
        self, message
    ):  # helper to append messages to GUI log with timestamp
        timestamp = time.strftime("[%H:%M:%S]")
        self.log.insert(END, f"{timestamp} {message}\n")
        self.log.see(END)


if __name__ == "__main__":  # launch the app
    root = Tk()
    app = Stack_Builder_Tester_GUI(root)
    root.mainloop()
