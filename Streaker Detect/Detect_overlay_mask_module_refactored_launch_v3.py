# ------------------------------------------------------------------------------
# Script Name:     Detect_overlay_mask_module_refactored_launch_v3.py
#
# Description:     GUI for tuning diff threshold and verifying mask application. 
#                  Launches Full_Detect_counts.py with selected parameters.
# ------------------------------------------------------------------------------


import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, Toplevel, messagebox
from PIL import Image, ImageTk
import traceback
import subprocess
import sys

class DetectionViewer:
    def __init__(self, frame_rate=10):
        print("DetectionViewer initialized.")
        self.root = tk.Tk()
        self.root.withdraw()

        self.stack_folder = None
        self.mask_path = "T:/Dahua_MKV_Streaker/mask.png"
        self.frame_rate = frame_rate

        self.frame_files = []
        self.frames = []
        self.frame_count = 0
        self.frame_index = 0
        self.paused = False

        self.original_height = 0
        self.original_width = 0
        self.external_mask = None

        self.edge_var = tk.BooleanVar(master=self.root, value=False)
        self.diff_var = tk.BooleanVar(master=self.root, value=True)

        self.edge_thresh = tk.IntVar(master=self.root, value=50)
        self.diff_thresh = tk.IntVar(master=self.root, value=25)

        self.canvas = None
        self.viewer_window = None
        self.current_display_image = None
        self.display_image_size = (1296, 972)
        self.aspect_ratio = None

        self.setup_gui()
        self.load_hardcoded_mask()

    def initialize_viewer_window(self):
        self.viewer_window = Toplevel(self.root)
        self.viewer_window.title(f"Detection Viewer - {os.path.basename(__file__)}")
        self.canvas = tk.Canvas(self.viewer_window, width=500, height=500)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        self.viewer_window.bind("<Right>", lambda e: self.step_frame())
        self.viewer_window.bind("<Left>", lambda e: self.step_frame(reverse=True))
        self.viewer_window.bind("<space>", lambda e: self.toggle_play())

    def on_canvas_resize(self, event):
        if self.aspect_ratio:
            canvas_ratio = event.width / event.height
            if canvas_ratio > self.aspect_ratio:
                new_height = event.height
                new_width = int(new_height * self.aspect_ratio)
            else:
                new_width = event.width
                new_height = int(new_width / self.aspect_ratio)
            self.canvas.config(width=new_width, height=new_height)
            self.display_image_size = (new_width, new_height)
        else:
            self.display_image_size = (event.width, event.height)
        self.apply_overlays()

    def setup_gui(self):
        self.control_window = Toplevel(self.root)
        self.control_window.title("Controls")
        self.control_window.geometry("580x180")

        tk.Button(self.control_window, text="Pause", command=self.toggle_play).grid(row=0, column=0, sticky="ew")
        tk.Button(self.control_window, text="Step", command=self.step_frame).grid(row=0, column=1, sticky="ew")
        tk.Button(self.control_window, text="Select Folder", command=self.load_stack_folder).grid(row=0, column=2, sticky="ew")
        tk.Button(self.control_window, text="Change Mask", command=self.select_mask_file).grid(row=0, column=3, sticky="ew")

        tk.Checkbutton(self.control_window, text="Edge Mask", variable=self.edge_var).grid(row=1, column=0, sticky="w")
        tk.Scale(self.control_window, from_=0, to=255, variable=self.edge_thresh, orient=tk.HORIZONTAL).grid(row=1, column=1, columnspan=4, sticky="ew")

        tk.Checkbutton(self.control_window, text="Diff Mask", variable=self.diff_var).grid(row=2, column=0, sticky="w")
        tk.Scale(self.control_window, from_=10, to=220, variable=self.diff_thresh, orient=tk.HORIZONTAL).grid(row=2, column=1, columnspan=4, sticky="ew")

        self.stack_path_label = tk.Label(self.control_window, text="", anchor="w")
        self.stack_path_label.grid(row=3, column=0, columnspan=5, sticky="ew", padx=5)

        self.mask_path_label = tk.Label(self.control_window, text=self.mask_path, fg="blue", cursor="hand2", anchor="w")
        self.mask_path_label.grid(row=4, column=0, columnspan=5, sticky="ew", padx=5)
        self.mask_path_label.bind("<Button-1>", lambda e: self.show_combined_mask_popup())

        tk.Button(self.control_window, text="Run Full Detection", command=self.launch_full_detection).grid(row=5, column=0, columnspan=5, sticky="ew", pady=(10, 5))

    def select_mask_file(self):
        path = filedialog.askopenfilename(title="Select External Mask", filetypes=[["PNG files", "*.png"]])
        if path:
            self.mask_path = path
            self.mask_path_label.config(text=self.mask_path)
            self.load_hardcoded_mask()
            self.apply_overlays()

    def load_hardcoded_mask(self):
        if os.path.isfile(self.mask_path):
            self.external_mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            self.external_mask = None

    def load_stack_folder(self):
        selected_folder = filedialog.askdirectory(title="Select Stack Folder")
        if not selected_folder:
            return
        self.stack_folder = selected_folder

        self.frame_files = sorted([f for f in os.listdir(self.stack_folder) if f.lower().endswith(".png")])[:50]
        self.frames = [cv2.imread(os.path.join(self.stack_folder, f), cv2.IMREAD_GRAYSCALE) for f in self.frame_files]
        if not self.frames:
            messagebox.showerror("Error", "No valid frames found in selected folder.")
            return

        self.initialize_viewer_window()

        self.frame_count = len(self.frames)
        self.original_height, self.original_width = self.frames[0].shape[:2]
        self.stack_path_label.config(text=self.stack_folder)
        self.aspect_ratio = self.original_width / self.original_height
        self.frame_index = 0
        self.apply_overlays()
        self.step_frame()

    def toggle_play(self):
        self.paused = not self.paused
        if not self.paused:
            self.play_loop()

    def play_loop(self):
        if not self.paused:
            self.step_frame()
            self.viewer_window.after(int(1000 / self.frame_rate), self.play_loop)

    def step_frame(self, reverse=False):
        if self.frame_count == 0:
            return
        self.frame_index = (self.frame_index - 1 if reverse else self.frame_index + 1) % self.frame_count
        self.apply_overlays()

    def apply_overlays(self):
        if not self.frames:
            return

        frame = self.frames[self.frame_index].copy()
        overlay_color = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

        if self.external_mask is not None:
            if self.external_mask.shape != frame.shape:
                self.external_mask = cv2.resize(self.external_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.edge_var.get():
            edges = cv2.Canny(frame, self.edge_thresh.get(), self.edge_thresh.get() * 2)
            overlay_color[edges > 0] = [0, 255, 255]

        if self.diff_var.get() and self.frame_count > 1:
            prev_index = (self.frame_index - 1) % self.frame_count
            diff = cv2.absdiff(frame, self.frames[prev_index])
            if self.external_mask is not None:
                diff = cv2.bitwise_and(diff, diff, mask=self.external_mask)
            _, diff_mask = cv2.threshold(diff, self.diff_thresh.get(), 255, cv2.THRESH_BINARY)
            overlay_color[diff_mask > 0] = [255, 0, 0]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        combined = cv2.addWeighted(frame_rgb, 1.0, overlay_color, 0.7, 0)
        frame_resized = cv2.resize(combined, self.display_image_size, interpolation=cv2.INTER_AREA)
        image = Image.fromarray(frame_resized)
        self.current_display_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.current_display_image)

    def show_combined_mask_popup(self):
        if self.external_mask is None:
            return
        mask_resized = cv2.resize(self.external_mask, (300, 300), interpolation=cv2.INTER_NEAREST)
        image = Image.fromarray(mask_resized)
        image = image.convert("L")
        mask_popup = Toplevel(self.root)
        mask_popup.title("Mask Preview")
        canvas = tk.Canvas(mask_popup, width=300, height=300)
        canvas.pack()
        mask_photo = ImageTk.PhotoImage(image=image)
        canvas.create_image(0, 0, anchor="nw", image=mask_photo)
        canvas.image = mask_photo

    def launch_full_detection(self):
        if not self.stack_folder or not self.mask_path:
            print("[ERROR] Stack folder or mask path not set.")
            return
        try:
            subprocess.Popen([
                sys.executable,
                "Full_Detect_counts.py",
                self.stack_folder,
                self.mask_path if self.mask_path else ""
            ])
            print("[DEBUG] Launched Full_Detect_counts.py with settings:",
                  self.stack_folder, self.mask_path)
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    try:
        print("Starting detection viewer...")
        viewer = DetectionViewer()
        print("Entering root mainloop.")
        viewer.root.mainloop()
    except Exception:
        print("Exception occurred:")
        traceback.print_exc()
