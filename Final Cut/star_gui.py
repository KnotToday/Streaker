# ------------------------------------------------------------------------------
# Script Name:     StarMaskGenerator.py
# Description:     Generates a star mask from grayscale PNG frames.
#                  Allows file/folder selection, navigates through frames with
#                  a ±10 buffered cache. Displays thresholded, blurred, and
#                  dilated result with live GUI controls.
# ------------------------------------------------------------------------------

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Toplevel, messagebox
from PIL import Image, ImageTk

# ------------------------------------------------------------------------------
# Class Definition
# ------------------------------------------------------------------------------

class StarMaskGenerator:
    def __init__(self):
        # -- Initialization --
        self.root = tk.Tk()
        self.root.withdraw()

        # -- Frame tracking --
        self.frame_list = []
        self.current_frame_index = 0
        self.frame_cache = {}

        # -- Image and processing --
        self.first_frame = None
        self.mask_image = None

        # -- GUI Controls --
        self.star_thresh = tk.IntVar(value=30)
        self.blur_radius = tk.IntVar(value=1)
        self.dilate_iter = tk.IntVar(value=1)
        self.show_original = tk.BooleanVar(value=False)

        # -- Viewer size and canvas --
        self.aspect_ratio = None
        self.display_image_size = (1296, 972)
        self.viewer_window = None
        self.canvas = None

        # -- GUI Setup --
        self.control_window = None
        self.setup_controls()

    # ------------------------------------------------------------------------------
    # GUI Control Panel Setup
    # ------------------------------------------------------------------------------

    def setup_controls(self):
        self.control_window = Toplevel(self.root)
        self.control_window.title("Star Mask Controls")
        self.control_window.geometry("600x200")

        # -- File/Folder Selection --
        tk.Button(self.control_window, text="Select Folder", command=self.select_stack_folder).grid(row=0, column=0, sticky="ew")
        tk.Button(self.control_window, text="Select File", command=self.select_single_file).grid(row=0, column=1, sticky="ew")

        # -- Threshold Controls --
        tk.Label(self.control_window, text="Threshold").grid(row=1, column=0, sticky="e")
        tk.Scale(self.control_window, from_=5, to=100, resolution=1, variable=self.star_thresh, orient=tk.HORIZONTAL, command=self.update_mask, length=400).grid(row=1, column=1, columnspan=2, sticky="ew")

        tk.Label(self.control_window, text="Blur").grid(row=2, column=0, sticky="e")
        tk.Scale(self.control_window, from_=0, to=10, variable=self.blur_radius, orient=tk.HORIZONTAL, command=self.update_mask).grid(row=2, column=1, sticky="ew")

        tk.Label(self.control_window, text="Dilate").grid(row=3, column=0, sticky="e")
        tk.Scale(self.control_window, from_=0, to=10, variable=self.dilate_iter, orient=tk.HORIZONTAL, command=self.update_mask).grid(row=3, column=1, sticky="ew")

        # -- Show/Hide Original Toggle --
        tk.Checkbutton(self.control_window, text="Show Original Image", variable=self.show_original, command=self.show_mask_on_canvas).grid(row=4, column=0, columnspan=3, sticky="w", padx=5)

        # -- Frame Navigation --
        nav_frame = tk.Frame(self.control_window)
        nav_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=5)
        tk.Button(nav_frame, text="Previous Frame", command=self.previous_frame).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next Frame", command=self.next_frame).pack(side=tk.LEFT, padx=5)

        # -- Save Mask --
        tk.Button(self.control_window, text="Save Mask", command=self.save_mask, bg="green", fg="white").grid(row=7, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        # -- Path Display --
        self.file_path_label = tk.Label(self.control_window, text="", anchor="w")
        self.file_path_label.grid(row=6, column=0, columnspan=3, sticky="ew", padx=5)

    # ------------------------------------------------------------------------------
    # Image Viewer Initialization
    # ------------------------------------------------------------------------------

    def initialize_viewer_window(self):
        if not self.viewer_window:
            self.viewer_window = Toplevel(self.root)
            self.viewer_window.title("Star Mask Viewer")
            self.canvas = tk.Canvas(self.viewer_window, width=1296, height=972)
            self.canvas.pack(fill=tk.BOTH, expand=True)
            self.canvas.bind("<Configure>", self.on_canvas_resize)

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
        self.show_mask_on_canvas()

    # ------------------------------------------------------------------------------
    # File & Folder Selection
    # ------------------------------------------------------------------------------

    def select_stack_folder(self):
        folder = filedialog.askdirectory(title="Select Stack Folder")
        if not folder:
            return
        self.load_frame_list(folder)
        self.current_frame_index = 0
        self.load_current_frame()
        self.initialize_viewer_window()
        self.update_mask()

    def select_single_file(self):
        filepath = filedialog.askopenfilename(title="Select PNG File", filetypes=[("PNG files", "*.png")])
        if not filepath:
            return
        folder = os.path.dirname(filepath)
        self.load_frame_list(folder)
        try:
            self.current_frame_index = self.frame_list.index(os.path.basename(filepath))
        except ValueError:
            messagebox.showerror("Error", "Selected file not found in folder listing.")
            return
        self.load_current_frame()
        self.initialize_viewer_window()
        self.update_mask()

    def load_frame_list(self, folder):
        self.frame_list = sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])
        self.stack_folder = folder
        if not self.frame_list:
            messagebox.showerror("Error", "No PNG files found.")
            return

    # ------------------------------------------------------------------------------
    # Frame Loading and Buffering
    # ------------------------------------------------------------------------------

    def load_frame_by_index(self, index):
        if index < 0 or index >= len(self.frame_list):
            return None
        if index in self.frame_cache:
            return self.frame_cache[index]
        filepath = os.path.join(self.stack_folder, self.frame_list[index])
        frame = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise FileNotFoundError(f"Could not load frame: {filepath}")
        self.frame_cache[index] = frame
        return frame

    def load_current_frame(self):
        # -- Cache ±10 frames around current index --
        self.frame_cache.clear()
        for offset in range(-10, 11):
            idx = self.current_frame_index + offset
            if 0 <= idx < len(self.frame_list):
                self.load_frame_by_index(idx)

        self.first_frame = self.frame_cache[self.current_frame_index]
        self.aspect_ratio = self.first_frame.shape[1] / self.first_frame.shape[0]
        full_path = os.path.join(self.stack_folder, self.frame_list[self.current_frame_index])
        self.file_path_label.config(text=full_path)

    # ------------------------------------------------------------------------------
    # Frame Navigation
    # ------------------------------------------------------------------------------

    def next_frame(self):
        if self.current_frame_index + 1 >= len(self.frame_list):
            return
        self.current_frame_index += 1
        self.load_current_frame()
        self.update_mask()

    def previous_frame(self):
        if self.current_frame_index - 1 < 0:
            return
        self.current_frame_index -= 1
        self.load_current_frame()
        self.update_mask()

    # ------------------------------------------------------------------------------
    # Mask Update and Display
    # ------------------------------------------------------------------------------

    def update_mask(self, *args):
        if self.first_frame is None:
            return

        _, mask = cv2.threshold(self.first_frame, self.star_thresh.get(), 255, cv2.THRESH_BINARY)

        if self.blur_radius.get() > 0:
            mask = cv2.GaussianBlur(mask, (2 * self.blur_radius.get() + 1,) * 2, 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        if self.dilate_iter.get() > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=self.dilate_iter.get())

        self.mask_image = mask
        self.show_mask_on_canvas()

    def save_mask(self):
        if self.mask_image is None:
            tk.messagebox.showerror("Error", "No mask to save.")
            return
        save_path = os.path.join(self.stack_folder, "star_mask.png")
        cv2.imwrite(save_path, self.mask_image)
        self.file_path_label.config(text=f"Saved: {save_path}")

    def show_mask_on_canvas(self):
        if self.first_frame is None or self.canvas is None:
            return
        image_to_show = self.first_frame if self.show_original.get() else self.mask_image
        if image_to_show is None:
            return
        image = Image.fromarray(image_to_show)
        image = image.resize(self.display_image_size, Image.NEAREST)
        image = image.convert("L")
        photo = ImageTk.PhotoImage(image=image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=photo)
        self.canvas.image = photo

# ------------------------------------------------------------------------------
# Launch the Application
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    app = StarMaskGenerator()
    app.root.mainloop()
 