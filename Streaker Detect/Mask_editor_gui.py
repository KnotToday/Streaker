# MaskEditor_FloatingControls_PanToggle.py — Mask editor with zoom, pan (toggle), polygon/rectangle tools

import os
import cv2
import numpy as np
from tkinter import (
    Tk, Toplevel, Label, Button, Canvas, filedialog, messagebox,
    Frame, StringVar
)
from PIL import Image, ImageTk
from screeninfo import get_monitors


class MaskEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("MaskEditor_FloatingControls_PanToggle.py")

        # Canvas only in main window
        self.canvas = Canvas(root, cursor="cross", bg="black")
        self.canvas.pack(fill="both", expand=True)

        # Floating control window
        self.ctrl_win = Toplevel(root)
        self.ctrl_win.title("Controls")
        self.ctrl_win.attributes("-topmost", True)

        controls_frame = Frame(self.ctrl_win)
        controls_frame.pack()

        Button(controls_frame, text="Load Image", command=self.load_image).pack(pady=2)
        Button(controls_frame, text="Load Existing Mask", command=self.load_mask).pack(pady=2)
        Button(controls_frame, text="Clear Mask", command=self.clear_mask).pack(pady=2)
        Button(controls_frame, text="Save Mask", command=self.save_mask).pack(pady=2)
        Button(controls_frame, text="Undo Last Point", command=self.undo_point).pack(pady=2)
        Button(controls_frame, text="Reset Rectangles", command=self.reset_rectangles).pack(pady=2)
        Button(controls_frame, text="Zoom In", command=self.zoom_in).pack(pady=2)
        Button(controls_frame, text="Zoom Out", command=self.zoom_out).pack(pady=2)

        self.mode = StringVar(value="Polygon")
        self.mode_button = Button(controls_frame, text="Switch to Rectangle Mode", command=self.toggle_mode)
        self.mode_button.pack(pady=10)

        self.pan_button = Button(controls_frame, text="Enter Pan Mode", command=self.toggle_pan_mode)
        self.pan_button.pack(pady=10)

        self.status = StringVar(value="Mode: Polygon | Zoom: 1.0x")
        Label(controls_frame, textvariable=self.status).pack(pady=10)

        # Bindings
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.end_drag)
        self.canvas.bind("<Button-3>", self.complete_polygon)

        # Image and mask
        self.image = None
        self.tk_image = None
        self.mask = None
        self.filename = None
        self.points = []
        self.rectangles = []

        # Drawing state
        self.height = 0
        self.width = 0
        self.zoom = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 3.0
        self.offset_x = 0
        self.offset_y = 0
        self.drag_start = None
        self.pan_mode = False

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not path:
            return
        self.filename = path
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Failed to load image.")
            return
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
        self.mask = np.ones((self.height, self.width), dtype=np.uint8) * 255
        self.points.clear()
        self.rectangles.clear()
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.update_canvas()

    def load_mask(self):
        path = filedialog.askopenfilename(filetypes=[("PNG Mask", "*.png")])
        if not path or self.image is None:
            return
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != self.mask.shape:
            messagebox.showerror("Error", "Failed to load mask or resolution mismatch.")
            return
        self.mask = mask.copy()
        self.points.clear()
        self.rectangles.clear()
        self.update_canvas()

    def clear_mask(self):
        if self.mask is not None:
            self.mask[:, :] = 255
            self.points.clear()
            self.rectangles.clear()
            self.update_canvas()

    def save_mask(self):
        if self.mask is None:
            messagebox.showerror("Error", "No mask to save.")
            return
        if self.filename:
            base_folder = os.path.abspath(os.path.join(self.filename, os.pardir, os.pardir))
            out_path = os.path.join(base_folder, "mask.png")
            cv2.imwrite(out_path, self.mask)
            messagebox.showinfo("Saved", f"Mask saved to:\n{out_path}")
        else:
            messagebox.showerror("Error", "No image loaded to determine save location.")

    def undo_point(self):
        if self.points:
            self.points.pop()
            self.update_canvas()
        else:
            messagebox.showinfo("Undo", "No more points to remove.")

    def reset_rectangles(self):
        self.rectangles.clear()
        self.rebuild_mask()

    def rebuild_mask(self):
        self.mask[:, :] = 255
        for rect in self.rectangles:
            cv2.rectangle(self.mask, rect[0], rect[1], 0, thickness=-1)
        self.update_canvas()

    def toggle_mode(self):
        if self.mode.get() == "Polygon":
            self.mode.set("Rectangle")
            self.mode_button.config(text="Switch to Polygon Mode")
        else:
            self.mode.set("Polygon")
            self.mode_button.config(text="Switch to Rectangle Mode")
        self.points.clear()
        self.update_canvas()
        self.update_status()

    def toggle_pan_mode(self):
        self.pan_mode = not self.pan_mode
        if self.pan_mode:
            self.pan_button.config(text="Exit Pan Mode")
        else:
            self.pan_button.config(text="Enter Pan Mode")
        self.update_status()

    def update_status(self):
        current_mode = "Pan" if self.pan_mode else self.mode.get()
        self.status.set(f"Mode: {current_mode} | Zoom: {self.zoom:.1f}x")

    def zoom_in(self):
        if self.zoom < self.max_zoom:
            self.zoom *= 1.25
            self.update_canvas()
            self.update_status()

    def zoom_out(self):
        if self.zoom > self.min_zoom:
            self.zoom /= 1.25
            self.update_canvas()
            self.update_status()

    def on_left_click(self, event):
        if self.image is None:
            return

        if self.pan_mode:
            self.drag_start = (event.x, event.y)
            return

        x = int((event.x - self.offset_x) / self.zoom)
        y = int((event.y - self.offset_y) / self.zoom)
        if not (0 <= x < self.width and 0 <= y < self.height):
            return

        if self.mode.get() == "Polygon":
            self.points.append((x, y))
        elif self.mode.get() == "Rectangle":
            self.points.append((x, y))
            if len(self.points) == 2:
                pt1, pt2 = self.points
                cv2.rectangle(self.mask, pt1, pt2, 0, thickness=-1)
                self.rectangles.append((pt1, pt2))
                self.points.clear()
        self.update_canvas()

    def on_drag(self, event):
        if self.pan_mode and self.drag_start:
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.drag_start = (event.x, event.y)
            self.update_canvas()

    def end_drag(self, event):
        self.drag_start = None

    def complete_polygon(self, event=None):
        if self.mode.get() != "Polygon" or len(self.points) < 3:
            return
        pts = np.array(self.points, dtype=np.int32)
        cv2.fillPoly(self.mask, [pts], 0)
        self.points.clear()
        self.update_canvas()

    def update_canvas(self):
        if self.image is None or self.mask is None:
            return
        overlay = self.image.copy()
        red_mask = np.zeros_like(overlay)
        red_mask[:, :, 0] = 255
        overlay[self.mask == 0] = red_mask[self.mask == 0]

        display_img = overlay.copy()
        if self.mode.get() == "Polygon":
            for i, pt in enumerate(self.points):
                cv2.circle(display_img, pt, 3, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(display_img, self.points[i - 1], pt, (0, 255, 0), 1)
        elif self.mode.get() == "Rectangle" and len(self.points) == 1:
            x1, y1 = self.points[0]
            x2 = int((self.canvas.winfo_pointerx() - self.canvas.winfo_rootx() - self.offset_x) / self.zoom)
            y2 = int((self.canvas.winfo_pointery() - self.canvas.winfo_rooty() - self.offset_y) / self.zoom)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        display_img = cv2.resize(display_img, (0, 0), fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_NEAREST)
        h, w = display_img.shape[:2]
        view = np.zeros((max(h, self.height), max(w, self.width), 3), dtype=np.uint8)
        view[:h, :w] = display_img
        pil_img = Image.fromarray(view)

        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_image)


if __name__ == "__main__":
    root = Tk()
    app = MaskEditor(root)
    root.mainloop()
