import os
import sys
import random
import cv2
import numpy as np
from tkinter import (
    Tk, Toplevel, Label, Button, Canvas, filedialog, messagebox,
    Frame, StringVar
)
from PIL import Image, ImageTk


class MaskEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Mask Editor")

        self.canvas = Canvas(root, cursor="cross", bg="black")
        self.canvas.pack(fill="both", expand=True)

        self.ctrl_win = Toplevel(root)
        self.ctrl_win.title("Controls")
        self.ctrl_win.attributes("-topmost", True)
        self.ctrl_win.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        frm = Frame(self.ctrl_win)
        frm.pack()

        Button(frm, text="Grab Frame from MKV",   command=self.grab_mkv_frame).pack(pady=2)
        Button(frm, text="Load Image",            command=self.load_image).pack(pady=2)
        Button(frm, text="Load Existing Mask",    command=self.load_mask).pack(pady=2)
        Button(frm, text="Clear Mask",            command=self.clear_mask).pack(pady=2)
        Button(frm, text="Save Mask",             command=self.save_mask).pack(pady=2)
        Button(frm, text="Undo",                  command=self.undo).pack(pady=2)
        Button(frm, text="Zoom In",               command=self.zoom_in).pack(pady=2)
        Button(frm, text="Zoom Out",              command=self.zoom_out).pack(pady=2)
        Button(frm, text="Fit to Window",         command=self.fit_to_window).pack(pady=2)

        self.mode = StringVar(value="Polygon")
        self.mode_button = Button(frm, text="Switch to Rectangle Mode", command=self.toggle_mode)
        self.mode_button.pack(pady=10)

        self.pan_button = Button(frm, text="Enter Pan Mode", command=self.toggle_pan_mode)
        self.pan_button.pack(pady=10)

        self.status = StringVar(value="Mode: Polygon | Zoom: 1.00x")
        Label(frm, textvariable=self.status).pack(pady=10)

        self.canvas.bind("<Button-1>",       self.on_left_click)
        self.canvas.bind("<B1-Motion>",      self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.end_drag)
        self.canvas.bind("<Button-3>",       self.complete_polygon)
        self.canvas.bind("<Motion>",         self.on_motion)
        self.canvas.bind("<MouseWheel>",     self.on_mousewheel)
        self.canvas.bind("<Configure>",      self._on_configure)

        self.image      = None
        self.tk_image   = None
        self.mask       = None
        self.filename   = None
        self.points     = []
        self.undo_stack = []

        self.height    = 0
        self.width     = 0
        self.zoom      = 1.0
        self.min_zoom  = 0.05
        self.max_zoom  = 10.0
        self.offset_x  = 0
        self.offset_y  = 0
        self.drag_start = None
        self.pan_mode  = False
        self.fit_mode  = True

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _on_close(self):
        for w in (self.ctrl_win, self.root):
            try:
                w.destroy()
            except Exception:
                pass

    # ── undo stack ───────────────────────────────────────────────────────────

    def _push_undo(self):
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)

    # ── image loading ────────────────────────────────────────────────────────

    def _fit_to_half_screen(self):
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.zoom = min((sw // 2) / self.width, (sh // 2) / self.height)
        self.root.geometry(f"{int(self.width * self.zoom)}x{int(self.height * self.zoom)}")
        self.root.update_idletasks()  # let canvas adopt new geometry before first render
        self.offset_x = 0
        self.offset_y = 0

    def load_image_path(self, path):
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Failed to load image:\n{path}")
            return
        self.filename = path
        self.image    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
        self.mask = np.full((self.height, self.width), 255, dtype=np.uint8)
        self.points.clear()
        self.undo_stack.clear()
        self.fit_mode = True
        self._fit_to_half_screen()
        self.update_canvas()

    def grab_mkv_frame(self):
        path = filedialog.askopenfilename(
            title="Select MKV File",
            filetypes=[("MKV files", "*.mkv"), ("All video files", "*.mkv;*.mp4;*.avi")])
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open:\n{path}")
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # pick a random frame from the middle 80% to avoid black edges
        lo = int(total * 0.1)
        hi = max(lo + 1, int(total * 0.9))
        frame_num = random.randint(lo, hi - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, raw = cap.read()
        cap.release()
        if not ret or raw is None:
            messagebox.showerror("Error", f"Could not read frame {frame_num}.")
            return

        self.filename = path
        self.image    = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
        self.mask = np.full((self.height, self.width), 255, dtype=np.uint8)
        self.points.clear()
        self.undo_stack.clear()
        self.fit_mode = True
        self._fit_to_half_screen()
        self.update_canvas()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if path:
            self.load_image_path(path)

    def load_mask(self):
        path = filedialog.askopenfilename(filetypes=[("PNG Mask", "*.png")])
        if not path or self.image is None:
            return
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != self.mask.shape:
            messagebox.showerror("Error", "Failed to load mask or resolution mismatch.")
            return
        self._push_undo()
        self.mask = mask.copy()
        self.points.clear()
        self.update_canvas()

    def clear_mask(self):
        if self.mask is not None:
            self._push_undo()
            self.mask[:] = 255
            self.points.clear()
            self.update_canvas()

    def save_mask(self):
        if self.mask is None:
            messagebox.showerror("Error", "No mask to save.")
            return
        out_path = os.path.join(os.path.dirname(self.filename), "mask.png")
        cv2.imwrite(out_path, self.mask)
        messagebox.showinfo("Saved", f"Mask saved to:\n{out_path}")

    # ── undo ─────────────────────────────────────────────────────────────────

    def undo(self):
        if self.points:
            self.points.pop()
            self.update_canvas()
        elif self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.update_canvas()
        else:
            messagebox.showinfo("Undo", "Nothing to undo.")

    # ── mode / zoom ──────────────────────────────────────────────────────────

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
        self.pan_button.config(text="Exit Pan Mode" if self.pan_mode else "Enter Pan Mode")
        self.update_status()

    def _on_configure(self, event):
        if self.fit_mode:
            self._recalculate_fit_zoom()
        self.update_canvas()

    def _recalculate_fit_zoom(self):
        if self.image is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self.zoom = min(cw / self.width, ch / self.height)
        self.offset_x = 0
        self.offset_y = 0

    def fit_to_window(self):
        self.fit_mode = True
        self._recalculate_fit_zoom()
        self.update_canvas()
        self.update_status()

    def update_status(self):
        mode = "Pan" if self.pan_mode else self.mode.get()
        fit = " | Fit" if self.fit_mode else ""
        self.status.set(f"Mode: {mode} | Zoom: {self.zoom:.2f}x{fit}")

    def zoom_in(self):
        self.fit_mode = False
        self.zoom = min(self.zoom * 1.25, self.max_zoom)
        self.update_canvas()
        self.update_status()

    def zoom_out(self):
        self.fit_mode = False
        self.zoom = max(self.zoom / 1.25, self.min_zoom)
        self.update_canvas()
        self.update_status()

    def on_mousewheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    # ── canvas interaction ───────────────────────────────────────────────────

    def on_motion(self, event):
        if not self.pan_mode and self.mode.get() == "Rectangle" and len(self.points) == 1:
            self.update_canvas()

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
                self._push_undo()
                cv2.rectangle(self.mask, self.points[0], self.points[1], 0, thickness=-1)
                self.points.clear()
        self.update_canvas()

    def on_drag(self, event):
        if self.pan_mode and self.drag_start:
            self.offset_x += event.x - self.drag_start[0]
            self.offset_y += event.y - self.drag_start[1]
            self.drag_start = (event.x, event.y)
            self.update_canvas()

    def end_drag(self, event):
        self.drag_start = None

    def complete_polygon(self, event=None):
        if self.mode.get() != "Polygon" or len(self.points) < 3:
            return
        self._push_undo()
        cv2.fillPoly(self.mask, [np.array(self.points, dtype=np.int32)], 0)
        self.points.clear()
        self.update_canvas()

    # ── rendering ────────────────────────────────────────────────────────────

    def update_canvas(self):
        if self.image is None or self.mask is None:
            return

        overlay = self.image.copy()
        overlay[self.mask == 0] = (255, 0, 0)

        # Draw in-progress shape
        thickness = max(1, int(1 / self.zoom))
        radius    = max(1, int(3 / self.zoom))
        if not self.pan_mode:
            if self.mode.get() == "Polygon" and self.points:
                for i, pt in enumerate(self.points):
                    cv2.circle(overlay, pt, radius, (0, 255, 0), -1)
                    if i > 0:
                        cv2.line(overlay, self.points[i - 1], pt, (0, 255, 0), thickness)
            elif self.mode.get() == "Rectangle" and len(self.points) == 1:
                mx = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
                my = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
                x2 = int((mx - self.offset_x) / self.zoom)
                y2 = int((my - self.offset_y) / self.zoom)
                cv2.rectangle(overlay, self.points[0], (x2, y2), (0, 255, 0), thickness)

        # Scale to zoom
        new_w = max(1, int(self.width  * self.zoom))
        new_h = max(1, int(self.height * self.zoom))
        scaled = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Viewport-clip so we never allocate more than canvas size
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        view  = np.zeros((ch, cw, 3), dtype=np.uint8)
        src_x = max(0, -self.offset_x)
        src_y = max(0, -self.offset_y)
        dst_x = max(0, self.offset_x)
        dst_y = max(0, self.offset_y)
        cpy_w = min(new_w - src_x, cw - dst_x)
        cpy_h = min(new_h - src_y, ch - dst_y)
        if cpy_w > 0 and cpy_h > 0:
            view[dst_y:dst_y + cpy_h, dst_x:dst_x + cpy_w] = \
                scaled[src_y:src_y + cpy_h, src_x:src_x + cpy_w]

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(view))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)


if __name__ == "__main__":
    root = Tk()
    app = MaskEditor(root)
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        app.load_image_path(sys.argv[1])
    root.mainloop()
