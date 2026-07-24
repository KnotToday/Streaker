import os
import sys
import random
import cv2
import numpy as np
from tkinter import Tk, Canvas, filedialog, messagebox, Frame, StringVar, Label, Button
from PIL import Image, ImageTk

# ── Design tokens ──────────────────────────────────────────────────────────────
BG    = '#0c0c0c'
SURF  = '#141414'
BTN   = '#1e1e1e'
BTN_A = '#2a2a2a'
FG    = '#e0e0e0'
FG2   = '#555555'
GREEN = '#22c55e'
AMBER = '#f59e0b'
FONT  = 'Segoe UI'


def _btn(parent, text, cmd, fg=FG, bg=BTN, **kw):
    b = Button(parent, text=text, command=cmd,
               bg=bg, fg=fg, relief='flat',
               font=(FONT, 9), padx=0, pady=7,
               activebackground=BTN_A, activeforeground='#ffffff',
               cursor='hand2', **kw)
    b.pack(fill='x', padx=10, pady=2)
    return b


def _divider(parent):
    Frame(parent, bg='#222222', height=1).pack(fill='x', padx=10, pady=8)


def _section(parent, text):
    Label(parent, text=text, bg=SURF, fg=FG2,
          font=(FONT, 7, 'bold'), anchor='w').pack(fill='x', padx=14, pady=(6, 1))


class MaskEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Mask Editor")
        self.root.configure(bg=BG)
        self.root.geometry('1280x820')
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Layout ────────────────────────────────────────────────────────────
        sidebar = Frame(root, bg=SURF, width=176)
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)

        right = Frame(root, bg=BG)
        right.pack(side='left', fill='both', expand=True)

        self.canvas = Canvas(right, cursor='crosshair', bg='#000000',
                             highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        bar = Frame(root, bg=SURF, height=26)
        bar.pack(side='bottom', fill='x')
        bar.pack_propagate(False)
        self._status = StringVar(value="Load an image to begin")
        Label(bar, textvariable=self._status, bg=SURF, fg=FG2,
              font=(FONT, 8), anchor='w').pack(side='left', padx=10)

        # ── Sidebar ───────────────────────────────────────────────────────────
        Label(sidebar, text='MASK EDITOR', bg=SURF, fg='#333333',
              font=(FONT, 8, 'bold')).pack(pady=(16, 2))
        Frame(sidebar, bg='#222222', height=1).pack(fill='x')

        _section(sidebar, 'FILE')
        _btn(sidebar, 'Load Image',         self.load_image)
        _btn(sidebar, 'Grab MKV Frame',     self.grab_mkv_frame)
        _btn(sidebar, 'Load Existing Mask', self.load_mask)
        _btn(sidebar, 'Save Mask',          self.save_mask,
             fg='#ffffff', bg='#14532d')
        _btn(sidebar, 'Export Masked Image', self.export_masked_image,
             fg='#ffffff', bg='#1e3a5f')

        _divider(sidebar)
        _section(sidebar, 'EDIT')
        _btn(sidebar, 'Undo',               self.undo)
        _btn(sidebar, 'Clear Mask',         self.clear_mask, fg='#fca5a5')

        _divider(sidebar)
        _section(sidebar, 'DRAW MODE')
        self._mode = StringVar(value='Polygon')
        self._mode_btn = _btn(sidebar, 'Polygon', self.toggle_mode,
                              fg='#93c5fd', bg='#172554')

        _divider(sidebar)
        _section(sidebar, 'VIEW')
        _btn(sidebar, 'Zoom In',            self.zoom_in)
        _btn(sidebar, 'Zoom Out',           self.zoom_out)
        _btn(sidebar, 'Fit to Window',      self.fit_to_window)
        self._pan_btn = _btn(sidebar, 'Pan Mode',   self.toggle_pan_mode)

        Label(sidebar, text='scroll = zoom\nmiddle-drag = pan\nright-click = finish poly',
              bg=SURF, fg=FG2, font=(FONT, 7), justify='center').pack(pady=(12, 0))

        # ── Canvas bindings ───────────────────────────────────────────────────
        self.canvas.bind('<Button-1>',        self.on_left_click)
        self.canvas.bind('<B1-Motion>',       self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.end_drag)
        self.canvas.bind('<Button-3>',        self.complete_polygon)
        self.canvas.bind('<Motion>',          self.on_motion)
        self.canvas.bind('<MouseWheel>',      self.on_mousewheel)
        self.canvas.bind('<Configure>',       self._on_configure)
        self.canvas.bind('<Button-2>',        lambda e: self._pan_start(e))
        self.canvas.bind('<B2-Motion>',       lambda e: self._pan_drag(e))
        self.canvas.bind('<ButtonRelease-2>', lambda e: self._pan_end(e))

        # ── State ─────────────────────────────────────────────────────────────
        self.image       = None
        self.tk_image    = None
        self.mask        = None
        self.filename    = None
        self.points      = []
        self.undo_stack  = []
        self.height      = 0
        self.width       = 0
        self.zoom        = 1.0
        self.min_zoom    = 0.02
        self.max_zoom    = 20.0
        self.offset_x    = 0
        self.offset_y    = 0
        self.drag_start  = None
        self.pan_mode    = False
        self.fit_mode    = True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _on_close(self):
        self.root.destroy()

    def _push_undo(self):
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)

    # ── Image loading ──────────────────────────────────────────────────────────

    def _fit_to_canvas(self):
        self.root.update_idletasks()
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self.zoom = min(cw / self.width, ch / self.height)
        self.offset_x = 0
        self.offset_y = 0

    def load_image_path(self, path):
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Failed to load:\n{path}")
            return
        self.filename = path
        self.image    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
        self.mask = np.full((self.height, self.width), 255, dtype=np.uint8)
        self.points.clear()
        self.undo_stack.clear()
        self.fit_mode = True
        self.root.after(50, self._fit_and_draw)

    def _fit_and_draw(self):
        self._fit_to_canvas()
        self.update_canvas()
        self._update_status()

    def grab_mkv_frame(self):
        path = filedialog.askopenfilename(
            title="Select MKV File",
            filetypes=[("Video files", "*.mkv;*.mp4;*.avi"), ("All files", "*.*")])
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open:\n{path}")
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        lo, hi = int(total * 0.1), max(1, int(total * 0.9))
        cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(lo, hi - 1))
        ret, raw = cap.read()
        cap.release()
        if not ret or raw is None:
            messagebox.showerror("Error", "Could not grab frame.")
            return
        self.filename = path
        self.image    = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
        self.mask = np.full((self.height, self.width), 255, dtype=np.uint8)
        self.points.clear()
        self.undo_stack.clear()
        self.fit_mode = True
        self.root.after(50, self._fit_and_draw)

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")])
        if path:
            self.load_image_path(path)

    def load_mask(self):
        path = filedialog.askopenfilename(filetypes=[("PNG Mask", "*.png")])
        if not path or self.image is None:
            return
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != self.mask.shape:
            messagebox.showerror("Error", "Cannot load mask or size mismatch.")
            return
        self._push_undo()
        self.mask = m.copy()
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
        out = os.path.join(os.path.dirname(self.filename), "mask.png")
        cv2.imwrite(out, self.mask)
        messagebox.showinfo("Saved", f"Mask saved:\n{out}")

    def export_masked_image(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded.")
            return
        if self.mask is None:
            messagebox.showerror("Error", "No mask to apply.")
            return
        out = filedialog.asksaveasfilename(
            title="Save masked image as…",
            initialdir=os.path.dirname(self.filename),
            initialfile="masked_" + os.path.basename(self.filename),
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")])
        if not out:
            return
        result = self.image.copy()
        result[self.mask == 0] = 0
        cv2.imwrite(out, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        messagebox.showinfo("Exported", f"Masked image saved:\n{out}")

    # ── Undo ──────────────────────────────────────────────────────────────────

    def undo(self):
        if self.points:
            self.points.pop()
            self.update_canvas()
        elif self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.update_canvas()

    # ── Mode / zoom ───────────────────────────────────────────────────────────

    def toggle_mode(self):
        if self._mode.get() == 'Polygon':
            self._mode.set('Rectangle')
            self._mode_btn.config(text='Rectangle', fg='#c4b5fd', bg='#2e1065')
        else:
            self._mode.set('Polygon')
            self._mode_btn.config(text='Polygon', fg='#93c5fd', bg='#172554')
        self.points.clear()
        self.update_canvas()
        self._update_status()

    def toggle_pan_mode(self):
        self.pan_mode = not self.pan_mode
        if self.pan_mode:
            self._pan_btn.config(text='Pan: ON', fg=AMBER, bg='#292300')
        else:
            self._pan_btn.config(text='Pan Mode', fg=FG, bg=BTN)
        self._update_status()

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
        self._update_status()

    def _update_status(self):
        if self.image is None:
            self.status.set("Load an image to begin")  # type: ignore[attr-defined]
            return
        mode = 'Pan' if self.pan_mode else self._mode.get()
        name = os.path.basename(self.filename) if self.filename else '—'
        self._status.set(
            f"{name}  ·  {mode}  ·  {self.zoom:.2f}×"
            f"  ·  scroll=zoom  middle-drag=pan  right-click=finish polygon")

    def zoom_in(self):
        self.fit_mode = False
        self.zoom = min(self.zoom * 1.25, self.max_zoom)
        self.update_canvas()
        self._update_status()

    def zoom_out(self):
        self.fit_mode = False
        self.zoom = max(self.zoom / 1.25, self.min_zoom)
        self.update_canvas()
        self._update_status()

    def on_mousewheel(self, event):
        if self.image is None:
            return
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self.fit_mode = False
        img_x = (event.x - self.offset_x) / self.zoom
        img_y = (event.y - self.offset_y) / self.zoom
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * factor))
        self.offset_x = int(event.x - img_x * self.zoom)
        self.offset_y = int(event.y - img_y * self.zoom)
        self.update_canvas()
        self._update_status()

    def _pan_start(self, event):
        self.drag_start = (event.x, event.y)

    def _pan_drag(self, event):
        if self.drag_start:
            self.fit_mode = False
            self.offset_x += event.x - self.drag_start[0]
            self.offset_y += event.y - self.drag_start[1]
            self.drag_start = (event.x, event.y)
            self.update_canvas()

    def _pan_end(self, event):
        self.drag_start = None

    # ── Canvas interaction ────────────────────────────────────────────────────

    def on_motion(self, event):
        if not self.pan_mode and self._mode.get() == 'Rectangle' and len(self.points) == 1:
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
        if self._mode.get() == 'Polygon':
            self.points.append((x, y))
        elif self._mode.get() == 'Rectangle':
            self.points.append((x, y))
            if len(self.points) == 2:
                self._push_undo()
                cv2.rectangle(self.mask, self.points[0], self.points[1], 0, -1)
                self.points.clear()
        self.update_canvas()

    def on_drag(self, event):
        if self.pan_mode and self.drag_start:
            self.fit_mode = False
            self.offset_x += event.x - self.drag_start[0]
            self.offset_y += event.y - self.drag_start[1]
            self.drag_start = (event.x, event.y)
            self.update_canvas()

    def end_drag(self, event):
        self.drag_start = None

    def complete_polygon(self, event=None):
        if self._mode.get() != 'Polygon' or len(self.points) < 3:
            return
        self._push_undo()
        cv2.fillPoly(self.mask, [np.array(self.points, dtype=np.int32)], 0)
        self.points.clear()
        self.update_canvas()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def update_canvas(self):
        if self.image is None or self.mask is None:
            return

        overlay = self.image.copy()
        # Masked regions: deep red tint
        mx = self.mask == 0
        overlay[mx] = (overlay[mx].astype(np.float32) * 0.2 +
                       np.array([180, 30, 30], dtype=np.float32) * 0.8).clip(0, 255).astype(np.uint8)

        tk = max(1, int(1.5 / self.zoom))
        rk = max(1, int(5 / self.zoom))
        if not self.pan_mode:
            if self._mode.get() == 'Polygon' and self.points:
                for i, pt in enumerate(self.points):
                    cv2.circle(overlay, pt, rk, (80, 220, 120), -1)
                    if i > 0:
                        cv2.line(overlay, self.points[i - 1], pt, (80, 220, 120), tk)
                mx2 = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
                my2 = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
                xp = int((mx2 - self.offset_x) / self.zoom)
                yp = int((my2 - self.offset_y) / self.zoom)
                cv2.line(overlay, self.points[-1], (xp, yp), (80, 220, 120), tk)
            elif self._mode.get() == 'Rectangle' and len(self.points) == 1:
                mx2 = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
                my2 = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
                x2 = int((mx2 - self.offset_x) / self.zoom)
                y2 = int((my2 - self.offset_y) / self.zoom)
                cv2.rectangle(overlay, self.points[0], (x2, y2), (80, 220, 120), tk)

        nw = max(1, int(self.width  * self.zoom))
        nh = max(1, int(self.height * self.zoom))
        scaled = cv2.resize(overlay, (nw, nh), interpolation=cv2.INTER_LINEAR)

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        view  = np.zeros((ch, cw, 3), dtype=np.uint8)
        sx = max(0, -self.offset_x)
        sy = max(0, -self.offset_y)
        dx = max(0,  self.offset_x)
        dy = max(0,  self.offset_y)
        cpw = min(nw - sx, cw - dx)
        cph = min(nh - sy, ch - dy)
        if cpw > 0 and cph > 0:
            view[dy:dy+cph, dx:dx+cpw] = scaled[sy:sy+cph, sx:sx+cpw]

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(view))
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)


if __name__ == '__main__':
    root = Tk()
    app = MaskEditor(root)
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        app.load_image_path(sys.argv[1])
    root.mainloop()
