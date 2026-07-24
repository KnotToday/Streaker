import os
import sys
import json
import subprocess
import threading
import time
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk

VERSION = "0.2.0"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
if getattr(sys, 'frozen', False):
    _BASE = Path(os.path.dirname(sys.executable))
else:
    _BASE = Path(__file__).parent.resolve()

_CONFIG_PATH  = _BASE / "shared_config.json"
_EXAMPLE_PATH = _BASE / "shared_config.example.json"
_CAMERAS_PATH = _BASE / "cameras.json"

# Sibling apps live next to SkyEye/ inside the Streaker repo
_REPO = _BASE.parent
_STREAKER_DETECT_DIR = _REPO / "Streaker Detect"
_STREAKER_DETECT_EXE = _STREAKER_DETECT_DIR / "dist" / "StreakerDetect.exe"
_STREAKER_DETECT_PY  = _STREAKER_DETECT_DIR / "StreakerDetect.py"

_RECORD_DIR = _REPO / "Record"
_RECORD_EXE = _RECORD_DIR / "dist" / "STREAKERrec.exe"
_RECORD_PY  = _RECORD_DIR / "Streaker_Record_Launch.py"

_DISPLAY_DIR = _REPO / "Streaker_Display"
_DISPLAY_EXE = _DISPLAY_DIR / "dist" / "StreakerDisplay.exe"
_DISPLAY_PY  = _DISPLAY_DIR / "StreakerDisplay.py"

NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG       = "#0a1520"
BG2      = "#0d1b2a"
BG3      = "#111f30"
ACCENT   = "#1a6fa8"
ACCENT2  = "#1f8acb"
FG       = "#d0e8ff"
FG2      = "#7a9ab8"
GREEN    = "#2ecc71"
RED      = "#e74c3c"
YELLOW   = "#f1c40f"
BORDER   = "#1a3a5a"

BTN_H    = {"bg": BG3, "fg": FG, "activebackground": ACCENT, "activeforeground": "white",
            "relief": "flat", "bd": 0, "cursor": "hand2", "font": ("Arial", 10)}
BTN_BIG  = {"bg": ACCENT, "fg": "white", "activebackground": ACCENT2, "activeforeground": "white",
            "relief": "flat", "bd": 0, "cursor": "hand2", "font": ("Arial", 11, "bold"),
            "padx": 18, "pady": 10}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
_DEFAULT_CFG = {
    "rtsp_url": "rtsp://user:password@192.168.x.x:554/cam/realmonitor?channel=1&subtype=0",
    "latitude": 0.0,
    "longitude": 0.0,
    "output_dir": "D:/",
    "opensky_username": "",
    "opensky_password": "",
}

def _load_config():
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH) as f:
                data = json.load(f)
            cfg = dict(_DEFAULT_CFG)
            cfg.update(data)
            return cfg
        except Exception:
            pass
    return dict(_DEFAULT_CFG)

def _save_config(cfg):
    with open(_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

# ---------------------------------------------------------------------------
# Live camera preview thread
# ---------------------------------------------------------------------------
class CameraThread(threading.Thread):
    def __init__(self, rtsp_url, frame_q):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.frame_q  = frame_q
        self._stop    = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        cap = None
        while not self._stop.is_set():
            try:
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    self.frame_q.put(("error", "Cannot connect to camera"))
                    self._stop.wait(5)
                    continue
                self.frame_q.put(("connected", None))
                while not self._stop.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        self.frame_q.put(("error", "Stream lost"))
                        break
                    # Drain old frames, keep only latest
                    try:
                        while True:
                            self.frame_q.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_q.put(("frame", frame))
                    time.sleep(0.033)  # ~30 fps cap
            except Exception as e:
                self.frame_q.put(("error", str(e)))
            finally:
                if cap:
                    cap.release()
            if not self._stop.is_set():
                self._stop.wait(3)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class SkyEye:
    def __init__(self, root):
        self.root = root
        self.root.title(f"SkyEye v{VERSION}")
        self.root.configure(bg=BG)
        self.root.minsize(800, 560)
        self.root.resizable(True, True)

        self.cfg = _load_config()
        self._cam_thread  = None
        self._frame_q     = queue.Queue(maxsize=2)
        self._cam_photo   = None
        self._logo_photo  = None
        self._preview_on  = False
        self._processes   = {}   # name → Popen
        self._cameras     = []
        self._camera_var  = tk.StringVar()

        self._build_ui()
        self._load_cameras()
        self._update_status_indicators()
        self.root.after(100, self._poll_camera)
        self.root.after(200, lambda: self._status(
            f"Config: {_CONFIG_PATH}  ({'found' if _CONFIG_PATH.exists() else 'NOT FOUND'})"))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ---- top bar ----
        top = tk.Frame(self.root, bg=BG, pady=6, padx=12)
        top.pack(fill='x', side='top')

        tk.Label(top, text="SkyEye", bg=BG, fg=FG,
                 font=("Arial", 16, "bold")).pack(side='left')
        tk.Label(top, text=f"v{VERSION}", bg=BG, fg=FG2,
                 font=("Arial", 9)).pack(side='left', padx=(6, 0), pady=(5, 0))

        tk.Button(top, text="⚙ Settings", command=self._open_settings,
                  **BTN_H, padx=10, pady=4).pack(side='right')

        # Camera profile selector
        tk.Button(top, text="🗑", command=self._camera_delete,
                  **BTN_H, padx=6, pady=4).pack(side='right', padx=(2, 0))
        tk.Button(top, text="✎", command=self._camera_edit,
                  **BTN_H, padx=6, pady=4).pack(side='right', padx=2)
        tk.Button(top, text="+ Camera", command=self._camera_add,
                  bg=ACCENT, fg="white", activebackground=ACCENT2, activeforeground="white",
                  relief='flat', bd=0, cursor='hand2', font=("Arial", 9),
                  padx=8, pady=4).pack(side='right', padx=2)
        self._camera_combo = ttk.Combobox(top, textvariable=self._camera_var,
                                          state='readonly', width=24,
                                          font=("Arial", 9))
        self._camera_combo.pack(side='right', padx=(12, 4))
        self._camera_combo.bind('<<ComboboxSelected>>', self._camera_on_select)
        tk.Label(top, text="Camera:", bg=BG, fg=FG2,
                 font=("Arial", 9)).pack(side='right', padx=(12, 2))

        # ---- main body (left sidebar + center preview) ----
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True, padx=10, pady=(0, 6))

        # Left: app launch panel
        left = tk.Frame(body, bg=BG2, width=200, relief='flat',
                        highlightthickness=1, highlightbackground=BORDER)
        left.pack(side='left', fill='y', padx=(0, 8))
        left.pack_propagate(False)
        self._build_left(left)

        # Center: camera preview
        center = tk.Frame(body, bg=BG3, relief='flat',
                          highlightthickness=1, highlightbackground=BORDER)
        center.pack(side='left', fill='both', expand=True)
        self._build_center(center)

        # ---- status bar ----
        self._status_var = tk.StringVar(value="Ready.")
        sb = tk.Frame(self.root, bg=BG, pady=3)
        sb.pack(fill='x', side='bottom')
        self._status_lbl = tk.Label(sb, textvariable=self._status_var, bg=BG, fg=FG2,
                                    font=("Arial", 8), anchor='w')
        self._status_lbl.pack(side='left', padx=8)

    def _build_left(self, parent):
        tk.Label(parent, text="APPS", bg=BG2, fg=FG2,
                 font=("Arial", 8, "bold")).pack(pady=(12, 4), padx=10, anchor='w')

        sep = lambda: tk.Frame(parent, bg=BORDER, height=1).pack(fill='x', padx=10, pady=4)

        # Recording
        self._rec_dot = self._dot_label(parent)
        self._rec_dot.pack(side='top', anchor='w', padx=10, pady=(8, 0))
        tk.Button(parent, text="⏺  Start Recording", command=self._launch_record,
                  **BTN_H, padx=10, pady=8, anchor='w').pack(fill='x', padx=10, pady=(2, 0))

        sep()

        # Detection
        self._det_dot = self._dot_label(parent)
        self._det_dot.pack(side='top', anchor='w', padx=10, pady=(4, 0))
        tk.Button(parent, text="🔍  Run Detection", command=self._launch_detect,
                  **BTN_H, padx=10, pady=8, anchor='w').pack(fill='x', padx=10, pady=(2, 0))

        sep()

        # Camera display
        self._disp_dot = self._dot_label(parent)
        self._disp_dot.pack(side='top', anchor='w', padx=10, pady=(4, 0))
        tk.Button(parent, text="📷  Camera Display", command=self._launch_display,
                  **BTN_H, padx=10, pady=8, anchor='w').pack(fill='x', padx=10, pady=(2, 0))

        sep()

        # Preview toggle
        self._prev_btn = tk.Button(parent, text="▶  Live Preview",
                                   command=self._toggle_preview,
                                   **BTN_H, padx=10, pady=8, anchor='w')
        self._prev_btn.pack(fill='x', padx=10, pady=(4, 0))

    def _dot_label(self, parent):
        lbl = tk.Label(parent, text="● Idle", bg=BG2, fg=FG2,
                       font=("Arial", 8))
        return lbl

    def _build_center(self, parent):
        self._preview_canvas = tk.Canvas(parent, bg=BG3, bd=0, highlightthickness=0)
        self._preview_canvas.pack(fill='both', expand=True)
        self._preview_canvas.bind("<Configure>", self._on_canvas_resize)
        self.root.after(50, self._show_placeholder)  # defer until canvas has size

    # ------------------------------------------------------------------
    # Preview / camera
    # ------------------------------------------------------------------
    def _toggle_preview(self):
        if self._preview_on:
            self._stop_preview()
        else:
            self._start_preview()

    def _start_preview(self):
        name = self._camera_var.get()
        cam  = next((c for c in self._cameras if c['name'] == name), None)
        if not cam:
            messagebox.showwarning("No Camera",
                "Select a camera profile from the dropdown before starting preview.")
            return
        url = cam.get('rtsp_url', '')
        if not url or 'user:password' in url:
            messagebox.showwarning("No RTSP URL",
                f"Camera '{name}' has no valid RTSP URL. Edit the profile to add one.")
            return
        self._preview_on = True
        self._prev_btn.config(text="⏹  Stop Preview")
        self._status("Connecting to camera…")
        self._frame_q = queue.Queue(maxsize=2)
        self._cam_thread = CameraThread(url, self._frame_q)
        self._cam_thread.start()

    def _show_placeholder(self):
        c = self._preview_canvas
        cw = c.winfo_width() or 800
        ch = c.winfo_height() or 560
        c.delete("all")
        logo_path = _BASE / "skyeye_logo.png"
        if logo_path.exists():
            try:
                img = Image.open(logo_path)
                scale = min(cw / img.width, ch / img.height, 1.0)
                nw, nh = int(img.width * scale), int(img.height * scale)
                img = img.resize((nw, nh), Image.LANCZOS)
                self._logo_photo = ImageTk.PhotoImage(img)
                c.create_image(cw // 2, ch // 2, anchor='center',
                               image=self._logo_photo)
                return
            except Exception:
                pass
        c.create_text(cw // 2, ch // 2,
                      text="▶  Click 'Live Preview' to connect to camera",
                      fill=FG2, font=("Arial", 12), tags="placeholder")

    def _stop_preview(self):
        self._preview_on = False
        self._prev_btn.config(text="▶  Live Preview")
        if self._cam_thread:
            self._cam_thread.stop()
            self._cam_thread = None
        self._show_placeholder()
        self._status("Preview stopped.")

    def _poll_camera(self):
        try:
            while True:
                kind, data = self._frame_q.get_nowait()
                if kind == "frame":
                    self._show_frame(data)
                elif kind == "connected":
                    self._status("Camera connected.")
                elif kind == "error":
                    self._status(f"Camera error: {data}", error=True)
        except queue.Empty:
            pass
        self.root.after(33, self._poll_camera)

    def _show_frame(self, frame):
        cw = self._preview_canvas.winfo_width()
        ch = self._preview_canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._cam_photo = ImageTk.PhotoImage(img)
        self._preview_canvas.delete("all")
        x = (cw - nw) // 2
        y = (ch - nh) // 2
        self._preview_canvas.create_image(x, y, anchor='nw', image=self._cam_photo)

    def _on_canvas_resize(self, _evt):
        if not self._preview_on:
            self._show_placeholder()

    # ------------------------------------------------------------------
    # App launchers
    # ------------------------------------------------------------------
    def _launch(self, name, exe_path, py_path, py_dir):
        if name in self._processes and self._processes[name].poll() is None:
            self._status(f"{name} is already running.")
            return
        if exe_path.exists():
            cmd = [str(exe_path)]
            cwd = str(exe_path.parent)
        elif py_path.exists():
            cmd = [sys.executable, str(py_path)]
            cwd = str(py_dir)
        else:
            messagebox.showerror("Not Found",
                f"Could not find {name}.\nLooked for:\n  {exe_path}\n  {py_path}")
            return
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, creationflags=NO_WINDOW)
            self._processes[name] = proc
            self._status(f"{name} launched (pid {proc.pid}).")
            self._update_status_indicators()
            threading.Thread(target=self._watch_proc, args=(name, proc),
                             daemon=True).start()
        except Exception as e:
            messagebox.showerror("Launch Error", str(e))

    def _watch_proc(self, name, proc):
        proc.wait()
        self.root.after(100, self._update_status_indicators)

    def _launch_record(self):
        self._launch("STREAKERrec", _RECORD_EXE, _RECORD_PY, _RECORD_DIR)

    def _launch_detect(self):
        self._launch("StreakerDetect", _STREAKER_DETECT_EXE,
                     _STREAKER_DETECT_PY, _STREAKER_DETECT_DIR)

    def _launch_display(self):
        self._launch("StreakerDisplay", _DISPLAY_EXE, _DISPLAY_PY, _DISPLAY_DIR)

    # ------------------------------------------------------------------
    # Status indicators
    # ------------------------------------------------------------------
    def _update_status_indicators(self):
        pairs = [
            (self._rec_dot,  "STREAKERrec",      "Recording"),
            (self._det_dot,  "StreakerDetect",    "Detection"),
            (self._disp_dot, "StreakerDisplay",   "Camera Display"),
        ]
        for lbl, name, label in pairs:
            proc = self._processes.get(name)
            if proc and proc.poll() is None:
                lbl.config(text=f"● {label}", fg=GREEN)
            else:
                lbl.config(text=f"● {label}", fg=FG2)

    # ------------------------------------------------------------------
    # Settings dialog
    # ------------------------------------------------------------------
    def _open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.configure(bg=BG)
        win.resizable(False, False)
        win.grab_set()

        fields = [
            ("Latitude",          "latitude",           20, False),
            ("Longitude",         "longitude",          20, False),
            ("OpenSky Username",  "opensky_username",   30, False),
            ("OpenSky Password",  "opensky_password",   30, True),
        ]

        vars_ = {}
        for row, (label, key, width, secret) in enumerate(fields):
            tk.Label(win, text=label, bg=BG, fg=FG,
                     font=("Arial", 9), anchor='e', width=18).grid(
                row=row, column=0, padx=(12, 4), pady=5, sticky='e')
            sv = tk.StringVar(value=str(self.cfg.get(key, "")))
            show = "*" if secret else ""
            ent = tk.Entry(win, textvariable=sv, width=width,
                           bg=BG3, fg=FG, insertbackground=FG,
                           relief='flat', show=show,
                           highlightthickness=1, highlightbackground=BORDER)
            ent.grid(row=row, column=1, padx=(0, 6), pady=5, sticky='w')
            vars_[key] = sv

        tk.Label(win, text=f"Config: {_CONFIG_PATH}", bg=BG, fg=FG2,
                 font=("Consolas", 7)).grid(row=len(fields), column=0, columnspan=3,
                                            padx=12, pady=(0, 4), sticky='w')

        def _browse_dir(sv):
            d = filedialog.askdirectory(initialdir=sv.get() or "/")
            if d:
                sv.set(d)

        def _save():
            for key, sv in vars_.items():
                val = sv.get().strip()
                if key in ("latitude", "longitude"):
                    try:
                        val = float(val)
                    except ValueError:
                        messagebox.showerror("Invalid", f"{key} must be a number.", parent=win)
                        return
                self.cfg[key] = val
            _save_config(self.cfg)
            self._status("Settings saved.")
            win.destroy()

        tk.Button(win, text="Save", command=_save,
                  **BTN_BIG).grid(row=len(fields) + 1, column=0, columnspan=3, pady=12)

    # ------------------------------------------------------------------
    # Camera profiles
    # ------------------------------------------------------------------
    def _load_cameras(self):
        if not _CAMERAS_PATH.exists():
            return
        try:
            with open(_CAMERAS_PATH) as f:
                data = json.load(f)
            self._cameras = data.get('cameras', [])
            self._camera_combo['values'] = [c['name'] for c in self._cameras]
            last = data.get('last_used', '')
            if last and last in [c['name'] for c in self._cameras]:
                self._camera_var.set(last)
                self._apply_camera_profile(last)
        except Exception:
            pass

    def _save_cameras(self):
        try:
            with open(_CAMERAS_PATH, 'w') as f:
                json.dump({'cameras': self._cameras,
                           'last_used': self._camera_var.get()}, f, indent=2)
        except Exception:
            pass

    def _apply_camera_profile(self, name):
        for cam in self._cameras:
            if cam['name'] == name:
                if cam.get('rtsp_url'):
                    self.cfg['rtsp_url'] = cam['rtsp_url']
                if cam.get('output_dir'):
                    self.cfg['output_dir'] = cam['output_dir']
                _save_config(self.cfg)
                return

    def _camera_on_select(self, _event=None):
        name = self._camera_var.get()
        self._apply_camera_profile(name)
        self._save_cameras()
        self._status(f"Camera profile: {name}")

    def _camera_add(self):
        self._camera_dialog("Add Camera")

    def _camera_edit(self):
        name = self._camera_var.get()
        if not name:
            messagebox.showwarning("No Selection", "Select a camera profile to edit.")
            return
        profile = next((c for c in self._cameras if c['name'] == name), None)
        self._camera_dialog("Edit Camera", profile)

    def _camera_delete(self):
        name = self._camera_var.get()
        if not name:
            messagebox.showwarning("No Selection", "Select a camera profile to delete.")
            return
        if not messagebox.askyesno("Delete Camera", f"Delete profile '{name}'?"):
            return
        self._cameras = [c for c in self._cameras if c['name'] != name]
        self._camera_var.set('')
        self._camera_combo['values'] = [c['name'] for c in self._cameras]
        self._save_cameras()

    def _camera_dialog(self, title, profile=None):
        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.grab_set()

        def lbl(text):
            tk.Label(dlg, text=text, bg=BG, fg=FG2,
                     font=("Arial", 8)).pack(anchor='w', padx=12, pady=(8, 0))

        def entry_row(var, width=44, secret=False):
            f = tk.Frame(dlg, bg=BG)
            f.pack(fill='x', padx=12, pady=(0, 2))
            e = tk.Entry(f, textvariable=var, bg=BG3, fg=FG,
                         insertbackground=FG, relief='flat', width=width,
                         highlightthickness=1, highlightbackground=BORDER,
                         show='*' if secret else '')
            e.pack(side='left', fill='x', expand=True)
            return f

        def browse_dir(var):
            p = filedialog.askdirectory(initialdir=var.get() or '/')
            if p:
                var.set(p)

        v_name   = tk.StringVar(value=profile['name']          if profile else '')
        v_rtsp   = tk.StringVar(value=profile.get('rtsp_url',  '') if profile else '')
        v_output = tk.StringVar(value=profile.get('output_dir', '') if profile else '')

        lbl("Camera name:")
        entry_row(v_name)

        lbl("RTSP URL:")
        entry_row(v_rtsp)

        lbl("Output directory (recordings):")
        fr = entry_row(v_output)
        tk.Button(fr, text="…", command=lambda: browse_dir(v_output),
                  **BTN_H, padx=6, pady=2).pack(side='left', padx=(4, 0))

        def _save():
            name = v_name.get().strip()
            if not name:
                messagebox.showerror("Error", "Camera name is required.", parent=dlg)
                return
            new_profile = {
                'name':       name,
                'rtsp_url':   v_rtsp.get().strip(),
                'output_dir': v_output.get().strip(),
            }
            if profile:
                for i, c in enumerate(self._cameras):
                    if c['name'] == profile['name']:
                        self._cameras[i] = new_profile
                        break
            else:
                if any(c['name'] == name for c in self._cameras):
                    messagebox.showerror("Error", f"A camera named '{name}' already exists.",
                                         parent=dlg)
                    return
                self._cameras.append(new_profile)
            self._camera_combo['values'] = [c['name'] for c in self._cameras]
            self._camera_var.set(name)
            self._apply_camera_profile(name)
            self._save_cameras()
            dlg.destroy()

        btn_f = tk.Frame(dlg, bg=BG)
        btn_f.pack(pady=12)
        tk.Button(btn_f, text="Save", command=_save, **BTN_BIG).pack(side='left', padx=6)
        tk.Button(btn_f, text="Cancel", command=dlg.destroy,
                  **BTN_H, padx=12, pady=8).pack(side='left')

        dlg.wait_window()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _status(self, msg, error=False):
        self._status_var.set(msg)
        self._status_lbl.config(fg=RED if error else FG2)

    def _on_close(self):
        self._stop_preview()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    root.geometry("900x580")
    app = SkyEye(root)
    root.mainloop()


if __name__ == "__main__":
    main()
