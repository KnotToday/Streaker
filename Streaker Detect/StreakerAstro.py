# StreakerAstro.py — Plate solve via astrometry.net and measure meteor angular velocity

import os
import sys
import json
import subprocess
import time
import math
import threading
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

try:
    from StreakerLocalSolve import LocalSolveWindow
    _LOCAL_SOLVE = True
except ImportError:
    _LOCAL_SOLVE = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from astropy.io import fits
    from astropy.wcs import WCS
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

ASTROMETRY_URL = "https://nova.astrometry.net/api"
import sys as _sys
def _config_path():
    # Prefer --config-dir arg (passed by StreakerPlayer when launching as subprocess)
    for i, arg in enumerate(_sys.argv[1:], 1):
        if arg == '--config-dir' and i < len(_sys.argv):
            return Path(_sys.argv[i + 1]) / "streaker_astro_config.json"
    if getattr(_sys, 'frozen', False):
        return Path(_sys.executable).parent / "streaker_astro_config.json"
    return Path(__file__).parent / "streaker_astro_config.json"

CONFIG_PATH = _config_path()

BG       = '#1a1a2e'
FG       = '#e0e0e0'
ENTRY_BG = '#2a2a4a'
ACCENT   = '#336699'


def _load_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_config(cfg):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)


def _stretch_image(path, mode='blackpoint', bp_percentile=90):
    """Stretch image before upload. Returns temp file path."""
    import tempfile
    import numpy as np
    from PIL import Image as PILImage
    img = np.array(PILImage.open(path).convert('RGB')).astype(np.float32)

    if mode == 'blackpoint':
        bg = np.percentile(img, bp_percentile)
        img = np.clip(img - bg, 0, None)
        peak = img.max()
        if peak > 0:
            img = img / peak * 255
    elif mode == 'sqrt':
        lo = np.percentile(img, 1)
        hi = np.percentile(img, 99.5)
        img = (img - lo) / max(hi - lo, 1)
        img = np.sqrt(np.clip(img, 0, 1)) * 255

    stretched = img.clip(0, 255).astype(np.uint8)
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.close()
    PILImage.fromarray(stretched).save(tmp.name)
    return tmp.name


def _angular_sep_deg(ra1, dec1, ra2, dec2):
    r = math.pi / 180
    ra1, dec1, ra2, dec2 = ra1*r, dec1*r, ra2*r, dec2*r
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = math.sin(ddec/2)**2 + math.cos(dec1)*math.cos(dec2)*math.sin(dra/2)**2
    return 2 * math.degrees(math.asin(math.sqrt(min(a, 1.0))))


class StreakerAstro:
    def __init__(self, root, stack_path=None, event_dir=None):
        self.root = root
        self.root.title("StreakerAstro — Plate Solve & Measure")
        self.root.configure(bg=BG)
        self.root.geometry("680x660")
        self.root.resizable(True, False)

        self.cfg        = _load_config()
        # Prefer contrast image saved by Local Solve over raw stack
        contrast_img = self.cfg.get('last_contrast_image', '')
        if contrast_img and os.path.isfile(contrast_img):
            self.stack_path = contrast_img
        else:
            self.stack_path = stack_path
        self.event_dir  = event_dir
        self._wcs_path  = None

        self._build_ui()

        if self.stack_path:
            self.stack_lbl.config(text=str(self.stack_path), fg=FG)
        if event_dir:
            self.event_lbl.config(text=os.path.basename(str(event_dir)), fg=FG)

    def _build_ui(self):
        pad = dict(padx=14, pady=5)

        # ── API key ───────────────────────────────────────────────────
        row = tk.Frame(self.root, bg=BG)
        row.pack(fill='x', **pad)
        tk.Label(row, text="API Key:", bg=BG, fg=FG,
                 font=('Arial', 9), width=10, anchor='w').pack(side='left')
        self.key_var = tk.StringVar(value=self.cfg.get('api_key', ''))
        self._key_entry = tk.Entry(row, textvariable=self.key_var, width=42,
                                   bg=ENTRY_BG, fg=FG, insertbackground=FG,
                                   relief='flat', show='*')
        self._key_entry.pack(side='left', padx=4)
        tk.Button(row, text="Show/Hide",
                  command=self._toggle_key_vis,
                  bg='#333344', fg='white', relief='flat', padx=6).pack(side='left', padx=2)
        tk.Button(row, text="Save",
                  command=self._save_key,
                  bg='#224422', fg='white', relief='flat', padx=8).pack(side='left', padx=4)

        # ── Stack file ────────────────────────────────────────────────
        row2 = tk.Frame(self.root, bg=BG)
        row2.pack(fill='x', **pad)
        tk.Label(row2, text="Stack:", bg=BG, fg=FG,
                 font=('Arial', 9), width=10, anchor='w').pack(side='left')
        self.stack_lbl = tk.Label(row2, text="None selected", bg=BG, fg='#666666',
                                  font=('Arial', 8), anchor='w')
        self.stack_lbl.pack(side='left', padx=4, fill='x', expand=True)
        tk.Button(row2, text="Browse…", command=self._browse_stack,
                  bg='#334455', fg='white', relief='flat', padx=6).pack(side='right')

        # ── Event dir ─────────────────────────────────────────────────
        row3 = tk.Frame(self.root, bg=BG)
        row3.pack(fill='x', **pad)
        tk.Label(row3, text="Event:", bg=BG, fg=FG,
                 font=('Arial', 9), width=10, anchor='w').pack(side='left')
        self.event_lbl = tk.Label(row3, text="None selected", bg=BG, fg='#666666',
                                  font=('Arial', 8), anchor='w')
        self.event_lbl.pack(side='left', padx=4, fill='x', expand=True)
        tk.Button(row3, text="Browse…", command=self._browse_event,
                  bg='#334455', fg='white', relief='flat', padx=6).pack(side='right')

        # ── Mask file ─────────────────────────────────────────────────
        row_mask = tk.Frame(self.root, bg=BG)
        row_mask.pack(fill='x', **pad)
        tk.Label(row_mask, text="Mask:", bg=BG, fg=FG,
                 font=('Arial', 9), width=10, anchor='w').pack(side='left')
        self.mask_path = self.cfg.get('mask_path', '')
        self.mask_lbl = tk.Label(row_mask,
                                 text=os.path.basename(self.mask_path) if self.mask_path else "None (optional)",
                                 bg=BG, fg=FG if self.mask_path else '#666666',
                                 font=('Arial', 8), anchor='w')
        self.mask_lbl.pack(side='left', padx=4, fill='x', expand=True)
        tk.Button(row_mask, text="Clear",
                  command=self._clear_mask,
                  bg='#443333', fg='white', relief='flat', padx=6).pack(side='right', padx=(2, 0))
        tk.Button(row_mask, text="Browse…", command=self._browse_mask,
                  bg='#334455', fg='white', relief='flat', padx=6).pack(side='right')

        # ── Sky center hint ───────────────────────────────────────────
        center_row = tk.Frame(self.root, bg=BG)
        center_row.pack(fill='x', padx=14, pady=(2, 0))
        tk.Label(center_row, text="Center RA (°):", bg=BG, fg=FG,
                 font=('Arial', 9), width=14, anchor='w').pack(side='left')
        self.ra_var = tk.StringVar(value=self.cfg.get('center_ra', ''))
        tk.Entry(center_row, textvariable=self.ra_var, width=8,
                 bg=ENTRY_BG, fg=FG, insertbackground=FG,
                 relief='flat').pack(side='left', padx=4)
        tk.Label(center_row, text="Dec (°):", bg=BG, fg=FG,
                 font=('Arial', 9)).pack(side='left', padx=(8, 2))
        self.dec_var = tk.StringVar(value=self.cfg.get('center_dec', ''))
        tk.Entry(center_row, textvariable=self.dec_var, width=8,
                 bg=ENTRY_BG, fg=FG, insertbackground=FG,
                 relief='flat').pack(side='left', padx=4)
        tk.Label(center_row, text="Radius (°):", bg=BG, fg=FG,
                 font=('Arial', 9)).pack(side='left', padx=(8, 2))
        self.radius_var = tk.StringVar(value=self.cfg.get('center_radius', '30'))
        tk.Entry(center_row, textvariable=self.radius_var, width=5,
                 bg=ENTRY_BG, fg=FG, insertbackground=FG,
                 relief='flat').pack(side='left', padx=4)
        tk.Label(center_row, text="(auto-filled after first solve)",
                 bg=BG, fg='#666666', font=('Arial', 8)).pack(side='left', padx=4)

        # ── FOV hint ──────────────────────────────────────────────────
        fov_row = tk.Frame(self.root, bg=BG)
        fov_row.pack(fill='x', padx=14, pady=(2, 0))
        tk.Label(fov_row, text="FOV width (°):", bg=BG, fg=FG,
                 font=('Arial', 9), width=14, anchor='w').pack(side='left')
        self.fov_var = tk.StringVar(value=self.cfg.get('fov_deg', '90'))
        tk.Entry(fov_row, textvariable=self.fov_var, width=8,
                 bg=ENTRY_BG, fg=FG, insertbackground=FG,
                 relief='flat').pack(side='left', padx=4)
        tk.Label(fov_row, text="(leave blank = no hint;  ±20% range used)",
                 bg=BG, fg='#666666', font=('Arial', 8)).pack(side='left', padx=4)

        # ── Stretch option ────────────────────────────────────────────
        stretch_row = tk.Frame(self.root, bg=BG)
        stretch_row.pack(fill='x', padx=14, pady=(2, 0))
        self.stretch_var = tk.StringVar(value='none')
        tk.Label(stretch_row, text="Stretch:", bg=BG, fg=FG,
                 font=('Arial', 9), width=14, anchor='w').pack(side='left')
        tk.Radiobutton(stretch_row, text="Black point clip",
                       variable=self.stretch_var, value='blackpoint',
                       bg=BG, fg=FG, selectcolor=ENTRY_BG,
                       activebackground=BG, activeforeground=FG,
                       font=('Arial', 9)).pack(side='left')
        tk.Radiobutton(stretch_row, text="Percentile+sqrt",
                       variable=self.stretch_var, value='sqrt',
                       bg=BG, fg=FG, selectcolor=ENTRY_BG,
                       activebackground=BG, activeforeground=FG,
                       font=('Arial', 9)).pack(side='left', padx=(8, 0))
        tk.Radiobutton(stretch_row, text="None",
                       variable=self.stretch_var, value='none',
                       bg=BG, fg=FG, selectcolor=ENTRY_BG,
                       activebackground=BG, activeforeground=FG,
                       font=('Arial', 9)).pack(side='left', padx=(8, 0))

        self._bp_row = tk.Frame(self.root, bg=BG)
        self._bp_row.pack(fill='x', padx=14, pady=(0, 2))
        tk.Label(self._bp_row, text="Black point %:", bg=BG, fg=FG,
                 font=('Arial', 9), width=14, anchor='w').pack(side='left')
        self.bp_var = tk.IntVar(value=int(self.cfg.get('bp_percentile', 90)))
        self._bp_scale = tk.Scale(self._bp_row, from_=50, to=99, orient='horizontal',
                                  variable=self.bp_var, length=160,
                                  bg=BG, fg=FG, troughcolor=ENTRY_BG,
                                  highlightthickness=0, font=('Arial', 8),
                                  showvalue=True)
        self._bp_scale.pack(side='left', padx=4)
        self._bp_hint = tk.Label(self._bp_row, text="← more stars   fewer stars →",
                                 bg=BG, fg='#666666', font=('Arial', 8))
        self._bp_hint.pack(side='left', padx=6)
        self.stretch_var.trace_add('write', lambda *_: self._update_bp_state())

        # ── Preview + Solve buttons ───────────────────────────────────
        btn_row = tk.Frame(self.root, bg=BG)
        btn_row.pack(fill='x', padx=14, pady=8)
        tk.Button(btn_row, text="👁 Preview Stretch",
                  command=self._preview_stretch,
                  bg='#3a3a1a', fg='white', relief='flat',
                  font=('Arial', 10), pady=6).pack(side='left', padx=(0, 8))
        self.solve_btn = tk.Button(btn_row, text="🔭  Solve & Measure",
                                   command=self._start_solve,
                                   bg='#1a3a5a', fg='white', relief='flat',
                                   font=('Arial', 11, 'bold'), pady=6)
        self.solve_btn.pack(side='left', fill='x', expand=True, padx=(0, 8))
        tk.Button(btn_row, text="📐 Measure (WCS)",
                  command=self._measure_saved_wcs,
                  bg='#2a1a4a', fg='white', relief='flat',
                  font=('Arial', 10), pady=6).pack(side='left', padx=(0, 8))
        tk.Button(btn_row, text="📏 Manual Calibrate",
                  command=self._manual_calibrate,
                  bg='#2a3a1a', fg='white', relief='flat',
                  font=('Arial', 10), pady=6).pack(side='left', padx=(0, 8))
        tk.Button(btn_row, text="🌟 Local Solve",
                  command=self._local_solve,
                  bg='#3a2a1a', fg='white', relief='flat',
                  font=('Arial', 10), pady=6,
                  state='normal' if _LOCAL_SOLVE else 'disabled').pack(side='left')

        # ── Status ────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(self.root, textvariable=self.status_var,
                 bg=BG, fg='#aaaaaa', font=('Courier', 8),
                 anchor='w').pack(fill='x', padx=14)

        tk.Label(self.root, text='─' * 90, bg=BG, fg='#333366').pack(fill='x', padx=14, pady=(8, 0))

        # ── Results ───────────────────────────────────────────────────
        self.result_text = tk.Text(self.root, bg='#0d0d1a', fg=FG,
                                   font=('Courier', 9), relief='flat',
                                   height=13, state='disabled')
        self.result_text.pack(fill='both', expand=True, padx=14, pady=4)

        tk.Button(self.root, text="📋 Copy Results", command=self._copy_results,
                  bg='#333344', fg='white', relief='flat',
                  padx=8, pady=4).pack(pady=6)

        self._update_bp_state()

    # ── UI helpers ────────────────────────────────────────────────────

    def _update_bp_state(self):
        state = 'normal' if self.stretch_var.get() == 'blackpoint' else 'disabled'
        self._bp_scale.config(state=state)
        self._bp_hint.config(fg='#666666' if state == 'normal' else '#444444')

    def _local_solve(self):
        """Open offline plate solver: click named stars → WCS → pixel scale + FOV."""
        if not self.stack_path or not os.path.isfile(self.stack_path):
            messagebox.showwarning("No stack", "Load a stack image first.", parent=self.root)
            return

        def _on_result(result):
            px = result.get('pixel_scale_arcsec', 0)
            fov_w = result.get('fov_width_deg', 0)
            fov_h = result.get('fov_height_deg', 0)
            cra  = result.get('center_ra', 0)
            cdec = result.get('center_dec', 0)
            # Pre-fill FOV hint with the solved value
            self.fov_var.set(f"{fov_w:.1f}")
            # Pre-fill RA/Dec center hint fields if they exist
            try:
                self.ra_var.set(f"{cra:.4f}")
                self.dec_var.set(f"{cdec:.4f}")
            except AttributeError:
                pass
            self.status_var.set(
                f"Local solve: {px:.2f}\"/px  FOV {fov_w:.2f}°×{fov_h:.2f}°  "
                f"Center {cra:.3f}° {cdec:.3f}°  ({result.get('n_matched',0)} stars)")

        LocalSolveWindow(self.root, self.stack_path, on_result=_on_result)

    def _manual_calibrate(self):
        """Open click-to-calibrate window: two stars → pixel scale → meteor velocity."""
        import numpy as np
        from PIL import Image, ImageTk

        stack_path = self.stack_path
        if not stack_path or not os.path.isfile(stack_path):
            messagebox.showerror("No Stack", "Browse to a stack image first.")
            return

        img_pil = Image.open(stack_path).convert('RGB')
        img_w, img_h = img_pil.size
        img_arr = np.array(img_pil)          # (H, W, 3) uint8 for centroid snapping

        # Size canvas to ~90% of screen, leave room for controls (~160px)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        CANVAS_W = int(sw * 0.88)
        CANVAS_H = int(sh * 0.72)
        scale = min(CANVAS_W / img_w, CANVAS_H / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        img_disp = img_pil.resize((disp_w, disp_h), Image.LANCZOS)

        win = tk.Toplevel(self.root)
        saved_px = self.cfg.get('px_arcsec')
        title = "Manual Pixel-Scale Calibration"
        if saved_px:
            title += f"  —  last calibration: {saved_px} arcsec/px"
        win.title(title)
        win.configure(bg=BG)
        win.resizable(True, True)

        # Canvas
        canvas = tk.Canvas(win, width=disp_w, height=disp_h,
                            bg='black', highlightthickness=0, cursor='crosshair')
        canvas.pack(padx=8, pady=(8, 2))
        img_tk = ImageTk.PhotoImage(img_disp)
        canvas.create_image(0, 0, anchor='nw', image=img_tk)
        canvas._img_ref = img_tk  # prevent GC

        # Click mode
        mode_var = tk.StringVar(value='star1')
        points = {'star1': None, 'star2': None, 'meteor_start': None, 'meteor_end': None}
        marker_ids = {}
        snap_radius = 30  # original-image pixels to search for centroid

        mode_frame = tk.Frame(win, bg=BG)
        mode_frame.pack(fill='x', padx=8, pady=2)
        for val, lbl, col in [
            ('star1',        '⭐ Star 1',       '#1a4a1a'),
            ('star2',        '⭐ Star 2',       '#1a4a1a'),
            ('meteor_start', '→ Meteor Start', '#1a1a4a'),
            ('meteor_end',   '→ Meteor End',   '#1a1a4a'),
        ]:
            tk.Radiobutton(mode_frame, text=lbl, variable=mode_var, value=val,
                           bg=BG, fg=FG, selectcolor=col, activebackground=BG,
                           font=('Arial', 9)).pack(side='left', padx=8)

        # Overlay clip stack onto reference image
        img_arr_ref = img_arr.copy()   # keep original for star centroid snapping

        def overlay_clip_stack():
            try:
                from platform_utils import FFMPEG_PATH as _ffp
            except ImportError:
                import shutil as _sh
                _ffp = _sh.which('ffmpeg') or 'ffmpeg'

            # Prefer existing stack PNG, then clip.mkv, then browse
            stack_png, clip_mkv = None, None
            if self.event_dir:
                _png = os.path.join(self.event_dir, 'astrometry_grabs', 'stack_mean.png')
                _mkv = os.path.join(self.event_dir, 'clip.mkv')
                if os.path.isfile(_png):
                    stack_png = _png
                elif os.path.isfile(_mkv):
                    clip_mkv = _mkv
            if not stack_png and not clip_mkv:
                choice = filedialog.askopenfilename(
                    title="Select clip.mkv or stack PNG",
                    filetypes=[("Video/PNG", "*.mkv *.png"), ("All", "*.*")],
                    parent=win)
                if not choice:
                    return
                if choice.lower().endswith('.mkv'):
                    clip_mkv = choice
                else:
                    stack_png = choice

            result_var.set("Building meteor trail composite…")
            win.update_idletasks()

            try:
                if stack_png:
                    clip_arr = np.array(
                        Image.open(stack_png).convert('RGB')
                            .resize((disp_w, disp_h), Image.LANCZOS))
                else:
                    # Pipe clip.mkv frames at display resolution, max-composite
                    nbytes = disp_w * disp_h * 3
                    cmd = [_ffp, '-i', clip_mkv,
                           '-vf', f'scale={disp_w}:{disp_h}',
                           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-an', 'pipe:1']
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL)
                    accum = None
                    while True:
                        raw = proc.stdout.read(nbytes)
                        if len(raw) < nbytes:
                            break
                        fr = np.frombuffer(raw, dtype=np.uint8).reshape(
                            disp_h, disp_w, 3).copy()
                        accum = fr if accum is None else np.maximum(accum, fr)
                    try:
                        proc.stdout.close(); proc.kill(); proc.wait()
                    except Exception:
                        pass
                    if accum is None:
                        result_var.set("No frames extracted from clip.mkv.")
                        return
                    clip_arr = accum

                ref_disp = np.array(
                    Image.fromarray(img_arr_ref).resize((disp_w, disp_h), Image.LANCZOS))
                combined = np.maximum(ref_disp, clip_arr)
                _views['ref']    = ImageTk.PhotoImage(Image.fromarray(ref_disp))
                _views['meteor'] = ImageTk.PhotoImage(Image.fromarray(clip_arr))
                _views['both']   = ImageTk.PhotoImage(Image.fromarray(combined))
                toggle_btn.config(state='normal')
                _show_view('both')
                result_var.set("Loaded — 🔁 toggle between Stars / Meteor / Both")
            except Exception as e:
                result_var.set(f"Overlay failed: {e}")

        # Stored display arrays for toggling (populated after overlay_clip_stack)
        _views = {'ref': None, 'meteor': None, 'both': None}
        _view_cycle = ['ref', 'meteor', 'both']
        _cur_view   = ['ref']   # mutable for closure

        def _show_view(name):
            tk_img = _views[name]
            if tk_img is None:
                return
            canvas.create_image(0, 0, anchor='nw', image=tk_img)
            canvas._img_ref = tk_img
            _cur_view[0] = name
            labels = {'ref': 'Stars', 'meteor': 'Meteor', 'both': 'Both'}
            toggle_btn.config(text=f"🔁 View: {labels[name]}")

        def toggle_view():
            idx = _view_cycle.index(_cur_view[0])
            _show_view(_view_cycle[(idx + 1) % len(_view_cycle)])

        toggle_btn = tk.Button(mode_frame, text="🔁 View: Stars",
                               command=toggle_view, state='disabled',
                               bg='#1a2a3a', fg='white', relief='flat',
                               font=('Arial', 8), padx=6)
        toggle_btn.pack(side='right', padx=4)

        tk.Button(mode_frame, text="🌠 Overlay clip stack",
                  command=overlay_clip_stack,
                  bg='#2a1a3a', fg='white', relief='flat',
                  font=('Arial', 8), padx=6).pack(side='right', padx=4)

        # Snap-radius control
        snap_var = tk.IntVar(value=snap_radius)
        tk.Label(mode_frame, text="  Snap r:", bg=BG, fg='#888888',
                 font=('Arial', 8)).pack(side='left', padx=(16, 2))
        tk.Scale(mode_frame, from_=5, to=80, orient='horizontal',
                 variable=snap_var, length=100,
                 bg=BG, fg=FG, troughcolor=ENTRY_BG, highlightthickness=0,
                 font=('Arial', 7), showvalue=True).pack(side='left')
        tk.Label(mode_frame, text="px (stars only)", bg=BG, fg='#888888',
                 font=('Arial', 8)).pack(side='left', padx=2)

        # Star coordinate row (paste from Stellarium → auto-fill sep)
        coord_frame = tk.Frame(win, bg=BG)
        coord_frame.pack(fill='x', padx=8, pady=(2, 0))
        tk.Label(coord_frame, text="Star 1 RA°:", bg=BG, fg='#aaaaaa',
                 font=('Arial', 8)).pack(side='left')
        s1ra_var  = tk.StringVar(value=self.cfg.get('cal_s1ra',  ''))
        tk.Entry(coord_frame, textvariable=s1ra_var, width=9, bg=ENTRY_BG, fg=FG,
                 insertbackground=FG, font=('Arial', 8)).pack(side='left', padx=(2, 4))
        tk.Label(coord_frame, text="Dec°:", bg=BG, fg='#aaaaaa',
                 font=('Arial', 8)).pack(side='left')
        s1dec_var = tk.StringVar(value=self.cfg.get('cal_s1dec', ''))
        tk.Entry(coord_frame, textvariable=s1dec_var, width=9, bg=ENTRY_BG, fg=FG,
                 insertbackground=FG, font=('Arial', 8)).pack(side='left', padx=(2, 12))
        tk.Label(coord_frame, text="Star 2 RA°:", bg=BG, fg='#aaaaaa',
                 font=('Arial', 8)).pack(side='left')
        s2ra_var  = tk.StringVar(value=self.cfg.get('cal_s2ra',  ''))
        tk.Entry(coord_frame, textvariable=s2ra_var, width=9, bg=ENTRY_BG, fg=FG,
                 insertbackground=FG, font=('Arial', 8)).pack(side='left', padx=(2, 4))
        tk.Label(coord_frame, text="Dec°:", bg=BG, fg='#aaaaaa',
                 font=('Arial', 8)).pack(side='left')
        s2dec_var = tk.StringVar(value=self.cfg.get('cal_s2dec', ''))
        tk.Entry(coord_frame, textvariable=s2dec_var, width=9, bg=ENTRY_BG, fg=FG,
                 insertbackground=FG, font=('Arial', 8)).pack(side='left', padx=(2, 8))

        def _parse_coord(s, is_ra=False):
            """Parse RA/Dec from Stellarium in any format:
               - Decimal degrees (RA > 24 → already degrees; RA ≤ 24 → decimal hours → ×15)
               - HMS/DMS: '5h34m32s', '5:34:32', '+22d00m52s', '+22:00:52'
            """
            s = s.strip().replace('°', 'd').replace("'", 'm').replace('"', 's')
            s = s.replace('h', ':').replace('d', ':').replace('m', ':').replace('s', '')
            parts = [p for p in s.replace(' ', ':').split(':') if p]
            if len(parts) == 1:
                val = float(parts[0])
                if is_ra:
                    # >24 → decimal degrees already; ≤24 → decimal hours, convert
                    return val if val > 24 else val * 15.0
                return val
            sign = -1 if parts[0].startswith('-') else 1
            nums = [abs(float(p)) for p in parts]
            val  = nums[0] + (nums[1] / 60 if len(nums) > 1 else 0) + (nums[2] / 3600 if len(nums) > 2 else 0)
            val *= sign
            return val * 15.0 if is_ra else val

        def compute_sep_from_coords():
            try:
                ra1  = _parse_coord(s1ra_var.get(),  is_ra=True)
                dec1 = _parse_coord(s1dec_var.get(), is_ra=False)
                ra2  = _parse_coord(s2ra_var.get(),  is_ra=True)
                dec2 = _parse_coord(s2dec_var.get(), is_ra=False)
                sep  = _angular_sep_deg(ra1, dec1, ra2, dec2)
                sep_var.set(f"{sep:.6f}")
                result_var.set(f"Sep = {sep:.6f}°  ({sep*60:.3f} arcmin)  — ready to Calculate")
                self.cfg.update({'cal_s1ra': s1ra_var.get(), 'cal_s1dec': s1dec_var.get(),
                                 'cal_s2ra': s2ra_var.get(), 'cal_s2dec': s2dec_var.get()})
                _save_config(self.cfg)
            except (ValueError, IndexError):
                result_var.set("Could not parse coordinates. Use decimal° or hh:mm:ss / dd:mm:ss.")

        tk.Button(coord_frame, text="→ Fill Sep", command=compute_sep_from_coords,
                  bg='#2a3a2a', fg='white', relief='flat',
                  font=('Arial', 8), padx=6).pack(side='left')
        tk.Label(coord_frame, text="(decimal °, from Stellarium info panel)",
                 bg=BG, fg='#555566', font=('Arial', 7)).pack(side='left', padx=6)

        # Inputs row
        inp_frame = tk.Frame(win, bg=BG)
        inp_frame.pack(fill='x', padx=8, pady=4)
        tk.Label(inp_frame, text="Star sep (°):", bg=BG, fg=FG,
                 font=('Arial', 9)).pack(side='left')
        sep_var = tk.StringVar()
        tk.Entry(inp_frame, textvariable=sep_var, width=8, bg=ENTRY_BG, fg=FG,
                 insertbackground=FG, font=('Arial', 9)).pack(side='left', padx=(4, 16))
        tk.Label(inp_frame, text="Duration (s):", bg=BG, fg=FG,
                 font=('Arial', 9)).pack(side='left')
        dur_var = tk.StringVar()
        # Try to prefill duration from event metadata
        if self.event_dir:
            meta_path = os.path.join(self.event_dir, 'metadata.json')
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    dets = [d for d in meta.get('detections', []) if d.get('bboxes')]
                    fps  = meta.get('fps', 20.0)
                    if len(dets) >= 2:
                        dur_frames = dets[-1]['frame'] - dets[0]['frame']
                        dur_var.set(f"{dur_frames / fps:.2f}")
                except Exception:
                    pass
        tk.Entry(inp_frame, textvariable=dur_var, width=8, bg=ENTRY_BG, fg=FG,
                 insertbackground=FG, font=('Arial', 9)).pack(side='left', padx=(4, 16))
        calc_btn = tk.Button(inp_frame, text="Calculate ▶", font=('Arial', 9, 'bold'),
                             bg='#1a3a5a', fg='white', relief='flat', padx=10)
        calc_btn.pack(side='left')

        # Results label
        result_var = tk.StringVar(value="Click Star 1 on the image above.")
        tk.Label(win, textvariable=result_var, bg=BG, fg='#aaaaff',
                 font=('Courier', 9), anchor='w', justify='left',
                 wraplength=disp_w).pack(fill='x', padx=10, pady=(2, 8))

        # Marker drawing
        COLORS  = {'star1': 'lime', 'star2': 'lime', 'meteor_start': 'cyan', 'meteor_end': 'cyan'}
        LABELS  = {'star1': '1',    'star2': '2',    'meteor_start': 'S',    'meteor_end': 'E'}
        ORDER   = ['star1', 'star2', 'meteor_start', 'meteor_end']

        def draw_marker(key, cx, cy):
            for item_id in marker_ids.pop(key, []):
                canvas.delete(item_id)
            c = COLORS[key]
            r = 9
            ids = [
                canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline=c, width=2),
                canvas.create_line(cx-r-5, cy, cx+r+5, cy, fill=c),
                canvas.create_line(cx, cy-r-5, cx, cy+r+5, fill=c),
                canvas.create_text(cx+r+5, cy-r, text=LABELS[key], fill=c,
                                   font=('Arial', 8, 'bold')),
            ]
            marker_ids[key] = ids

        def draw_meteor_line():
            for item_id in marker_ids.pop('meteor_line', []):
                canvas.delete(item_id)
            if points['meteor_start'] and points['meteor_end']:
                sx = int(points['meteor_start'][0] * scale)
                sy = int(points['meteor_start'][1] * scale)
                ex = int(points['meteor_end'][0] * scale)
                ey = int(points['meteor_end'][1] * scale)
                marker_ids['meteor_line'] = [
                    canvas.create_line(sx, sy, ex, ey, fill='cyan', width=1, dash=(4, 2))
                ]

        def star_centroid(orig_x, orig_y):
            """Brightness-weighted centroid in a snap_var radius around the click."""
            r = snap_var.get()
            x0 = max(0, orig_x - r);  x1 = min(img_w, orig_x + r)
            y0 = max(0, orig_y - r);  y1 = min(img_h, orig_y + r)
            patch = img_arr[y0:y1, x0:x1].astype(np.float32)
            gray  = patch.mean(axis=2)
            thresh = gray.max() * 0.5
            mask  = gray >= thresh
            if not mask.any():
                return orig_x, orig_y
            ys, xs = np.where(mask)
            w = gray[ys, xs]
            cx = int(round((xs * w).sum() / w.sum())) + x0
            cy = int(round((ys * w).sum() / w.sum())) + y0
            return cx, cy

        def on_click(event):
            mode = mode_var.get()
            orig_x = int(event.x / scale)
            orig_y = int(event.y / scale)
            if mode in ('star1', 'star2'):
                cx, cy = star_centroid(orig_x, orig_y)
                snapped = (cx != orig_x or cy != orig_y)
            else:
                cx, cy = orig_x, orig_y
                snapped = False
            points[mode] = (cx, cy)
            disp_cx = int(cx * scale)
            disp_cy = int(cy * scale)
            draw_marker(mode, disp_cx, disp_cy)
            draw_meteor_line()
            snap_msg = f" (snapped {round(math.dist((orig_x,orig_y),(cx,cy)),1)}px)" if snapped else ""
            result_var.set(f"Set {mode} → pixel ({cx}, {cy}){snap_msg}")
            idx = ORDER.index(mode)
            if idx < len(ORDER) - 1:
                mode_var.set(ORDER[idx + 1])

        canvas.bind('<Button-1>', on_click)

        # Auto-load meteor endpoints from metadata.json
        if self.event_dir:
            meta_path = os.path.join(self.event_dir, 'metadata.json')
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    dets = [d for d in meta.get('detections', []) if d.get('bboxes')]
                    if len(dets) >= 2:
                        def _centroid(d):
                            bx, by, bw, bh = d['bboxes'][0]
                            return int(bx + bw / 2), int(by + bh / 2)
                        points['meteor_start'] = _centroid(dets[0])
                        points['meteor_end']   = _centroid(dets[-1])
                        for key in ('meteor_start', 'meteor_end'):
                            cx = int(points[key][0] * scale)
                            cy = int(points[key][1] * scale)
                            draw_marker(key, cx, cy)
                        draw_meteor_line()
                        mode_var.set('star1')
                        result_var.set(
                            f"Meteor endpoints loaded from metadata "
                            f"(S={points['meteor_start']}, E={points['meteor_end']}).  "
                            f"Now click Star 1 and Star 2."
                        )
                except Exception:
                    pass

        def dist(p1, p2):
            return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

        def calculate():
            if not (points['star1'] and points['star2']):
                result_var.set("Click both star points first.")
                return
            try:
                sep_deg = float(sep_var.get())
            except ValueError:
                result_var.set("Enter star angular separation in degrees.")
                return

            star_px = dist(points['star1'], points['star2'])
            if star_px < 1:
                result_var.set("Star points too close together.")
                return

            px_arcsec = sep_deg * 3600.0 / star_px
            px_deg    = sep_deg / star_px

            lines = [
                f"Pixel scale : {px_arcsec:.2f} arcsec/px  ({px_deg*60:.3f} arcmin/px)",
                f"Stars       : {star_px:.1f} px apart  →  {sep_deg:.4f}°",
            ]

            if points['meteor_start'] and points['meteor_end']:
                m_px  = dist(points['meteor_start'], points['meteor_end'])
                m_deg = m_px * px_deg
                lines.append(f"Meteor trail: {m_px:.1f} px  →  {m_deg:.3f}°  ({m_deg*60:.2f} arcmin)")
                try:
                    dur = float(dur_var.get())
                    if dur > 0:
                        vel_deg  = m_deg / dur
                        vel_amin = vel_deg * 60
                        lines.append(f"Duration    : {dur:.2f} s")
                        lines.append(f"Angular vel : {vel_deg:.3f} °/s  =  {vel_amin:.1f} arcmin/s")
                except ValueError:
                    lines.append("(Enter duration for velocity)")
            else:
                lines.append("(Click meteor start + end for velocity)")

            result_var.set("  |  ".join(lines))
            self._set_result('\n'.join(lines))
            # Persist pixel scale for future reference
            self.cfg['px_arcsec'] = round(px_arcsec, 4)
            _save_config(self.cfg)

        calc_btn.config(command=calculate)

    def _toggle_key_vis(self):
        self._key_entry.config(
            show='' if self._key_entry.cget('show') == '*' else '*')

    def _save_key(self):
        self.cfg['api_key'] = self.key_var.get().strip()
        fov = self.fov_var.get().strip()
        if fov:
            self.cfg['fov_deg'] = fov
        try:
            _save_config(self.cfg)
            self.status_var.set(f"Saved → {CONFIG_PATH}")
            messagebox.showinfo("Saved", f"Config saved to:\n{CONFIG_PATH}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _preview_stretch(self):
        if not self.stack_path or not os.path.exists(self.stack_path):
            messagebox.showerror("No stack", "Select a stacked image first.")
            return
        mode = self.stretch_var.get()
        if mode == 'none':
            os.startfile(self.stack_path)
            return
        self.status_var.set("Generating preview…")
        self.root.update_idletasks()
        preview_path = os.path.join(
            os.path.dirname(self.stack_path),
            f"preview_{mode}_{self.bp_var.get()}.png")
        tmp = _stretch_image(self.stack_path, mode=mode, bp_percentile=self.bp_var.get())
        import shutil
        shutil.copy(tmp, preview_path)
        try:
            os.remove(tmp)
        except Exception:
            pass
        self.status_var.set(f"Preview saved: {os.path.basename(preview_path)}")
        os.startfile(preview_path)

    def _browse_stack(self):
        path = filedialog.askopenfilename(
            title="Select stacked PNG",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")])
        if path:
            self.stack_path = path
            self.stack_lbl.config(text=path, fg=FG)

    def _browse_mask(self):
        path = filedialog.askopenfilename(
            title="Select mask PNG",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")])
        if path:
            self.mask_path = path
            self.mask_lbl.config(text=os.path.basename(path), fg=FG)
            self.cfg['mask_path'] = path
            _save_config(self.cfg)

    def _clear_mask(self):
        self.mask_path = ''
        self.mask_lbl.config(text="None (optional)", fg='#666666')
        self.cfg.pop('mask_path', None)
        _save_config(self.cfg)

    def _browse_event(self):
        path = filedialog.askdirectory(title="Select event folder (contains metadata.json)")
        if path:
            self.event_dir = path
            self.event_lbl.config(text=os.path.basename(path), fg=FG)

    def _set_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _append(self, text):
        def _do():
            self.result_text.config(state='normal')
            self.result_text.insert('end', text)
            self.result_text.config(state='disabled')
            self.result_text.see('end')
        self.root.after(0, _do)

    def _set_result(self, text):
        self.result_text.config(state='normal')
        self.result_text.delete('1.0', 'end')
        self.result_text.insert('end', text)
        self.result_text.config(state='disabled')

    def _clear_results(self):
        def _do():
            self.result_text.config(state='normal')
            self.result_text.delete('1.0', 'end')
            self.result_text.config(state='disabled')
        self.root.after(0, _do)

    # ── Solve ─────────────────────────────────────────────────────────

    def _start_solve(self):
        if not HAS_REQUESTS:
            messagebox.showerror("Missing dependency",
                "Install requests:\n  pip install requests")
            return
        if not HAS_ASTROPY:
            messagebox.showerror("Missing dependency",
                "Install astropy:\n  pip install astropy")
            return
        key = self.key_var.get().strip()
        if not key:
            messagebox.showerror("No API key",
                "Enter your astrometry.net API key.\n"
                "Get one free at: nova.astrometry.net")
            return
        if not self.stack_path or not os.path.exists(self.stack_path):
            messagebox.showerror("No stack", "Select a stacked PNG first.")
            return

        self.solve_btn.config(state='disabled', text="Solving…")
        self._clear_results()
        threading.Thread(target=self._solve_worker, daemon=True).start()

    def _measure_saved_wcs(self):
        if not HAS_ASTROPY:
            messagebox.showerror("Missing dependency",
                "Install astropy:\n  pip install astropy")
            return
        # Auto-detect wcs.fits next to stack
        wcs_path = None
        if self.stack_path:
            candidate = os.path.join(os.path.dirname(self.stack_path), 'wcs.fits')
            if os.path.isfile(candidate):
                wcs_path = candidate
        if not wcs_path:
            wcs_path = filedialog.askopenfilename(
                title="Select WCS FITS file",
                filetypes=[("FITS", "*.fits *.fit"), ("All files", "*.*")])
            if not wcs_path:
                return
        self._wcs_path = wcs_path
        self._clear_results()
        self.status_var.set(f"Measuring with {os.path.basename(wcs_path)}…")
        threading.Thread(target=lambda: self._measure_thread(wcs_path), daemon=True).start()

    def _measure_thread(self, wcs_path):
        try:
            self._measure(wcs_path)
        except Exception as e:
            self._set_status(f"Error: {e}")
            self._append(f"\nERROR: {e}\n")

    def _solve_worker(self):
        try:
            key = self.key_var.get().strip()

            # 1. Login
            self._set_status("Logging in to nova.astrometry.net…")
            r = requests.post(f"{ASTROMETRY_URL}/login",
                              data={'request-json': json.dumps({"apikey": key})},
                              timeout=30)
            resp = r.json()
            if resp.get('status') != 'success':
                raise RuntimeError(f"Login failed: {resp.get('errormessage', resp)}")
            session = resp['session']

            # 2. Upload (optionally mask + stretch first)
            upload_path = self.stack_path
            tmp_stretch = None
            stretch_mode = self.stretch_var.get()
            if stretch_mode != 'none':
                self._set_status("Stretching image…")
                tmp_stretch = _stretch_image(
                    self.stack_path, mode=stretch_mode,
                    bp_percentile=self.bp_var.get())
                upload_path = tmp_stretch
                self.cfg['bp_percentile']  = self.bp_var.get()
                _save_config(self.cfg)
            if self.mask_path and os.path.isfile(self.mask_path):
                self._set_status("Applying mask…")
                import tempfile, numpy as np
                from PIL import Image as _PIL
                import cv2 as _cv2
                img = np.array(_PIL.open(upload_path).convert('RGB'))
                mask = _cv2.imread(self.mask_path, _cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if mask.shape[:2] != img.shape[:2]:
                        mask = _cv2.resize(mask, (img.shape[1], img.shape[0]),
                                           interpolation=_cv2.INTER_NEAREST)
                    img[mask == 0] = 0
                tmp_masked = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                tmp_masked.close()
                _PIL.fromarray(img).save(tmp_masked.name)
                if tmp_stretch:
                    try:
                        os.remove(tmp_stretch)
                    except Exception:
                        pass
                    tmp_stretch = None
                upload_path = tmp_masked.name
                tmp_stretch = tmp_masked.name
            self._set_status("Uploading stack…")
            upload_params = {
                'session': session,
                'publicly_visible': 'n',
                'allow_modifications': 'd',
            }
            fov_str = self.fov_var.get().strip()
            if fov_str:
                try:
                    fov = float(fov_str)
                    upload_params['scale_units'] = 'degwidth'
                    upload_params['scale_lower'] = round(fov * 0.8, 2)
                    upload_params['scale_upper'] = round(fov * 1.2, 2)
                except ValueError:
                    pass
            ra_str  = self.ra_var.get().strip()
            dec_str = self.dec_var.get().strip()
            rad_str = self.radius_var.get().strip()
            if ra_str and dec_str:
                try:
                    upload_params['center_ra']  = float(ra_str)
                    upload_params['center_dec'] = float(dec_str)
                    upload_params['radius']     = float(rad_str) if rad_str else 30.0
                except ValueError:
                    pass
            hint_info = []
            if 'scale_lower' in upload_params:
                hint_info.append(f"FOV {upload_params['scale_lower']}–{upload_params['scale_upper']}°")
            if 'center_ra' in upload_params:
                hint_info.append(f"center RA {upload_params['center_ra']:.1f} Dec {upload_params['center_dec']:+.1f}")
            self._set_status("Uploading stack" + (f" ({', '.join(hint_info)})" if hint_info else "") + "…")
            with open(upload_path, 'rb') as f:
                r = requests.post(
                    f"{ASTROMETRY_URL}/upload",
                    data={'request-json': json.dumps(upload_params)},
                    files={'file': f},
                    timeout=120)
            resp = r.json()
            if tmp_stretch:
                try:
                    os.remove(tmp_stretch)
                except Exception:
                    pass
            if resp.get('status') != 'success':
                raise RuntimeError(f"Upload failed: {resp}")
            sub_id = resp['subid']
            self._set_status(f"Uploaded (submission #{sub_id}) — waiting for solver…")

            # 3. Wait for job ID
            job_id = None
            for _ in range(40):
                time.sleep(5)
                r = requests.get(f"{ASTROMETRY_URL}/submissions/{sub_id}", timeout=15)
                jobs = r.json().get('jobs', [])
                if jobs and jobs[0]:
                    job_id = jobs[0]
                    break
            if not job_id:
                raise RuntimeError("Timed out waiting for solver job to start.")
            self._set_status(f"Job #{job_id} assigned — solving…")

            # 4. Wait for solve
            for attempt in range(180):
                time.sleep(5)
                r = requests.get(f"{ASTROMETRY_URL}/jobs/{job_id}", timeout=15)
                status = r.json().get('status', '')
                elapsed = attempt * 5
                self._set_status(
                    f"Job #{job_id} — {status}  ({elapsed}s elapsed)")
                if status == 'success':
                    break
                if status == 'failure':
                    raise RuntimeError(
                        "Plate solve failed — try a brighter/sharper stack, "
                        "more frames, or set a FOV hint.")
            else:
                raise RuntimeError("Solve timed out after 15 minutes.")

            # 5. Download WCS
            self._set_status("Downloading WCS…")
            r = requests.get(
                f"https://nova.astrometry.net/wcs_file/{job_id}", timeout=30)
            wcs_path = os.path.join(os.path.dirname(self.stack_path), 'wcs.fits')
            with open(wcs_path, 'wb') as f:
                f.write(r.content)
            self._wcs_path = wcs_path
            self._set_status("WCS saved — computing measurement…")

            # 6. Measure
            self._measure(wcs_path)

        except Exception as e:
            self._set_status(f"Error: {e}")
            self._append(f"\nERROR: {e}\n")
        finally:
            self.root.after(0, lambda: self.solve_btn.config(
                state='normal', text="🔭  Solve & Measure"))

    # ── Measurement ───────────────────────────────────────────────────

    def _measure(self, wcs_path):
        hdul = fits.open(wcs_path)
        wcs = WCS(hdul[0].header)
        hdul.close()

        # Auto-save center coords from WCS for future solves
        try:
            ny, nx = wcs.array_shape or (1080, 1920)
            cra, cdec = wcs.pixel_to_world_values(nx / 2, ny / 2)
            self.cfg['center_ra']     = round(float(cra), 4)
            self.cfg['center_dec']    = round(float(cdec), 4)
            self.cfg['center_radius'] = '15'
            _save_config(self.cfg)
            self.root.after(0, lambda: self.ra_var.set(str(self.cfg['center_ra'])))
            self.root.after(0, lambda: self.dec_var.set(str(self.cfg['center_dec'])))
            self.root.after(0, lambda: self.radius_var.set('15'))
        except Exception:
            pass

        # Pixel scale from WCS
        px_arcsec = None
        try:
            psm = wcs.pixel_scale_matrix          # degrees/pixel, 2×2
            px_deg = math.sqrt(abs(
                psm[0, 0] * psm[1, 1] - psm[0, 1] * psm[1, 0]))
            px_arcsec = px_deg * 3600.0
        except Exception:
            pass

        lines = [f"WCS file: {os.path.basename(wcs_path)}\n"]
        if px_arcsec is not None:
            lines.append(f"Pixel scale: {px_arcsec:.2f} arcsec/px"
                         f"  ({px_arcsec / 60:.3f} arcmin/px)\n")
        lines.append("\n")

        if self.event_dir:
            meta_path = os.path.join(self.event_dir, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)

                dets = [d for d in meta.get('detections', []) if d.get('bboxes')]
                fps  = meta.get('fps', 20.0)

                if len(dets) >= 2:
                    def centroid(d):
                        bx, by, bw, bh = d['bboxes'][0]
                        return bx + bw / 2, by + bh / 2

                    cx1, cy1 = centroid(dets[0])
                    cx2, cy2 = centroid(dets[-1])
                    frame1   = dets[0]['frame']
                    frame2   = dets[-1]['frame']

                    ra1, dec1 = wcs.pixel_to_world_values(cx1, cy1)
                    ra2, dec2 = wcs.pixel_to_world_values(cx2, cy2)

                    sep  = _angular_sep_deg(ra1, dec1, ra2, dec2)
                    dt   = (frame2 - frame1) / fps
                    vel  = sep / dt if dt > 0 else 0.0

                    lines += [
                        f"Start  pixel ({cx1:7.1f}, {cy1:7.1f})  →"
                        f"  RA {ra1:9.4f}°  Dec {dec1:+9.4f}°  [frame {frame1}]\n",
                        f"End    pixel ({cx2:7.1f}, {cy2:7.1f})  →"
                        f"  RA {ra2:9.4f}°  Dec {dec2:+9.4f}°  [frame {frame2}]\n",
                        f"\n",
                        f"Angular separation :  {sep:.4f}°"
                        f"  ({sep * 60:.2f} arcmin)\n",
                        f"Duration           :  {dt:.3f} s"
                        f"  ({frame2 - frame1} frames @ {fps:.1f} fps)\n",
                        f"Angular velocity   :  {vel:.4f}°/s"
                        f"  ({vel * 60:.2f} arcmin/s)\n",
                    ]
                elif len(dets) == 1:
                    lines.append("Only one detection frame — cannot measure velocity.\n")
                else:
                    lines.append("No detections with bounding boxes in metadata.json.\n")
            else:
                lines.append("metadata.json not found — WCS saved but no measurement.\n")
        else:
            lines.append("No event folder selected — WCS saved but no measurement.\n")

        self._append("".join(lines))
        self._set_status("Done.")

    # ── Clipboard ─────────────────────────────────────────────────────

    def _copy_results(self):
        text = self.result_text.get('1.0', 'end').strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_var.set("Copied to clipboard.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stack', default=None, help="Path to stacked PNG")
    parser.add_argument('--event', default=None, help="Path to event folder")
    parser.add_argument('--config-dir', default=None, help="Directory for config file")
    args = parser.parse_args()

    root = tk.Tk()
    StreakerAstro(root, stack_path=args.stack, event_dir=args.event)
    root.mainloop()


if __name__ == '__main__':
    import traceback
    _log = os.path.join(os.path.expanduser('~'), 'streaker_astro_crash.log')
    try:
        main()
    except Exception:
        with open(_log, 'w') as _f:
            _f.write(traceback.format_exc())
        raise
