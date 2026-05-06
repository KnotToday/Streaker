# ------------------------------------------------------------------------------
# Script Name:     StreakerCompare.py
# Description:     Side-by-side detection parameter comparison GUI.
#                  Runs two independent detection passes on the same MKV clip
#                  and presents matched / unmatched events in a split view.
# ------------------------------------------------------------------------------

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import json
import subprocess
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------
# Import shared detection primitives from StreakerDetect
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from StreakerDetect import (
    process_frame, TrackManager, AdaptiveCloudDetector,
    make_thumbnail, FFMPEG_PATH, THUMB_W, THUMB_H, INFO_H,
)
try:
    from platform_utils import HWACCEL_ARGS
except ImportError:
    HWACCEL_ARGS = []

# ---------------------------------------------------------------------------
# Theme colours
# ---------------------------------------------------------------------------
BG        = '#111111'
BG2       = '#1a1a1a'
FG        = '#dddddd'
FG2       = '#888888'
ENTRY_BG  = '#252525'

COL_BOTH   = '#0d2a0d'
COL_ONLY_A = '#0d0d2a'
COL_ONLY_B = '#2a0d0d'

HL_BOTH = '#33aa33'
HL_A    = '#3333aa'
HL_B    = '#aa3333'

# ---------------------------------------------------------------------------
# Detection parameter definitions
# (key, label, lo, hi, resolution, default)
# ---------------------------------------------------------------------------
PARAM_DEFS = [
    ('threshold',    'MOG2 Thresh',   10,    150,    1,   40),
    ('history',      'History',      100,   1000,   50,  500),
    ('min_area',     'Min Area',      10,    500,   10,  120),
    ('max_area',     'Max Area',     500,  20000,  100, 1400),
    ('min_aspect',   'Aspect',       1.0,    8.0,  0.1,  2.0),
    ('max_track',    'Max Track',      1,     30,    1,   10),
    ('warmup',       'Warmup',        50,    500,   50,  200),
    ('pre_buffer',   'Pre Buf',        5,    120,    5,   30),
    ('post_buffer',  'Post Buf',       5,    120,    5,   30),
    ('min_bright',   'Min Bright',     0,    255,    5,    0),
    ('min_move',     'Min Move',       0,     50,    1,    0),
    ('min_travel',   'Min Travel',     0,    100,    5,    0),
    ('cloud_thresh', 'Cloud Sens',    20,    200,    5,   40),
    ('cloud_ratio',  'Cloud Ratio', 0.01,   0.50, 0.01, 0.15),
    ('scale',        'Scale',       0.25,    1.0, 0.25,  0.5),
]

# Thumbnail dimensions for the comparison grid
GRID_THUMB_W = 220
GRID_THUMB_H = 150

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
                           'streaker_config.json')

# ---------------------------------------------------------------------------
# Detection pass — runs MOG2 on an MKV and returns a list of event dicts
# ---------------------------------------------------------------------------

def run_detection_pass(mkv_path, mask_path, params, progress_cb=None, stop_event=None):
    """
    Run a full MOG2 detection pass on *mkv_path*.

    Returns a list of event dicts, each with:
        start_frame, end_frame, fps,
        frames_bgr  (list of BGR numpy arrays — full-resolution),
        thumbnail   (BGR numpy array, GRID_THUMB_W × GRID_THUMB_H)
    """
    events = []

    cap = cv2.VideoCapture(mkv_path)
    if not cap.isOpened():
        return events
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh_px  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
    cap.release()

    # Load and scale mask once
    mask = None
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    scale      = float(params.get('scale', 0.5))
    area_scale = scale * scale

    work_params = dict(params)
    work_params['min_area'] = max(1, int(work_params['min_area'] * area_scale))
    work_params['max_area'] = max(1, int(work_params['max_area'] * area_scale))
    work_params['min_move_scaled'] = work_params.get('min_move', 0) * scale
    # Detection runs at scale; tell process_frame we are already at scale=1
    work_params['scale'] = 1.0

    mask_small = None
    if mask is not None:
        if scale != 1.0:
            mask_small = cv2.resize(mask,
                                    (int(fw * scale), int(fh_px * scale)),
                                    interpolation=cv2.INTER_NEAREST)
        else:
            mask_small = mask

    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=int(params.get('history', 500)),
        varThreshold=float(params.get('threshold', 40)),
        detectShadows=False)
    tracker       = TrackManager(max_frames=int(params.get('max_track', 10)))
    kernel        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cloud_det     = AdaptiveCloudDetector(window=200)

    warmup      = int(params.get('warmup', 200))
    pre_buf_sz  = int(params.get('pre_buffer', 30))
    post_buf_sz = int(params.get('post_buffer', 30))

    # FFmpeg pipe — grayscale
    ffmpeg_cmd = [FFMPEG_PATH] + HWACCEL_ARGS + [
        '-i', mkv_path,
        '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1',
    ]
    proc = subprocess.Popen(ffmpeg_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    frame_bytes = fw * fh_px

    pre_buffer = deque(maxlen=pre_buf_sz)
    pending    = []        # list of (frame_idx, gray_full, count, bboxes)
    post_cd    = 0
    frame_idx  = 0

    try:
        while True:
            if stop_event and stop_event.is_set():
                break

            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break

            gray_full = np.frombuffer(raw, dtype=np.uint8).reshape(fh_px, fw)

            # Downscale for detection
            if scale != 1.0:
                small = cv2.resize(gray_full,
                                   (int(fw * scale), int(fh_px * scale)),
                                   interpolation=cv2.INTER_AREA)
            else:
                small = gray_full

            count, bboxes, _overlay, _cloudy, _fg = process_frame(
                small, mog2, tracker, mask_small, kernel, work_params, cloud_det)

            if progress_cb and frame_idx % 30 == 0:
                progress_cb(frame_idx, total)

            # Warmup — still feed mog2 but skip detection logic
            if frame_idx < warmup:
                frame_idx += 1
                continue

            entry = (frame_idx, gray_full, count, bboxes)

            if count > 0:
                if post_cd == 0:
                    pending.extend(list(pre_buffer))
                    pre_buffer.clear()
                pending.append(entry)
                post_cd = post_buf_sz
            elif post_cd > 0:
                pending.append(entry)
                post_cd -= 1
                if post_cd == 0:
                    ev = _build_event(pending, fps)
                    if ev:
                        events.append(ev)
                    pending = []
            else:
                pre_buffer.append(entry)

            frame_idx += 1

    finally:
        proc.stdout.close()
        proc.wait()

    # Flush any trailing event
    if pending:
        ev = _build_event(pending, fps)
        if ev:
            events.append(ev)

    return events


def _build_event(pending, fps):
    """Convert a list of (frame_idx, gray, count, bboxes) tuples into an event dict."""
    if not pending:
        return None
    start_frame = pending[0][0]
    end_frame   = pending[-1][0]

    # Collect full-res BGR frames for the split player
    frames_bgr = []
    for (_fidx, gray, _c, _b) in pending:
        frames_bgr.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    # Build thumbnail from grayscale frames (max-blend composite)
    grays = [e[1] for e in pending]
    thumb_full = make_thumbnail(grays)  # returns THUMB_W × THUMB_H BGR

    # Resize thumbnail to grid size
    if thumb_full is not None:
        thumb = cv2.resize(thumb_full, (GRID_THUMB_W, GRID_THUMB_H),
                           interpolation=cv2.INTER_AREA)
    else:
        thumb = np.zeros((GRID_THUMB_H, GRID_THUMB_W, 3), dtype=np.uint8)

    return {
        'start_frame': start_frame,
        'end_frame':   end_frame,
        'fps':         fps,
        'frames_bgr':  frames_bgr,
        'thumbnail':   thumb,
    }


def match_events(events_a, events_b, window=60):
    """
    Match events from config A to events from config B by frame-range overlap.

    Returns a list of tuples: (ev_a_or_None, ev_b_or_None)
    Each tuple represents one row in the comparison grid.
    """
    used_b = set()
    rows   = []

    for ea in events_a:
        best_idx   = -1
        best_overlap = 0
        for i, eb in enumerate(events_b):
            if i in used_b:
                continue
            # Overlap = intersection of [start, end] ranges extended by window
            lo = max(ea['start_frame'] - window, eb['start_frame'] - window)
            hi = min(ea['end_frame']   + window, eb['end_frame']   + window)
            overlap = hi - lo
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx     = i
        if best_idx >= 0:
            rows.append((ea, events_b[best_idx]))
            used_b.add(best_idx)
        else:
            rows.append((ea, None))

    for i, eb in enumerate(events_b):
        if i not in used_b:
            rows.append((None, eb))

    return rows


# ===========================================================================
# ParamPanel — scrollable LabelFrame with Scale/Combobox widgets
# ===========================================================================

class ParamPanel(tk.LabelFrame):
    """
    Scrollable parameter panel for one detection configuration.
    Contains one widget per entry in PARAM_DEFS.
    """

    def __init__(self, parent, title, **kw):
        super().__init__(parent, text=title,
                         bg=BG2, fg=FG, font=('Arial', 9, 'bold'),
                         relief='flat', bd=1, **kw)

        # Scrollable canvas inside the frame
        self._canvas = tk.Canvas(self, bg=BG2, highlightthickness=0,
                                 width=230)
        sb = ttk.Scrollbar(self, orient='vertical',
                           command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self._canvas.pack(side='left', fill='both', expand=True)

        self._inner = tk.Frame(self._canvas, bg=BG2)
        self._win_id = self._canvas.create_window(
            (0, 0), window=self._inner, anchor='nw')
        self._inner.bind('<Configure>', self._on_inner_resize)
        self._canvas.bind('<Configure>', self._on_canvas_resize)
        self._canvas.bind('<Enter>',
                          lambda _e: self._canvas.bind_all(
                              '<MouseWheel>', self._on_mousewheel))
        self._canvas.bind('<Leave>',
                          lambda _e: self._canvas.unbind_all('<MouseWheel>'))

        # Build one row per param
        self._vars = {}   # key → tk variable
        self._build_widgets()

    def _on_inner_resize(self, _e):
        self._canvas.configure(scrollregion=self._canvas.bbox('all'))

    def _on_canvas_resize(self, e):
        self._canvas.itemconfig(self._win_id, width=e.width)

    def _on_mousewheel(self, e):
        self._canvas.yview_scroll(-1 * (e.delta // 120), 'units')

    def _build_widgets(self):
        lp = dict(bg=BG2, fg=FG2, font=('Arial', 7))
        for (key, label, lo, hi, res, default) in PARAM_DEFS:
            row = tk.Frame(self._inner, bg=BG2)
            row.pack(fill='x', padx=4, pady=1)
            tk.Label(row, text=label, width=11, anchor='w', **lp).pack(side='left')

            if key == 'scale':
                var = tk.DoubleVar(value=default)
                cb  = ttk.Combobox(row, textvariable=var,
                                   values=[1.0, 0.75, 0.5, 0.25],
                                   width=6, state='readonly')
                cb.pack(side='left', padx=2)
            else:
                if isinstance(res, float) or isinstance(lo, float):
                    var = tk.DoubleVar(value=default)
                else:
                    var = tk.IntVar(value=default)
                sc = tk.Scale(row, from_=lo, to=hi, resolution=res,
                              variable=var, orient='horizontal',
                              bg=BG2, fg=FG, troughcolor='#333333',
                              highlightthickness=0, length=140,
                              showvalue=True, font=('Arial', 7))
                sc.pack(side='left', padx=2)

            self._vars[key] = var

    def get_params(self):
        """Return a plain dict of all current parameter values."""
        result = {}
        for key, var in self._vars.items():
            result[key] = var.get()
        return result

    def set_params(self, d):
        """Set parameter values from a dict (missing keys are left unchanged)."""
        for key, var in self._vars.items():
            if key in d:
                try:
                    var.set(d[key])
                except Exception:
                    pass


# ===========================================================================
# CompareGrid — scrollable list of event pair rows
# ===========================================================================

class CompareGrid(tk.Frame):
    """
    Scrollable list of event rows.  Each row shows:
      left thumbnail | centre label | right thumbnail
    Row background reflects match status:
      BOTH → COL_BOTH,  ONLY A → COL_ONLY_A,  ONLY B → COL_ONLY_B
    """

    def __init__(self, parent, on_select, **kw):
        super().__init__(parent, bg=BG2, **kw)
        self.on_select = on_select

        self._canvas = tk.Canvas(self, bg=BG2, highlightthickness=0,
                                 width=490)
        sb = ttk.Scrollbar(self, orient='vertical',
                           command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self._canvas.pack(side='left', fill='both', expand=True)

        self._inner = tk.Frame(self._canvas, bg=BG2)
        self._win_id = self._canvas.create_window(
            (0, 0), window=self._inner, anchor='nw')
        self._inner.bind('<Configure>',
                         lambda _e: self._canvas.configure(
                             scrollregion=self._canvas.bbox('all')))
        self._canvas.bind('<Configure>',
                          lambda e: self._canvas.itemconfig(
                              self._win_id, width=e.width))
        self._canvas.bind('<Enter>',
                          lambda _e: self._canvas.bind_all(
                              '<MouseWheel>', self._scroll))
        self._canvas.bind('<Leave>',
                          lambda _e: self._canvas.unbind_all('<MouseWheel>'))

        self._photo_refs = []   # prevent GC

    def _scroll(self, e):
        self._canvas.yview_scroll(-1 * (e.delta // 120), 'units')

    def clear(self):
        for w in self._inner.winfo_children():
            w.destroy()
        self._photo_refs.clear()

    def populate(self, rows):
        """
        rows: list of (ev_a_or_None, ev_b_or_None)
        """
        self.clear()
        for idx, (ea, eb) in enumerate(rows):
            self._add_row(idx, ea, eb)
        self._canvas.yview_moveto(0)

    def _add_row(self, idx, ea, eb):
        if ea and eb:
            bg_col = COL_BOTH
            hl_col = HL_BOTH
            label  = 'BOTH'
        elif ea:
            bg_col = COL_ONLY_A
            hl_col = HL_A
            label  = 'ONLY A'
        else:
            bg_col = COL_ONLY_B
            hl_col = HL_B
            label  = 'ONLY B'

        row_frame = tk.Frame(self._inner, bg=bg_col,
                             highlightbackground=hl_col,
                             highlightthickness=1,
                             cursor='hand2')
        row_frame.pack(fill='x', padx=4, pady=2)

        # Left thumbnail (Config A)
        left_lbl = self._make_thumb_label(row_frame, ea, bg_col)
        left_lbl.pack(side='left', padx=2, pady=2)

        # Centre info
        info_f = tk.Frame(row_frame, bg=bg_col, width=70)
        info_f.pack(side='left', fill='y', expand=False)
        info_f.pack_propagate(False)

        tk.Label(info_f, text=f'#{idx+1}',
                 bg=bg_col, fg=hl_col,
                 font=('Arial', 9, 'bold')).pack(pady=(6, 0))
        tk.Label(info_f, text=label,
                 bg=bg_col, fg=hl_col,
                 font=('Arial', 7, 'bold')).pack()

        if ea:
            tk.Label(info_f,
                     text=f'A:{ea["start_frame"]}-{ea["end_frame"]}',
                     bg=bg_col, fg=FG2,
                     font=('Courier', 6)).pack()
        if eb:
            tk.Label(info_f,
                     text=f'B:{eb["start_frame"]}-{eb["end_frame"]}',
                     bg=bg_col, fg=FG2,
                     font=('Courier', 6)).pack()

        # Right thumbnail (Config B)
        right_lbl = self._make_thumb_label(row_frame, eb, bg_col)
        right_lbl.pack(side='left', padx=2, pady=2)

        # Click binding — propagate to all children
        def _click(_e, a=ea, b=eb):
            self.on_select({'frames_a': a['frames_bgr'] if a else [],
                            'frames_b': b['frames_bgr'] if b else [],
                            'start_a':  a['start_frame'] if a else 0,
                            'start_b':  b['start_frame'] if b else 0,
                            'fps':      (a or b)['fps']})
        for widget in (row_frame, info_f):
            widget.bind('<Button-1>', _click)
        for child in list(row_frame.winfo_children()) + list(info_f.winfo_children()):
            child.bind('<Button-1>', _click)
        left_lbl.bind('<Button-1>', _click)
        right_lbl.bind('<Button-1>', _click)

    def _make_thumb_label(self, parent, ev, bg_col):
        if ev and ev.get('thumbnail') is not None:
            thumb_bgr = ev['thumbnail']
        else:
            thumb_bgr = np.zeros((GRID_THUMB_H, GRID_THUMB_W, 3), dtype=np.uint8)
            cv2.putText(thumb_bgr, 'No event',
                        (40, GRID_THUMB_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (80, 80, 80), 1, cv2.LINE_AA)

        img = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)))
        self._photo_refs.append(img)
        lbl = tk.Label(parent, image=img, bg=bg_col)
        lbl.image = img
        return lbl


# ===========================================================================
# SplitPlayer — two synchronized canvases for event playback
# ===========================================================================

class SplitPlayer(tk.Frame):
    """
    Side-by-side synchronized player for two event clip sequences.
    """

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)

        self._frames_a  = []
        self._frames_b  = []
        self._idx       = 0
        self._playing   = False
        self._speed_ms  = 80
        self._sync      = True
        self._after_id  = None
        self._photo_a   = None
        self._photo_b   = None

        self._build()

    def _build(self):
        # Header labels
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill='x')
        self._lbl_a = tk.Label(hdr, text='Config A',
                               bg=BG, fg=HL_A,
                               font=('Arial', 9, 'bold'))
        self._lbl_a.pack(side='left', expand=True)
        self._lbl_b = tk.Label(hdr, text='Config B',
                               bg=BG, fg=HL_B,
                               font=('Arial', 9, 'bold'))
        self._lbl_b.pack(side='right', expand=True)

        # Canvas area
        cv_row = tk.Frame(self, bg='#080808')
        cv_row.pack(fill='both', expand=True)

        self._cv_a = tk.Canvas(cv_row, bg='black', highlightthickness=0)
        self._cv_a.pack(side='left', fill='both', expand=True)

        divider = tk.Frame(cv_row, bg='#333333', width=3)
        divider.pack(side='left', fill='y')
        divider.pack_propagate(False)

        self._cv_b = tk.Canvas(cv_row, bg='black', highlightthickness=0)
        self._cv_b.pack(side='right', fill='both', expand=True)

        # Controls
        ctrl = tk.Frame(self, bg=BG2)
        ctrl.pack(fill='x')

        btn_kw = dict(bg='#2a2a2a', fg=FG, relief='flat', padx=6, pady=2)
        tk.Button(ctrl, text='|◀',
                  command=self._go_first, **btn_kw).pack(side='left', padx=2, pady=3)
        tk.Button(ctrl, text='◀',
                  command=lambda: self._step(-1), **btn_kw).pack(side='left', padx=2)
        self._play_btn = tk.Button(ctrl, text='▶ Play',
                                   command=self._toggle_play, **btn_kw)
        self._play_btn.pack(side='left', padx=2)
        tk.Button(ctrl, text='▶|',
                  command=self._go_last, **btn_kw).pack(side='left', padx=2)

        tk.Label(ctrl, text='Speed:', bg=BG2, fg=FG2,
                 font=('Arial', 7)).pack(side='left', padx=(8, 2))
        self._speed_var = tk.IntVar(value=self._speed_ms)
        tk.Scale(ctrl, from_=10, to=500, orient='horizontal',
                 variable=self._speed_var, length=100, showvalue=False,
                 bg=BG2, fg=FG2, troughcolor='#333333',
                 highlightthickness=0,
                 command=lambda v: setattr(self, '_speed_ms', int(v))
                 ).pack(side='left')

        self._sync_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text='Sync', variable=self._sync_var,
                       bg=BG2, fg=FG2, selectcolor='#333333',
                       font=('Arial', 8),
                       activebackground=BG2,
                       command=lambda: setattr(self, '_sync', self._sync_var.get())
                       ).pack(side='left', padx=6)

        # Scrubber
        self._scrub_var = tk.DoubleVar(value=0)
        self._scrubber  = ttk.Scale(ctrl, from_=0, to=1,
                                    variable=self._scrub_var,
                                    orient='horizontal',
                                    command=self._on_scrub)
        self._scrubber.pack(side='left', fill='x', expand=True, padx=6)

        self._status_var = tk.StringVar(value='No event loaded')
        tk.Label(ctrl, textvariable=self._status_var,
                 bg=BG2, fg=FG2,
                 font=('Courier', 7), width=18).pack(side='right', padx=4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, frames_a, frames_b, start_a=0, start_b=0, fps=20.0):
        """Load new frame sequences and reset playback."""
        self._stop_playback()
        self._frames_a = list(frames_a)
        self._frames_b = list(frames_b)
        self._fps      = fps
        self._idx      = 0

        n = max(len(self._frames_a), len(self._frames_b), 1)
        self._scrubber.configure(to=n - 1)

        # Update header labels with frame ranges
        self._lbl_a.config(
            text=f'Config A  (start={start_a}  {len(frames_a)} frames)')
        self._lbl_b.config(
            text=f'Config B  (start={start_b}  {len(frames_b)} frames)')

        self._goto(0)

    def _max_frames(self):
        if self._sync:
            return max(len(self._frames_a), len(self._frames_b))
        return max(len(self._frames_a), len(self._frames_b))

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------

    def _toggle_play(self):
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        self._playing = True
        self._play_btn.config(text='⏸ Pause')
        self._play_loop()

    def _stop_playback(self):
        self._playing = False
        self._play_btn.config(text='▶ Play')
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None

    def _play_loop(self):
        if not self._playing:
            return
        n = self._max_frames()
        if n == 0:
            self._stop_playback()
            return
        next_idx = self._idx + 1
        if next_idx >= n:
            next_idx = 0   # loop
        self._goto(next_idx)
        self._after_id = self.after(self._speed_ms, self._play_loop)

    def _step(self, d):
        n = self._max_frames()
        if n == 0:
            return
        self._goto((self._idx + d) % n)

    def _go_first(self):
        self._goto(0)

    def _go_last(self):
        n = self._max_frames()
        if n > 0:
            self._goto(n - 1)

    def _on_scrub(self, v):
        self._goto(int(float(v)))

    def _goto(self, idx):
        n = self._max_frames()
        if n == 0:
            return
        self._idx = max(0, min(idx, n - 1))

        # Update scrubber without triggering _on_scrub re-entrantly
        self._scrubber.configure(command='')
        self._scrub_var.set(self._idx)
        self._scrubber.configure(command=self._on_scrub)

        fa = self._get_frame(self._frames_a, self._idx)
        fb = self._get_frame(self._frames_b, self._idx)

        self._render(self._cv_a, fa)
        self._render(self._cv_b, fb)

        na = len(self._frames_a)
        nb = len(self._frames_b)
        self._status_var.set(
            f'A:{min(self._idx, na-1) if na else "-"}/{max(na-1,0)}'
            f'  B:{min(self._idx, nb-1) if nb else "-"}/{max(nb-1,0)}')

    @staticmethod
    def _get_frame(frames, idx):
        if not frames:
            return None
        return frames[min(idx, len(frames) - 1)]

    def _render(self, canvas, frame_bgr):
        cw = max(canvas.winfo_width(),  200)
        ch = max(canvas.winfo_height(), 150)

        if frame_bgr is None:
            # Black frame with placeholder text
            canvas.delete('all')
            canvas.create_text(cw // 2, ch // 2,
                               text='No event', fill='#444444',
                               font=('Arial', 11))
            return

        fh, fw = frame_bgr.shape[:2]
        sc = min(cw / max(fw, 1), ch / max(fh, 1))
        nw = max(1, int(fw * sc))
        nh = max(1, int(fh * sc))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        img = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))

        x_off = (cw - nw) // 2
        y_off = (ch - nh) // 2
        canvas.delete('all')
        canvas.create_image(x_off, y_off, anchor='nw', image=img)

        # Keep reference on canvas widget to prevent GC
        if canvas is self._cv_a:
            self._photo_a = img
        else:
            self._photo_b = img


# ===========================================================================
# StreakerCompareApp — main application
# ===========================================================================

class StreakerCompareApp:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Streaker Compare')
        self.root.geometry('1400x900')
        self.root.configure(bg=BG)
        self.root.minsize(900, 600)

        self._mkv_path  = tk.StringVar()
        self._mask_path = tk.StringVar()

        self._result_q  = queue.Queue()
        self._stop_evt  = threading.Event()
        self._running   = False

        self._build_ui()
        self._load_config_defaults()
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Row 1: source / mask pickers + run button ──────────────────
        row1 = tk.Frame(self.root, bg=BG2)
        row1.pack(fill='x', side='top', padx=4, pady=3)

        tk.Label(row1, text='STREAKER COMPARE',
                 bg=BG2, fg=FG,
                 font=('Arial', 11, 'bold')).pack(side='left', padx=8)

        self._add_file_picker(row1, 'Source MKV', self._mkv_path,
                              lambda: self._browse_file(
                                  self._mkv_path,
                                  [('MKV files', '*.mkv'),
                                   ('All files', '*.*')]))
        self._add_file_picker(row1, 'Mask (opt)', self._mask_path,
                              lambda: self._browse_file(
                                  self._mask_path,
                                  [('PNG files', '*.png'),
                                   ('All files', '*.*')]))

        self._run_btn = tk.Button(row1, text='▶ RUN COMPARISON',
                                  command=self._on_run,
                                  bg='#006600', fg='white',
                                  font=('Arial', 10, 'bold'),
                                  relief='flat', padx=12, pady=4)
        self._run_btn.pack(side='left', padx=12)

        self._stop_btn = tk.Button(row1, text='■ STOP',
                                   command=self._on_stop,
                                   bg='#660000', fg='white',
                                   font=('Arial', 10, 'bold'),
                                   relief='flat', padx=12, pady=4,
                                   state='disabled')
        self._stop_btn.pack(side='left', padx=2)

        # ── Row 2: param panels + copy button ──────────────────────────
        row2 = tk.Frame(self.root, bg=BG)
        row2.pack(fill='x', side='top', padx=4, pady=2)

        self._panel_a = ParamPanel(row2, 'Config A', height=260)
        self._panel_a.pack(side='left', fill='both', expand=True, padx=(0, 2))

        btn_col = tk.Frame(row2, bg=BG, width=120)
        btn_col.pack(side='left', fill='y', padx=4)
        btn_col.pack_propagate(False)

        btn_kw = dict(bg='#2a2a2a', fg=FG, relief='flat',
                      font=('Arial', 8), pady=4, padx=6)
        tk.Button(btn_col, text='Copy A → B',
                  command=self._copy_a_to_b,
                  **btn_kw).pack(fill='x', pady=3, padx=4)
        tk.Button(btn_col, text='Load A',
                  command=lambda: self._load_config(self._panel_a),
                  **btn_kw).pack(fill='x', pady=3, padx=4)
        tk.Button(btn_col, text='Load B',
                  command=lambda: self._load_config(self._panel_b),
                  **btn_kw).pack(fill='x', pady=3, padx=4)

        self._panel_b = ParamPanel(row2, 'Config B', height=260)
        self._panel_b.pack(side='left', fill='both', expand=True, padx=(2, 0))

        # ── Status / summary / diff ─────────────────────────────────────
        self._status_var  = tk.StringVar(value='Ready')
        self._summary_var = tk.StringVar(value='')
        self._diff_var    = tk.StringVar(value='')

        status_bar = tk.Frame(self.root, bg='#0a0a0a')
        status_bar.pack(fill='x', side='top')
        tk.Label(status_bar, textvariable=self._status_var,
                 bg='#0a0a0a', fg='#aaaaaa',
                 font=('Arial', 8), anchor='w').pack(fill='x', padx=6, pady=1)
        tk.Label(status_bar, textvariable=self._summary_var,
                 bg='#0a0a0a', fg='#aaffaa',
                 font=('Courier', 9, 'bold'), anchor='w').pack(fill='x', padx=6)
        tk.Label(status_bar, textvariable=self._diff_var,
                 bg='#0a0a0a', fg='#aaaaff',
                 font=('Courier', 7), anchor='w').pack(fill='x', padx=6, pady=1)

        # ── Bottom area: grid | player ─────────────────────────────────
        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill='both', expand=True, padx=4, pady=4)

        # Grid (fixed width ~500)
        grid_frame = tk.Frame(bottom, bg=BG2, width=500)
        grid_frame.pack(side='left', fill='y')
        grid_frame.pack_propagate(False)

        tk.Label(grid_frame, text='EVENTS',
                 bg=BG2, fg=FG2,
                 font=('Arial', 8, 'bold')).pack(anchor='w', padx=4, pady=2)

        self._grid = CompareGrid(grid_frame, on_select=self._on_row_select)
        self._grid.pack(fill='both', expand=True)

        # Divider
        tk.Frame(bottom, bg='#333333', width=2).pack(side='left', fill='y')

        # Split player (fills remainder)
        self._player = SplitPlayer(bottom)
        self._player.pack(side='left', fill='both', expand=True)

    @staticmethod
    def _add_file_picker(parent, label, var, cmd):
        f = tk.Frame(parent, bg=BG2)
        f.pack(side='left', padx=4)
        tk.Label(f, text=label, bg=BG2, fg=FG2,
                 font=('Arial', 7)).pack(anchor='w')
        row = tk.Frame(f, bg=BG2)
        row.pack()
        tk.Entry(row, textvariable=var, width=28,
                 bg=ENTRY_BG, fg=FG,
                 relief='flat',
                 insertbackground=FG).pack(side='left')
        tk.Button(row, text='…', command=cmd,
                  bg='#3a3a3a', fg=FG, relief='flat',
                  width=2).pack(side='left', padx=(1, 0))

    @staticmethod
    def _browse_file(var, filetypes):
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _load_config_defaults(self):
        """Load streaker_config.json into both panels on startup."""
        self._load_config(self._panel_a)
        self._load_config(self._panel_b)

    def _load_config(self, panel):
        if not os.path.exists(CONFIG_PATH):
            messagebox.showwarning('No config',
                                   f'Config not found:\n{CONFIG_PATH}')
            return
        try:
            with open(CONFIG_PATH) as fh:
                d = json.load(fh)
            panel.set_params(d)
        except Exception as exc:
            messagebox.showerror('Load error', str(exc))

    def _copy_a_to_b(self):
        self._panel_b.set_params(self._panel_a.get_params())

    # ------------------------------------------------------------------
    # Run / stop
    # ------------------------------------------------------------------

    def _on_run(self):
        mkv = self._mkv_path.get().strip()
        if not mkv or not os.path.exists(mkv):
            messagebox.showerror('Missing source',
                                 'Please select a valid MKV file.')
            return
        if self._running:
            return

        self._running = True
        self._stop_evt.clear()
        self._run_btn.config(state='disabled')
        self._stop_btn.config(state='normal')
        self._status_var.set('Running detection pass A …')
        self._summary_var.set('')
        self._diff_var.set('')
        self._grid.clear()

        params_a = self._panel_a.get_params()
        params_b = self._panel_b.get_params()
        mask     = self._mask_path.get().strip() or None

        # Compute and display param diff immediately
        self._show_diff(params_a, params_b)

        t = threading.Thread(
            target=self._worker,
            args=(mkv, mask, params_a, params_b),
            daemon=True)
        t.start()
        self._poll_results()

    def _on_stop(self):
        self._stop_evt.set()
        self._status_var.set('Stopping …')

    def _worker(self, mkv, mask, params_a, params_b):
        try:
            def progress_a(fi, tot):
                self._result_q.put(('progress', f'Config A: frame {fi}/{tot}'))

            def progress_b(fi, tot):
                self._result_q.put(('progress', f'Config B: frame {fi}/{tot}'))

            events_a = run_detection_pass(
                mkv, mask, params_a,
                progress_cb=progress_a,
                stop_event=self._stop_evt)

            if self._stop_evt.is_set():
                self._result_q.put(('stopped', None))
                return

            self._result_q.put(('progress', 'Config A done — running Config B …'))

            events_b = run_detection_pass(
                mkv, mask, params_b,
                progress_cb=progress_b,
                stop_event=self._stop_evt)

            if self._stop_evt.is_set():
                self._result_q.put(('stopped', None))
                return

            rows = match_events(events_a, events_b)
            self._result_q.put(('done', (events_a, events_b, rows)))

        except Exception as exc:
            import traceback
            self._result_q.put(('error',
                                 f'{exc}\n{traceback.format_exc()}'))

    def _poll_results(self):
        try:
            while True:
                msg_type, payload = self._result_q.get_nowait()
                if msg_type == 'progress':
                    self._status_var.set(payload)
                elif msg_type == 'done':
                    self._on_done(*payload)
                    return
                elif msg_type == 'stopped':
                    self._status_var.set('Stopped.')
                    self._finish_run()
                    return
                elif msg_type == 'error':
                    messagebox.showerror('Detection error', payload)
                    self._finish_run()
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll_results)

    def _on_done(self, events_a, events_b, rows):
        n_a    = len(events_a)
        n_b    = len(events_b)
        both   = sum(1 for ea, eb in rows if ea and eb)
        only_a = sum(1 for ea, eb in rows if ea and not eb)
        only_b = sum(1 for ea, eb in rows if not ea and eb)

        self._summary_var.set(
            f'A: {n_a} events    B: {n_b}    Both: {both}'
            f'    Only A: {only_a}    Only B: {only_b}')
        self._status_var.set('Detection complete.')
        self._grid.populate(rows)
        self._finish_run()

    def _finish_run(self):
        self._running = False
        self._run_btn.config(state='normal')
        self._stop_btn.config(state='disabled')

    # ------------------------------------------------------------------
    # Param diff display
    # ------------------------------------------------------------------

    def _show_diff(self, pa, pb):
        diffs = []
        for (key, label, *_rest) in PARAM_DEFS:
            va = pa.get(key)
            vb = pb.get(key)
            if va != vb:
                diffs.append(f'{label}: {va} vs {vb}')
        if diffs:
            self._diff_var.set('Diff: ' + '  |  '.join(diffs))
        else:
            self._diff_var.set('Diff: (configs are identical)')

    # ------------------------------------------------------------------
    # Grid row selection → split player
    # ------------------------------------------------------------------

    def _on_row_select(self, data):
        self._player.load(
            frames_a=data['frames_a'],
            frames_b=data['frames_b'],
            start_a=data['start_a'],
            start_b=data['start_b'],
            fps=data.get('fps', 20.0))

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def _on_close(self):
        self._stop_evt.set()
        self.root.destroy()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--source', default='')
    args, _ = ap.parse_known_args()
    app = StreakerCompareApp()
    if args.source and os.path.exists(args.source):
        app._mkv_path.set(args.source)
    app.run()
