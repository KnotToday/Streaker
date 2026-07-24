# StreakerPlayer.py — MKV timeline player with event markers and annotations

import os
import sys
import json
import math
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import re

from platform_utils import FFMPEG_PATH, HWACCEL_ARGS, find_ffprobe, NO_WINDOW, launch_companion

try:
    from StreakerLocalSolve import launch_local_solve
    _LOCAL_SOLVE_AVAILABLE = True
except ImportError:
    _LOCAL_SOLVE_AVAILABLE = False

_BRAVE = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"


def _angular_sep_deg(ra1, dec1, ra2, dec2):
    r = math.pi / 180
    ra1, dec1, ra2, dec2 = ra1*r, dec1*r, ra2*r, dec2*r
    a = (math.sin((dec2-dec1)/2)**2 +
         math.cos(dec1)*math.cos(dec2)*math.sin((ra2-ra1)/2)**2)
    return 2*math.degrees(math.asin(math.sqrt(min(a, 1.0))))


def _fit_wcs(star_pairs):
    """Fit a linear WCS from identified star pairs.
    star_pairs: list of {'px', 'py', 'ra', 'dec'}
    Returns pixel_to_radec(x, y) callable, or None if < 3 stars."""
    if len(star_pairs) < 3:
        return None
    xs  = np.array([s['px'] for s in star_pairs], dtype=float)
    ys  = np.array([s['py'] for s in star_pairs], dtype=float)
    ras = np.array([s['ra']  for s in star_pairs], dtype=float)
    dcs = np.array([s['dec'] for s in star_pairs], dtype=float)
    cx, cy = np.mean(xs), np.mean(ys)
    A = np.column_stack([xs - cx, ys - cy, np.ones(len(xs))])
    cr, *_ = np.linalg.lstsq(A, ras, rcond=None)
    cd, *_ = np.linalg.lstsq(A, dcs, rcond=None)
    def pixel_to_radec(x, y):
        v = np.array([x - cx, y - cy, 1.0])
        return float(cr @ v), float(cd @ v)
    return pixel_to_radec


def _open_in_browser(path):
    """Open a file in Brave browser (new tab), falling back to os.startfile."""
    import pathlib
    url = pathlib.Path(path).as_uri()
    if os.path.isfile(_BRAVE):
        subprocess.Popen([_BRAVE, url], creationflags=NO_WINDOW)
    else:
        os.startfile(path)

# Sentinel returned by read_buffered() when the queue is momentarily empty.
# Callers should retry after a short delay rather than treating this as EOF.
_BUF_WAIT = object()
BUFFER_FRAMES = 30   # ~1.5 s at 20 fps — enough to absorb NAS latency spikes

BG       = '#1a1a1a'
BG2      = '#111111'
FG       = '#cccccc'
ACCENT   = '#336699'

LABEL_COLORS = {
    'unreviewed': '#555555',
    'interesting': '#ffcc00',
    'junk':        '#cc3333',
    'meteor':      '#00ccff',
    'plane':       '#00ff88',
    'satellite':   '#ff88ff',
    'unknown':     '#ff8800',
}

LABELS = ['unreviewed', 'interesting', 'junk', 'meteor', 'plane', 'satellite', 'unknown']

# Map identification.json classifications → player label names
_IDENT_TO_LABEL = {
    'meteor':                 'meteor',
    'aircraft':               'plane',
    'satellite':              'satellite',
    'noise':                  'junk',
    'unidentified':           'unknown',
    'unidentified satellite?':'satellite',
    'unidentified aircraft?': 'plane',
}


def ffprobe_info(ffmpeg_path, video_path):
    """Return (fps, total_frames, width, height) via ffprobe."""
    ffprobe = find_ffprobe(ffmpeg_path)
    try:
        out = subprocess.check_output([
            ffprobe, '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-show_format', video_path],
            stderr=subprocess.DEVNULL)
        info = json.loads(out)
        fmt_dur = float(info.get('format', {}).get('duration', 0) or 0)
        for s in info.get('streams', []):
            if s.get('codec_type') == 'video':
                w = int(s.get('width', 1920))
                h = int(s.get('height', 1080))
                fr = s.get('r_frame_rate', '20/1')
                num, den = fr.split('/')
                fps = float(num) / float(den)
                nb  = s.get('nb_frames')
                dur = float(s.get('duration', 0) or 0) or fmt_dur
                if nb:
                    frames = int(nb)
                elif dur:
                    frames = int(dur * fps)
                else:
                    frames = 0
                return fps, frames, w, h
    except Exception:
        pass
    return 20.0, 0, 1920, 1080


class FrameReader:
    """FFmpeg-backed frame reader with a background seek thread.

    Seeks are non-blocking: seek_async() posts a request and returns immediately;
    the worker thread kills the old process, launches a new one, reads the first
    frame, and delivers it via callback.  Sequential read() for playback runs on
    the caller's thread and touches _proc directly — safe because playback is
    always stopped before any seek is issued.
    """

    def __init__(self, ffmpeg_path, video_path, width, height, fps):
        self.ffmpeg_path   = ffmpeg_path
        self.video_path    = video_path
        self.width         = width
        self.height        = height
        self.fps           = fps
        self._nbytes       = width * height * 3
        self._proc         = None
        self.current_frame = 0
        self._seek_q       = queue.Queue()
        self._worker       = threading.Thread(target=self._seek_worker, daemon=True)
        self._worker.start()
        # Frame buffer for smooth NAS playback
        self._buf_q       = queue.Queue(maxsize=BUFFER_FRAMES)
        self._buf_stop    = threading.Event()
        self._fill_thread = None
        self._fill_gen    = 0   # incremented each start; old threads check this and discard writes

    # ── internal helpers ──────────────────────────────────────────────────────

    def _launch(self, frame_num, hw=True):
        ss = frame_num / max(self.fps, 1)
        cmd = [self.ffmpeg_path]
        if hw and HWACCEL_ARGS:
            cmd += HWACCEL_ARGS
        cmd += ['-ss', f'{ss:.4f}', '-i', self.video_path,
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-an', 'pipe:1']
        return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                                creationflags=NO_WINDOW)

    @staticmethod
    def _kill(proc):
        if proc:
            try: proc.stdout.close()
            except Exception: pass
            try: proc.kill(); proc.wait(timeout=2)
            except Exception: pass

    # ── public API ────────────────────────────────────────────────────────────

    def seek_async(self, frame_num, callback):
        """Non-blocking seek.  Drains any pending requests (latest wins).
        callback(frame_or_None, frame_num) is called from the worker thread;
        callers must dispatch UI updates via root.after(0, ...)."""
        while not self._seek_q.empty():
            try: self._seek_q.get_nowait()
            except queue.Empty: break
        self._seek_q.put((frame_num, callback))

    def seek(self, frame_num):
        """Blocking seek — dispatches to worker and waits (used for initial load)."""
        done   = threading.Event()
        result = [None]
        def _cb(frame, fn):
            result[0] = frame
            done.set()
        self.seek_async(frame_num, _cb)
        done.wait(timeout=15)
        return result[0]

    def read(self):
        """Read next sequential frame.  Call only from main thread during playback
        (i.e. never concurrently with an in-flight seek_async)."""
        if not self._proc:
            return None
        raw = self._proc.stdout.read(self._nbytes)
        if len(raw) < self._nbytes:
            # CUDA may have failed mid-stream — fall back to software
            self._kill(self._proc)
            self._proc = self._launch(self.current_frame, hw=False)
            raw = self._proc.stdout.read(self._nbytes)
            if len(raw) < self._nbytes:
                self._proc = None
                return None
        self.current_frame += 1
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)

    def skip_frames(self, n):
        """Read and discard n frames for fast-forward. Returns False if stream ends."""
        if not self._proc:
            return False
        for _ in range(n):
            raw = self._proc.stdout.read(self._nbytes)
            if len(raw) < self._nbytes:
                return False
            self.current_frame += 1
        return True

    def start_buffering(self):
        """Launch background fill thread so read_buffered() never blocks on I/O."""
        self._stop_fill()
        self._buf_stop.clear()
        self._fill_gen += 1
        gen = self._fill_gen
        self._fill_thread = threading.Thread(
            target=self._fill_loop, args=(gen,), daemon=True)
        self._fill_thread.start()

    def stop_buffering(self):
        """Stop the fill thread and drain the queue (call before seek or stop)."""
        self._stop_fill()

    def _stop_fill(self):
        if self._fill_thread and self._fill_thread.is_alive():
            self._buf_stop.set()
            # Drain one slot so a blocked put() can unblock and exit
            try: self._buf_q.get_nowait()
            except queue.Empty: pass
            self._fill_thread.join(timeout=1.0)
        self._fill_thread = None
        # Clear residual items
        while True:
            try: self._buf_q.get_nowait()
            except queue.Empty: break

    def _fill_loop(self, gen):
        """Background: read frames from FFmpeg pipe into queue ahead of playback.
        gen is this thread's generation; any write is skipped if a newer generation exists."""
        fn = self.current_frame
        while not self._buf_stop.is_set() and self._fill_gen == gen:
            if not self._proc:
                if self._fill_gen == gen:
                    try: self._buf_q.put(None, timeout=0.5)
                    except queue.Full: pass
                break
            raw = self._proc.stdout.read(self._nbytes)
            # After a blocking read, re-check generation before touching the queue
            if self._fill_gen != gen or self._buf_stop.is_set():
                break
            if len(raw) < self._nbytes:
                try: self._buf_q.put(None, timeout=0.5)
                except queue.Full: pass
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                self.height, self.width, 3).copy()
            try:
                self._buf_q.put((fn, frame), timeout=0.2)
                fn += 1
            except queue.Full:
                if self._buf_stop.is_set() or self._fill_gen != gen:
                    break

    def read_buffered(self):
        """Non-blocking read for playback. Returns:
          - ndarray frame on success
          - None on EOF
          - _BUF_WAIT sentinel when queue is momentarily empty (caller should retry)
        Falls back to direct read() when fill thread is not running."""
        if self._fill_thread and self._fill_thread.is_alive():
            try:
                item = self._buf_q.get_nowait()
            except queue.Empty:
                return _BUF_WAIT
            if item is None:
                return None
            fn, frame = item
            self.current_frame = fn + 1
            return frame
        return self.read()

    def close(self):
        """Shut down the worker thread and release the ffmpeg process."""
        self._stop_fill()
        try: self._seek_q.put(None)
        except Exception: pass
        self._kill(self._proc)
        self._proc = None

    # ── background worker ─────────────────────────────────────────────────────

    def _seek_worker(self):
        proc = None
        while True:
            item = self._seek_q.get()
            if item is None:          # shutdown signal
                self._kill(proc)
                return
            frame_num, callback = item

            # Drain queue — only honour the latest request
            while True:
                try:
                    item2 = self._seek_q.get_nowait()
                    if item2 is None:
                        self._kill(proc)
                        callback(None, frame_num)
                        return
                    frame_num, callback = item2
                except queue.Empty:
                    break

            # Kill previous process before launching a new one
            self._kill(proc)
            proc = None

            # Try hardware decode first, fall back to software
            frame = None
            for hw in ([True, False] if HWACCEL_ARGS else [False]):
                p   = self._launch(frame_num, hw=hw)
                raw = p.stdout.read(self._nbytes)
                if len(raw) == self._nbytes:
                    proc = p
                    self.current_frame = frame_num + 1
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        self.height, self.width, 3).copy()
                    break
                self._kill(p)

            # Expose proc so read() can continue from here during playback
            self._proc = proc
            callback(frame, frame_num)


class StreakerPlayer:
    TIMELINE_H   = 60
    CTRL_H       = 40

    def __init__(self, root, initial_folder=None, initial_mkv=None):
        self.root = root
        self.root.title("Streaker Player")
        self.root.configure(bg=BG)
        self.root.state('zoomed')
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._initial_mkv = initial_mkv
        self.run_folder   = None
        self.events       = []       # all loaded events (dicts)
        self.clip_events  = {}       # clip_path -> [event, ...]
        self.clip_list    = []       # ordered list of unique clip paths
        self.annotations     = {}    # event_dir -> label string (manual)
        self.identifications = {}    # event_dir -> identification.json dict
        self.annot_path      = None

        self.current_clip_idx = 0
        self.reader       = None
        self.fps          = 20.0
        self.total_frames = 0
        self.frame_w      = 1920
        self.frame_h      = 1080
        self.current_frame_num = 0

        self.playing      = False
        self._play_id     = None
        self.play_speed   = 1.0      # multiplier
        self.current_event_idx = -1  # index into self.events for current clip
        self._displayed_events = []  # events currently shown in listbox (may be filtered/sorted)
        self._sort_mode = tk.StringVar(value='time')
        self._last_raw_frame  = None  # full-res BGR frame for grab/export
        self._last_stack_path = None  # path to most recent stack_mean.png
        self._measure_mode  = False
        self._measure_pt_a  = None   # (img_x, img_y, frame_num) or None
        self._wcs_fn        = None   # pixel_to_radec(x,y) from Local Solve

        self.tk_image     = None
        self._frame_q     = queue.Queue(maxsize=4)
        self._reader_thread = None
        self._stop_reader = threading.Event()

        self._build_ui()

        if initial_folder and os.path.isdir(initial_folder):
            mkv = getattr(self, '_initial_mkv', None)
            if mkv:
                # Load folder for events, then jump to the specific clip
                self.root.after(100, lambda: self._load_folder_then_clip(initial_folder, mkv))
            else:
                self.root.after(100, lambda: self.load_run_folder(initial_folder))
        elif getattr(self, '_initial_mkv', None):
            self.root.after(100, lambda: self._open_direct_mkv(self._initial_mkv))

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Top bar row 1: file open + clip navigation ────────────────
        row1 = tk.Frame(self.root, bg=BG)
        row1.pack(fill='x', side='top', padx=6, pady=(4, 0))

        tk.Label(row1, text="STREAKER PLAYER", bg=BG, fg='white',
                 font=('Arial', 10, 'bold')).pack(side='left', padx=4)
        tk.Label(row1, text="│", bg=BG, fg='#444').pack(side='left', padx=2)
        tk.Button(row1, text="📁 Folder",
                  command=self._browse_run_folder,
                  bg='#334455', fg='white', relief='flat',
                  padx=6, pady=2).pack(side='left', padx=2)
        tk.Button(row1, text="🎬 MKV",
                  command=self._browse_mkv,
                  bg='#334455', fg='white', relief='flat',
                  padx=6, pady=2).pack(side='left', padx=2)
        tk.Button(row1, text="↺ Reload",
                  command=self._reload_identifications,
                  bg='#334455', fg='white', relief='flat',
                  padx=6, pady=2).pack(side='left', padx=2)

        # Clip navigation — right side of row 1
        nav = tk.Frame(row1, bg=BG)
        nav.pack(side='right', padx=4)
        tk.Button(nav, text="◀ Prev", command=self._prev_clip,
                  bg='#333333', fg='white', relief='flat', padx=5).pack(side='left', padx=2)
        self.clip_lbl = tk.Label(nav, text="—", bg=BG, fg=FG,
                                 font=('Arial', 8), width=28)
        self.clip_lbl.pack(side='left', padx=4)
        tk.Button(nav, text="Next ▶", command=self._next_clip,
                  bg='#333333', fg='white', relief='flat', padx=5).pack(side='left', padx=2)

        # ── Top bar row 2: tools + path info ─────────────────────────
        row2 = tk.Frame(self.root, bg=BG)
        row2.pack(fill='x', side='top', padx=6, pady=(2, 2))

        tk.Button(row2, text="📷 Grab",
                  command=self._grab_frame,
                  bg='#334433', fg='white', relief='flat',
                  padx=6, pady=1).pack(side='left', padx=2)
        tk.Button(row2, text="📷×N Stack",
                  command=self._grab_series,
                  bg='#334433', fg='white', relief='flat',
                  padx=6, pady=1).pack(side='left', padx=2)
        tk.Label(row2, text="│", bg=BG, fg='#444').pack(side='left', padx=2)
        tk.Button(row2, text="🔭 Solve",
                  command=self._launch_astro,
                  bg='#1a3a5a', fg='white', relief='flat',
                  padx=6, pady=1).pack(side='left', padx=2)
        tk.Button(row2, text="🌐 Local Solve",
                  command=self._launch_local_solve,
                  bg='#1a3a5a', fg='white', relief='flat',
                  padx=6, pady=1).pack(side='left', padx=2)
        tk.Button(row2, text="✏ Mask",
                  command=self._launch_mask_editor,
                  bg='#2a2a2a', fg='#cccccc', relief='flat',
                  padx=6, pady=1).pack(side='left', padx=2)
        self._measure_btn = tk.Button(row2, text="📐 Measure",
                  command=self._toggle_measure_mode,
                  bg='#2a2a2a', fg='#cccccc', relief='flat',
                  padx=6, pady=1)
        self._measure_btn.pack(side='left', padx=2)
        tk.Label(row2, text="│", bg=BG, fg='#444').pack(side='left', padx=2)
        tk.Button(row2, text="🔺 Triangulate",
                  command=self._launch_triangulate,
                  bg='#2a1a3a', fg='#cc88ff', relief='flat',
                  padx=6, pady=1).pack(side='left', padx=2)
        tk.Label(row2, text="│", bg=BG, fg='#444').pack(side='left', padx=2)

        self.folder_lbl = tk.Label(row2, text="No folder loaded",
                                   bg=BG, fg='#888888', font=('Arial', 8))
        self.folder_lbl.pack(side='left', padx=4, fill='x', expand=True)

        # ── Main area: video + event list ────────────────────────────
        # NOTE: main.pack() is called AFTER the bottom widgets are packed,
        # so that side='bottom' frames claim their space first.
        main = tk.Frame(self.root, bg=BG)

        # Video canvas
        self.canvas = tk.Canvas(main, bg='black', cursor='crosshair')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.bind('<Button-1>', self._on_canvas_click)
        self.canvas.bind('<Configure>', self._on_canvas_resize)

        # Event list panel
        ep = tk.Frame(main, bg=BG2, width=220)
        ep.pack(side='right', fill='y')
        ep.pack_propagate(False)

        tk.Label(ep, text="EVENTS", bg=BG2, fg='#aaaaaa',
                 font=('Arial', 8, 'bold')).pack(pady=(6, 2))

        self.event_count_lbl = tk.Label(ep, text="0 events", bg=BG2,
                                        fg='#666666', font=('Arial', 7))
        self.event_count_lbl.pack()

        # Annotation buttons
        ann_f = tk.Frame(ep, bg=BG2)
        ann_f.pack(fill='x', padx=4, pady=4)
        tk.Label(ann_f, text="Mark as:", bg=BG2, fg='#888888',
                 font=('Arial', 7)).pack(anchor='w')
        for label in LABELS[1:]:
            color = LABEL_COLORS[label]
            tk.Button(ann_f, text=label.capitalize(),
                      bg='#222222', fg=color, relief='flat',
                      font=('Arial', 7), pady=1,
                      command=lambda l=label: self._annotate_current(l)
                      ).pack(fill='x', pady=1)

        tk.Button(ann_f, text="💾 Save Annotations",
                  command=self._save_annotations,
                  bg='#224422', fg='white', relief='flat',
                  font=('Arial', 7), pady=3).pack(fill='x', pady=(6, 2))

        # Sort / filter controls
        sort_row = tk.Frame(ep, bg=BG2)
        sort_row.pack(fill='x', padx=4, pady=(2, 0))
        tk.Label(sort_row, text='Sort:', bg=BG2, fg='#666666',
                 font=('Arial', 7)).pack(side='left', padx=(0, 3))
        sort_cb = ttk.Combobox(sort_row, textvariable=self._sort_mode,
                               values=['time', 'flagged_first', 'flagged_only'],
                               state='readonly', width=13,
                               font=('Arial', 7))
        sort_cb.pack(side='left')
        sort_cb.bind('<<ComboboxSelected>>', lambda _e: self._on_sort_change())

        # Scrollable event list
        lf = tk.Frame(ep, bg=BG2)
        lf.pack(fill='both', expand=True, padx=2)
        sb = tk.Scrollbar(lf)
        sb.pack(side='right', fill='y')
        self.event_listbox = tk.Listbox(lf, bg=BG2, fg=FG,
                                        font=('Courier', 7),
                                        selectbackground=ACCENT,
                                        yscrollcommand=sb.set,
                                        relief='flat', borderwidth=0)
        self.event_listbox.pack(side='left', fill='both', expand=True)
        sb.config(command=self.event_listbox.yview)
        self.event_listbox.bind('<<ListboxSelect>>', self._on_event_select)

        # ── Timeline ─────────────────────────────────────────────────
        tl_frame = tk.Frame(self.root, bg='#0d0d0d', height=self.TIMELINE_H)
        tl_frame.pack(fill='x', side='bottom')
        tl_frame.pack_propagate(False)

        self.timeline = tk.Canvas(tl_frame, bg='#0d0d0d',
                                  height=self.TIMELINE_H, cursor='hand2')
        self.timeline.pack(fill='both', expand=True, padx=4, pady=4)
        self.timeline.bind('<Button-1>', self._on_timeline_click)
        self.timeline.bind('<Configure>', lambda e: self._draw_timeline())

        self.time_lbl = tk.Label(tl_frame, text="00:00 / 00:00",
                                 bg='#0d0d0d', fg='#888888',
                                 font=('Courier', 7))
        self.time_lbl.place(relx=1.0, rely=0.0, anchor='ne', x=-6, y=4)

        # ── Scrubber ──────────────────────────────────────────────────
        scrub_frame = tk.Frame(self.root, bg='#0d0d0d')
        scrub_frame.pack(fill='x', side='bottom')
        self._scrub_updating = False
        self.scrubber_var = tk.DoubleVar(value=0)
        self.scrubber = tk.Scale(scrub_frame, from_=0, to=1000,
                                 variable=self.scrubber_var,
                                 orient='horizontal', bg='#0d0d0d',
                                 troughcolor='#444', fg='#888',
                                 highlightthickness=0, showvalue=False,
                                 command=self._on_scrub)
        self.scrubber.pack(fill='x', padx=6, pady=2)

        # ── Playback controls ─────────────────────────────────────────
        ctrl = tk.Frame(self.root, bg='#0d0d0d')
        ctrl.pack(fill='x', side='bottom', pady=2)

        tk.Button(ctrl, text="|◀", command=self._goto_clip_start,
                  bg='#222', fg='white', relief='flat', width=3).pack(side='left', padx=2, pady=2)
        tk.Button(ctrl, text="−10s", command=lambda: self._step_seconds(-10),
                  bg='#222', fg='white', relief='flat', width=5).pack(side='left', padx=2)
        tk.Button(ctrl, text="◀", command=lambda: self._step(-1),
                  bg='#222', fg='white', relief='flat', width=3).pack(side='left', padx=2)

        self.play_btn = tk.Button(ctrl, text="▶ Play",
                                  command=self._toggle_play,
                                  bg='#006600', fg='white',
                                  relief='flat', padx=10, pady=3,
                                  font=('Arial', 8, 'bold'))
        self.play_btn.pack(side='left', padx=4)

        tk.Button(ctrl, text="▶", command=lambda: self._step(1),
                  bg='#222', fg='white', relief='flat', width=3).pack(side='left', padx=2)
        tk.Button(ctrl, text="+10s", command=lambda: self._step_seconds(10),
                  bg='#222', fg='white', relief='flat', width=5).pack(side='left', padx=2)
        tk.Button(ctrl, text="▶|", command=self._goto_clip_end,
                  bg='#222', fg='white', relief='flat', width=3).pack(side='left', padx=2)

        # Event navigation
        tk.Label(ctrl, text="│", bg='#0d0d0d', fg='#444').pack(side='left', padx=4)
        tk.Button(ctrl, text="◀ Ev", command=self._prev_event,
                  bg='#334455', fg='white', relief='flat', padx=6).pack(side='left', padx=2)
        tk.Button(ctrl, text="Ev ▶", command=self._next_event,
                  bg='#334455', fg='white', relief='flat', padx=6).pack(side='left', padx=2)

        tk.Label(ctrl, text="│", bg='#0d0d0d', fg='#444').pack(side='left', padx=4)
        tk.Label(ctrl, text="Speed:", bg='#0d0d0d', fg='#888',
                 font=('Arial', 7)).pack(side='left', padx=(4, 2))
        self.speed_var = tk.DoubleVar(value=1.0)
        tk.Scale(ctrl, from_=0.25, to=8.0, resolution=0.25,
                 variable=self.speed_var, orient='horizontal',
                 bg='#0d0d0d', fg='white', troughcolor='#333',
                 highlightthickness=0, length=120, showvalue=True,
                 font=('Arial', 7),
                 command=self._on_speed_change
                 ).pack(side='left')

        self.status_lbl = tk.Label(ctrl, text="No clip loaded",
                                   bg='#0d0d0d', fg='#888',
                                   font=('Courier', 7))
        self.status_lbl.pack(side='left', padx=12)

        # Legend
        leg = tk.Frame(ctrl, bg='#0d0d0d')
        leg.pack(side='right', padx=8)
        for label, color in list(LABEL_COLORS.items())[1:]:
            tk.Label(leg, text='█', fg=color, bg='#0d0d0d',
                     font=('Arial', 9)).pack(side='left')
            tk.Label(leg, text=label, fg='#666', bg='#0d0d0d',
                     font=('Arial', 6)).pack(side='left', padx=(0, 4))

        # Pack main last so it fills whatever vertical space the bottom widgets leave.
        main.pack(fill='both', expand=True)

        # Keyboard bindings
        self.root.bind('<Left>',  lambda e: self._step(-1))
        self.root.bind('<Right>', lambda e: self._step(1))
        self.root.bind('<space>', lambda e: self._toggle_play())
        self.root.bind('<comma>',  lambda e: self._prev_event())   # < key
        self.root.bind('<period>', lambda e: self._next_event())   # > key
        self.root.bind('<p>', lambda e: self._grab_frame())
        self.root.bind('<P>', lambda e: self._grab_series())
        self.root.bind('<r>', lambda e: self._reload_identifications())

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _open_direct_mkv(self, mkv_path):
        """Load a single MKV directly — mirrors _browse_mkv without the dialog."""
        try:
            self._stop_playback()
            fps, total, w, h = ffprobe_info(FFMPEG_PATH, mkv_path)
            self.fps, self.total_frames = fps, total
            self.frame_w, self.frame_h  = w, h
            self.clip_list    = [mkv_path]
            self.clip_events  = {mkv_path: []}
            self.events       = []
            self.current_clip_idx  = 0
            self.current_event_idx = -1
            if self.reader:
                self.reader.close()
            self.reader = FrameReader(FFMPEG_PATH, mkv_path, w, h, fps)
            self.reader.seek(0)
            name = os.path.basename(mkv_path)
            self.folder_lbl.config(text=os.path.dirname(mkv_path))
            self.clip_lbl.config(text=name)
            self.event_listbox.delete(0, 'end')
            self.event_count_lbl.config(text="0 events (direct clip)")
            self._draw_timeline()
            self._seek_and_show(0)
        except Exception as exc:
            import traceback
            messagebox.showerror("Player Error",
                                 f"Failed to open:\n{mkv_path}\n\n{traceback.format_exc()}")

    def _scan_events_in(self, folder):
        """Scan folder for event_* subdirs, appending to self.events / self.clip_events / self.identifications."""
        try:
            entries = sorted(os.listdir(folder))
        except Exception:
            return
        for d in entries:
            meta_path  = os.path.join(folder, d, 'metadata.json')
            ident_path = os.path.join(folder, d, 'identification.json')
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                continue
            ev_dir = os.path.join(folder, d)
            if os.path.exists(ident_path):
                try:
                    with open(ident_path) as f:
                        self.identifications[ev_dir] = json.load(f)
                except Exception:
                    pass
            clip = meta.get('source_clip', '')
            clip_missing = not clip or not os.path.exists(clip)
            fps = meta.get('fps', 20.0)
            ev = {
                'dir':          ev_dir,
                'name':         d,
                'source_clip':  clip,
                'clip_missing': clip_missing,
                'start_frame':  meta.get('start_frame', 0),
                'end_frame':    meta.get('end_frame', 0),
                'start_sec':    meta.get('start_frame', 0) / fps,
                'end_sec':      meta.get('end_frame', 0) / fps,
                'fps':          fps,
                'detections':   meta.get('detections', []),
                'detect_scale': meta.get('detect_scale', 1.0),
            }
            self.events.append(ev)
            # Group under clip path; missing clips group under their stored path
            self.clip_events.setdefault(clip or f'__missing__{ev_dir}', []).append(ev)

    def load_run_folder(self, folder):
        self.run_folder = folder
        self.annot_path = os.path.join(folder, 'annotations.json')
        self.annotations = {}
        if os.path.exists(self.annot_path):
            try:
                with open(self.annot_path) as f:
                    self.annotations = json.load(f)
            except Exception:
                pass

        self.events = []
        self.clip_events = {}
        self.identifications = {}

        self._scan_events_in(folder)

        # Auto-descend: if nothing found directly, try one level of subdirectories
        if not self.events:
            try:
                subs = sorted([
                    os.path.join(folder, d) for d in os.listdir(folder)
                    if os.path.isdir(os.path.join(folder, d))
                ])
            except Exception:
                subs = []
            for sub in subs:
                self._scan_events_in(sub)

        # Build clip list — exclude placeholder missing-clip keys from video list
        self.clip_list = sorted(
            k for k in self.clip_events if not k.startswith('__missing__'))

        n   = len(self.events)
        n_missing = sum(1 for ev in self.events if ev['clip_missing'])
        nc  = len(self.clip_list)
        # Show last two path components so date folder is visible: 07-20-2026 › detect_xxx
        parts = os.path.normpath(folder).replace('\\', '/').split('/')
        short = ' › '.join(p for p in parts[-2:] if p)
        label = f"{short} — {n} events, {nc} clip(s)"
        if n_missing:
            label += f"  ({n_missing} not found)"
        self.folder_lbl.config(text=label)
        self.event_count_lbl.config(text=f"{n} events total")

        if self.clip_list:
            self.current_clip_idx = 0
            self._load_clip(self.clip_list[0])
        elif self.events:
            # All clips missing — still populate the event list so user can see identifications
            self.folder_lbl.config(
                text=f"{os.path.basename(folder)} — {n} events (source clips not found)")
            self._refresh_event_list(None)

    def _load_folder_then_clip(self, folder, mkv_path):
        """Load run folder for events, then jump to the specific source clip."""
        print(f"[PLAYER] load_folder_then_clip folder={folder!r}")
        print(f"[PLAYER] load_folder_then_clip mkv_path={mkv_path!r}")
        print(f"[PLAYER] folder isdir={os.path.isdir(folder)}")
        self.load_run_folder(folder)
        print(f"[PLAYER] after load_run_folder: {len(self.events)} events, clip_list={self.clip_list}")
        if mkv_path in self.clip_list:
            idx = self.clip_list.index(mkv_path)
            self.current_clip_idx = idx
            self._load_clip(mkv_path)
        else:
            print(f"[PLAYER] mkv_path NOT in clip_list — clip_list={self.clip_list[:3]}")

    def _display_label(self, ev_dir):
        """Return (label, color, is_auto, flagged) for an event.
        Manual annotation takes priority; falls back to identification.json."""
        manual = self.annotations.get(ev_dir)
        if manual:
            return manual, LABEL_COLORS.get(manual, LABEL_COLORS['unreviewed']), False, False
        ident = self.identifications.get(ev_dir)
        if ident:
            cls   = ident.get('classification', '')
            label = _IDENT_TO_LABEL.get(cls, 'unknown')
            color = LABEL_COLORS.get(label, LABEL_COLORS['unreviewed'])
            return label, color, True, ident.get('flagged', False)
        return 'unreviewed', LABEL_COLORS['unreviewed'], False, False

    def _load_clip(self, clip_path):
        self._stop_playback()
        if self.reader:
            self.reader.close()

        if not clip_path or not os.path.exists(clip_path):
            self.reader = None
            self.fps = 20.0; self.total_frames = 0
            self.frame_w = 640; self.frame_h = 480
            name = os.path.basename(clip_path) if clip_path else '(unknown)'
            idx  = self.current_clip_idx + 1
            self.clip_lbl.config(text=f"[{idx}/{len(self.clip_list)}] {name} — NOT FOUND")
            self._refresh_event_list(clip_path)
            self.current_frame_num = 0
            self.current_event_idx = -1
            self._draw_timeline()
            return

        self.fps, self.total_frames, self.frame_w, self.frame_h = \
            ffprobe_info(FFMPEG_PATH, clip_path)
        self.reader = FrameReader(FFMPEG_PATH, clip_path,
                                  self.frame_w, self.frame_h, self.fps)

        name = os.path.basename(clip_path)
        idx  = self.current_clip_idx + 1
        total = len(self.clip_list)
        self.clip_lbl.config(text=f"[{idx}/{total}] {name}")

        # Populate event list
        self._refresh_event_list(clip_path)

        self.current_frame_num = 0
        self.current_event_idx = -1
        self._draw_timeline()
        self._seek_and_show(0)

    def _reload_identifications(self):
        """Re-read identification.json files from disk without closing the folder."""
        folder = getattr(self, 'run_folder', None)
        if not folder or not os.path.isdir(folder):
            return
        self.identifications = {}
        for d in sorted(os.listdir(folder)):
            ident_path = os.path.join(folder, d, 'identification.json')
            if os.path.exists(ident_path):
                try:
                    with open(ident_path) as f:
                        self.identifications[os.path.join(folder, d)] = json.load(f)
                except Exception:
                    pass
        clip = self.clip_list[self.current_clip_idx] if self.clip_list else None
        if clip:
            self._refresh_event_list(clip)
            self._draw_timeline()

    def _browse_run_folder(self):
        folder = filedialog.askdirectory(title="Select Run Folder")
        if folder:
            self.load_run_folder(folder)

    def _browse_mkv(self):
        path = filedialog.askopenfilename(
            title="Open Video File",
            filetypes=[("Video files", "*.mkv *.mp4 *.avi *.mov"), ("All files", "*.*")])
        if not path:
            return
        self._stop_playback()
        fps, total, w, h = ffprobe_info(FFMPEG_PATH, path)
        self.fps          = fps
        self.total_frames = total
        self.frame_w      = w
        self.frame_h      = h
        self.clip_list    = [path]
        self.clip_events  = {path: []}
        self.events       = []
        self.current_clip_idx = 0
        self.current_event_idx = -1
        if self.reader:
            self.reader.close()
        self.reader = FrameReader(FFMPEG_PATH, path, w, h, fps)
        self.reader.seek(0)
        self.folder_lbl.config(text=os.path.dirname(path))
        self.clip_lbl.config(text=os.path.basename(path))
        self.event_listbox.delete(0, 'end')
        self.event_count_lbl.config(text="0 events (bare clip)")
        self._draw_timeline()
        self._seek_and_show(0)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _seek_and_show(self, frame_num):
        self.current_frame_num = max(0, frame_num)
        self._draw_timeline()
        self._update_time_label()
        self._update_scrubber()
        if self.reader:
            self.reader.stop_buffering()  # must stop before seek touches _proc
            self.reader.seek_async(self.current_frame_num, self._on_seek_done)

    def _on_seek_done(self, frame, frame_num):
        """Called from FrameReader worker thread — dispatch display back to main thread."""
        if frame is not None:
            self._last_raw_frame = frame
            self.root.after(0, lambda f=frame: self._display_frame(f))

    def _display_frame(self, frame_bgr):
        cw = max(self.canvas.winfo_width(), 640)
        ch = max(self.canvas.winfo_height(), 480)
        h, w = frame_bgr.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        img = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.canvas.delete('all')
        self.canvas.create_image(cw // 2, ch // 2, anchor='center', image=img)
        self.canvas.image = img
        self.tk_image = img

        # Overlay event marker if within an event
        self._overlay_event_info(frame_bgr.shape[1], frame_bgr.shape[0], scale, cw, ch)
        # Redraw measure markers
        if self._measure_pt_a:
            ix, iy, _ = self._measure_pt_a
            cx2, cy2 = self._image_to_canvas(ix, iy)
            if cx2 is not None:
                r = 6
                self.canvas.create_oval(cx2-r, cy2-r, cx2+r, cy2+r,
                                        outline='#00ffcc', width=2)
                self.canvas.create_text(cx2+10, cy2, text='A',
                                        fill='#00ffcc', font=('Arial', 10, 'bold'))

    def _overlay_event_info(self, fw, fh, scale, cw, ch):
        clip = self.clip_list[self.current_clip_idx] if self.clip_list else None
        if not clip:
            return
        for ev in self.clip_events.get(clip, []):
            if ev['start_frame'] <= self.current_frame_num <= ev['end_frame']:
                label, color, is_auto, flagged = self._display_label(ev['dir'])
                ds = ev.get('detect_scale', 1.0) or 1.0
                ox = (cw - int(fw * scale)) // 2
                oy = (ch - int(fh * scale)) // 2
                for det in ev['detections']:
                    if det['frame'] == self.current_frame_num and det.get('bboxes'):
                        for bx, by, bw, bh in det['bboxes']:
                            # bboxes are stored in detect_scale coords; scale up to full-res first
                            x1 = int(bx / ds * scale) + ox
                            y1 = int(by / ds * scale) + oy
                            x2 = int((bx + bw) / ds * scale) + ox
                            y2 = int((by + bh) / ds * scale) + oy
                            self.canvas.create_rectangle(
                                x1, y1, x2, y2, outline=color, width=2)
                prefix = '~' if is_auto else ''
                flag   = '  ⚑' if flagged else ''
                self.canvas.create_text(
                    10, 10, anchor='nw',
                    text=f"EVENT: {ev['name']}  [{prefix}{label}]{flag}",
                    fill=color, font=('Courier', 9, 'bold'))
                break

    def _toggle_play(self):
        if self.playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if not self.reader or not self.clip_list:
            return
        self.playing = True
        self.play_btn.config(text="⏸ Pause", bg='#664400')
        self.reader.start_buffering()
        self._play_tick()

    def _stop_playback(self):
        self.playing = False
        self.play_btn.config(text="▶ Play", bg='#006600')
        if self._play_id:
            self.root.after_cancel(self._play_id)
            self._play_id = None
        if self.reader:
            self.reader.stop_buffering()

    def _play_tick(self):
        if not self.playing or not self.reader:
            return
        speed = self.play_speed
        # Fast-forward: discard intermediate frames from the buffer
        if speed > 1.5:
            skip = max(0, int(round(speed)) - 1)
            for _ in range(skip):
                dropped = self.reader.read_buffered()
                if dropped is None:
                    self._stop_playback()
                    return
                if dropped is _BUF_WAIT:
                    break  # buffer momentarily empty — display next available frame
        frame = self.reader.read_buffered()
        if frame is _BUF_WAIT:
            # Buffer momentarily empty — retry soon without advancing frame counter
            self._play_id = self.root.after(20, self._play_tick)
            return
        if frame is None:
            self._stop_playback()
            return
        self.current_frame_num = max(0, self.reader.current_frame - 1)
        self._last_raw_frame = frame
        self._display_frame(frame)
        self._draw_timeline()
        self._update_time_label()
        self._update_scrubber()
        ms = max(1, int(1000 / (self.fps * min(speed, 1.0)))) if speed < 1.0 else max(1, int(1000 / self.fps))
        self._play_id = self.root.after(ms, self._play_tick)

    def _on_speed_change(self, val):
        self.play_speed = float(val)

    def _step(self, delta):
        self._stop_playback()
        self._seek_and_show(self.current_frame_num + delta)

    def _step_seconds(self, seconds):
        self._stop_playback()
        self._seek_and_show(self.current_frame_num + int(seconds * self.fps))

    def _on_scrub(self, val):
        if self._scrub_updating or self.total_frames == 0:
            return
        if hasattr(self, '_scrub_job') and self._scrub_job:
            self.root.after_cancel(self._scrub_job)
        self._scrub_job = self.root.after(150, lambda v=float(val): self._do_scrub(v))

    def _do_scrub(self, val):
        self._scrub_job = None
        if self.total_frames == 0:
            return
        frame = int(val / 1000 * self.total_frames)
        self._stop_playback()
        self._seek_and_show(frame)

    def _update_scrubber(self):
        if self.total_frames > 0:
            self._scrub_updating = True
            self.scrubber_var.set(self.current_frame_num / self.total_frames * 1000)
            self._scrub_updating = False

    def _goto_clip_start(self):
        self._stop_playback()
        self._seek_and_show(0)

    def _goto_clip_end(self):
        self._stop_playback()
        self._seek_and_show(max(0, self.total_frames - 1))

    def _prev_event(self):
        self._stop_playback()
        clip = self.clip_list[self.current_clip_idx] if self.clip_list else None
        if not clip:
            return
        evs = self.clip_events.get(clip, [])
        for ev in reversed(evs):
            if ev['start_frame'] < self.current_frame_num - 5:
                self._seek_and_show(max(0, ev['start_frame'] - 20))
                return
        if evs:
            self._seek_and_show(max(0, evs[0]['start_frame'] - 20))

    def _next_event(self):
        self._stop_playback()
        clip = self.clip_list[self.current_clip_idx] if self.clip_list else None
        if not clip:
            return
        evs = self.clip_events.get(clip, [])
        for ev in evs:
            if ev['start_frame'] > self.current_frame_num + 5:
                self._seek_and_show(max(0, ev['start_frame'] - 20))
                return

    def _prev_clip(self):
        if not self.clip_list:
            return
        self.current_clip_idx = (self.current_clip_idx - 1) % len(self.clip_list)
        self._load_clip(self.clip_list[self.current_clip_idx])

    def _next_clip(self):
        if not self.clip_list:
            return
        self.current_clip_idx = (self.current_clip_idx + 1) % len(self.clip_list)
        self._load_clip(self.clip_list[self.current_clip_idx])

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def _draw_timeline(self):
        tl = self.timeline
        tl.delete('all')
        w = tl.winfo_width()
        h = tl.winfo_height()
        if w < 10 or not self.clip_list:
            return

        # Background track
        tl.create_rectangle(0, 0, w, h, fill='#0d0d0d', outline='')
        tl.create_rectangle(4, h//2 - 3, w - 4, h//2 + 3,
                            fill='#333333', outline='')

        clip = self.clip_list[self.current_clip_idx]
        total = max(self.total_frames, 1)

        # Draw event markers
        for ev in self.clip_events.get(clip, []):
            label, color, is_auto, flagged = self._display_label(ev['dir'])
            x1 = int(ev['start_frame'] / total * w)
            x2 = max(x1 + 2, int(ev['end_frame'] / total * w))
            tl.create_rectangle(x1, 4, x2, h - 4, fill=color, outline='')
            if flagged:
                tl.create_rectangle(x1, 4, x1 + 3, h - 4, fill='#ffdd00', outline='')

        # Current position cursor
        cx = int(self.current_frame_num / total * w)
        tl.create_line(cx, 0, cx, h, fill='white', width=2)

    def _on_timeline_click(self, event):
        w = self.timeline.winfo_width()
        if w < 1 or not self.total_frames:
            return
        frac = event.x / w
        frame = int(frac * self.total_frames)
        self._stop_playback()
        self._seek_and_show(frame)

    def _update_time_label(self):
        cur = self.current_frame_num / max(self.fps, 1)
        tot = self.total_frames / max(self.fps, 1)
        cm, cs = int(cur) // 60, int(cur) % 60
        tm, ts = int(tot) // 60, int(tot) % 60
        self.time_lbl.config(text=f"{cm:02d}:{cs:02d} / {tm:02d}:{ts:02d}")
        self.status_lbl.config(
            text=f"Frame {self.current_frame_num} / {self.total_frames}  "
                 f"({self.fps:.1f} fps)")

    # ------------------------------------------------------------------
    # Events list
    # ------------------------------------------------------------------

    def _on_event_select(self, event):
        sel = self.event_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx < len(self._displayed_events):
            ev = self._displayed_events[idx]
            self.current_event_idx = idx
            self._stop_playback()
            src = ev.get('source_clip', '')
            if src:
                self.root.title(f"StreakerPlayer — {os.path.basename(src)}")
            if not ev.get('clip_missing'):
                self._seek_and_show(max(0, ev['start_frame'] - 20))

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def _annotate_current(self, label):
        clip = self.clip_list[self.current_clip_idx] if self.clip_list else None
        if not clip:
            return
        evs = self.clip_events.get(clip, [])
        # Find which event we're currently in or closest to
        best = None
        best_dist = float('inf')
        for ev in evs:
            dist = abs(ev['start_frame'] - self.current_frame_num)
            if dist < best_dist:
                best_dist, best = dist, ev
        if best and best_dist < self.fps * 60:  # within 60 seconds
            self.annotations[best['dir']] = label
            self._refresh_event_list(clip)
            self._draw_timeline()
            color = LABEL_COLORS.get(label, FG)
            self.status_lbl.config(
                text=f"Labeled '{best['name']}' → {label.upper()}",
                fg=color)
            self.root.after(3000, lambda: self.status_lbl.config(
                text="", fg=FG))
        else:
            self.status_lbl.config(text="No event near current position to label", fg='#cc8800')
            self.root.after(3000, lambda: self.status_lbl.config(text="", fg=FG))

    def _refresh_event_list(self, clip):
        clip_evs = self.events if clip is None else self.clip_events.get(clip, [])
        mode = self._sort_mode.get()

        if mode == 'flagged_only':
            evs = [ev for ev in clip_evs
                   if self._display_label(ev['dir'])[3]]
        elif mode == 'flagged_first':
            flagged = [ev for ev in clip_evs if self._display_label(ev['dir'])[3]]
            rest    = [ev for ev in clip_evs if not self._display_label(ev['dir'])[3]]
            evs = flagged + rest
        else:
            evs = list(clip_evs)

        self._displayed_events = evs
        self.event_listbox.delete(0, 'end')
        for ev in evs:
            mm = int(ev['start_sec']) // 60
            ss = int(ev['start_sec']) % 60
            label, color, is_auto, flagged = self._display_label(ev['dir'])
            dur = ev['end_sec'] - ev['start_sec']
            prefix = '~' if is_auto else ' '
            flag   = ' ⚑' if flagged else ''
            self.event_listbox.insert('end',
                f"  {mm:02d}:{ss:02d}  {dur:4.1f}s  {prefix}{label[:9]}{flag}")
            self.event_listbox.itemconfig('end', fg=color)

    def _on_sort_change(self):
        clip = self.clip_list[self.current_clip_idx] if self.clip_list else None
        if clip:
            self._refresh_event_list(clip)

    def _save_annotations(self):
        if not self.annot_path:
            return
        try:
            with open(self.annot_path, 'w') as f:
                json.dump(self.annotations, f, indent=2)
            messagebox.showinfo("Saved",
                f"Annotations saved to:\n{self.annot_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------------------------------------------------------------------
    # Frame grab / astrometry export
    # ------------------------------------------------------------------

    def _grab_dir(self):
        if not self.clip_list:
            return None
        clip = self.clip_list[self.current_clip_idx]
        d = os.path.join(os.path.dirname(clip), 'astrometry_grabs')
        os.makedirs(d, exist_ok=True)
        return d

    def _grab_frame(self):
        if self._last_raw_frame is None:
            return
        d = self._grab_dir()
        if not d:
            return
        fname = f"frame_{self.current_frame_num:06d}.png"
        cv2.imwrite(os.path.join(d, fname), self._last_raw_frame)
        self.status_lbl.config(text=f"Saved: {fname}")

    def _grab_series(self):
        if not self.clip_list:
            return

        # Dialog: frame count + stack mode
        dlg = tk.Toplevel(self.root)
        dlg.title("Stack Grab")
        dlg.configure(bg='#1a1a1a')
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text="Frames to grab:", bg='#1a1a1a', fg='#cccccc',
                 font=('Arial', 9)).grid(row=0, column=0, padx=12, pady=8, sticky='w')
        count_var = tk.IntVar(value=20)
        tk.Spinbox(dlg, from_=1, to=5000, textvariable=count_var, width=6,
                   bg='#2a2a2a', fg='white', insertbackground='white',
                   buttonbackground='#333').grid(row=0, column=1, padx=8, pady=8, sticky='w')

        tk.Label(dlg, text="Stack mode:", bg='#1a1a1a', fg='#cccccc',
                 font=('Arial', 9)).grid(row=1, column=0, padx=12, pady=4, sticky='w')
        mode_var = tk.StringVar(value='max')
        tk.Radiobutton(dlg, text="Max — brightest pixel (stars pop)",
                       variable=mode_var, value='max',
                       bg='#1a1a1a', fg='#cccccc', selectcolor='#333',
                       activebackground='#1a1a1a', activeforeground='white',
                       font=('Arial', 9)).grid(row=1, column=1, padx=8, sticky='w')
        tk.Radiobutton(dlg, text="Mean — noise reduction",
                       variable=mode_var, value='mean',
                       bg='#1a1a1a', fg='#cccccc', selectcolor='#333',
                       activebackground='#1a1a1a', activeforeground='white',
                       font=('Arial', 9)).grid(row=2, column=1, padx=8, sticky='w')

        confirmed = [False]
        def on_ok():
            confirmed[0] = True
            dlg.destroy()
        def on_cancel():
            dlg.destroy()

        btn_row = tk.Frame(dlg, bg='#1a1a1a')
        btn_row.grid(row=3, column=0, columnspan=2, pady=10)
        tk.Button(btn_row, text="Grab", command=on_ok,
                  bg='#334433', fg='white', relief='flat', padx=12).pack(side='left', padx=6)
        tk.Button(btn_row, text="Cancel", command=on_cancel,
                  bg='#333333', fg='white', relief='flat', padx=12).pack(side='left', padx=6)

        dlg.bind('<Return>', lambda e: on_ok())
        dlg.bind('<Escape>', lambda e: on_cancel())
        self.root.wait_window(dlg)

        if not confirmed[0]:
            return

        count = count_var.get()
        mode  = mode_var.get()

        try:
            d = self._grab_dir()
            if not d:
                messagebox.showerror("Stack Grab", "No clip loaded.")
                return
            clip = self.clip_list[self.current_clip_idx]
            subfolder = os.path.join(d, f"stack_{self.current_frame_num:06d}_{count}f")
            os.makedirs(subfolder, exist_ok=True)

            self._stop_playback()
            temp = FrameReader(FFMPEG_PATH, clip, self.frame_w, self.frame_h, self.fps)
            temp.seek(self.current_frame_num)
            saved = 0
            stack_img = None
            for i in range(count):
                frame = temp.read()
                if frame is None:
                    break
                arr = frame.astype(np.float32)
                if stack_img is None:
                    stack_img = arr
                elif mode == 'max':
                    stack_img = np.maximum(stack_img, arr)
                else:
                    stack_img = stack_img + arr
                saved += 1
                self.status_lbl.config(text=f"Stacking… {saved}/{count}")
                self.root.update_idletasks()
            temp.close()

            if saved == 0:
                messagebox.showerror("Stack Grab", f"No frames could be read from:\n{clip}")
                return

            if mode == 'mean':
                stack_img = stack_img / saved
            result = stack_img.clip(0, 255).astype(np.uint8)
            fname = f"stack_{mode}.png"
            self._last_stack_path = os.path.join(subfolder, fname)
            cv2.imwrite(self._last_stack_path, result)
            self.status_lbl.config(
                text=f"Stacked {saved} frames ({mode}) → {os.path.basename(subfolder)}/")
            messagebox.showinfo("Stack Grab Done",
                                f"Stacked {saved} frames ({mode}):\n{self._last_stack_path}")
            _open_in_browser(self._last_stack_path)
        except Exception as e:
            import traceback
            messagebox.showerror("Stack Grab Error", traceback.format_exc())

    # ------------------------------------------------------------------
    # Astrometry launch
    # ------------------------------------------------------------------

    def _current_event(self):
        clip = self.clip_list[self.current_clip_idx] if self.clip_list else None
        if not clip:
            return None
        evs = self.clip_events.get(clip, [])
        for ev in evs:
            if ev['start_frame'] <= self.current_frame_num <= ev['end_frame']:
                return ev
        if evs:
            return min(evs, key=lambda e: abs(e['start_frame'] - self.current_frame_num))
        return None

    def _launch_astro(self):
        cfg_dir = (os.path.dirname(sys.executable)
                   if getattr(sys, 'frozen', False)
                   else os.path.dirname(os.path.abspath(__file__)))
        extra = ['--config-dir', cfg_dir]
        if self._last_stack_path and os.path.exists(self._last_stack_path):
            extra += ['--stack', self._last_stack_path]
        ev = self._current_event()
        if ev:
            extra += ['--event', ev['dir']]
        elif self.clip_list:
            clip_dir = os.path.dirname(self.clip_list[self.current_clip_idx])
            if os.path.isfile(os.path.join(clip_dir, 'metadata.json')):
                extra += ['--event', clip_dir]
        try:
            launch_companion('StreakerAstro.py', extra)
        except Exception as e:
            messagebox.showerror("Astro Launch Error", str(e))

    def _launch_local_solve(self):
        if not _LOCAL_SOLVE_AVAILABLE:
            messagebox.showerror("Not Available", "StreakerLocalSolve not found.")
            return
        initial_dir = None
        initial_file = None
        if self._last_stack_path and os.path.exists(self._last_stack_path):
            initial_dir = os.path.dirname(self._last_stack_path)
            initial_file = self._last_stack_path
        elif getattr(self, 'clip_list', None):
            initial_dir = os.path.dirname(self.clip_list[0])
        image_path = filedialog.askopenfilename(
            parent=self.root,
            title="Select image for Local Solve",
            initialdir=initial_dir,
            initialfile=os.path.basename(initial_file) if initial_file else None,
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.fit *.fits"),
                       ("All files", "*.*")],
        )
        if not image_path:
            return
        launch_local_solve(self.root, image_path)

    # ------------------------------------------------------------------
    # Measure mode
    # ------------------------------------------------------------------

    def _load_wcs_from_config(self):
        import sys as _sys
        if getattr(_sys, 'frozen', False):
            cfg_path = os.path.join(os.path.dirname(_sys.executable),
                                    'streaker_astro_config.json')
        else:
            cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'streaker_astro_config.json')
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            stars = cfg.get('identified_stars', [])
            self._wcs_fn = _fit_wcs(stars)
            return len(stars)
        except Exception:
            self._wcs_fn = None
            return 0

    def _canvas_to_image(self, cx, cy):
        if not getattr(self, 'frame_w', None):
            return None, None
        cw = max(self.canvas.winfo_width(), 640)
        ch = max(self.canvas.winfo_height(), 480)
        scale = min(cw / self.frame_w, ch / self.frame_h)
        ox = (cw - int(self.frame_w * scale)) // 2
        oy = (ch - int(self.frame_h * scale)) // 2
        return (cx - ox) / scale, (cy - oy) / scale

    def _image_to_canvas(self, ix, iy):
        if not getattr(self, 'frame_w', None):
            return None, None
        cw = max(self.canvas.winfo_width(), 640)
        ch = max(self.canvas.winfo_height(), 480)
        scale = min(cw / self.frame_w, ch / self.frame_h)
        ox = (cw - int(self.frame_w * scale)) // 2
        oy = (ch - int(self.frame_h * scale)) // 2
        return ox + ix * scale, oy + iy * scale

    def _toggle_measure_mode(self):
        if self._measure_mode:
            self._measure_mode = False
            self._measure_pt_a = None
            self._measure_btn.config(bg='#2a2a2a', fg='#cccccc')
            self.status_lbl.config(text="Measure mode off")
            return
        n = self._load_wcs_from_config()
        if not self._wcs_fn:
            messagebox.showwarning("No WCS",
                f"Need ≥3 identified stars from Local Solve.\n"
                f"Config has {n} star(s).", parent=self.root)
            return
        self._measure_mode = True
        self._measure_pt_a = None
        self._measure_btn.config(bg='#1a4a3a', fg='#00ffcc')
        self.status_lbl.config(
            text=f"📐 Measure mode ({n} stars) — click START of streak")

    def _on_canvas_resize(self, event):
        if self._last_raw_frame is not None:
            self._display_frame(self._last_raw_frame)

    def _on_canvas_click(self, event):
        if not self._measure_mode or not self._wcs_fn:
            return
        ix, iy = self._canvas_to_image(event.x, event.y)
        if ix is None:
            return
        frame_num = getattr(self, 'current_frame_num', 0)
        if self._measure_pt_a is None:
            self._measure_pt_a = (ix, iy, frame_num)
            self.status_lbl.config(
                text="📐 Point A set — navigate to end frame, then click END of streak")
            if self._last_raw_frame is not None:
                self._display_frame(self._last_raw_frame)
        else:
            ax, ay, af = self._measure_pt_a
            ra_a, dec_a = self._wcs_fn(ax, ay)
            ra_b, dec_b = self._wcs_fn(ix, iy)
            sep_deg = _angular_sep_deg(ra_a, dec_a, ra_b, dec_b)
            sep_arcmin = sep_deg * 60
            fps = getattr(self, 'fps', 20.0)
            dt = abs(frame_num - af) / max(fps, 1)
            if dt > 0:
                vel_deg_s   = sep_deg / dt
                vel_arcmin_s = sep_arcmin / dt
                vel_str = (f"Angular velocity:  {vel_deg_s:.3f} °/s  "
                           f"({vel_arcmin_s:.1f} arcmin/s)\n"
                           f"Duration:          {dt:.3f} s  "
                           f"({abs(frame_num - af)} frames @ {fps:.1f} fps)")
            else:
                vel_str = "Angular velocity:  (same frame — navigate between frames for velocity)"
            body = (
                f"Angular separation:  {sep_deg:.4f}°  ({sep_arcmin:.2f} arcmin)\n"
                f"{vel_str}\n\n"
                f"Point A:  pixel ({ax:.0f}, {ay:.0f})  →  RA {ra_a:.4f}°  Dec {dec_a:.4f}°\n"
                f"Point B:  pixel ({ix:.0f}, {iy:.0f})  →  RA {ra_b:.4f}°  Dec {dec_b:.4f}°"
            )
            messagebox.showinfo("Streak Measurement", body, parent=self.root)
            self._measure_pt_a = None
            self.status_lbl.config(
                text=f"📐 Measure mode — click START of next streak")

    def _launch_mask_editor(self):
        try:
            launch_companion('Mask_editor_gui.py')
        except Exception as e:
            messagebox.showerror("Mask Editor Error", str(e))

    def _launch_triangulate(self):
        extra = ['--run-dir', self.run_folder] if (
            self.run_folder and os.path.isdir(self.run_folder)) else []
        try:
            launch_companion('StreakerTriangulate.py', extra)
        except Exception as e:
            messagebox.showerror("Triangulate Error", str(e))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _on_close(self):
        self._stop_playback()
        if self.reader:
            self.reader.close()
        self.root.destroy()


# ── Launch from StreakerDetect or standalone ──────────────────────────

def launch_player(initial_folder=None, mkv_path=None):
    root = tk.Toplevel() if tk._default_root else tk.Tk()
    app = StreakerPlayer(root, initial_folder=initial_folder, initial_mkv=mkv_path)
    if not tk._default_root or tk._default_root is root:
        root.mainloop()


if __name__ == '__main__':
    root = tk.Tk()
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    app = StreakerPlayer(root, initial_folder=folder)
    root.mainloop()
