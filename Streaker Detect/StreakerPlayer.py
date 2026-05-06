# StreakerPlayer.py — MKV timeline player with event markers and annotations

import os
import json
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import re

from platform_utils import FFMPEG_PATH, HWACCEL_ARGS, find_ffprobe
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
    """Wraps an FFmpeg pipe; supports seek by reopening."""

    def __init__(self, ffmpeg_path, video_path, width, height, fps):
        self.ffmpeg_path = ffmpeg_path
        self.video_path  = video_path
        self.width       = width
        self.height      = height
        self.fps         = fps
        self._proc       = None
        self.current_frame = 0

    def seek(self, frame_num):
        self.close()
        self.current_frame = max(0, frame_num)
        ss = self.current_frame / self.fps
        self._proc = subprocess.Popen([
            self.ffmpeg_path,
            *HWACCEL_ARGS,
            '-ss', f'{ss:.4f}',
            '-i', self.video_path,
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-an', 'pipe:1'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self):
        if not self._proc:
            return None
        nbytes = self.width * self.height * 3
        raw = self._proc.stdout.read(nbytes)
        if len(raw) < nbytes:
            return None
        self.current_frame += 1
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)

    def close(self):
        if self._proc:
            try:
                self._proc.stdout.close()
                self._proc.kill()
                self._proc.wait()
            except Exception:
                pass
            self._proc = None


class StreakerPlayer:
    TIMELINE_H   = 60
    CTRL_H       = 40

    def __init__(self, root, initial_folder=None):
        self.root = root
        self.root.title("Streaker Player")
        self.root.geometry("1200x820")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.run_folder   = None
        self.events       = []       # all loaded events (dicts)
        self.clip_events  = {}       # clip_path -> [event, ...]
        self.clip_list    = []       # ordered list of unique clip paths
        self.annotations  = {}       # event_dir -> label string
        self.annot_path   = None

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

        self.tk_image     = None
        self._frame_q     = queue.Queue(maxsize=4)
        self._reader_thread = None
        self._stop_reader = threading.Event()

        self._build_ui()

        if initial_folder and os.path.isdir(initial_folder):
            self.root.after(100, lambda: self.load_run_folder(initial_folder))

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill='x', side='top', padx=6, pady=4)

        tk.Label(top, text="STREAKER PLAYER", bg=BG, fg='white',
                 font=('Arial', 11, 'bold')).pack(side='left', padx=6)

        tk.Button(top, text="📁 Open Run Folder",
                  command=self._browse_run_folder,
                  bg='#334455', fg='white', relief='flat',
                  padx=8, pady=2).pack(side='left', padx=4)
        tk.Button(top, text="🎬 Open MKV",
                  command=self._browse_mkv,
                  bg='#334455', fg='white', relief='flat',
                  padx=8, pady=2).pack(side='left', padx=4)

        self.folder_lbl = tk.Label(top, text="No folder loaded",
                                   bg=BG, fg='#888888', font=('Arial', 8))
        self.folder_lbl.pack(side='left', padx=6)

        # Clip navigation
        nav = tk.Frame(top, bg=BG)
        nav.pack(side='right', padx=6)
        tk.Button(nav, text="◀ Prev Clip", command=self._prev_clip,
                  bg='#333333', fg='white', relief='flat', padx=6).pack(side='left', padx=2)
        self.clip_lbl = tk.Label(nav, text="—", bg=BG, fg=FG,
                                 font=('Arial', 8), width=30)
        self.clip_lbl.pack(side='left', padx=4)
        tk.Button(nav, text="Next Clip ▶", command=self._next_clip,
                  bg='#333333', fg='white', relief='flat', padx=6).pack(side='left', padx=2)

        # ── Main area: video + event list ────────────────────────────
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True)

        # Video canvas
        self.canvas = tk.Canvas(main, bg='black', cursor='crosshair')
        self.canvas.pack(side='left', fill='both', expand=True)

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

        # Keyboard bindings
        self.root.bind('<Left>',  lambda e: self._step(-1))
        self.root.bind('<Right>', lambda e: self._step(1))
        self.root.bind('<space>', lambda e: self._toggle_play())
        self.root.bind('<comma>',  lambda e: self._prev_event())   # < key
        self.root.bind('<period>', lambda e: self._next_event())   # > key

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

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

        for d in sorted(os.listdir(folder)):
            meta_path = os.path.join(folder, d, 'metadata.json')
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                continue

            clip = meta.get('source_clip', '')
            if not clip or not os.path.exists(clip):
                continue

            fps = meta.get('fps', 20.0)
            ev = {
                'dir':         os.path.join(folder, d),
                'name':        d,
                'source_clip': clip,
                'start_frame': meta.get('start_frame', 0),
                'end_frame':   meta.get('end_frame', 0),
                'start_sec':   meta.get('start_frame', 0) / fps,
                'end_sec':     meta.get('end_frame', 0) / fps,
                'fps':         fps,
                'detections':  meta.get('detections', []),
            }
            self.events.append(ev)
            self.clip_events.setdefault(clip, []).append(ev)

        self.clip_list = sorted(self.clip_events.keys())

        n = len(self.events)
        nc = len(self.clip_list)
        self.folder_lbl.config(
            text=f"{os.path.basename(folder)} — {n} events across {nc} clip(s)")
        self.event_count_lbl.config(text=f"{n} events total")

        if self.clip_list:
            self.current_clip_idx = 0
            self._load_clip(self.clip_list[0])

    def _load_clip(self, clip_path):
        self._stop_playback()
        if self.reader:
            self.reader.close()

        self.fps, self.total_frames, self.frame_w, self.frame_h = \
            ffprobe_info(FFMPEG_PATH, clip_path)
        self.reader = FrameReader(FFMPEG_PATH, clip_path,
                                  self.frame_w, self.frame_h, self.fps)

        name = os.path.basename(clip_path)
        idx  = self.current_clip_idx + 1
        total = len(self.clip_list)
        self.clip_lbl.config(text=f"[{idx}/{total}] {name}")

        # Populate event list
        self.event_listbox.delete(0, 'end')
        clip_evs = self.clip_events.get(clip_path, [])
        for ev in clip_evs:
            mm = int(ev['start_sec']) // 60
            ss = int(ev['start_sec']) % 60
            label = self.annotations.get(ev['dir'], 'unreviewed')
            color = LABEL_COLORS.get(label, LABEL_COLORS['unreviewed'])
            dur = ev['end_sec'] - ev['start_sec']
            self.event_listbox.insert('end',
                f"  {mm:02d}:{ss:02d}  {dur:4.1f}s  {label[:10]}")
            self.event_listbox.itemconfig('end', fg=color)

        self.current_frame_num = 0
        self.current_event_idx = -1
        self._draw_timeline()
        self._seek_and_show(0)

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
        self.folder_lbl.config(text=os.path.basename(path))
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
        if self.reader:
            self.reader.seek(self.current_frame_num)
            frame = self.reader.read()
            if frame is not None:
                self._display_frame(frame)
        self._draw_timeline()
        self._update_time_label()
        self._update_scrubber()

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

    def _overlay_event_info(self, fw, fh, scale, cw, ch):
        clip = self.clip_list[self.current_clip_idx] if self.clip_list else None
        if not clip:
            return
        for ev in self.clip_events.get(clip, []):
            if ev['start_frame'] <= self.current_frame_num <= ev['end_frame']:
                label = self.annotations.get(ev['dir'], 'unreviewed')
                color = LABEL_COLORS.get(label, LABEL_COLORS['unreviewed'])
                # Draw bboxes for this frame
                for det in ev['detections']:
                    if det['frame'] == self.current_frame_num and det.get('bboxes'):
                        for bx, by, bw, bh in det['bboxes']:
                            x1 = int(bx * scale) + (cw - int(fw * scale)) // 2
                            y1 = int(by * scale) + (ch - int(fh * scale)) // 2
                            x2 = x1 + int(bw * scale)
                            y2 = y1 + int(bh * scale)
                            self.canvas.create_rectangle(
                                x1, y1, x2, y2, outline=color, width=2)
                name = ev['name']
                self.canvas.create_text(
                    10, 10, anchor='nw', text=f"EVENT: {name}  [{label}]",
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
        self._play_tick()

    def _stop_playback(self):
        self.playing = False
        self.play_btn.config(text="▶ Play", bg='#006600')
        if self._play_id:
            self.root.after_cancel(self._play_id)
            self._play_id = None

    def _play_tick(self):
        if not self.playing or not self.reader:
            return
        speed = self.play_speed
        # Skip intermediate frames for fast-forward (reads+discards from open pipe)
        if speed > 1.5 and self.reader._proc:
            skip = max(0, int(round(speed)) - 1)
            nbytes = self.frame_w * self.frame_h * 3
            for _ in range(skip):
                raw = self.reader._proc.stdout.read(nbytes)
                if len(raw) < nbytes:
                    self._stop_playback()
                    return
                self.reader.current_frame += 1
        frame = self.reader.read()
        if frame is None:
            self._stop_playback()
            return
        self.current_frame_num = self.reader.current_frame
        self._display_frame(frame)
        self._draw_timeline()
        self._update_time_label()
        self._update_scrubber()
        # Slow-mo uses longer delay; fast-forward uses frame-skipping above
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
            label = self.annotations.get(ev['dir'], 'unreviewed')
            color = LABEL_COLORS.get(label, LABEL_COLORS['unreviewed'])
            x1 = int(ev['start_frame'] / total * w)
            x2 = max(x1 + 2, int(ev['end_frame'] / total * w))
            tl.create_rectangle(x1, 4, x2, h - 4, fill=color, outline='')

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
        clip = self.clip_list[self.current_clip_idx] if self.clip_list else None
        if not clip:
            return
        evs = self.clip_events.get(clip, [])
        idx = sel[0]
        if idx < len(evs):
            ev = evs[idx]
            self.current_event_idx = idx
            self._stop_playback()
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

    def _refresh_event_list(self, clip):
        self.event_listbox.delete(0, 'end')
        for ev in self.clip_events.get(clip, []):
            mm = int(ev['start_sec']) // 60
            ss = int(ev['start_sec']) % 60
            label = self.annotations.get(ev['dir'], 'unreviewed')
            color = LABEL_COLORS.get(label, LABEL_COLORS['unreviewed'])
            dur = ev['end_sec'] - ev['start_sec']
            self.event_listbox.insert('end',
                f"  {mm:02d}:{ss:02d}  {dur:4.1f}s  {label[:10]}")
            self.event_listbox.itemconfig('end', fg=color)

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
    # Cleanup
    # ------------------------------------------------------------------

    def _on_close(self):
        self._stop_playback()
        if self.reader:
            self.reader.close()
        self.root.destroy()


# ── Launch from StreakerDetect or standalone ──────────────────────────

def launch_player(initial_folder=None):
    root = tk.Toplevel() if tk._default_root else tk.Tk()
    app = StreakerPlayer(root, initial_folder=initial_folder)
    if not tk._default_root or tk._default_root is root:
        root.mainloop()


if __name__ == '__main__':
    root = tk.Tk()
    app = StreakerPlayer(root)
    root.mainloop()
