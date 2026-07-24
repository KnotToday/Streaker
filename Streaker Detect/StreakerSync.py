# StreakerSync.py — Camera time synchronisation for multi-camera triangulation
#
# Workflow:
#   1. Load each camera's video file
#   2. Scrub to the frame where a GPS clock is visible
#   3. Type the UTC time shown on the clock
#   4. Enter the camera's GPS coordinates
#   5. Save session JSON → used by StreakerTriangulate for event matching
#
# Output: sync_session.json
#   { "cameras": [ { "video", "ref_frame", "ref_utc", "fps", "gps": {lat,lon,alt} }, ... ] }

import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import threading
import queue
import json
import os
import numpy as np
import cv2
from PIL import Image, ImageTk
from datetime import datetime, timezone

from platform_utils import FFMPEG_PATH, NO_WINDOW, find_ffprobe

# ── Appearance ────────────────────────────────────────────────────────────────

BG      = '#1a1a2e'
BG2     = '#16213e'
FG      = '#e0e0e0'
ACCENT  = '#0f3460'
DIM     = '#555555'
OK_COL  = '#00cc88'
WARN    = '#cc8800'

MAX_CAMERAS = 3

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ffprobe_info(video_path):
    ffprobe = find_ffprobe(FFMPEG_PATH)
    try:
        out = subprocess.check_output(
            [ffprobe, '-v', 'quiet', '-print_format', 'json',
             '-show_streams', '-show_format', video_path],
            stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
        info = json.loads(out)
        fmt_dur = float(info.get('format', {}).get('duration', 0) or 0)
        for s in info.get('streams', []):
            if s.get('codec_type') == 'video':
                w   = int(s.get('width', 1920))
                h   = int(s.get('height', 1080))
                fr  = s.get('r_frame_rate', '120/1')
                num, den = fr.split('/')
                fps = float(num) / float(den)
                nb  = s.get('nb_frames')
                dur = float(s.get('duration', 0) or 0) or fmt_dur
                frames = int(nb) if nb else (int(dur * fps) if dur else 0)
                return fps, frames, w, h
    except Exception:
        pass
    return 120.0, 0, 3840, 2160


class _FrameReader:
    """Minimal seek-and-display reader (no buffer needed for sync tool)."""

    def __init__(self, video_path, width, height, fps):
        self.video_path  = video_path
        self.width       = width
        self.height      = height
        self.fps         = fps
        self._nbytes     = width * height * 3
        self._proc       = None
        self._seek_q     = queue.Queue()
        self._worker     = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _launch(self, frame_num):
        ss = frame_num / max(self.fps, 1)
        return subprocess.Popen(
            [FFMPEG_PATH, '-ss', f'{ss:.4f}', '-i', self.video_path,
             '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-an', 'pipe:1'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=NO_WINDOW)

    @staticmethod
    def _kill(proc):
        if proc:
            try: proc.stdout.close()
            except Exception: pass
            try: proc.kill(); proc.wait(timeout=2)
            except Exception: pass

    def _run(self):
        proc = None
        while True:
            item = self._seek_q.get()
            if item is None:
                self._kill(proc)
                return
            frame_num, cb = item
            while not self._seek_q.empty():
                try:
                    nxt = self._seek_q.get_nowait()
                    if nxt is None:
                        self._kill(proc)
                        cb(None, frame_num)
                        return
                    frame_num, cb = nxt
                except queue.Empty:
                    break
            self._kill(proc)
            p   = self._launch(frame_num)
            raw = p.stdout.read(self._nbytes)
            if len(raw) == self._nbytes:
                proc = p
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    self.height, self.width, 3).copy()
                cb(frame, frame_num)
            else:
                self._kill(p)
                proc = None
                cb(None, frame_num)

    def seek(self, frame_num, callback):
        self._seek_q.put((frame_num, callback))

    def close(self):
        self._seek_q.put(None)


# ── Main app ──────────────────────────────────────────────────────────────────

class StreakerSync:

    def __init__(self, root):
        self.root = root
        self.root.title("Streaker Sync")
        self.root.configure(bg=BG)
        self.root.state('zoomed')

        # Per-camera state
        self.readers     = [None] * MAX_CAMERAS
        self.cam_data    = [self._empty_cam() for _ in range(MAX_CAMERAS)]
        self.active_cam  = 0
        self.frame_num   = 0
        self.total_frames = 0
        self.fps         = 120.0
        self.tk_image    = None
        self._session_path = None

        self._build_ui()

    @staticmethod
    def _empty_cam():
        return {
            'video':     None,
            'ref_frame': None,
            'ref_utc':   None,
            'fps':       None,
            'gps':       {'lat': '', 'lon': '', 'alt': ''},
        }

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill='x', side='top', padx=6, pady=(4, 0))

        tk.Label(top, text="STREAKER SYNC", bg=BG, fg='white',
                 font=('Arial', 10, 'bold')).pack(side='left', padx=4)
        tk.Label(top, text="│", bg=BG, fg=DIM).pack(side='left', padx=2)

        tk.Button(top, text="📂 Load Session",
                  command=self._load_session,
                  bg='#334455', fg='white', relief='flat',
                  padx=6, pady=2).pack(side='left', padx=2)
        tk.Button(top, text="💾 Save Session",
                  command=self._save_session,
                  bg='#224422', fg='white', relief='flat',
                  padx=6, pady=2).pack(side='left', padx=2)

        # ── Camera tabs ──────────────────────────────────────────────────────
        tab_row = tk.Frame(self.root, bg=BG2)
        tab_row.pack(fill='x', side='top', padx=0, pady=(4, 0))

        self._tab_btns = []
        for i in range(MAX_CAMERAS):
            b = tk.Button(tab_row, text=f"Camera {i+1}",
                          command=lambda idx=i: self._switch_cam(idx),
                          bg=ACCENT if i == 0 else BG2,
                          fg='white', relief='flat', padx=12, pady=4,
                          font=('Arial', 9, 'bold'))
            b.pack(side='left', padx=1)
            self._tab_btns.append(b)

        self._tab_status = [
            tk.Label(tab_row, text="⏳ no video", bg=BG2, fg=DIM,
                     font=('Arial', 7))
            for _ in range(MAX_CAMERAS)
        ]
        # Status labels shown below each tab button — packed separately
        self._rebuild_tab_status()

        # ── Bottom panels (packed before main so expand works) ────────────────

        # Controls row
        ctrl = tk.Frame(self.root, bg='#0d0d0d')
        ctrl.pack(fill='x', side='bottom', pady=2)

        tk.Button(ctrl, text="|◀", command=self._goto_start,
                  bg='#222', fg='white', relief='flat', width=3).pack(side='left', padx=2, pady=2)
        tk.Button(ctrl, text="−10s", command=lambda: self._step_seconds(-10),
                  bg='#222', fg='white', relief='flat', width=5).pack(side='left', padx=2)
        tk.Button(ctrl, text="◀◀", command=lambda: self._step(-10),
                  bg='#222', fg='white', relief='flat', width=3).pack(side='left', padx=2)
        tk.Button(ctrl, text="◀", command=lambda: self._step(-1),
                  bg='#222', fg='white', relief='flat', width=3).pack(side='left', padx=2)
        tk.Button(ctrl, text="▶", command=lambda: self._step(1),
                  bg='#222', fg='white', relief='flat', width=3).pack(side='left', padx=2)
        tk.Button(ctrl, text="▶▶", command=lambda: self._step(10),
                  bg='#222', fg='white', relief='flat', width=3).pack(side='left', padx=2)
        tk.Button(ctrl, text="+10s", command=lambda: self._step_seconds(10),
                  bg='#222', fg='white', relief='flat', width=5).pack(side='left', padx=2)
        tk.Button(ctrl, text="▶|", command=self._goto_end,
                  bg='#222', fg='white', relief='flat', width=3).pack(side='left', padx=2)

        self.frame_lbl = tk.Label(ctrl, text="Frame 0 / 0",
                                  bg='#0d0d0d', fg='#888', font=('Courier', 8))
        self.frame_lbl.pack(side='left', padx=12)

        # Scrubber
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

        # Sync panel
        sync_panel = tk.Frame(self.root, bg=BG2)
        sync_panel.pack(fill='x', side='bottom', padx=6, pady=(0, 2))

        # GPS row
        gps_row = tk.Frame(sync_panel, bg=BG2)
        gps_row.pack(fill='x', padx=4, pady=(6, 2))
        tk.Label(gps_row, text="Camera GPS →", bg=BG2, fg=DIM,
                 font=('Arial', 8)).pack(side='left', padx=(0, 6))
        tk.Label(gps_row, text="Lat:", bg=BG2, fg=FG, font=('Arial', 8)).pack(side='left')
        self.lat_var = tk.StringVar()
        tk.Entry(gps_row, textvariable=self.lat_var, width=12,
                 bg='#222', fg=FG, insertbackground=FG,
                 font=('Courier', 8)).pack(side='left', padx=(2, 8))
        tk.Label(gps_row, text="Lon:", bg=BG2, fg=FG, font=('Arial', 8)).pack(side='left')
        self.lon_var = tk.StringVar()
        tk.Entry(gps_row, textvariable=self.lon_var, width=12,
                 bg='#222', fg=FG, insertbackground=FG,
                 font=('Courier', 8)).pack(side='left', padx=(2, 8))
        tk.Label(gps_row, text="Alt (m):", bg=BG2, fg=FG, font=('Arial', 8)).pack(side='left')
        self.alt_var = tk.StringVar()
        tk.Entry(gps_row, textvariable=self.alt_var, width=7,
                 bg='#222', fg=FG, insertbackground=FG,
                 font=('Courier', 8)).pack(side='left', padx=(2, 0))

        # Clock / sync row
        clock_row = tk.Frame(sync_panel, bg=BG2)
        clock_row.pack(fill='x', padx=4, pady=(2, 6))
        tk.Label(clock_row, text="UTC clock shown:", bg=BG2, fg=FG,
                 font=('Arial', 8)).pack(side='left', padx=(0, 4))
        self.utc_var = tk.StringVar(value='YYYY-MM-DD HH:MM:SS.mmm')
        utc_entry = tk.Entry(clock_row, textvariable=self.utc_var, width=26,
                             bg='#222', fg=FG, insertbackground=FG,
                             font=('Courier', 8))
        utc_entry.pack(side='left', padx=(0, 8))
        utc_entry.bind('<FocusIn>', self._clear_utc_placeholder)

        tk.Button(clock_row, text="📍 Set Sync Point",
                  command=self._set_sync,
                  bg='#1a3a5a', fg='white', relief='flat',
                  padx=8, pady=2).pack(side='left', padx=4)

        tk.Button(clock_row, text="📁 Load Video",
                  command=self._load_video,
                  bg='#334455', fg='white', relief='flat',
                  padx=8, pady=2).pack(side='left', padx=4)

        self.status_lbl = tk.Label(clock_row, text="No video loaded",
                                   bg=BG2, fg=DIM, font=('Courier', 8))
        self.status_lbl.pack(side='left', padx=12)

        # ── Main canvas (packed last — fills remaining space) ─────────────────
        self.canvas = tk.Canvas(self.root, bg='black', cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', self._on_resize)

        # Keyboard bindings
        self.root.bind('<Left>',  lambda e: self._step(-1))
        self.root.bind('<Right>', lambda e: self._step(1))
        self.root.bind('<Prior>', lambda e: self._step(-100))   # Page Up
        self.root.bind('<Next>',  lambda e: self._step(100))    # Page Down

    def _rebuild_tab_status(self):
        for lbl in self._tab_status:
            lbl.destroy()
        self._tab_status = []
        bar = self._tab_btns[0].master
        for i in range(MAX_CAMERAS):
            cd = self.cam_data[i]
            if cd['ref_utc'] and cd['ref_frame'] is not None:
                txt = f"✓ {os.path.basename(cd['video'])[:20]}" if cd['video'] else "✓ synced"
                col = OK_COL
            elif cd['video']:
                txt = f"▶ {os.path.basename(cd['video'])[:20]}"
                col = WARN
            else:
                txt = "⏳ no video"
                col = DIM
            lbl = tk.Label(bar, text=txt, bg=BG2, fg=col, font=('Arial', 6))
            lbl.pack(side='left', padx=(0, 8))
            self._tab_status.append(lbl)

    # ── Camera switching ──────────────────────────────────────────────────────

    def _switch_cam(self, idx):
        # Save current GPS fields to current camera
        self._save_gps_fields()
        self.active_cam = idx
        for i, b in enumerate(self._tab_btns):
            b.config(bg=ACCENT if i == idx else BG2)
        # Restore GPS fields for new camera
        cd = self.cam_data[idx]
        self.lat_var.set(cd['gps'].get('lat', ''))
        self.lon_var.set(cd['gps'].get('lon', ''))
        self.alt_var.set(cd['gps'].get('alt', ''))
        # Restore video state
        if cd['video']:
            self.total_frames = int((cd['fps'] or 120) * 99999)  # will be overwritten on load
            self.fps = cd['fps'] or 120.0
            self._show_status()
        else:
            self.total_frames = 0
            self.frame_num = 0
            self._clear_canvas()
            self.status_lbl.config(text="No video loaded", fg=DIM)
            self.frame_lbl.config(text="Frame 0 / 0")

    def _save_gps_fields(self):
        cd = self.cam_data[self.active_cam]
        cd['gps']['lat'] = self.lat_var.get().strip()
        cd['gps']['lon'] = self.lon_var.get().strip()
        cd['gps']['alt'] = self.alt_var.get().strip()

    # ── Video loading ─────────────────────────────────────────────────────────

    def _load_video(self):
        path = filedialog.askopenfilename(
            title=f"Load video for Camera {self.active_cam + 1}",
            filetypes=[('Video files', '*.mp4 *.MP4 *.mkv *.MKV *.mov *.MOV *.mxf *.MXF'),
                       ('All files', '*.*')])
        if not path:
            return
        self.status_lbl.config(text="Reading video info…", fg=WARN)
        self.root.update()

        fps, total, w, h = _ffprobe_info(path)
        cd = self.cam_data[self.active_cam]
        cd['video'] = path
        cd['fps']   = fps

        if self.readers[self.active_cam]:
            self.readers[self.active_cam].close()
        self.readers[self.active_cam] = _FrameReader(path, w, h, fps)

        self.fps          = fps
        self.total_frames = total
        self.frame_num    = 0

        self.status_lbl.config(
            text=f"{os.path.basename(path)}  ({w}×{h} @ {fps:.2f}fps, {total} frames)",
            fg=FG)
        self._seek(0)
        self._rebuild_tab_status()

    # ── Seeking / display ─────────────────────────────────────────────────────

    def _seek(self, frame_num):
        reader = self.readers[self.active_cam]
        if not reader:
            return
        self.frame_num = max(0, min(frame_num, max(0, self.total_frames - 1)))
        self.frame_lbl.config(text=f"Frame {self.frame_num} / {self.total_frames}")
        self._scrub_updating = True
        if self.total_frames > 0:
            self.scrubber_var.set(self.frame_num / self.total_frames * 1000)
        self._scrub_updating = False
        reader.seek(self.frame_num, self._on_frame)

    def _on_frame(self, frame, frame_num):
        if frame is not None:
            self.root.after(0, lambda f=frame: self._display(f))

    def _display(self, frame_bgr):
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

        # Overlay sync marker if this is the set sync frame
        cd = self.cam_data[self.active_cam]
        if cd['ref_frame'] == self.frame_num and cd['ref_utc']:
            self.canvas.create_text(
                10, 10, anchor='nw',
                text=f"📍 SYNC FRAME  {cd['ref_utc']}",
                fill=OK_COL, font=('Courier', 10, 'bold'))

    def _clear_canvas(self):
        self.canvas.delete('all')
        cw = max(self.canvas.winfo_width(), 640)
        ch = max(self.canvas.winfo_height(), 480)
        self.canvas.create_text(cw // 2, ch // 2, anchor='center',
                                text="No video loaded", fill=DIM,
                                font=('Arial', 14))

    def _on_resize(self, _e):
        if not self.readers[self.active_cam]:
            self._clear_canvas()

    def _on_scrub(self, val):
        if self._scrub_updating:
            return
        if self.total_frames > 0:
            self._seek(int(float(val) / 1000 * self.total_frames))

    def _step(self, delta):
        self._seek(self.frame_num + delta)

    def _step_seconds(self, s):
        self._seek(self.frame_num + int(s * self.fps))

    def _goto_start(self):
        self._seek(0)

    def _goto_end(self):
        self._seek(max(0, self.total_frames - 1))

    # ── Sync point ────────────────────────────────────────────────────────────

    def _clear_utc_placeholder(self, _e):
        if self.utc_var.get() == 'YYYY-MM-DD HH:MM:SS.mmm':
            self.utc_var.set('')

    def _set_sync(self):
        cd = self.cam_data[self.active_cam]
        if not cd['video']:
            messagebox.showwarning("No video", "Load a video for this camera first.")
            return

        self._save_gps_fields()

        utc_str = self.utc_var.get().strip()
        # Accept several formats
        for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
                    '%H:%M:%S.%f', '%H:%M:%S'):
            try:
                dt = datetime.strptime(utc_str, fmt)
                # If no date provided, use today
                if fmt.startswith('%H'):
                    today = datetime.now(timezone.utc).date()
                    dt = datetime.combine(today, dt.time())
                cd['ref_frame'] = self.frame_num
                cd['ref_utc']   = dt.strftime('%Y-%m-%dT%H:%M:%S.') + \
                                  f"{dt.microsecond // 1000:03d}Z"
                self._show_status()
                self._display_current()
                self._rebuild_tab_status()
                return
            except ValueError:
                continue

        messagebox.showerror("Invalid time",
            "Enter time as:  YYYY-MM-DD HH:MM:SS.mmm\n"
            "           or:  HH:MM:SS.mmm  (today's date assumed)")

    def _show_status(self):
        cd = self.cam_data[self.active_cam]
        if cd['ref_utc'] and cd['ref_frame'] is not None:
            fps = cd['fps'] or self.fps
            self.status_lbl.config(
                text=f"✓ Sync set — frame {cd['ref_frame']} = {cd['ref_utc']}  "
                     f"(fps {fps:.3f})",
                fg=OK_COL)
        elif cd['video']:
            self.status_lbl.config(
                text=f"Video loaded — scrub to clock frame, enter UTC, click Set Sync",
                fg=WARN)

    def _display_current(self):
        reader = self.readers[self.active_cam]
        if reader:
            reader.seek(self.frame_num, self._on_frame)

    # ── Session save / load ───────────────────────────────────────────────────

    def _save_session(self):
        self._save_gps_fields()

        if not self._session_path:
            path = filedialog.asksaveasfilename(
                title="Save sync session",
                defaultextension='.json',
                filetypes=[('JSON', '*.json')],
                initialfile='sync_session.json')
            if not path:
                return
            self._session_path = path

        session = {'cameras': []}
        for i, cd in enumerate(self.cam_data):
            entry = {
                'id':        i + 1,
                'video':     cd['video'],
                'ref_frame': cd['ref_frame'],
                'ref_utc':   cd['ref_utc'],
                'fps':       cd['fps'],
                'gps': {
                    'lat': self._parse_float(cd['gps'].get('lat')),
                    'lon': self._parse_float(cd['gps'].get('lon')),
                    'alt': self._parse_float(cd['gps'].get('alt')),
                },
            }
            session['cameras'].append(entry)

        with open(self._session_path, 'w') as f:
            json.dump(session, f, indent=2)

        messagebox.showinfo("Saved", f"Session saved to:\n{self._session_path}")

    def _load_session(self):
        path = filedialog.askopenfilename(
            title="Load sync session",
            filetypes=[('JSON', '*.json'), ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path) as f:
                session = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self._session_path = path
        for i, entry in enumerate(session.get('cameras', [])[:MAX_CAMERAS]):
            cd = self.cam_data[i]
            cd['video']     = entry.get('video')
            cd['ref_frame'] = entry.get('ref_frame')
            cd['ref_utc']   = entry.get('ref_utc')
            cd['fps']       = entry.get('fps')
            gps = entry.get('gps', {})
            cd['gps'] = {
                'lat': str(gps.get('lat', '')),
                'lon': str(gps.get('lon', '')),
                'alt': str(gps.get('alt', '')),
            }
            # Reload video reader if file still exists
            if cd['video'] and os.path.exists(cd['video']):
                fps, total, w, h = _ffprobe_info(cd['video'])
                if self.readers[i]:
                    self.readers[i].close()
                self.readers[i] = _FrameReader(cd['video'], w, h, fps)
                if i == self.active_cam:
                    self.fps = fps
                    self.total_frames = total

        self._switch_cam(self.active_cam)
        self._rebuild_tab_status()

    @staticmethod
    def _parse_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _on_close(self):
        for r in self.readers:
            if r:
                try: r.close()
                except Exception: pass
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = StreakerSync(root)
    root.protocol("WM_DELETE_WINDOW", app._on_close)
    root.mainloop()


if __name__ == '__main__':
    main()
