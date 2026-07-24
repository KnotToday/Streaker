# StreakerMatch.py — Multi-camera event matcher for triangulation
#
# Workflow:
#   1. Load sync_session.json (from StreakerSync)
#   2. Load detect output folder for each camera
#   3. Click Find Matches — events from different cameras that overlap in
#      absolute UTC time are grouped into match candidates
#   4. Review matches (thumbnails, timestamps, camera count)
#   5. Save matches.json → input for StreakerTriangulate
#
# Match output format:
#   { "matches": [
#       { "start_utc": "...", "end_utc": "...", "n_cams": 2,
#         "cameras": { "1": {event fields}, "2": {...} } }, ... ] }

import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from PIL import Image, ImageTk

# ── Appearance ────────────────────────────────────────────────────────────────

BG      = '#1a1a2e'
BG2     = '#16213e'
BG3     = '#0f0f1e'
FG      = '#e0e0e0'
ACCENT  = '#0f3460'
DIM     = '#555555'
OK_COL  = '#00cc88'
WARN    = '#ffcc00'
ERR     = '#cc3333'

MAX_CAMERAS  = 3
THUMB_W, THUMB_H = 160, 90
CAM_COLORS   = ['#00aaff', '#ff8800', '#cc44ff']   # cam 1/2/3

# ── Time helpers ──────────────────────────────────────────────────────────────

_DT_FMTS = [
    '%Y-%m-%dT%H:%M:%S.%fZ',
    '%Y-%m-%dT%H:%M:%SZ',
    '%Y-%m-%dT%H:%M:%S.%f',
    '%Y-%m-%dT%H:%M:%S',
]

def _parse_utc(s):
    for fmt in _DT_FMTS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None

def _abs_time(frame_num, ref_frame, ref_utc_dt, fps):
    """Absolute UTC datetime for frame_num given sync reference."""
    return ref_utc_dt + timedelta(seconds=(frame_num - ref_frame) / max(fps, 1))

def _fmt_utc(dt):
    if dt is None:
        return '—'
    return dt.strftime('%H:%M:%S.') + f'{dt.microsecond // 1000:03d}'

# ── Matching algorithm ────────────────────────────────────────────────────────

def _events_overlap(a, b, tol_s):
    tol = timedelta(seconds=tol_s)
    return a['start_utc'] <= b['end_utc'] + tol and b['start_utc'] <= a['end_utc'] + tol

def find_matches(events_by_cam, tol_s=2.0):
    """
    Group events from different cameras whose UTC windows overlap.
    Returns list of dicts: {'events': {cam_idx: ev}, 'start_utc', 'end_utc', 'n_cams'}
    Only groups with events from ≥2 cameras are returned.
    """
    all_events = []
    for cam_idx, evs in events_by_cam.items():
        for ev in evs:
            all_events.append((cam_idx, ev))

    n = len(all_events)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        cam_i, ev_i = all_events[i]
        for j in range(i + 1, n):
            cam_j, ev_j = all_events[j]
            if cam_i == cam_j:
                continue
            if _events_overlap(ev_i, ev_j, tol_s):
                union(i, j)

    # Group indices by root
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    results = []
    for idxs in groups.values():
        cam_map = {}
        for i in idxs:
            cam_idx, ev = all_events[i]
            # If same camera appears twice in a group, keep the earlier-starting one
            if cam_idx not in cam_map or ev['start_utc'] < cam_map[cam_idx]['start_utc']:
                cam_map[cam_idx] = ev
        if len(cam_map) < 2:
            continue
        start = min(ev['start_utc'] for ev in cam_map.values())
        end   = max(ev['end_utc']   for ev in cam_map.values())
        results.append({
            'events':    cam_map,
            'start_utc': start,
            'end_utc':   end,
            'n_cams':    len(cam_map),
        })

    results.sort(key=lambda m: m['start_utc'])
    return results

# ── Event scanning ────────────────────────────────────────────────────────────

def scan_events(folder, cam_sync):
    """
    Scan a detect output folder for event_* subdirs.
    cam_sync: {'ref_frame', 'ref_utc' (datetime), 'fps'}
    Returns list of event dicts with start_utc / end_utc computed.
    """
    ref_frame = cam_sync.get('ref_frame')
    ref_utc   = cam_sync.get('ref_utc_dt')
    fps       = cam_sync.get('fps') or 120.0
    events    = []
    if not folder or not os.path.isdir(folder):
        return events
    try:
        entries = sorted(os.listdir(folder))
    except Exception:
        return events
    for d in entries:
        meta_path = os.path.join(folder, d, 'metadata.json')
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            continue
        ev_fps = meta.get('fps', fps)
        sf = meta.get('start_frame', 0)
        ef = meta.get('end_frame', 0)
        if ref_frame is None or ref_utc is None:
            # No sync — use relative seconds from file start
            start_utc = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=sf / max(ev_fps, 1))
            end_utc   = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=ef / max(ev_fps, 1))
        else:
            start_utc = _abs_time(sf, ref_frame, ref_utc, ev_fps)
            end_utc   = _abs_time(ef, ref_frame, ref_utc, ev_fps)
        thumb = os.path.join(folder, d, '_thumbnail.jpg')
        events.append({
            'name':        d,
            'dir':         os.path.join(folder, d),
            'start_frame': sf,
            'end_frame':   ef,
            'fps':         ev_fps,
            'start_utc':   start_utc,
            'end_utc':     end_utc,
            'duration_s':  (ef - sf) / max(ev_fps, 1),
            'source_clip': meta.get('source_clip', ''),
            'thumb':       thumb if os.path.exists(thumb) else None,
        })
    return events

# ── Main app ──────────────────────────────────────────────────────────────────

class StreakerMatch:

    def __init__(self, root):
        self.root   = root
        self.root.title("Streaker Match")
        self.root.configure(bg=BG)
        self.root.state('zoomed')

        self.session_path   = None
        self.session        = None          # raw session dict
        self.cam_sync       = [None] * MAX_CAMERAS   # per-cam sync info
        self.cam_folders    = [None] * MAX_CAMERAS
        self.cam_events     = [[] for _ in range(MAX_CAMERAS)]
        self.matches        = []
        self._thumb_images  = []            # keep refs alive

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill='x', side='top', padx=6, pady=(4, 0))

        tk.Label(top, text="STREAKER MATCH", bg=BG, fg='white',
                 font=('Arial', 10, 'bold')).pack(side='left', padx=4)
        tk.Label(top, text="│", bg=BG, fg=DIM).pack(side='left', padx=2)
        tk.Button(top, text="📂 Load Session",
                  command=self._load_session,
                  bg='#334455', fg='white', relief='flat',
                  padx=6, pady=2).pack(side='left', padx=2)
        tk.Button(top, text="💾 Save Matches",
                  command=self._save_matches,
                  bg='#224422', fg='white', relief='flat',
                  padx=6, pady=2).pack(side='left', padx=2)

        self.summary_lbl = tk.Label(top, text="Load a sync session to begin",
                                    bg=BG, fg=DIM, font=('Courier', 8))
        self.summary_lbl.pack(side='left', padx=12)

        # ── Camera setup row ─────────────────────────────────────────────────
        cam_row = tk.Frame(self.root, bg=BG2)
        cam_row.pack(fill='x', side='top', padx=0, pady=(4, 0))

        self._cam_folder_btns  = []
        self._cam_status_lbls  = []

        for i in range(MAX_CAMERAS):
            col = CAM_COLORS[i]
            cell = tk.Frame(cam_row, bg=BG2)
            cell.pack(side='left', padx=6, pady=4)

            tk.Label(cell, text=f"Camera {i+1}", bg=BG2, fg=col,
                     font=('Arial', 8, 'bold')).pack(anchor='w')
            btn = tk.Button(cell, text="📁 Detect folder",
                            command=lambda idx=i: self._load_folder(idx),
                            bg=ACCENT, fg='white', relief='flat',
                            padx=6, pady=1, font=('Arial', 7))
            btn.pack(anchor='w')
            self._cam_folder_btns.append(btn)

            lbl = tk.Label(cell, text="not loaded", bg=BG2, fg=DIM,
                           font=('Arial', 6))
            lbl.pack(anchor='w')
            self._cam_status_lbls.append(lbl)

        # Tolerance + run button
        run_cell = tk.Frame(cam_row, bg=BG2)
        run_cell.pack(side='right', padx=12, pady=4)

        tol_row = tk.Frame(run_cell, bg=BG2)
        tol_row.pack(anchor='e', pady=(0, 4))
        tk.Label(tol_row, text="Tolerance (s):", bg=BG2, fg=FG,
                 font=('Arial', 8)).pack(side='left', padx=(0, 4))
        self.tol_var = tk.StringVar(value='2.0')
        tk.Entry(tol_row, textvariable=self.tol_var, width=5,
                 bg='#222', fg=FG, insertbackground=FG,
                 font=('Courier', 8)).pack(side='left')

        tk.Button(run_cell, text="🔍 Find Matches",
                  command=self._find_matches,
                  bg='#006600', fg='white', relief='flat',
                  padx=10, pady=4, font=('Arial', 9, 'bold')).pack(anchor='e')

        # ── Match list (scrollable canvas) ───────────────────────────────────
        list_frame = tk.Frame(self.root, bg=BG3)
        list_frame.pack(fill='both', expand=True, padx=4, pady=4)

        tk.Label(list_frame, text="MATCHES", bg=BG3, fg='#888',
                 font=('Arial', 8, 'bold')).pack(anchor='w', padx=6, pady=(4, 0))

        scroll_area = tk.Frame(list_frame, bg=BG3)
        scroll_area.pack(fill='both', expand=True)

        self.list_canvas = tk.Canvas(scroll_area, bg=BG3,
                                     highlightthickness=0)
        sb = tk.Scrollbar(scroll_area, orient='vertical',
                          command=self.list_canvas.yview)
        sb.pack(side='right', fill='y')
        self.list_canvas.pack(side='left', fill='both', expand=True)
        self.list_canvas.configure(yscrollcommand=sb.set)

        self.list_inner = tk.Frame(self.list_canvas, bg=BG3)
        self._list_window = self.list_canvas.create_window(
            (0, 0), window=self.list_inner, anchor='nw')
        self.list_inner.bind('<Configure>', self._on_inner_resize)
        self.list_canvas.bind('<Configure>', self._on_canvas_resize)
        self.list_canvas.bind('<MouseWheel>',
                              lambda e: self.list_canvas.yview_scroll(
                                  -1 * (e.delta // 120), 'units'))

        self._render_placeholder("No matches yet — load folders and click Find Matches")

    def _on_inner_resize(self, _e):
        self.list_canvas.configure(scrollregion=self.list_canvas.bbox('all'))

    def _on_canvas_resize(self, e):
        self.list_canvas.itemconfig(self._list_window, width=e.width)

    # ── Session loading ───────────────────────────────────────────────────────

    def _load_session(self):
        path = filedialog.askopenfilename(
            title="Load sync session",
            filetypes=[('JSON', '*.json'), ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path) as f:
                self.session = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.session_path = path
        for i, cam in enumerate(self.session.get('cameras', [])[:MAX_CAMERAS]):
            ref_utc_dt = _parse_utc(cam.get('ref_utc', ''))
            self.cam_sync[i] = {
                'ref_frame':  cam.get('ref_frame'),
                'ref_utc_dt': ref_utc_dt,
                'fps':        cam.get('fps') or 120.0,
                'gps':        cam.get('gps', {}),
                'video':      cam.get('video', ''),
            }
            synced = ref_utc_dt is not None and cam.get('ref_frame') is not None
            col = OK_COL if synced else WARN
            info = f"{'✓' if synced else '!'} sync ref: {cam.get('ref_utc', 'none')}"
            self._cam_status_lbls[i].config(text=info, fg=col)

        self.summary_lbl.config(
            text=f"Session loaded: {os.path.basename(path)}  — now load detect folders",
            fg=OK_COL)

    # ── Folder loading ────────────────────────────────────────────────────────

    def _load_folder(self, cam_idx):
        folder = filedialog.askdirectory(
            title=f"Select detect output folder for Camera {cam_idx + 1}")
        if not folder:
            return
        self.cam_folders[cam_idx] = folder
        sync = self.cam_sync[cam_idx] or {}
        events = scan_events(folder, sync)
        self.cam_events[cam_idx] = events
        lbl = self._cam_status_lbls[cam_idx]
        if events:
            lbl.config(text=f"✓ {len(events)} events  ({os.path.basename(folder)})",
                       fg=OK_COL)
        else:
            lbl.config(text=f"⚠ 0 events found in folder", fg=WARN)
        self._update_summary()

    def _update_summary(self):
        loaded = sum(1 for evs in self.cam_events if evs)
        total  = sum(len(evs) for evs in self.cam_events)
        self.summary_lbl.config(
            text=f"{loaded} camera(s) loaded, {total} total events",
            fg=FG if loaded else DIM)

    # ── Matching ──────────────────────────────────────────────────────────────

    def _find_matches(self):
        try:
            tol = float(self.tol_var.get())
        except ValueError:
            tol = 2.0

        loaded = {i: evs for i, evs in enumerate(self.cam_events) if evs}
        if len(loaded) < 2:
            messagebox.showwarning("Need more data",
                "Load detect folders for at least 2 cameras first.")
            return

        self.matches = find_matches(loaded, tol_s=tol)
        n3 = sum(1 for m in self.matches if m['n_cams'] == 3)
        n2 = sum(1 for m in self.matches if m['n_cams'] == 2)
        self.summary_lbl.config(
            text=f"{len(self.matches)} matches found  "
                 f"({n3} × 3-cam,  {n2} × 2-cam)  — tolerance {tol}s",
            fg=OK_COL if self.matches else WARN)
        self._render_matches()

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_placeholder(self, text):
        for w in self.list_inner.winfo_children():
            w.destroy()
        self._thumb_images.clear()
        tk.Label(self.list_inner, text=text, bg=BG3, fg=DIM,
                 font=('Arial', 10)).pack(pady=40)

    def _render_matches(self):
        for w in self.list_inner.winfo_children():
            w.destroy()
        self._thumb_images.clear()

        if not self.matches:
            self._render_placeholder("No matches found — try increasing the tolerance")
            return

        for idx, match in enumerate(self.matches):
            self._render_match_card(idx, match)

    def _render_match_card(self, idx, match):
        n_cams = match['n_cams']
        dot_col = OK_COL if n_cams == 3 else WARN

        card = tk.Frame(self.list_inner, bg=BG2,
                        highlightbackground=dot_col, highlightthickness=1)
        card.pack(fill='x', padx=6, pady=3)

        # Header row
        hdr = tk.Frame(card, bg=BG2)
        hdr.pack(fill='x', padx=6, pady=(4, 2))

        tk.Label(hdr, text=f"#{idx+1}", bg=BG2, fg=DIM,
                 font=('Courier', 8)).pack(side='left', padx=(0, 8))
        tk.Label(hdr, text=_fmt_utc(match['start_utc']), bg=BG2, fg=FG,
                 font=('Courier', 10, 'bold')).pack(side='left')
        tk.Label(hdr, text=" UTC", bg=BG2, fg=DIM,
                 font=('Courier', 8)).pack(side='left')

        dur = (match['end_utc'] - match['start_utc']).total_seconds()
        tk.Label(hdr, text=f"  {dur:.2f}s", bg=BG2, fg='#aaaaaa',
                 font=('Courier', 9)).pack(side='left', padx=8)

        cam_str = ', '.join(f"Cam {ci+1}" for ci in sorted(match['events']))
        tk.Label(hdr, text=f"● {cam_str}", bg=BG2, fg=dot_col,
                 font=('Arial', 8, 'bold')).pack(side='left', padx=8)

        # Thumbnail row
        thumb_row = tk.Frame(card, bg=BG2)
        thumb_row.pack(fill='x', padx=6, pady=(0, 6))

        for cam_idx in range(MAX_CAMERAS):
            col = CAM_COLORS[cam_idx]
            cell = tk.Frame(thumb_row, bg=BG2)
            cell.pack(side='left', padx=4)

            ev = match['events'].get(cam_idx)
            if ev is None:
                # Camera didn't catch this event
                blank = tk.Frame(cell, bg='#111', width=THUMB_W, height=THUMB_H)
                blank.pack_propagate(False)
                blank.pack()
                tk.Label(blank, text="—", bg='#111', fg=DIM,
                         font=('Arial', 12)).place(relx=0.5, rely=0.5, anchor='center')
                tk.Label(cell, text=f"Cam {cam_idx+1}  (not detected)",
                         bg=BG2, fg=DIM, font=('Arial', 6)).pack()
            else:
                # Load thumbnail
                img = self._load_thumb(ev.get('thumb'))
                lbl_img = tk.Label(cell, bg='black', image=img,
                                   highlightbackground=col, highlightthickness=2)
                lbl_img.image = img
                lbl_img.pack()
                self._thumb_images.append(img)

                ev_dur = ev['duration_s']
                tk.Label(cell,
                         text=f"Cam {cam_idx+1}  {ev_dur:.2f}s  "
                              f"fr {ev['start_frame']}–{ev['end_frame']}",
                         bg=BG2, fg=col, font=('Arial', 6)).pack()
                tk.Label(cell, text=os.path.basename(ev['dir']),
                         bg=BG2, fg=DIM, font=('Arial', 5)).pack()

    @staticmethod
    def _load_thumb(thumb_path):
        """Load and resize thumbnail, return PhotoImage (or blank placeholder)."""
        try:
            if thumb_path and os.path.exists(thumb_path):
                img = Image.open(thumb_path).convert('RGB')
                img.thumbnail((THUMB_W, THUMB_H), Image.LANCZOS)
                # Pad to exact size
                padded = Image.new('RGB', (THUMB_W, THUMB_H), (17, 17, 30))
                ox = (THUMB_W - img.width)  // 2
                oy = (THUMB_H - img.height) // 2
                padded.paste(img, (ox, oy))
                return ImageTk.PhotoImage(padded)
        except Exception:
            pass
        # Blank placeholder
        blank = Image.new('RGB', (THUMB_W, THUMB_H), (30, 30, 50))
        return ImageTk.PhotoImage(blank)

    # ── Save matches ──────────────────────────────────────────────────────────

    def _save_matches(self):
        if not self.matches:
            messagebox.showwarning("No matches", "Run Find Matches first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save matches",
            defaultextension='.json',
            filetypes=[('JSON', '*.json')],
            initialfile='matches.json')
        if not path:
            return

        out = {'matches': []}
        for m in self.matches:
            entry = {
                'start_utc': m['start_utc'].strftime('%Y-%m-%dT%H:%M:%S.') +
                             f"{m['start_utc'].microsecond // 1000:03d}Z",
                'end_utc':   m['end_utc'].strftime('%Y-%m-%dT%H:%M:%S.') +
                             f"{m['end_utc'].microsecond // 1000:03d}Z",
                'duration_s': (m['end_utc'] - m['start_utc']).total_seconds(),
                'n_cams':    m['n_cams'],
                'cameras':   {},
            }
            for cam_idx, ev in m['events'].items():
                sync = self.cam_sync[cam_idx] or {}
                entry['cameras'][str(cam_idx + 1)] = {
                    'event_dir':   ev['dir'],
                    'start_frame': ev['start_frame'],
                    'end_frame':   ev['end_frame'],
                    'fps':         ev['fps'],
                    'start_utc':   ev['start_utc'].strftime('%Y-%m-%dT%H:%M:%S.') +
                                   f"{ev['start_utc'].microsecond // 1000:03d}Z",
                    'end_utc':     ev['end_utc'].strftime('%Y-%m-%dT%H:%M:%S.') +
                                   f"{ev['end_utc'].microsecond // 1000:03d}Z",
                    'source_clip': ev['source_clip'],
                    'gps':         sync.get('gps', {}),
                }
            out['matches'].append(entry)

        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        messagebox.showinfo("Saved", f"Matches saved to:\n{path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app  = StreakerMatch(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == '__main__':
    main()
