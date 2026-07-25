# StreakerReel.py — Starred-event highlight reel builder.
#
# Scan:    finds detect folders with _flags.json but no compiled reel yet.
# Compile: concatenates starred clips (smallest → largest streak) per folder,
#          saves as _reel.mp4 inside the detect folder.
# Rank:    shows all compiled folders in two simultaneous rankings —
#          by largest max-streak and by highest event count.

import os
import json
import math
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
_mb_showerror   = messagebox.showerror
_mb_showwarning = messagebox.showwarning
_CTRLC_TIP = "\n\nTip: press Ctrl+C to copy this message."
messagebox.showerror   = lambda t, m, **k: _mb_showerror(t, str(m) + _CTRLC_TIP, **k)
messagebox.showwarning = lambda t, m, **k: _mb_showwarning(t, str(m) + _CTRLC_TIP, **k)

from platform_utils import FFMPEG_PATH, NO_WINDOW

BG     = '#1a1a1a'
BG2    = '#111111'
BG3    = '#0d0d0d'
FG     = '#cccccc'
ACCENT = '#336699'
GREEN  = '#226622'
DIM    = '#555555'

REEL_FILENAME = '_reel.mp4'


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _streak_px(metadata: dict) -> float:
    scale = metadata.get('detect_scale', 0.5)
    best = 0.0
    for det in metadata.get('detections', []):
        for bbox in det.get('bboxes', []):
            _, _, w, h = bbox
            d = math.sqrt(w * w + h * h) / scale
            if d > best:
                best = d
    return best


def scan_root(root_dir: str) -> list[dict]:
    """
    Walk root_dir and return one dict per detect folder that contains _flags.json.
    Each dict:
      folder, display_name, event_count, max_streak_px, starred_clips, has_reel, reel_path
    """
    results = []
    for dirpath, _, filenames in os.walk(root_dir):
        if '_flags.json' not in filenames:
            continue
        flags_path = os.path.join(dirpath, '_flags.json')
        try:
            with open(flags_path) as f:
                starred_paths = json.load(f)
        except Exception:
            continue

        clips = []        # (streak_px, clip_path)
        for ev_path in starred_paths:
            ev_path = ev_path.replace('/', os.sep).replace('\\', os.sep)
            meta_path = os.path.join(ev_path, 'metadata.json')
            clip_path = os.path.join(ev_path, 'clip.mkv')
            if not os.path.isfile(meta_path) or not os.path.isfile(clip_path):
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                continue
            clips.append((_streak_px(meta), clip_path))

        if not clips:
            continue

        clips.sort()                          # smallest streak first
        max_streak = clips[-1][0]
        reel_path  = os.path.join(dirpath, REEL_FILENAME)

        # Display name: "MM-DD-YYYY  /  detect_YYYYMMDD_HHMMSS"
        date_part   = os.path.basename(os.path.dirname(dirpath))
        detect_part = os.path.basename(dirpath)
        display     = f"{date_part}  /  {detect_part}"

        results.append({
            'folder':        dirpath,
            'display_name':  display,
            'event_count':   len(clips),
            'max_streak_px': max_streak,
            'starred_clips': [c for _, c in clips],   # ordered smallest→largest
            'has_reel':      os.path.isfile(reel_path),
            'reel_path':     reel_path,
        })

    return results


def compile_folder(info: dict) -> None:
    """Run ffmpeg concat for this folder's starred clips → _reel.mp4."""
    clips = info['starred_clips']
    output = info['reel_path']

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                     delete=False, encoding='utf-8') as f:
        list_path = f.name
        for clip in clips:
            safe = clip.replace("'", r"\'")
            f.write(f"file '{safe}'\n")

    cmd = [
        FFMPEG_PATH, '-y',
        '-f', 'concat', '-safe', '0',
        '-i', list_path,
        '-c', 'copy',
        output,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors='replace')[-600:])
    finally:
        os.unlink(list_path)

    info['has_reel'] = True


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class StreakerReelApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('StreakerReel')
        self.root.configure(bg=BG)
        self.root.geometry('1000x660')

        self.all_folders: list[dict] = []
        self._build_ui()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── top bar ──────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill='x', padx=8, pady=6)

        tk.Label(top, text='STREAKER REEL', bg=BG, fg='white',
                 font=('Arial', 12, 'bold')).pack(side='left', padx=6)

        tk.Button(top, text='Select Root Folder', command=self._pick_folder,
                  bg=ACCENT, fg='white', relief='flat', padx=8).pack(side='left', padx=4)

        self.folder_lbl = tk.Label(top, text='No folder selected',
                                   bg=BG, fg='#888888', font=('Arial', 8))
        self.folder_lbl.pack(side='left', padx=6)

        # ── notebook tabs ────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook',        background=BG,  borderwidth=0)
        style.configure('TNotebook.Tab',    background=BG2, foreground=FG,
                        padding=[10, 4])
        style.map('TNotebook.Tab',
                  background=[('selected', ACCENT)],
                  foreground=[('selected', 'white')])

        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True, padx=8, pady=4)

        self.compile_tab  = tk.Frame(nb, bg=BG)
        self.rankings_tab = tk.Frame(nb, bg=BG)
        nb.add(self.compile_tab,  text='  Compile Queue  ')
        nb.add(self.rankings_tab, text='  Rankings  ')

        self._build_compile_tab()
        self._build_rankings_tab()

        # ── status bar ───────────────────────────────────────────────
        self.prog_var = tk.StringVar(value='Select a root folder to begin.')
        tk.Label(self.root, textvariable=self.prog_var,
                 bg=BG, fg='#aaaaaa', font=('Arial', 8),
                 anchor='w').pack(fill='x', padx=10, pady=(0, 2))

        self.progress = ttk.Progressbar(self.root, mode='indeterminate', length=980)
        self.progress.pack(padx=8, pady=(0, 6))

    # ------------------------------------------------------------------
    # Compile tab
    # ------------------------------------------------------------------

    def _build_compile_tab(self):
        t = self.compile_tab

        # header row
        hdr = tk.Frame(t, bg=BG3)
        hdr.pack(fill='x', padx=4, pady=(4, 0))
        for text, width in [('Detect folder', 55), ('Events', 8), ('Max streak', 12), ('', 12)]:
            tk.Label(hdr, text=text, bg=BG3, fg='#888888',
                     font=('Arial', 8, 'bold'), width=width,
                     anchor='w').pack(side='left', padx=2)

        # scrollable list
        canvas = tk.Canvas(t, bg=BG2, highlightthickness=0)
        vsb    = tk.Scrollbar(t, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True, padx=4, pady=4)

        self.compile_inner  = tk.Frame(canvas, bg=BG2)
        self.compile_window = canvas.create_window((0, 0), window=self.compile_inner, anchor='nw')
        self.compile_inner.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
            lambda e: canvas.itemconfig(self.compile_window, width=e.width))
        self.compile_canvas = canvas

        # bottom buttons
        bot = tk.Frame(t, bg=BG)
        bot.pack(fill='x', padx=4, pady=4)
        tk.Button(bot, text='Compile All Unprocessed',
                  command=self._compile_all,
                  bg=GREEN, fg='white', relief='flat',
                  font=('Arial', 9, 'bold'), padx=10).pack(side='right', padx=4)

        self.compile_count_lbl = tk.Label(bot, text='', bg=BG, fg='#888888', font=('Arial', 8))
        self.compile_count_lbl.pack(side='left', padx=4)

    def _populate_compile_tab(self):
        for w in self.compile_inner.winfo_children():
            w.destroy()

        uncompiled = [f for f in self.all_folders if not f['has_reel']]
        self.compile_count_lbl.config(
            text=f'{len(uncompiled)} unprocessed  /  {len(self.all_folders)} total detect folders')

        for info in uncompiled:
            row = tk.Frame(self.compile_inner, bg=BG2)
            row.pack(fill='x', pady=1)

            tk.Label(row, text=info['display_name'], bg=BG2, fg=FG,
                     font=('Arial', 8), width=55, anchor='w').pack(side='left', padx=4)
            tk.Label(row, text=str(info['event_count']), bg=BG2, fg='#aaddaa',
                     font=('Courier', 8), width=8, anchor='w').pack(side='left')
            tk.Label(row, text=f"{info['max_streak_px']:.0f} px", bg=BG2, fg='#aaccff',
                     font=('Courier', 8), width=12, anchor='w').pack(side='left')
            tk.Button(row, text='Compile',
                      command=lambda i=info: self._compile_one(i),
                      bg='#333355', fg='white', relief='flat',
                      font=('Arial', 7), padx=6).pack(side='left', padx=4)

        if not uncompiled:
            tk.Label(self.compile_inner, text='All folders already compiled.',
                     bg=BG2, fg=DIM, font=('Arial', 9)).pack(pady=20)

    # ------------------------------------------------------------------
    # Rankings tab
    # ------------------------------------------------------------------

    def _build_rankings_tab(self):
        t = self.rankings_tab
        panes = tk.Frame(t, bg=BG)
        panes.pack(fill='both', expand=True, padx=4, pady=4)

        # Left: by streak
        left = tk.Frame(panes, bg=BG)
        left.pack(side='left', fill='both', expand=True, padx=(0, 4))

        tk.Label(left, text='Largest Streak  →  Smallest',
                 bg=BG, fg='#aaccff', font=('Arial', 9, 'bold')).pack(pady=(4, 2))

        self.streak_list = self._make_rank_list(left)

        # Right: by event count
        right = tk.Frame(panes, bg=BG)
        right.pack(side='right', fill='both', expand=True, padx=(4, 0))

        tk.Label(right, text='Most Events  →  Fewest',
                 bg=BG, fg='#aaddaa', font=('Arial', 9, 'bold')).pack(pady=(4, 2))

        self.count_list = self._make_rank_list(right)

    def _make_rank_list(self, parent: tk.Frame) -> tk.Frame:
        canvas = tk.Canvas(parent, bg=BG2, highlightthickness=0)
        vsb    = tk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        inner = tk.Frame(canvas, bg=BG2)
        win   = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(win, width=e.width))
        inner._canvas = canvas   # stash for later resize
        return inner

    def _populate_rankings_tab(self):
        compiled = [f for f in self.all_folders if f['has_reel']]

        # By streak (largest first)
        by_streak = sorted(compiled, key=lambda f: f['max_streak_px'], reverse=True)
        self._fill_rank_list(self.streak_list, by_streak,
                             stat_fn=lambda f: f"{f['max_streak_px']:.0f} px",
                             stat_color='#aaccff')

        # By event count (most first)
        by_count = sorted(compiled, key=lambda f: f['event_count'], reverse=True)
        self._fill_rank_list(self.count_list, by_count,
                             stat_fn=lambda f: f"{f['event_count']} events",
                             stat_color='#aaddaa')

    def _fill_rank_list(self, inner: tk.Frame, folders: list[dict],
                        stat_fn, stat_color: str):
        for w in inner.winfo_children():
            w.destroy()

        if not folders:
            tk.Label(inner, text='No compiled reels yet.',
                     bg=BG2, fg=DIM, font=('Arial', 9)).pack(pady=20)
            return

        for rank, info in enumerate(folders, 1):
            row = tk.Frame(inner, bg=BG2)
            row.pack(fill='x', pady=1)

            tk.Label(row, text=f'#{rank}', bg=BG2, fg=DIM,
                     font=('Courier', 8), width=4, anchor='e').pack(side='left', padx=(4, 2))
            tk.Label(row, text=stat_fn(info), bg=BG2, fg=stat_color,
                     font=('Courier', 8), width=12, anchor='w').pack(side='left')
            tk.Label(row, text=info['display_name'], bg=BG2, fg=FG,
                     font=('Arial', 8), anchor='w').pack(side='left', padx=4, fill='x', expand=True)

    # ------------------------------------------------------------------
    # Folder picking
    # ------------------------------------------------------------------

    def _pick_folder(self):
        initial = (r'T:\Streaker_MKV' if os.path.isdir(r'T:\Streaker_MKV')
                   else r'T:\Dahua_MKV_Streaker' if os.path.isdir(r'T:\Dahua_MKV_Streaker')
                   else os.path.expanduser('~'))
        folder = filedialog.askdirectory(title='Select root folder to scan', initialdir=initial)
        if not folder:
            return
        self.folder_lbl.config(text=folder)
        self.prog_var.set('Scanning…')
        self.root.update()
        self.all_folders = scan_root(folder)
        self._populate_compile_tab()
        self._populate_rankings_tab()
        n_done = sum(1 for f in self.all_folders if f['has_reel'])
        self.prog_var.set(
            f'Found {len(self.all_folders)} detect folders — '
            f'{n_done} compiled, {len(self.all_folders) - n_done} pending.')

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def _compile_one(self, info: dict):
        self.progress.start(12)
        self.prog_var.set(f'Compiling {info["display_name"]}…')
        threading.Thread(target=self._compile_thread,
                         args=([info],), daemon=True).start()

    def _compile_all(self):
        queue = [f for f in self.all_folders if not f['has_reel']]
        if not queue:
            messagebox.showinfo('Nothing to do', 'All folders are already compiled.')
            return
        self.progress.start(12)
        self.prog_var.set(f'Compiling {len(queue)} folder(s)…')
        threading.Thread(target=self._compile_thread,
                         args=(queue,), daemon=True).start()

    def _compile_thread(self, queue: list[dict]):
        errors = []
        for i, info in enumerate(queue, 1):
            self.root.after(0, self.prog_var.set,
                            f'Compiling {i}/{len(queue)}: {info["display_name"]}')
            try:
                compile_folder(info)
            except Exception as exc:
                errors.append(f'{info["display_name"]}: {exc}')

        self.root.after(0, self._on_compile_done, errors)

    def _on_compile_done(self, errors: list[str]):
        self.progress.stop()
        self._populate_compile_tab()
        self._populate_rankings_tab()
        n_done = sum(1 for f in self.all_folders if f['has_reel'])
        if errors:
            self.prog_var.set(f'Done with {len(errors)} error(s). See details.')
            messagebox.showerror('Compile errors', '\n\n'.join(errors))
        else:
            self.prog_var.set(
                f'All done — {n_done}/{len(self.all_folders)} folders compiled.')


if __name__ == '__main__':
    root = tk.Tk()
    app = StreakerReelApp(root)
    root.mainloop()
