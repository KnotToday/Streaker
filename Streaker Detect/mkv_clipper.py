"""
mkv_clipper.py — Quick MKV clip cutter for Streaker test clips.

Select an MKV, set start time and duration, cut.  Uses stream copy so it's
fast and lossless.  Multiple clips can be queued and cut in one pass.
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from platform_utils import FFMPEG_PATH


BG, FG, ENTRY_BG = '#1a1a1a', '#dddddd', '#2a2a2a'


def parse_time(s):
    """Accept HH:MM:SS, MM:SS, or plain seconds. Returns float seconds."""
    s = s.strip()
    parts = s.split(':')
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        else:
            return float(s)
    except ValueError:
        raise ValueError(f"Invalid time format: '{s}'  (use HH:MM:SS, MM:SS, or seconds)")


def format_time(secs):
    secs = int(secs)
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def cut_clip(source, start_sec, duration_sec, output_path, log_fn):
    cmd = [
        FFMPEG_PATH,
        '-y',
        '-ss', f'{start_sec:.3f}',
        '-i', source,
        '-t', f'{duration_sec:.3f}',
        '-c', 'copy',
        output_path,
    ]
    log_fn(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding='utf-8', errors='replace')
    for line in proc.stdout:
        if line.strip():
            log_fn(line.rstrip())
    proc.wait()
    if proc.returncode == 0:
        log_fn(f"Saved: {output_path}\n")
        return True
    else:
        log_fn(f"ERROR: ffmpeg exited {proc.returncode}\n")
        return False


class MKVClipper:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MKV Clipper")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        lbl_kw   = dict(bg=BG, fg=FG, font=('Arial', 9), anchor='w')
        entry_kw = dict(bg=ENTRY_BG, fg=FG, insertbackground=FG,
                        font=('Arial', 9), relief='flat')

        # ── Source file ──────────────────────────────────────────────────────
        src_row = tk.Frame(self.root, bg=BG)
        src_row.pack(fill='x', padx=8, pady=(8, 2))
        tk.Label(src_row, text='Source MKV:', **lbl_kw).pack(side='left')
        self.v_source = tk.StringVar()
        tk.Entry(src_row, textvariable=self.v_source, width=60, **entry_kw).pack(
            side='left', fill='x', expand=True, padx=4)
        tk.Button(src_row, text='...', command=self._pick_source,
                  bg='#333', fg=FG, relief='flat', padx=6).pack(side='left')

        # ── Output folder ────────────────────────────────────────────────────
        out_row = tk.Frame(self.root, bg=BG)
        out_row.pack(fill='x', padx=8, pady=2)
        tk.Label(out_row, text='Output folder:', **lbl_kw).pack(side='left')
        self.v_outdir = tk.StringVar()
        tk.Entry(out_row, textvariable=self.v_outdir, width=60, **entry_kw).pack(
            side='left', fill='x', expand=True, padx=4)
        tk.Button(out_row, text='...', command=self._pick_outdir,
                  bg='#333', fg=FG, relief='flat', padx=6).pack(side='left')

        # ── Clip entry row ───────────────────────────────────────────────────
        clip_frame = tk.LabelFrame(self.root, text='Clip', bg=BG, fg='#888888',
                                   font=('Arial', 8), relief='flat')
        clip_frame.pack(fill='x', padx=8, pady=4)

        tk.Label(clip_frame, text='Start (HH:MM:SS or s):', **lbl_kw).grid(
            row=0, column=0, sticky='w', padx=6, pady=3)
        self.v_start = tk.StringVar(value='0')
        tk.Entry(clip_frame, textvariable=self.v_start, width=14, **entry_kw).grid(
            row=0, column=1, padx=4)

        tk.Label(clip_frame, text='Duration (HH:MM:SS or s):', **lbl_kw).grid(
            row=0, column=2, sticky='w', padx=6)
        self.v_dur = tk.StringVar(value='60')
        tk.Entry(clip_frame, textvariable=self.v_dur, width=14, **entry_kw).grid(
            row=0, column=3, padx=4)

        tk.Label(clip_frame, text='Output name:', **lbl_kw).grid(
            row=0, column=4, sticky='w', padx=6)
        self.v_name = tk.StringVar(value='clip_01')
        tk.Entry(clip_frame, textvariable=self.v_name, width=20, **entry_kw).grid(
            row=0, column=5, padx=4)

        tk.Button(clip_frame, text='Add to queue', command=self._add_to_queue,
                  bg='#2a4a6a', fg='white', relief='flat', padx=8).grid(
            row=0, column=6, padx=6)

        # ── Queue list ───────────────────────────────────────────────────────
        queue_frame = tk.Frame(self.root, bg=BG)
        queue_frame.pack(fill='both', expand=False, padx=8, pady=2)

        tk.Label(queue_frame, text='Queue:', **lbl_kw).pack(anchor='w')
        self.queue_box = tk.Listbox(queue_frame, bg=ENTRY_BG, fg=FG,
                                    font=('Courier', 8), height=6, relief='flat',
                                    selectbackground='#3a3a3a')
        self.queue_box.pack(fill='both', expand=True, side='left')
        sb = tk.Scrollbar(queue_frame, command=self.queue_box.yview)
        sb.pack(side='left', fill='y')
        self.queue_box.config(yscrollcommand=sb.set)

        remove_btn = tk.Button(queue_frame, text='Remove\nselected',
                               command=self._remove_selected,
                               bg='#333', fg=FG, relief='flat', padx=6)
        remove_btn.pack(side='left', padx=4)

        self._queue = []  # list of (start_sec, dur_sec, name)

        # ── Run / status ─────────────────────────────────────────────────────
        btn_row = tk.Frame(self.root, bg=BG)
        btn_row.pack(fill='x', padx=8, pady=4)
        self.run_btn = tk.Button(btn_row, text='Cut all clips', command=self._run,
                                 bg='#2a6a2a', fg='white', font=('Arial', 10, 'bold'),
                                 relief='flat', padx=16, pady=4)
        self.run_btn.pack(side='left')
        self.status = tk.Label(btn_row, text='', bg=BG, fg='#aaaaaa', font=('Arial', 9))
        self.status.pack(side='left', padx=10)

        self.log = scrolledtext.ScrolledText(self.root, height=10, bg='#0d0d0d', fg='#cccccc',
                                             font=('Courier', 8), relief='flat')
        self.log.pack(fill='both', expand=True, padx=8, pady=(0, 8))

    # ── helpers ──────────────────────────────────────────────────────────────

    def _pick_source(self):
        p = filedialog.askopenfilename(filetypes=[('MKV', '*.mkv'), ('All', '*.*')],
                                       title='Select source MKV')
        if p:
            self.v_source.set(p)
            if not self.v_outdir.get():
                self.v_outdir.set(os.path.dirname(p))

    def _pick_outdir(self):
        p = filedialog.askdirectory(title='Select output folder')
        if p:
            self.v_outdir.set(p)

    def _add_to_queue(self):
        try:
            start = parse_time(self.v_start.get())
            dur   = parse_time(self.v_dur.get())
        except ValueError as e:
            messagebox.showerror('Invalid time', str(e))
            return
        name = self.v_name.get().strip() or f'clip_{len(self._queue)+1:02d}'
        self._queue.append((start, dur, name))
        self.queue_box.insert('end',
            f"{name}  |  start {format_time(start)}  dur {format_time(dur)}")
        # Auto-increment the name suffix for convenience
        import re
        m = re.search(r'(\d+)$', name)
        if m:
            self.v_name.set(name[:m.start()] + f'{int(m.group())+1:02d}')

    def _remove_selected(self):
        sel = self.queue_box.curselection()
        for i in reversed(sel):
            self.queue_box.delete(i)
            self._queue.pop(i)

    def _log(self, msg):
        self.root.after(0, lambda: (
            self.log.insert('end', msg + '\n'),
            self.log.see('end')
        ))

    def _run(self):
        source = self.v_source.get().strip()
        outdir = self.v_outdir.get().strip()
        if not source or not os.path.isfile(source):
            messagebox.showerror('Missing', 'Please select a valid source MKV.')
            return
        if not outdir:
            messagebox.showerror('Missing', 'Please select an output folder.')
            return
        if not self._queue:
            messagebox.showerror('Empty queue', 'Add at least one clip to the queue.')
            return

        self.run_btn.config(state='disabled')
        self.status.config(text='Cutting...')

        queue_copy = list(self._queue)

        def worker():
            ok = 0
            for i, (start, dur, name) in enumerate(queue_copy):
                self.root.after(0, lambda i=i: self.status.config(
                    text=f'Clip {i+1}/{len(queue_copy)}...'))
                out_path = os.path.join(outdir, name + '.mkv')
                if cut_clip(source, start, dur, out_path, self._log):
                    ok += 1
            self.root.after(0, lambda: (
                self.run_btn.config(state='normal'),
                self.status.config(text=f'Done — {ok}/{len(queue_copy)} clips saved')
            ))

        threading.Thread(target=worker, daemon=True).start()

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    MKVClipper().run()
