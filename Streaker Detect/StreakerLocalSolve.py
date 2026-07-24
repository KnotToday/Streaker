"""
StreakerLocalSolve.py — offline plate solver
Click 2+ named stars → rough WCS → auto-match catalog → refined solution.
No internet required.
"""

import os
import math
import subprocess
from pathlib import Path
import tkinter as tk
from itertools import combinations as _combos
from tkinter import ttk, messagebox, filedialog
import numpy as np
import cv2
from PIL import Image, ImageTk
from platform_utils import launch_companion

try:
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS as _AstropyWCS
    import astropy.units as u
    _ASTROPY = True
except ImportError:
    _ASTROPY = False

# ── Design tokens ──────────────────────────────────────────────────────────────
BG    = '#0c0c0c'
SURF  = '#141414'
BTN   = '#1e1e1e'
BTN_A = '#2a2a2a'
FG    = '#e0e0e0'
FG2   = '#555555'
GREEN = '#22c55e'
FONT  = 'Segoe UI'

# ── Bright star catalog (name, RA_deg_J2000, Dec_deg_J2000, V_mag) ────────────
BRIGHT_STARS = [
    # mag < 0
    ('Sirius',            101.2875,  -16.7161,  -1.46),
    ('Canopus',            95.9879,  -52.6957,  -0.74),
    ('Rigil Kentaurus',   219.9021,  -60.8340,  -0.27),
    # mag 0–1
    ('Arcturus',          213.9153,   19.1822,  -0.05),
    ('Vega',              279.2347,   38.7836,   0.03),
    ('Capella',            79.1723,   45.9980,   0.08),
    ('Rigel',              78.6345,   -8.2016,   0.12),
    ('Procyon',           114.8255,    5.2250,   0.34),
    ('Achernar',           24.4288,  -57.2368,   0.46),
    ('Betelgeuse',         88.7929,    7.4071,   0.50),
    ('Hadar',             210.9560,  -60.3731,   0.61),
    ('Altair',            297.6958,    8.8683,   0.75),
    ('Acrux',             186.6496,  -63.0991,   0.77),
    ('Aldebaran',          68.9802,   16.5093,   0.85),
    ('Antares',           247.3519,  -26.4320,   0.96),
    ('Spica',             201.2983,  -11.1613,   0.98),
    # mag 1–1.5
    ('Pollux',            116.3290,   28.0262,   1.15),
    ('Fomalhaut',         344.4127,  -29.6223,   1.16),
    ('Mimosa',            191.9303,  -59.6886,   1.25),
    ('Deneb',             310.3579,   45.2803,   1.25),
    ('Regulus',           152.0929,   11.9672,   1.36),
    ('Adhara',            104.6565,  -28.9720,   1.50),
    # mag 1.5–2.0
    ('Castor',            113.6495,   31.8883,   1.58),
    ('Gacrux',            187.7915,  -57.1132,   1.63),
    ('Shaula',            263.4022,  -37.1038,   1.62),
    ('Bellatrix',          81.2828,    6.3497,   1.64),
    ('Elnath',             81.5731,   28.6082,   1.65),
    ('Miaplacidus',       138.2999,  -69.7172,   1.68),
    ('Alnilam',            84.0534,   -1.2019,   1.69),
    ('Alioth',            193.5073,   55.9598,   1.77),
    ('Regor',             122.3833,  -47.3367,   1.78),
    ('Alnitak',            85.1897,   -1.9426,   1.77),
    ('Dubhe',             165.9320,   61.7510,   1.79),
    ('Mirfak',             51.0807,   49.8612,   1.79),
    ('Wezen',             107.0978,  -26.3932,   1.84),
    ('Alkaid',            206.8851,   49.3133,   1.85),
    ('Kaus Australis',    276.0431,  -34.3843,   1.85),
    ('Avior',             125.6284,  -59.5095,   1.86),
    ('Sargas',            264.3298,  -42.9980,   1.86),
    ('Menkalinan',         89.8821,   44.9474,   1.90),
    ('Atria',             247.3520,  -69.0277,   1.91),
    ('Alhena',             99.4278,   16.3993,   1.93),
    ('Alsephina',         131.1757,  -54.7088,   1.93),
    ('Peacock',           306.4122,  -56.7350,   1.94),
    ('Polaris',            37.9546,   89.2641,   1.97),
    ('Mirzam',             95.6749,  -17.9559,   1.98),
    ('Alphard',           141.8968,   -8.6586,   1.98),
    # mag 2.0–2.5
    ('Hamal',              31.7933,   23.4624,   2.00),
    ('Algieba',           154.9929,   19.8415,   2.01),
    ('Diphda',             10.8974,  -17.9866,   2.04),
    ('Nunki',             283.8164,  -26.2967,   2.05),
    ('Menkent',           211.6706,  -36.3699,   2.06),
    ('Mirach',             17.4330,   35.6202,   2.06),
    ('Alpheratz',           2.0969,   29.0905,   2.07),
    ('Saiph',              86.9391,   -9.6697,   2.07),
    ('Rasalhague',        263.7335,   12.5600,   2.08),
    ('Kochab',            222.6763,   74.1555,   2.08),
    ('Algol',              47.0422,   40.9557,   2.09),
    ('Denebola',          177.2649,   14.5721,   2.14),
    ('Tiaki',             340.6670,  -46.8850,   2.11),
    ('Suhail',            136.0000,  -43.4322,   2.21),
    ('Alphecca',          233.6720,   26.7147,   2.23),
    ('Aspidiske',         139.2723,  -59.2750,   2.25),
    ('Mintaka',            83.0016,   -0.2991,   2.25),
    ('Naos',              120.8961,  -40.0032,   2.25),
    ('Sadr',              305.5571,   40.2567,   2.23),
    ('Eltanin',           269.1517,   51.4889,   2.24),
    ('Schedar',            10.1268,   56.5373,   2.24),
    ('Almach',             30.9751,   42.3297,   2.26),
    ('Caph',               2.2945,   59.1498,   2.27),
    ('Mizar',             200.9812,   54.9254,   2.27),
    ('Larawag',           252.5407,  -34.2931,   2.29),
    ('Dschubba',          240.0833,  -22.6217,   2.32),
    ('Enif',              326.0453,    9.8750,   2.38),
    ('Merak',             165.4603,   56.3824,   2.37),
    ('Izar',              221.2470,   27.0743,   2.37),
    ('Ankaa',               6.5716,  -42.3061,   2.40),
    ('Girtab',            265.6221,  -39.0298,   2.41),
    ('Navi',               14.1770,   60.7170,   2.47),
    ('Phecda',            178.4576,   53.6948,   2.43),
    ('Sabik',             257.5946,  -15.7248,   2.43),
    ('Alderamin',         319.6449,   62.5856,   2.45),
    ('Aludra',            111.0238,  -29.3031,   2.45),
    ('Scheat',            345.9886,   28.0828,   2.44),
    ('Gienah Cygni',      305.5700,   33.9703,   2.46),
    ('Markab',            346.1902,   15.2051,   2.49),
    # mag 2.5–3.0
    ('Acrab',             241.3593,  -19.8057,   2.56),
    ('Zosma',             168.5270,   20.5239,   2.56),
    ('Menkar',             45.5699,    4.0896,   2.54),
    ('Han',               247.7282,  -10.5671,   2.54),
    ('Ascella',           285.6532,  -29.8801,   2.60),
    ('Zubeneschamali',    229.2518,   -9.3829,   2.61),
    ('Kraz',              188.5965,  -23.3967,   2.65),
    ('Mahasim',            89.9300,   37.2130,   2.65),
    ('Sheratan',           28.6604,   20.8082,   2.64),
    ('Ruchbah',            21.4539,   60.2353,   2.66),
    ('Lesath',            262.6914,  -37.2960,   2.70),
    ('Mu Velorum',        161.6925,  -49.4201,   2.69),
    ('Muphrid',           208.6714,   18.3977,   2.68),
    ('Kaus Media',        274.4066,  -29.8281,   2.70),
    ('Tarazed',           296.5647,   10.6132,   2.72),
    ('Yed Prior',         243.5861,   -3.6943,   2.74),
    ('Zubenelgenubi',     222.7197,  -16.0417,   2.75),
    ('Cebalrai',          265.8678,    4.5671,   2.77),
    ('Porrima',           190.4152,   -1.4491,   2.74),
    ('Kornephoros',       247.5550,   21.4896,   2.77),
    ('Cursa',              76.9622,   -5.0864,   2.79),
    ('Rastaban',          262.6082,   52.3014,   2.79),
    ('Zeta Her',          250.3215,   31.6028,   2.81),
    ('Kaus Borealis',     276.9928,  -25.4217,   2.81),
    ('Algenib',             3.3086,   15.1836,   2.83),
    ('Vindemiatrix',      195.5440,   10.9592,   2.83),
    ('Fawaris',           297.0451,   45.1305,   2.87),
    ('Tejat',              95.7401,   22.5139,   2.86),
    ('Alcyone',            56.8712,   24.1051,   2.87),
    ('Gienah',            183.7862,  -17.5420,   2.59),
    ('Al Niyat',          245.2973,  -25.5925,   2.89),
    ('Albaldah',          290.9716,  -21.0236,   2.89),
    ('Matar',             355.6851,   30.2214,   2.94),
    ('Gomeisa',           111.7878,    8.2892,   2.90),
    ('Acamar',             44.5654,  -40.3047,   2.91),
    ('Sadalsuud',         322.8897,   -5.5712,   2.91),
    ('Algorab',           187.4660,  -16.5153,   2.95),
    ('Zaurak',             59.5074,  -13.5085,   2.95),
    ('Sadalmelik',        331.4453,   -0.3198,   2.96),
    ('Tianguan',           84.4111,   21.1426,   2.97),
    ('Mebsuda',           100.9830,   25.1313,   2.98),
    ('Almaaz',             75.4922,   43.8233,   2.99),
    ('Deneb Okab',        286.3521,   13.8638,   2.99),
    # mag 3.0–3.5
    ('Pherkad',           230.1821,   71.8342,   3.05),
    ('Albireo',           292.6802,   27.9597,   3.05),
    ('Altais',            288.1387,   67.6615,   3.07),
    ('Alnasl',            271.4523,  -30.4241,   2.99),
    ('Seginus',           218.0196,   38.3083,   3.03),
    ('Sarin',             236.0667,   24.8390,   3.14),
    ('Pi Her',            258.7618,   36.8092,   3.16),
    ('Phi Sagittarii',    290.0973,  -26.9864,   3.17),
    ('Haedus',             76.6287,   41.2343,   3.17),
    ('Errai',             322.1649,   77.6325,   3.21),
    ('Alfirk',            322.1649,   70.5609,   3.23),
    ('Sulafat',           284.7358,   32.6896,   3.24),
    ('Yed Posterior',     244.5796,   -4.6924,   3.24),
    ('Skat',              340.6453,  -15.8211,   3.27),
    ('Propus',             93.7192,   22.5060,   3.28),
    ('Edasich',           231.2323,   58.9659,   3.29),
    ('Megrez',            183.8565,   57.0326,   3.31),
    ('Chertan',           168.5603,   15.4297,   3.34),
    ('Segin',              28.5990,   63.6700,   3.37),
    ('Minelauva',         193.9013,    3.3972,   3.38),
    ('Adhafera',          154.1722,   23.4173,   3.44),
    ('Mu Her',            254.0694,   27.7226,   3.42),
    ('Rasalgethi',        258.6618,   14.3903,   3.48),
    ('Nekkar',            225.4862,   40.3904,   3.50),
    # mag 3.5–4.0
    ('Sheliak',           282.5196,   33.3627,   3.52),
    ('Eta Her',           247.5537,   38.9224,   3.53),
    ('Biham',             337.3826,    6.1978,   3.53),
    ('Alshain',           298.8283,    6.4068,   3.71),
    ('Rotanev',           309.3874,   14.5951,   3.64),
    ('Thuban',            211.0974,   64.3758,   3.65),
    ('Atlas',              57.2908,   24.0535,   3.63),
    ('Prima Hyadum',       67.1542,   15.6279,   3.65),
    ('Electra',            56.2194,   24.1137,   3.70),
    ('Ran',                53.2335,   -9.4584,   3.73),
    ('Sualocin',          309.9087,   15.9122,   3.77),
    ('Homam',             344.4128,   10.8316,   3.40),
    ('Sadalbari',         343.8000,   24.6010,   3.48),
    ('Nusakan',           232.9527,   29.1056,   3.68),
    ('Gamma Her',         245.4754,   19.1530,   3.75),
    ('Xi Her',            263.0513,   29.2478,   3.70),
    ('Iota Her',          238.7695,   46.0263,   3.80),
    ('Theta Her',         274.4043,   37.2505,   3.86),
    ('Omicron Her',       244.9308,   28.7625,   3.83),
    ('Epsilon Her',       240.1621,   30.9264,   3.92),
    ('Alrescha',           30.5119,    2.7636,   3.82),
]

def _load_catalog():
    """Load star_catalog.json if bundled/present; fall back to hardcoded list."""
    import json as _json
    import sys as _sys
    search = [Path(__file__).parent]
    if getattr(_sys, 'frozen', False):
        search.insert(0, Path(_sys.executable).parent)
    for d in search:
        p = d / 'star_catalog.json'
        if p.exists():
            try:
                data = _json.loads(p.read_text())
                # Returns {name: (ra, dec, mag, hr, flamsteed_or_None)}
                return {s['name']: (s['ra'], s['dec'], s['mag'],
                                    s.get('hr'), s.get('fl'))
                        for s in data}
            except Exception:
                pass
    return {name: (ra, dec, mag, None, None) for name, ra, dec, mag in BRIGHT_STARS}


_CATALOG = _load_catalog()
_STAR_NAMES = sorted(_CATALOG.keys())

# Rebuild BRIGHT_STARS from catalog so suggestion loop stays in sync
BRIGHT_STARS = [(n, ra, dec, mag) for n, (ra, dec, mag, *_) in _CATALOG.items()]


# ── Star detection ─────────────────────────────────────────────────────────────

def _detect_stars(gray, max_stars=400, min_sep=20, percentile=99):
    gray = gray.copy()
    gray[:min_sep, :] = 0
    gray[-min_sep:, :] = 0
    gray[:, :min_sep] = 0
    gray[:, -min_sep:] = 0
    threshold = np.percentile(gray, percentile)
    kernel = np.ones((min_sep * 2 + 1, min_sep * 2 + 1), np.uint8)
    dilated = cv2.dilate(gray, kernel)
    mask = (gray >= threshold) & (gray == dilated)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    bright = gray[ys, xs].astype(float)
    order = np.argsort(-bright)
    n = min(max_stars, len(order))
    return list(zip(xs[order[:n]].tolist(),
                    ys[order[:n]].tolist(),
                    bright[order[:n]].tolist()))


def _fmt_ids(hr, fl):
    """Format HR and Flamsteed numbers as a short string, e.g. 'HR 6168  22 Her'."""
    parts = []
    if hr:
        parts.append(f'HR {hr}')
    if fl:
        parts.append(f'FL {fl}')
    return '  '.join(parts)


def _fmt_lb(name, hr, fl, mag):
    """Format a listbox row: name + catalog IDs."""
    ids = _fmt_ids(hr, fl)
    return f"{name:<24}  {ids}" if ids else f"{name:<24}  mag {mag:.1f}"


# ── Main window ────────────────────────────────────────────────────────────────

class LocalSolveWindow:

    def __init__(self, parent, image_path, on_result=None):
        self.image_path = image_path
        self.on_result  = on_result
        self.identified = {}
        self.all_stars  = []
        self._full_disp = None
        self._pending_xy = None
        self._mask      = None
        self._gray_raw  = None
        self.img_w = self.img_h = 0

        # Zoom / pan state
        self.scale      = 1.0   # effective px-per-image-px (base × zoom_f)
        self._zoom_f    = 1.0
        self._off_x     = 0
        self._off_y     = 0
        self._drag_start = None

        self.win = tk.Toplevel(parent)
        self.win.title("Local Plate Solve")
        self.win.configure(bg=BG)
        self.win.geometry('1200x800')
        self.win.state('zoomed')
        self._build_ui()
        self.win.after(120, self._load_image)

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top toolbar
        top = tk.Frame(self.win, bg=SURF, height=40)
        top.pack(fill='x', side='top')
        top.pack_propagate(False)

        def _tbtn(text, cmd, fg=FG, bg=BTN, **kw):
            kw.setdefault('font', (FONT, 9))
            b = tk.Button(top, text=text, command=cmd,
                          bg=bg, fg=fg, relief='flat',
                          padx=12, pady=0,
                          activebackground=BTN_A, activeforeground='#ffffff',
                          cursor='hand2', **kw)
            b.pack(side='right', padx=3, pady=5)
            return b

        _tbtn('Solve ✦', self._solve, fg='#ffffff', bg='#14532d',
              font=(FONT, 9, 'bold'))
        _tbtn('Clear ↺', self._clear_all)
        self._mask_btn = _tbtn('Mask: none', self._load_mask, fg=FG2)
        _tbtn('Draw Mask', self._open_mask_editor)
        _tbtn('Fit', self._fit_to_window)
        _tbtn('Save Image', self._save_display_image, fg=FG2)
        self._show_markers = True
        self._markers_btn = _tbtn('Markers: ON', self._toggle_markers, fg=FG2)

        # Contrast slider (left side of toolbar)
        tk.Label(top, text="Contrast:", bg=SURF, fg=FG2,
                 font=(FONT, 8)).pack(side='left', padx=(10, 2))
        self._contrast_var = tk.DoubleVar(value=1.0)
        self._contrast_scale = tk.Scale(top, from_=0.5, to=3.0, resolution=0.02,
                 orient='horizontal', variable=self._contrast_var,
                 command=lambda _: self._apply_contrast(),
                 bg=SURF, fg=FG, troughcolor=BTN, highlightthickness=0,
                 length=160, showvalue=True, font=(FONT, 7))
        self._contrast_scale.pack(side='left', padx=(0, 6))

        self.status_lbl = tk.Label(top, text="Loading…",
                                   bg=SURF, fg=FG2, font=(FONT, 8), anchor='w')
        self.status_lbl.pack(side='left', padx=10)

        # Status bar — must pack BEFORE canvas so it claims bottom space
        bar = tk.Frame(self.win, bg=SURF, height=26)
        bar.pack(fill='x', side='bottom')
        bar.pack_propagate(False)
        tk.Label(bar, text='Identified:', bg=SURF, fg=FG2,
                 font=(FONT, 8)).pack(side='left', padx=10)
        self.id_lbl = tk.Label(
            bar, text='none — click an orange circle to name it',
            bg=SURF, fg=GREEN, font=(FONT, 8))
        self.id_lbl.pack(side='left')
        tk.Label(bar, text='scroll=zoom  right/middle-drag=pan',
                 bg=SURF, fg=FG2, font=(FONT, 7)).pack(side='right', padx=10)

        # Canvas — fills remaining space
        self.canvas = tk.Canvas(self.win, bg='#000000', cursor='crosshair',
                                highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        self.canvas.bind('<Button-1>',        self._on_click)
        self.canvas.bind('<Button-2>',        self._pan_start)
        self.canvas.bind('<B2-Motion>',       self._pan_drag)
        self.canvas.bind('<ButtonRelease-2>', self._pan_end)
        self.canvas.bind('<Button-3>',        self._pan_start)
        self.canvas.bind('<B3-Motion>',       self._pan_drag)
        self.canvas.bind('<ButtonRelease-3>', self._pan_end)
        self.canvas.bind('<MouseWheel>',      self._on_mousewheel)
        self.canvas.bind('<Configure>',       lambda e: self._render())

    # ── Zoom / pan ─────────────────────────────────────────────────────────────

    def _toggle_markers(self):
        self._show_markers = not self._show_markers
        self._markers_btn.config(
            text='Markers: ON' if self._show_markers else 'Markers: OFF',
            fg=FG2 if self._show_markers else '#f59e0b')
        self._render()

    def _save_display_image(self):
        if self._full_disp is None:
            return
        base = os.path.splitext(self.image_path)[0]
        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg')],
            initialfile=os.path.basename(base) + '_contrast.png',
            initialdir=os.path.dirname(self.image_path),
            parent=self.win)
        if not path:
            return
        cv2.imwrite(path, cv2.cvtColor(self._full_disp, cv2.COLOR_RGB2BGR))
        self.status_lbl.config(text=f"Saved: {os.path.basename(path)}")
        # Store path so StreakerAstro can pre-fill it
        _save_result_to_astro_config({'center_ra': 0, 'center_dec': 0,
                                      '_image_only': True, '_image_path': path})

    def _apply_contrast(self):
        if self._gray_raw is None:
            return
        factor = self._contrast_var.get()
        gray_disp = np.clip(
            (self._gray_raw.astype(np.float32) - 127.0) * factor + 127.0,
            0, 255).astype(np.uint8)
        self._full_disp = cv2.cvtColor(gray_disp, cv2.COLOR_GRAY2RGB)
        self._render()

    def _fit_to_window(self):
        if not self.img_w:
            return
        self.win.update_idletasks()   # ensure canvas has its real size
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        base = min(cw / self.img_w, ch / self.img_h)
        self._zoom_f = 1.0
        dw = int(self.img_w * base)
        dh = int(self.img_h * base)
        self._off_x = max(0, (cw - dw) // 2)
        self._off_y = max(0, (ch - dh) // 2)
        self._render()

    def _on_mousewheel(self, event):
        if not self.img_w:
            return
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        img_x = (event.x - self._off_x) / self.scale
        img_y = (event.y - self._off_y) / self.scale
        self._zoom_f = max(0.05, min(40.0, self._zoom_f * factor))
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        base = min(cw / self.img_w, ch / self.img_h)
        self.scale = base * self._zoom_f
        self._off_x = int(event.x - img_x * self.scale)
        self._off_y = int(event.y - img_y * self.scale)
        self._render()

    def _pan_start(self, event):
        self._drag_start = (event.x, event.y)

    def _pan_drag(self, event):
        if self._drag_start:
            self._off_x += event.x - self._drag_start[0]
            self._off_y += event.y - self._drag_start[1]
            self._drag_start = (event.x, event.y)
            self._render()

    def _pan_end(self, event):
        self._drag_start = None

    # ── Image load ─────────────────────────────────────────────────────────────

    def _open_mask_editor(self):
        extra = [self.image_path] if self.image_path and os.path.isfile(self.image_path) else []
        try:
            launch_companion('Mask_editor_gui.py', extra)
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self.win)

    def _load_mask(self):
        path = filedialog.askopenfilename(
            parent=self.win, title="Select mask PNG",
            filetypes=[("PNG mask", "*.png"), ("All files", "*.*")])
        if not path:
            return
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            messagebox.showerror("Mask Error",
                                 f"Cannot read:\n{path}", parent=self.win)
            return
        self._mask = m
        self._mask_btn.config(
            text=f"Mask: {os.path.basename(path)}", fg=GREEN)
        self._redetect()

    def _redetect(self):
        if self._full_disp is None:
            return
        img_bgr = cv2.imread(self.image_path)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if self._mask is not None:
            m = cv2.resize(self._mask, (gray.shape[1], gray.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
            mb = (m > 0).astype('uint8') * 255
            mb = cv2.erode(mb, np.ones((41, 41), np.uint8))
            gray = cv2.bitwise_and(gray, gray, mask=mb)
        self.all_stars = _detect_stars(gray)
        self.status_lbl.config(
            text=f"{len(self.all_stars)} candidates detected")
        self._render()

    def _load_image(self):
        try:
            self._load_image_inner()
        except Exception as e:
            import traceback
            messagebox.showerror("Load Error", traceback.format_exc(),
                                 parent=self.win)

    def _load_image_inner(self):
        img_bgr = cv2.imread(self.image_path)
        if img_bgr is None:
            messagebox.showerror("Error", f"Cannot read:\n{self.image_path}",
                                 parent=self.win)
            return
        self.img_h, self.img_w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Auto-find mask
        img_dir = os.path.dirname(self.image_path)
        for cand in ('mask.png', 'Mask.png'):
            mp = os.path.join(img_dir, cand)
            if os.path.isfile(mp):
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    self._mask = m
                    self._mask_btn.config(
                        text=f"Mask: {cand}", fg=GREEN)
                break

        if self._mask is not None:
            m = cv2.resize(self._mask, (gray.shape[1], gray.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
            mb = (m > 0).astype('uint8') * 255
            mb = cv2.erode(mb, np.ones((41, 41), np.uint8))
            gray = cv2.bitwise_and(gray, gray, mask=mb)

        self.status_lbl.config(text="Detecting stars…")
        self.win.update_idletasks()
        self.all_stars = _detect_stars(gray)
        self._gray_raw = gray.copy()
        # Sky level from non-masked pixels (lower 20th percentile = background)
        lit = gray[gray > 0]
        self._sky_level = float(np.percentile(lit, 20)) if len(lit) > 100 else 20.0
        # Slider max = factor that clips sky to black: solve (sky-127)*f+127=0
        if self._sky_level < 126:
            black_f = round(127.0 / (127.0 - self._sky_level), 2)
            self._contrast_scale.configure(to=max(1.5, black_f))
        self._contrast_var.set(1.0)
        self._apply_contrast()

        self.status_lbl.config(
            text=f"{len(self.all_stars)} candidates — "
                 f"click an orange circle, name it  |  need ≥ 2 to solve")
        self._fit_to_window()

    # ── Rendering ──────────────────────────────────────────────────────────────

    def _render(self, overlay_wcs=None, overlay_catalog=None):
        if self._full_disp is None:
            return
        cw = max(self.canvas.winfo_width(),  100)
        ch = max(self.canvas.winfo_height(), 100)

        # Compute effective scale (fit-to-window × user zoom)
        base = min(cw / self.img_w, ch / self.img_h)
        self.scale = base * self._zoom_f
        dw = int(self.img_w * self.scale)
        dh = int(self.img_h * self.scale)

        disp = cv2.resize(self._full_disp, (dw, dh),
                          interpolation=cv2.INTER_AREA).copy()

        # Detected candidates — circles (gated by marker toggle)
        if self._show_markers:
            for sx, sy, _ in self.all_stars:
                cx, cy = int(sx * self.scale), int(sy * self.scale)
                cv2.circle(disp, (cx, cy), 12, (255, 140, 0), 1)

        # Catalog overlay after solve
        if overlay_wcs and overlay_catalog:
            for name, ra, dec in overlay_catalog:
                if name in self.identified:
                    continue   # drawn below as a larger green circle
                try:
                    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg,
                                  frame='icrs')
                    px, py = overlay_wcs.world_to_pixel(sc)
                    cx = int(float(px) * self.scale)
                    cy = int(float(py) * self.scale)
                    if not (4 <= cx < dw - 4 and 4 <= cy < dh - 4):
                        continue
                    cv2.circle(disp, (cx, cy), 7, (160, 140, 0), 1)
                    txt_w = int(len(name) * 7)
                    tx = cx - 8 - txt_w if cx + 8 + txt_w > dw - 4 else cx + 8
                    ty = cy + 14 if cy - 4 < 10 else cy - 4
                    cv2.putText(disp, name, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 140, 0),
                                1, cv2.LINE_AA)
                except Exception:
                    pass

        # Named stars (green, larger)
        for name, (ix, iy) in self.identified.items():
            cx = int(ix * self.scale)
            cy = int(iy * self.scale)
            cv2.circle(disp, (cx, cy), 11, (80, 220, 120), 2)
            txt_w = int(len(name) * 9)
            tx = cx - 13 - txt_w if cx + 13 + txt_w > dw - 4 else cx + 13
            ty = cy + 18 if cy - 6 < 10 else cy - 6
            cv2.putText(disp, name, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 120),
                        1, cv2.LINE_AA)

        # Viewport-clip with pan offset
        view = np.zeros((ch, cw, 3), dtype=np.uint8)
        sx = max(0, -self._off_x)
        sy = max(0, -self._off_y)
        dx = max(0,  self._off_x)
        dy = max(0,  self._off_y)
        cpw = min(dw - sx, cw - dx)
        cph = min(dh - sy, ch - dy)
        if cpw > 0 and cph > 0:
            view[dy:dy+cph, dx:dx+cpw] = disp[sy:sy+cph, sx:sx+cpw]

        pil = Image.fromarray(view)
        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self._tk_img)

    # ── Click handling ─────────────────────────────────────────────────────────

    def _on_click(self, event):
        if not self.all_stars and not self.identified:
            return
        img_x = (event.x - self._off_x) / self.scale
        img_y = (event.y - self._off_y) / self.scale
        snap = 30 / self.scale

        # Check if click is near an already-identified star (allow re-labeling)
        existing_name = None
        if self.identified:
            nearest_id = min(self.identified.items(),
                             key=lambda kv: (kv[1][0]-img_x)**2 + (kv[1][1]-img_y)**2)
            dist_id = math.sqrt((nearest_id[1][0]-img_x)**2 + (nearest_id[1][1]-img_y)**2)
            if dist_id <= snap:
                existing_name = nearest_id[0]
                self._pending_xy = nearest_id[1]
                self._show_name_picker(event.x, event.y, existing_name)
                return

        if not self.all_stars:
            return
        nearest = min(self.all_stars,
                      key=lambda s: (s[0]-img_x)**2 + (s[1]-img_y)**2)
        dist = math.sqrt((nearest[0]-img_x)**2 + (nearest[1]-img_y)**2)
        if dist > snap:
            self._pending_xy = (img_x, img_y)
        else:
            self._pending_xy = (nearest[0], nearest[1])

        suggestion = None
        suggest_dist = None
        suggest_mag = None
        suggest_ids = ''
        if len(self.identified) >= 2 and _ASTROPY:
            try:
                rw = self._rough_wcs()
                if rw:
                    best_name, best_d = None, float('inf')
                    px0, py0 = self._pending_xy
                    for _nm, ra, dec, _ in BRIGHT_STARS:
                        if _nm in self.identified:
                            continue
                        sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
                        proj = rw.world_to_pixel(sc)
                        d = math.sqrt((float(proj[0])-px0)**2 + (float(proj[1])-py0)**2)
                        if d < best_d:
                            best_d, best_name = d, _nm
                    if best_name and best_d < 200:
                        suggestion = best_name
                        suggest_dist = int(best_d)
                        suggest_mag = None  # unused now
                        _ce = _CATALOG.get(best_name)
                        suggest_ids = _fmt_ids(_ce[3], _ce[4]) if _ce else ''
            except Exception:
                pass

        # Confidence: is the click inside the convex hull of already-identified stars?
        suggest_confident = True
        if suggestion and len(self.identified) >= 3:
            try:
                pts = np.array(list(self.identified.values()), dtype=np.float32)
                hull = cv2.convexHull(pts)
                px0, py0 = self._pending_xy
                d_hull = cv2.pointPolygonTest(hull, (float(px0), float(py0)), True)
                # d_hull > 0 = inside, < 0 = outside (magnitude = px distance outside)
                suggest_confident = d_hull >= -80
            except Exception:
                pass

        self._show_name_picker(event.x, event.y, suggestion=suggestion,
                               suggest_dist=suggest_dist, suggest_ids=suggest_ids,
                               suggest_confident=suggest_confident)

    def _show_name_picker(self, disp_x, disp_y, current_name=None, suggestion=None,
                          suggest_dist=None, suggest_ids='', suggest_confident=True):
        popup = tk.Toplevel(self.win)
        popup.title("Rename star" if current_name else "Name star")
        popup.configure(bg=SURF)
        popup.transient(self.win)
        popup.resizable(False, False)
        rx = self.win.winfo_rootx() + disp_x + 14
        ry = self.win.winfo_rooty() + disp_y + 14
        popup.geometry(f'+{int(rx)}+{int(ry)}')

        tk.Label(popup, text="Star name", bg=SURF, fg=FG2,
                 font=(FONT, 8)).pack(padx=14, pady=(12, 3), anchor='w')

        if suggestion and not current_name:
            ids_str   = f"  {suggest_ids}" if suggest_ids else ""
            dist_str  = f"  ({suggest_dist}px off)" if suggest_dist is not None else ""
            hint_col  = '#a78bfa' if suggest_confident else '#f59e0b'
            conf_str  = '' if suggest_confident else '  ⚠ extrapolating'
            tk.Label(popup,
                     text=f'✦ WCS suggests: {suggestion}{ids_str}{dist_str}{conf_str}',
                     bg=SURF, fg=hint_col,
                     font=(FONT, 7)).pack(padx=14, pady=(0, 4), anchor='w')

        search_var = tk.StringVar(value=current_name or suggestion or '')
        ent = tk.Entry(popup, textvariable=search_var, width=24,
                       bg=BTN, fg=FG, insertbackground=FG,
                       relief='flat', font=(FONT, 10), bd=6)
        ent.pack(padx=14, pady=(0, 4))
        ent.focus_set()

        list_area = tk.Frame(popup, bg=SURF)
        list_area.pack(fill='both', expand=True, padx=14, pady=(0, 2))

        frm = tk.Frame(list_area, bg=BTN)
        frm.pack(side='left', fill='both', expand=True)
        sb = tk.Scrollbar(frm, bg=SURF)
        sb.pack(side='right', fill='y')
        lb = tk.Listbox(frm, yscrollcommand=sb.set, height=12, width=44,
                        bg=BTN, fg=FG, selectbackground='#1d4ed8',
                        selectforeground='#ffffff', activestyle='none',
                        font=('Consolas', 9), relief='flat', bd=0)
        lb.pack(side='left', fill='both', expand=True)
        sb.config(command=lb.yview)

        # A–Z jump bar
        alpha_col = tk.Frame(list_area, bg=SURF, width=16)
        alpha_col.pack(side='right', fill='y', padx=(2, 0))
        alpha_col.pack_propagate(False)
        for _letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            def _jump(l=_letter):
                for i in range(lb.size()):
                    if lb.get(i).upper().startswith(l):
                        lb.see(i)
                        lb.selection_clear(0, 'end')
                        lb.selection_set(i)
                        break
            tk.Button(alpha_col, text=_letter, command=_jump,
                      bg=SURF, fg=FG2, relief='flat', bd=0,
                      font=('Consolas', 6), padx=0, pady=0
                      ).pack(fill='x', expand=True)

        _lb_names = []  # parallel list of actual names for the listbox rows

        def _refresh(*_):
            q = search_var.get().strip().lower()
            lb.delete(0, 'end')
            _lb_names.clear()
            for n in _STAR_NAMES:
                ce = _CATALOG.get(n)
                hr, fl, mag = (ce[3], ce[4], ce[2]) if ce else (None, None, 0.0)
                hr_str = str(hr) if hr else ''
                fl_str = str(fl) if fl else ''
                if not q or q in n.lower() or q in hr_str or q in fl_str:
                    lb.insert('end', _fmt_lb(n, hr, fl, mag))
                    _lb_names.append(n)
            if lb.size() == 1:
                lb.selection_set(0)

        search_var.trace_add('write', _refresh)
        _refresh()

        def _confirm(name=None):
            if name is None:
                sel = lb.curselection()
                if sel:
                    name = _lb_names[sel[0]]
                elif lb.size() == 1:
                    name = _lb_names[0]
                else:
                    # allow typing a free-form name not in the list
                    name = search_var.get().strip() or None
                if not name:
                    return
            if name and self._pending_xy:
                if current_name and current_name != name:
                    self.identified.pop(current_name, None)
                self.identified[name] = self._pending_xy
                self._update_id_label()
                self._render()
            popup.destroy()

        lb.bind('<Double-Button-1>', lambda e: _confirm())
        ent.bind('<Return>',
                 lambda e: _confirm(_lb_names[lb.curselection()[0]])
                 if lb.curselection() else _confirm())
        popup.bind('<Escape>', lambda e: popup.destroy())

        btn_row = tk.Frame(popup, bg=SURF)
        btn_row.pack(pady=8)
        tk.Button(btn_row, text="Assign", command=_confirm,
                  bg='#14532d', fg='#ffffff', relief='flat',
                  font=(FONT, 9), padx=14, pady=5,
                  activebackground='#166534', cursor='hand2').pack(side='left', padx=4)
        if current_name:
            def _remove():
                self.identified.pop(current_name, None)
                self._update_id_label()
                self._render()
                popup.destroy()
            tk.Button(btn_row, text="Remove", command=_remove,
                      bg='#7f1d1d', fg='#ffffff', relief='flat',
                      font=(FONT, 9), padx=10, pady=5,
                      activebackground='#991b1b', cursor='hand2').pack(side='left', padx=4)

    def _update_id_label(self):
        if not self.identified:
            self.id_lbl.config(
                text='none — click an orange circle to name it')
        else:
            self.id_lbl.config(text='  ·  '.join(self.identified.keys()))

    def _clear_all(self):
        self.identified.clear()
        self._update_id_label()
        self._render()

    # ── Rough WCS (for click suggestions) ─────────────────────────────────────

    def _rough_wcs(self):
        """2-star similarity-transform WCS, or full fit for 3+.  Returns None on failure."""
        names = [n for n in self.identified if n in _CATALOG]
        if len(names) < 2:
            return None
        pix_list = [list(self.identified[n]) for n in names]
        sky_list  = [SkyCoord(ra=_CATALOG[n][0]*u.deg, dec=_CATALOG[n][1]*u.deg,
                               frame='icrs') for n in names]
        if len(names) >= 3:
            try:
                return self._fit_wcs(pix_list, sky_list)
            except Exception:
                pass
        # Two-star similarity transform (scale + rotation only, no shear)
        p1 = np.array(pix_list[0], dtype=float)
        p2 = np.array(pix_list[1], dtype=float)
        s1, s2 = sky_list[0], sky_list[1]
        w_unit = _AstropyWCS(naxis=2)
        w_unit.wcs.crpix = [1.0, 1.0]
        w_unit.wcs.crval = [s1.ra.deg, s1.dec.deg]
        w_unit.wcs.cd    = np.eye(2)
        w_unit.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        w_unit.wcs.set()
        iwc2 = w_unit.world_to_pixel(s2)
        dxi, deta = float(iwc2[0]), float(iwc2[1])
        dpx, dpy  = float(p2[0] - p1[0]), float(p2[1] - p1[1])
        det = dpx**2 + dpy**2
        if det < 1e-12:
            return None
        a = ( dpx*dxi + dpy*deta) / det
        b = (-dpy*dxi + dpx*deta) / det
        CD = np.array([[a, -b], [b, a]])
        w = _AstropyWCS(naxis=2)
        w.wcs.crpix = [p1[0] + 1.0, p1[1] + 1.0]
        w.wcs.crval = [s1.ra.deg, s1.dec.deg]
        w.wcs.cd    = CD
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        w.wcs.set()
        return w

    # ── Solving ────────────────────────────────────────────────────────────────

    def _solve(self):
        if not _ASTROPY:
            messagebox.showerror(
                "Missing dependency",
                "astropy is not installed.\nRun: pip install astropy",
                parent=self.win)
            return
        if len(self.identified) < 2:
            messagebox.showwarning(
                "Need more stars",
                "Identify at least 2 named stars before solving.",
                parent=self.win)
            return

        self.status_lbl.config(text="Computing rough WCS…")
        self.win.update_idletasks()

        seed_pix, seed_sky = [], []
        for name, (px, py) in self.identified.items():
            if name not in _CATALOG:
                continue
            ra, dec, *_ = _CATALOG[name]
            seed_pix.append([px, py])
            seed_sky.append(
                SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs'))

        if len(seed_pix) < 2:
            messagebox.showerror("Catalog error",
                                 "None of the named stars are in the catalog.",
                                 parent=self.win)
            return

        # Keep the ordered name list in sync with seed_pix / seed_sky.
        seed_names_list = [n for n in self.identified if n in _CATALOG]
        rejected_seeds = []

        if len(seed_pix) < 2:
            messagebox.showerror(
                "Not enough catalog stars",
                "Need at least 2 identified stars that are in the catalog.",
                parent=self.win)
            return

        try:
            rough_wcs = self._fit_wcs(seed_pix, seed_sky)
        except Exception as e:
            messagebox.showerror("WCS error", str(e), parent=self.win)
            return

        self.status_lbl.config(text="Auto-matching catalog…")
        self.win.update_idletasks()
        matched_pix, matched_sky, matched_names = \
            self._auto_match(rough_wcs, seed_pix, seed_sky, seed_names_list)

        if len(matched_pix) >= 3:
            self.status_lbl.config(
                text=f"Refining with {len(matched_pix)} stars…")
            self.win.update_idletasks()
            try:
                final_wcs = self._fit_wcs(matched_pix, matched_sky)
            except Exception:
                final_wcs = rough_wcs

            # Outlier rejection: drop auto-matched stars with > 15px residual
            # from the final WCS, then refit.  Always keep manually-named seeds.
            if len(matched_pix) > len(seed_pix):
                keep_pix, keep_sky, keep_names = [], [], []
                for pix, sky, name in zip(matched_pix, matched_sky, matched_names):
                    is_seed = name in self.identified
                    try:
                        proj = final_wcs.world_to_pixel(sky)
                        resid = math.sqrt(
                            (float(proj[0]) - pix[0]) ** 2 +
                            (float(proj[1]) - pix[1]) ** 2)
                    except Exception:
                        resid = 0.0
                    if resid <= 15.0 or is_seed:
                        keep_pix.append(pix)
                        keep_sky.append(sky)
                        keep_names.append(name)
                if len(keep_pix) >= 3 and len(keep_pix) < len(matched_pix):
                    try:
                        final_wcs = self._fit_wcs(keep_pix, keep_sky)
                        matched_pix, matched_sky, matched_names = \
                            keep_pix, keep_sky, keep_names
                    except Exception:
                        pass
        else:
            final_wcs = rough_wcs

        self._show_result(final_wcs, matched_pix, matched_names, rejected_seeds)

    def _score_wcs(self, w):
        """Count catalog stars whose projected position matches a detected star within 20 px."""
        if not self.all_stars:
            return 0
        score = 0
        tol_sq = 20 ** 2
        for _name, ra, dec, _ in BRIGHT_STARS:
            try:
                sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
                proj = w.world_to_pixel(sc)
                px, py = float(proj[0]), float(proj[1])
            except Exception:
                continue
            if not (0 <= px < self.img_w and 0 <= py < self.img_h):
                continue
            nearest = min(self.all_stars,
                          key=lambda s, _px=px, _py=py: (s[0]-_px)**2 + (s[1]-_py)**2)
            if (nearest[0]-px)**2 + (nearest[1]-py)**2 <= tol_sq:
                score += 1
        return score

    def _ransac_wcs(self, pix_list, sky_list, inlier_thresh=120.0):
        """Try all C(N,3) triplets; return (best_wcs, inlier_indices).

        Scores each triplet's WCS by (manual inliers, auto-match count) so
        wrong identifications — whose catalog coords don't map to real stars —
        lose to the correct subset which auto-matches many detected stars.
        """
        n = len(pix_list)
        best_manual, best_auto = -1, -1
        best_inliers, best_wcs = list(range(n)), None

        for triplet in _combos(range(n), 3):
            try:
                w = self._fit_wcs([pix_list[i] for i in triplet],
                                  [sky_list[i]  for i in triplet])
            except Exception:
                continue

            inliers = []
            for i, (pix, sky) in enumerate(zip(pix_list, sky_list)):
                try:
                    proj = w.world_to_pixel(sky)
                    r = math.sqrt((float(proj[0])-pix[0])**2 +
                                  (float(proj[1])-pix[1])**2)
                    if r < inlier_thresh:
                        inliers.append(i)
                except Exception:
                    pass

            auto = self._score_wcs(w)
            if (len(inliers), auto) > (best_manual, best_auto):
                best_manual = len(inliers)
                best_auto   = auto
                best_inliers = inliers
                best_wcs     = w

        return best_wcs, best_inliers

    def _fit_wcs(self, pix_list, sky_list):
        """Fit TAN WCS from pixel/sky pairs.

        Uses a unit WCS (CD=identity) to ask astropy for the true TAN
        intermediate world coordinates, then fits a CD matrix with lstsq.
        This guarantees the round-trip is exact regardless of sign conventions
        in the gnomonic formulae.
        """
        pix = np.array(pix_list, dtype=float)
        ra_degs  = np.array([s.ra.deg  for s in sky_list])
        dec_degs = np.array([s.dec.deg for s in sky_list])

        # Circular mean for RA reference (handles 0/360 wrap)
        ra0_deg = math.degrees(math.atan2(
            float(np.mean(np.sin(np.radians(ra_degs)))),
            float(np.mean(np.cos(np.radians(ra_degs))))))
        if ra0_deg < 0:
            ra0_deg += 360.0
        dec0_deg = float(np.mean(dec_degs))

        # Unit WCS: CRPIX=(1,1), CD=identity — world_to_pixel returns IWC in °
        w_unit = _AstropyWCS(naxis=2)
        w_unit.wcs.crpix = [1.0, 1.0]
        w_unit.wcs.crval = [ra0_deg, dec0_deg]
        w_unit.wcs.cd    = np.eye(2)
        w_unit.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        w_unit.wcs.set()

        iwc = np.array([list(w_unit.world_to_pixel(s)) for s in sky_list],
                       dtype=float)
        xi  = iwc[:, 0]
        eta = iwc[:, 1]

        # Fit affine:  [px_x, px_y, 1] @ c = IWC  (0-indexed pixels)
        A = np.column_stack([pix[:, 0], pix[:, 1], np.ones(len(pix))])
        cx, _, _, _ = np.linalg.lstsq(A, xi,  rcond=None)
        cy, _, _, _ = np.linalg.lstsq(A, eta, rcond=None)

        CD = np.array([[cx[0], cx[1]], [cy[0], cy[1]]])
        try:
            crpix0 = np.linalg.solve(CD, np.array([-cx[2], -cy[2]]))
        except np.linalg.LinAlgError:
            crpix0 = pix.mean(axis=0)

        w = _AstropyWCS(naxis=2)
        w.wcs.crpix = [crpix0[0] + 1.0, crpix0[1] + 1.0]
        w.wcs.crval = [ra0_deg, dec0_deg]
        w.wcs.cd    = CD
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        w.wcs.set()
        return w

    def _auto_match(self, rough_wcs, seed_pix, seed_sky, seed_names):
        matched_pix   = list(seed_pix)
        matched_sky   = list(seed_sky)
        matched_names = list(seed_names)   # stays aligned with seed_pix
        used = {(int(p[0]), int(p[1])) for p in seed_pix}
        catalog_in_fov = []
        tol_sq = 20 ** 2

        for name, ra, dec, _ in BRIGHT_STARS:
            try:
                sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
                px_arr = rough_wcs.world_to_pixel(sc)
                proj_x = float(px_arr[0])
                proj_y = float(px_arr[1])
            except Exception:
                continue
            if not (0 <= proj_x < self.img_w and 0 <= proj_y < self.img_h):
                continue
            catalog_in_fov.append((name, ra, dec))
            if name in self.identified or not self.all_stars:
                continue
            nearest = min(self.all_stars,
                          key=lambda s: (s[0]-proj_x)**2 + (s[1]-proj_y)**2)
            if (nearest[0]-proj_x)**2 + (nearest[1]-proj_y)**2 > tol_sq:
                continue
            key = (int(nearest[0]), int(nearest[1]))
            if key in used:
                continue
            matched_pix.append([nearest[0], nearest[1]])
            matched_sky.append(sc)
            matched_names.append(name)
            used.add(key)

        self._render(overlay_wcs=rough_wcs, overlay_catalog=catalog_in_fov)
        return matched_pix, matched_sky, matched_names

    # ── Result ─────────────────────────────────────────────────────────────────

    def _show_result(self, wcs, matched_pix, matched_names, rejected_seeds=None):
        try:
            scales = wcs.proj_plane_pixel_scales()
            px_arcsec = float(
                (scales[0].to(u.arcsec).value +
                 scales[1].to(u.arcsec).value) / 2)
        except Exception:
            try:
                pm = wcs.pixel_scale_matrix
                px_arcsec = float(
                    math.sqrt(abs(np.linalg.det(pm)))) * 3600
            except Exception:
                px_arcsec = 0.0

        fov_w = self.img_w * px_arcsec / 3600
        fov_h = self.img_h * px_arcsec / 3600
        try:
            center = wcs.pixel_to_world(self.img_w / 2, self.img_h / 2)
            cra  = center.ra.deg
            cdec = center.dec.deg
        except Exception:
            cra = cdec = 0.0

        # RMS residual + per-star residuals for diagnostics.
        resid_sq = []
        star_lines = []
        for name, pix_xy in zip(matched_names, matched_pix):
            dpx, dpy = pix_xy[0], pix_xy[1]
            if name not in _CATALOG:
                continue
            ra, dec, *_ = _CATALOG[name]
            try:
                sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
                proj = wcs.world_to_pixel(sc)
                dx = float(proj[0]) - dpx
                dy = float(proj[1]) - dpy
                r = math.sqrt(dx * dx + dy * dy)
                resid_sq.append(r * r)
                flag = "manual" if name in self.identified else "auto  "
                star_lines.append(f"  {name:<20s} [{flag}] {r:6.1f} px")
            except Exception:
                pass
        rms = math.sqrt(sum(resid_sq) / len(resid_sq)) if resid_sq else 0.0
        rms_arcsec = rms * px_arcsec

        n = len(matched_pix)
        self.status_lbl.config(
            text=f"✓ Solved  {px_arcsec:.2f}\"/px  "
                 f"FOV {fov_w:.1f}°×{fov_h:.1f}°  "
                 f"RA {cra:.3f}°  Dec {cdec:.3f}°  "
                 f"({n} stars  RMS {rms:.1f}px / {rms_arcsec:.0f}\")")

        detail = "\n".join(star_lines) if star_lines else "  (none)"
        reject_str = ""
        if rejected_seeds:
            reject_str = ("\n\n⚠ Rejected as outliers (wrong identification):\n" +
                          "\n".join(f"  ✗ {n}" for n in rejected_seeds))
        body = (
            f"Pixel scale :  {px_arcsec:.2f} arcsec/px\n"
            f"FOV         :  {fov_w:.2f}° × {fov_h:.2f}°\n"
            f"Center RA   :  {cra:.4f}°\n"
            f"Center Dec  :  {cdec:.4f}°\n"
            f"Stars matched: {n}\n"
            f"RMS residual:  {rms:.2f} px  ({rms_arcsec:.1f}\")\n\n"
            f"Per-star residuals:\n{detail}"
            f"{reject_str}\n\n"
            f"{'✓ Good fit (< 2px)' if rms < 2 else '⚠ Check identifications (> 2px)'}"
        )
        dlg = tk.Toplevel(self.win)
        dlg.title("Plate Solve Result")
        dlg.configure(bg=SURF)
        dlg.transient(self.win)
        dlg.resizable(True, True)
        txt = tk.Text(dlg, bg=BTN, fg=FG, font=('Consolas', 9),
                      relief='flat', bd=8, wrap='none',
                      width=60, height=min(30, body.count('\n') + 4))
        sb_y = tk.Scrollbar(dlg, command=txt.yview)
        txt.configure(yscrollcommand=sb_y.set)
        txt.pack(side='left', fill='both', expand=True, padx=(10, 0), pady=10)
        sb_y.pack(side='right', fill='y', pady=10, padx=(0, 4))
        txt.insert('1.0', body)
        txt.configure(state='disabled')
        tk.Button(dlg, text="OK", command=dlg.destroy,
                  bg='#14532d', fg='#ffffff', relief='flat',
                  font=(FONT, 9), padx=20, pady=4,
                  cursor='hand2').pack(pady=(0, 10))
        dlg.bind('<Return>', lambda e: dlg.destroy())
        dlg.bind('<Escape>', lambda e: dlg.destroy())

        identified_stars = [
            {'name': nm, 'px': float(self.identified[nm][0]),
             'py': float(self.identified[nm][1]),
             'ra': float(_CATALOG[nm][0]), 'dec': float(_CATALOG[nm][1])}
            for nm in self.identified if nm in _CATALOG
        ]
        if self.on_result:
            self.on_result({
                'pixel_scale_arcsec': px_arcsec,
                'fov_width_deg':  fov_w,
                'fov_height_deg': fov_h,
                'center_ra':  cra,
                'center_dec': cdec,
                'n_matched':  n,
                'identified_stars': identified_stars,
            })


def _save_result_to_astro_config(result):
    """Write center RA/Dec (and optionally contrast image path) into streaker_astro_config.json."""
    import json as _json
    import sys as _sys
    # Locate config next to the exe (frozen) or script
    if getattr(_sys, 'frozen', False):
        cfg_path = Path(_sys.executable).parent / "streaker_astro_config.json"
    else:
        cfg_path = Path(__file__).parent / "streaker_astro_config.json"
    try:
        cfg = _json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    except Exception:
        cfg = {}
    if result.get('_image_only'):
        cfg['last_contrast_image'] = result['_image_path']
    else:
        cfg['center_ra']           = round(result['center_ra'],  4)
        cfg['center_dec']          = round(result['center_dec'], 4)
        cfg['center_radius']       = '15'
        if result.get('identified_stars'):
            cfg['identified_stars'] = result['identified_stars']
    try:
        cfg_path.write_text(_json.dumps(cfg, indent=2))
    except Exception:
        pass


def launch_local_solve(parent, image_path=None):
    if not image_path:
        image_path = filedialog.askopenfilename(
            title="Select image to plate solve",
            filetypes=[("PNG/JPG", "*.png *.jpg"), ("All", "*.*")],
            parent=parent)
    if image_path:
        LocalSolveWindow(parent, image_path, on_result=_save_result_to_astro_config)


if __name__ == '__main__':
    import sys
    root = tk.Tk()
    root.withdraw()
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    launch_local_solve(root, image_path)
    root.mainloop()

