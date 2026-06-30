#!/usr/bin/env python3
"""
StreakerDemo.py — two-tab interactive visualizer for StreakerDetect filters.

Tab 1  Per-Frame Filters : MOG2 threshold, area, aspect, cloud_ratio.
Tab 2  TrackManager      : how per-frame detections get linked into tracks
                           and gated by min_move / min_travel.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
from PIL import Image, ImageTk

# ── Shared constants ──────────────────────────────────────────────────────────
SW, SH   = 780, 440          # per-panel canvas size (scene / mask)
TW       = SW * 2 + 6        # total canvas width (both panels side by side)
FONT     = cv2.FONT_HERSHEY_SIMPLEX
BG       = "#0c0c1e"
CARD     = "#14142a"
TBL_BG   = "#080814"
ACCENT   = "#66aaee"
VAL_C    = "#88ddff"
FG       = "#d0d0e8"
T_PASS   = "#33ee66"
T_FAIL   = "#ff4444"
T_SKIP   = "#444460"
T_DIM    = "#555570"
T_WARN   = "#ffdd44"
T_HDR    = "#8899cc"
T_NAME_ON  = "#c8d4ff"
T_NAME_OFF = "#484860"


def _txt(img, text, pos, scale, color, thickness=1):
    """Text with solid black background box — always readable."""
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = pos
    p = 3
    cv2.rectangle(img, (x - p, y - th - p), (x + tw + p, y + bl + p), (0, 0, 0), -1)
    cv2.putText(img, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PER-FRAME FILTERS
# ══════════════════════════════════════════════════════════════════════════════

PF_OBJECTS = [
    dict(id="meteor",    contrast=230, type="streak",
         p1=(55, 410),  p2=(250, 145), width=4,
         color=(220, 240, 255), name="Meteor",       hint="bright · elongated"),
    dict(id="satellite", contrast=115, type="streak",
         p1=(490, 55),  p2=(700, 100), width=2,
         color=(140, 190, 220), name="Satellite",    hint="dim · elongated"),
    dict(id="cloud_big", contrast=190, type="blob",
         center=(590, 265), axes=(108, 84), angle=15,
         color=(130, 120, 105), name="Cloud",        hint="large · round"),
    dict(id="cloud_sm",  contrast=150, type="blob",
         center=(365, 370), axes=(30, 24), angle=0,
         color=(130, 120, 105), name="Small cloud",  hint="medium · round"),
    dict(id="bird",      contrast=138, type="blob",
         center=(188, 200), axes=(32, 11), angle=40,
         color=(80, 190, 240), name="Bird / plane",  hint="elongated blob"),
    dict(id="noise",     contrast=78,  type="point",
         center=(435, 190), radius=5,
         color=(160, 220, 160), name="Noise",        hint="tiny · round"),
]

PF_DEFAULTS = dict(threshold=40, min_area=120, max_area=1400,
                   min_aspect=2.0, cloud_ratio=0.15)

PASS_C = (60, 235,  80)
FAIL_C = (40,  80, 255)
GHOST  = (50,  50,  70)


def _centroid(obj):
    if obj["type"] == "streak":
        return ((obj["p1"][0] + obj["p2"][0]) // 2,
                (obj["p1"][1] + obj["p2"][1]) // 2)
    return obj["center"]


def _draw_obj(canvas, obj, col, alpha=1.0):
    t = obj["type"]
    target = canvas.copy() if alpha < 1.0 else canvas
    if t == "streak":
        cv2.line(target, obj["p1"], obj["p2"], col, obj.get("width", 2))
    elif t == "blob":
        cv2.ellipse(target, obj["center"], obj["axes"],
                    obj.get("angle", 0), 0, 360, col, -1)
    elif t == "point":
        cv2.circle(target, obj["center"], obj["radius"], col, -1)
    if alpha < 1.0:
        cv2.addWeighted(target, alpha, canvas, 1.0 - alpha, 0, canvas)


def _stamp(mask, obj):
    t = obj["type"]
    if t == "streak":
        cv2.line(mask, obj["p1"], obj["p2"], 255, obj.get("width", 2))
    elif t == "blob":
        cv2.ellipse(mask, obj["center"], obj["axes"],
                    obj.get("angle", 0), 0, 360, 255, -1)
    elif t == "point":
        cv2.circle(mask, obj["center"], obj["radius"], 255, -1)


def _eval_pf(obj, cnt, min_a, max_a, min_s):
    d = dict(thr_pass=True, area=None, area_pass=None,
             maxarea_pass=None, asp=None, asp_pass=None,
             passed=False, reason="")
    if cnt is None:
        d["thr_pass"] = False
        return d
    area = cv2.contourArea(cnt)
    d["area"] = area
    if area < min_a:
        d["area_pass"] = False
        d["reason"] = f"area {area:.0f} < {min_a:.0f}"
        return d
    d["area_pass"] = True
    if area > max_a:
        d["maxarea_pass"] = False
        d["reason"] = f"area {area:.0f} > {max_a:.0f}"
        return d
    d["maxarea_pass"] = True
    rw, rh = cv2.minAreaRect(cnt)[1]
    if min(rw, rh) < 1:
        d["asp_pass"] = False
        d["reason"] = "degenerate"
        return d
    asp = max(rw, rh) / min(rw, rh)
    d["asp"] = asp
    if asp < min_s:
        d["asp_pass"] = False
        d["reason"] = f"aspect {asp:.1f} < {min_s:.1f}"
        return d
    d["asp_pass"] = True
    d["passed"] = True
    d["reason"] = f"area {area:.0f}  asp {asp:.1f}"
    return d


def render_pf(params):
    thr   = params["threshold"]
    min_a = params["min_area"]
    max_a = params["max_area"]
    min_s = params["min_aspect"]
    c_rat = params["cloud_ratio"]
    mask     = np.zeros((SH, SW), dtype=np.uint8)
    per_obj  = []
    for obj in PF_OBJECTS:
        if obj["contrast"] < thr:
            per_obj.append((obj, None, _eval_pf(obj, None, min_a, max_a, min_s)))
            continue
        omask = np.zeros((SH, SW), dtype=np.uint8)
        _stamp(omask, obj)
        mask = cv2.bitwise_or(mask, omask)
        ocnts, _ = cv2.findContours(omask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(ocnts, key=cv2.contourArea) if ocnts else None
        per_obj.append((obj, cnt, _eval_pf(obj, cnt, min_a, max_a, min_s)))
    active_ratio = np.count_nonzero(mask) / (SW * SH)
    suppressed = active_ratio > c_rat
    scene = np.full((SH, SW, 3), (18, 18, 32), dtype=np.uint8)
    for obj, cnt, d in per_obj:
        col   = obj["color"] if d["thr_pass"] else GHOST
        alpha = 1.0          if d["thr_pass"] else 0.45
        _draw_obj(scene, obj, col, alpha)
        if cnt is None:
            cx, cy = _centroid(obj)
            _txt(scene, obj["name"], (cx - 30, cy - 6), 0.44, (70, 70, 95))
            continue
        outline = PASS_C if d["passed"] else FAIL_C
        cv2.drawContours(scene, [cnt], -1, outline, 2)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        _txt(scene, obj["name"], (bx, max(by - 8, 16)), 0.50, (220, 230, 255))
        tag = "PASS → TrackManager" if d["passed"] else d["reason"]
        _txt(scene, tag, (bx, min(by + bh + 20, SH - 6)), 0.46, outline)
    if suppressed:
        ov = scene.copy()
        cv2.rectangle(ov, (0, 0), (SW, 50), (8, 0, 130), -1)
        cv2.addWeighted(ov, 0.75, scene, 0.25, 0, scene)
        _txt(scene,
             f"FRAME SUPPRESSED — {active_ratio:.1%} active > cloud_ratio {c_rat:.0%}",
             (10, 34), 0.50, (255, 215, 60))
    else:
        _txt(scene, f"cloud: {active_ratio:.1%} active  (limit {c_rat:.0%})",
             (10, 24), 0.42, (110, 135, 165))
    _txt(scene, "STAGE 1 — Per-Frame Filters", (10, SH - 10), 0.40, (80, 100, 140))
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    _txt(mask_bgr, "Foreground Mask", (10, SH - 10), 0.40, (140, 140, 140))
    info = dict(per_obj=per_obj, active_ratio=active_ratio, suppressed=suppressed)
    return scene, mask_bgr, info


PF_SLIDER_DEFS = [
    ("threshold",   "MOG2 Threshold",   5,   250, 1,
     "How different a pixel must be\nfrom background to flag as\nforeground.\n\n"
     "↑ High → bright objects only\n↓ Low  → faint objects visible"),
    ("min_area",    "Min Area (px²)",   5,   600, 5,
     "Reject contours smaller than\nthis (noise, stars).\n\n"
     "↑ High → reject small blobs\n↓ Low  → allow tiny detections"),
    ("max_area",    "Max Area (px²)",   200, 5000, 50,
     "Reject contours larger than\nthis (cloud patches).\n\n"
     "↑ High → allow large objects\n↓ Low  → reject big blobs"),
    ("min_aspect",  "Min Aspect Ratio", 1.0,  8.0, 0.1,
     "Minimum length÷width ratio.\n\nMeteors: 4–20+\nSatellites: 3–8\n"
     "Clouds: 1.0–2.0\n\n"
     "↑ High → streaks only\n↓ Low  → round blobs pass"),
    ("cloud_ratio", "Cloud Ratio",      0.01, 0.50, 0.01,
     "Active-pixel fraction that\ntriggers frame suppression.\n\n"
     "↑ High → only suppress heavy cloud\n↓ Low  → suppress early"),
]


class PerFrameTab:
    DIVW = 6

    def __init__(self, parent):
        self.parent = parent
        self._pending = None

        # Stage label
        tk.Label(parent,
                 text="STAGE 1 — PER-FRAME FILTERS   "
                      "(applied every frame before TrackManager sees anything)",
                 fg="#66aaee", bg=BG,
                 font=("Consolas", 10, "bold")).pack(anchor="w", padx=14, pady=(8, 2))

        # Canvas
        self.canvas = tk.Canvas(parent, width=TW, height=SH,
                                bg="#000", highlightthickness=0)
        self.canvas.pack(padx=10, pady=(2, 4))
        self.img_id = self.canvas.create_image(0, 0, anchor="nw")

        # Legend
        leg = tk.Frame(parent, bg=BG)
        leg.pack(fill="x", padx=14, pady=(0, 4))
        for col, txt in [
            (T_PASS, "■  PASS → goes to TrackManager"),
            (T_FAIL, "■  FAIL → discarded, never reaches TrackManager"),
            (T_SKIP, "■  Below threshold — invisible to detector"),
        ]:
            tk.Label(leg, text=txt, fg=col, bg=BG,
                     font=("Consolas", 10, "bold")).pack(side="left", padx=12)

        # Sliders
        sf = tk.Frame(parent, bg=BG)
        sf.pack(fill="x", padx=10, pady=(0, 6))
        self.vars = {}
        for col, (key, label, lo, hi, res, tip) in enumerate(PF_SLIDER_DEFS):
            self.vars[key] = tk.DoubleVar(value=PF_DEFAULTS[key])
            sf.columnconfigure(col, weight=1)
            fr = tk.Frame(sf, bg=CARD, padx=8, pady=6)
            fr.grid(row=0, column=col, padx=4, sticky="nsew")
            tk.Label(fr, text=label, fg=ACCENT, bg=CARD,
                     font=("Consolas", 10, "bold"), justify="center").pack()
            val_lbl = tk.Label(fr, text=f"{PF_DEFAULTS[key]:.3g}",
                               fg=VAL_C, bg=CARD,
                               font=("Consolas", 16, "bold"))
            val_lbl.pack(pady=(2, 1))
            tk.Label(fr, text=tip, fg="#7080a0", bg=CARD,
                     font=("Consolas", 7), justify="left").pack(pady=(2, 4))

            def _cmd(v, vl=val_lbl):
                vl.config(text=f"{float(v):.3g}")
                if self._pending:
                    self.parent.after_cancel(self._pending)
                self._pending = self.parent.after(100, self._refresh)

            tk.Scale(fr, variable=self.vars[key], from_=lo, to=hi,
                     resolution=res, orient="horizontal", showvalue=False,
                     bg=CARD, fg=FG, troughcolor="#2a3a5a",
                     activebackground=ACCENT, highlightthickness=0,
                     command=_cmd).pack(fill="x")

        # Pipeline table
        tbl = tk.Frame(parent, bg=TBL_BG, bd=1, relief="flat")
        tbl.pack(fill="x", padx=10, pady=(0, 10))
        self.cloud_banner = tk.Label(tbl, text="", anchor="w",
                                     bg=TBL_BG, fg="#88aacc",
                                     font=("Consolas", 10, "bold"), padx=10, pady=4)
        self.cloud_banner.grid(row=0, column=0, columnspan=6, sticky="ew")
        cols   = ["Object", "Threshold", "Min Area", "Max Area", "Aspect", "Result"]
        widths = [16, 12, 14, 14, 14, 34]
        for c, (h, w) in enumerate(zip(cols, widths)):
            tk.Label(tbl, text=h, bg="#181830", fg=T_HDR,
                     font=("Consolas", 9, "bold"), width=w,
                     anchor="center", pady=4).grid(
                row=1, column=c, padx=2, pady=(0, 2), sticky="ew")
            tbl.columnconfigure(c, weight=1 if c == 5 else 0)
        self.cells = []
        for r in range(len(PF_OBJECTS)):
            row = []
            for c, w in enumerate(widths):
                lbl = tk.Label(tbl, text="", bg="#0e0e22", fg=T_DIM,
                               font=("Consolas", 10), width=w,
                               anchor="center", pady=3)
                lbl.grid(row=r + 2, column=c, padx=2, pady=1, sticky="ew")
                row.append(lbl)
            self.cells.append(row)

        self._refresh()

    def _refresh(self):
        p = {k: v.get() for k, v in self.vars.items()}
        scene, mask_bgr, info = render_pf(p)
        div = np.full((SH, self.DIVW, 3), 45, dtype=np.uint8)
        combined = np.hstack([scene, div, mask_bgr])
        img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self.img_id, image=self.tk_img)

        ar   = info["active_ratio"]
        supp = info["suppressed"]
        cr   = p["cloud_ratio"]
        if supp:
            self.cloud_banner.config(
                text=f"  ⚠  FRAME SUPPRESSED — {ar:.1%} active > cloud_ratio {cr:.0%}"
                     "  —  all events blocked",
                fg=T_WARN, bg="#200010")
        else:
            self.cloud_banner.config(
                text=f"  Cloud: {ar:.1%} active  (limit {cr:.0%})  —  frame not suppressed",
                fg="#88aacc", bg=TBL_BG)

        for r, (obj, cnt, d) in enumerate(info["per_obj"]):
            row = self.cells[r]
            row[0].config(text=obj["name"],
                          fg=T_NAME_ON if d["thr_pass"] else T_NAME_OFF)
            row[1].config(text="✓" if d["thr_pass"] else "✗  below",
                          fg=T_PASS if d["thr_pass"] else T_DIM)

            def _cell(lbl, val, passed):
                if val is None:   lbl.config(text="—",    fg=T_SKIP)
                elif passed:      lbl.config(text="✓",    fg=T_PASS)
                else:             lbl.config(text="FAIL", fg=T_FAIL)

            _cell(row[2], d["area"], d["area_pass"])
            _cell(row[3], d["area"], d["maxarea_pass"])
            _cell(row[4], d["asp"],  d["asp_pass"])

            if not d["thr_pass"]:
                row[5].config(text="(below threshold)", fg=T_DIM)
            elif supp:
                row[5].config(text="⚠  frame suppressed", fg=T_WARN)
            elif d["passed"]:
                row[5].config(text="✓  PASS → TrackManager", fg=T_PASS)
            else:
                row[5].config(text=f"✗  {d['reason']}", fg=T_FAIL)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRACK MANAGER
# ══════════════════════════════════════════════════════════════════════════════

N_FRAMES = 40

# Synthetic objects: positions over 40 frames. None = not detected.
TM_OBJECTS = [
    dict(name="Meteor",       color_cv=(210, 230, 255), color_tk="#50c8ff",
         positions={f: (int(100 + (f-5)*30), int(390 - (f-5)*25))
                    for f in range(5, 15)}),
    dict(name="Satellite",    color_cv=(90, 210, 90),   color_tk="#66dd66",
         positions={f: (int(35 + f*17), 240) for f in range(40)}),
    dict(name="Star flicker", color_cv=(60, 150, 255),  color_tk="#ffaa44",
         positions={2: (540, 155), 3: (540, 156),
                    9: (541, 155), 10: (540, 155),
                    19: (540, 154), 20: (540, 155),
                    29: (541, 156), 30: (540, 155)}),
    dict(name="Cloud edge",   color_cv=(210, 90, 200),  color_tk="#cc88ff",
         positions={f: (int(570 + (f-12)*3), int(335 + (f-12)*2))
                    for f in range(12, 29)}),
]

TM_DEFAULTS = dict(min_move=0, min_travel=30, ghost_frames=3)

TM_SLIDER_DEFS = [
    ("min_move",     "Min Move (px/frame)",  0,  40, 1,
     "Minimum distance an object must\nmove frame-to-frame to extend\nan active track.\n\n"
     "Stationary scintillation = 0–2px\nSatellites = 17px/frame\nMeteors = 40px/frame\n\n"
     "↑ High → only fast movers tracked\n↓ 0   → stationary blobs build tracks"),
    ("min_travel",   "Min Travel (px total)", 0, 300, 5,
     "Minimum end-to-end displacement\nfor a closed track to be saved\nas an event clip.\n\n"
     "Star flicker = ~0px\nCloud edge ≈ 50px\nSatellite = 600+ px\nMeteor = 400+ px\n\n"
     "↑ High → only large movers saved\n↓ Low  → short drifts saved too"),
    ("ghost_frames", "Ghost Tolerance (frames)", 0, 10, 1,
     "How many consecutive frames a\ntrack can go unmatched before\nbeing closed.\n\n"
     "Star flicker gaps = 5–9 frames\n\n"
     "↑ High → bridges long gaps\n↓ Low  → tight, breaks on gaps"),
]


def _dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5


def run_tm_sim(min_move, min_travel, ghost_tol):
    """Simulate TrackManager over N_FRAMES. Returns list of track dicts."""
    tracks = {}
    nxt = [0]

    for frame in range(N_FRAMES):
        dets = [(oi, *obj["positions"][frame])
                for oi, obj in enumerate(TM_OBJECTS)
                if frame in obj["positions"]]

        matched_t = set()
        matched_d = set()

        for tid, tr in list(tracks.items()):
            if not tr["active"]:
                continue
            lx, ly = tr["last_pos"]
            best, best_di = 1e9, None
            for di, (oi, cx, cy) in enumerate(dets):
                if di in matched_d:
                    continue
                d = _dist((cx, cy), (lx, ly))
                if d < best:
                    best, best_di = d, di

            if best_di is not None and best < 120:
                oi, cx, cy = dets[best_di]
                move = _dist((cx, cy), (lx, ly))
                if move >= min_move:
                    tr["positions"].append((frame, cx, cy))
                    tr["last_pos"] = (cx, cy)
                    tr["ghost_count"] = 0
                    matched_t.add(tid)
                    matched_d.add(best_di)
                    continue
            tr["ghost_count"] += 1
            tr["ghost_frames"].append(frame)
            if tr["ghost_count"] > ghost_tol:
                tr["active"] = False

        for di, (oi, cx, cy) in enumerate(dets):
            if di in matched_d:
                continue
            tid = nxt[0]; nxt[0] += 1
            tracks[tid] = dict(
                id=tid, obj_idx=oi, origin=(cx, cy),
                positions=[(frame, cx, cy)], last_pos=(cx, cy),
                ghost_count=0, ghost_frames=[], active=True)

    for tr in tracks.values():
        if tr["active"]:
            tr["active"] = False
        ox, oy = tr["origin"]
        _, lx, ly = tr["positions"][-1]
        tr["travel"] = _dist((lx, ly), (ox, oy))
        tr["passed"] = tr["travel"] >= min_travel and len(tr["positions"]) >= 2

    return list(tracks.values())


def render_timeline(tracks, params):
    """Render the timeline canvas (TW × 300)."""
    TH      = 300
    LMARGIN = 140
    RMARGIN = 160
    HMARGIN = 30
    INNER_W = TW - LMARGIN - RMARGIN
    CELL_W  = INNER_W / N_FRAMES
    N_OBJ   = len(TM_OBJECTS)
    ROW_H   = (TH - HMARGIN * 2) / N_OBJ

    img = np.full((TH, TW, 3), (10, 10, 22), dtype=np.uint8)

    # Grid lines
    for f in range(N_FRAMES + 1):
        x = int(LMARGIN + f * CELL_W)
        cv2.line(img, (x, HMARGIN - 4), (x, TH - HMARGIN + 4), (28, 28, 48), 1)
    for r in range(N_OBJ + 1):
        y = int(HMARGIN + r * ROW_H)
        cv2.line(img, (LMARGIN - 5, y), (TW - RMARGIN + 5, y), (28, 28, 48), 1)

    # Frame number labels
    for f in range(0, N_FRAMES + 1, 5):
        x = int(LMARGIN + f * CELL_W)
        cv2.putText(img, str(f), (x - 6, 18), FONT, 0.32, (80, 90, 120), 1)

    # Per-object rows
    for oi, obj in enumerate(TM_OBJECTS):
        row_cy = int(HMARGIN + (oi + 0.5) * ROW_H)
        col    = obj["color_cv"]

        # Object label
        cv2.putText(img, obj["name"], (6, row_cy + 5), FONT, 0.44, col, 1, cv2.LINE_AA)

        # Detected-frame dots (dim)
        for frame, (cx, cy) in [(f, p) for f, p in obj["positions"].items()]:
            px = int(LMARGIN + (frame + 0.5) * CELL_W)
            cv2.circle(img, (px, row_cy), 4, (col[0]//3, col[1]//3, col[2]//3), -1)

        # Track segments for this object
        obj_tracks = [t for t in tracks if t["obj_idx"] == oi]
        for tr in obj_tracks:
            if len(tr["positions"]) < 1:
                continue
            pts = [(int(LMARGIN + (f + 0.5) * CELL_W), row_cy)
                   for f, cx, cy in tr["positions"]]
            line_col = col if tr["passed"] else (60, 60, 80)
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i+1], line_col, 2)
            for px, py in pts:
                cv2.circle(img, (px, py), 5, line_col, -1)

            # Ghost frame markers
            for gf in tr["ghost_frames"]:
                px = int(LMARGIN + (gf + 0.5) * CELL_W)
                cv2.drawMarker(img, (px, row_cy), (70, 70, 100),
                               cv2.MARKER_CROSS, 8, 1)

            # Result badge
            last_f = tr["positions"][-1][0]
            bx = int(LMARGIN + (last_f + 1.5) * CELL_W)
            bx = max(bx, TW - RMARGIN + 8)
            if tr["passed"]:
                cv2.putText(img, f"EVENT  ({tr['travel']:.0f}px)",
                            (TW - RMARGIN + 8, row_cy + 5),
                            FONT, 0.40, (50, 220, 80), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, f"discard  ({tr['travel']:.0f}px)",
                            (TW - RMARGIN + 8, row_cy + 5),
                            FONT, 0.36, (70, 70, 100), 1, cv2.LINE_AA)

    # Axis labels
    cv2.putText(img, "Frame  →",
                (LMARGIN, TH - 6), FONT, 0.36, (70, 80, 110), 1)
    cv2.putText(img, "STAGE 2 — TrackManager",
                (TW - 240, TH - 6), FONT, 0.36, (80, 100, 140), 1)

    return img


class TrackManagerTab:
    def __init__(self, parent):
        self.parent = parent
        self._pending = None

        # Stage label
        tk.Label(parent,
                 text="STAGE 2 — TRACK MANAGER   "
                      "(receives only PASS detections from Stage 1)",
                 fg="#88ee88", bg=BG,
                 font=("Consolas", 10, "bold")).pack(anchor="w", padx=14, pady=(8, 2))

        # Explanation
        tk.Label(parent,
                 text="  Each dot = one PASS detection.  "
                      "Connected dots = TrackManager linked them across frames.  "
                      "Result = saved event or discarded.",
                 fg="#6070a0", bg=BG,
                 font=("Consolas", 9), justify="left").pack(anchor="w", padx=14, pady=(0, 4))

        # Timeline canvas
        self.canvas = tk.Canvas(parent, width=TW, height=300,
                                bg="#000", highlightthickness=0)
        self.canvas.pack(padx=10, pady=(0, 4))
        self.img_id = self.canvas.create_image(0, 0, anchor="nw")

        # Legend
        leg = tk.Frame(parent, bg=BG)
        leg.pack(fill="x", padx=14, pady=(0, 6))
        for col, txt in [
            ("#33ee66", "■  EVENT saved  (travel ≥ min_travel)"),
            ("#444460", "■  discarded  (travel < min_travel)"),
            ("#606090", "+  ghost frame  (no detection, track kept alive)"),
        ]:
            tk.Label(leg, text=txt, fg=col, bg=BG,
                     font=("Consolas", 10, "bold")).pack(side="left", padx=12)

        # Sliders
        sf = tk.Frame(parent, bg=BG)
        sf.pack(fill="x", padx=10, pady=(0, 8))
        self.vars = {}
        for col, (key, label, lo, hi, res, tip) in enumerate(TM_SLIDER_DEFS):
            self.vars[key] = tk.DoubleVar(value=TM_DEFAULTS[key])
            sf.columnconfigure(col, weight=1)
            fr = tk.Frame(sf, bg=CARD, padx=8, pady=6)
            fr.grid(row=0, column=col, padx=4, sticky="nsew")
            tk.Label(fr, text=label, fg="#88ee88", bg=CARD,
                     font=("Consolas", 10, "bold"), justify="center").pack()
            val_lbl = tk.Label(fr, text=f"{TM_DEFAULTS[key]:.3g}",
                               fg=VAL_C, bg=CARD,
                               font=("Consolas", 16, "bold"))
            val_lbl.pack(pady=(2, 1))
            tk.Label(fr, text=tip, fg="#7080a0", bg=CARD,
                     font=("Consolas", 7), justify="left").pack(pady=(2, 4))

            def _cmd(v, vl=val_lbl):
                vl.config(text=f"{float(v):.3g}")
                if self._pending:
                    self.parent.after_cancel(self._pending)
                self._pending = self.parent.after(100, self._refresh)

            tk.Scale(fr, variable=self.vars[key], from_=lo, to=hi,
                     resolution=res, orient="horizontal", showvalue=False,
                     bg=CARD, fg=FG, troughcolor="#1a3a2a",
                     activebackground="#88ee88", highlightthickness=0,
                     command=_cmd).pack(fill="x")

        # Results table
        tbl = tk.Frame(parent, bg=TBL_BG)
        tbl.pack(fill="x", padx=10, pady=(0, 10))
        hdrs = ["Object", "Frames tracked", "Total travel (px)",
                "min_move gate", "min_travel gate", "Result"]
        wids = [16, 16, 18, 16, 16, 30]
        for c, (h, w) in enumerate(zip(hdrs, wids)):
            tk.Label(tbl, text=h, bg="#181830", fg=T_HDR,
                     font=("Consolas", 9, "bold"), width=w,
                     anchor="center", pady=4).grid(
                row=0, column=c, padx=2, pady=(4, 2), sticky="ew")
            tbl.columnconfigure(c, weight=1 if c == 5 else 0)
        self.res_cells = []
        for r in range(len(TM_OBJECTS)):
            row = []
            for c, w in enumerate(wids):
                lbl = tk.Label(tbl, text="", bg="#0e0e22", fg=T_DIM,
                               font=("Consolas", 10), width=w,
                               anchor="center", pady=3)
                lbl.grid(row=r + 1, column=c, padx=2, pady=1, sticky="ew")
                row.append(lbl)
            self.res_cells.append(row)

        self._refresh()

    def _refresh(self):
        p = {k: v.get() for k, v in self.vars.items()}
        tracks = run_tm_sim(int(p["min_move"]), p["min_travel"], int(p["ghost_frames"]))
        tl = render_timeline(tracks, p)
        img = Image.fromarray(cv2.cvtColor(tl, cv2.COLOR_BGR2RGB))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self.img_id, image=self.tk_img)

        # Update results table — one row per TM_OBJECT (merged across tracks)
        for oi, obj in enumerate(TM_OBJECTS):
            obj_tracks = [t for t in tracks if t["obj_idx"] == oi]
            row = self.res_cells[oi]
            row[0].config(text=obj["name"], fg=T_NAME_ON)
            if not obj_tracks:
                for c in range(1, 6):
                    row[c].config(text="—", fg=T_SKIP)
                continue
            total_frames = sum(len(t["positions"]) for t in obj_tracks)
            max_travel   = max(t["travel"] for t in obj_tracks)
            any_pass     = any(t["passed"] for t in obj_tracks)
            row[1].config(text=str(total_frames), fg=FG)
            row[2].config(text=f"{max_travel:.0f}", fg=FG)
            # min_move gate: did movement per frame ever fail?
            mm = int(p["min_move"])
            row[3].config(
                text="✓ passed" if mm == 0 else f"≥ {mm}px/frame",
                fg=T_PASS if mm == 0 else T_HDR)
            mt = p["min_travel"]
            row[4].config(
                text=f"{max_travel:.0f} {'≥' if max_travel >= mt else '<'} {mt:.0f}",
                fg=T_PASS if max_travel >= mt else T_FAIL)
            row[5].config(
                text="✓  EVENT SAVED" if any_pass else "✗  discarded",
                fg=T_PASS if any_pass else T_FAIL)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    root.title("StreakerDetect — Filter Demo")
    root.configure(bg=BG)
    root.resizable(False, False)

    style = ttk.Style()
    style.theme_use("default")
    style.configure("TNotebook",          background=BG, borderwidth=0)
    style.configure("TNotebook.Tab",      background="#1a1a30", foreground=FG,
                    padding=[14, 6], font=("Consolas", 10, "bold"))
    style.map("TNotebook.Tab",
              background=[("selected", "#2a3a5a")],
              foreground=[("selected", "#aaddff")])

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)

    tab1 = tk.Frame(nb, bg=BG)
    tab2 = tk.Frame(nb, bg=BG)
    nb.add(tab1, text="  Stage 1 — Per-Frame Filters  ")
    nb.add(tab2, text="  Stage 2 — TrackManager  ")

    PerFrameTab(tab1)
    TrackManagerTab(tab2)

    root.mainloop()


if __name__ == "__main__":
    main()
