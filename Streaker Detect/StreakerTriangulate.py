# StreakerTriangulate.py — 3D position solver for multi-camera events
#
# Workflow:
#   1. Load matches.json (from StreakerMatch)
#   2. Per camera: click "Load WCS" → picks streaker_astro_config.json from
#      a Local Solve session (supplies center_ra, center_dec, pixel_scale).
#      Rotation defaults to 0 (camera level); override if tilted.
#   3. Load sync_session.json (GPS + ref UTC per camera)
#   4. Select a match → Triangulate
#      • center RA/Dec is converted to Az/El at the match's UTC + observer GPS
#      • centroids → pixel offsets → Az/El → ECEF ray per camera
#      • least-squares ray intersection → lat/lon/altitude
#   5. Export result JSON
#
# Math: each camera produces a line-of-sight ray in ECEF space.
# The least-squares closest point to all rays is the object's 3D position.
# Coordinates: WGS84 GPS → ECEF → local ENU for display.

import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import math
import numpy as np
from datetime import datetime, timezone
from PIL import Image, ImageTk

# ── Appearance ────────────────────────────────────────────────────────────────

BG     = '#1a1a2e'
BG2    = '#16213e'
BG3    = '#0f0f1e'
FG     = '#e0e0e0'
ACCENT = '#0f3460'
DIM    = '#555555'
OK     = '#00cc88'
WARN   = '#ffcc00'
ERR    = '#cc3333'

MAX_CAMERAS = 3
CAM_COLORS  = ['#00aaff', '#ff8800', '#cc44ff']

# Default pixel scale for Sony A7S III + 24mm on full-frame, 4K 3840px
# HFOV = 2*arctan(35.6/(2*24)) = 73.74°  →  73.74°/3840 px × 3600 arcsec/° = 69.1 arcsec/px
DEFAULT_PIXEL_SCALE = 69.1
DEFAULT_FRAME_W     = 3840
DEFAULT_FRAME_H     = 2160

# ── Geodesy / ECEF math ───────────────────────────────────────────────────────

_WGS84_A  = 6378137.0           # semi-major axis (m)
_WGS84_E2 = 0.00669437999014    # eccentricity²

def gps_to_ecef(lat_deg, lon_deg, alt_m):
    """WGS84 geodetic → ECEF (metres)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    N   = _WGS84_A / math.sqrt(1 - _WGS84_E2 * math.sin(lat)**2)
    x   = (N + alt_m) * math.cos(lat) * math.cos(lon)
    y   = (N + alt_m) * math.cos(lat) * math.sin(lon)
    z   = (N * (1 - _WGS84_E2) + alt_m) * math.sin(lat)
    return np.array([x, y, z])

def ecef_to_gps(ecef):
    """ECEF (metres) → (lat_deg, lon_deg, alt_m) — Bowring iterative."""
    x, y, z = ecef
    lon = math.atan2(y, x)
    p   = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - _WGS84_E2))
    for _ in range(10):
        N   = _WGS84_A / math.sqrt(1 - _WGS84_E2 * math.sin(lat)**2)
        lat = math.atan2(z + _WGS84_E2 * N * math.sin(lat), p)
    N   = _WGS84_A / math.sqrt(1 - _WGS84_E2 * math.sin(lat)**2)
    alt = (p / math.cos(lat) - N) if abs(math.cos(lat)) > 1e-10 else \
          (abs(z) / math.sin(lat) - N * (1 - _WGS84_E2))
    return math.degrees(lat), math.degrees(lon), alt

def enu_to_ecef_dir(lat_deg, lon_deg, enu):
    """ENU unit vector → ECEF direction vector (at given surface point)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    sn, cn = math.sin(lon), math.cos(lon)
    # Rotation columns: East, North, Up in ECEF
    R = np.array([[-sn,  -sl*cn,  cl*cn],
                  [ cn,  -sl*sn,  cl*sn],
                  [  0,   cl,     sl   ]])
    return R @ np.array(enu)

def azel_to_enu(az_deg, el_deg):
    """Azimuth (0=N CW) + elevation → ENU unit vector."""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    return (math.cos(el)*math.sin(az),   # East
            math.cos(el)*math.cos(az),   # North
            math.sin(el))                # Up

def radec_to_azel(ra_deg, dec_deg, lat_deg, lon_deg, utc_dt):
    """
    Convert RA/Dec (J2000, degrees) to topocentric Az/El.

    Uses low-precision GMST formula (good to ~0.1° — sufficient for
    a pointing direction that will be refined by the centroid offset anyway).

    az  : degrees, 0 = North, clockwise
    el  : degrees, 0 = horizon
    """
    # Days since J2000.0
    J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    d = (utc_dt - J2000).total_seconds() / 86400.0

    # Greenwich Mean Sidereal Time (degrees)
    gmst = (280.46061837 + 360.98564736629 * d) % 360.0

    # Local Sidereal Time and Hour Angle
    lst = (gmst + lon_deg) % 360.0
    ha  = (lst - ra_deg) % 360.0
    if ha > 180:
        ha -= 360.0

    ha_r  = math.radians(ha)
    dec_r = math.radians(dec_deg)
    lat_r = math.radians(lat_deg)

    sin_el = (math.sin(dec_r) * math.sin(lat_r) +
              math.cos(dec_r) * math.cos(lat_r) * math.cos(ha_r))
    sin_el = max(-1.0, min(1.0, sin_el))
    el = math.degrees(math.asin(sin_el))

    cos_el = math.cos(math.radians(el))
    if cos_el < 1e-9:
        return 0.0, el   # at zenith, az is undefined

    cos_az = (math.sin(dec_r) - math.sin(lat_r) * sin_el) / \
             (math.cos(lat_r) * cos_el)
    cos_az = max(-1.0, min(1.0, cos_az))
    az = math.degrees(math.acos(cos_az))
    if math.sin(ha_r) > 0:
        az = 360.0 - az

    return az, el

def pixel_to_azel(px, py, frame_w, frame_h,
                  az_center, el_center, rot_deg, scale_arcsec_px):
    """
    Convert a pixel position to azimuth/elevation.

    px, py         : pixel in full-res frame
    frame_w/h      : frame dimensions
    az_center      : azimuth of frame centre (degrees, 0=N clockwise)
    el_center      : elevation of frame centre (degrees, 0=horizon)
    rot_deg        : camera rotation (degrees, CW positive)
    scale_arcsec_px: arcseconds per pixel
    """
    cx, cy = frame_w / 2, frame_h / 2
    dx =  (px - cx)     # positive = right in image
    dy = -(py - cy)     # positive = up (image y flipped)

    # Apply camera rotation
    rot = math.radians(rot_deg)
    dx_r = dx * math.cos(rot) - dy * math.sin(rot)
    dy_r = dx * math.sin(rot) + dy * math.cos(rot)

    scale_deg = scale_arcsec_px / 3600.0
    cos_el    = math.cos(math.radians(el_center))

    # Azimuth offset is foreshortened at high elevation
    daz = (dx_r * scale_deg) / max(cos_el, 0.01)
    del_ = dy_r * scale_deg

    return az_center + daz, el_center + del_

def triangulate_rays(positions, directions):
    """
    Least-squares closest point to N rays (positions + unit directions in ECEF).

    Minimises sum of squared perpendicular distances from point X to each ray.
    System: (Σ Mᵢ)·X = Σ Mᵢ·Pᵢ   where Mᵢ = I − dᵢdᵢᵀ

    Returns (best_point_ecef, mean_residual_m).
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for pos, d in zip(positions, directions):
        d = d / np.linalg.norm(d)
        M  = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ pos
    try:
        X = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        X = np.linalg.lstsq(A, b, rcond=None)[0]

    residuals = []
    for pos, d in zip(positions, directions):
        d = d / np.linalg.norm(d)
        w    = X - pos
        perp = w - np.dot(w, d) * d
        residuals.append(float(np.linalg.norm(perp)))
    return X, float(np.mean(residuals))

# ── Event data helpers ────────────────────────────────────────────────────────

def load_event_centroids(event_dir, detect_scale=1.0):
    """
    Load all centroids from metadata.json, converted to full-res pixels.
    Returns list of (frame_num, x_fullres, y_fullres).
    """
    meta_path = os.path.join(event_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        return []
    with open(meta_path) as f:
        meta = json.load(f)
    ds = meta.get('detect_scale', detect_scale) or 1.0
    pts = []
    for det in meta.get('detections', []):
        fn = det.get('frame', 0)
        for cx, cy in det.get('centroids', []):
            pts.append((fn, cx / ds, cy / ds))
    return pts

def _parse_utc(s):
    fmts = ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%f',  '%Y-%m-%dT%H:%M:%S']
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None

# ── Map canvas helper ─────────────────────────────────────────────────────────

def _draw_map(canvas, cam_positions_gps, object_gps, cam_colors):
    """
    Draw a north-up overhead map of camera positions and object.
    cam_positions_gps: list of (lat, lon) tuples (None = not participating)
    object_gps: (lat, lon) or None
    """
    canvas.delete('all')
    cw = canvas.winfo_width()  or 300
    ch = canvas.winfo_height() or 300
    margin = 30

    # Collect all valid points
    points = [(lat, lon) for lat, lon in cam_positions_gps if lat is not None]
    if object_gps:
        points.append(object_gps)
    if not points:
        canvas.create_text(cw//2, ch//2, text="No positions", fill=DIM)
        return

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    lat_c = sum(lats) / len(lats)
    lon_c = sum(lons) / len(lons)

    # Meters per degree at this latitude
    m_per_lat = 111320.0
    m_per_lon = 111320.0 * math.cos(math.radians(lat_c))

    # Convert to local metres (North-up: y = north, x = east)
    def to_local(lat, lon):
        return (lon - lon_c) * m_per_lon, (lat - lat_c) * m_per_lat

    local_pts = [to_local(lat, lon) for lat, lon in points]
    xs = [p[0] for p in local_pts]
    ys = [p[1] for p in local_pts]
    span_x = max(abs(max(xs) - min(xs)), 1)
    span_y = max(abs(max(ys) - min(ys)), 1)
    scale   = min((cw - 2*margin) / span_x, (ch - 2*margin) / span_y)

    def to_canvas(x, y):
        # North = up → flip y
        cx_ = cw // 2 + x * scale
        cy_ = ch // 2 - y * scale
        return cx_, cy_

    # Draw compass rose
    canvas.create_text(cw//2, margin//2, text="N", fill=DIM, font=('Arial', 7))

    # Draw cameras
    for i, (lat, lon) in enumerate(cam_positions_gps):
        if lat is None:
            continue
        lx, ly = to_local(lat, lon)
        cx_, cy_ = to_canvas(lx, ly)
        col = cam_colors[i]
        r = 8
        canvas.create_oval(cx_-r, cy_-r, cx_+r, cy_+r, fill=col, outline='white', width=1)
        canvas.create_text(cx_, cy_-r-6, text=f"C{i+1}", fill=col, font=('Arial', 7, 'bold'))

    # Draw object
    if object_gps:
        lx, ly = to_local(*object_gps)
        cx_, cy_ = to_canvas(lx, ly)
        r = 7
        canvas.create_oval(cx_-r, cy_-r, cx_+r, cy_+r,
                           fill='white', outline=OK, width=2)
        canvas.create_text(cx_, cy_-r-7, text="OBJ", fill=OK, font=('Arial', 7, 'bold'))

        # Draw lines from each camera to object
        for i, (lat, lon) in enumerate(cam_positions_gps):
            if lat is None:
                continue
            lx2, ly2 = to_local(lat, lon)
            cx2, cy2 = to_canvas(lx2, ly2)
            canvas.create_line(cx2, cy2, cx_, cy_, fill=cam_colors[i],
                               dash=(4, 3), width=1)

    # Scale bar: 100m or 1km depending on span
    span_m = max(span_x, span_y)
    bar_m  = 100 if span_m < 1000 else 1000
    bar_px = bar_m * scale
    bx1, by_ = margin, ch - margin//2
    canvas.create_line(bx1, by_, bx1 + bar_px, by_, fill=DIM, width=2)
    canvas.create_text(bx1 + bar_px//2, by_ - 8,
                       text=f"{bar_m}m" if bar_m < 1000 else "1 km",
                       fill=DIM, font=('Arial', 6))

# ── Main app ──────────────────────────────────────────────────────────────────

class StreakerTriangulate:

    def __init__(self, root, run_dir=None):
        self.root = root
        self.root.title("Streaker Triangulate")
        self.root.configure(bg=BG)
        self.root.state('zoomed')

        self.matches_path = None
        self.matches      = []           # raw match dicts from JSON
        self.session      = None         # sync session dict (optional, for GPS)
        self._result      = None         # last triangulation result
        self._run_dir     = run_dir      # hint dir from StreakerPlayer

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill='x', side='top', padx=6, pady=(4, 0))

        tk.Label(top, text="STREAKER TRIANGULATE", bg=BG, fg='white',
                 font=('Arial', 10, 'bold')).pack(side='left', padx=4)
        tk.Label(top, text="│", bg=BG, fg=DIM).pack(side='left', padx=2)
        tk.Button(top, text="📂 Load Matches",
                  command=self._load_matches,
                  bg='#334455', fg='white', relief='flat',
                  padx=6, pady=2).pack(side='left', padx=2)
        tk.Button(top, text="📂 Load Session (GPS)",
                  command=self._load_session,
                  bg='#334455', fg='white', relief='flat',
                  padx=6, pady=2).pack(side='left', padx=2)
        tk.Button(top, text="💾 Export Results",
                  command=self._export_results,
                  bg='#224422', fg='white', relief='flat',
                  padx=6, pady=2).pack(side='left', padx=2)

        self.top_lbl = tk.Label(top, text="Load matches.json to begin",
                                bg=BG, fg=DIM, font=('Courier', 8))
        self.top_lbl.pack(side='left', padx=12)

        # ── Camera orientations panel ─────────────────────────────────────────
        # Orientation comes from plate solve (RA/Dec + pixel scale).
        # Az/El is computed at triangulation time using GPS + match UTC.
        orient_frame = tk.Frame(self.root, bg=BG2)
        orient_frame.pack(fill='x', side='top', padx=0, pady=(4, 0))

        tk.Label(orient_frame, text="Camera Orientations  (from plate solve)",
                 bg=BG2, fg='#888', font=('Arial', 8, 'bold')).grid(
                     row=0, column=0, padx=8, pady=(4, 0), columnspan=2, sticky='w')
        tk.Label(orient_frame,
                 text="RA center (°)   Dec center (°)   Rot (°)   Scale (\"/px)   W px   H px",
                 bg=BG2, fg=DIM, font=('Courier', 7)).grid(
                     row=0, column=2, padx=4, pady=(4, 0), columnspan=7, sticky='w')

        # _orient_vars[i] = [ra_var, dec_var, rot_var, scale_var, fw_var, fh_var, gps_lbl]
        self._orient_vars = []
        for i in range(MAX_CAMERAS):
            color = CAM_COLORS[i]
            tk.Label(orient_frame, text=f"Camera {i+1}", bg=BG2, fg=color,
                     font=('Arial', 8, 'bold')).grid(row=i+1, column=0, padx=(8, 2), pady=2)

            tk.Button(orient_frame, text="Load WCS",
                      command=lambda idx=i: self._load_wcs(idx),
                      bg='#1a3a5a', fg='white', relief='flat',
                      font=('Arial', 7), padx=4).grid(row=i+1, column=1, padx=2, pady=2)

            vars_ = []
            defaults = ['0.0', '0.0', '0.0', str(DEFAULT_PIXEL_SCALE),
                        str(DEFAULT_FRAME_W), str(DEFAULT_FRAME_H)]
            widths_  = [10, 10, 6, 6, 6, 6]
            for col_idx, (default, width) in enumerate(zip(defaults, widths_)):
                v = tk.StringVar(value=default)
                tk.Entry(orient_frame, textvariable=v, width=width,
                         bg='#222', fg=FG, insertbackground=FG,
                         font=('Courier', 8)).grid(
                             row=i+1, column=2+col_idx, padx=2, pady=2)
                vars_.append(v)

            gps_lbl = tk.Label(orient_frame, text="GPS: —", bg=BG2, fg=DIM,
                               font=('Arial', 6), width=30, anchor='w')
            gps_lbl.grid(row=i+1, column=8, padx=4)
            vars_.append(gps_lbl)   # index 6

            self._orient_vars.append(vars_)

        # ── Main area: match list (left) + results (right) ────────────────────
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True, padx=4, pady=4)

        # Match list
        list_frame = tk.Frame(main, bg=BG3, width=340)
        list_frame.pack(side='left', fill='y')
        list_frame.pack_propagate(False)

        tk.Label(list_frame, text="MATCHES", bg=BG3, fg='#888',
                 font=('Arial', 8, 'bold')).pack(anchor='w', padx=6, pady=(4, 0))

        scroll_area = tk.Frame(list_frame, bg=BG3)
        scroll_area.pack(fill='both', expand=True)
        sb = tk.Scrollbar(scroll_area, orient='vertical')
        sb.pack(side='right', fill='y')
        self.list_canvas = tk.Canvas(scroll_area, bg=BG3,
                                     highlightthickness=0, yscrollcommand=sb.set)
        sb.config(command=self.list_canvas.yview)
        self.list_canvas.pack(side='left', fill='both', expand=True)
        self.list_canvas.bind('<MouseWheel>',
                              lambda e: self.list_canvas.yview_scroll(
                                  -1*(e.delta//120), 'units'))
        self.list_inner = tk.Frame(self.list_canvas, bg=BG3)
        self._lwin = self.list_canvas.create_window((0,0), window=self.list_inner, anchor='nw')
        self.list_inner.bind('<Configure>',
            lambda e: self.list_canvas.configure(
                scrollregion=self.list_canvas.bbox('all')))

        # Results panel
        res_frame = tk.Frame(main, bg=BG2)
        res_frame.pack(side='right', fill='both', expand=True, padx=(4, 0))

        tk.Label(res_frame, text="TRIANGULATION RESULT", bg=BG2, fg='#888',
                 font=('Arial', 8, 'bold')).pack(anchor='w', padx=8, pady=(4, 0))

        # Text results
        self.result_text = tk.Text(res_frame, bg=BG3, fg=FG,
                                   font=('Courier', 9), height=12,
                                   relief='flat', state='disabled',
                                   wrap='word', padx=8, pady=8)
        self.result_text.pack(fill='x', padx=6, pady=4)

        # Map canvas
        tk.Label(res_frame, text="OVERHEAD MAP  (north up)",
                 bg=BG2, fg=DIM, font=('Arial', 7)).pack(anchor='w', padx=8)
        self.map_canvas = tk.Canvas(res_frame, bg='#0a0a1a',
                                    highlightthickness=1,
                                    highlightbackground='#333')
        self.map_canvas.pack(fill='both', expand=True, padx=6, pady=(0, 6))
        self.map_canvas.bind('<Configure>', lambda e: self._redraw_map())
        self._map_data = None   # (cam_gps_list, object_gps) for redraw

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_matches(self):
        path = filedialog.askopenfilename(
            title="Load matches.json",
            initialdir=self._run_dir or None,
            filetypes=[('JSON', '*.json'), ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.matches_path = path
        self.matches = data.get('matches', [])
        self.top_lbl.config(
            text=f"{len(self.matches)} matches — {os.path.basename(path)}",
            fg=OK)
        self._render_match_list()

    def _load_wcs(self, cam_idx):
        """Load plate solve result from streaker_astro_config.json for one camera."""
        path = filedialog.askopenfilename(
            title=f"Load plate solve config for Camera {cam_idx + 1}",
            filetypes=[('JSON config', '*.json'), ('All files', '*.*')],
            initialfile='streaker_astro_config.json')
        if not path:
            return
        try:
            with open(path) as f:
                cfg = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        vs = self._orient_vars[cam_idx]
        ra  = cfg.get('center_ra')
        dec = cfg.get('center_dec')
        px  = cfg.get('pixel_scale_arcsec')

        if ra is None or dec is None:
            messagebox.showwarning("Incomplete",
                "File must contain center_ra and center_dec (run Local Solve first).")
            return

        vs[0].set(str(round(ra,  6)))   # RA
        vs[1].set(str(round(dec, 6)))   # Dec
        if px:
            vs[3].set(str(round(px, 2)))  # pixel scale

        color = CAM_COLORS[cam_idx]
        vs[6].config(text=f"WCS RA {ra:.3f}  Dec {dec:.3f}", fg=color)

    def _load_session(self):
        path = filedialog.askopenfilename(
            title="Load sync_session.json (for GPS)",
            filetypes=[('JSON', '*.json'), ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path) as f:
                self.session = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        # Populate GPS labels in orientation panel
        for i, cam in enumerate(self.session.get('cameras', [])[:MAX_CAMERAS]):
            gps = cam.get('gps', {})
            lat = gps.get('lat', '')
            lon = gps.get('lon', '')
            alt = gps.get('alt', '')
            lbl = self._orient_vars[i][6]   # GPS label
            if lat and lon:
                lbl.config(text=f"GPS {lat:.5f}, {lon:.5f} {alt}m", fg=OK)
            else:
                lbl.config(text="GPS: not set", fg=WARN)
        self.top_lbl.config(text=self.top_lbl.cget('text') + "  + session GPS", fg=OK)

    # ── Match list rendering ──────────────────────────────────────────────────

    def _render_match_list(self):
        for w in self.list_inner.winfo_children():
            w.destroy()
        if not self.matches:
            tk.Label(self.list_inner, text="No matches", bg=BG3, fg=DIM).pack(pady=20)
            return
        for idx, m in enumerate(self.matches):
            self._render_match_row(idx, m)

    def _render_match_row(self, idx, m):
        n   = m.get('n_cams', len(m.get('cameras', {})))
        col = OK if n == 3 else WARN
        start_utc = m.get('start_utc', '')
        dur = m.get('duration_s', 0)

        row = tk.Frame(self.list_inner, bg=BG2,
                       highlightbackground=col, highlightthickness=1)
        row.pack(fill='x', padx=4, pady=2)

        info = tk.Frame(row, bg=BG2)
        info.pack(side='left', fill='x', expand=True, padx=6, pady=4)

        time_str = start_utc[11:23] if len(start_utc) > 11 else start_utc
        cams_str = ', '.join(f"C{c}" for c in sorted(m.get('cameras', {}).keys()))
        tk.Label(info, text=f"#{idx+1}  {time_str}", bg=BG2, fg=FG,
                 font=('Courier', 9, 'bold')).pack(anchor='w')
        tk.Label(info, text=f"● {cams_str}   {dur:.2f}s", bg=BG2, fg=col,
                 font=('Arial', 7)).pack(anchor='w')

        tk.Button(row, text="Triangulate",
                  command=lambda i=idx: self._triangulate(i),
                  bg='#006600', fg='white', relief='flat',
                  padx=6, pady=2, font=('Arial', 7, 'bold')).pack(
                      side='right', padx=4, pady=4)

    # ── Triangulation ─────────────────────────────────────────────────────────

    def _get_orient(self, cam_key):
        """Parse orientation vars for camera key (string '1','2','3' or 0-based int)."""
        idx = int(cam_key) - 1 if str(cam_key).isdigit() else int(cam_key)
        if idx < 0 or idx >= MAX_CAMERAS:
            return None
        vs = self._orient_vars[idx]
        try:
            return {
                'ra':    float(vs[0].get()),
                'dec':   float(vs[1].get()),
                'rot':   float(vs[2].get()),
                'scale': float(vs[3].get()),
                'fw':    int(vs[4].get()),
                'fh':    int(vs[5].get()),
            }
        except ValueError:
            return None

    def _get_gps(self, cam_key):
        """Get GPS for camera from session or matches.json."""
        idx = int(cam_key) - 1 if str(cam_key).isdigit() else int(cam_key)
        # Try session first
        if self.session:
            cams = self.session.get('cameras', [])
            if idx < len(cams):
                gps = cams[idx].get('gps', {})
                lat = gps.get('lat')
                lon = gps.get('lon')
                alt = gps.get('alt') or 0.0
                if lat is not None and lon is not None:
                    return float(lat), float(lon), float(alt)
        return None

    def _triangulate(self, match_idx):
        m = self.matches[match_idx]
        cameras = m.get('cameras', {})

        cam_positions_ecef = []
        cam_directions_ecef = []
        cam_gps_for_map = [None] * MAX_CAMERAS
        details = []

        for cam_key, cam_ev in cameras.items():
            idx = int(cam_key) - 1
            orient = self._get_orient(cam_key)
            if orient is None:
                messagebox.showwarning("Missing orientation",
                    f"Check orientation values for Camera {cam_key}.")
                return

            gps = self._get_gps(cam_key)
            if gps is None:
                # Try matches.json embedded GPS
                raw_gps = cam_ev.get('gps', {})
                lat = raw_gps.get('lat')
                lon = raw_gps.get('lon')
                alt = raw_gps.get('alt') or 0.0
                if lat is None:
                    messagebox.showwarning("Missing GPS",
                        f"No GPS for Camera {cam_key}. "
                        "Load sync session or check matches.json.")
                    return
                gps = (float(lat), float(lon), float(alt))

            lat, lon, alt = gps
            cam_gps_for_map[idx] = (lat, lon)

            # Mean centroid in full-res pixels
            ev_dir = cam_ev.get('event_dir', '')
            centroids = load_event_centroids(ev_dir)
            if not centroids:
                messagebox.showwarning("No centroids",
                    f"Camera {cam_key}: no detection centroids in {ev_dir}")
                return

            mean_x = sum(c[1] for c in centroids) / len(centroids)
            mean_y = sum(c[2] for c in centroids) / len(centroids)

            # Convert plate-solve RA/Dec → Az/El at match UTC + observer GPS
            match_utc = _parse_utc(m.get('start_utc', ''))
            if match_utc is None:
                messagebox.showwarning("No time",
                    "Match has no start_utc — cannot convert RA/Dec to Az/El.")
                return
            az_center, el_center = radec_to_azel(
                orient['ra'], orient['dec'], lat, lon, match_utc)

            # Pixel → azimuth/elevation
            az_obj, el_obj = pixel_to_azel(
                mean_x, mean_y,
                orient['fw'], orient['fh'],
                az_center, el_center,
                orient['rot'], orient['scale'])

            # AZ/EL → ENU → ECEF direction
            enu = azel_to_enu(az_obj, el_obj)
            d_ecef = enu_to_ecef_dir(lat, lon, enu)
            d_ecef = d_ecef / np.linalg.norm(d_ecef)

            pos_ecef = gps_to_ecef(lat, lon, alt)
            cam_positions_ecef.append(pos_ecef)
            cam_directions_ecef.append(d_ecef)

            # Angular velocity (first→last centroid)
            ang_vel = None
            if len(centroids) >= 2:
                fps = cam_ev.get('fps', 24.0)
                fn0, x0, y0 = centroids[0]
                fn1, x1, y1 = centroids[-1]
                dt = (fn1 - fn0) / max(fps, 1)
                if dt > 0:
                    dpx = math.hypot(x1-x0, y1-y0)
                    ang_vel = dpx * orient['scale'] / 3600.0 / dt  # deg/s

            details.append({
                'cam':        cam_key,
                'lat':        lat, 'lon': lon, 'alt': alt,
                'ra_center':  orient['ra'],  'dec_center': orient['dec'],
                'az_center':  az_center,     'el_center':  el_center,
                'az_obj':     az_obj,        'el_obj':     el_obj,
                'mean_px':    (mean_x, mean_y),
                'ang_vel':    ang_vel,
                'n_pts':      len(centroids),
            })

        if len(cam_positions_ecef) < 2:
            messagebox.showwarning("Need ≥2 cameras",
                "At least 2 cameras with GPS + orientation required.")
            return

        # Solve
        obj_ecef, residual_m = triangulate_rays(cam_positions_ecef,
                                                cam_directions_ecef)
        obj_lat, obj_lon, obj_alt = ecef_to_gps(obj_ecef)

        # Per-camera distances
        distances = [float(np.linalg.norm(obj_ecef - p))
                     for p in cam_positions_ecef]

        self._result = {
            'match_idx':  match_idx,
            'match_utc':  m.get('start_utc', ''),
            'lat':   obj_lat, 'lon': obj_lon, 'alt': obj_alt,
            'residual_m': residual_m,
            'cameras':    details,
            'distances_m': distances,
        }

        self._display_result(details, obj_lat, obj_lon, obj_alt,
                             residual_m, distances, cam_gps_for_map)

    def _display_result(self, details, lat, lon, alt,
                        residual_m, distances, cam_gps_for_map):
        lines = []
        lines.append(f"Object position")
        lines.append(f"  Lat  : {lat:+.6f}°")
        lines.append(f"  Lon  : {lon:+.6f}°")
        lines.append(f"  Alt  : {alt:,.0f} m  ({alt/1000:.2f} km)")
        lines.append(f"  Residual: {residual_m:.1f} m")
        lines.append("")
        for i, (d, cam_d) in enumerate(zip(distances, details)):
            lines.append(f"Camera {cam_d['cam']}")
            lines.append(f"  GPS    : {cam_d['lat']:.5f}, {cam_d['lon']:.5f}, "
                         f"{cam_d['alt']:.0f}m")
            lines.append(f"  Plate solve: RA {cam_d['ra_center']:.3f}°  "
                         f"Dec {cam_d['dec_center']:.3f}°")
            lines.append(f"  Frame center: Az {cam_d['az_center']:.2f}°  "
                         f"El {cam_d['el_center']:.2f}°  (at match UTC)")
            lines.append(f"  Object Az/El: {cam_d['az_obj']:.2f}°  "
                         f"{cam_d['el_obj']:.2f}°")
            lines.append(f"  Pixel  : ({cam_d['mean_px'][0]:.0f}, "
                         f"{cam_d['mean_px'][1]:.0f})  "
                         f"from {cam_d['n_pts']} detections")
            lines.append(f"  Distance: {d/1000:.3f} km")
            if cam_d['ang_vel'] is not None:
                spd = cam_d['ang_vel'] * math.pi/180 * d   # m/s
                lines.append(f"  Angular vel: {cam_d['ang_vel']:.3f} °/s  "
                             f"≈ {spd:.1f} m/s")
            lines.append("")

        self.result_text.config(state='normal')
        self.result_text.delete('1.0', 'end')
        self.result_text.insert('end', '\n'.join(lines))
        self.result_text.config(state='disabled')

        self._map_data = (cam_gps_for_map, (lat, lon))
        self._redraw_map()

    def _redraw_map(self):
        if self._map_data is None:
            return
        cam_gps, obj_gps = self._map_data
        _draw_map(self.map_canvas, cam_gps, obj_gps, CAM_COLORS)

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_results(self):
        if self._result is None:
            messagebox.showwarning("No result", "Triangulate a match first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export result",
            defaultextension='.json',
            filetypes=[('JSON', '*.json')],
            initialfile='triangulation_result.json')
        if not path:
            return
        with open(path, 'w') as f:
            json.dump(self._result, f, indent=2, default=str)
        messagebox.showinfo("Saved", f"Result saved to:\n{path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import sys
    run_dir = None
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == '--run-dir' and i + 1 < len(args):
            run_dir = args[i + 1]
    root = tk.Tk()
    app  = StreakerTriangulate(root, run_dir=run_dir)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == '__main__':
    main()
