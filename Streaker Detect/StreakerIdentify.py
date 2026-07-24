# ------------------------------------------------------------------------------
# StreakerIdentify.py
# Post-detection identification pass.
# Reads metadata.json from each event in a detect run folder, classifies each
# event (meteor / aircraft / satellite / unknown), flags anomalies, and writes
# identification.json alongside each event's metadata.
# ------------------------------------------------------------------------------

import os
import sys
import json
import math
import datetime
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path
from collections import defaultdict

try:
    import tkinter as tk
    from tkinter import filedialog
    _TK_AVAILABLE = True
except ImportError:
    _TK_AVAILABLE = False

# Add sibling directory so we can import platform_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from platform_utils import FFMPEG_PATH, NO_WINDOW, find_ffprobe

FFPROBE_PATH = find_ffprobe(FFMPEG_PATH)

VERSION = "1.0.0"

# ── Thresholds ─────────────────────────────────────────────────────────────────
METEOR_RATE_MIN      = 15.0   # deg/s — above this: meteor
SATELLITE_RATE_MIN   =  0.1   # deg/s — below this: not a LEO satellite
AIRCRAFT_RATE_MAX    =  8.0   # deg/s — above this: unlikely aircraft
RATE_MATCH_TOL       =  0.35  # deg/s — how close a match must be to count
TLE_MAX_EPOCH_DAYS   = 30     # ignore TLEs older than this

# ── Camera calibration ─────────────────────────────────────────────────────────

class CameraCalibration:
    """Pinhole camera model loaded from YAML or hardcoded values."""

    def __init__(self, fx, fy, cx, cy, width, height):
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.width, self.height = width, height
        self.hfov = 2 * math.degrees(math.atan(width  / (2 * fx)))
        self.vfov = 2 * math.degrees(math.atan(height / (2 * fy)))
        self.deg_per_px_x = self.hfov / width
        self.deg_per_px_y = self.vfov / height

    @classmethod
    def from_yaml(cls, path):
        import re
        with open(path) as f:
            txt = f.read()
        def g(key):
            m = re.search(rf'{key}:\s*([\d.]+)', txt)
            return float(m.group(1)) if m else None
        size_m = re.search(r'image_size:\s*\n\s*-\s*(\d+)\s*\n\s*-\s*(\d+)', txt)
        w, h = (int(size_m.group(1)), int(size_m.group(2))) if size_m else (2592, 1944)
        return cls(fx=g('fx'), fy=g('fy'), cx=g('cx'), cy=g('cy'), width=w, height=h)

    @classmethod
    def dahua_n53(cls):
        return cls(fx=2072.3336555748688, fy=2010.877781963462,
                   cx=1183.57385460179,   cy=1065.1708417590967,
                   width=2592, height=1944)

    @classmethod
    def canon_r_24mm(cls):
        return cls(fx=2411.0661819244197, fy=2417.4637415487236,
                   cx=1689.2134956346508, cy=1134.9890125610566,
                   width=3360, height=2240)

    def angular_velocity(self, vx_px_frame, vy_px_frame, fps, detect_scale=1.0):
        """Convert (pixels/frame at detect_scale) → deg/s."""
        full_vx = vx_px_frame / max(detect_scale, 1e-6)
        full_vy = vy_px_frame / max(detect_scale, 1e-6)
        deg_s = math.sqrt((full_vx * self.deg_per_px_x)**2 +
                          (full_vy * self.deg_per_px_y)**2) * fps
        return deg_s

    def pixel_to_angle(self, px, py):
        """Pixel → (angle_x, angle_y) relative to optical axis, in degrees."""
        return (math.degrees(math.atan((px - self.cx) / self.fx)),
                math.degrees(math.atan((py - self.cy) / self.fy)))


# ── Timestamp recovery ─────────────────────────────────────────────────────────

def get_clip_start_utc(clip_path):
    """
    Return UTC datetime for when the clip started, or None.
    Tries ffprobe container metadata first, then falls back to file mtime.
    """
    try:
        result = subprocess.run(
            [FFPROBE_PATH, '-v', 'quiet', '-print_format', 'json', '-show_format',
             clip_path],
            capture_output=True, text=True, timeout=10,
            creationflags=NO_WINDOW)
        data = json.loads(result.stdout)
        ts = data.get('format', {}).get('tags', {}).get('creation_time', '')
        if ts:
            return datetime.datetime.fromisoformat(
                ts.replace('Z', '+00:00')).astimezone(datetime.timezone.utc)
    except Exception:
        pass
    # Fallback: file modification time (approximate)
    try:
        mtime = os.path.getmtime(clip_path)
        return datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
    except Exception:
        return None


# ── Event analysis ─────────────────────────────────────────────────────────────

def analyze_event(metadata, calib):
    """
    Fit a linear track through detection centroids.
    Returns angular rate, direction, duration, and displacement.
    """
    dets = [d for d in metadata.get('detections', []) if d.get('centroids')]
    fps   = metadata.get('fps', 20)
    scale = metadata.get('detect_scale', 0.5)

    empty = {'angular_rate_deg_s': 0.0, 'direction_image_deg': None,
             'duration_s': 0.0, 'total_displacement_deg': 0.0,
             'n_detections': len(dets), 'vx': 0.0, 'vy': 0.0}
    if len(dets) < 2:
        return empty

    xs = [d['centroids'][0][0] for d in dets]
    ys = [d['centroids'][0][1] for d in dets]
    fs = [d['frame']           for d in dets]

    n   = len(fs)
    sf  = sum(fs);  sx = sum(xs);  sy = sum(ys)
    sf2 = sum(f*f for f in fs)
    sfx = sum(fs[i]*xs[i] for i in range(n))
    sfy = sum(fs[i]*ys[i] for i in range(n))
    denom = n * sf2 - sf * sf

    vx = vy = 0.0
    if denom:
        vx = (n * sfx - sf * sx) / denom   # px/frame at detect scale
        vy = (n * sfy - sf * sy) / denom

    rate     = calib.angular_velocity(vx, vy, fps, scale)
    dir_img  = math.degrees(math.atan2(vy, vx)) % 360  # 0=right, 90=down

    inv = 1.0 / max(scale, 1e-6)
    dx  = (xs[-1] - xs[0]) * inv
    dy  = (ys[-1] - ys[0]) * inv
    disp_deg = math.sqrt((dx * calib.deg_per_px_x)**2 +
                         (dy * calib.deg_per_px_y)**2)
    duration  = (fs[-1] - fs[0]) / max(fps, 1)

    return {
        'angular_rate_deg_s':    round(rate, 3),
        'direction_image_deg':   round(dir_img, 1),
        'duration_s':            round(duration, 2),
        'total_displacement_deg': round(disp_deg, 3),
        'n_detections': n,
        'vx': round(vx, 4),
        'vy': round(vy, 4),
        'start_centroid': [round(xs[0], 1), round(ys[0], 1)],
        'end_centroid':   [round(xs[-1], 1), round(ys[-1], 1)],
    }


# ── Coordinate math ────────────────────────────────────────────────────────────

def latlon_to_ecef(lat_deg, lon_deg, alt_m=0.0):
    a  = 6378137.0; f = 1 / 298.257223563
    e2 = 2*f - f*f
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    N  = a / math.sqrt(1 - e2 * math.sin(lat)**2)
    x  = (N + alt_m) * math.cos(lat) * math.cos(lon)
    y  = (N + alt_m) * math.cos(lat) * math.sin(lon)
    z  = (N * (1 - e2) + alt_m) * math.sin(lat)
    return x, y, z

def ecef_to_enu(dx, dy, dz, lat_deg, lon_deg):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    e = -math.sin(lon)*dx + math.cos(lon)*dy
    n = (-math.sin(lat)*math.cos(lon)*dx
         - math.sin(lat)*math.sin(lon)*dy
         + math.cos(lat)*dz)
    u = (math.cos(lat)*math.cos(lon)*dx
         + math.cos(lat)*math.sin(lon)*dy
         + math.sin(lat)*dz)
    return e, n, u

def enu_to_azel(e, n, u):
    rng = math.sqrt(e*e + n*n + u*u)
    if rng < 1:
        return 0.0, 0.0
    az = math.degrees(math.atan2(e, n)) % 360
    el = math.degrees(math.asin(max(-1.0, min(1.0, u / rng))))
    return az, el

def gmst_rad(dt_utc):
    """Greenwich Mean Sidereal Time in radians (J2000 formula)."""
    j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    t_days = (dt_utc - j2000).total_seconds() / 86400.0
    return math.radians((280.46061837 + 360.98564736629 * t_days) % 360)

def teme_to_ecef(x, y, z, dt_utc):
    """Rotate TEME (km) → ECEF (m) using GMST."""
    theta = gmst_rad(dt_utc)
    xe = (x * math.cos(theta) + y * math.sin(theta)) * 1000
    ye = (-x * math.sin(theta) + y * math.cos(theta)) * 1000
    ze = z * 1000
    return xe, ye, ze


# ── FlightAware AeroAPI ───────────────────────────────────────────────────────

def query_flightaware(lat, lon, dt_utc, api_key, radius_deg=2.5):
    """
    Query FlightAware AeroAPI v4 for flights in a bounding box around (lat, lon)
    within ±5 minutes of dt_utc.
    Returns list of aircraft dicts (same schema as query_opensky) or [].
    """
    if not api_key:
        return []

    lamin = lat - radius_deg
    lamax = lat + radius_deg
    lomin = lon - radius_deg
    lomax = lon + radius_deg

    start = (dt_utc - datetime.timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end   = (dt_utc + datetime.timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
    fql   = f'-latlong "{lamin} {lomin} {lamax} {lomax}"'

    url = ('https://aeroapi.flightaware.com/aeroapi/flights/search?'
           + urllib.parse.urlencode({'query': fql, 'start': start, 'end': end}))
    req = urllib.request.Request(url, headers={
        'x-apikey': api_key,
        'Accept':   'application/json; charset=UTF-8',
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f'    [FlightAware] query failed: HTTP {e.code} — {e.reason}')
        return []
    except Exception as e:
        print(f'    [FlightAware] query failed: {e}')
        return []

    obs_x, obs_y, obs_z = latlon_to_ecef(lat, lon, 100)
    result = []
    for flight in (data.get('flights') or []):
        pos = flight.get('last_position') or {}
        ac_lat = pos.get('latitude')
        ac_lon = pos.get('longitude')
        if ac_lat is None or ac_lon is None:
            continue
        alt_ft = pos.get('altitude') or 0      # feet MSL
        alt_m  = alt_ft * 0.3048
        spd_kt = pos.get('groundspeed') or 0   # knots
        spd_ms = spd_kt * 0.514444

        ax, ay, az_ = latlon_to_ecef(ac_lat, ac_lon, alt_m)
        e, n, u = ecef_to_enu(ax - obs_x, ay - obs_y, az_ - obs_z, lat, lon)
        if u <= 0:
            continue

        az_deg, el_deg = enu_to_azel(e, n, u)
        dist     = math.sqrt(e*e + n*n + u*u)
        ang_rate = math.degrees(spd_ms / dist) if dist > 0 and spd_ms > 0 else 0.0

        result.append({
            'icao24':             (flight.get('registration') or '').strip(),
            'callsign':           (flight.get('ident') or '').strip(),
            'lat':                ac_lat,
            'lon':                ac_lon,
            'alt_m':              round(alt_m),
            'speed_ms':           round(spd_ms, 1),
            'az':                 round(az_deg, 1),
            'el':                 round(el_deg, 1),
            'dist_m':             round(dist),
            'angular_rate_deg_s': round(ang_rate, 3),
        })
    return result


# ── TLE / SGP4 ────────────────────────────────────────────────────────────────

def _parse_tle_epoch(line1):
    """Return epoch as UTC datetime from TLE line 1."""
    try:
        yr2 = int(line1[18:20])
        year = 2000 + yr2 if yr2 < 57 else 1900 + yr2
        day_frac = float(line1[20:32])
        return (datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
                + datetime.timedelta(days=day_frac - 1))
    except Exception:
        return None

def load_tle_catalog(tle_path, max_epoch_days=TLE_MAX_EPOCH_DAYS):
    """Load TLE file; optionally filter by epoch age."""
    entries = []
    cutoff  = (datetime.datetime.now(datetime.timezone.utc)
               - datetime.timedelta(days=max_epoch_days)) if max_epoch_days else None
    try:
        with open(tle_path, encoding='utf-8', errors='replace') as f:
            lines = [l.rstrip() for l in f if l.strip()]
    except Exception as e:
        print(f'[TLE] Cannot read {tle_path}: {e}')
        return []

    i = 0
    while i < len(lines) - 2:
        if lines[i+1].startswith('1 ') and lines[i+2].startswith('2 '):
            name, l1, l2 = lines[i], lines[i+1], lines[i+2]
            if cutoff:
                ep = _parse_tle_epoch(l1)
                if ep and ep < cutoff:
                    i += 3
                    continue
            entries.append((name.strip(), l1, l2))
            i += 3
        else:
            i += 1
    return entries

def _propagate_one(name, l1, l2, dt_utc, obs_lat, obs_lon):
    """
    Propagate one satellite at dt_utc. Returns dict or None (below horizon /
    error). Requires sgp4 library.
    """
    try:
        from sgp4.earth_gravity import wgs84
        from sgp4.io import twoline2rv
    except ImportError:
        return None

    try:
        sat = twoline2rv(l1, l2, wgs84)
        dt  = dt_utc.replace(tzinfo=None)
        pos, vel = sat.propagate(dt.year, dt.month, dt.day,
                                 dt.hour, dt.minute,
                                 dt.second + dt.microsecond / 1e6)
        if sat.error or pos is None:
            return None

        # TEME (km) → ECEF (m)
        px, py, pz = teme_to_ecef(*pos, dt_utc)
        vx, vy, vz = (v * 1000 for v in vel)   # km/s → m/s (still in TEME;
        # for rate we only need magnitude, so rotation doesn't change it)

        obs_x, obs_y, obs_z = latlon_to_ecef(obs_lat, obs_lon, 100)
        dx, dy, dz = px - obs_x, py - obs_y, pz - obs_z
        e, n, u = ecef_to_enu(dx, dy, dz, obs_lat, obs_lon)
        if u <= 0:
            return None   # below horizon

        az, el = enu_to_azel(e, n, u)
        dist   = math.sqrt(dx*dx + dy*dy + dz*dz)
        speed  = math.sqrt(vx*vx + vy*vy + vz*vz)
        # Radial component (towards/away from observer) — not useful for angular rate
        radial = abs(dx*vx + dy*vy + dz*vz) / max(dist, 1)
        transverse = math.sqrt(max(0.0, speed*speed - radial*radial))
        ang_rate = math.degrees(transverse / dist) if dist > 0 else 0.0

        return {
            'name':              name,
            'az':                round(az, 2),
            'el':                round(el, 2),
            'angular_rate_deg_s': round(ang_rate, 4),
        }
    except Exception:
        return None

def match_satellites(tle_entries, dt_utc, obs_lat, obs_lon, target_rate,
                     rate_tol=RATE_MATCH_TOL):
    """
    Propagate all TLE entries at dt_utc, return list of matches sorted by rate
    difference. Skips entries that are below the horizon.
    """
    matches = []
    for name, l1, l2 in tle_entries:
        sat = _propagate_one(name, l1, l2, dt_utc, obs_lat, obs_lon)
        if sat is None:
            continue
        diff = abs(sat['angular_rate_deg_s'] - target_rate)
        if diff <= rate_tol:
            matches.append({**sat, 'rate_diff_deg_s': round(diff, 4)})
    matches.sort(key=lambda s: s['rate_diff_deg_s'])
    return matches


# ── Classifier ────────────────────────────────────────────────────────────────

def classify(analysis, ac_above_horizon, sat_matches):
    """
    Apply classification rules.
    Returns dict: classification, confidence, flagged, flag_reason,
    aircraft_match, satellite_match.
    """
    rate = analysis.get('angular_rate_deg_s', 0.0)
    dur  = analysis.get('duration_s', 0.0)

    out = {
        'classification': 'unknown',
        'confidence':     'none',
        'flagged':        False,
        'flag_reason':    None,
        'aircraft_match': None,
        'satellite_match': None,
    }

    # ── Meteor: very fast ────────────────────────────────────────────────────
    if rate >= METEOR_RATE_MIN:
        out.update(classification='meteor', confidence='high',
                   flagged=True, flag_reason='meteor — always review')
        return out

    # ── Stationary / noise ───────────────────────────────────────────────────
    if rate < SATELLITE_RATE_MIN:
        out.update(classification='noise', confidence='low',
                   flagged=True, flag_reason=f'nearly stationary ({rate:.3f} deg/s)')
        return out

    # ── Best aircraft match by angular rate ─────────────────────────────────
    best_ac = min(ac_above_horizon,
                  key=lambda a: abs(a.get('angular_rate_deg_s', 0) - rate),
                  default=None)
    ac_diff = (abs(best_ac['angular_rate_deg_s'] - rate)
               if best_ac else float('inf'))
    ac_match = ac_diff <= RATE_MATCH_TOL

    # ── Best satellite match (already filtered by rate tolerance) ────────────
    sat_match = bool(sat_matches)
    best_sat  = sat_matches[0] if sat_matches else None

    # ── Decide ──────────────────────────────────────────────────────────────
    if ac_match and not sat_match:
        out['classification'] = 'aircraft'
        out['confidence']     = 'confirmed'
        out['aircraft_match'] = {
            'callsign': best_ac['callsign'],
            'icao24':   best_ac['icao24'],
            'az':       best_ac['az'],
            'el':       best_ac['el'],
            'rate_diff_deg_s': round(ac_diff, 4),
        }

    elif sat_match and not ac_match:
        out['classification']  = 'satellite'
        out['confidence']      = 'confirmed'
        out['satellite_match'] = {
            'name':          best_sat['name'],
            'az':            best_sat['az'],
            'el':            best_sat['el'],
            'rate_diff_deg_s': best_sat['rate_diff_deg_s'],
        }

    elif ac_match and sat_match:
        # Both match — pick better one, flag ambiguity
        if ac_diff <= best_sat['rate_diff_deg_s']:
            out['classification'] = 'aircraft'
            out['aircraft_match'] = {'callsign': best_ac['callsign'],
                                     'icao24': best_ac['icao24'],
                                     'rate_diff_deg_s': round(ac_diff, 4)}
        else:
            out['classification'] = 'satellite'
            out['satellite_match'] = {'name': best_sat['name'],
                                      'rate_diff_deg_s': best_sat['rate_diff_deg_s']}
        out['confidence']  = 'uncertain'
        out['flagged']     = True
        out['flag_reason'] = 'ambiguous: matches both aircraft and satellite rates'

    else:
        # No match found — flag it
        out['flagged'] = True
        if rate > AIRCRAFT_RATE_MAX:
            out['classification'] = 'unidentified'
            out['flag_reason']    = (f'faster than any aircraft ({rate:.2f} deg/s), '
                                     'no TLE match')
        elif not ac_above_horizon:
            out['classification'] = 'unidentified'
            out['flag_reason']    = 'no FlightAware data — check API key or query limit'
        else:
            # Speed is plausible but nothing matches
            if rate > 3.0:
                out['classification'] = 'unidentified satellite?'
            else:
                out['classification'] = 'unidentified aircraft?'
            out['flag_reason'] = 'no FlightAware or TLE match at this angular rate'

    return out


# ── Main pass ─────────────────────────────────────────────────────────────────

def identify_folder(events_folder, config, calib, tle_path=None,
                    progress_cb=None, cancel_event=None):
    """
    Run identification on all events in events_folder.
    progress_cb(msg: str) is called with status updates if provided.
    Returns summary dict.
    """
    def log(msg):
        print(msg)
        if progress_cb:
            progress_cb(msg)

    lat    = config.get('latitude',  0.0)
    lon    = config.get('longitude', 0.0)
    fa_key = config.get('flightaware_api_key', '')

    # Discover event directories
    event_dirs = sorted([
        os.path.join(events_folder, d)
        for d in os.listdir(events_folder)
        if (os.path.isdir(os.path.join(events_folder, d))
            and os.path.exists(os.path.join(events_folder, d, 'metadata.json')))
    ])
    if not event_dirs:
        log('No events with metadata.json found.')
        return {}

    log(f'Found {len(event_dirs)} event(s) in {os.path.basename(events_folder)}')

    # Load TLE catalog once
    tle_entries = []
    try_sgp4 = False
    if tle_path and os.path.exists(tle_path):
        try:
            from sgp4.earth_gravity import wgs84  # noqa — just check availability
            try_sgp4 = True
        except ImportError:
            log('[TLE] sgp4 not installed — satellite matching disabled')
        if try_sgp4:
            tle_entries = load_tle_catalog(tle_path)
            log(f'[TLE] Loaded {len(tle_entries)} satellites '
                f'(epoch ≤ {TLE_MAX_EPOCH_DAYS} days old) from '
                f'{os.path.basename(tle_path)}')

    # Group events by source clip to batch timestamp + OpenSky queries
    by_clip = defaultdict(list)
    for ed in event_dirs:
        with open(os.path.join(ed, 'metadata.json')) as f:
            meta = json.load(f)
        by_clip[meta.get('source_clip', '')].append((ed, meta))

    totals = defaultdict(int)

    for clip_path, events in by_clip.items():
        log(f'\n── {os.path.basename(clip_path)} '
            f'({len(events)} event(s)) ──────────────')

        clip_start = get_clip_start_utc(clip_path)
        fps        = events[0][1].get('fps', 20)

        if clip_start:
            log(f'   Clip start (UTC): {clip_start.strftime("%Y-%m-%d %H:%M:%S")}')
        else:
            log('   WARNING: could not determine clip start time — '
                'OpenSky/TLE queries disabled for this clip')

        # OpenSky cache: keyed by 60-second bucket
        opensky_cache = {}

        for event_dir, meta in events:
            if cancel_event and cancel_event.is_set():
                log('Identification cancelled.')
                return dict(totals)
            event_time = None
            if clip_start:
                event_time = (clip_start
                              + datetime.timedelta(
                                  seconds=meta.get('start_frame', 0) / max(fps, 1)))

            analysis = analyze_event(meta, calib)

            # ── Aircraft query (FlightAware) ─────────────────────────────────
            ac_above_horizon = []
            if event_time and fa_key:
                bucket = int(event_time.timestamp() // 60)
                if bucket not in opensky_cache:
                    log(f'   [FlightAware] querying '
                        f'{event_time.strftime("%H:%M:%S UTC")}…')
                    raw = query_flightaware(lat, lon, event_time, fa_key)
                    opensky_cache[bucket] = raw
                    log(f'               {len(raw)} aircraft above horizon')
                ac_above_horizon = opensky_cache[bucket]

            # ── TLE / satellite matching ─────────────────────────────────────
            sat_matches = []
            if event_time and tle_entries:
                rate = analysis.get('angular_rate_deg_s', 0)
                if SATELLITE_RATE_MIN <= rate < METEOR_RATE_MIN:
                    sat_matches = match_satellites(
                        tle_entries, event_time, lat, lon, rate)

            # ── Classify ────────────────────────────────────────────────────
            result = classify(analysis, ac_above_horizon, sat_matches)
            result['event_time_utc'] = (event_time.isoformat()
                                         if event_time else None)
            result['analysis'] = analysis

            # Write identification.json
            with open(os.path.join(event_dir, 'identification.json'), 'w') as f:
                json.dump(result, f, indent=2)

            cls     = result['classification']
            conf    = result['confidence']
            flag    = ' ⚑' if result['flagged'] else ''
            rate_s  = f"{analysis['angular_rate_deg_s']:.2f} deg/s"
            dur_s   = f"{analysis['duration_s']:.1f}s"
            log(f'   {os.path.basename(event_dir)}: '
                f'{cls} ({conf})  {rate_s}  {dur_s}{flag}')
            if result.get('flag_reason'):
                log(f'     ↳ {result["flag_reason"]}')

            totals[cls.split()[0]] += 1
            if result['flagged']:
                totals['flagged'] += 1

    log('\n── Summary ' + '─' * 50)
    for k, v in sorted(totals.items()):
        log(f'  {k}: {v}')

    return dict(totals)


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_config():
    """Load shared_config.json — next to the exe when frozen, else next to this script."""
    if getattr(sys, 'frozen', False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).parent
    candidates = [
        base / 'shared_config.json',
        base.parent.parent / 'SkyEye' / 'shared_config.json',
    ]
    for p in candidates:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return {}

def _find_tle(config):
    """Return most recent TLE file from the configured save_dir."""
    tle_cfg_path = Path(__file__).parent.parent / 'Get_TLE' / 'tle_config.json'
    save_dir = None
    if tle_cfg_path.exists():
        with open(tle_cfg_path) as f:
            save_dir = json.load(f).get('save_dir', '')
    if not save_dir:
        save_dir = config.get('tle_dir', '')
    if not save_dir or not os.path.isdir(save_dir):
        return None
    files = sorted(
        [p for p in Path(save_dir).iterdir()
         if p.suffix.lower() in ('.txt', '.tle')],
        key=lambda p: p.stat().st_mtime)
    return str(files[-1]) if files else None

def _load_calib(config):
    """Load calibration YAML from cameras config or fall back to Dahua N53."""
    calib_path = config.get('calibration_yaml', '')
    if calib_path and os.path.exists(calib_path):
        try:
            return CameraCalibration.from_yaml(calib_path)
        except Exception:
            pass
    return CameraCalibration.dahua_n53()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Identify events in a Streaker detect run folder')
    parser.add_argument('folder', nargs='?',
                        help='Detect run folder (or parent containing event_* dirs)')
    parser.add_argument('--tle', metavar='FILE',
                        help='TLE catalog file (auto-detected if omitted)')
    parser.add_argument('--calib', metavar='YAML',
                        help='Camera calibration YAML (defaults to Dahua N53)')
    args = parser.parse_args()

    config = _load_config()

    calib = (_load_calib({'calibration_yaml': args.calib})
             if args.calib else _load_calib(config))

    tle_path = args.tle or _find_tle(config)
    if tle_path:
        print(f'[TLE] Using: {tle_path}')
    else:
        print('[TLE] No TLE file found — satellite matching disabled')

    folder = args.folder
    if not folder:
        if _TK_AVAILABLE:
            root = tk.Tk(); root.withdraw()
            folder = filedialog.askdirectory(
                title='Select detect run folder to identify')
            root.destroy()
        else:
            print('No folder specified. Pass folder as argument.')
            sys.exit(1)

    if not folder or not os.path.isdir(folder):
        print('Folder not found.')
        sys.exit(1)

    identify_folder(folder, config, calib, tle_path)


if __name__ == '__main__':
    main()
