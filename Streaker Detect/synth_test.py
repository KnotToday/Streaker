"""
synth_test.py — Synthetic streak injection test for the Streaker detection pipeline.

Injects known fake streaks into real dark-sky frames, runs detection, and reports
what was found vs. what was planted.  Use this to tune thresholds systematically.

Usage:
    python synth_test.py --source <mkv_or_frame_folder> [options]

Options:
    --source        MKV file or folder of frame_*.jpg / frame_*.png
    --mask          Optional mask PNG (same as used during real detection)
    --config        streaker_config.json to load params from (default: streaker_config.json)
    --output        Output folder for results (default: synth_test_results/)
    --count         Number of synthetic streaks to inject per run (default: 20)
    --brightness    Comma-separated list of peak pixel values to test (default: 30,50,80,120,200)
    --length        Comma-separated list of streak lengths in px to test (default: 20,40,80)
    --angles        Comma-separated list of angles in degrees to test (default: 30,60,120,150)
    --seed          Random seed for reproducible placement (default: 42)
"""

import argparse
import json
import os
import random
import subprocess
import sys
import math
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Import detection primitives from StreakerDetect in the same directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from StreakerDetect import process_frame, TrackManager, AdaptiveCloudDetector, make_thumbnail

DEFAULT_PARAMS = {
    'threshold':      40,
    'min_area':       120,
    'max_area':       1400,
    'min_aspect':     2.0,
    'min_bright':     0,
    'min_move':       0,
    'min_travel':     0,
    'cloud_thresh':   40.0,
    'cloud_ratio':    0.0,
    'cloud_min_bright': 0,
    'cloud_min_travel': 0,
    'scale':          1.0,
    'warmup':         50,
    'pre_buffer':     30,
    'post_buffer':    30,
}


def load_frames_from_folder(folder):
    paths = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.png')) and f.startswith('frame_')
    ])
    frames = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames.append(img)
    return frames


def load_frames_from_mkv(mkv_path, start_sec=0.0, duration_sec=0.0, max_frames=600, scale=0.5):
    """Load frames into a list (for synth injection).  Downscales during load to save memory."""
    cap = cv2.VideoCapture(mkv_path)
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    limit = int(duration_sec * fps) if duration_sec > 0 else max_frames
    frames = []
    while len(frames) < limit:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            h, w = gray.shape
            gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                              interpolation=cv2.INTER_AREA)
        frames.append(gray)
    cap.release()
    return frames


def draw_streak(frame, x0, y0, angle_deg, length, brightness, width=2):
    """Draw a bright anti-aliased line streak onto a copy of frame."""
    out = frame.copy()
    rad = math.radians(angle_deg)
    x1 = int(x0 + length * math.cos(rad))
    y1 = int(y0 + length * math.sin(rad))
    x0, y0, x1, y1 = (max(0, min(v, dim - 1))
                       for v, dim in zip([x0, y0, x1, y1],
                                         [out.shape[1], out.shape[0],
                                          out.shape[1], out.shape[0]]))
    cv2.line(out, (x0, y0), (x1, y1), int(brightness), width, cv2.LINE_AA)
    return out, (x0, y0, x1, y1)


def run_detection_on_frames(frames, mask, params):
    """Run MOG2 + TrackManager detection on a list of gray frames.
    Returns list of (frame_idx, bboxes) for every frame with detections."""
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=params['threshold'], detectShadows=False)
    tracker = TrackManager(max_frames=params.get('max_track', 10))
    cloud_detector = AdaptiveCloudDetector()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    scale  = params.get('scale', 1.0)
    warmup = params.get('warmup', 50)
    results = []
    for idx, gray in enumerate(frames):
        if scale != 1.0:
            h, w = gray.shape
            small = cv2.resize(gray, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = gray
        if idx < warmup:
            mog2.apply(small)   # train background, skip detection
            continue
        mask_small = mask
        if mask is not None and scale != 1.0:
            mh, mw = mask.shape
            mask_small = cv2.resize(mask, (int(mw * scale), int(mh * scale)),
                                    interpolation=cv2.INTER_NEAREST)
        count, bboxes, _, _, fg_mask = process_frame(
            small, mog2, tracker, mask_small, kernel, params, cloud_detector)
        if count > 0:
            results.append((idx, bboxes, fg_mask))
    return results


def bbox_contains_point(bbox, x, y, margin=10):
    bx, by, bw, bh = bbox
    return (bx - margin) <= x <= (bx + bw + margin) and \
           (by - margin) <= y <= (by + bh + margin)


def streak_detected(detections, inject_frame_idx, x0, y0, x1, y1, margin=15):
    """Check if any detection bbox overlaps the injected streak centroid."""
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    for (fidx, bboxes, *_) in detections:
        if abs(fidx - inject_frame_idx) > 3:
            continue
        for bbox in bboxes:
            if bbox_contains_point(bbox, cx, cy, margin):
                return True
    return False


def run_test(frames, mask, params, n_streaks, brightnesses, lengths, angles, seed, out_dir):
    rng = random.Random(seed)
    h, w = frames[0].shape[:2]
    warmup = params.get('warmup', 50)

    # Inject streaks into frames after warmup so MOG2 is trained on clean sky
    inject_start = warmup + 20
    if inject_start >= len(frames) - 10:
        inject_start = len(frames) // 3

    results = []
    report_frames = []  # (frame_bgr, label) for visual report

    for i in range(n_streaks):
        brightness = rng.choice(brightnesses)
        length = rng.choice(lengths)
        angle = rng.choice(angles)

        # Random position — keep streak fully inside frame with margin
        margin = length + 10
        fx = rng.randint(margin, max(margin + 1, w - margin))
        fy = rng.randint(margin, max(margin + 1, h - margin))
        inject_idx = rng.randint(inject_start, len(frames) - 5)

        # Build modified frame list: inject streak into one frame
        test_frames = list(frames)
        modified, (ax0, ay0, ax1, ay1) = draw_streak(
            frames[inject_idx], fx, fy, angle, length, brightness)
        test_frames[inject_idx] = modified

        detections = run_detection_on_frames(test_frames, mask, params)
        found = streak_detected(detections, inject_idx, ax0, ay0, ax1, ay1)

        results.append({
            'streak_id':    i,
            'brightness':   brightness,
            'length':       length,
            'angle':        angle,
            'inject_frame': inject_idx,
            'x0': ax0, 'y0': ay0, 'x1': ax1, 'y1': ay1,
            'detected':     found,
            'n_detections': len(detections),
        })

        # Build event folder with thumbnail — same format as the detector
        pre_buf  = params.get('pre_buffer', 30)
        post_buf = params.get('post_buffer', 30)
        win_start = max(0, inject_idx - pre_buf)
        win_end   = min(len(test_frames) - 1, inject_idx + post_buf)
        det_lookup = {fidx: (bboxes, fg) for fidx, bboxes, fg in detections
                      if win_start <= fidx <= win_end}
        pending = []
        for fidx in range(win_start, win_end + 1):
            bboxes_f, fg_f = det_lookup.get(fidx, ([], None))
            pending.append((fidx, test_frames[fidx], len(bboxes_f), bboxes_f, fg_f))
        label = "found" if found else "missed"
        # Prefix event folder name with result so they sort visually
        src_tag = f"{label}_streak{i:03d}"
        event_dir_path = _save_event_folder(pending, out_dir, src_tag, params,
                                            lambda m: None)
        if event_dir_path:
            # Append synth info to metadata.json
            meta_path = os.path.join(event_dir_path, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as mf:
                    meta = json.load(mf)
                meta['synth_streak'] = {
                    'brightness': brightness, 'length': length,
                    'angle': angle, 'inject_frame': inject_idx,
                    'found': found,
                    'x0': ax0, 'y0': ay0, 'x1': ax1, 'y1': ay1,
                }
                with open(meta_path, 'w') as mf:
                    json.dump(meta, mf, indent=2)

    return results


def print_summary(results, params, out_dir):
    total = len(results)
    found = sum(1 for r in results if r['detected'])
    print(f"\n{'='*60}")
    print(f"SYNTH TEST RESULTS  —  {found}/{total} detected ({100*found/total:.1f}%)")
    print(f"{'='*60}")
    print(f"Params: thr={params['threshold']}  area={params['min_area']}-{params['max_area']}"
          f"  asp={params['min_aspect']}  bright={params['min_bright']}"
          f"  move={params['min_move']}  scale={params['scale']}")
    print()

    # Break down by brightness
    by_bright = {}
    for r in results:
        b = r['brightness']
        by_bright.setdefault(b, []).append(r['detected'])
    print("By brightness:")
    for b in sorted(by_bright):
        vals = by_bright[b]
        n = len(vals)
        d = sum(vals)
        print(f"  {b:4d}px : {d}/{n}  ({100*d/n:.0f}%)")

    # Break down by length
    by_len = {}
    for r in results:
        l = r['length']
        by_len.setdefault(l, []).append(r['detected'])
    print("By length:")
    for l in sorted(by_len):
        vals = by_len[l]
        n = len(vals)
        d = sum(vals)
        print(f"  {l:4d}px : {d}/{n}  ({100*d/n:.0f}%)")

    print(f"\nVisual evidence frames saved to: {out_dir}")

    report = {
        'summary': {'total': total, 'detected': found, 'pct': round(100*found/total, 1)},
        'params': params,
        'by_brightness': {str(k): {'detected': sum(v), 'total': len(v)} for k, v in by_bright.items()},
        'by_length':     {str(k): {'detected': sum(v), 'total': len(v)} for k, v in by_len.items()},
        'streaks': results,
    }
    report_path = os.path.join(out_dir, 'synth_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Full JSON report: {report_path}")


import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from PIL import Image, ImageTk


def _parse_time_gui(s):
    """Accept HH:MM:SS, MM:SS, or plain seconds. Returns float."""
    s = s.strip()
    if not s or s == '0':
        return 0.0
    parts = s.split(':')
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(s)
    except ValueError:
        raise ValueError(f"Invalid time '{s}' — use HH:MM:SS, MM:SS, or seconds")


def _save_event_folder(pending, out_dir, source, params, log_fn):
    """Save one event in StreakerDetect's standard format: frame_*.jpg + _thumbnail.jpg + metadata.json"""
    from collections import deque as _deque
    if not pending:
        return None
    first_frame = pending[0][0]
    fps = params.get('fps', 20)
    total_secs = int(first_frame / fps)
    mm, ss = total_secs // 60, total_secs % 60
    src_name = os.path.splitext(os.path.basename(source))[0]
    event_dir = os.path.join(out_dir, f"event_{src_name}_{mm:02d}m{ss:02d}s_{first_frame:06d}")
    os.makedirs(event_dir, exist_ok=True)

    detect_scale = params.get('scale', 1.0)
    gray_frames, det_meta, all_detections = [], [], []
    for (fidx, gray, count, bboxes, fg_mask) in pending:
        fpath = os.path.join(event_dir, f"frame_{fidx:06d}.jpg")
        cv2.imwrite(fpath, gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
        gray_frames.append(gray if len(gray.shape) == 2
                           else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY))
        if bboxes:
            centroids = [[x + w//2, y + h//2] for (x, y, w, h) in bboxes]
            det_meta.append({'frame': fidx, 'centroids': centroids,
                             'bboxes': [list(b) for b in bboxes], 'count': count})
            all_detections.append((bboxes, fg_mask))

    thumb = make_thumbnail(gray_frames, all_detections, detect_scale, params=params)
    if thumb is not None:
        cv2.imwrite(os.path.join(event_dir, '_thumbnail.jpg'), thumb)

    meta = {
        'source_clip':  source,
        'start_frame':  pending[0][0],
        'end_frame':    pending[-1][0],
        'fps':          fps,
        'detect_scale': detect_scale,
        'detections':   det_meta,
        'params':       dict(params),
    }
    with open(os.path.join(event_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    log_fn(f"  Saved: {os.path.basename(event_dir)}  ({len(pending)} frames, {len(det_meta)} detection frames)")
    return event_dir


def run_detect_only(source, mask_path, config_path, out_dir,
                    log_fn, done_fn, start_sec=0.0, duration_sec=0.0):
    """Stream frames one at a time with pre/post buffers — saves events in StreakerDetect format."""
    from collections import deque
    try:
        params = dict(DEFAULT_PARAMS)
        if os.path.exists(config_path):
            with open(config_path) as f:
                params.update(json.load(f))
            log_fn(f"Loaded params from {config_path}")
        else:
            log_fn("Config not found, using defaults")

        scale      = params.get('scale', 1.0)
        warmup     = params.get('warmup', 50)
        pre_buf    = params.get('pre_buffer', 30)
        post_buf   = params.get('post_buffer', 30)
        kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mog2       = cv2.createBackgroundSubtractorMOG2(
                         history=500, varThreshold=params['threshold'], detectShadows=False)
        tracker    = TrackManager(max_frames=params.get('max_track', 10))
        cloud_det  = AdaptiveCloudDetector()

        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            log_fn(f"Loaded mask: {mask_path}")

        is_mkv = os.path.isfile(source) and source.lower().endswith('.mkv')
        if is_mkv:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                log_fn(f"ERROR: could not open {source}"); done_fn(None); return
            if start_sec > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            frame_limit = int(duration_sec * fps) if duration_sec > 0 else 0
            range_str = f"start={start_sec:.1f}s" + (f"  dur={duration_sec:.1f}s" if duration_sec > 0 else "  dur=all")
            log_fn(f"Streaming: {source}  ({range_str})")
            frame_paths = None
        elif os.path.isdir(source):
            frame_paths = sorted([
                os.path.join(source, f) for f in os.listdir(source)
                if f.lower().endswith(('.jpg', '.png')) and f.startswith('frame_')
            ])
            log_fn(f"Folder: {source}  ({len(frame_paths)} frames)")
            cap = None; fps = 20.0; frame_limit = 0
        else:
            log_fn("ERROR: source must be MKV or frame folder"); done_fn(None); return

        params['fps'] = fps

        # Create timestamped run subfolder
        from datetime import datetime as _dt
        ts = _dt.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(out_dir, f"test_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        run_info = {
            'type':      'test',
            'source':    os.path.basename(source),
            'timestamp': _dt.now().isoformat(timespec='seconds'),
            'start_sec': start_sec,
            'duration_sec': duration_sec,
            'params':    dict(params),
        }
        with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
            json.dump(run_info, f, indent=2)
        log_fn(f"Run folder: {run_dir}\n")
        log_fn("Running detection (streaming)...\n")

        pre_buffer = deque(maxlen=pre_buf)  # (fidx, gray, count, bboxes, fg_mask)
        pending    = []
        post_cd    = 0
        frame_idx  = 0
        mask_small = None
        saved_dirs = []

        def read_gray():
            if is_mkv:
                ret, frame = cap.read()
                if not ret:
                    return None
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                if frame_idx >= len(frame_paths):
                    return None
                g = cv2.imread(frame_paths[frame_idx], cv2.IMREAD_GRAYSCALE)
                return g

        while True:
            if frame_limit and frame_idx >= frame_limit:
                break
            gray = read_gray()
            if gray is None:
                break

            if scale != 1.0:
                h, w = gray.shape
                small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                                   interpolation=cv2.INTER_AREA)
            else:
                small = gray

            if mask_small is None and mask is not None:
                mh, mw = mask.shape
                mask_small = cv2.resize(mask, (max(1, int(mw * scale)), max(1, int(mh * scale))),
                                        interpolation=cv2.INTER_NEAREST) if scale != 1.0 else mask

            if frame_idx >= warmup:
                count, bboxes, _, fg_mask_out, fg_mask = process_frame(
                    small, mog2, tracker, mask_small, kernel, params, cloud_det)
                entry = (frame_idx, gray, count, bboxes, fg_mask)

                if count > 0:
                    if post_cd == 0:
                        pending.extend(list(pre_buffer))
                        pre_buffer.clear()
                    pending.append(entry)
                    post_cd = post_buf
                elif post_cd > 0:
                    pending.append(entry)
                    post_cd -= 1
                    if post_cd == 0:
                        d = _save_event_folder(pending, run_dir, source, params, log_fn)
                        if d:
                            saved_dirs.append(d)
                        pending = []
                else:
                    pre_buffer.append(entry)
            else:
                mog2.apply(small)

            if frame_idx % 500 == 0 and frame_idx > 0:
                log_fn(f"  frame {frame_idx}  events saved: {len(saved_dirs)}")

            frame_idx += 1

        # Flush trailing pending
        if pending:
            d = _save_event_folder(pending, run_dir, source, params, log_fn)
            if d:
                saved_dirs.append(d)

        if is_mkv:
            cap.release()

        log_fn(f"\nProcessed {frame_idx} frames.  {len(saved_dirs)} event(s) saved to:\n  {run_dir}")
        done_fn({'events': len(saved_dirs), 'out_dir': run_dir})

    except Exception as e:
        import traceback
        log_fn(f"ERROR: {e}\n{traceback.format_exc()}")
        done_fn(None)


def run_core(source, mask_path, config_path, out_dir, count,
             brightnesses, lengths, angles, seed, log_fn, done_fn,
             start_sec=0.0, duration_sec=0.0):
    try:
        params = dict(DEFAULT_PARAMS)
        if os.path.exists(config_path):
            with open(config_path) as f:
                params.update(json.load(f))
            log_fn(f"Loaded params from {config_path}")
        else:
            log_fn(f"Config not found, using defaults")

        range_info = ""
        if start_sec > 0 or duration_sec > 0:
            range_info = f"  (start={start_sec:.1f}s" + (f"  dur={duration_sec:.1f}s)" if duration_sec > 0 else ")")
        # Load frames at detection resolution (params.scale) so detection
        # runs at scale=1.0 — avoids double-scaling and threshold mismatch
        load_scale = params.get('scale', 0.5)
        log_fn(f"Loading frames from: {source}{range_info}")
        if os.path.isfile(source) and source.lower().endswith('.mkv'):
            frames = load_frames_from_mkv(source, start_sec=start_sec,
                                          duration_sec=duration_sec, scale=load_scale)
        elif os.path.isdir(source):
            frames = load_frames_from_folder(source)
        else:
            log_fn(f"ERROR: source must be MKV or frame folder")
            done_fn(None)
            return

        if len(frames) < 100:
            log_fn(f"ERROR: need at least 100 frames, got {len(frames)}")
            done_fn(None)
            return
        log_fn(f"Loaded {len(frames)} frames  ({frames[0].shape[1]}x{frames[0].shape[0]})")

        # Frames are already at detection resolution — run detection without
        # re-scaling, but adjust area thresholds to match what DetectionWorker does
        params['scale'] = 1.0
        area_scale = load_scale * load_scale
        params['min_area'] = max(1, int(params['min_area'] * area_scale))
        params['max_area'] = max(1, int(params['max_area'] * area_scale))
        log_fn(f"Detection scale {load_scale:.2f} → area thresholds {params['min_area']}–{params['max_area']} px²")

        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            log_fn(f"Loaded mask: {mask_path}")

        # Create timestamped run subfolder
        from datetime import datetime as _dt
        ts = _dt.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(out_dir, f"synth_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        run_info = {
            'type':        'synth',
            'source':      os.path.basename(source),
            'timestamp':   _dt.now().isoformat(timespec='seconds'),
            'start_sec':   start_sec,
            'duration_sec': duration_sec,
            'count':       count,
            'brightnesses': brightnesses,
            'lengths':     lengths,
            'angles':      angles,
            'seed':        seed,
            'params':      dict(params),
        }
        with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
            json.dump(run_info, f, indent=2)
        log_fn(f"Run folder: {run_dir}\n")
        log_fn(f"Injecting {count} streaks  brightness={brightnesses}  length={lengths}")

        results = run_test(frames, mask, params, count,
                           brightnesses, lengths, angles, seed, run_dir)

        total = len(results)
        found = sum(1 for r in results if r['detected'])
        log_fn(f"\n{'='*50}")
        log_fn(f"RESULTS: {found}/{total} detected ({100*found/total:.1f}%)")
        log_fn(f"{'='*50}")

        by_bright = {}
        for r in results:
            by_bright.setdefault(r['brightness'], []).append(r['detected'])
        log_fn("By brightness:")
        for b in sorted(by_bright):
            v = by_bright[b]; d = sum(v); n = len(v)
            log_fn(f"  {b:4d}px : {d}/{n}  ({100*d/n:.0f}%)")

        by_len = {}
        for r in results:
            by_len.setdefault(r['length'], []).append(r['detected'])
        log_fn("By length:")
        for l in sorted(by_len):
            v = by_len[l]; d = sum(v); n = len(v)
            log_fn(f"  {l:4d}px : {d}/{n}  ({100*d/n:.0f}%)")

        report = {
            'summary': {'total': total, 'detected': found, 'pct': round(100*found/total, 1)},
            'params': params,
            'by_brightness': {str(k): {'detected': sum(v), 'total': len(v)} for k, v in by_bright.items()},
            'by_length':     {str(k): {'detected': sum(v), 'total': len(v)} for k, v in by_len.items()},
            'streaks': results,
        }
        report_path = os.path.join(run_dir, 'synth_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        log_fn(f"\nReport: {report_path}")
        log_fn(f"Frames: {run_dir}")
        done_fn(report)

    except Exception as e:
        import traceback
        log_fn(f"ERROR: {e}\n{traceback.format_exc()}")
        done_fn(None)


class ClipScrubber:
    """Embeddable video scrubber with Mark Start / Mark End buttons."""
    CW, CH = 480, 270   # canvas size

    def __init__(self, parent, on_range_set, bg='#1a1a1a'):
        self.on_range_set = on_range_set   # callback(start_sec, duration_sec)
        self.cap          = None
        self.total_frames = 0
        self.fps          = 20.0
        self.mark_in      = 0
        self.mark_out     = 0
        self._after_id    = None
        self._photo       = None

        self.frame = tk.Frame(parent, bg=bg)

        self.canvas = tk.Canvas(self.frame, width=self.CW, height=self.CH,
                                bg='#000000', highlightthickness=0)
        self.canvas.pack()

        self.slider_var = tk.IntVar(value=0)
        self.slider = tk.Scale(self.frame, from_=0, to=1000,
                               orient='horizontal', variable=self.slider_var,
                               command=self._on_scrub,
                               bg=bg, fg='#888888', troughcolor='#333333',
                               activebackground='#555555',
                               highlightthickness=0, showvalue=False,
                               length=self.CW)
        self.slider.pack(fill='x')

        ctrl = tk.Frame(self.frame, bg=bg)
        ctrl.pack(fill='x', pady=2)

        tk.Button(ctrl, text='◀ Mark Start', command=self._mark_start,
                  bg='#1a3a1a', fg='#88ff88', relief='flat',
                  font=('Arial', 9), padx=8).pack(side='left', padx=6)

        self.time_lbl = tk.Label(ctrl, text='00:00:00', bg=bg, fg='#aaaaaa',
                                 font=('Courier', 10))
        self.time_lbl.pack(side='left', expand=True)

        tk.Button(ctrl, text='Mark End ▶', command=self._mark_end,
                  bg='#3a1a1a', fg='#ff8888', relief='flat',
                  font=('Arial', 9), padx=8).pack(side='right', padx=6)

        self.range_lbl = tk.Label(self.frame,
                                  text='Scrub to a position then click Mark Start / Mark End',
                                  bg=bg, fg='#555555', font=('Arial', 8))
        self.range_lbl.pack(pady=(0, 4))

    def load(self, path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.cap = None
            return False
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
        self.mark_in  = 0
        self.mark_out = self.total_frames
        self.slider.config(to=max(1, self.total_frames - 1))
        self.slider_var.set(0)
        self._seek(0)
        return True

    def _seek(self, frame_no):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_MSEC, (frame_no / self.fps) * 1000)
        ret, frame = self.cap.read()
        if not ret:
            return
        h, w = frame.shape[:2]
        scale = min(self.CW / w, self.CH / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete('all')
        self.canvas.create_image(self.CW // 2, self.CH // 2,
                                 anchor='center', image=self._photo)
        # Draw in/out markers on the slider track
        self._draw_range_overlay()
        self.time_lbl.config(text=self._fmt(frame_no / self.fps))

    def _draw_range_overlay(self):
        """Tint the canvas border to show marked region."""
        if self.mark_out <= self.mark_in:
            return
        in_x  = int(self.mark_in  / max(self.total_frames, 1) * self.CW)
        out_x = int(self.mark_out / max(self.total_frames, 1) * self.CW)
        # Green bar at bottom of canvas showing selected range
        self.canvas.delete('range_bar')
        self.canvas.create_rectangle(in_x, self.CH - 6, out_x, self.CH,
                                     fill='#44aa44', outline='', tags='range_bar')

    def _on_scrub(self, val):
        if self._after_id:
            self.slider.after_cancel(self._after_id)
        self._after_id = self.slider.after(60, lambda: self._seek(int(val)))

    def _mark_start(self):
        self.mark_in = self.slider_var.get()
        self._update_range()

    def _mark_end(self):
        self.mark_out = self.slider_var.get()
        self._update_range()

    def _update_range(self):
        in_sec  = self.mark_in  / self.fps
        out_sec = self.mark_out / self.fps
        dur     = max(0.0, out_sec - in_sec)
        if dur > 0:
            self.range_lbl.config(
                text=f'Start: {self._fmt(in_sec)}  →  End: {self._fmt(out_sec)}'
                     f'  ({dur:.1f}s)',
                fg='#88ff88')
            if self.on_range_set:
                self.on_range_set(in_sec, dur)
        else:
            self.range_lbl.config(text='End must be after start', fg='#ff6666')
        self._draw_range_overlay()

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    @staticmethod
    def _fmt(secs):
        s = int(secs)
        return f'{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}'


class SynthTestGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Synth Streak Test")
        self.root.configure(bg='#1a1a1a')
        self.root.resizable(True, True)

        BG, FG, ENTRY_BG = '#1a1a1a', '#dddddd', '#2a2a2a'
        lbl_kw  = dict(bg=BG, fg=FG, font=('Arial', 9), anchor='w')
        entry_kw = dict(bg=ENTRY_BG, fg=FG, insertbackground=FG, font=('Arial', 9), relief='flat')

        def row(parent, label, var, browse_fn=None, row_i=None):
            tk.Label(parent, text=label, **lbl_kw).grid(row=row_i, column=0, sticky='w', padx=6, pady=2)
            e = tk.Entry(parent, textvariable=var, width=55, **entry_kw)
            e.grid(row=row_i, column=1, sticky='ew', padx=4, pady=2)
            if browse_fn:
                tk.Button(parent, text='...', command=browse_fn, bg='#333', fg=FG,
                          relief='flat', padx=4).grid(row=row_i, column=2, padx=2)

        top = tk.Frame(self.root, bg=BG)
        top.pack(fill='x', padx=8, pady=6)
        top.columnconfigure(1, weight=1)

        self.v_source  = tk.StringVar()
        self.v_mask    = tk.StringVar()
        self.v_config  = tk.StringVar(value=os.path.join(os.path.dirname(__file__), 'streaker_config.json'))
        self.v_output  = tk.StringVar(value=os.path.join(os.path.dirname(__file__), 'synth_test_results'))
        self.v_start   = tk.StringVar(value='0')
        self.v_dur     = tk.StringVar(value='0')
        self.v_count   = tk.StringVar(value='20')
        self.v_bright  = tk.StringVar(value='30,50,80,120,200')
        self.v_length  = tk.StringVar(value='20,40,80')
        self.v_angles  = tk.StringVar(value='30,60,120,150')
        self.v_seed    = tk.StringVar(value='42')

        row(top, 'Source (MKV or folder)',        self.v_source, self._pick_source, 0)
        row(top, 'Mask PNG (optional)',            self.v_mask,   self._pick_mask,   1)
        row(top, 'Config JSON',                    self.v_config, self._pick_config, 2)
        row(top, 'Output folder',                  self.v_output, self._pick_output, 3)
        row(top, 'Clip start (s or HH:MM:SS)',     self.v_start,  None,              4)
        row(top, 'Clip duration (s, 0 = all)',     self.v_dur,    None,              5)

        # Detect-only toggle
        self.v_detect_only = tk.BooleanVar(value=False)
        det_chk = tk.Checkbutton(top, text='Detect only  (no injection — just run detector on real frames)',
                                 variable=self.v_detect_only, command=self._toggle_detect_only,
                                 bg=BG, fg='#aaaaaa', selectcolor='#2a2a2a',
                                 activebackground=BG, font=('Arial', 9))
        det_chk.grid(row=6, column=0, columnspan=3, sticky='w', padx=6, pady=4)

        row(top, 'Streak count',                   self.v_count,  None,              7)
        row(top, 'Brightness values',              self.v_bright, None,              8)
        row(top, 'Length values (px)',             self.v_length, None,              9)
        row(top, 'Angles (deg)',                   self.v_angles, None,              10)
        row(top, 'Random seed',                    self.v_seed,   None,              11)

        self._synth_rows = []  # populated after widgets exist

        # ── Clip Preview scrubber ────────────────────────────────────────────
        self._scrubber_toggle = tk.Button(
            self.root, text='▶ Clip Preview',
            command=self._toggle_scrubber,
            bg='#282828', fg='#777777', font=('Arial', 8),
            relief='flat', padx=8, pady=2, anchor='w')
        self._scrubber_toggle.pack(fill='x', padx=8)

        self._scrubber_section = tk.Frame(self.root, bg=BG)
        self._scrubber = ClipScrubber(self._scrubber_section,
                                      on_range_set=self._on_scrubber_range, bg=BG)
        self._scrubber.frame.pack(fill='x')
        # section hidden by default; toggle shows it

        self._btn_row = tk.Frame(self.root, bg=BG)
        btn_row = self._btn_row
        btn_row.pack(fill='x', padx=8, pady=4)
        self.run_btn = tk.Button(btn_row, text='Run Test', command=self._run,
                                 bg='#2a6a2a', fg='white', font=('Arial', 10, 'bold'),
                                 relief='flat', padx=16, pady=4)
        self.run_btn.pack(side='left')
        self.status_lbl = tk.Label(btn_row, text='', bg=BG, fg='#aaaaaa', font=('Arial', 9))
        self.status_lbl.pack(side='left', padx=10)

        self.log = scrolledtext.ScrolledText(self.root, height=18, bg='#0d0d0d', fg='#cccccc',
                                             font=('Courier', 8), relief='flat')
        self.log.pack(fill='both', expand=True, padx=8, pady=(0, 8))

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    def _pick_source(self):
        p = filedialog.askopenfilename(filetypes=[('MKV', '*.mkv'), ('All', '*.*')], title='Select source MKV')
        if not p:
            p = filedialog.askdirectory(title='Or select frame folder')
        if p:
            self.v_source.set(p)
            if p.lower().endswith('.mkv') and os.path.isfile(p):
                self._scrubber.load(p)
                if not self._scrubber_section.winfo_ismapped():
                    self._toggle_scrubber()

    def _pick_mask(self):
        p = filedialog.askopenfilename(filetypes=[('PNG', '*.png'), ('All', '*.*')], title='Select mask PNG')
        if p:
            self.v_mask.set(p)

    def _pick_config(self):
        p = filedialog.askopenfilename(filetypes=[('JSON', '*.json')], title='Select config JSON')
        if p:
            self.v_config.set(p)

    def _pick_output(self):
        p = filedialog.askdirectory(title='Select output folder')
        if p:
            self.v_output.set(p)

    def _toggle_scrubber(self):
        if self._scrubber_section.winfo_ismapped():
            self._scrubber_section.pack_forget()
            self._scrubber_toggle.config(text='▶ Clip Preview')
        else:
            self._scrubber_section.pack(fill='x', padx=8, pady=4,
                                        after=self._scrubber_toggle)
            self._scrubber_toggle.config(text='▼ Clip Preview')

    def _on_scrubber_range(self, start_sec, duration_sec):
        h = int(start_sec) // 3600
        m = (int(start_sec) % 3600) // 60
        s = int(start_sec) % 60
        self.v_start.set(f'{h:02d}:{m:02d}:{s:02d}' if h > 0 else f'{m:02d}:{s:02d}')
        self.v_dur.set(f'{duration_sec:.1f}')

    def _on_close(self):
        self._scrubber.close()
        self.root.destroy()

    def _toggle_detect_only(self):
        self.run_btn.config(
            text='Run Detection' if self.v_detect_only.get() else 'Run Test')

    def _open_in_viewer(self, out_dir):
        subprocess.Popen([sys.executable,
                          os.path.join(os.path.dirname(__file__), 'StreakerDetect.py'),
                          '--load', out_dir])

    def _log(self, msg):
        self.root.after(0, lambda: (
            self.log.insert('end', msg + '\n'),
            self.log.see('end')
        ))

    def _run(self):
        src = self.v_source.get().strip()
        if not src:
            messagebox.showerror('Missing', 'Please select a source MKV or frame folder.')
            return
        try:
            brightnesses = [int(x) for x in self.v_bright.get().split(',')]
            lengths      = [int(x) for x in self.v_length.get().split(',')]
            angles       = [int(x) for x in self.v_angles.get().split(',')]
            count        = int(self.v_count.get())
            seed         = int(self.v_seed.get())
            start_sec    = _parse_time_gui(self.v_start.get())
            duration_sec = _parse_time_gui(self.v_dur.get())
        except ValueError as e:
            messagebox.showerror('Invalid input', str(e))
            return

        self.run_btn.config(state='disabled')
        self.status_lbl.config(text='Running...')
        self.log.delete('1.0', 'end')

        if self.v_detect_only.get():
            def done(report):
                def _update():
                    self.run_btn.config(state='normal')
                    if report and report.get('events', 0) > 0:
                        self.status_lbl.config(text=f"{report['events']} event(s) found")
                        if not hasattr(self, '_viewer_btn'):
                            self._viewer_btn = tk.Button(
                                self._btn_row, text='Open in Viewer',
                                command=lambda: self._open_in_viewer(report['out_dir']),
                                bg='#2a4a6a', fg='white', font=('Arial', 10, 'bold'),
                                relief='flat', padx=12, pady=4)
                            self._viewer_btn.pack(side='left', padx=8)
                        else:
                            self._viewer_btn.config(
                                command=lambda: self._open_in_viewer(report['out_dir']))
                    else:
                        self.status_lbl.config(text='No events found' if report else 'Failed — see log')
                self.root.after(0, _update)

            threading.Thread(
                target=run_detect_only,
                args=(src, self.v_mask.get().strip(), self.v_config.get().strip(),
                      self.v_output.get().strip(), self._log, done),
                kwargs={'start_sec': start_sec, 'duration_sec': duration_sec},
                daemon=True
            ).start()
        else:
            def done(report):
                self.root.after(0, lambda: (
                    self.run_btn.config(state='normal'),
                    self.status_lbl.config(
                        text=f"Done — {report['summary']['detected']}/{report['summary']['total']} detected"
                        if report else 'Failed — see log')
                ))
            threading.Thread(
                target=run_core,
                args=(src, self.v_mask.get().strip(), self.v_config.get().strip(),
                      self.v_output.get().strip(), count, brightnesses, lengths,
                      angles, seed, self._log, done),
                kwargs={'start_sec': start_sec, 'duration_sec': duration_sec},
                daemon=True
            ).start()

    def run(self):
        self.root.mainloop()


def main():
    # CLI mode when --source is provided, GUI mode when double-clicked
    if '--source' in sys.argv:
        parser = argparse.ArgumentParser(description='Synthetic streak injection test')
        parser.add_argument('--source',     required=True)
        parser.add_argument('--mask',       default=None)
        parser.add_argument('--config',     default=os.path.join(os.path.dirname(__file__), 'streaker_config.json'))
        parser.add_argument('--output',     default=os.path.join(os.path.dirname(__file__), 'synth_test_results'))
        parser.add_argument('--count',      type=int, default=20)
        parser.add_argument('--brightness', default='30,50,80,120,200')
        parser.add_argument('--length',     default='20,40,80')
        parser.add_argument('--angles',     default='30,60,120,150')
        parser.add_argument('--seed',       type=int, default=42)
        args = parser.parse_args()

        brightnesses = [int(x) for x in args.brightness.split(',')]
        lengths      = [int(x) for x in args.length.split(',')]
        angles       = [int(x) for x in args.angles.split(',')]

        run_core(args.source, args.mask, args.config, args.output, args.count,
                 brightnesses, lengths, angles, args.seed,
                 log_fn=print, done_fn=lambda r: None)
    else:
        gui = SynthTestGUI()
        # Pre-fill fields when launched from StreakerDetect
        for flag, var in [('--presource', gui.v_source), ('--preoutput', gui.v_output)]:
            if flag in sys.argv:
                idx = sys.argv.index(flag)
                if idx + 1 < len(sys.argv):
                    var.set(sys.argv[idx + 1])
        src = gui.v_source.get()
        if src and src.lower().endswith('.mkv') and os.path.isfile(src):
            gui.root.after(150, lambda: (gui._scrubber.load(src),
                                         gui._toggle_scrubber() if not gui._scrubber_section.winfo_ismapped() else None))
        gui.run()


if __name__ == '__main__':
    main()
