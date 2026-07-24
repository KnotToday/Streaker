# ------------------------------------------------------------------------------
# Script Name:     StreakerDetect.py
# Description:     Unified detection GUI — file selection, parameter tuning,
#                  live preview, event thumbnail gallery, and clip playback.
# ------------------------------------------------------------------------------

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import subprocess
import json
import re
from collections import deque
from datetime import datetime
import time
import gc

try:
    from StreakerPlayer import launch_player
    _PLAYER_AVAILABLE = True
except ImportError:
    _PLAYER_AVAILABLE = False

if getattr(sys, 'frozen', False):
    _BASE_DIR = os.path.dirname(sys.executable)
else:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH   = os.path.join(_BASE_DIR, 'streaker_config.json')
SHARED_CONFIG_PATH = os.path.join(_BASE_DIR, 'shared_config.json')
CAMERAS_PATH  = os.path.join(_BASE_DIR, 'cameras.json')

from platform_utils import (FFMPEG_PATH, HWACCEL_ARGS, NO_WINDOW,
                            play_completion_sound, launch_companion)

# ------------------------------------------------------------------------------
# Detection Parameters (defaults — all tunable in GUI)
# ------------------------------------------------------------------------------

DEFAULT_MOG2_HISTORY       = 250
DEFAULT_MOG2_THRESHOLD     = 40
DEFAULT_MIN_CONTOUR_AREA   = 100
DEFAULT_MAX_CONTOUR_AREA   = 5000
DEFAULT_MIN_ASPECT_RATIO   = 2.0
DEFAULT_MAX_TRACK_FRAMES   = 5
DEFAULT_MAX_MATCH_DIST     = 0
DEFAULT_PRE_BUFFER         = 30
DEFAULT_POST_BUFFER        = 30
DEFAULT_WARMUP_FRAMES      = 200
DEFAULT_CLOUD_THRESH       = 80
DEFAULT_CLOUD_RATIO        = 0.15

VERSION      = "1.2.0"
GITHUB_REPO  = "KnotToday/Streaker"

THUMB_W  = 320
THUMB_H  = 200   # image portion height
INFO_H   = 38    # params strip below image (3 lines × ~11 px + padding)
DETECTION_COLORS = [
    (0, 255, 0),    # green
    (0, 165, 255),  # orange
    (255, 0, 255),  # magenta
    (0, 255, 255),  # cyan
    (255, 128, 0),  # blue
]
PREVIEW_EVERY_N = 5   # update live preview every N frames

# ------------------------------------------------------------------------------
# Detection Engine
# ------------------------------------------------------------------------------

class TrackManager:
    def __init__(self, max_frames, iou_threshold=0.3, max_match_dist=0):
        self.max_frames = max_frames
        self.iou_threshold = iou_threshold
        self.max_match_dist = max_match_dist  # px; 0 = disabled
        self.tracks = []

    def _iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(ax, bx); iy = max(ay, by)
        iw = min(ax+aw, bx+bw) - ix
        ih = min(ay+ah, by+bh) - iy
        if iw <= 0 or ih <= 0: return 0.0
        inter = iw * ih
        union = aw*ah + bw*bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _center(bbox):
        x, y, w, h = bbox
        return x + w / 2, y + h / 2

    @staticmethod
    def _dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def update(self, detections, min_displacement=0):
        matched = set()
        for track in self.tracks:
            best_iou, best_i = 0, -1
            for i, det in enumerate(detections):
                if i in matched: continue
                iou = self._iou(track['bbox'], det)
                if iou > best_iou:
                    best_iou, best_i = iou, i

            if best_iou < self.iou_threshold and self.max_match_dist > 0:
                # IoU failed — fall back to nearest centroid within max_match_dist
                best_dist, best_i = float('inf'), -1
                tc = self._center(track['bbox'])
                for i, det in enumerate(detections):
                    if i in matched: continue
                    d = self._dist(tc, self._center(det))
                    if d < best_dist:
                        best_dist, best_i = d, i
                if best_dist <= self.max_match_dist:
                    best_iou = self.iou_threshold  # treat as matched

            if best_iou >= self.iou_threshold and best_i >= 0:
                track['bbox'] = detections[best_i]
                track['age'] += 1
                track['ghost'] = 0
                matched.add(best_i)
                cx, cy = self._center(detections[best_i])
                ox, oy = track['origin']
                disp = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                track['max_disp'] = max(track['max_disp'], disp)
            else:
                track['ghost'] = track.get('ghost', 0) + 1

        for i, det in enumerate(detections):
            if i not in matched:
                self.tracks.append({
                    'bbox': det, 'age': 1, 'ghost': 0,
                    'origin': self._center(det), 'max_disp': 0.0,
                })

        # Keep tracks alive during ghost period for re-matching; drop after
        self.tracks = [t for t in self.tracks if t.get('ghost', 0) <= 3]

        # Only emit tracks that have moved — ghost frames never count as detections
        return [t['bbox'] for t in self.tracks
                if t.get('ghost', 0) == 0 and t['max_disp'] >= min_displacement]

    def reset(self):
        self.tracks = []


def _peak_brightness(frame, bbox):
    x, y, w, h = bbox
    roi = frame[max(0, y):y+h, max(0, x):x+w]
    return int(roi.max()) if roi.size > 0 else 0


def passes_shape_filter(contour, min_area, max_area, min_aspect):
    area = cv2.contourArea(contour)
    if area < min_area or area > max_area:
        return False
    if len(contour) >= 5:
        _, (ew, eh), _ = cv2.fitEllipse(contour)
        major = max(ew, eh); minor = min(ew, eh)
    else:
        _, _, bw, bh = cv2.boundingRect(contour)
        major = max(bw, bh); minor = min(bw, bh)
    if minor > 0 and (major / minor) < min_aspect:
        return False
    return True


class AdaptiveCloudDetector:
    """Rolling mean brightness — suppresses frames that spike above baseline."""
    def __init__(self, window=200):
        self.window   = window
        self.history  = deque(maxlen=window)

    def is_cloudy(self, frame, sensitivity):
        mean = float(np.mean(frame))
        if len(self.history) < 10:
            self.history.append(mean)
            return False
        # Median baseline: resistant to sustained brightness contaminating the reference.
        # Check BEFORE appending so the current frame doesn't skew its own baseline.
        baseline = np.median(self.history)
        std      = max(np.std(self.history), 1.0)
        sigma_thresh = sensitivity / 40.0
        cloudy = mean > baseline + sigma_thresh * std
        self.history.append(mean)
        return cloudy


def process_frame(frame, mog2, tracker, mask, kernel, params, cloud_detector):
    is_cloudy_frame = cloud_detector.is_cloudy(frame, params['cloud_thresh'])
    if is_cloudy_frame:
        mog2.apply(frame, learningRate=0)  # freeze background model during clouds — prevents bright sky becoming "normal"
        blank = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        return 0, [], cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR), True, None
    fg_mask = mog2.apply(frame)
    if mask is not None:
        fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=mask)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    # Cloud ratio check — suppress if too much of the valid area is active (clouds cause mass triggering)
    cloud_ratio = params.get('cloud_ratio', 0)
    if cloud_ratio > 0:
        total_px = int(cv2.countNonZero(mask)) if mask is not None else int(fg_mask.size)
        if total_px > 0 and cv2.countNonZero(fg_mask) / total_px > cloud_ratio:
            return 0, [], cv2.cvtColor(np.zeros_like(fg_mask), cv2.COLOR_GRAY2BGR), True, None
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shaped = [c for c in contours if passes_shape_filter(
        c, params['min_area'], params['max_area'], params['min_aspect'])]
    bboxes = [cv2.boundingRect(c) for c in shaped]
    filtered = tracker.update(bboxes, min_displacement=params.get('min_move_scaled', params.get('min_move', 0)))
    min_bright = params.get('min_bright', 0)
    if min_bright > 0:
        filtered = [b for b in filtered
                    if _peak_brightness(frame, b) >= min_bright]
    overlay = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in filtered:
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return len(filtered), filtered, overlay, False, fg_mask


def make_thumbnail(gray_frames, all_detections=None, detect_scale=1.0, params=None):
    if not gray_frames:
        return None
    fh_img, fw_img = gray_frames[0].shape[:2]
    scale = min(THUMB_W / fw_img, THUMB_H / fh_img)
    tw = max(1, int(fw_img * scale))
    th = max(1, int(fh_img * scale))
    # Resize each frame to thumbnail size first, then blend — much faster than
    # blending full-res frames and resizing at the end
    composite = cv2.resize(gray_frames[0], (tw, th),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
    for f in gray_frames[1:]:
        small = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
        np.maximum(composite, small, out=composite)
    bgr = cv2.cvtColor(composite.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    total_h = THUMB_H + (INFO_H if params else 0)
    canvas = np.zeros((total_h, THUMB_W, 3), dtype=np.uint8)
    y_off = (THUMB_H - th) // 2
    x_off = (THUMB_W - tw) // 2
    canvas[y_off:y_off+th, x_off:x_off+tw] = bgr
    if all_detections:
        det_to_thumb = scale / max(detect_scale, 1e-6)
        for (bboxes, fg_mask) in all_detections:
            if fg_mask is None or not bboxes:
                continue
            fg_scaled = cv2.resize(fg_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            for i, (x, y, w, h) in enumerate(bboxes):
                color = DETECTION_COLORS[i % len(DETECTION_COLORS)]
                bx = max(0, int(x * det_to_thumb))
                by = max(0, int(y * det_to_thumb))
                bx2 = min(tw, bx + max(1, int(w * det_to_thumb)))
                by2 = min(th, by + max(1, int(h * det_to_thumb)))
                roi_mask = fg_scaled[by:by2, bx:bx2]
                active = roi_mask > 0
                region = canvas[y_off + by:y_off + by2, x_off + bx:x_off + bx2]
                for c in range(3):
                    region[:, :, c] = np.where(active, color[c], region[:, :, c])
                canvas[y_off + by:y_off + by2, x_off + bx:x_off + bx2] = region
                cv2.rectangle(canvas, (x_off + bx, y_off + by), (x_off + bx2, y_off + by2), color, 1)
    if params:
        # Dark info strip below the image
        canvas[THUMB_H:, :] = (18, 18, 18)
        lines = [
            f"thr={params.get('threshold','?')}  area={params.get('min_area','?')}-{params.get('max_area','?')}  asp={params.get('min_aspect','?')}",
            f"bright={params.get('min_bright','?')}  move={params.get('min_move','?')}  travel={params.get('min_travel','?')}  scale={params.get('scale','?')}",
            f"pre={params.get('pre_buffer','?')}  post={params.get('post_buffer','?')}  cld={params.get('cloud_thresh','?')}  ratio={params.get('cloud_ratio','?')}",
        ]
        font, fscale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1
        lh, margin = 11, 4
        for li, line in enumerate(lines):
            y_text = THUMB_H + margin + lh + li * lh
            cv2.putText(canvas, line, (margin, y_text), font, fscale, (0, 0, 0), thick + 1, cv2.LINE_AA)
            cv2.putText(canvas, line, (margin, y_text), font, fscale, (190, 190, 190), thick, cv2.LINE_AA)
    return canvas


def _drain_save_queue(futures):
    """Wait for all pending saves to finish before continuing detection.
    Prevents simultaneous FFmpeg decode + encode competing for memory and CPU."""
    for f in futures:
        try:
            f.result()
        except Exception:
            pass
    futures.clear()


class Tooltip:
    """Hover tooltip for any Tkinter widget."""
    def __init__(self, widget, text):
        self._widget = widget
        self._text   = text
        self._win    = None
        widget.bind('<Enter>', self._show, add='+')
        widget.bind('<Leave>', self._hide, add='+')
        widget.bind('<ButtonPress>', self._hide, add='+')

    def _show(self, _event=None):
        if self._win:
            return
        x = self._widget.winfo_rootx() + 10
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._win = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f'+{x}+{y}')
        tk.Label(tw, text=self._text, justify='left',
                 background='#ffffe0', foreground='#111111',
                 relief='solid', borderwidth=1,
                 font=('Arial', 8), wraplength=320,
                 padx=6, pady=4).pack()

    def _hide(self, _event=None):
        if self._win:
            self._win.destroy()
            self._win = None


def build_dark_frame(source, n_frames=300, progress_cb=None):
    """
    Read up to n_frames grayscale frames from source (MKV path or RTSP URL),
    compute the per-pixel median, and return a float32 ndarray.
    progress_cb(done, total) is called after each frame if provided.
    Returns None on failure.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fw <= 0 or fh <= 0:
        return None

    cmd = [FFMPEG_PATH, '-i', source,
           '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            creationflags=NO_WINDOW)
    frame_bytes = fw * fh
    frames = []
    try:
        while len(frames) < n_frames:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(fh, fw).copy())
            if progress_cb:
                progress_cb(len(frames), n_frames)
    finally:
        try: proc.stdout.close()
        except Exception: pass
        try: proc.kill(); proc.wait()
        except Exception: pass

    if not frames:
        return None
    return np.median(np.stack(frames, axis=0).astype(np.float32), axis=0).astype(np.float32)


class DetectionWorker:
    def __init__(self, input_path, mask_path, output_dir, params,
                 preview_q, event_q, done_q, stop_event, sw_decode=False,
                 dark_frame_path=None):
        self.input_path      = input_path
        self.mask_path       = mask_path
        self.dark_frame_path = dark_frame_path
        self.output_dir      = output_dir
        self.params          = params
        self.preview_q       = preview_q
        self.event_q         = event_q
        self.done_q          = done_q
        self.stop_event      = stop_event
        self.sw_decode       = sw_decode

    def run(self):
        try:
            self._run()
        except Exception as e:
            import traceback
            self.done_q.put({'error': str(e), 'trace': traceback.format_exc()})
        finally:
            proc = getattr(self, '_ffmpeg_proc', None)
            if proc is not None:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                try:
                    proc.kill()
                    proc.wait()
                except Exception:
                    pass
                self._ffmpeg_proc = None

    def _run(self):
        # Use cv2 briefly just to get video dimensions and frame count
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            self.done_q.put({'error': f'Could not open: {self.input_path}'})
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 20
        self.params['fps'] = fps
        cap.release()

        mask = (cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
                if self.mask_path and os.path.exists(self.mask_path) else None)

        dark_frame = None
        if self.dark_frame_path and os.path.exists(self.dark_frame_path):
            try:
                df = np.load(self.dark_frame_path).astype(np.float32)
                if df.shape != (fh, fw):
                    df = cv2.resize(df, (fw, fh), interpolation=cv2.INTER_LINEAR)
                dark_frame = df
                print(f"[DARK] Loaded dark frame {df.shape} from {self.dark_frame_path}")
            except Exception as e:
                print(f"[DARK] Failed to load dark frame: {e}")

        # Check for checkpoint
        src_base = os.path.splitext(self.input_path)[0]
        checkpoint_path = src_base + '.checkpoint'
        resume_from = 0
        if os.path.exists(checkpoint_path):
            try:
                resume_from = int(open(checkpoint_path).read().strip())
                print(f"[RESUME] Resuming from frame {resume_from}")
            except Exception:
                resume_from = 0

        warmup = self.params['warmup']
        seek_frame = max(0, resume_from - warmup)
        seek_sec   = seek_frame / fps

        # Open FFmpeg pipe; uses CUDA hardware decoding unless SW Decode is checked
        ffmpeg_cmd = [FFMPEG_PATH] + ([] if self.sw_decode else HWACCEL_ARGS)
        if seek_frame > 0:
            ffmpeg_cmd += ['-ss', f'{seek_sec:.3f}']
        ffmpeg_cmd += ['-i', self.input_path,
                       '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1']
        self._ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=NO_WINDOW)
        frame_bytes = fw * fh

        scale = self.params['scale']
        area_scale = scale * scale  # area shrinks by scale²
        self.params['min_area'] = max(1, int(self.params['min_area'] * area_scale))
        self.params['max_area'] = max(1, int(self.params['max_area'] * area_scale))
        # Keep slider value for logging; store scaled version separately for detection
        self.params['min_move_scaled'] = self.params.get('min_move', 0) * scale
        mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self.params['history'],
            varThreshold=self.params['threshold'],
            detectShadows=False)
        tracker = TrackManager(max_frames=self.params['max_track'],
                               max_match_dist=self.params.get('max_match_dist', 0))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cloud_detector = AdaptiveCloudDetector(window=200)

        # Pre-scale mask once
        if mask is not None and scale != 1.0:
            mh, mw = mask.shape[:2]
            mask_small = cv2.resize(mask,
                                    (int(mw * scale), int(mh * scale)),
                                    interpolation=cv2.INTER_NEAREST)
        else:
            mask_small = mask

        pre_buffer   = deque(maxlen=self.params['pre_buffer'])
        cloud_rolling = deque(maxlen=50)
        post_cd      = 0
        pending      = []   # (idx, gray, count, bboxes)
        pending_gray = []   # gray frames for thumbnail

        total_detections = 0
        detected_frames  = 0
        cloudy_frames    = 0
        frame_idx        = seek_frame
        checkpoint_every = 500
        t_start          = time.time()
        executor    = ThreadPoolExecutor(max_workers=1)
        save_futures = []

        while not self.stop_event.is_set():
            raw_bytes = self._ffmpeg_proc.stdout.read(frame_bytes)
            if len(raw_bytes) < frame_bytes:
                break

            gray = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(fh, fw)

            if dark_frame is not None:
                gray = np.clip(gray.astype(np.float32) - dark_frame, 0, 255).astype(np.uint8)

            # Downscale for detection
            if scale != 1.0:
                h, w = gray.shape[:2]
                small = cv2.resize(gray, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            else:
                small = gray

            count, bboxes, overlay, was_cloudy, fg_mask = process_frame(
                small, mog2, tracker, mask_small, kernel, self.params,
                cloud_detector)

            cloud_rolling.append(was_cloudy)
            if was_cloudy:
                cloudy_frames += 1

            # Send preview update every N frames
            if frame_idx % PREVIEW_EVERY_N == 0:
                elapsed = time.time() - t_start
                fps = frame_idx / elapsed if elapsed > 0 else 0
                eta = (total - frame_idx) / fps if fps > 0 and total > 0 else 0
                stats = {
                    'frame': frame_idx,
                    'total': total,
                    'detections': total_detections,
                    'cloudy': cloudy_frames,
                    'elapsed': elapsed,
                    'eta': eta,
                }
                if not self.preview_q.full():
                    # Resize preview to save queue memory
                    prev = cv2.resize(overlay, (640, 480),
                                      interpolation=cv2.INTER_AREA)
                    self.preview_q.put_nowait((prev, stats))

            # Save checkpoint periodically
            if frame_idx % checkpoint_every == 0 and frame_idx > seek_frame:
                try:
                    open(checkpoint_path, 'w').write(str(frame_idx))
                except Exception:
                    pass

            # Skip detection during warmup or until resume point
            if frame_idx < seek_frame + warmup or frame_idx < warmup:
                frame_idx += 1
                continue

            entry = (frame_idx, gray, count, bboxes, fg_mask)  # gray = full-res; fg_mask = detect-scale

            if count > 0:
                detected_frames += 1
                total_detections += count
                if post_cd == 0:
                    pending.extend(list(pre_buffer))
                    pending_gray.extend([e[1] for e in pre_buffer])
                    pre_buffer.clear()
                pending.append(entry)
                pending_gray.append(gray)
                post_cd = self.params['post_buffer']

                # Safety cap — flush at 100 frames (~480MB at 2592x1944 grayscale)
                # Tracker stays alive across the flush; stitcher merges resulting clips
                if len(pending) > 100:
                    _cfrac = sum(cloud_rolling) / max(len(cloud_rolling), 1)
                    _drain_save_queue(save_futures)
                    save_futures.append(
                        executor.submit(self._save_event,
                                        list(pending), list(pending_gray), _cfrac))
                    pending.clear()
                    pending_gray.clear()
                    # post_cd intentionally not reset — keeps the event going

            elif post_cd > 0:
                pending.append(entry)
                pending_gray.append(gray)
                post_cd -= 1
                if post_cd == 0:
                    _cfrac = sum(cloud_rolling) / max(len(cloud_rolling), 1)
                    _drain_save_queue(save_futures)
                    save_futures.append(
                        executor.submit(self._save_event,
                                        list(pending), list(pending_gray), _cfrac))
                    pending.clear()
                    pending_gray.clear()
            else:
                pre_buffer.append(entry)

            frame_idx += 1

        # Flush remaining pending frames
        if pending:
            _cfrac = sum(cloud_rolling) / max(len(cloud_rolling), 1)
            save_futures.append(
                executor.submit(self._save_event,
                                list(pending), list(pending_gray), _cfrac))

        self._ffmpeg_proc.stdout.close()
        self._ffmpeg_proc.wait()

        # Wait for all background saves to finish before writing logs
        executor.shutdown(wait=True)
        saved_events = sum(1 for f in save_futures if f.result() is not False)

        elapsed = time.time() - t_start

        if self.stop_event.is_set():
            # Save checkpoint so we can resume later
            try:
                open(checkpoint_path, 'w').write(str(frame_idx))
            except Exception:
                pass
        else:
            # Completed — remove checkpoint
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

        self._write_logs(frame_idx, total_detections, detected_frames,
                         cloudy_frames, elapsed, saved_events)

        self.done_q.put({
            'frames': frame_idx,
            'detections': total_detections,
            'detected_frames': detected_frames,
            'cloudy': cloudy_frames,
            'elapsed': elapsed,
            'saved_events': saved_events,
        })

    def _write_logs(self, frames, detections, det_frames, cloudy, elapsed, saved_events=0):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        p = self.params

        summary_lines = [
            f"processed : {ts}",
            f"source    : {self.input_path}",
            f"frames    : {frames}",
            f"event_folders: {saved_events}",
            f"events    : {det_frames}",
            f"detections: {detections}",
            f"cloudy    : {cloudy}",
            f"elapsed   : {int(elapsed//60):02d}m{int(elapsed%60):02d}s",
            f"scale     : {p['scale']}",
            f"thresh    : {p['threshold']}",
            f"min_area  : {p['min_area']}",
            f"max_area  : {p['max_area']}",
            f"aspect    : {p['min_aspect']}",
            f"pre_buf   : {p['pre_buffer']}",
            f"post_buf  : {p['post_buffer']}",
            f"warmup    : {p['warmup']}",
            f"cld_thresh: {p['cloud_thresh']}",
            f"cld_ratio : {p.get('cloud_ratio', 0)}",
            f"min_move  : {p.get('min_move', 0)}",
            f"max_track : {p.get('max_track', 0)}",
        ]
        summary = "\n".join(summary_lines)

        # Marker file next to source clip
        src_dir = (os.path.dirname(self.input_path)
                   if os.path.isfile(self.input_path) else self.input_path)
        src_name = os.path.splitext(os.path.basename(self.input_path))[0]
        marker_path = os.path.join(src_dir, f"{src_name}.processed")
        with open(marker_path, 'w') as f:
            f.write(summary)

        # Full JSON log in output folder
        log_data = {
            'processed': ts,
            'source': self.input_path,
            'output_dir': self.output_dir,
            'frames': frames,
            'event_folders': saved_events,
            'events': det_frames,
            'detections': detections,
            'cloudy_suppressed': cloudy,
            'elapsed_s': round(elapsed, 1),
            'params': {k: v for k, v in p.items()},
        }
        log_path = os.path.join(self.output_dir, f"{src_name}_detection_log.json")
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    def _save_event(self, pending, gray_frames, cloud_frac=0.0):
        # Determine if adaptive cloud mode is active
        cloud_min_bright = self.params.get('cloud_min_bright', 0)
        cloud_min_travel = self.params.get('cloud_min_travel', 0)
        cloudy_mode = cloud_frac >= 0.3 and (cloud_min_bright > 0 or cloud_min_travel > 0)

        # Pre-filter: require minimum centroid travel within the event window
        # In cloudy mode use cloud_min_travel if set, otherwise fall back to min_travel
        min_travel = self.params.get('min_travel', 0)
        effective_min_travel = (cloud_min_travel if (cloudy_mode and cloud_min_travel > 0)
                                else min_travel)
        if effective_min_travel > 0:
            inv_scale = 1.0 / max(self.params.get('scale', 1.0), 1e-6)
            det_pts = []
            for (_, _, _, fbboxes, _) in pending:
                if fbboxes:
                    x, y, w, h = fbboxes[0]
                    det_pts.append(((x + w / 2) * inv_scale, (y + h / 2) * inv_scale))
            travel = 0.0
            if len(det_pts) >= 2:
                dx = det_pts[-1][0] - det_pts[0][0]
                dy = det_pts[-1][1] - det_pts[0][1]
                travel = (dx*dx + dy*dy) ** 0.5
            if travel < effective_min_travel:
                return False  # skip — too stationary

        # In cloudy mode, require peak brightness above cloud_min_bright threshold
        if cloudy_mode and cloud_min_bright > 0:
            inv_scale = 1.0 / max(self.params.get('scale', 1.0), 1e-6)
            max_peak = 0
            for (_, gframe, _, fbboxes, _) in pending:
                for (x, y, w, h) in fbboxes:
                    fx = int(x * inv_scale); fy = int(y * inv_scale)
                    fw = max(1, int(w * inv_scale)); fh = max(1, int(h * inv_scale))
                    max_peak = max(max_peak, _peak_brightness(gframe, (fx, fy, fw, fh)))
            if max_peak < cloud_min_bright:
                return False  # skip — too dim for cloudy conditions

        last_det_pos = max(
            (i for i, (_, _, _, fbboxes, _) in enumerate(pending) if fbboxes),
            default=-1)
        if last_det_pos >= 0:
            trim_end = last_det_pos + 11  # keep 10 frames after last detection
            if trim_end < len(pending):
                pending     = pending[:trim_end]
                gray_frames = gray_frames[:trim_end]

        src = os.path.splitext(os.path.basename(self.input_path))[0]
        first_frame = pending[0][0] if pending else 0
        fps = self.params.get('fps', 20)
        total_secs = int(first_frame / fps)
        mm = total_secs // 60
        ss = total_secs % 60
        event_dir = os.path.join(self.output_dir, f"event_{src}_{mm:02d}m{ss:02d}s_{first_frame:06d}")
        os.makedirs(event_dir, exist_ok=True)

        det_meta = []
        for (fidx, gframe, fcount, fbboxes, _) in pending:
            if fbboxes:
                centroids = [[x + w//2, y + h//2] for (x, y, w, h) in fbboxes]
                det_meta.append({
                    'frame': fidx,
                    'centroids': centroids,
                    'bboxes': [list(b) for b in fbboxes],
                    'count': fcount,
                })

        # Write clip.mkv by piping grayscale frames into FFmpeg
        clip_path = os.path.join(event_dir, 'clip.mkv')
        fh, fw = pending[0][1].shape[:2]
        ffmpeg_cmd = [
            FFMPEG_PATH, '-y',
            '-f', 'rawvideo', '-pix_fmt', 'gray',
            '-s', f'{fw}x{fh}', '-r', str(fps),
            '-i', 'pipe:0',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-pix_fmt', 'yuv420p',
            clip_path,
        ]
        proc = None
        try:
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    creationflags=NO_WINDOW)
            for (_, gframe, _, _, _) in pending:
                proc.stdin.write(gframe.tobytes())
            proc.stdin.close()
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg exited with code {proc.returncode}")
        except Exception:
            import traceback
            tb = traceback.format_exc()
            try:
                log_path = os.path.join(_BASE_DIR, 'streaker_error.log')
                with open(log_path, 'a') as f:
                    f.write(f"\n--- {__import__('datetime').datetime.now()} [mkv-fallback] ---\n{tb}\n")
            except Exception:
                pass
            self.event_q.put({'warning': f"MKV write failed for {os.path.basename(event_dir)} — fell back to JPEGs. Check streaker_error.log."})
            if proc is not None:
                try: proc.stdin.close()
                except Exception: pass
                proc.kill()
                proc.wait()
            # Fall back to individual JPEGs
            for (fidx, gframe, _, _, _) in pending:
                cv2.imwrite(os.path.join(event_dir, f"frame_{fidx:06d}.jpg"),
                            gframe, [cv2.IMWRITE_JPEG_QUALITY, 90])

        detect_scale = self.params.get('scale', 1.0)
        all_detections = [(fbboxes, fmask) for (_, _, _, fbboxes, fmask) in pending if fbboxes]
        thumb_bgr = make_thumbnail(gray_frames, all_detections, detect_scale, params=self.params)
        thumb_path = os.path.join(event_dir, "_thumbnail.jpg")
        if thumb_bgr is not None:
            cv2.imwrite(thumb_path, thumb_bgr)

        # Save metadata for stitching
        meta = {
            'source_clip':  self.input_path,
            'start_frame':  pending[0][0],
            'end_frame':    pending[-1][0],
            'n_frames':     len(pending),
            'fps':          self.params.get('fps', 20),
            'detect_scale': detect_scale,
            'detections':   det_meta,
            'params':       dict(self.params),
        }
        with open(os.path.join(event_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        total = sum(c for _, _, c, _, _ in pending)
        self.event_q.put({
            'dir':    event_dir,
            'thumb':  thumb_path,
            'frames': len(pending),
            'count':  total,
        })

# ------------------------------------------------------------------------------
# Event Viewer — popup playback for a single event clip
# ------------------------------------------------------------------------------

class EventViewer:
    def __init__(self, parent, event_dir):
        self.event_dir = event_dir
        self.frame_paths = sorted([
            os.path.join(event_dir, f) for f in os.listdir(event_dir)
            if f.startswith('frame_') and f.endswith('.jpg')])

        if not self.frame_paths:
            messagebox.showinfo("Empty", "No frames in this event.")
            return

        self.idx      = 0
        self.paused   = True
        self.speed_ms = 80
        self.loop_id  = None
        self.show_composite = False
        self._scrubbing = False


        # Max-blend composite at full resolution (not thumbnail-sized)
        raw_frames = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in self.frame_paths]
        raw_frames = [f for f in raw_frames if f is not None]
        if raw_frames:
            comp = raw_frames[0].copy().astype(np.float32)
            for f in raw_frames[1:]:
                np.maximum(comp, f.astype(np.float32), out=comp)
            self.composite = cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            self.composite = None

        self.win = tk.Toplevel(parent)
        self.win.title(f"Event — {os.path.basename(event_dir)}")
        self.win.geometry("900x700")
        self._build()
        self._show_frame()
        self.win.after(100, self._toggle_play)

    def _build(self):
        self.canvas = tk.Canvas(self.win, bg='black')
        self.canvas.pack(fill='both', expand=True)

        ctrl = tk.Frame(self.win)
        ctrl.pack(fill='x', pady=4)

        tk.Button(ctrl, text="|◀", command=lambda: self._goto(0)).pack(side='left', padx=2)
        tk.Button(ctrl, text="◀", command=lambda: self._step(-1)).pack(side='left', padx=2)
        self.play_btn = tk.Button(ctrl, text="▶ Play", command=self._toggle_play)
        self.play_btn.pack(side='left', padx=2)
        tk.Button(ctrl, text="▶|", command=lambda: self._goto(-1)).pack(side='left', padx=2)

        tk.Label(ctrl, text="Speed:").pack(side='left', padx=(10, 2))
        self.speed_var = tk.IntVar(value=self.speed_ms)
        tk.Scale(ctrl, from_=10, to=500, orient='horizontal', variable=self.speed_var,
                 length=120, showvalue=False,
                 command=lambda v: setattr(self, 'speed_ms', int(v))).pack(side='left')
        tk.Button(ctrl, text="Real Time",
                  command=self._set_realtime).pack(side='left', padx=4)

        tk.Button(ctrl, text="Max-Blend Toggle",
                  command=self._toggle_composite).pack(side='left', padx=10)
        tk.Button(ctrl, text="Save Stack",
                  command=self._save_stack).pack(side='left', padx=4)

        self.scrubber = ttk.Scale(self.win, from_=0, to=len(self.frame_paths)-1,
                                  orient='horizontal',
                                  command=lambda v: self._goto(int(float(v))))
        self.scrubber.pack(fill='x', padx=8, pady=2)

        self.status_var = tk.StringVar()
        tk.Label(self.win, textvariable=self.status_var, anchor='w').pack(fill='x', padx=8)

        self.win.bind('<Left>',  lambda e: self._step(-1))
        self.win.bind('<Right>', lambda e: self._step(1))
        self.win.bind('<space>', lambda e: self._toggle_play())

    def _goto(self, idx):
        if self._scrubbing:
            return
        if idx == -1:
            idx = len(self.frame_paths) - 1
        self.idx = max(0, min(idx, len(self.frame_paths) - 1))
        self._scrubbing = True
        self.scrubber.set(self.idx)
        self._scrubbing = False
        self._show_frame()

    def _step(self, d):
        self._goto(self.idx + d)

    def _toggle_play(self):
        self.paused = not self.paused
        self.play_btn.config(text="⏸ Pause" if not self.paused else "▶ Play")
        if not self.paused:
            self._play_loop()

    def _play_loop(self):
        if self.paused:
            return
        if self.idx >= len(self.frame_paths) - 1:
            self.idx = 0  # loop back to start
        self._step(1)
        self.loop_id = self.win.after(self.speed_ms, self._play_loop)

    def _set_realtime(self):
        # Try to read FPS from detection log in parent folder
        fps = 20
        try:
            parent = os.path.dirname(self.event_dir)
            for f in os.listdir(parent):
                if f.endswith('_detection_log.json'):
                    import json
                    with open(os.path.join(parent, f)) as fh:
                        data = json.load(fh)
                    fps = data.get('params', {}).get('fps', 20)
                    break
        except Exception:
            pass
        self.speed_ms = max(10, int(1000 / fps))
        self.speed_var.set(self.speed_ms)

    def _toggle_composite(self):
        self.show_composite = not self.show_composite
        self._show_frame()

    def _save_stack(self):
        if self.composite is None:
            tk.messagebox.showerror("No Stack", "No composite available for this event.")
            return
        out_path = os.path.join(self.event_dir, "stack.png")
        cv2.imwrite(out_path, self.composite)
        tk.messagebox.showinfo("Saved", f"Stack saved to:\n{out_path}")

    def _show_frame(self):
        if self.show_composite and self.composite is not None:
            cw = max(self.canvas.winfo_width(), 860)
            ch = max(self.canvas.winfo_height(), 580)
            h, w = self.composite.shape[:2]
            scale = min(cw / w, ch / h)
            img_bgr = cv2.resize(self.composite,
                                 (max(1, int(w * scale)), max(1, int(h * scale))),
                                 interpolation=cv2.INTER_AREA)
        else:
            img_bgr = cv2.imread(self.frame_paths[self.idx], cv2.IMREAD_COLOR)
            if img_bgr is None:
                return
            cw = max(self.canvas.winfo_width(), 860)
            ch = max(self.canvas.winfo_height(), 580)
            h, w = img_bgr.shape[:2]
            scale = min(cw/w, ch/h)
            img_bgr = cv2.resize(img_bgr, (max(1, int(w*scale)), max(1, int(h*scale))))

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
        self.canvas.delete('all')
        cw = self.canvas.winfo_width() or 860
        ch = self.canvas.winfo_height() or 580
        self.canvas.create_image(cw//2, ch//2, anchor='center', image=img)
        self.canvas.image = img
        self.status_var.set(
            f"Frame {self.idx+1}/{len(self.frame_paths)}  |  "
            f"{'MAX-BLEND COMPOSITE' if self.show_composite else os.path.basename(self.frame_paths[self.idx])}")

# ------------------------------------------------------------------------------
# Thumbnail Panel
# ------------------------------------------------------------------------------

PAGE_SIZE = 50

class ThumbnailPanel:
    def __init__(self, parent, on_click, on_view_clip=None):
        self.on_click      = on_click
        self.on_view_clip  = on_view_clip
        self.thumbnails  = []   # PhotoImage refs for current page
        self._flag_vars  = []   # StringVar refs for flag buttons (prevent GC)
        self.all_events  = []   # every event_info ever added
        self.visited    = set()
        self.flagged    = set()
        self._flags_path = None
        self.page = 0
        self._sort_mode = tk.StringVar(value='all')

        BG = '#1a1a1a'
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill='both', expand=True)

        # Header row: title + sort + page navigation
        hdr = tk.Frame(frame, bg=BG)
        hdr.pack(fill='x')
        tk.Label(hdr, text="DETECTED EVENTS", bg=BG, fg='#aaaaaa',
                 font=('Arial', 9, 'bold')).pack(side='left', pady=(4, 0), padx=4)
        sort_cb = ttk.Combobox(hdr, textvariable=self._sort_mode,
                               values=['all', 'flagged_first', 'flagged_only'],
                               state='readonly', width=13, font=('Arial', 7))
        sort_cb.pack(side='left', padx=4, pady=2)
        self._sort_mode.trace_add('write', lambda *_: self._on_sort_change())
        self.prev_btn = tk.Button(hdr, text="◀", command=self._prev_page,
                                  bg='#333333', fg='white', relief='flat',
                                  width=2, state='disabled')
        self.prev_btn.pack(side='right', padx=(0, 2), pady=2)
        self.next_btn = tk.Button(hdr, text="▶", command=self._next_page,
                                  bg='#333333', fg='white', relief='flat',
                                  width=2, state='disabled')
        self.next_btn.pack(side='right', padx=(0, 2), pady=2)
        self.page_var = tk.StringVar(value="")
        tk.Label(hdr, textvariable=self.page_var, bg=BG, fg='#666666',
                 font=('Arial', 8)).pack(side='right', padx=4)

        # Footer row: flag count + delete unflagged button
        footer = tk.Frame(frame, bg=BG)
        footer.pack(fill='x', side='bottom')
        self.flag_count_lbl = tk.Label(footer, text="", bg=BG, fg='#888888',
                                       font=('Arial', 8))
        self.flag_count_lbl.pack(side='left', padx=6, pady=2)
        self.delete_btn = tk.Button(footer, text="🗑 Delete Unflagged",
                                    command=self._delete_unflagged,
                                    bg='#3a1a1a', fg='#ff6666',
                                    font=('Arial', 8), relief='flat', padx=6,
                                    state='disabled')
        self.delete_btn.pack(side='right', padx=4, pady=2)

        container = tk.Frame(frame, bg=BG)
        container.pack(fill='both', expand=True)

        self.canvas = tk.Canvas(container, bg=BG, highlightthickness=0)
        scroll = ttk.Scrollbar(container, orient='vertical',
                               command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)

        self.inner = tk.Frame(self.canvas, bg=BG)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.inner, anchor='nw')
        self.inner.bind('<Configure>', self._on_inner_resize)
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        self.canvas.bind_all('<MouseWheel>',
                             lambda e: self.canvas.yview_scroll(-1*(e.delta//120), 'units'))

    def _on_inner_resize(self, _):
        self.canvas.after_idle(self._refresh_scrollregion)

    def _refresh_scrollregion(self):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _on_canvas_resize(self, e):
        self.canvas.itemconfig(self.canvas_window, width=e.width)

    @staticmethod
    def _flag_key(path):
        """Use just the event folder basename as the flag key.
        This is robust against UNC vs drive-letter path differences (e.g. \\tnas\G509 vs T:)."""
        return os.path.basename(os.path.normcase(os.path.normpath(path)))

    def _view_events(self):
        mode = self._sort_mode.get()
        if mode == 'flagged_only':
            return [e for e in self.all_events if self._flag_key(e['dir']) in self.flagged]
        if mode == 'flagged_first':
            flagged = [e for e in self.all_events if self._flag_key(e['dir']) in self.flagged]
            rest    = [e for e in self.all_events if self._flag_key(e['dir']) not in self.flagged]
            return flagged + rest
        return self.all_events

    def _on_sort_change(self):
        mode = self._sort_mode.get()
        print(f'[SORT] mode={mode!r}  flagged={len(self.flagged)}  total={len(self.all_events)}  view={len(self._view_events())}')
        self.page = 0
        self._render_page()

    def _page_count(self):
        return max(1, (len(self._view_events()) + PAGE_SIZE - 1) // PAGE_SIZE)

    def _update_nav(self):
        total = len(self._view_events())
        pages = self._page_count()
        if total == 0:
            self.page_var.set("")
        else:
            start = self.page * PAGE_SIZE + 1
            end   = min((self.page + 1) * PAGE_SIZE, total)
            self.page_var.set(f"{start}–{end} / {total}")
        self.prev_btn.config(state='normal' if self.page > 0 else 'disabled')
        self.next_btn.config(state='normal' if self.page < pages - 1 else 'disabled')

    def _prev_page(self):
        if self.page > 0:
            self.page -= 1
            self._render_page()

    def _next_page(self):
        if self.page < self._page_count() - 1:
            self.page += 1
            self._render_page()

    def _render_page(self):
        for w in self.inner.winfo_children():
            w.destroy()
        self.thumbnails.clear()
        self._flag_vars.clear()
        self.canvas.yview_moveto(0)
        start = self.page * PAGE_SIZE
        for i, ev in enumerate(self._view_events()[start:start + PAGE_SIZE]):
            self._render_card(ev, start + i)
        self._update_nav()
        self.canvas.after_idle(self._refresh_scrollregion)

    def _card_colors(self, event_dir):
        """Return (bg, border_color, border_thickness) based on flagged/visited state."""
        if self._flag_key(event_dir) in self.flagged:
            return '#2a2410', '#ffd700', 2
        elif event_dir in self.visited:
            return '#1a3a22', '#33aa55', 2
        else:
            return '#2a2a2a', '#2a2a2a', 0

    def _apply_card_colors(self, card, event_dir):
        bg, border, thickness = self._card_colors(event_dir)
        card.config(bg=bg, highlightbackground=border, highlightthickness=thickness)
        for child in card.winfo_children():
            try: child.config(bg=bg)
            except Exception: pass
            for gc in child.winfo_children():
                try: gc.config(bg=bg)
                except Exception: pass
                for ggc in gc.winfo_children():
                    try: ggc.config(bg=bg)
                    except Exception: pass

    def _render_card(self, event_info, global_idx):
        event_dir    = event_info['dir']
        thumb_path   = event_info['thumb']
        n_frames     = event_info['frames']
        n_detections = event_info['count']

        card_bg, border, thickness = self._card_colors(event_dir)
        card = tk.Frame(self.inner, bg=card_bg,
                        highlightbackground=border,
                        highlightthickness=thickness,
                        relief='flat', bd=0)
        card._event_dir = event_dir
        card.pack(fill='x', padx=6, pady=4)

        def _click(d=event_dir, c=card):
            self.visited.add(d)
            self._apply_card_colors(c, d)
            self.on_click(d)

        if os.path.exists(thumb_path):
            img_bgr = cv2.imread(thumb_path)
            if img_bgr is not None:
                img = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
                lbl = tk.Label(card, image=img, bg=card_bg, cursor='hand2')
                lbl.image = img
                lbl.pack(pady=(4, 0))
                self.thumbnails.append(img)
                lbl.bind('<Button-1>', lambda e, fn=_click: fn())

        info = tk.Frame(card, bg=card_bg)
        info.pack(fill='x', padx=6, pady=(2, 4))

        top_row = tk.Frame(info, bg=card_bg)
        top_row.pack(fill='x')
        tk.Label(top_row, text=f"Event {global_idx + 1}",
                 bg=card_bg, fg='white',
                 font=('Arial', 10, 'bold')).pack(side='left')
        flag_text = tk.StringVar(value="⭐" if self._flag_key(event_dir) in self.flagged else "☆")
        self._flag_vars.append(flag_text)
        def _toggle(d=event_dir, c=card, v=flag_text):
            self._toggle_flag(d, c, v)
        tk.Button(top_row, textvariable=flag_text, command=_toggle,
                  bg=card_bg, fg='#ffd700', activebackground=card_bg,
                  font=('Arial', 12), relief='flat', cursor='hand2',
                  bd=0).pack(side='right')

        tk.Label(info, text=os.path.basename(event_dir),
                 bg=card_bg, fg='#888888',
                 font=('Arial', 8)).pack(anchor='w')
        tk.Label(info, text=f"{n_frames} frames  |  {n_detections} detections",
                 bg=card_bg, fg='#aaffaa',
                 font=('Arial', 9)).pack(anchor='w', pady=(2, 0))
        def _view_in_player(d=event_dir):
            if self.on_view_clip:
                self.on_view_clip(d)
        tk.Button(info, text="▶ View Clip",
                  command=_view_in_player,
                  bg='#1a3a5a', fg='white', relief='flat',
                  cursor='hand2').pack(anchor='w', pady=(4, 0))

    def _highlight_card(self, card):
        self._apply_card_colors(card, card._event_dir)

    def select_event(self, event_dir):
        """Highlight the card for event_dir; switch page if needed and scroll to it."""
        idx = next((i for i, e in enumerate(self._view_events()) if e['dir'] == event_dir), -1)
        if idx < 0:
            return
        target_page = idx // PAGE_SIZE
        if target_page != self.page:
            self.page = target_page
            self._render_page()
        self.visited.add(event_dir)
        for card in self.inner.winfo_children():
            if getattr(card, '_event_dir', None) == event_dir:
                self._highlight_card(card)
                self.canvas.after_idle(lambda c=card: self._scroll_to_card(c))
                break

    def _scroll_to_card(self, card):
        self.canvas.update_idletasks()
        total = self.inner.winfo_height()
        if total <= 0:
            return
        frac = max(0.0, min(1.0, (card.winfo_y() - 40) / total))
        self.canvas.yview_moveto(frac)

    def _toggle_flag(self, event_dir, card, flag_var):
        key = self._flag_key(event_dir)
        if key in self.flagged:
            self.flagged.discard(key)
            flag_var.set("☆")
        else:
            self.flagged.add(key)
            flag_var.set("⭐")
        self._apply_card_colors(card, event_dir)
        self._save_flags()
        self._update_delete_btn()

    def _load_flags(self):
        print(f'[FLAGS] load → path={self._flags_path!r}  exists={self._flags_path and os.path.exists(self._flags_path)}')
        if self._flags_path and os.path.exists(self._flags_path):
            try:
                with open(self._flags_path) as f:
                    # Stored values may be full paths (old format) or basenames (new format)
                    self.flagged = set(
                        os.path.basename(os.path.normcase(os.path.normpath(p)))
                        for p in json.load(f))
                print(f'[FLAGS] loaded {len(self.flagged)} flag(s): {self.flagged}')
            except Exception as e:
                print(f'[FLAGS] load FAILED: {e}')
                self.flagged = set()

    def _save_flags(self):
        print(f'[FLAGS] save → path={self._flags_path!r}  flagged={self.flagged}')
        if self._flags_path:
            try:
                with open(self._flags_path, 'w') as f:
                    json.dump(list(self.flagged), f)
                print(f'[FLAGS] saved OK')
            except Exception as e:
                print(f'[FLAGS] save FAILED: {e}')

    def _update_delete_btn(self):
        n_flagged   = sum(1 for e in self.all_events if self._flag_key(e['dir']) in self.flagged)
        n_unflagged = len(self.all_events) - n_flagged
        if n_flagged > 0 and n_unflagged > 0:
            self.delete_btn.config(state='normal', text="🗑 Delete Unflagged")
            self.flag_count_lbl.config(text=f"{n_flagged} flagged  ·  {n_unflagged} unflagged")
        elif n_flagged > 0:
            self.delete_btn.config(state='disabled', text="🗑 Delete Unflagged")
            self.flag_count_lbl.config(text=f"All {n_flagged} flagged")
        elif len(self.all_events) > 0:
            self.delete_btn.config(state='normal', text="🗑 Delete All")
            self.flag_count_lbl.config(text=f"{len(self.all_events)} clips  ·  none flagged")
        else:
            self.delete_btn.config(state='disabled', text="🗑 Delete Unflagged")
            self.flag_count_lbl.config(text="")

    def _delete_unflagged(self):
        import shutil
        unflagged = [e for e in self.all_events if self._flag_key(e['dir']) not in self.flagged]
        n = len(unflagged)
        n_keep = len(self.all_events) - n
        if n_keep == 0:
            title, msg = "Delete All", (
                f"Permanently delete all {n} clip folder{'s' if n != 1 else ''}?\n\n"
                "This cannot be undone.")
        else:
            title, msg = "Delete Unflagged", (
                f"Permanently delete {n} unflagged clip folder{'s' if n != 1 else ''}?\n"
                f"({n_keep} flagged clip{'s' if n_keep != 1 else ''} will be kept)\n\n"
                "This cannot be undone.")
        if not messagebox.askyesno(title, msg):
            return
        errors = []
        for e in unflagged:
            try:
                shutil.rmtree(e['dir'])
            except Exception as ex:
                errors.append(f"{os.path.basename(e['dir'])}: {ex}")
        self.all_events = [e for e in self.all_events if self._flag_key(e['dir']) in self.flagged]
        self._render_page()
        self._update_delete_btn()
        if errors:
            messagebox.showerror("Delete Errors", "\n".join(errors))

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()
        self.thumbnails.clear()
        self._flag_vars.clear()
        self.all_events.clear()
        self.visited.clear()
        self.flagged.clear()
        self._flags_path = None
        self.page = 0
        self.canvas.yview_moveto(0)
        self._update_nav()
        self._update_delete_btn()

    def add_event(self, event_info):
        event_info = dict(event_info)
        event_info['dir'] = os.path.normpath(event_info['dir'])
        if self._flags_path is None:
            clips_dir = os.path.dirname(event_info['dir'])
            self._flags_path = os.path.join(clips_dir, '_flags.json')
            self._load_flags()
        self.all_events.append(event_info)
        # Only render if this event falls on the current page
        idx = len(self.all_events) - 1
        if self.page * PAGE_SIZE <= idx < (self.page + 1) * PAGE_SIZE:
            self._render_card(event_info, idx)
        self._update_nav()
        self._update_delete_btn()

# ------------------------------------------------------------------------------
# Event Stitcher — merges fragmented events from the same object
# ------------------------------------------------------------------------------

class EventStitcher:
    def __init__(self, events_folder, ffmpeg_path=FFMPEG_PATH,
                 max_gap_frames=300, position_tolerance=80):
        self.events_folder     = events_folder
        self.ffmpeg_path       = ffmpeg_path
        self.max_gap_frames    = max_gap_frames   # max frames between events to consider merging
        self.pos_tolerance     = position_tolerance  # max pixel distance for predicted vs actual

    def run(self):
        # Load all metadata files
        events = []
        for d in sorted(os.listdir(self.events_folder)):
            meta_path = os.path.join(self.events_folder, d, 'metadata.json')
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            meta['event_dir'] = os.path.join(self.events_folder, d)
            events.append(meta)

        if len(events) < 2:
            return 0

        # Group by source clip
        from collections import defaultdict
        by_clip = defaultdict(list)
        for ev in events:
            by_clip[ev['source_clip']].append(ev)

        merged_count = 0
        for clip, clip_events in by_clip.items():
            clip_events.sort(key=lambda e: e['start_frame'])
            merged_count += self._stitch_clip_events(clip, clip_events)

        return merged_count

    def _get_track(self, event):
        dets = [d for d in event['detections'] if d['centroids']]
        if not dets:
            return []
        return [(d['frame'], d['centroids'][0]) for d in dets]

    def _predict_position(self, track, target_frame):
        if len(track) < 2:
            if track:
                return track[-1][1]
            return None
        # Linear fit through last min(10, len) points
        pts = track[-10:]
        frames = [p[0] for p in pts]
        xs     = [p[1][0] for p in pts]
        ys     = [p[1][1] for p in pts]
        n = len(frames)
        sf = sum(frames); sx = sum(xs); sy = sum(ys)
        sf2 = sum(f*f for f in frames)
        sfx = sum(frames[i]*xs[i] for i in range(n))
        sfy = sum(frames[i]*ys[i] for i in range(n))
        denom = n * sf2 - sf * sf
        if denom == 0:
            return xs[-1], ys[-1]
        vx = (n * sfx - sf * sx) / denom
        vy = (n * sfy - sf * sy) / denom
        bx = (sx - vx * sf) / n
        by = (sy - vy * sf) / n
        pred_x = vx * target_frame + bx
        pred_y = vy * target_frame + by
        return [pred_x, pred_y]

    def _distance(self, a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    def _stitch_clip_events(self, clip_path, events):
        merged = 0
        i = 0
        while i < len(events) - 1:
            ev_a = events[i]
            ev_b = events[i + 1]

            gap = ev_b['start_frame'] - ev_a['end_frame']
            if gap > self.max_gap_frames:
                i += 1
                continue

            track_a = self._get_track(ev_a)
            track_b = self._get_track(ev_b)
            if not track_a or not track_b:
                i += 1
                continue

            predicted = self._predict_position(track_a, ev_b['start_frame'])
            actual    = track_b[0][1]

            if self._distance(predicted, actual) > self.pos_tolerance:
                i += 1
                continue

            # Match — merge ev_a and ev_b
            print(f"[STITCH] Merging {os.path.basename(ev_a['event_dir'])} "
                  f"+ {os.path.basename(ev_b['event_dir'])} (gap={gap} frames)")
            merged_event = self._merge(ev_a, ev_b, clip_path)
            if merged_event:
                events[i] = merged_event
                events.pop(i + 1)
                merged += 1
            else:
                i += 1

        return merged

    def _merge(self, ev_a, ev_b, clip_path):
        ts = datetime.now().strftime("%H%M%S")
        src = os.path.splitext(os.path.basename(clip_path))[0]
        fps = ev_a.get('fps', 20)
        start_sec = int(ev_a['start_frame'] / fps)
        mm = start_sec // 60; ss = start_sec % 60
        merged_dir = os.path.join(self.events_folder,
                                  f"event_{src}_{mm:02d}m{ss:02d}s_stitched_{ts}")
        os.makedirs(merged_dir, exist_ok=True)

        clip_a = os.path.join(ev_a['event_dir'], 'clip.mkv')
        clip_b = os.path.join(ev_b['event_dir'], 'clip.mkv')
        if not os.path.exists(clip_a) or not os.path.exists(clip_b):
            return None

        gap_start        = ev_a['end_frame'] + 1
        gap_end          = ev_b['start_frame'] - 1
        gap_frame_indices = list(range(gap_start, gap_end + 1))

        # Extract gap frames from original recording as a temp MKV
        gap_clip = None
        if gap_frame_indices and os.path.exists(clip_path):
            gap_clip = os.path.join(merged_dir, '_gap.mkv')
            self._extract_gap_mkv(clip_path, clip_a, gap_start,
                                   len(gap_frame_indices), fps, gap_clip)
            if not os.path.exists(gap_clip) or os.path.getsize(gap_clip) == 0:
                gap_clip = None

        # Concatenate: ev_a clip + gap + ev_b clip → merged clip.mkv
        concat_parts = [clip_a]
        if gap_clip:
            concat_parts.append(gap_clip)
        concat_parts.append(clip_b)

        concat_txt = os.path.join(merged_dir, '_concat.txt')
        with open(concat_txt, 'w', encoding='utf-8') as f:
            for p in concat_parts:
                f.write(f"file '{p.replace(chr(92), chr(47))}'\n")

        merged_clip = os.path.join(merged_dir, 'clip.mkv')
        try:
            subprocess.run(
                [self.ffmpeg_path, '-y', '-f', 'concat', '-safe', '0',
                 '-i', concat_txt, '-c', 'copy', merged_clip],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=NO_WINDOW)
        except Exception:
            pass

        for tmp in [concat_txt, gap_clip]:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

        # Mark gap frame indices
        if gap_frame_indices:
            with open(os.path.join(merged_dir, '_gap_frames.txt'), 'w') as f:
                f.write('\n'.join(str(i) for i in gap_frame_indices))

        # Combined metadata
        combined_dets = (ev_a['detections'] +
                         [{'frame': i, 'centroids': [], 'bboxes': [], 'count': 0}
                          for i in gap_frame_indices] +
                         ev_b['detections'])
        meta = {
            'source_clip': clip_path,
            'start_frame': ev_a['start_frame'],
            'end_frame':   ev_b['end_frame'],
            'fps':         fps,
            'stitched':    True,
            'detections':  combined_dets,
            'event_dir':   merged_dir,
        }
        with open(os.path.join(merged_dir, 'metadata.json'), 'w') as f:
            json.dump({k: v for k, v in meta.items() if k != 'event_dir'}, f, indent=2)

        if os.path.exists(merged_clip):
            thumb = self._make_thumbnail_from_mkv(merged_clip)
            if thumb is not None:
                cv2.imwrite(os.path.join(merged_dir, '_thumbnail.jpg'), thumb)

            import shutil
            for orig_dir in (ev_a['event_dir'], ev_b['event_dir']):
                try:
                    shutil.rmtree(orig_dir)
                except Exception as e:
                    print(f'[STITCH] Could not remove {orig_dir}: {e}')

        return meta

    def _extract_gap_mkv(self, src_clip, ref_clip, start_frame, frame_count, fps, out_path):
        """Extract frame_count frames from src_clip at start_frame into out_path (MKV).
        Matches resolution and grayscale encoding of ref_clip (an existing event clip.mkv)."""
        cap = cv2.VideoCapture(ref_clip)
        if not cap.isOpened():
            return
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if fw == 0 or fh == 0:
            return

        start_sec = start_frame / fps
        reader = None
        writer = None
        try:
            reader = subprocess.Popen(
                [self.ffmpeg_path, *HWACCEL_ARGS,
                 '-ss', f'{start_sec:.3f}', '-i', src_clip,
                 '-frames:v', str(frame_count),
                 '-vf', f'scale={fw}:{fh},format=gray',
                 '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                creationflags=NO_WINDOW)
            writer = subprocess.Popen(
                [self.ffmpeg_path, '-y',
                 '-f', 'rawvideo', '-pix_fmt', 'gray',
                 '-s', f'{fw}x{fh}', '-r', str(fps), '-i', 'pipe:0',
                 '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                 '-pix_fmt', 'yuv420p', out_path],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
            frame_bytes = fw * fh
            for _ in range(frame_count):
                raw = reader.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                writer.stdin.write(raw)
            reader.stdout.close()
            reader.wait(timeout=15)
            writer.stdin.close()
            writer.wait(timeout=30)
        except Exception:
            pass
        finally:
            for p in [reader, writer]:
                if p is not None:
                    try:
                        p.kill()
                    except Exception:
                        pass

    def _make_thumbnail_from_mkv(self, mkv_path, max_frames=300):
        """Decode up to max_frames from mkv_path and return a max-blend thumbnail."""
        cap = cv2.VideoCapture(mkv_path)
        if not cap.isOpened():
            return None
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if fw == 0 or fh == 0:
            return None

        gray_frames = []
        proc = None
        try:
            proc = subprocess.Popen(
                [self.ffmpeg_path, '-i', mkv_path,
                 '-vframes', str(max_frames),
                 '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                creationflags=NO_WINDOW)
            frame_bytes = fw * fh
            while True:
                raw = proc.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                gray_frames.append(
                    np.frombuffer(raw, dtype=np.uint8).reshape(fh, fw))
            proc.stdout.close()
            proc.wait(timeout=15)
        except Exception:
            pass
        finally:
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
        return make_thumbnail(gray_frames) if gray_frames else None


# ==============================================================================
# Main Application
# ==============================================================================

class StreakerDetectApp:
    def __init__(self, master):
        self.root = master
        master.title(f"StreakerDetect  v{VERSION}")
        master.configure(bg='#111111')

        self.input_path      = tk.StringVar()
        self.mask_path       = tk.StringVar()
        self.dark_frame_path = tk.StringVar()
        self.fa_key          = tk.StringVar()
        self.output_dir      = tk.StringVar()
        self._cameras    = []        # list of profile dicts
        self._camera_var = tk.StringVar()

        self.p_history   = tk.IntVar(value=DEFAULT_MOG2_HISTORY)
        self.p_threshold = tk.IntVar(value=DEFAULT_MOG2_THRESHOLD)
        self.p_min_area  = tk.IntVar(value=DEFAULT_MIN_CONTOUR_AREA)
        self.p_max_area  = tk.IntVar(value=DEFAULT_MAX_CONTOUR_AREA)
        self.p_min_asp   = tk.DoubleVar(value=DEFAULT_MIN_ASPECT_RATIO)
        self.p_max_track      = tk.IntVar(value=DEFAULT_MAX_TRACK_FRAMES)
        self.p_max_match_dist = tk.IntVar(value=DEFAULT_MAX_MATCH_DIST)
        self.p_pre_buf   = tk.IntVar(value=DEFAULT_PRE_BUFFER)
        self.p_post_buf  = tk.IntVar(value=DEFAULT_POST_BUFFER)
        self.p_warmup    = tk.IntVar(value=DEFAULT_WARMUP_FRAMES)
        self.p_cld_thr   = tk.IntVar(value=DEFAULT_CLOUD_THRESH)
        self.p_cld_rat   = tk.DoubleVar(value=DEFAULT_CLOUD_RATIO)
        self.p_scale     = tk.DoubleVar(value=0.5)
        self.p_stitch_gap   = tk.IntVar(value=300)
        self.p_stitch_tol   = tk.IntVar(value=80)
        self.p_min_move     = tk.IntVar(value=0)
        self.p_min_travel       = tk.IntVar(value=0)
        self.p_min_bright       = tk.IntVar(value=0)
        self.p_cloud_min_bright = tk.IntVar(value=0)
        self.p_cloud_min_travel = tk.IntVar(value=0)
        self.p_force_rerun      = tk.BooleanVar(value=False)
        self.p_no_hwaccel       = tk.BooleanVar(value=False)
        self.p_auto_stitch      = tk.BooleanVar(value=False)

        self.preview_q = queue.Queue(maxsize=2)
        self.event_q   = queue.Queue()
        self.done_q    = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None

        # Embedded player state
        self._player_frames      = []
        self._player_frame_cache = []   # pre-loaded RGB arrays indexed by frame position
        self._player_composite   = None
        self._player_idx         = 0
        self._player_paused      = True
        self._canvas_mode        = 'detect'  # 'detect' | 'player'
        self._player_speed_ms    = 80
        self._player_show_comp   = False
        self._player_loop_id     = None
        self._player_event_dir   = None
        self._player_mkv_path    = None   # set when event was saved as MKV
        self._play_clock_start   = 0.0
        self._play_clock_frame   = 0
        self._player_scrubbing   = False
        self._player_fps       = 20.0
        self._player_tester_clip = None

        self._build_ui()
        self._load_config()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_queues()
        if getattr(sys, 'frozen', False):
            self.root.after(4000, self._check_for_update)

    # --------------------------------------------------------------------------
    # Config persistence
    # --------------------------------------------------------------------------

    def _load_config(self):
        self._load_cameras()
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH) as f:
                    c = json.load(f)
                if c.get('mask_path') and os.path.exists(c['mask_path']):
                    self.mask_path.set(c['mask_path'])
                if c.get('dark_frame_path') and os.path.exists(c['dark_frame_path']):
                    self.dark_frame_path.set(c['dark_frame_path'])
                if c.get('output_dir') and os.path.exists(c['output_dir']):
                    self.output_dir.set(c['output_dir'])
                self.p_threshold.set(c.get('threshold',  DEFAULT_MOG2_THRESHOLD))
                self.p_min_area.set(c.get('min_area',    DEFAULT_MIN_CONTOUR_AREA))
                self.p_max_area.set(c.get('max_area',    DEFAULT_MAX_CONTOUR_AREA))
                self.p_min_asp.set(c.get('min_aspect',   DEFAULT_MIN_ASPECT_RATIO))
                self.p_max_track.set(c.get('max_track',       DEFAULT_MAX_TRACK_FRAMES))
                self.p_max_match_dist.set(c.get('max_match_dist', DEFAULT_MAX_MATCH_DIST))
                self.p_pre_buf.set(c.get('pre_buffer',   DEFAULT_PRE_BUFFER))
                self.p_post_buf.set(c.get('post_buffer', DEFAULT_POST_BUFFER))
                self.p_warmup.set(c.get('warmup',        DEFAULT_WARMUP_FRAMES))
                self.p_cld_thr.set(c.get('cloud_thresh',  DEFAULT_CLOUD_THRESH))
                self.p_cld_rat.set(c.get('cloud_ratio',   DEFAULT_CLOUD_RATIO))
                self.p_scale.set(c.get('scale',           0.5))
                self.p_stitch_gap.set(c.get('stitch_gap', 300))
                self.p_stitch_tol.set(c.get('stitch_tol', 80))
                self.p_min_move.set(c.get('min_move', 0))
                self.p_min_travel.set(c.get('min_travel', 0))
                self.p_min_bright.set(c.get('min_bright', 0))
                self.p_cloud_min_bright.set(c.get('cloud_min_bright', 0))
                self.p_cloud_min_travel.set(c.get('cloud_min_travel', 0))
        except Exception:
            pass
        try:
            if os.path.exists(SHARED_CONFIG_PATH):
                with open(SHARED_CONFIG_PATH) as f:
                    sc = json.load(f)
                if sc.get('flightaware_api_key'):
                    self.fa_key.set(sc['flightaware_api_key'])
        except Exception:
            pass

    def _save_config(self):
        try:
            c = {
                'mask_path':        self.mask_path.get(),
                'dark_frame_path':  self.dark_frame_path.get(),
                'output_dir':       self.output_dir.get(),
                'threshold':   self.p_threshold.get(),
                'min_area':    self.p_min_area.get(),
                'max_area':    self.p_max_area.get(),
                'min_aspect':  self.p_min_asp.get(),
                'max_track':      self.p_max_track.get(),
                'max_match_dist': self.p_max_match_dist.get(),
                'pre_buffer':  self.p_pre_buf.get(),
                'post_buffer': self.p_post_buf.get(),
                'warmup':      self.p_warmup.get(),
                'cloud_thresh':self.p_cld_thr.get(),
                'cloud_ratio': self.p_cld_rat.get(),
                'scale':       self.p_scale.get(),
                'stitch_gap':  self.p_stitch_gap.get(),
                'stitch_tol':  self.p_stitch_tol.get(),
                'min_move':         self.p_min_move.get(),
                'min_travel':       self.p_min_travel.get(),
                'min_bright':       self.p_min_bright.get(),
                'cloud_min_bright': self.p_cloud_min_bright.get(),
                'cloud_min_travel': self.p_cloud_min_travel.get(),
            }
            with open(CONFIG_PATH, 'w') as f:
                json.dump(c, f, indent=2)
        except Exception:
            pass
        try:
            sc = {}
            if os.path.exists(SHARED_CONFIG_PATH):
                with open(SHARED_CONFIG_PATH) as f:
                    sc = json.load(f)
            sc['flightaware_api_key'] = self.fa_key.get().strip()
            with open(SHARED_CONFIG_PATH, 'w') as f:
                json.dump(sc, f, indent=2)
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # Camera profiles
    # --------------------------------------------------------------------------

    def _load_cameras(self):
        if not os.path.exists(CAMERAS_PATH):
            return
        try:
            with open(CAMERAS_PATH) as f:
                data = json.load(f)
            self._cameras = data.get('cameras', [])
            self._camera_combo['values'] = [c['name'] for c in self._cameras]
            last = data.get('last_used', '')
            if last and last in [c['name'] for c in self._cameras]:
                self._camera_var.set(last)
                self._apply_camera_profile(last)
        except Exception:
            pass

    def _save_cameras(self):
        try:
            with open(CAMERAS_PATH, 'w') as f:
                json.dump({'cameras': self._cameras,
                           'last_used': self._camera_var.get()}, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def _newest_date_folder(base_dir):
        """Return the newest MM-DD-YYYY subfolder in base_dir, or None."""
        import re
        pat = re.compile(r'^\d{2}-\d{2}-\d{4}$')
        try:
            subs = [d for d in os.listdir(base_dir)
                    if pat.match(d) and os.path.isdir(os.path.join(base_dir, d))]
        except Exception:
            return None
        if not subs:
            return None
        def _date_key(name):
            try:
                m, d, y = name.split('-')
                return (int(y), int(m), int(d))
            except Exception:
                return (0, 0, 0)
        return os.path.join(base_dir, max(subs, key=_date_key))

    def _apply_camera_profile(self, name):
        for cam in self._cameras:
            if cam['name'] == name:
                if cam.get('input_dir'):
                    newest = self._newest_date_folder(cam['input_dir'])
                    self.input_path.set(newest or cam['input_dir'])
                if cam.get('output_dir'):
                    self.output_dir.set(cam['output_dir'])
                if cam.get('mask_path'):
                    self.mask_path.set(cam['mask_path'])
                return

    def _camera_on_select(self, _event=None):
        name = self._camera_var.get()
        self._apply_camera_profile(name)
        self._save_cameras()

    def _camera_add(self):
        self._camera_dialog("Add Camera")

    def _camera_edit(self):
        name = self._camera_var.get()
        if not name:
            messagebox.showwarning("No Selection", "Select a camera profile to edit.")
            return
        profile = next((c for c in self._cameras if c['name'] == name), None)
        self._camera_dialog("Edit Camera", profile)

    def _camera_delete(self):
        name = self._camera_var.get()
        if not name:
            messagebox.showwarning("No Selection", "Select a camera profile to delete.")
            return
        if not messagebox.askyesno("Delete Camera",
                                   f"Delete profile '{name}'?"):
            return
        self._cameras = [c for c in self._cameras if c['name'] != name]
        self._camera_var.set('')
        self._camera_combo['values'] = [c['name'] for c in self._cameras]
        self._save_cameras()

    def _camera_dialog(self, title, profile=None):
        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.configure(bg='#1e1e1e')
        dlg.resizable(False, False)
        dlg.grab_set()

        BG, FG = '#1e1e1e', '#dddddd'
        pad = dict(padx=8, pady=4)

        def lbl(text):
            tk.Label(dlg, text=text, bg=BG, fg='#888888',
                     font=('Arial', 8)).pack(anchor='w', padx=10, pady=(6, 0))

        def entry_row(var):
            f = tk.Frame(dlg, bg=BG)
            f.pack(fill='x', padx=10, pady=(0, 2))
            e = tk.Entry(f, textvariable=var, bg='#2a2a2a', fg=FG,
                         insertbackground=FG, relief='flat', width=42)
            e.pack(side='left', fill='x', expand=True)
            return f, e

        def browse_dir(var):
            p = filedialog.askdirectory()
            if p:
                var.set(p)

        def browse_file(var):
            p = filedialog.askopenfilename(filetypes=[("PNG", "*.png"), ("All", "*.*")])
            if p:
                var.set(p)

        v_name   = tk.StringVar(value=profile['name']       if profile else '')
        v_input  = tk.StringVar(value=profile.get('input_dir',  '') if profile else '')
        v_output = tk.StringVar(value=profile.get('output_dir', '') if profile else '')
        v_mask   = tk.StringVar(value=profile.get('mask_path',  '') if profile else '')

        lbl("Camera name:")
        _, e_name = entry_row(v_name)
        e_name.focus_set()

        lbl("Input folder (where recordings land):")
        fr, _ = entry_row(v_input)
        tk.Button(fr, text="…", command=lambda: browse_dir(v_input),
                  bg='#444', fg=FG, relief='flat', width=2).pack(side='left', padx=(2, 0))

        lbl("Output folder (detection results):")
        fr, _ = entry_row(v_output)
        tk.Button(fr, text="…", command=lambda: browse_dir(v_output),
                  bg='#444', fg=FG, relief='flat', width=2).pack(side='left', padx=(2, 0))

        lbl("Mask path (optional):")
        fr, _ = entry_row(v_mask)
        tk.Button(fr, text="…", command=lambda: browse_file(v_mask),
                  bg='#444', fg=FG, relief='flat', width=2).pack(side='left', padx=(2, 0))
        def _launch_mask_editor():
            launch_companion('Mask_editor_gui.py')
        tk.Button(fr, text="✎ Edit", command=_launch_mask_editor,
                  bg='#444', fg=FG, relief='flat').pack(side='left', padx=(2, 0))

        def _save():
            name = v_name.get().strip()
            if not name:
                messagebox.showerror("Error", "Camera name is required.", parent=dlg)
                return
            new_profile = {
                'name':       name,
                'input_dir':  v_input.get().strip(),
                'output_dir': v_output.get().strip(),
                'mask_path':  v_mask.get().strip(),
            }
            if profile:
                # Replace existing
                for i, c in enumerate(self._cameras):
                    if c['name'] == profile['name']:
                        self._cameras[i] = new_profile
                        break
            else:
                if any(c['name'] == name for c in self._cameras):
                    messagebox.showerror("Error", f"A camera named '{name}' already exists.",
                                         parent=dlg)
                    return
                self._cameras.append(new_profile)
            self._camera_combo['values'] = [c['name'] for c in self._cameras]
            self._camera_var.set(name)
            self._apply_camera_profile(name)
            self._save_cameras()
            dlg.destroy()

        btn_f = tk.Frame(dlg, bg=BG)
        btn_f.pack(pady=10)
        tk.Button(btn_f, text="Save", command=_save,
                  bg='#0057a8', fg='white', relief='flat',
                  font=('Arial', 9, 'bold'), padx=12).pack(side='left', padx=6)
        tk.Button(btn_f, text="Cancel", command=dlg.destroy,
                  bg='#444', fg=FG, relief='flat', padx=8).pack(side='left')

        dlg.wait_window()

    def _on_close(self):
        self._save_config()
        self.root.destroy()

    # --------------------------------------------------------------------------
    # UI Construction
    # --------------------------------------------------------------------------

    def _build_ui(self):
        BG = '#111111'

        def sep(parent):
            ttk.Separator(parent, orient='vertical').pack(
                side='left', fill='y', padx=6, pady=3)

        def file_input(parent, label, var, cmd):
            f = tk.Frame(parent, bg=BG)
            f.pack(side='left', padx=3, pady=3)
            tk.Label(f, text=label, bg=BG, fg='#888888',
                     font=('Arial', 7)).pack(anchor='w')
            r = tk.Frame(f, bg=BG)
            r.pack()
            tk.Entry(r, textvariable=var, bg='#2a2a2a', fg='white',
                     relief='flat', width=30).pack(side='left')
            tk.Button(r, text="…", command=cmd, bg='#444444', fg='white',
                      relief='flat', width=2).pack(side='left', padx=(1, 0))

        def slider(parent, label, var, lo, hi, res, tip=None):
            f = tk.Frame(parent, bg=BG)
            f.pack(side='left', padx=3, pady=2)
            lbl = tk.Label(f, text=label, bg=BG, fg='white',
                           font=('Arial', 7))
            lbl.pack(anchor='w')
            sc = tk.Scale(f, from_=lo, to=hi, resolution=res, variable=var,
                          orient='horizontal', bg=BG, fg='white',
                          troughcolor='#333333', highlightthickness=0,
                          length=150, showvalue=True,
                          font=('Arial', 7))
            sc.pack()
            if tip:
                Tooltip(lbl, tip)
                Tooltip(sc, tip)

        # ── Outer split: left controls+canvas | right thumbnails ──────────
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill='both', expand=True)

        # Thumbnails — packed first so it claims full height on the right
        self.thumb_col = tk.Frame(outer, bg='#1a1a1a', width=360)
        self.thumb_col.pack(side='right', fill='y', padx=(2, 4), pady=0)
        self.thumb_col.pack_propagate(False)

        # Collapse toggle button at top of the panel
        _thumb_hdr = tk.Frame(self.thumb_col, bg='#111111')
        _thumb_hdr.pack(fill='x', side='top')
        self._thumb_toggle_btn = tk.Button(
            _thumb_hdr, text="◀ hide events", command=self._toggle_thumb_panel,
            bg='#2a2a2a', fg='#cccccc', relief='flat',
            font=('Arial', 8), padx=8, pady=3, cursor='hand2')
        self._thumb_toggle_btn.pack(side='left')

        # Content wrapper — what gets hidden on collapse
        self._thumb_content = tk.Frame(self.thumb_col, bg='#1a1a1a')
        self._thumb_content.pack(fill='both', expand=True)
        self.thumb_panel = ThumbnailPanel(self._thumb_content, self._open_event_viewer,
                                          on_view_clip=self._view_clip_in_player)
        self._thumb_visible = True

        # Left section — rows + canvas
        left_section = tk.Frame(outer, bg=BG)
        left_section.pack(side='left', fill='both', expand=True)

        # ── Camera profile row ────────────────────────────────────────────────
        r0 = tk.Frame(left_section, bg=BG)
        r0.pack(fill='x', side='top', padx=6, pady=(3, 0))
        tk.Label(r0, text="Camera:", bg=BG, fg='#888888',
                 font=('Arial', 8)).pack(side='left', padx=(0, 4))
        self._camera_combo = ttk.Combobox(r0, textvariable=self._camera_var,
                                          state='readonly', width=28, font=('Arial', 9))
        self._camera_combo.pack(side='left', padx=(0, 4))
        self._camera_combo.bind('<<ComboboxSelected>>', self._camera_on_select)
        tk.Button(r0, text="+ Add",   command=self._camera_add,
                  bg='#226622', fg='white', relief='flat',
                  font=('Arial', 8), padx=4).pack(side='left', padx=2)
        tk.Button(r0, text="✎ Edit",  command=self._camera_edit,
                  bg='#444444', fg='white', relief='flat',
                  font=('Arial', 8), padx=4).pack(side='left', padx=2)
        tk.Button(r0, text="🗑 Delete", command=self._camera_delete,
                  bg='#662222', fg='white', relief='flat',
                  font=('Arial', 8), padx=4).pack(side='left', padx=2)

        # ── Row 1 sub-row A: title | input + output | run/stop/stitch/open ─
        r1a = tk.Frame(left_section, bg=BG)
        r1a.pack(fill='x', side='top')

        tk.Label(r1a, text="STREAKER DETECT", bg=BG, fg='white',
                 font=('Arial', 11, 'bold')).pack(side='left', padx=10, pady=4)
        sep(r1a)

        # Input path
        inp_f = tk.Frame(r1a, bg=BG)
        inp_f.pack(side='left', padx=3, pady=3)
        tk.Label(inp_f, text="Input (MKV or folder)", bg=BG, fg='#888888',
                 font=('Arial', 7)).pack(anchor='w')
        inp_r = tk.Frame(inp_f, bg=BG)
        inp_r.pack()
        tk.Entry(inp_r, textvariable=self.input_path, bg='#2a2a2a', fg='white',
                 relief='flat', width=30).pack(side='left')

        tk.Button(inp_r, text="…", command=self._browse_input, bg='#444444', fg='white',
                  relief='flat', width=2).pack(side='left', padx=(1, 0))
        tk.Button(inp_r, text="📁", command=self._browse_input_folder, bg='#444444', fg='white',
                  relief='flat', width=2).pack(side='left', padx=(1, 0))
        self.input_path.trace_add('write', lambda *_: self.root.after(200, self._queue_populate_from_var))

        file_input(r1a, "Output folder", self.output_dir, self._browse_output)
        sep(r1a)

        btn_f = tk.Frame(r1a, bg=BG)
        btn_f.pack(side='left', padx=4, pady=2)
        _cb_force = tk.Checkbutton(btn_f, text="Force Re-run", variable=self.p_force_rerun,
                                   bg=BG, fg='#aaaaaa', selectcolor='#333333',
                                   font=('Arial', 8), activebackground=BG)
        _cb_force.pack(side='left', padx=(2, 6))
        Tooltip(_cb_force, "Re-process clips that were already completed. "
                "Also clears any partial checkpoints so clips restart from frame 0.")
        _cb_sw = tk.Checkbutton(btn_f, text="SW Decode", variable=self.p_no_hwaccel,
                                bg=BG, fg='#aaaaaa', selectcolor='#333333',
                                font=('Arial', 8), activebackground=BG)
        _cb_sw.pack(side='left', padx=(0, 6))
        Tooltip(_cb_sw, "Use software (CPU) decoding instead of NVIDIA CUDA hardware decoding. "
                "Enable if you see phantom streaks or corrupted frames in event clips.")
        self.run_btn = tk.Button(btn_f, text="▶ RUN",
                                 command=self._start_detection,
                                 bg='#006600', fg='white',
                                 font=('Arial', 9, 'bold'),
                                 relief='flat', padx=10, pady=3)
        self.run_btn.pack(side='left', padx=2)
        Tooltip(self.run_btn, "Start detection on the selected input folder or MKV clip.")
        self.stop_btn = tk.Button(btn_f, text="■ STOP",
                                  command=self._stop_detection,
                                  bg='#660000', fg='white',
                                  font=('Arial', 9, 'bold'),
                                  relief='flat', padx=10, pady=3,
                                  state='disabled')
        self.stop_btn.pack(side='left', padx=2)
        Tooltip(self.stop_btn, "Stop detection after the current clip finishes. "
                "Progress is saved — you can resume from where it stopped next time.")
        _btn_stitch = tk.Button(btn_f, text="🔗 STITCH",
                                command=self._run_stitcher,
                                bg='#334455', fg='white',
                                font=('Arial', 9, 'bold'),
                                relief='flat', padx=10, pady=3)
        _btn_stitch.pack(side='left', padx=2)
        Tooltip(_btn_stitch, "Merge nearby event clips from the same object into one. "
                "Useful when a plane or satellite was split into multiple short clips.")
        _auto_cb = tk.Checkbutton(btn_f, text="auto", variable=self.p_auto_stitch,
                                   bg=BG, fg='#888888', selectcolor='#334455',
                                   activebackground=BG, activeforeground='#dddddd',
                                   font=('Arial', 8), cursor='hand2')
        _auto_cb.pack(side='left', padx=(0, 6))
        Tooltip(_auto_cb, "Automatically stitch events after each detection run completes.")
        self._btn_identify = tk.Button(btn_f, text="🔎 IDENTIFY",
                                   command=self._run_identify,
                                   bg='#1a3a1a', fg='white',
                                   font=('Arial', 9, 'bold'),
                                   relief='flat', padx=10, pady=3)
        self._btn_identify.pack(side='left', padx=2)
        Tooltip(self._btn_identify,
                "Run the identification pass on a detect folder — classifies each event "
                "as meteor / aircraft / satellite / unknown and flags anomalies. "
                "Writes identification.json per event.")
        self._identify_cancel = threading.Event()
        _btn_open = tk.Button(btn_f, text="📂 OPEN EVENTS",
                              command=self._open_events_folder,
                              bg='#1a4a6a', fg='white',
                              font=('Arial', 9, 'bold'),
                              relief='flat', padx=10, pady=3)
        _btn_open.pack(side='left', padx=2)
        Tooltip(_btn_open, "Open the current detection output folder in Windows Explorer.")

        _btn_mask_ed = tk.Button(btn_f, text="✏ MASK",
                                 command=self._open_mask_editor,
                                 bg='#2a2a2a', fg='#cccccc',
                                 font=('Arial', 9, 'bold'),
                                 relief='flat', padx=10, pady=3)
        _btn_mask_ed.pack(side='left', padx=2)
        Tooltip(_btn_mask_ed, "Open Mask Editor — draw regions to exclude from detection.")
        self._events_folder_var     = tk.StringVar(value="")
        self._events_folder_full    = ""
        _folder_lbl = tk.Label(btn_f, textvariable=self._events_folder_var, bg=BG,
                               fg="#0A66C3", font=('Consolas', 10))
        _folder_lbl.pack(side='left', padx=(6, 0))
        self._events_folder_tip = Tooltip(_folder_lbl, "")

        # ── Row 1 sub-row B: queue/mask | force re-run/logs/player/synth/compare ─
        r1b = tk.Frame(left_section, bg=BG)
        r1b.pack(fill='x', side='top')

        # MKV queue panel
        q_frame = tk.Frame(r1b, bg=BG)
        q_frame.pack(side='left', padx=3, pady=(0, 3))
        self._queue_count_lbl = tk.Label(q_frame, text="", bg=BG, fg='#888888',
                                         font=('Arial', 7))
        self._queue_count_lbl.pack(anchor='w')
        q_inner = tk.Frame(q_frame, bg=BG)
        q_inner.pack(fill='x')
        self._queue_listbox = tk.Listbox(q_inner, height=2, width=55, selectmode='extended',
                                         bg='#0d1b2a', fg='white',
                                         selectbackground='#1a3a5a',
                                         font=('Consolas', 8), relief='flat',
                                         highlightthickness=1,
                                         highlightbackground='#1a3a5a')
        q_sb = tk.Scrollbar(q_inner, orient='vertical', command=self._queue_listbox.yview)
        self._queue_listbox.config(yscrollcommand=q_sb.set)
        self._queue_listbox.pack(side='left', fill='x', expand=True)
        q_sb.pack(side='left', fill='y')
        tk.Button(q_inner, text="✕ Remove", command=self._queue_remove_selected,
                  bg='#2a1a1a', fg='#ff6666',
                  font=('Arial', 8), relief='flat', padx=6).pack(side='left', padx=(4, 0))
        tk.Button(q_inner, text="+ Add", command=self._queue_add_files,
                  bg='#1a2a1a', fg='#66ff66',
                  font=('Arial', 8), relief='flat', padx=6).pack(side='left', padx=(2, 0))

        file_input(r1b, "Mask (optional)", self.mask_path, self._browse_mask)

        # Dark frame controls
        df_f = tk.Frame(r1b, bg=BG)
        df_f.pack(side='left', padx=3, pady=3)
        tk.Label(df_f, text="Dark frame (.npy)", bg=BG, fg='#888888',
                 font=('Arial', 7)).pack(anchor='w')
        df_r = tk.Frame(df_f, bg=BG)
        df_r.pack()
        tk.Entry(df_r, textvariable=self.dark_frame_path, bg='#2a2a2a', fg='white',
                 relief='flat', width=26).pack(side='left')
        tk.Button(df_r, text="…",   command=self._browse_dark_frame,
                  bg='#444444', fg='white', relief='flat', width=2).pack(side='left', padx=(1, 0))
        tk.Button(df_r, text="MKV", command=self._build_dark_from_mkv,
                  bg='#334455', fg='white', relief='flat', width=4).pack(side='left', padx=(1, 0))
        tk.Button(df_r, text="CAM", command=self._capture_dark_from_rtsp,
                  bg='#334433', fg='white', relief='flat', width=4).pack(side='left', padx=(1, 0))

        # FlightAware API key
        fa_f = tk.Frame(r1b, bg=BG)
        fa_f.pack(side='left', padx=3, pady=3)
        tk.Label(fa_f, text="FlightAware key", bg=BG, fg='#888888',
                 font=('Arial', 7)).pack(anchor='w')
        tk.Entry(fa_f, textvariable=self.fa_key, bg='#2a2a2a', fg='white',
                 relief='flat', width=28, show='*').pack()

        sep(r1b)

        btn_f2 = tk.Frame(r1b, bg=BG)
        btn_f2.pack(side='left', padx=4, pady=2)
        _btn_logs = tk.Button(btn_f2, text="📋 VIEW LOGS",
                              command=self._show_logs_popup,
                              bg='#443300', fg='white',
                              font=('Arial', 9, 'bold'),
                              relief='flat', padx=10, pady=3)
        _btn_logs.pack(side='left', padx=2)
        Tooltip(_btn_logs, "View detection logs from previous runs — "
                "shows what was found, frame counts, cloud stats, and any errors.")
        _btn_player = tk.Button(btn_f2, text="📺 PLAYER",
                                command=self._launch_player,
                                bg='#2a1a4a', fg='white',
                                font=('Arial', 9, 'bold'),
                                relief='flat', padx=10, pady=3,
                                state='normal' if _PLAYER_AVAILABLE else 'disabled')
        _btn_player.pack(side='left', padx=2)
        Tooltip(_btn_player, "Open the MKV timeline player to review full recording clips "
                "with color-coded event markers (cyan=meteor, green=plane, magenta=satellite).")
        _btn_synth = tk.Button(btn_f2, text="🧪 SYNTH TEST",
                               command=self._launch_synth_test,
                               bg='#1a3a1a', fg='white',
                               font=('Arial', 9, 'bold'),
                               relief='flat', padx=10, pady=3)
        _btn_synth.pack(side='left', padx=2)
        Tooltip(_btn_synth, "Run a synthetic test: injects a fake streak into a blank frame "
                "to verify the current detection settings can find it.")
        _btn_demo = tk.Button(btn_f2, text="🎛 DEMO",
                              command=self._launch_demo,
                              bg='#2a2a1a', fg='white',
                              font=('Arial', 9, 'bold'),
                              relief='flat', padx=10, pady=3)
        _btn_demo.pack(side='left', padx=2)
        Tooltip(_btn_demo, "Interactive filter demo: drag sliders and see in real-time "
                "which synthetic objects (meteor, cloud, noise) pass or fail each filter.")
        _btn_compare = tk.Button(btn_f2, text="⚖ COMPARE",
                                 command=self._launch_compare,
                                 bg='#1a2a4a', fg='white',
                                 font=('Arial', 9, 'bold'),
                                 relief='flat', padx=10, pady=3)
        _btn_compare.pack(side='left', padx=2)
        Tooltip(_btn_compare, "Side-by-side comparison of two detection runs — "
                "useful for seeing how changing settings affects what gets detected.")
        tk.Label(btn_f2, text="│", bg=BG, fg='#444').pack(side='left', padx=4)
        _btn_sync = tk.Button(btn_f2, text="📡 SYNC",
                              command=self._launch_sync,
                              bg='#1a3a4a', fg='#88ddff',
                              font=('Arial', 9, 'bold'),
                              relief='flat', padx=10, pady=3)
        _btn_sync.pack(side='left', padx=2)
        Tooltip(_btn_sync, "StreakerSync — scrub each camera to a GPS clock reference "
                "frame and enter the UTC time shown, for post-hoc multi-camera alignment.")
        _btn_match = tk.Button(btn_f2, text="🔗 MATCH",
                               command=self._launch_match,
                               bg='#1a3a4a', fg='#88ddff',
                               font=('Arial', 9, 'bold'),
                               relief='flat', padx=10, pady=3)
        _btn_match.pack(side='left', padx=2)
        Tooltip(_btn_match, "StreakerMatch — compare events across cameras and group "
                "ones that occurred at the same time for triangulation.")

        self.stats_var = tk.StringVar(value="")

        self._update_btn = tk.Button(btn_f2, text="",
                                     bg=BG, fg='#ffaa00',
                                     font=('Arial', 8, 'bold'), relief='flat',
                                     padx=6)
        # not packed — only shown when an update is available

        # ── Row 2: detection & tracking sliders ───────────────────────────
        row2 = tk.Frame(left_section, bg=BG)
        row2.pack(fill='x', side='top')

        for args in [
            ("MOG2 Thresh",   self.p_threshold,      10,  150,    1,
             "Background sensitivity. Lower = more sensitive, more false positives. "
             "Raise if stars or noise are triggering detections."),
            ("Min Area",      self.p_min_area,        10,  500,   10,
             "Minimum contour size in pixels² to be considered a detection. "
             "Raise to ignore tiny noise specks."),
            ("Max Area",      self.p_max_area,       500, 20000,  100,
             "Maximum contour size in pixels². Detections larger than this are ignored. "
             "Raise if large meteors are being missed."),
            ("Aspect Ratio",  self.p_min_asp,        1.0,   8.0,  0.1,
             "Minimum elongation of a detection (length ÷ width). "
             "Higher values require streakier shapes, filtering out round blobs."),
            ("Max Track Fr",  self.p_max_track,        1,    30,    1,
             "How many frames a track can stay 'alive' without a new detection before it ends. "
             "Higher values keep slow-moving objects tracked longer."),
            ("Match Dist px", self.p_max_match_dist,   0,   200,    5,
             "Fallback match distance in pixels. When two detections don't overlap at all, "
             "they can still be linked as the same object if their centres are within this distance. "
             "Set to 0 to disable. Useful for fast-moving planes and satellites."),
            ("Pre Buffer",    self.p_pre_buf,           5,   120,    5,
             "Frames to include BEFORE a detection fires. Captures the approach of the object."),
            ("Post Buffer",   self.p_post_buf,          5,   120,    5,
             "Frames to include AFTER the last detection. Captures the object leaving the frame."),
            ("Min Bright",    self.p_min_bright,        0,   255,    5,
             "Minimum peak pixel brightness for a detection to be saved. "
             "Raise to filter out dim noise that passes shape filters."),
        ]:
            slider(row2, *args)

        # ── Row 3: cloud, filter & stitch sliders ─────────────────────────
        row3 = tk.Frame(left_section, bg=BG)
        row3.pack(fill='x', side='top')

        for args in [
            ("Warmup Fr",     self.p_warmup,      50,   500,   50,
             "Frames to skip at the start of each clip while the background model learns. "
             "During warmup, detections are suppressed to avoid false triggers."),
            ("Cloud Sens",    self.p_cld_thr,     20,   200,    5,
             "Sensitivity of the cloud detector (sigma multiplier). "
             "Lower = easier to declare a frame cloudy. Raise if clear-sky frames are being suppressed."),
            ("Cloud Ratio",   self.p_cld_rat,   0.01,  0.50, 0.01,
             "Fraction of active pixels that triggers cloud suppression. "
             "e.g. 0.15 = suppress the frame if more than 15%% of pixels are 'active'."),
            ("Min Move px",   self.p_min_move,     0,    50,    1,
             "Minimum movement of a detection between frames to count as real. "
             "Filters out stationary hot pixels."),
            ("Min Travel px", self.p_min_travel,   0,   100,    5,
             "Minimum total distance a track must travel (start to end) to be saved as an event. "
             "Filters out flickering pixels that don't go anywhere."),
            ("Stitch Gap",    self.p_stitch_gap,   0,  1000,   10,
             "Maximum gap in frames between two clips to be stitched together into one event. "
             "Set to 0 to disable stitching."),
            ("Stitch Tol",    self.p_stitch_tol,   5,   300,    5,
             "Maximum pixel distance between the end of one clip and the start of the next "
             "for them to be considered the same object during stitching."),
        ]:
            slider(row3, *args)

        # Scale dropdown (row 3)
        sf = tk.Frame(row3, bg=BG)
        sf.pack(side='left', padx=8, pady=2)
        tk.Label(sf, text="Detect Scale", bg=BG, fg='white',
                 font=('Arial', 7)).pack(anchor='w')
        scale_menu = ttk.Combobox(sf, textvariable=self.p_scale,
                                  values=[1.0, 0.75, 0.5, 0.25],
                                  width=5, state='readonly')
        scale_menu.pack()
        Tooltip(scale_menu, "Downscale frames before detection. "
                "0.5 = half resolution, runs ~4× faster with slightly reduced sensitivity. "
                "Use 1.0 for maximum accuracy on faint meteors.")

        # ── Row 4: adaptive cloud overrides ───────────────────────────────
        row4 = tk.Frame(left_section, bg=BG)
        row4.pack(fill='x', side='top')
        tk.Label(row4, text="ADAPTIVE CLOUD:", bg=BG, fg='#5599bb',
                 font=('Arial', 7, 'bold')).pack(side='left', padx=(8, 4), pady=2)
        tk.Label(row4, text="When ≥30% of recent frames are cloudy, override these filters:",
                 bg=BG, fg='#888888', font=('Arial', 7)).pack(side='left', padx=(0, 8))
        for args in [
            ("Cloud Min Bright", self.p_cloud_min_bright, 0, 255, 5,
             "When ≥30%% of recent frames are cloudy, override Min Bright with this value. "
             "Set higher than normal Min Bright to reject dim cloud-lit false positives."),
            ("Cloud Min Travel", self.p_cloud_min_travel, 0, 100, 5,
             "When ≥30%% of recent frames are cloudy, override Min Travel with this value. "
             "Set higher to require more movement before saving a cloudy-sky event."),
        ]:
            slider(row4, *args)

        # ── Progress bar ───────────────────────────────────────────────────
        prog = tk.Frame(left_section, bg='#0a0a0a', pady=1)
        prog.pack(fill='x', side='top')
        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(prog, variable=self.progress_var,
                        maximum=100).pack(fill='x', padx=6, pady=1)
        self._active_folder_var = tk.StringVar(value="")
        tk.Label(prog, textvariable=self._active_folder_var, bg='#0a0a0a',
                 fg='#557799', font=('Consolas', 7), anchor='w').pack(fill='x', padx=6)
        self.progress_lbl = tk.StringVar(value="Ready")
        tk.Label(prog, textvariable=self.progress_lbl, bg='#0a0a0a',
                 fg='#aaaaaa', font=('Arial', 8)).pack()

        # ── Main area: preview canvas ──────────────────────────────────────
        main = tk.Frame(left_section, bg='#1a1a1a')
        main.pack(fill='both', expand=True)

        left_pane = tk.Frame(main, bg='#1a1a1a')
        left_pane.pack(side='left', fill='both', expand=True, padx=(4, 2), pady=4)

        self.preview_canvas = tk.Canvas(left_pane, bg='black', highlightthickness=0)
        self.preview_canvas.pack(fill='both', expand=True)

        # ── Embedded player controls (hidden until event is loaded) ────────
        BP = '#0e0e0e'
        lp = dict(bg=BP, fg='#999999', font=('Arial', 8))

        self._player_ctrl = tk.Frame(left_pane, bg=BP)
        # not packed yet

        ev_nav = tk.Frame(self._player_ctrl, bg=BP)
        ev_nav.pack(fill='x', pady=(3, 0))
        tk.Button(ev_nav, text='◀ Prev Event', command=self._player_prev_event,
                  bg='#252525', fg='#cccccc', relief='flat',
                  font=('Arial', 8), padx=6).pack(side='left', padx=4)
        self._player_ev_lbl = tk.Label(ev_nav, text='', **lp)
        self._player_ev_lbl.pack(side='left', expand=True)
        tk.Button(ev_nav, text='Next Event ▶', command=self._player_next_event,
                  bg='#252525', fg='#cccccc', relief='flat',
                  font=('Arial', 8), padx=6).pack(side='right', padx=4)

        ctrl_bar = tk.Frame(self._player_ctrl, bg=BP)
        ctrl_bar.pack(fill='x', pady=2)
        tk.Button(ctrl_bar, text='|◀', command=lambda: self._player_goto(0),
                  bg='#252525', fg='white', relief='flat', padx=4).pack(side='left', padx=2)
        #tk.Button(ctrl_bar, text='◀',  command=lambda: self._player_step(-1),
                  #bg='#252525', fg='white', relief='flat', padx=4).pack(side='left', padx=2)
        self._player_play_btn = tk.Button(ctrl_bar, text='▶ Play',
                                          command=self._player_toggle_play,
                                          bg='#1a4a1a', fg='white', relief='flat', padx=8)
        self._player_play_btn.pack(side='left', padx=2)
        tk.Button(ctrl_bar, text='▶|', command=lambda: self._player_goto(-1),
                  bg='#252525', fg='white', relief='flat', padx=4).pack(side='left', padx=2)
        tk.Label(ctrl_bar, text='Speed:', **lp).pack(side='left', padx=(10, 2))
        self._player_speed_var = tk.StringVar(value='80ms')
        tk.Button(ctrl_bar, text='−', command=self._player_speed_slower,
                  bg='#252525', fg='white', relief='flat', padx=6).pack(side='left')
        tk.Label(ctrl_bar, textvariable=self._player_speed_var,
                 bg=BP, fg='white', font=('Consolas', 8), width=6).pack(side='left')
        tk.Button(ctrl_bar, text='+', command=self._player_speed_faster,
                  bg='#252525', fg='white', relief='flat', padx=6).pack(side='left')
        tk.Button(ctrl_bar, text='Real Time', command=self._player_set_realtime,
                  bg='#252525', fg='#cccccc', relief='flat', padx=4).pack(side='left', padx=3)
        self._player_blend_btn = tk.Button(ctrl_bar, text='Max-Blend',
                                           command=self._player_toggle_composite,
                                           bg='#252525', fg='#cccccc', relief='flat', padx=4)
        self._player_blend_btn.pack(side='left', padx=3)
        self._player_tester_btn = tk.Button(ctrl_bar, text='📊 Send to Compare',
                                            command=self._player_send_to_compare,
                                            bg='#1a2a4a', fg='white', relief='flat', padx=6)
        self._player_tester_btn.pack(side='right', padx=6)

        scrub_row = tk.Frame(self._player_ctrl, bg=BP)
        scrub_row.pack(fill='x', pady=(0, 3))
        self._player_scrubber = ttk.Scale(scrub_row, from_=0, to=1, orient='horizontal',
                                          command=self._player_on_scrub)
        self._player_scrubber.pack(side='left', fill='x', expand=True, padx=6)
        self._player_status = tk.Label(scrub_row, text='', width=12, **lp)
        self._player_status.pack(side='right', padx=6)

        self.root.bind('<Left>',  lambda e: self._player_step(-1) if self._player_event_dir else None)
        self.root.bind('<Right>', lambda e: self._player_step(1)  if self._player_event_dir else None)
        self.root.bind('<space>', lambda e: self._player_toggle_play() if self._player_event_dir else None)


    # --------------------------------------------------------------------------
    # File Browsers
    # --------------------------------------------------------------------------

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select MKV File",
            filetypes=[("MKV files", "*.mkv"), ("All files", "*.*")])
        if path:
            self.input_path.set(path)
            folder = os.path.dirname(path)
            self.output_dir.set(folder)
            auto_mask = os.path.join(folder, "mask.png")
            if os.path.exists(auto_mask):
                self.mask_path.set(auto_mask)

    def _browse_input_folder(self):
        path = filedialog.askdirectory(title="Select Folder of MKV Clips", parent=self.root)
        if path:
            self.input_path.set(path)
            self.output_dir.set(path)
            auto_mask = os.path.join(path, "mask.png")
            if os.path.exists(auto_mask):
                self.mask_path.set(auto_mask)
            self._queue_populate(path)

    def _queue_populate_from_var(self):
        self._queue_populate(self.input_path.get().strip())

    def _queue_populate(self, folder):
        self._queue_listbox.delete(0, 'end')
        if not os.path.isdir(folder):
            self._queue_count_lbl.config(text="")
            return
        clips = sorted(f for f in os.listdir(folder) if f.lower().endswith('.mkv'))
        for c in clips:
            self._queue_listbox.insert('end', c)
        n = len(clips)
        self._queue_count_lbl.config(text=f"{n} MKV{'s' if n != 1 else ''} queued")

    def _queue_remove_selected(self):
        for i in reversed(self._queue_listbox.curselection()):
            self._queue_listbox.delete(i)
        n = self._queue_listbox.size()
        self._queue_count_lbl.config(
            text=f"{n} MKV{'s' if n != 1 else ''} queued" if n else "")

    def _queue_add_files(self):
        paths = filedialog.askopenfilenames(
            title="Add MKV files to queue",
            filetypes=[("MKV files", "*.mkv")])
        if not paths:
            return
        existing = set(self._queue_listbox.get(0, 'end'))
        for p in paths:
            if p not in existing:
                self._queue_listbox.insert('end', p)
                existing.add(p)
        n = self._queue_listbox.size()
        self._queue_count_lbl.config(text=f"{n} MKV{'s' if n != 1 else ''} queued")

    def _browse_mask(self):
        path = filedialog.askopenfilename(
            title="Select Mask PNG",
            filetypes=[("PNG files", "*.png")])
        if path:
            self.mask_path.set(path)

    def _browse_dark_frame(self):
        path = filedialog.askopenfilename(
            title="Select Dark Frame",
            filetypes=[("NumPy array", "*.npy")])
        if path:
            self.dark_frame_path.set(path)

    def _build_dark_from_mkv(self):
        mkv = filedialog.askopenfilename(
            title="Select dark MKV (lens cap on)",
            filetypes=[("MKV files", "*.mkv"), ("All files", "*.*")])
        if not mkv:
            return
        out_path = os.path.splitext(mkv)[0] + '_dark.npy'
        self._dark_capture_dialog(mkv, out_path, n_frames=300)

    def _capture_dark_from_rtsp(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Capture Dark Frame from Camera")
        dlg.configure(bg='#1a1a1a')
        dlg.resizable(False, False)
        dlg.grab_set()

        BG = '#1a1a1a'
        FG = 'white'

        # Prefill RTSP from cameras list if available
        default_url = ''
        if self._cameras:
            default_url = self._cameras[0].get('rtsp_url', '')

        tk.Label(dlg, text="RTSP URL:", bg=BG, fg=FG,
                 font=('Arial', 9)).grid(row=0, column=0, sticky='e', padx=8, pady=6)
        url_var = tk.StringVar(value=default_url)
        tk.Entry(dlg, textvariable=url_var, bg='#2a2a2a', fg=FG, relief='flat',
                 width=52).grid(row=0, column=1, columnspan=2, padx=(0, 8), pady=6)

        tk.Label(dlg, text="Frames:", bg=BG, fg=FG,
                 font=('Arial', 9)).grid(row=1, column=0, sticky='e', padx=8, pady=4)
        frames_var = tk.IntVar(value=300)
        tk.Spinbox(dlg, from_=30, to=1000, increment=30, textvariable=frames_var,
                   bg='#2a2a2a', fg=FG, relief='flat',
                   width=8).grid(row=1, column=1, sticky='w', padx=(0, 8), pady=4)

        tk.Label(dlg, text="Save as:", bg=BG, fg=FG,
                 font=('Arial', 9)).grid(row=2, column=0, sticky='e', padx=8, pady=4)
        save_var = tk.StringVar(value=os.path.join(_BASE_DIR, 'dark_frame.npy'))
        tk.Entry(dlg, textvariable=save_var, bg='#2a2a2a', fg=FG, relief='flat',
                 width=44).grid(row=2, column=1, padx=(0, 4), pady=4)
        def _browse_save():
            p = filedialog.asksaveasfilename(defaultextension='.npy',
                filetypes=[("NumPy array", "*.npy")],
                initialfile='dark_frame.npy')
            if p:
                save_var.set(p)
        tk.Button(dlg, text="…", command=_browse_save,
                  bg='#444', fg=FG, relief='flat', width=2).grid(row=2, column=2, padx=(0, 8))

        status_var = tk.StringVar(value="Ready")
        tk.Label(dlg, textvariable=status_var, bg=BG, fg='#aaaaaa',
                 font=('Arial', 8)).grid(row=3, column=0, columnspan=3, pady=4)

        def _do_capture():
            url = url_var.get().strip()
            if not url:
                status_var.set("Enter an RTSP URL first.")
                return
            btn_go.config(state='disabled')
            status_var.set("Connecting…")
            dlg.update()

            def _run():
                def _progress(done, total):
                    status_var.set(f"Capturing {done}/{total} frames…")
                    dlg.update()
                dark = build_dark_frame(url, n_frames=frames_var.get(), progress_cb=_progress)
                if dark is None:
                    status_var.set("Failed — check URL and camera connection.")
                    btn_go.config(state='normal')
                    return
                np.save(save_var.get(), dark)
                self.dark_frame_path.set(save_var.get())
                self._save_config()
                status_var.set(f"Saved {dark.shape[1]}×{dark.shape[0]} dark frame.")
                btn_go.config(state='normal')

            threading.Thread(target=_run, daemon=True).start()

        btn_f = tk.Frame(dlg, bg=BG)
        btn_f.grid(row=4, column=0, columnspan=3, pady=8)
        btn_go = tk.Button(btn_f, text="Capture", command=_do_capture,
                           bg='#226622', fg=FG, relief='flat', padx=12, pady=4)
        btn_go.pack(side='left', padx=4)
        tk.Button(btn_f, text="Close", command=dlg.destroy,
                  bg='#444', fg=FG, relief='flat', padx=12, pady=4).pack(side='left', padx=4)

    def _dark_capture_dialog(self, source, out_path, n_frames=300):
        dlg = tk.Toplevel(self.root)
        dlg.title("Building Dark Frame…")
        dlg.configure(bg='#1a1a1a')
        dlg.resizable(False, False)
        dlg.grab_set()

        BG, FG = '#1a1a1a', 'white'
        status_var = tk.StringVar(value="Starting…")
        tk.Label(dlg, textvariable=status_var, bg=BG, fg=FG,
                 font=('Arial', 10), wraplength=360).pack(padx=20, pady=20)
        bar = ttk.Progressbar(dlg, length=340, maximum=n_frames)
        bar.pack(padx=20, pady=(0, 16))
        dlg.update()

        def _run():
            def _progress(done, total):
                bar['value'] = done
                status_var.set(f"Reading frame {done}/{total}…")
                dlg.update()
            dark = build_dark_frame(source, n_frames=n_frames, progress_cb=_progress)
            if dark is None:
                status_var.set("Failed — could not read frames.")
                return
            np.save(out_path, dark)
            self.dark_frame_path.set(out_path)
            self._save_config()
            status_var.set(f"Done — saved to {os.path.basename(out_path)}")
            dlg.after(1500, dlg.destroy)

        threading.Thread(target=_run, daemon=True).start()

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select Output Folder", parent=self.root)
        if path:
            self.output_dir.set(path)

    # --------------------------------------------------------------------------
    # Detection Control
    # --------------------------------------------------------------------------

    def _get_params(self):
        return {
            'history':     self.p_history.get(),
            'threshold':   self.p_threshold.get(),
            'min_area':    self.p_min_area.get(),
            'max_area':    self.p_max_area.get(),
            'min_aspect':  self.p_min_asp.get(),
            'max_track':      self.p_max_track.get(),
            'max_match_dist': self.p_max_match_dist.get(),
            'pre_buffer':  self.p_pre_buf.get(),
            'post_buffer': self.p_post_buf.get(),
            'warmup':      self.p_warmup.get(),
            'cloud_thresh': self.p_cld_thr.get(),
            'cloud_ratio':  self.p_cld_rat.get(),
            'min_move':          self.p_min_move.get(),
            'min_travel':        self.p_min_travel.get(),
            'min_bright':        self.p_min_bright.get(),
            'cloud_min_bright':  self.p_cloud_min_bright.get(),
            'cloud_min_travel':  self.p_cloud_min_travel.get(),
            'scale':             self.p_scale.get(),
        }

    def _start_detection(self):
        inp = self.input_path.get().strip()
        if not inp or not os.path.exists(inp):
            messagebox.showerror("Error", "Select a valid folder of MKV clips.")
            return
        base_out = self.output_dir.get().strip() or inp

        # Create timestamped run subfolder
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(base_out, f"detect_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        self._current_run_dir = run_dir

        # Save run_info.json so the folder is self-describing
        src_label = (os.path.basename(inp) if os.path.isfile(inp)
                     else os.path.basename(inp.rstrip('/\\')))
        run_info = {
            'type':      'detect',
            'source':    src_label,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'params':    self._get_params(),
        }
        with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
            json.dump(run_info, f, indent=2)

        force = self.p_force_rerun.get()

        # Collect MKVs
        # When forcing, delete checkpoint files so clips restart from frame 0
        if os.path.isdir(inp):
            if self._queue_listbox.size() > 0:
                clips = [
                    e if os.path.isabs(e) else os.path.join(inp, e)
                    for e in (self._queue_listbox.get(i)
                              for i in range(self._queue_listbox.size()))
                ]
            else:
                clips = sorted([
                    os.path.join(inp, f) for f in os.listdir(inp)
                    if f.lower().endswith('.mkv')])
            if force:
                pending = clips
                skipped = 0
                for c in clips:
                    cp = os.path.splitext(c)[0] + '.checkpoint'
                    if os.path.exists(cp):
                        os.remove(cp)
            else:
                pending = [c for c in clips
                           if not os.path.exists(os.path.splitext(c)[0] + '.processed')]
                skipped = len(clips) - len(pending)
        else:
            pending = [inp]
            skipped = 0
            if force:
                cp = os.path.splitext(inp)[0] + '.checkpoint'
                if os.path.exists(cp):
                    os.remove(cp)

        if not pending:
            messagebox.showinfo("All Done", "All clips in this folder are already processed.")
            try:
                os.rmdir(run_dir)  # remove only if truly empty
            except OSError:
                pass
            return

        self._player_stop_loop()

        self.stop_event.clear()
        while not self.preview_q.empty(): self.preview_q.get_nowait()
        while not self.event_q.empty():   self.event_q.get_nowait()
        while not self.done_q.empty():    self.done_q.get_nowait()

        self.thumb_panel.clear()

        self._batch_queue = pending
        self._batch_total = len(pending)
        self._batch_done  = 0
        lbl = f"{os.path.basename(run_dir)} — {len(pending)} clip(s)"
        if skipped:
            lbl += f", {skipped} already processed skipped"
        if force:
            lbl += " (force re-run)"
        self.progress_lbl.set(lbl)
        self._active_folder_var.set(run_dir)
        self.run_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self._run_next_batch(run_dir)

    def _stop_detection(self):
        self.stop_event.set()
        self._identify_cancel.set()
        worker = getattr(self, '_current_worker', None)
        if worker:
            proc = getattr(worker, '_ffmpeg_proc', None)
            if proc:
                proc.kill()
        self.run_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.progress_lbl.set("Stopped.")
        self._active_folder_var.set("")
        self.root.title(f"StreakerDetect  v{VERSION}")

    # --------------------------------------------------------------------------
    # Queue Polling
    # --------------------------------------------------------------------------

    def _poll_queues(self):
        try:
            # Preview — drain queue, only render the latest frame
            latest_preview = None
            try:
                while True:
                    latest_preview = self.preview_q.get_nowait()
            except queue.Empty:
                pass
            if latest_preview is not None:
                self._update_preview(*latest_preview)

            # New events
            try:
                while True:
                    ev = self.event_q.get_nowait()
                    if 'warning' in ev:
                        self.progress_lbl.set(f"⚠ {ev['warning']}")
                    else:
                        self.thumb_panel.add_event(ev)
            except queue.Empty:
                pass

            # Done
            try:
                result = self.done_q.get_nowait()
                if '_stitch_result' in result:
                    n = result['_stitch_result']
                    self.progress_lbl.set(
                        f"Stitch complete — {n} event(s) merged")
                    if result.get('_then_ufo'):
                        self._show_ufo()
                elif result.get('_identify_done'):
                    self._btn_identify.config(state='normal')
                    if 'error' in result:
                        messagebox.showerror("Identify Error", result['error'])
                        self.progress_lbl.set("Identify failed.")
                    elif self._identify_cancel.is_set():
                        self.progress_lbl.set("Identification cancelled.")
                    else:
                        self.progress_lbl.set("Identification complete.")
                else:
                    self._on_done(result)
            except queue.Empty:
                pass

        except Exception:
            import traceback
            msg = traceback.format_exc()
            print(msg)
            try:
                log_path = os.path.join(_BASE_DIR, 'streaker_error.log')
                with open(log_path, 'a') as f:
                    f.write(f'\n--- {datetime.now()} [poll] ---\n{msg}')
            except Exception:
                pass
        finally:
            self.root.after(250, self._poll_queues)

    def _update_preview(self, frame_bgr, stats):
        if self._canvas_mode == 'player':
            return  # player owns the canvas
        cw = self.preview_canvas.winfo_width()
        ch = self.preview_canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        try:
            h, w = frame_bgr.shape[:2]
            scale = min(cw/w, ch/h)
            disp = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)))
            rgb  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            del disp                            # free before PIL alloc
            pil  = Image.fromarray(rgb)
            del rgb                             # free before PhotoImage alloc
            img  = ImageTk.PhotoImage(pil)
            del pil
            old_img = getattr(self.preview_canvas, 'image', None)
            self.preview_canvas.delete('all')
            self.preview_canvas.create_image(cw//2, ch//2, anchor='center', image=img)
            self.preview_canvas.image = img
            if old_img is not None:
                del old_img
        except Exception:
            pass

        total = stats['total']
        frame = stats['frame']
        pct = (frame / total * 100) if total > 0 else 0
        self.progress_var.set(pct)
        eta_s = int(stats['eta'])
        clip_info = ""
        if getattr(self, '_batch_total', 1) > 1:
            clip_info = f"Clip {self._batch_done}/{self._batch_total}  |  "
        clip_name = os.path.basename(self.input_path.get().strip())
        self.progress_lbl.set(
            f"{clip_info}{clip_name}  |  Frame {frame}/{total}  ({pct:.1f}%)  "
            f"ETA {eta_s//60:02d}:{eta_s%60:02d}")
        el = stats['elapsed']
        self.root.title(
            f"StreakerDetect  v{VERSION}  |  {clip_info}"
            f"Frame {frame}/{total}  |  "
            f"{stats['detections']} det  {stats['cloudy']} cloudy  |  "
            f"Elapsed {int(el//60):02d}:{int(el%60):02d}  ETA {eta_s//60:02d}:{eta_s%60:02d}"
        )

    def _on_done(self, result):
        self.root.title(f"StreakerDetect  v{VERSION}")
        if 'error' in result:
            trace = result.get('trace', result['error'])
            try:
                log_path = os.path.join(_BASE_DIR, 'streaker_error.log')
                with open(log_path, 'a') as f:
                    f.write(f'\n--- {datetime.now()} [worker] ---\n{trace}')
            except Exception:
                pass
            messagebox.showerror("Detection Error", result['error'])
            self.progress_lbl.set("Error.")
            self.run_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            return
        self.progress_var.set(100)
        self.progress_lbl.set(
            f"Complete — {result['frames']} frames | "
            f"{result['detections']} detections | "
            f"{result['detected_frames']} event frames")

        # Continue batch or finish
        if hasattr(self, '_batch_queue'):
            self._run_next_batch(self._current_run_dir)
        else:
            self.run_btn.config(state='normal')
            self.stop_btn.config(state='disabled')

    def _show_logs_popup(self):
        folder = self.input_path.get().strip()
        if not folder or not os.path.isdir(folder):
            folder = filedialog.askdirectory(title="Select folder to read logs from", parent=self.root)
        if not folder:
            return

        markers = sorted([
            f for f in os.listdir(folder) if f.endswith('.processed')])

        out_folder = self.output_dir.get().strip() or folder

        popup = tk.Toplevel(self.root)
        popup.title(f"Detection Logs — {os.path.basename(folder)}")
        popup.configure(bg='#1a1a1a')
        popup.geometry("860x600")

        # Scrollable text area
        frame = tk.Frame(popup, bg='#1a1a1a')
        frame.pack(fill='both', expand=True, padx=8, pady=(8, 4))
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side='right', fill='y')
        text = tk.Text(frame, bg='#111111', fg='#cccccc',
                       font=('Courier', 8), yscrollcommand=scrollbar.set,
                       relief='flat', wrap='none')
        text.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=text.yview)

        # ── Section 1: Per-clip log summary ───────────────────────────────
        text.insert('end', "  DETECTION LOGS\n", 'section')
        if not markers:
            text.insert('end', "  No .processed log files found in input folder.\n\n")
        else:
            text.insert('end', f"{'CLIP':<35} {'DATE':<20} {'FRAMES':>8} "
                               f"{'LOGGED':>7} {'DETECTIONS':>11} {'CLOUDY':>7} "
                               f"{'TIME':>7} {'SCALE':>6} {'THRESH':>7}\n", 'header')
            text.insert('end', "─" * 112 + "\n", 'sep')

            total_det = 0
            total_logged = 0
            for fname in markers:
                path = os.path.join(folder, fname)
                kv = {}
                try:
                    with open(path) as f:
                        for line in f:
                            if ':' in line:
                                k, _, v = line.partition(':')
                                kv[k.strip()] = v.strip()
                except Exception:
                    continue

                clip = fname.replace('.processed', '')[:34]
                date = kv.get('processed', '?')[:19]
                frames = kv.get('frames', '?')
                logged = kv.get('event_folders', '?')
                dets = kv.get('detections', '?')
                cloudy = kv.get('cloudy', '?')
                elapsed = kv.get('elapsed', '?')
                scale = kv.get('scale', '?')
                thresh = kv.get('thresh', '?')
                try:
                    total_det += int(dets)
                except (ValueError, TypeError):
                    pass
                try:
                    total_logged += int(logged)
                except (ValueError, TypeError):
                    pass

                text.insert('end',
                    f"{clip:<35} {date:<20} {frames:>8} {logged:>7} "
                    f"{dets:>11} {cloudy:>7} {elapsed:>7} {scale:>6} {thresh:>7}\n")

            text.insert('end', "─" * 112 + "\n", 'sep')
            text.insert('end',
                f"  {len(markers)} clip(s) logged — {total_logged} folders recorded — "
                f"{total_det} total detections\n", 'footer')

        # ── Section 2: Run comparison ──────────────────────────────────────
        text.insert('end', "\n  RUN COMPARISON  (output: {})\n".format(out_folder), 'section')

        # Find run_NNN subfolders
        try:
            run_dirs = sorted([
                e.name for e in os.scandir(out_folder)
                if e.is_dir() and re.match(r'^run_\d+$', e.name)])
        except OSError:
            run_dirs = []

        if not run_dirs:
            # No run subfolders — fall back to flat scan of out_folder
            text.insert('end', "  No run_ subfolders found yet. Run detection to start tracking.\n")
        else:
            # Collect stats per run from JSON logs
            # run_stats[run] = {clip: {event_folders, detections, cloudy, params}}
            run_stats = {}
            run_params = {}
            all_clips = set()
            for rname in run_dirs:
                rpath = os.path.join(out_folder, rname)
                run_stats[rname] = {}
                run_params[rname] = {}
                for jf in os.listdir(rpath):
                    if not jf.endswith('_detection_log.json'):
                        continue
                    try:
                        with open(os.path.join(rpath, jf)) as jh:
                            d = json.load(jh)
                        clip = os.path.splitext(os.path.basename(d.get('source', jf)))[0]
                        run_stats[rname][clip] = {
                            'folders':    d.get('event_folders', '?'),
                            'detections': d.get('detections', '?'),
                            'cloudy':     d.get('cloudy_suppressed', '?'),
                        }
                        run_params[rname] = d.get('params', {})
                        all_clips.add(clip)
                    except Exception:
                        continue

            if not all_clips:
                text.insert('end', "  Run folders exist but contain no detection logs yet.\n")
            else:
                # Per-clip comparison table
                col_w = 12
                header = f"  {'CLIP':<34}"
                for rname in run_dirs:
                    header += f" {rname:^{col_w}}"
                text.insert('end', header + "\n", 'header')

                subhdr = f"  {'':34}"
                for _ in run_dirs:
                    subhdr += f" {'fld/det/cld':^{col_w}}"
                text.insert('end', subhdr + "\n", 'header')
                text.insert('end', "  " + "─" * (34 + len(run_dirs) * (col_w + 1)) + "\n", 'sep')

                for clip in sorted(all_clips):
                    row = f"  {clip[:33]:<34}"
                    for rname in run_dirs:
                        s = run_stats[rname].get(clip)
                        if s:
                            cell = f"{s['folders']}/{s['detections']}/{s['cloudy']}"
                        else:
                            cell = "—"
                        row += f" {cell:^{col_w}}"
                    text.insert('end', row + "\n")

                # Settings diff per run
                text.insert('end', "\n  SETTINGS PER RUN\n", 'section')
                key_params = ['scale', 'threshold', 'cloud_thresh', 'cloud_ratio',
                              'min_move', 'min_area', 'max_area', 'pre_buffer', 'post_buffer']
                phdr = f"  {'PARAM':<18}"
                for rname in run_dirs:
                    phdr += f" {rname:>10}"
                text.insert('end', phdr + "\n", 'header')
                text.insert('end', "  " + "─" * (18 + len(run_dirs) * 11) + "\n", 'sep')
                for key in key_params:
                    prow = f"  {key:<18}"
                    for rname in run_dirs:
                        val = run_params[rname].get(key, '—')
                        prow += f" {str(val):>10}"
                    text.insert('end', prow + "\n")

        text.tag_config('section', foreground='#ffaa00', font=('Courier', 9, 'bold'))
        text.tag_config('header', foreground='#ffcc44', font=('Courier', 8, 'bold'))
        text.tag_config('sep', foreground='#444444')
        text.tag_config('footer', foreground='#aaffaa', font=('Courier', 8, 'bold'))
        text.config(state='disabled')

        tk.Button(popup, text="Close", command=popup.destroy,
                  bg='#333333', fg='white', relief='flat',
                  padx=12, pady=3).pack(pady=(4, 8))

    def _view_clip_in_player(self, event_dir):
        if not _PLAYER_AVAILABLE:
            messagebox.showerror("Player Unavailable", "StreakerPlayer.py not found.")
            return
        meta_path = os.path.join(event_dir, 'metadata.json')
        source_clip = None
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                source_clip = meta.get('source_clip', '')
            except Exception:
                pass
        run_folder = os.path.dirname(event_dir)
        print(f"[DETECT] view_clip event_dir={event_dir!r}")
        print(f"[DETECT] view_clip source_clip={source_clip!r} isfile={os.path.isfile(source_clip) if source_clip else 'N/A'}")
        print(f"[DETECT] view_clip run_folder={run_folder!r} isdir={os.path.isdir(run_folder)}")
        if source_clip and os.path.isfile(source_clip):
            launch_player(initial_folder=run_folder, mkv_path=source_clip)
        else:
            launch_player(initial_folder=run_folder)

    def _launch_player(self):
        if not _PLAYER_AVAILABLE:
            messagebox.showerror("Player Unavailable", "StreakerPlayer.py not found.")
            return
        # If an event is currently previewed, open the player to that event's source clip
        event_dir = getattr(self, '_player_event_dir', None)
        if event_dir and os.path.isdir(event_dir):
            self._view_clip_in_player(event_dir)
            return
        # Fall back to current run/detect folder
        folder = getattr(self, '_current_run_dir', None)
        if not folder or not os.path.isdir(folder):
            initial = self.output_dir.get() or None
            folder = filedialog.askdirectory(
                title="Select detect folder to open in Player",
                initialdir=initial, parent=self.root)
            if not folder:
                return
        launch_player(initial_folder=folder)

    def _launch_synth_test(self):
        src    = self.input_path.get().strip()
        outdir = self.output_dir.get().strip()
        extra  = []
        if src:
            extra += ['--presource', src]
        if outdir:
            extra += ['--preoutput', outdir]
        launch_companion('synth_test.py', extra)

    def _check_for_update(self):
        import urllib.request
        def _worker():
            try:
                url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
                req = urllib.request.Request(url, headers={'User-Agent': 'StreakerDetect'})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read())
                tag = data.get('tag_name', '').lstrip('v.')
                if not tag or tag == VERSION:
                    return
                assets = data.get('assets', [])
                exe_url = next((a['browser_download_url'] for a in assets
                                if a['name'].lower().endswith('.exe')), None)
                if exe_url:
                    def _show_update(u=exe_url, t=tag):
                        self.root.title(f"StreakerDetect  v{VERSION}  ⬆ v{t} available")
                        self._update_btn.config(
                            text=f"⬆ Update v{t}", bg='#2a1a00',
                            command=lambda: self._do_update(u, t))
                        self._update_btn.pack(side='right', padx=6)
                    self.root.after(0, _show_update)
            except Exception:
                pass
        threading.Thread(target=_worker, daemon=True).start()

    def _do_update(self, download_url, new_version):
        import urllib.request
        if not messagebox.askyesno("Update Available",
                f"Download and install v{new_version}?\n\nThe app will restart automatically."):
            return
        self._update_btn.config(text="Downloading…", fg='#888888', bg=BG)

        exe_path = sys.executable

        def _download():
            try:
                tmp_path = exe_path + '.new'
                urllib.request.urlretrieve(download_url, tmp_path)
                bat = (
                    f'@echo off\r\n'
                    f'timeout /t 2 /nobreak >nul\r\n'
                    f'move /Y "{tmp_path}" "{exe_path}"\r\n'
                    f'start "" "{exe_path}"\r\n'
                )
                bat_path = exe_path + '_update.bat'
                with open(bat_path, 'w') as f:
                    f.write(bat)
                subprocess.Popen(['cmd', '/c', bat_path],
                                 creationflags=subprocess.CREATE_NO_WINDOW)
                self.root.after(0, self.root.quit)
            except Exception as e:
                self.root.after(0, lambda: (
                    messagebox.showerror("Update Failed", str(e)),
                    self._update_btn.config(text=f"⬆ Update v{new_version}",
                                            fg='#ffaa00', bg='#2a1a00')
                ))
        threading.Thread(target=_download, daemon=True).start()

    def _launch_demo(self):
        launch_companion('StreakerDemo.py')

    def _launch_compare(self):
        src   = self.input_path.get().strip()
        extra = ['--source', src] if src else []
        launch_companion('StreakerCompare.py', extra)

    def _launch_sync(self):
        launch_companion('StreakerSync.py')

    def _launch_match(self):
        launch_companion('StreakerMatch.py')

    def _open_mask_editor(self):
        try:
            launch_companion('Mask_editor_gui.py')
        except Exception as e:
            messagebox.showerror("Launch Error", str(e))

    def _open_streaker_player(self):
        folder = self._events_folder_full or self.output_dir.get().strip() or None
        extra  = [folder] if folder and os.path.isdir(folder) else []
        try:
            launch_companion('StreakerPlayer.py', extra)
        except Exception as e:
            messagebox.showerror("Launch Error", str(e))

    def _toggle_thumb_panel(self):
        if self._thumb_visible:
            self._thumb_content.pack_forget()
            self.thumb_col.config(width=26)
            self._thumb_toggle_btn.config(text="▶")
            self._thumb_visible = False
        else:
            self._thumb_content.pack(fill='both', expand=True)
            self.thumb_col.config(width=360)
            self._thumb_toggle_btn.config(text="◀ hide")
            self._thumb_visible = True

    def _open_events_folder(self):
        folder = filedialog.askdirectory(title="Select Events Folder", initialdir=self.output_dir.get() or None, parent=self.root)
        if not folder:
            return
        self._open_events_folder_path(folder)

    def _open_events_folder_path(self, folder):
        if not folder or not os.path.isdir(folder):
            return
        folder = os.path.normpath(folder)
        self._events_folder_var.set(os.path.basename(folder))
        self._events_folder_tip._text = folder

        entries = os.listdir(folder)

        # First look for event_ folders directly in this folder
        event_dirs = sorted(
            os.path.normpath(os.path.join(folder, d)) for d in entries
            if d.startswith('event_') and os.path.isdir(os.path.join(folder, d)))
        source_label = os.path.basename(folder)

        if not event_dirs:
            # No direct events — search one level deeper in every subfolder
            sub_dirs = sorted(d for d in entries if os.path.isdir(os.path.join(folder, d)))
            for sd in sub_dirs:
                sd_path = os.path.normpath(os.path.join(folder, sd))
                event_dirs += sorted(
                    os.path.normpath(os.path.join(sd_path, d)) for d in os.listdir(sd_path)
                    if d.startswith('event_') and os.path.isdir(os.path.join(sd_path, d)))
            if event_dirs:
                source_label = f"{os.path.basename(folder)} ({len(sub_dirs)} subfolders)"

        if not event_dirs:
            messagebox.showinfo("No Events", "No event_ subfolders found in that folder.")
            return

        self.thumb_panel.clear()
        self.progress_lbl.set(f"Loading {len(event_dirs)} events…")

        total  = len(event_dirs)
        _label = source_label
        _q     = queue.Queue()

        def _load_events():
            for event_dir in event_dirs:
                thumb_path  = os.path.join(event_dir, '_thumbnail.jpg')
                clip_path   = os.path.join(event_dir, 'clip.mkv')
                meta_path   = os.path.join(event_dir, 'metadata.json')
                frame_files = []

                n_frames = 0
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path) as f:
                            meta = json.load(f)
                        n_frames = meta.get('end_frame', 0) - meta.get('start_frame', 0)
                    except Exception:
                        pass

                if n_frames == 0:
                    if os.path.exists(clip_path):
                        cap = cv2.VideoCapture(clip_path)
                        n_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                        cap.release()
                    else:
                        frame_files = [f for f in os.listdir(event_dir)
                                       if f.startswith('frame_') and f.endswith('.jpg')]
                        n_frames = len(frame_files)

                if not os.path.exists(thumb_path):
                    if os.path.exists(clip_path):
                        cap = cv2.VideoCapture(clip_path)
                        grays = []
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                        cap.release()
                        thumb_bgr = make_thumbnail(grays)
                        if thumb_bgr is not None:
                            cv2.imwrite(thumb_path, thumb_bgr)
                    else:
                        if not frame_files:
                            frame_files = [f for f in os.listdir(event_dir)
                                           if f.startswith('frame_') and f.endswith('.jpg')]
                        grays = [cv2.imread(os.path.join(event_dir, f), cv2.IMREAD_GRAYSCALE)
                                 for f in sorted(frame_files)]
                        grays = [g for g in grays if g is not None]
                        thumb_bgr = make_thumbnail(grays)
                        if thumb_bgr is not None:
                            cv2.imwrite(thumb_path, thumb_bgr)

                _q.put({'dir': event_dir, 'thumb': thumb_path,
                        'frames': n_frames, 'count': 0})
            _q.put(None)  # sentinel

        def _drain():
            BATCH = 25
            added = 0
            try:
                for _ in range(BATCH):
                    ev = _q.get_nowait()
                    if ev is None:
                        self.progress_lbl.set(f"Loaded {total} events from {_label}")
                        return
                    self.thumb_panel.add_event(ev)
                    added += 1
            except queue.Empty:
                pass
            if added:
                done = len(self.thumb_panel.all_events)
                self.progress_lbl.set(f"Loading… {done}/{total}")
            self.root.after(30, _drain)

        threading.Thread(target=_load_events, daemon=True).start()
        self.root.after(30, _drain)

    def _open_event_viewer(self, event_dir):
        self._load_event(os.path.normpath(event_dir))

    # --------------------------------------------------------------------------
    # Embedded Player
    # --------------------------------------------------------------------------

    def _load_event(self, event_dir):
        clip_path   = os.path.join(event_dir, 'clip.mkv')
        use_mkv     = os.path.exists(clip_path)
        frame_paths = [] if use_mkv else sorted([
            os.path.join(event_dir, f) for f in os.listdir(event_dir)
            if f.startswith('frame_') and f.endswith('.jpg')])

        if use_mkv:
            cap = cv2.VideoCapture(clip_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            # CAP_PROP_FRAME_COUNT is unreliable for H.264 MKV — fall back to metadata
            if n_frames <= 1:
                try:
                    meta_path = os.path.join(event_dir, 'metadata.json')
                    with open(meta_path) as f:
                        _m = json.load(f)
                    n_meta = (_m.get('n_frames') or
                              int(_m.get('end_frame', 0)) - int(_m.get('start_frame', 0)) + 1)
                    if n_meta > 1:
                        n_frames = n_meta
                except Exception:
                    pass
            n_frames = max(1, n_frames)
        else:
            n_frames = len(frame_paths)

        if n_frames == 0:
            return

        self._player_stop_loop()
        self._player_frames      = list(range(n_frames)) if use_mkv else frame_paths
        self._player_frame_cache = [None] * n_frames
        self._player_event_dir   = event_dir
        self._player_mkv_path    = clip_path if use_mkv else None
        self._player_idx         = 0
        self._player_paused      = True
        self._player_show_comp   = False
        self._player_composite   = None

        if use_mkv:
            def _preload_mkv(path, target_dir):
                cap = cv2.VideoCapture(path)
                cw = self.preview_canvas.winfo_width()  or 640
                ch = self.preview_canvas.winfo_height() or 480
                i = 0
                while i < len(self._player_frame_cache):
                    if self._player_event_dir != target_dir:
                        break
                    ret, frame = cap.read()
                    if not ret:
                        break
                    h, w = frame.shape[:2]
                    scale = min(cw / w, ch / h)
                    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                    self._player_frame_cache[i] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    i += 1
                cap.release()
            threading.Thread(target=_preload_mkv, args=(clip_path, event_dir), daemon=True).start()
        else:
            def _preload_frames(paths, target_dir):
                cw = self.preview_canvas.winfo_width()  or 640
                ch = self.preview_canvas.winfo_height() or 480
                for i, p in enumerate(paths):
                    if self._player_event_dir != target_dir:
                        return
                    img = cv2.imread(p, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    scale = min(cw / w, ch / h)
                    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
                    self._player_frame_cache[i] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            threading.Thread(target=_preload_frames, args=(frame_paths, event_dir), daemon=True).start()
        self._player_tester_clip = None
        self._player_blend_btn.config(bg='#252525')
        self._player_tester_btn.config(text='✂ Cutting…', state='disabled', bg='#2a2a2a')
        threading.Thread(target=self._player_auto_cut, args=(event_dir,), daemon=True).start()

        try:
            meta_path = os.path.join(event_dir, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    self._player_fps = float(json.load(f).get('fps', 20.0))
        except Exception:
            self._player_fps = 20.0

        def _build_comp():
            if use_mkv:
                cap = cv2.VideoCapture(clip_path)
                raw = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                cap.release()
            else:
                raw = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in frame_paths]
                raw = [f for f in raw if f is not None]
            if not raw:
                return None
            comp = raw[0].copy().astype(np.float32)
            for f in raw[1:]:
                np.maximum(comp, f.astype(np.float32), out=comp)
            return cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        def _comp_done(comp):
            self._player_composite = comp

        def _comp_thread():
            comp = _build_comp()
            self.root.after(0, lambda: _comp_done(comp))

        threading.Thread(target=_comp_thread, daemon=True).start()

        events = self.thumb_panel.all_events
        ev_idx = next((i for i, e in enumerate(events) if e['dir'] == event_dir), -1)
        lbl = (f"Event {ev_idx+1} / {len(events)}  —  {os.path.basename(event_dir)}"
               if ev_idx >= 0 else os.path.basename(event_dir))
        self._player_ev_lbl.config(text=lbl)
        self.thumb_panel.select_event(event_dir)

        if not self._player_ctrl.winfo_ismapped():
            self._player_ctrl.pack(fill='x')

        self._player_scrubbing = True
        self._player_scrubber.config(to=max(1, len(self._player_frames) - 1))
        self._player_scrubber.set(0)
        self._player_scrubbing = False

        self._player_show_frame()
        self._player_play_btn.config(text='▶ Play')
        self._player_toggle_play()

    def _player_show_frame(self):
        if self._player_show_comp and self._player_composite is not None:
            img_bgr = self._player_composite
            cw = self.preview_canvas.winfo_width()  or 640
            ch = self.preview_canvas.winfo_height() or 480
            h, w = img_bgr.shape[:2]
            scale = min(cw / w, ch / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        elif self._player_frames:
            cached = (self._player_frame_cache[self._player_idx]
                      if self._player_idx < len(self._player_frame_cache) else None)
            if cached is not None:
                rgb = cached
                cw = self.preview_canvas.winfo_width()  or 640
                ch = self.preview_canvas.winfo_height() or 480
            elif isinstance(self._player_frames[self._player_idx], str):
                img_bgr = cv2.imread(self._player_frames[self._player_idx], cv2.IMREAD_COLOR)
                if img_bgr is None:
                    return
                cw = self.preview_canvas.winfo_width()  or 640
                ch = self.preview_canvas.winfo_height() or 480
                h, w = img_bgr.shape[:2]
                scale = min(cw / w, ch / h)
                nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            else:
                return  # MKV cache not yet loaded for this frame
        else:
            return
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview_canvas.delete('all')
        self.preview_canvas.create_image(cw // 2, ch // 2, anchor='center', image=img)
        self.preview_canvas.image = img
        n = len(self._player_frames)
        self._player_status.config(text=f"{self._player_idx + 1} / {n}")

    def _player_on_scrub(self, val):
        if not self._player_scrubbing:
            self._player_goto(int(float(val)))

    def _player_goto(self, idx):
        if not self._player_frames:
            return
        if idx == -1:
            idx = len(self._player_frames) - 1
        self._player_idx = max(0, min(idx, len(self._player_frames) - 1))
        self._player_scrubbing = True
        self._player_scrubber.set(self._player_idx)
        self._player_scrubbing = False
        self._player_show_frame()

    def _player_step(self, d):
        self._player_goto(self._player_idx + d)

    def _player_toggle_play(self):
        self._player_paused = not self._player_paused
        self._canvas_mode = 'player' if not self._player_paused else 'detect'
        self._player_play_btn.config(text='⏸ Pause' if not self._player_paused else '▶ Play')
        if not self._player_paused:
            # Anchor the clock to now so the loop can compensate for render overhead
            self._play_clock_start = time.perf_counter()
            self._play_clock_frame = self._player_idx
            self._player_play_loop()

    def _player_play_loop(self):
        if self._player_paused:
            return
        if self._player_idx >= len(self._player_frames) - 1:
            self._player_idx = 0
            self._play_clock_start = time.perf_counter()
            self._play_clock_frame = 0
        self._player_step(1)
        # Calculate exact time the next frame should fire based on wall clock
        frames_played = self._player_idx - self._play_clock_frame
        ideal_next = self._play_clock_start + (frames_played + 1) * (self._player_speed_ms / 1000)
        delay = max(0, int((ideal_next - time.perf_counter()) * 1000))
        self._player_loop_id = self.root.after(delay, self._player_play_loop)

    def _player_stop_loop(self):
        if self._player_loop_id:
            self.root.after_cancel(self._player_loop_id)
            self._player_loop_id = None
        self._player_paused = True
        self._canvas_mode = 'detect'
        if hasattr(self, '_player_play_btn'):
            self._player_play_btn.config(text='▶ Play')

    def _player_set_realtime(self):
        ms = max(10, int(1000 / max(self._player_fps, 1)))
        self._player_speed_ms = ms
        self._player_speed_var.set(f'{ms}ms')

    def _player_speed_faster(self):
        ms = max(10, self._player_speed_ms - 10)
        self._player_speed_ms = ms
        self._player_speed_var.set(f'{ms}ms')

    def _player_speed_slower(self):
        ms = min(500, self._player_speed_ms + 10)
        self._player_speed_ms = ms
        self._player_speed_var.set(f'{ms}ms')

    def _player_toggle_composite(self):
        self._player_show_comp = not self._player_show_comp
        self._player_blend_btn.config(
            bg='#1a3a5a' if self._player_show_comp else '#252525')
        self._player_show_frame()

    def _player_prev_event(self):
        events = self.thumb_panel.all_events
        idx = next((i for i, e in enumerate(events) if e['dir'] == self._player_event_dir), -1)
        if idx > 0:
            self._load_event(events[idx - 1]['dir'])

    def _player_next_event(self):
        events = self.thumb_panel.all_events
        idx = next((i for i, e in enumerate(events) if e['dir'] == self._player_event_dir), -1)
        if 0 <= idx < len(events) - 1:
            self._load_event(events[idx + 1]['dir'])

    def _player_send_to_compare(self):
        event_dir = self._player_event_dir

        # Try to open the source MKV with a pre-roll window around the event
        source_mkv = None
        start_sec  = 0.0
        duration   = 300.0  # default 5 minutes
        pre_roll   = 180.0  # 3 minutes before event so MOG2 can warm up

        if event_dir and os.path.isdir(event_dir):
            meta_path = os.path.join(event_dir, 'metadata.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    src = meta.get('source_clip', '')
                    if src and os.path.isfile(src):
                        source_mkv = src
                        fps = float(meta.get('fps', 20))
                        event_sec = meta.get('start_frame', 0) / max(fps, 1)
                        start_sec = max(0.0, event_sec - pre_roll)
                except Exception:
                    pass

        if not source_mkv:
            messagebox.showwarning('No source MKV',
                'Could not locate the source recording MKV for this event.\n'
                'Open StreakerCompare manually and select the MKV file.')
            return

        launch_companion('StreakerCompare.py', [
            '--source',   source_mkv,
            '--start',    str(int(start_sec)),
            '--duration', str(int(duration)),
        ])

    def _player_auto_cut(self, event_dir):
        """Background thread: cut tester clip as soon as a thumbnail is clicked."""
        meta_path = os.path.join(event_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            return

        with open(meta_path) as f:
            meta = json.load(f)

        source_clip = meta.get('source_clip', '')
        if not os.path.isfile(source_clip):
            def _missing():
                if self._player_event_dir != event_dir:
                    return
                self._player_tester_btn.config(
                    text='📂 Locate MKV', state='normal', bg='#5a3a1a',
                    command=lambda: self._player_locate_and_cut(event_dir, meta))
            self.root.after(0, _missing)
            return

        out_path = self._do_ffmpeg_cut(event_dir, meta, source_clip)

        def _update():
            if self._player_event_dir != event_dir:
                return
            if out_path:
                self._player_tester_clip = out_path
                self._player_tester_btn.config(
                    text='📊 Send to Compare', state='normal', bg='#1a2a4a',
                    command=self._player_send_to_compare)
            else:
                self._player_tester_btn.config(
                    text='✂ Cut failed', state='disabled', bg='#3a1a1a')

        self.root.after(0, _update)

    def _player_locate_and_cut(self, event_dir, meta):
        source_clip = filedialog.askopenfilename(
            title='Locate source MKV for this event',
            filetypes=[('MKV', '*.mkv'), ('All', '*.*')])
        if not source_clip:
            return
        self._player_tester_btn.config(text='✂ Cutting…', state='disabled', bg='#2a2a2a')
        threading.Thread(
            target=lambda: self._player_auto_cut_with_source(event_dir, meta, source_clip),
            daemon=True).start()

    def _player_auto_cut_with_source(self, event_dir, meta, source_clip):
        out_path = self._do_ffmpeg_cut(event_dir, meta, source_clip)

        def _update():
            if self._player_event_dir != event_dir:
                return
            if out_path:
                self._player_tester_clip = out_path
                self._player_tester_btn.config(
                    text='📊 Send to Compare', state='normal', bg='#1a2a4a',
                    command=self._player_send_to_compare)
            else:
                self._player_tester_btn.config(
                    text='✂ Cut failed', state='disabled', bg='#3a1a1a')

        self.root.after(0, _update)

    def _do_ffmpeg_cut(self, event_dir, meta, source_clip):
        """Cut a tester clip around the event. Returns output path or None."""
        fps         = float(meta.get('fps', 20.0))
        start_frame = int(meta.get('start_frame', 0))
        end_frame   = int(meta.get('end_frame', start_frame))

        run_start_sec = 0.0
        run_info_path = os.path.join(os.path.dirname(event_dir), 'run_info.json')
        try:
            if os.path.exists(run_info_path):
                with open(run_info_path) as f:
                    run_start_sec = float(json.load(f).get('start_sec', 0.0))
        except Exception:
            pass

        params      = meta.get('params', {})
        event_start = run_start_sec + start_frame / max(fps, 1)
        event_dur   = (end_frame - start_frame) / max(fps, 1)
        pre_frames  = params.get('history', 500) + params.get('warmup', 200)
        pre_sec     = pre_frames / max(fps, 1) + 10
        post_sec    = params.get('post_buffer', 30) / max(fps, 1) + 2
        clip_start  = max(0.0, event_start - pre_sec)
        clip_dur    = (event_start - clip_start) + event_dur + post_sec

        out_dir  = os.path.join(os.path.dirname(event_dir), 'tester_clips')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'tester_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mkv')

        cmd = [FFMPEG_PATH, '-y',
               '-ss', f'{clip_start:.3f}',
               '-i', source_clip,
               '-t', f'{clip_dur:.3f}',
               '-c', 'copy', out_path]
        ok = subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW).returncode == 0
        return out_path if ok and os.path.isfile(out_path) else None

    def _has_events(self, d):
        """True if d contains at least one event_* subdir with metadata.json."""
        try:
            return any(
                os.path.exists(os.path.join(d, sub, 'metadata.json'))
                for sub in os.listdir(d)
                if sub.startswith('event_') and os.path.isdir(os.path.join(d, sub)))
        except OSError:
            return False

    def _run_stitcher(self):
        folder = filedialog.askdirectory(
            title="Select events folder (or date folder containing several runs) to Stitch",
            parent=self.root)
        if not folder:
            return

        if self._has_events(folder):
            run_folders = [folder]
        else:
            run_folders = sorted([
                os.path.join(folder, d) for d in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, d))
                and self._has_events(os.path.join(folder, d))])
            if not run_folders:
                messagebox.showinfo("No Events",
                    "No event_ subfolders found in the selected folder or its subdirectories.")
                return

        self.progress_lbl.set(f"Stitching {len(run_folders)} folder(s)…")

        def do_stitch():
            total = 0
            for run_dir in run_folders:
                stitcher = EventStitcher(
                    run_dir,
                    max_gap_frames=self.p_stitch_gap.get(),
                    position_tolerance=self.p_stitch_tol.get())
                total += stitcher.run()
            self.done_q.put({'_stitch_result': total, 'folder': folder})

        threading.Thread(target=do_stitch, daemon=True).start()

    def _auto_stitch_then_ufo(self, run_dir):
        self.progress_lbl.set("Auto-stitching events…")

        def _do():
            stitcher = EventStitcher(
                run_dir,
                max_gap_frames=self.p_stitch_gap.get(),
                position_tolerance=self.p_stitch_tol.get())
            n = stitcher.run()
            self.done_q.put({'_stitch_result': n, 'folder': run_dir, '_then_ufo': True})

        threading.Thread(target=_do, daemon=True).start()

    def _run_identify(self):
        folder = filedialog.askdirectory(
            title="Select detect run folder (or date folder containing several runs)",
            initialdir=self.output_dir.get() or None, parent=self.root)
        if not folder:
            return

        try:
            import importlib.util
            # When frozen, companion .py files are in sys._MEIPASS, not next to the exe
            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                identify_script = os.path.join(sys._MEIPASS, 'StreakerIdentify.py')
            else:
                identify_script = os.path.join(_BASE_DIR, 'StreakerIdentify.py')
            spec = importlib.util.spec_from_file_location(
                'StreakerIdentify', identify_script)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            messagebox.showerror("Identify Error",
                                 f"Could not load StreakerIdentify.py:\n{e}")
            return

        # Determine which folders to process:
        #  • folder itself has events → process just it
        #  • folder contains subdirs with events → process each subdir (date folder)
        if self._has_events(folder):
            run_folders = [folder]
        else:
            run_folders = sorted([
                os.path.join(folder, d) for d in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, d))
                and self._has_events(os.path.join(folder, d))])
            if not run_folders:
                messagebox.showinfo("No Events",
                    "No event_ subfolders found in the selected folder or its subdirectories.")
                return

        self.progress_lbl.set(
            f"Identifying {len(run_folders)} run folder(s)…")

        self._identify_cancel.clear()
        self._btn_identify.config(state='disabled')

        def _do():
            try:
                config = mod._load_config()
                fa = self.fa_key.get().strip()
                if fa:
                    config['flightaware_api_key'] = fa
                calib  = mod._load_calib(config)
                tle    = mod._find_tle(config)
                lines  = []

                def _log(msg):
                    lines.append(msg)
                    self.root.after(0, lambda m=msg: self.progress_lbl.set(m))

                for i, run_dir in enumerate(run_folders, 1):
                    if self._identify_cancel.is_set():
                        break
                    _log(f'[{i}/{len(run_folders)}] {os.path.basename(run_dir)}')
                    mod.identify_folder(run_dir, config, calib, tle,
                                        progress_cb=_log,
                                        cancel_event=self._identify_cancel)
                summary = '\n'.join(lines[-10:])
                self.done_q.put({'_identify_done': True, 'summary': summary})
            except Exception as e:
                import traceback
                self.done_q.put({'_identify_done': True,
                                 'error': f'{e}\n{traceback.format_exc()}'})

        threading.Thread(target=_do, daemon=True).start()


    # --------------------------------------------------------------------------
    # Batch Processing
    # --------------------------------------------------------------------------

    def _start_batch(self):
        folder = filedialog.askdirectory(title="Select Folder of MKV Clips", parent=self.root)
        if not folder:
            return

        clips = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith('.mkv')])

        if not clips:
            messagebox.showinfo("No Clips", "No MKV files found in that folder.")
            return

        # Filter out already processed clips
        pending = []
        skipped = 0
        for clip in clips:
            src = os.path.splitext(clip)[0]
            if os.path.exists(src + '.processed'):
                skipped += 1
            else:
                pending.append(clip)

        if not pending:
            messagebox.showinfo("All Done",
                                f"All {skipped} clips already processed.")
            return

        msg = f"{len(pending)} clip(s) to process"
        if skipped:
            msg += f", {skipped} already done (skipping)"
        if not messagebox.askokcancel("Batch Run", msg + ". Start?"):
            return

        out = self.output_dir.get().strip() or folder
        self.output_dir.set(out)

        self._batch_queue = list(pending)
        self._batch_total = len(pending)
        self._batch_done = 0
        self.run_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self._run_next_batch(out)

    def _run_next_batch(self, out):
        if self.stop_event.is_set() or not self._batch_queue:
            self.run_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            if not self.stop_event.is_set():
                if self.p_auto_stitch.get() and out and os.path.isdir(out):
                    self._auto_stitch_then_ufo(out)
                else:
                    self._show_ufo()
            return

        clip = self._batch_queue.pop(0)
        self._batch_done += 1
        self.input_path.set(clip)

        self.progress_lbl.set(
            f"Batch {self._batch_done}/{self._batch_total}: "
            f"{os.path.basename(clip)}")

        self.stop_event.clear()
        while not self.preview_q.empty(): self.preview_q.get_nowait()
        while not self.event_q.empty():   self.event_q.get_nowait()
        while not self.done_q.empty():    self.done_q.get_nowait()

        # Release previous worker's MOG2 model and frame buffers before starting next clip
        if hasattr(self, 'worker_thread') and self.worker_thread is not None:
            self.worker_thread.join(timeout=5)
        self._current_worker = None
        gc.collect()

        worker = DetectionWorker(
            input_path=clip,
            mask_path=self.mask_path.get().strip() or None,
            dark_frame_path=self.dark_frame_path.get().strip() or None,
            output_dir=out,
            params=self._get_params(),
            preview_q=self.preview_q,
            event_q=self.event_q,
            done_q=self.done_q,
            stop_event=self.stop_event,
            sw_decode=self.p_no_hwaccel.get(),
        )
        self._current_worker = worker
        self.worker_thread = threading.Thread(target=worker.run, daemon=True)
        self.worker_thread.start()

    # --------------------------------------------------------------------------
    # UFO Celebration
    # --------------------------------------------------------------------------

    def _show_ufo(self):
        play_completion_sound()

        win = tk.Toplevel(self.root)
        win.title("Batch Complete!")
        win.geometry("500x400")
        win.configure(bg='black')
        win.resizable(False, False)

        tk.Label(win, text="BATCH COMPLETE", bg='black', fg='#00ff00',
                 font=('Arial', 18, 'bold')).pack(pady=(10, 0))
        tk.Label(win,
                 text=f"{self._batch_total} clip(s) processed",
                 bg='black', fg='#aaffaa',
                 font=('Arial', 11)).pack()

        c = tk.Canvas(win, width=500, height=300, bg='black',
                      highlightthickness=0)
        c.pack(pady=10)
        tk.Button(win, text="OK", command=win.destroy,
                  bg='#333333', fg='white', font=('Arial', 11),
                  relief='flat', padx=20, pady=5).pack()

        # Stars
        import random
        stars = [(random.randint(0, 500), random.randint(0, 300))
                 for _ in range(80)]
        for sx, sy in stars:
            c.create_oval(sx, sy, sx+1, sy+1, fill='white', outline='')

        # UFO body
        ufo = c.create_oval(175, 30, 325, 90, fill='#888888', outline='#cccccc', width=2)
        dome = c.create_oval(210, 10, 290, 55, fill='#aaddff', outline='#cccccc', width=2)
        l1 = c.create_oval(185, 75, 205, 85, fill='yellow', outline='')
        l2 = c.create_oval(220, 78, 240, 88, fill='yellow', outline='')
        l3 = c.create_oval(255, 78, 275, 88, fill='yellow', outline='')
        l4 = c.create_oval(290, 75, 310, 85, fill='yellow', outline='')
        ufo_parts = [ufo, dome, l1, l2, l3, l4]

        # Beam
        beam = c.create_polygon(220, 90, 280, 90, 310, 220, 190, 220,
                                 fill='#ffffaa', stipple='gray25', outline='')

        # Cow (simple ASCII-art style shapes)
        cow_body = c.create_oval(215, 230, 285, 265, fill='white', outline='#333333', width=2)
        cow_head = c.create_oval(275, 220, 305, 250, fill='white', outline='#333333', width=2)
        cow_l1   = c.create_rectangle(220, 263, 232, 285, fill='white', outline='#333333')
        cow_l2   = c.create_rectangle(240, 263, 252, 285, fill='white', outline='#333333')
        cow_l3   = c.create_rectangle(258, 263, 270, 285, fill='white', outline='#333333')
        cow_l4   = c.create_rectangle(274, 263, 284, 285, fill='white', outline='#333333')
        cow_spot = c.create_oval(230, 235, 255, 255, fill='#333333', outline='')
        cow_parts = [cow_body, cow_head, cow_l1, cow_l2, cow_l3, cow_l4, cow_spot]

        state = {'y_cow': 0, 'ufo_dy': 0.0, 'phase': 'hover', 'tick': 0}

        def animate():
            if not win.winfo_exists():
                return
            t = state['tick']
            state['tick'] += 1

            # UFO hover wobble
            wobble = 3 if (t // 10) % 2 == 0 else -3
            dy = wobble * 0.3
            for part in ufo_parts:
                c.move(part, 0, dy)
            state['ufo_dy'] += dy

            if state['phase'] == 'hover' and t > 20:
                state['phase'] = 'beam'

            if state['phase'] == 'beam' and t > 35:
                state['phase'] = 'lift'

            if state['phase'] == 'lift':
                for part in cow_parts:
                    c.move(part, 0, -4)
                state['y_cow'] += 4
                if state['y_cow'] > 180:
                    # Reset everything and loop
                    for part in cow_parts:
                        c.move(part, 0, state['y_cow'])
                    for part in ufo_parts:
                        c.move(part, 0, -state['ufo_dy'])
                    state['y_cow'] = 0
                    state['ufo_dy'] = 0.0
                    state['tick'] = 0
                    state['phase'] = 'hover'

            win.after(40, animate)

        win.after(200, animate)


# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------

def main():
    import traceback
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'streaker_error.log')

    def _log_exception(exc_type, exc_value, exc_tb):
        msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        try:
            with open(log_path, 'a') as f:
                f.write(f'\n--- {datetime.now()} ---\n{msg}')
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _log_exception

    # Catch unhandled exceptions in daemon/background threads
    def _thread_exception(args):
        msg = ''.join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        try:
            with open(log_path, 'a') as f:
                name = getattr(args.thread, 'name', 'unknown')
                f.write(f'\n--- {datetime.now()} [thread:{name}] ---\n{msg}')
        except Exception:
            pass
    threading.excepthook = _thread_exception

    try:
        root = tk.Tk()
        root.title(f"StreakerDetect  v{VERSION}")
        root.state('zoomed')

        def _tk_cb_exception(exc_type, exc_value, exc_tb):
            msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            try:
                with open(log_path, 'a') as f:
                    f.write(f'\n--- {datetime.now()} [tk] ---\n{msg}')
            except Exception:
                pass
            # Show the normal tkinter error dialog so it's still visible
            import tkinter.messagebox
            tkinter.messagebox.showerror("Unexpected Error", msg[:2000])

        root.report_callback_exception = _tk_cb_exception

        app = StreakerDetectApp(root)

        # Allow synth_test / other tools to pre-load an events folder on launch
        if '--load' in sys.argv:
            idx = sys.argv.index('--load')
            if idx + 1 < len(sys.argv):
                folder = sys.argv[idx + 1]
                root.after(200, lambda: app._open_events_folder_path(folder))

        root.mainloop()
    except Exception:
        _log_exception(*sys.exc_info())

if __name__ == "__main__":
    # --companion <ScriptName.py> [args...]: dispatch to a bundled companion script.
    # When frozen, other tools call `sys.executable --companion X.py` instead of
    # launching a venv Python, so no Python installation is needed on target machines.
    if len(sys.argv) >= 3 and sys.argv[1] == '--companion':
        import runpy
        _companion = sys.argv[2]
        sys.argv = [sys.argv[0]] + sys.argv[3:]
        _script_dir = (sys._MEIPASS if (getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'))
                       else os.path.dirname(os.path.abspath(__file__)))
        if _script_dir not in sys.path:
            sys.path.insert(0, _script_dir)
        runpy.run_path(os.path.join(_script_dir, _companion), run_name='__main__')
        raise SystemExit(0)

    try:
        main()
    except Exception:
        import traceback
        _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'streaker_error.log')
        try:
            with open(_log_path, 'a') as f:
                f.write(f'\n--- {datetime.now()} [__main__] ---\n{traceback.format_exc()}')
        except Exception:
            pass
        raise
