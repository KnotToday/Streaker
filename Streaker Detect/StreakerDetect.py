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

try:
    from StreakerPlayer import launch_player
    _PLAYER_AVAILABLE = True
except ImportError:
    _PLAYER_AVAILABLE = False

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
                           'streaker_config.json')

from platform_utils import FFMPEG_PATH, HWACCEL_ARGS, play_completion_sound

# ------------------------------------------------------------------------------
# Detection Parameters (defaults — all tunable in GUI)
# ------------------------------------------------------------------------------

DEFAULT_MOG2_HISTORY       = 500
DEFAULT_MOG2_THRESHOLD     = 40
DEFAULT_MIN_CONTOUR_AREA   = 100
DEFAULT_MAX_CONTOUR_AREA   = 5000
DEFAULT_MIN_ASPECT_RATIO   = 2.0
DEFAULT_MAX_TRACK_FRAMES   = 5
DEFAULT_PRE_BUFFER         = 30
DEFAULT_POST_BUFFER        = 30
DEFAULT_WARMUP_FRAMES      = 200
DEFAULT_CLOUD_THRESH       = 80
DEFAULT_CLOUD_RATIO        = 0.15

VERSION      = "1.0.0"
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
    def __init__(self, max_frames, iou_threshold=0.3):
        self.max_frames = max_frames
        self.iou_threshold = iou_threshold
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

    def update(self, detections, min_displacement=0):
        matched = set()
        for track in self.tracks:
            best_iou, best_i = 0, -1
            for i, det in enumerate(detections):
                if i in matched: continue
                iou = self._iou(track['bbox'], det)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_iou >= self.iou_threshold:
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
    """Wait for all but the currently-running save to finish before queuing another.
    Keeps at most one batch of frame data in memory beyond what's actively being saved."""
    done = [f for f in futures if f.done()]
    for f in done:
        futures.remove(f)
    if len(futures) >= 1:
        futures[0].result()
        futures.pop(0)


class DetectionWorker:
    def __init__(self, input_path, mask_path, output_dir, params,
                 preview_q, event_q, done_q, stop_event):
        self.input_path  = input_path
        self.mask_path   = mask_path
        self.output_dir  = output_dir
        self.params      = params
        self.preview_q   = preview_q
        self.event_q     = event_q
        self.done_q      = done_q
        self.stop_event  = stop_event

    def run(self):
        try:
            self._run()
        except Exception as e:
            import traceback
            self.done_q.put({'error': str(e), 'trace': traceback.format_exc()})

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

        # Open FFmpeg pipe; uses CUDA hardware decoding when available
        ffmpeg_cmd = [FFMPEG_PATH] + HWACCEL_ARGS
        if seek_frame > 0:
            ffmpeg_cmd += ['-ss', f'{seek_sec:.3f}']
        ffmpeg_cmd += ['-i', self.input_path,
                       '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1']
        self._ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
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
        tracker = TrackManager(max_frames=self.params['max_track'])
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

                # Safety cap — flush if pending grows too large (cloud flooding)
                if len(pending) > 200:
                    _cfrac = sum(cloud_rolling) / max(len(cloud_rolling), 1)
                    _drain_save_queue(save_futures)
                    save_futures.append(
                        executor.submit(self._save_event,
                                        list(pending), list(pending_gray), _cfrac))
                    pending.clear()
                    pending_gray.clear()
                    post_cd = 0

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
            fpath = os.path.join(event_dir, f"frame_{fidx:06d}.jpg")
            cv2.imwrite(fpath, gframe, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if fbboxes:
                centroids = [[x + w//2, y + h//2] for (x, y, w, h) in fbboxes]
                det_meta.append({
                    'frame': fidx,
                    'centroids': centroids,
                    'bboxes': [list(b) for b in fbboxes],
                    'count': fcount,
                })

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

PAGE_SIZE = 100

class ThumbnailPanel:
    def __init__(self, parent, on_click):
        self.on_click   = on_click
        self.thumbnails = []   # PhotoImage refs for current page
        self.all_events = []   # every event_info ever added
        self.visited    = set()
        self.page = 0

        BG = '#1a1a1a'
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill='both', expand=True)

        # Header row: title + page navigation
        hdr = tk.Frame(frame, bg=BG)
        hdr.pack(fill='x')
        tk.Label(hdr, text="DETECTED EVENTS", bg=BG, fg='#aaaaaa',
                 font=('Arial', 9, 'bold')).pack(side='left', pady=(4, 0), padx=4)
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

    def _page_count(self):
        return max(1, (len(self.all_events) + PAGE_SIZE - 1) // PAGE_SIZE)

    def _update_nav(self):
        total = len(self.all_events)
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
        self.canvas.yview_moveto(0)
        start = self.page * PAGE_SIZE
        for i, ev in enumerate(self.all_events[start:start + PAGE_SIZE]):
            self._render_card(ev, start + i)
        self._update_nav()
        self.canvas.after_idle(self._refresh_scrollregion)

    def _render_card(self, event_info, global_idx):
        event_dir    = event_info['dir']
        thumb_path   = event_info['thumb']
        n_frames     = event_info['frames']
        n_detections = event_info['count']

        visited   = event_dir in self.visited
        card_bg   = '#1a3a22' if visited else '#2a2a2a'
        card = tk.Frame(self.inner, bg=card_bg,
                        highlightbackground='#33aa55' if visited else '#2a2a2a',
                        highlightthickness=2 if visited else 0,
                        relief='flat', bd=0)
        card._event_dir = event_dir
        card.pack(fill='x', padx=6, pady=4)

        def _click(d=event_dir, c=card):
            self.visited.add(d)
            self._highlight_card(c)
            self.on_click(d)

        if os.path.exists(thumb_path):
            img_bgr = cv2.imread(thumb_path)
            if img_bgr is not None:
                img = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
                lbl = tk.Label(card, image=img, bg=card_bg, cursor='hand2')
                lbl.image = img
                lbl.pack(side='left', padx=4, pady=4)
                self.thumbnails.append(img)
                lbl.bind('<Button-1>', lambda e, fn=_click: fn())

        info = tk.Frame(card, bg=card_bg)
        info.pack(side='left', fill='both', expand=True, padx=6)
        tk.Label(info, text=f"Event {global_idx + 1}",
                 bg=card_bg, fg='white',
                 font=('Arial', 10, 'bold')).pack(anchor='w')
        tk.Label(info, text=os.path.basename(event_dir),
                 bg=card_bg, fg='#888888',
                 font=('Arial', 8)).pack(anchor='w')
        tk.Label(info, text=f"{n_frames} frames  |  {n_detections} detections",
                 bg=card_bg, fg='#aaffaa',
                 font=('Arial', 9)).pack(anchor='w', pady=(4, 0))
        tk.Button(info, text="▶ View Clip",
                  command=_click,
                  bg='#3a3a3a', fg='white', relief='flat',
                  cursor='hand2').pack(anchor='w', pady=(6, 0))

    def _highlight_card(self, card):
        bg = '#1a3a22'
        card.config(bg=bg, highlightbackground='#33aa55', highlightthickness=2)
        for child in card.winfo_children():
            try: child.config(bg=bg)
            except Exception: pass
            for grandchild in child.winfo_children():
                try: grandchild.config(bg=bg)
                except Exception: pass

    def select_event(self, event_dir):
        """Highlight the card for event_dir; switch page if needed and scroll to it."""
        idx = next((i for i, e in enumerate(self.all_events) if e['dir'] == event_dir), -1)
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

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()
        self.thumbnails.clear()
        self.all_events.clear()
        self.visited.clear()
        self.page = 0
        self._update_nav()

    def add_event(self, event_info):
        self.all_events.append(event_info)
        # Only render if this event falls on the current page
        idx = len(self.all_events) - 1
        if self.page * PAGE_SIZE <= idx < (self.page + 1) * PAGE_SIZE:
            self._render_card(event_info, idx)
        self._update_nav()

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

        # Copy ev_a frames
        for f in sorted(os.listdir(ev_a['event_dir'])):
            if f.startswith('frame_') and f.endswith('.jpg'):
                src_path = os.path.join(ev_a['event_dir'], f)
                cv2.imwrite(os.path.join(merged_dir, f),
                            cv2.imread(src_path, cv2.IMREAD_GRAYSCALE),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Extract gap frames from original MKV via FFmpeg
        gap_start = ev_a['end_frame'] + 1
        gap_end   = ev_b['start_frame'] - 1
        gap_frame_indices = list(range(gap_start, gap_end + 1))
        if gap_frame_indices and os.path.exists(clip_path):
            self._extract_gap_frames(clip_path, gap_frame_indices,
                                     merged_dir, fps)

        # Copy ev_b frames
        for f in sorted(os.listdir(ev_b['event_dir'])):
            if f.startswith('frame_') and f.endswith('.jpg'):
                src_path = os.path.join(ev_b['event_dir'], f)
                cv2.imwrite(os.path.join(merged_dir, f),
                            cv2.imread(src_path, cv2.IMREAD_GRAYSCALE),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Mark gap frames
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
        }
        with open(os.path.join(merged_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        # Regenerate thumbnail from all frames
        all_frames = sorted([
            os.path.join(merged_dir, f) for f in os.listdir(merged_dir)
            if f.startswith('frame_') and f.endswith('.jpg')])
        grays = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in all_frames]
        grays = [g for g in grays if g is not None]
        thumb = make_thumbnail(grays)
        if thumb is not None:
            cv2.imwrite(os.path.join(merged_dir, '_thumbnail.jpg'), thumb)

        return meta

    def _extract_gap_frames(self, clip_path, frame_indices, out_dir, fps):
        if not frame_indices:
            return
        start = frame_indices[0]
        count = len(frame_indices)
        start_sec = start / fps
        cmd = [
            self.ffmpeg_path,
            *HWACCEL_ARGS,
            '-ss', str(start_sec),
            '-i', clip_path,
            '-frames:v', str(count),
            '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            'pipe:1',
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL)
            # Get frame dimensions from first existing frame in out_dir
            sample = sorted([f for f in os.listdir(out_dir)
                             if f.startswith('frame_')])[0]
            sample_img = cv2.imread(os.path.join(out_dir, sample),
                                    cv2.IMREAD_GRAYSCALE)
            if sample_img is None:
                proc.kill(); return
            fh, fw = sample_img.shape
            frame_bytes = fw * fh
            for fidx in frame_indices:
                raw = proc.stdout.read(frame_bytes)
                if len(raw) == frame_bytes:
                    frame_arr = np.frombuffer(raw, dtype=np.uint8).reshape(fh, fw)
                    fpath = os.path.join(out_dir, f"frame_{fidx:06d}.jpg")
                    cv2.imwrite(fpath, frame_arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            proc.stdout.close()
            proc.wait()
        except Exception:
            pass


# ==============================================================================
# Main Application
# ==============================================================================

class StreakerDetectApp:
    def __init__(self, master):
        self.root = master
        master.title("Streaker Detect")
        master.configure(bg='#111111')

        self.input_path = tk.StringVar()
        self.mask_path  = tk.StringVar()
        self.output_dir = tk.StringVar()

        self.p_history   = tk.IntVar(value=DEFAULT_MOG2_HISTORY)
        self.p_threshold = tk.IntVar(value=DEFAULT_MOG2_THRESHOLD)
        self.p_min_area  = tk.IntVar(value=DEFAULT_MIN_CONTOUR_AREA)
        self.p_max_area  = tk.IntVar(value=DEFAULT_MAX_CONTOUR_AREA)
        self.p_min_asp   = tk.DoubleVar(value=DEFAULT_MIN_ASPECT_RATIO)
        self.p_max_track = tk.IntVar(value=DEFAULT_MAX_TRACK_FRAMES)
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

        self.preview_q = queue.Queue(maxsize=2)
        self.event_q   = queue.Queue()
        self.done_q    = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None

        # Embedded player state
        self._player_frames    = []
        self._player_composite = None
        self._player_idx       = 0
        self._player_paused    = True
        self._canvas_mode      = 'detect'  # 'detect' | 'player'
        self._player_speed_ms  = 80
        self._player_show_comp = False
        self._player_loop_id   = None
        self._player_event_dir = None
        self._player_scrubbing = False
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
        if not os.path.exists(CONFIG_PATH):
            return
        try:
            with open(CONFIG_PATH) as f:
                c = json.load(f)
            if c.get('mask_path') and os.path.exists(c['mask_path']):
                self.mask_path.set(c['mask_path'])
            if c.get('output_dir') and os.path.exists(c['output_dir']):
                self.output_dir.set(c['output_dir'])
            self.p_threshold.set(c.get('threshold',  DEFAULT_MOG2_THRESHOLD))
            self.p_min_area.set(c.get('min_area',    DEFAULT_MIN_CONTOUR_AREA))
            self.p_max_area.set(c.get('max_area',    DEFAULT_MAX_CONTOUR_AREA))
            self.p_min_asp.set(c.get('min_aspect',   DEFAULT_MIN_ASPECT_RATIO))
            self.p_max_track.set(c.get('max_track',  DEFAULT_MAX_TRACK_FRAMES))
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

    def _save_config(self):
        try:
            c = {
                'mask_path':   self.mask_path.get(),
                'output_dir':  self.output_dir.get(),
                'threshold':   self.p_threshold.get(),
                'min_area':    self.p_min_area.get(),
                'max_area':    self.p_max_area.get(),
                'min_aspect':  self.p_min_asp.get(),
                'max_track':   self.p_max_track.get(),
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

        def slider(parent, label, var, lo, hi, res):
            f = tk.Frame(parent, bg=BG)
            f.pack(side='left', padx=3, pady=2)
            tk.Label(f, text=label, bg=BG, fg='#888888',
                     font=('Arial', 7)).pack(anchor='w')
            tk.Scale(f, from_=lo, to=hi, resolution=res, variable=var,
                     orient='horizontal', bg=BG, fg='white',
                     troughcolor='#333333', highlightthickness=0,
                     length=150, showvalue=True,
                     font=('Arial', 7)).pack()

        # ── Row 1: title + file inputs + buttons + stats ──────────────────
        row1 = tk.Frame(self.root, bg=BG)
        row1.pack(fill='x', side='top')

        tk.Label(row1, text="STREAKER DETECT", bg=BG, fg='white',
                 font=('Arial', 11, 'bold')).pack(side='left', padx=10, pady=4)
        sep(row1)

        # Input with separate file and folder browse buttons
        inp_f = tk.Frame(row1, bg=BG)
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
        file_input(row1, "Mask (optional)",        self.mask_path,  self._browse_mask)
        file_input(row1, "Output folder",          self.output_dir, self._browse_output)
        sep(row1)

        # Right column: two stacked button rows, aligned together
        right_col = tk.Frame(row1, bg=BG)
        right_col.pack(side='left', padx=4, pady=2)

        btn_f = tk.Frame(right_col, bg=BG)
        btn_f.pack(side='top', anchor='w', pady=(2, 0))
        self.run_btn = tk.Button(btn_f, text="▶ RUN",
                                 command=self._start_detection,
                                 bg='#006600', fg='white',
                                 font=('Arial', 9, 'bold'),
                                 relief='flat', padx=10, pady=3)
        self.run_btn.pack(side='left', padx=2)
        self.stop_btn = tk.Button(btn_f, text="■ STOP",
                                  command=self._stop_detection,
                                  bg='#660000', fg='white',
                                  font=('Arial', 9, 'bold'),
                                  relief='flat', padx=10, pady=3,
                                  state='disabled')
        self.stop_btn.pack(side='left', padx=2)
        tk.Button(btn_f, text="🔗 STITCH",
                  command=self._run_stitcher,
                  bg='#334455', fg='white',
                  font=('Arial', 9, 'bold'),
                  relief='flat', padx=10, pady=3).pack(side='left', padx=2)
        tk.Button(btn_f, text="📂 OPEN EVENTS",
                  command=self._open_events_folder,
                  bg='#1a4a6a', fg='white',
                  font=('Arial', 9, 'bold'),
                  relief='flat', padx=10, pady=3).pack(side='left', padx=2)

        btn_f2 = tk.Frame(right_col, bg=BG)
        btn_f2.pack(side='top', anchor='w', pady=(0, 2))
        tk.Checkbutton(btn_f2, text="Force Re-run", variable=self.p_force_rerun,
                       bg=BG, fg='#aaaaaa', selectcolor='#333333',
                       font=('Arial', 8), activebackground=BG).pack(side='left', padx=(2, 6))
        tk.Button(btn_f2, text="📋 VIEW LOGS",
                  command=self._show_logs_popup,
                  bg='#443300', fg='white',
                  font=('Arial', 9, 'bold'),
                  relief='flat', padx=10, pady=3).pack(side='left', padx=2)
        tk.Button(btn_f2, text="📺 PLAYER",
                  command=self._launch_player,
                  bg='#2a1a4a', fg='white',
                  font=('Arial', 9, 'bold'),
                  relief='flat', padx=10, pady=3,
                  state='normal' if _PLAYER_AVAILABLE else 'disabled'
                  ).pack(side='left', padx=2)
        tk.Button(btn_f2, text="🧪 SYNTH TEST",
                  command=self._launch_synth_test,
                  bg='#1a3a1a', fg='white',
                  font=('Arial', 9, 'bold'),
                  relief='flat', padx=10, pady=3).pack(side='left', padx=2)
        tk.Button(btn_f2, text="⚖ COMPARE",
                  command=self._launch_compare,
                  bg='#1a2a4a', fg='white',
                  font=('Arial', 9, 'bold'),
                  relief='flat', padx=10, pady=3).pack(side='left', padx=2)

        self.stats_var = tk.StringVar(value="—")
        tk.Label(btn_f2, textvariable=self.stats_var, bg=BG,
                 fg='#aaffaa', font=('Courier', 14),
                 justify='left').pack(side='left', padx=8)

        self._update_btn = tk.Button(btn_f2, text=f"v{VERSION}",
                                     bg=BG, fg='#444444',
                                     font=('Arial', 8), relief='flat',
                                     padx=4, state='disabled')
        self._update_btn.pack(side='right', padx=6)

        # ── Row 2: detection & tracking sliders ───────────────────────────
        row2 = tk.Frame(self.root, bg=BG)
        row2.pack(fill='x', side='top')

        for args in [
            ("MOG2 Thresh",  self.p_threshold,  10,  150,    1),
            ("Min Area",     self.p_min_area,   10,  500,   10),
            ("Max Area",     self.p_max_area,  500, 20000,  100),
            ("Aspect Ratio", self.p_min_asp,   1.0,   8.0,  0.1),
            ("Max Track Fr", self.p_max_track,   1,    30,    1),
            ("Pre Buffer",   self.p_pre_buf,     5,   120,    5),
            ("Post Buffer",  self.p_post_buf,    5,   120,    5),
            ("Min Bright",   self.p_min_bright,  0,   255,    5),
        ]:
            slider(row2, *args)

        # ── Row 3: cloud, filter & stitch sliders ─────────────────────────
        row3 = tk.Frame(self.root, bg=BG)
        row3.pack(fill='x', side='top')

        for args in [
            ("Warmup Fr",    self.p_warmup,     50,   500,   50),
            ("Cloud Sens",   self.p_cld_thr,    20,   200,    5),
            ("Cloud Ratio",  self.p_cld_rat,  0.01,  0.50, 0.01),
            ("Min Move px",  self.p_min_move,    0,    50,    1),
            ("Min Travel px", self.p_min_travel,  0,   100,    5),
            ("Stitch Gap",   self.p_stitch_gap,  0,  1000,   10),
            ("Stitch Tol",   self.p_stitch_tol,  5,   300,    5),
        ]:
            slider(row3, *args)

        # Scale dropdown (row 3)
        sf = tk.Frame(row3, bg=BG)
        sf.pack(side='left', padx=8, pady=2)
        tk.Label(sf, text="Detect Scale", bg=BG, fg='#888888',
                 font=('Arial', 7)).pack(anchor='w')
        scale_menu = ttk.Combobox(sf, textvariable=self.p_scale,
                                  values=[1.0, 0.75, 0.5, 0.25],
                                  width=5, state='readonly')
        scale_menu.pack()

        # ── Row 4: adaptive cloud overrides ───────────────────────────────
        row4 = tk.Frame(self.root, bg=BG)
        row4.pack(fill='x', side='top')
        tk.Label(row4, text="ADAPTIVE CLOUD:", bg=BG, fg='#5599bb',
                 font=('Arial', 7, 'bold')).pack(side='left', padx=(8, 4), pady=2)
        tk.Label(row4, text="When ≥30% of recent frames are cloudy, override these filters:",
                 bg=BG, fg='#555555', font=('Arial', 7)).pack(side='left', padx=(0, 8))
        for args in [
            ("Cloud Min Bright", self.p_cloud_min_bright, 0, 255, 5),
            ("Cloud Min Travel", self.p_cloud_min_travel, 0, 100, 5),
        ]:
            slider(row4, *args)

        # ── Progress bar ───────────────────────────────────────────────────
        prog = tk.Frame(self.root, bg='#0a0a0a', pady=1)
        prog.pack(fill='x', side='top')
        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(prog, variable=self.progress_var,
                        maximum=100).pack(fill='x', padx=6, pady=1)
        self.progress_lbl = tk.StringVar(value="Ready")
        tk.Label(prog, textvariable=self.progress_lbl, bg='#0a0a0a',
                 fg='#aaaaaa', font=('Arial', 8)).pack()

        # ── Main area: preview + thumbnails ───────────────────────────────
        main = tk.Frame(self.root, bg='#1a1a1a')
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
        tk.Button(ctrl_bar, text='◀',  command=lambda: self._player_step(-1),
                  bg='#252525', fg='white', relief='flat', padx=4).pack(side='left', padx=2)
        self._player_play_btn = tk.Button(ctrl_bar, text='▶ Play',
                                          command=self._player_toggle_play,
                                          bg='#1a4a1a', fg='white', relief='flat', padx=8)
        self._player_play_btn.pack(side='left', padx=2)
        tk.Button(ctrl_bar, text='▶|', command=lambda: self._player_goto(-1),
                  bg='#252525', fg='white', relief='flat', padx=4).pack(side='left', padx=2)
        tk.Label(ctrl_bar, text='Speed:', **lp).pack(side='left', padx=(10, 2))
        self._player_speed_var = tk.IntVar(value=80)
        tk.Scale(ctrl_bar, from_=10, to=500, orient='horizontal',
                 variable=self._player_speed_var, length=90, showvalue=False,
                 bg=BP, fg='#666', troughcolor='#2a2a2a', highlightthickness=0,
                 command=lambda v: setattr(self, '_player_speed_ms', int(v))
                 ).pack(side='left')
        tk.Button(ctrl_bar, text='Real Time', command=self._player_set_realtime,
                  bg='#252525', fg='#cccccc', relief='flat', padx=4).pack(side='left', padx=3)
        self._player_blend_btn = tk.Button(ctrl_bar, text='Max-Blend',
                                           command=self._player_toggle_composite,
                                           bg='#252525', fg='#cccccc', relief='flat', padx=4)
        self._player_blend_btn.pack(side='left', padx=3)
        self._player_tester_btn = tk.Button(ctrl_bar, text='🔬 Send to Tester',
                                            command=self._player_send_to_tester,
                                            bg='#3a1a5a', fg='white', relief='flat', padx=6)
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

        right = tk.Frame(main, bg='#1a1a1a', width=360)
        right.pack(side='right', fill='y', padx=(2, 4), pady=4)
        right.pack_propagate(False)
        self.thumb_panel = ThumbnailPanel(right, self._open_event_viewer)

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
        path = filedialog.askdirectory(title="Select Folder of MKV Clips")
        if path:
            self.input_path.set(path)
            self.output_dir.set(path)
            auto_mask = os.path.join(path, "mask.png")
            if os.path.exists(auto_mask):
                self.mask_path.set(auto_mask)

    def _browse_mask(self):
        path = filedialog.askopenfilename(
            title="Select Mask PNG",
            filetypes=[("PNG files", "*.png")])
        if path:
            self.mask_path.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select Output Folder")
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
            'max_track':   self.p_max_track.get(),
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
        if os.path.isdir(inp):
            clips = sorted([
                os.path.join(inp, f) for f in os.listdir(inp)
                if f.lower().endswith('.mkv')])
            if force:
                pending = clips
                skipped = 0
            else:
                pending = [c for c in clips
                           if not os.path.exists(os.path.splitext(c)[0] + '.processed')]
                skipped = len(clips) - len(pending)
        else:
            pending = [inp]
            skipped = 0

        if not pending:
            messagebox.showinfo("All Done", "All clips in this folder are already processed.")
            os.rmdir(run_dir)  # nothing to do — remove empty run folder
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
        self.run_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self._run_next_batch(run_dir)

    def _stop_detection(self):
        self.stop_event.set()
        worker = getattr(self, '_current_worker', None)
        if worker:
            proc = getattr(worker, '_ffmpeg_proc', None)
            if proc:
                proc.kill()
        self.run_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.progress_lbl.set("Stopped.")

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
                else:
                    self._on_done(result)
            except queue.Empty:
                pass

        except Exception:
            import traceback
            msg = traceback.format_exc()
            print(msg)
            try:
                log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'streaker_error.log')
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
        self.stats_var.set(
            f"Detections : {stats['detections']}\n"
            f"Cloudy suppressed: {stats['cloudy']}\n"
            f"Elapsed : {int(stats['elapsed']//60):02d}:{int(stats['elapsed']%60):02d}")

    def _on_done(self, result):
        if 'error' in result:
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
            folder = filedialog.askdirectory(title="Select folder to read logs from")
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

    def _launch_player(self):
        if not _PLAYER_AVAILABLE:
            messagebox.showerror("Player Unavailable", "StreakerPlayer.py not found.")
            return
        folder = getattr(self, '_current_run_dir', None) or self.output_dir.get() or None
        launch_player(initial_folder=folder)

    def _launch_synth_test(self):
        synth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synth_test.py')
        if not os.path.exists(synth_path):
            messagebox.showerror("Not Found", f"synth_test.py not found at:\n{synth_path}")
            return
        # Pre-fill source and output from current UI state
        src    = self.input_path.get().strip()
        outdir = self.output_dir.get().strip()
        cmd = [sys.executable, synth_path]
        if src:
            cmd += ['--presource', src]
        if outdir:
            cmd += ['--preoutput', outdir]
        subprocess.Popen(cmd)

    def _check_for_update(self):
        import urllib.request
        def _worker():
            try:
                url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
                req = urllib.request.Request(url, headers={'User-Agent': 'StreakerDetect'})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read())
                tag = data.get('tag_name', '').lstrip('v')
                if not tag or tag == VERSION:
                    return
                assets = data.get('assets', [])
                exe_url = next((a['browser_download_url'] for a in assets
                                if a['name'].lower().endswith('.exe')), None)
                if exe_url:
                    self.root.after(0, lambda: self._update_btn.config(
                        text=f"⬆ Update v{tag}", fg='#ffaa00', bg='#2a1a00',
                        state='normal',
                        command=lambda: self._do_update(exe_url, tag)))
            except Exception:
                pass
        threading.Thread(target=_worker, daemon=True).start()

    def _do_update(self, download_url, new_version):
        import urllib.request
        if not messagebox.askyesno("Update Available",
                f"Download and install v{new_version}?\n\nThe app will restart automatically."):
            return
        self._update_btn.config(text="Downloading…", state='disabled', fg='#888888', bg=BG)

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
                                            state='normal', fg='#ffaa00', bg='#2a1a00')
                ))
        threading.Thread(target=_download, daemon=True).start()

    def _launch_compare(self):
        compare_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'StreakerCompare.py')
        if not os.path.exists(compare_path):
            messagebox.showerror("Not Found", f"StreakerCompare.py not found at:\n{compare_path}")
            return
        src = self.input_path.get().strip()
        cmd = [sys.executable, compare_path]
        if src:
            cmd += ['--source', src]
        subprocess.Popen(cmd)

    def _open_events_folder(self):
        folder = filedialog.askdirectory(title="Select Events Folder", initialdir=self.output_dir.get() or None)
        if not folder:
            return
        self._open_events_folder_path(folder)

    def _open_events_folder_path(self, folder):
        if not folder or not os.path.isdir(folder):
            return

        event_dirs = sorted([
            os.path.join(folder, d) for d in os.listdir(folder)
            if d.startswith('event_') and
            os.path.isdir(os.path.join(folder, d))])

        if not event_dirs:
            messagebox.showinfo("No Events", "No event_ subfolders found in that folder.")
            return

        # Clear existing thumbnails
        self.thumb_panel.clear()

        for event_dir in event_dirs:
            thumb_path = os.path.join(event_dir, '_thumbnail.jpg')
            frame_files = [f for f in os.listdir(event_dir)
                           if f.startswith('frame_') and f.endswith('.jpg')]
            n_frames = len(frame_files)

            # Regenerate thumbnail if missing
            if not os.path.exists(thumb_path) and frame_files:
                grays = [cv2.imread(os.path.join(event_dir, f),
                                    cv2.IMREAD_GRAYSCALE)
                         for f in sorted(frame_files)]
                grays = [g for g in grays if g is not None]
                thumb_bgr = make_thumbnail(grays)
                if thumb_bgr is not None:
                    cv2.imwrite(thumb_path, thumb_bgr)

            self.thumb_panel.add_event({
                'dir':    event_dir,
                'thumb':  thumb_path,
                'frames': n_frames,
                'count':  0,
            })

        self.progress_lbl.set(f"Loaded {len(event_dirs)} events from {os.path.basename(folder)}")

    def _open_event_viewer(self, event_dir):
        self._load_event(event_dir)

    # --------------------------------------------------------------------------
    # Embedded Player
    # --------------------------------------------------------------------------

    def _load_event(self, event_dir):
        frame_paths = sorted([
            os.path.join(event_dir, f) for f in os.listdir(event_dir)
            if f.startswith('frame_') and f.endswith('.jpg')])
        if not frame_paths:
            return

        self._player_stop_loop()
        self._player_frames      = frame_paths
        self._player_event_dir   = event_dir
        self._player_idx         = 0
        self._player_paused      = True
        self._player_show_comp   = False
        self._player_composite   = None
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
        self._player_scrubber.config(to=max(1, len(frame_paths) - 1))
        self._player_scrubber.set(0)
        self._player_scrubbing = False

        self._player_show_frame()
        self._player_play_btn.config(text='▶ Play')
        self._player_toggle_play()

    def _player_show_frame(self):
        if self._player_show_comp and self._player_composite is not None:
            img_bgr = self._player_composite
        elif self._player_frames:
            img_bgr = cv2.imread(self._player_frames[self._player_idx], cv2.IMREAD_COLOR)
            if img_bgr is None:
                return
        else:
            return
        cw = self.preview_canvas.winfo_width()  or 640
        ch = self.preview_canvas.winfo_height() or 480
        h, w = img_bgr.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
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
            self._player_play_loop()

    def _player_play_loop(self):
        if self._player_paused:
            return
        if self._player_idx >= len(self._player_frames) - 1:
            self._player_idx = 0
        self._player_step(1)
        self._player_loop_id = self.root.after(self._player_speed_ms, self._player_play_loop)

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
        self._player_speed_var.set(ms)

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

    def _player_send_to_tester(self):
        if not self._player_tester_clip or not os.path.isfile(self._player_tester_clip):
            return
        synth_py    = os.path.join(os.path.dirname(__file__), 'synth_test.py')
        output_root = self.output_dir.get().strip() or os.path.dirname(__file__)
        subprocess.Popen([sys.executable, synth_py,
                          '--presource', self._player_tester_clip,
                          '--preoutput', output_root])

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
                    text='🔬 Send to Tester', state='normal', bg='#3a1a5a',
                    command=self._player_send_to_tester)
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
                    text='🔬 Send to Tester', state='normal', bg='#3a1a5a',
                    command=self._player_send_to_tester)
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
        ok = subprocess.run(cmd, capture_output=True).returncode == 0
        return out_path if ok and os.path.isfile(out_path) else None

    def _run_stitcher(self):
        folder = filedialog.askdirectory(title="Select Events Folder to Stitch")
        if not folder:
            return
        self.progress_lbl.set("Stitching events...")
        self.root.update()

        def do_stitch():
            stitcher = EventStitcher(
                folder,
                max_gap_frames=self.p_stitch_gap.get(),
                position_tolerance=self.p_stitch_tol.get())
            n = stitcher.run()
            self.done_q.put({'_stitch_result': n, 'folder': folder})

        threading.Thread(target=do_stitch, daemon=True).start()

    # --------------------------------------------------------------------------
    # Batch Processing
    # --------------------------------------------------------------------------

    def _start_batch(self):
        folder = filedialog.askdirectory(title="Select Folder of MKV Clips")
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

        worker = DetectionWorker(
            input_path=clip,
            mask_path=self.mask_path.get().strip() or None,
            output_dir=out,
            params=self._get_params(),
            preview_q=self.preview_q,
            event_q=self.event_q,
            done_q=self.done_q,
            stop_event=self.stop_event,
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

    try:
        root = tk.Tk()

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
    main()
