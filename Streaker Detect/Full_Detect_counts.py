# ------------------------------------------------------------------------------
# Script Name:     Full_Detect_counts.py
# Description:     Runs full detection on a folder of PNG frames or an MKV file
#                  using MOG2 background subtraction with shape filtering and
#                  duration-based track filtering. Saves candidate event clips.
# ------------------------------------------------------------------------------

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from glob import glob
from collections import deque
from StreakerPlayBack_V2 import DetectionVideoPlayer

# ------------------------------------------------------------------------------
# Detection Parameters
# ------------------------------------------------------------------------------

MOG2_HISTORY        = 500     # frames MOG2 uses to build background model
MOG2_VAR_THRESHOLD  = 40      # sensitivity — lower = more sensitive
MOG2_DETECT_SHADOWS = False   # shadow detection not needed for night sky

MIN_CONTOUR_AREA    = 100     # px² — ignore tiny noise/compression blobs
MAX_CONTOUR_AREA    = 5000    # px² — ignore large cloud fills
MIN_ASPECT_RATIO    = 2.0     # streaks are elongated; blobs < this are filtered
MAX_TRACK_FRAMES    = 5       # blobs persisting longer than this = plane/artifact

CLOUD_PIXEL_THRESH  = 80      # brightness value considered "lit sky"
CLOUD_FRAME_RATIO   = 0.15    # if >15% of pixels exceed threshold, suppress detection

# For MKV mode: how many frames to save before and after each detection event
PRE_EVENT_BUFFER    = 30
POST_EVENT_BUFFER   = 30
MOG2_WARMUP_FRAMES  = 200    # frames to feed MOG2 before flagging detections

# ------------------------------------------------------------------------------
# Track Manager — filters detections by duration across frames
# ------------------------------------------------------------------------------

class TrackManager:
    def __init__(self, max_frames, iou_threshold=0.3):
        self.max_frames = max_frames
        self.iou_threshold = iou_threshold
        self.tracks = []

    def _iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(ax, bx)
        iy = max(ay, by)
        iw = min(ax + aw, bx + bw) - ix
        ih = min(ay + ah, by + bh) - iy
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections):
        matched = set()
        for track in self.tracks:
            track['active'] = False
            for i, det in enumerate(detections):
                if i in matched:
                    continue
                if self._iou(track['bbox'], det) >= self.iou_threshold:
                    track['bbox'] = det
                    track['age'] += 1
                    track['active'] = True
                    matched.add(i)
                    break

        for i, det in enumerate(detections):
            if i not in matched:
                self.tracks.append({'bbox': det, 'age': 1, 'active': True})

        self.tracks = [t for t in self.tracks if t['active']]

        return [t['bbox'] for t in self.tracks if t['age'] <= self.max_frames]

# ------------------------------------------------------------------------------
# Shape Filter
# ------------------------------------------------------------------------------

def passes_shape_filter(contour):
    area = cv2.contourArea(contour)
    if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
        return False
    if len(contour) >= 5:
        _, (w, h), _ = cv2.fitEllipse(contour)
        major = max(w, h)
        minor = min(w, h)
        if minor > 0 and (major / minor) < MIN_ASPECT_RATIO:
            return False
    return True


def is_cloudy(frame):
    bright_pixels = cv2.countNonZero(cv2.threshold(frame, CLOUD_PIXEL_THRESH, 255, cv2.THRESH_BINARY)[1])
    return (bright_pixels / frame.size) > CLOUD_FRAME_RATIO

# ------------------------------------------------------------------------------
# Frame processing — shared logic for both input modes
# ------------------------------------------------------------------------------

def process_frame(frame, mog2, tracker, mask, kernel):
    # Always update MOG2 model even during cloudy frames
    fg_mask = mog2.apply(frame)

    # Suppress detections on cloudy frames but keep model updating
    if is_cloudy(frame):
        empty = cv2.cvtColor(fg_mask * 0, cv2.COLOR_GRAY2BGR)
        return 0, empty

    if mask is not None:
        fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=mask)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shaped = [cnt for cnt in contours if passes_shape_filter(cnt)]
    bboxes = [cv2.boundingRect(cnt) for cnt in shaped]
    filtered_bboxes = tracker.update(bboxes)

    debug_overlay = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in filtered_bboxes:
        cv2.rectangle(debug_overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return len(filtered_bboxes), debug_overlay

# ------------------------------------------------------------------------------
# PNG Folder Mode
# ------------------------------------------------------------------------------

def run_detection_folder(input_folder, mask_path, master,
                         mog2_history=MOG2_HISTORY,
                         mog2_threshold=MOG2_VAR_THRESHOLD):

    frame_paths = sorted(glob(os.path.join(input_folder, "*.png")))
    if not frame_paths:
        messagebox.showerror("Error", f"No PNG frames found in {input_folder}")
        return

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path and os.path.exists(mask_path) else None
    diff_mask_dir = os.path.join(input_folder, "diff_masks")
    os.makedirs(diff_mask_dir, exist_ok=True)

    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=mog2_history, varThreshold=mog2_threshold,
        detectShadows=MOG2_DETECT_SHADOWS)
    tracker = TrackManager(max_frames=MAX_TRACK_FRAMES)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    object_counts = []
    diff_mask_paths = []
    total_detections = 0
    detected_frames = 0

    for i, frame_path in enumerate(frame_paths):
        print(f"[DEBUG] Frame {i + 1}/{len(frame_paths)}: {os.path.basename(frame_path)}")
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            object_counts.append(0)
            diff_mask_paths.append(None)
            continue

        count, overlay = process_frame(frame, mog2, tracker, mask, kernel)
        object_counts.append(count)

        if count > 0:
            detected_frames += 1
            total_detections += count
            print(f"[DETECTED] Frame {i:04d} — {count} candidate(s)")

        diff_mask_path = os.path.join(diff_mask_dir, f"diff_mask_{i:05d}.png")
        cv2.imwrite(diff_mask_path, overlay)
        diff_mask_paths.append(diff_mask_path)

    print(f"[SUMMARY] {detected_frames} frames with detections | Total: {total_detections}")
    DetectionVideoPlayer(master, frame_paths, object_counts=object_counts,
                         diff_masks=diff_mask_paths, autoplay=True)

# ------------------------------------------------------------------------------
def _flush_pending(pending, clips_dir, mog2, tracker, mask, kernel,
                   out_paths, out_counts, out_overlays):
    for (fidx, gframe, fcount) in pending:
        _, foverlay = process_frame(gframe, mog2, tracker, mask, kernel)
        fpath = os.path.join(clips_dir, f"frame_{fidx:06d}.jpg")
        opath = os.path.join(clips_dir, f"overlay_{fidx:06d}.jpg")
        cv2.imwrite(fpath, gframe, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(opath, foverlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        out_paths.append(fpath)
        out_counts.append(fcount)
        out_overlays.append(opath)


# MKV Mode — saves only event clips (pre/post buffer around detections)
# ------------------------------------------------------------------------------

def run_detection_mkv(mkv_path, mask_path, master,
                      mog2_history=MOG2_HISTORY,
                      mog2_threshold=MOG2_VAR_THRESHOLD):

    cap = cv2.VideoCapture(mkv_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open video: {mkv_path}")
        return

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path and os.path.exists(mask_path) else None

    from datetime import datetime
    base_name = os.path.splitext(os.path.basename(mkv_path))[0]
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    clips_dir = os.path.join(os.path.dirname(mkv_path), f"{base_name}_events_{run_ts}")
    os.makedirs(clips_dir, exist_ok=True)

    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=mog2_history, varThreshold=mog2_threshold,
        detectShadows=MOG2_DETECT_SHADOWS)
    tracker = TrackManager(max_frames=MAX_TRACK_FRAMES)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Ring buffer holds (frame_index, gray_frame) only — no overlays to save RAM
    pre_buffer = deque(maxlen=PRE_EVENT_BUFFER)
    post_countdown = 0
    pending_save = []  # list of (frame_idx, gray_frame, count)

    saved_frame_paths = []
    saved_counts = []
    saved_overlays = []

    total_detections = 0
    detected_frames = 0
    frame_idx = 0

    print(f"[INFO] Processing MKV: {mkv_path}")
    print(f"[INFO] Warming up MOG2 for {MOG2_WARMUP_FRAMES} frames...")

    while True:
        ret, raw = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        # Always feed MOG2 to build background model
        count, overlay = process_frame(gray, mog2, tracker, mask, kernel)

        # Skip detections during warmup
        if frame_idx < MOG2_WARMUP_FRAMES:
            frame_idx += 1
            continue

        entry = (frame_idx, gray, count)

        if count > 0:
            detected_frames += 1
            total_detections += count
            print(f"[DETECTED] Frame {frame_idx:06d} — {count} candidate(s)")

            if post_countdown == 0:
                pending_save.extend(list(pre_buffer))
                pre_buffer.clear()

            pending_save.append(entry)
            post_countdown = POST_EVENT_BUFFER

        elif post_countdown > 0:
            pending_save.append(entry)
            post_countdown -= 1

            if post_countdown == 0:
                _flush_pending(pending_save, clips_dir, mog2, tracker,
                               mask, kernel, saved_frame_paths,
                               saved_counts, saved_overlays)
                pending_save.clear()

        else:
            pre_buffer.append(entry)

        frame_idx += 1

    # Flush any remaining pending frames
    if pending_save:
        _flush_pending(pending_save, clips_dir, mog2, tracker,
                       mask, kernel, saved_frame_paths,
                       saved_counts, saved_overlays)

    cap.release()

    print(f"[SUMMARY] Processed {frame_idx} frames | "
          f"{detected_frames} with detections | "
          f"Saved {len(saved_frame_paths)} event frames to {clips_dir}")

    if not saved_frame_paths:
        messagebox.showinfo("No Detections", "No candidate events found in this MKV.")
        return

    DetectionVideoPlayer(master, saved_frame_paths, object_counts=saved_counts,
                         diff_masks=saved_overlays, autoplay=True)

# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------

def run_detection(input_path, mask_path, master,
                  mog2_history=MOG2_HISTORY,
                  mog2_threshold=MOG2_VAR_THRESHOLD):
    if input_path.lower().endswith(".mkv"):
        run_detection_mkv(input_path, mask_path, master, mog2_history, mog2_threshold)
    else:
        run_detection_folder(input_path, mask_path, master, mog2_history, mog2_threshold)


if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.withdraw()

        if len(sys.argv) < 2:
            messagebox.showerror(
                "Usage Error",
                "Usage: Full_Detect_counts.py <mkv_file_or_frames_folder> [mask_path]"
            )
            sys.exit(1)

        input_path = sys.argv[1]
        mask_path  = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].strip() else None
        history    = int(sys.argv[3]) if len(sys.argv) > 3 else MOG2_HISTORY
        threshold  = int(sys.argv[4]) if len(sys.argv) > 4 else MOG2_VAR_THRESHOLD

        run_detection(input_path, mask_path, root, history, threshold)
        root.mainloop()

    except Exception as e:
        import traceback
        print("[ERROR] Exception in Full_Detect_counts.py:", e)
        traceback.print_exc()
