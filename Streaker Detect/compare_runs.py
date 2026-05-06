"""
compare_runs.py — Side-by-side detection comparison for two parameter sets.

Runs StreakerDetect's detection engine twice on the same source (MKV or frame folder),
once per config, then produces a comparison report showing which events each config
found, missed, or uniquely detected.

Usage:
    python compare_runs.py --source <mkv_or_folder> --config-a <json> --config-b <json> [options]

Options:
    --source        MKV file or folder of frame_*.jpg / frame_*.png
    --config-a      JSON file with first param set  (or use --label-a / --label-b for display)
    --config-b      JSON file with second param set
    --label-a       Display name for config A (default: "A")
    --label-b       Display name for config B (default: "B")
    --mask          Optional mask PNG
    --output        Output folder (default: compare_results/)
    --match-window  Frames within which two events are considered the "same" event (default: 60)

Examples:
    # Compare threshold=30 vs threshold=50
    python compare_runs.py --source night.mkv --config-a cfg_thr30.json --config-b cfg_thr50.json

    # Quick inline override (edit the DEFAULT_PARAMS_A/B dicts at the top of this file)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from StreakerDetect import process_frame, TrackManager, AdaptiveCloudDetector

DEFAULT_PARAMS = {
    'threshold':        40,
    'min_area':         120,
    'max_area':         1400,
    'min_aspect':       2.0,
    'min_bright':       0,
    'min_move':         0,
    'min_travel':       0,
    'cloud_thresh':     40.0,
    'cloud_ratio':      0.0,
    'cloud_min_bright': 0,
    'cloud_min_travel': 0,
    'scale':            1.0,
    'warmup':           50,
    'pre_buffer':       30,
    'post_buffer':      30,
}

THUMB_W, THUMB_H, INFO_H = 320, 200, 22


def load_frames(source):
    if os.path.isfile(source) and source.lower().endswith('.mkv'):
        cap = cv2.VideoCapture(source)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()
        return frames
    elif os.path.isdir(source):
        paths = sorted([
            os.path.join(source, f) for f in os.listdir(source)
            if f.lower().endswith(('.jpg', '.png')) and f.startswith('frame_')
        ])
        frames = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
        return [f for f in frames if f is not None]
    else:
        raise ValueError(f"Source must be MKV file or frame folder: {source}")


def run_detection(frames, mask, params, label):
    """Detect events in frames. Returns list of event dicts with start/peak frame and bboxes."""
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=params['threshold'], detectShadows=False)
    tracker = TrackManager(max_frames=params.get('max_track', 10))
    cloud_detector = AdaptiveCloudDetector()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    scale = params.get('scale', 1.0)
    warmup = params.get('warmup', 50)

    print(f"  Running config {label} on {len(frames)} frames...")

    active_event = None
    post_cd = 0
    post_buf = params.get('post_buffer', 30)
    events = []
    detection_frames = []  # (idx, bboxes, gray) for current event

    for idx, gray in enumerate(frames):
        if scale != 1.0:
            h, w = gray.shape
            small = cv2.resize(gray, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = gray

        mask_small = mask
        if mask is not None and scale != 1.0:
            mh, mw = mask.shape
            mask_small = cv2.resize(mask, (int(mw * scale), int(mh * scale)),
                                    interpolation=cv2.INTER_NEAREST)

        count, bboxes, _, _, _ = process_frame(
            small, mog2, tracker, mask_small, kernel, params, cloud_detector)

        if idx < warmup:
            continue

        if count > 0:
            if active_event is None:
                active_event = {'start_frame': idx, 'peak_frame': idx,
                                'peak_count': count, 'frame_grays': []}
            else:
                if count > active_event['peak_count']:
                    active_event['peak_frame'] = idx
                    active_event['peak_count'] = count
            active_event['frame_grays'].append(gray)
            detection_frames.append((idx, bboxes))
            post_cd = post_buf
        elif post_cd > 0:
            if active_event:
                active_event['frame_grays'].append(gray)
            post_cd -= 1
            if post_cd == 0 and active_event:
                active_event['end_frame'] = idx
                active_event['total_detections'] = len(detection_frames)
                active_event['detection_frames'] = detection_frames
                events.append(active_event)
                active_event = None
                detection_frames = []

    if active_event:
        active_event['end_frame'] = frames[-1] if frames else 0
        active_event['total_detections'] = len(detection_frames)
        active_event['detection_frames'] = detection_frames
        events.append(active_event)

    print(f"  Config {label}: {len(events)} events found")
    return events


def make_comparison_thumb(gray_frames, label, color, params):
    """Make a small thumbnail composite with a colored border and label."""
    total_h = THUMB_H + INFO_H
    if not gray_frames:
        return np.zeros((total_h, THUMB_W, 3), dtype=np.uint8)
    composite = cv2.resize(gray_frames[0], (THUMB_W, THUMB_H),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
    for f in gray_frames[1:min(len(gray_frames), 30)]:
        small = cv2.resize(f, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA).astype(np.float32)
        np.maximum(composite, small, out=composite)
    bgr = cv2.cvtColor(composite.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(bgr, (0, 0), (THUMB_W - 1, THUMB_H - 1), color, 3)
    cv2.putText(bgr, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(bgr, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    # Info strip below image
    canvas = np.zeros((total_h, THUMB_W, 3), dtype=np.uint8)
    canvas[:THUMB_H] = bgr
    canvas[THUMB_H:] = (18, 18, 18)
    info = (f"thr={params.get('threshold','?')} area={params.get('min_area','?')}-"
            f"{params.get('max_area','?')} bright={params.get('min_bright','?')}")
    cv2.putText(canvas, info, (4, THUMB_H + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, info, (4, THUMB_H + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                (190, 190, 190), 1, cv2.LINE_AA)
    return canvas


def events_overlap(ev_a, ev_b, window):
    """True if two events' frame ranges overlap within window frames."""
    a_start, a_end = ev_a['start_frame'], ev_a.get('end_frame', ev_a['start_frame'])
    b_start, b_end = ev_b['start_frame'], ev_b.get('end_frame', ev_b['start_frame'])
    return not (a_end + window < b_start or b_end + window < a_start)


def match_events(events_a, events_b, window):
    """Return (matched, only_a, only_b) as lists of event dicts."""
    matched = []
    used_b = set()
    only_a = []

    for ev_a in events_a:
        best = None
        for i, ev_b in enumerate(events_b):
            if i in used_b:
                continue
            if events_overlap(ev_a, ev_b, window):
                best = i
                break
        if best is not None:
            matched.append((ev_a, events_b[best]))
            used_b.add(best)
        else:
            only_a.append(ev_a)

    only_b = [ev for i, ev in enumerate(events_b) if i not in used_b]
    return matched, only_a, only_b


def save_comparison_grid(matched, only_a, only_b,
                         params_a, params_b, label_a, label_b, out_dir):
    """Save a visual grid image showing matched and unique events."""
    color_a = (255, 180, 0)   # blue-ish
    color_b = (0, 200, 255)   # yellow
    color_match = (0, 255, 80)  # green

    rows = []

    for ev_a, ev_b in matched:
        thumb_a = make_comparison_thumb(ev_a['frame_grays'],
                                        f"{label_a} fr{ev_a['start_frame']}", color_match, params_a)
        thumb_b = make_comparison_thumb(ev_b['frame_grays'],
                                        f"{label_b} fr{ev_b['start_frame']}", color_match, params_b)
        row = np.hstack([thumb_a, np.full((THUMB_H, 4, 3), 40, dtype=np.uint8), thumb_b])
        rows.append(row)

    for ev_a in only_a:
        thumb_a = make_comparison_thumb(ev_a['frame_grays'],
                                        f"{label_a} only  fr{ev_a['start_frame']}", color_a, params_a)
        blank = np.full((THUMB_H, THUMB_W, 3), 20, dtype=np.uint8)
        cv2.putText(blank, f"not in {label_b}", (10, THUMB_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        row = np.hstack([thumb_a, np.full((THUMB_H, 4, 3), 40, dtype=np.uint8), blank])
        rows.append(row)

    for ev_b in only_b:
        blank = np.full((THUMB_H, THUMB_W, 3), 20, dtype=np.uint8)
        cv2.putText(blank, f"not in {label_a}", (10, THUMB_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        thumb_b = make_comparison_thumb(ev_b['frame_grays'],
                                        f"{label_b} only  fr{ev_b['start_frame']}", color_b, params_b)
        row = np.hstack([blank, np.full((THUMB_H, 4, 3), 40, dtype=np.uint8), thumb_b])
        rows.append(row)

    if not rows:
        print("  No events to display in grid.")
        return

    # Header row
    header_w = THUMB_W * 2 + 4
    header = np.full((30, header_w, 3), 20, dtype=np.uint8)
    cv2.putText(header, f"{label_a}  (left)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_a, 1)
    cv2.putText(header, f"{label_b}  (right)", (THUMB_W + 14, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_b, 1)

    grid = np.vstack([header] + [np.vstack([r, np.full((3, header_w, 3), 30, dtype=np.uint8)])
                                  for r in rows])
    grid_path = os.path.join(out_dir, 'comparison_grid.jpg')
    cv2.imwrite(grid_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  Visual grid saved: {grid_path}")


def main():
    parser = argparse.ArgumentParser(description='Side-by-side detection comparison')
    parser.add_argument('--source',       required=True)
    parser.add_argument('--config-a',     required=True, dest='config_a')
    parser.add_argument('--config-b',     required=True, dest='config_b')
    parser.add_argument('--label-a',      default='A',   dest='label_a')
    parser.add_argument('--label-b',      default='B',   dest='label_b')
    parser.add_argument('--mask',         default=None)
    parser.add_argument('--output',       default='compare_results')
    parser.add_argument('--match-window', type=int, default=60, dest='match_window')
    args = parser.parse_args()

    def load_config(path):
        p = dict(DEFAULT_PARAMS)
        if os.path.exists(path):
            with open(path) as f:
                p.update(json.load(f))
            print(f"  Loaded {path}")
        else:
            print(f"  WARNING: config not found ({path}), using defaults")
        return p

    params_a = load_config(args.config_a)
    params_b = load_config(args.config_b)

    print(f"\nLoading frames from: {args.source}")
    frames = load_frames(args.source)
    print(f"Loaded {len(frames)} frames")

    mask = None
    if args.mask and os.path.exists(args.mask):
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        print(f"Loaded mask: {args.mask}")

    out_dir = args.output
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(__file__), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nRunning detection...")
    events_a = run_detection(frames, mask, params_a, args.label_a)
    events_b = run_detection(frames, mask, params_b, args.label_b)

    matched, only_a, only_b = match_events(events_a, events_b, args.match_window)

    print(f"\n{'='*60}")
    print(f"COMPARISON:  {args.label_a} vs {args.label_b}")
    print(f"{'='*60}")
    print(f"  {args.label_a} total events : {len(events_a)}")
    print(f"  {args.label_b} total events : {len(events_b)}")
    print(f"  Matched (both found)   : {len(matched)}")
    print(f"  Only in {args.label_a}            : {len(only_a)}")
    print(f"  Only in {args.label_b}            : {len(only_b)}")

    # Param diff
    all_keys = sorted(set(params_a) | set(params_b))
    diff_keys = [k for k in all_keys if params_a.get(k) != params_b.get(k)]
    if diff_keys:
        print(f"\nParam differences:")
        for k in diff_keys:
            print(f"  {k:25s}  {args.label_a}={params_a.get(k,'—')}  {args.label_b}={params_b.get(k,'—')}")

    save_comparison_grid(matched, only_a, only_b,
                         params_a, params_b, args.label_a, args.label_b, out_dir)

    report = {
        'source': args.source,
        'label_a': args.label_a, 'label_b': args.label_b,
        'params_a': params_a, 'params_b': params_b,
        'param_diffs': {k: {'a': params_a.get(k), 'b': params_b.get(k)} for k in diff_keys},
        'events_a': len(events_a), 'events_b': len(events_b),
        'matched': len(matched), 'only_a': len(only_a), 'only_b': len(only_b),
        'matched_frames': [(e[0]['start_frame'], e[1]['start_frame']) for e in matched],
        'only_a_frames': [e['start_frame'] for e in only_a],
        'only_b_frames': [e['start_frame'] for e in only_b],
    }
    report_path = os.path.join(out_dir, 'comparison_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report: {report_path}")


if __name__ == '__main__':
    main()
