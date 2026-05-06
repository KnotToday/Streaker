# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streaker is a night-sky object detection system for a Dahua IP camera (Sony DSLRs planned). It records RTSP streams during twilight windows, stacks frames into composites, runs MOG2-based background subtraction to detect meteors/planes/satellites, and provides GUIs for reviewing and labeling events.

## Two Working Directories

- **`Final Cut/`** — backend pipeline scripts and legacy tools; the canonical home for `Full_Detect_counts.py` and supporting utilities
- **`Streaker Detect/`** — unified GUI applications (`StreakerDetect.py`, `StreakerPlayer.py`) that incorporate the same algorithms; these are packaged as standalone `.exe` files

Some files (e.g. `Full_Detect_counts.py`, `extract_monitor_advanced.py`, `StreakerPlayBack_V2.py`) exist in both directories. `Streaker Detect/` versions are the more feature-complete copies.

## Running the Pipeline

```powershell
# 1. Record (opens GUI; set RTSP URL, GPS coords, twilight offsets, then Start)
python Record/Streaker_Record_Launch.py

# 2. Extract MKV → PNG frames
python extract_monitor_advanced.py

# 3. (Optional) Stack frames into composites
python stack_monitor_advanced.py

# 4a. Detect via unified GUI (preferred)
cd "c:\Users\piein\PythonTrials\Streaker\Streaker Detect"
python StreakerDetect.py

# 4b. Or CLI detection
python Full_Detect_counts.py <input_folder_or_mkv> [mask_path]

# 4c. Or detection tuner GUI (tune thresholds before full run)
python Detect_overlay_mask_module_refactored_launch_v3.py

# 5. Review / annotate results
python StreakerPlayer.py <mkv_file>
```

## Building Executables

PyInstaller spec files are provided for both standalone apps:

```powershell
pyinstaller Record/STREAKERrec.spec          # → Record/dist/STREAKERrec.exe
pyinstaller "Streaker Detect/StreakerDetect.spec"  # → Streaker Detect/dist/StreakerDetect.exe
```

## Key Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | MOG2 background subtraction, contour detection |
| `numpy` | Array math |
| `Pillow` | Frame display in Tkinter |
| `tkinter` | All GUIs (stdlib) |
| `astral==3.2` | Sunrise/twilight time calculations |
| `pytz==2025.2` | Timezone handling for astral |
| `subprocess` / FFmpeg | MKV recording, frame extraction, CUDA playback |

FFmpeg hardcoded path: `C:\Program Files\FFMPEG\ffmpeg-2024-03-04-git-...\bin\ffmpeg.exe`

Install Python deps: `pip install opencv-python numpy Pillow astral==3.2 pytz==2025.2`

## Configuration Files

**`stream_capture_config.json`** — recorder settings (RTSP URL, GPS lat/lon, output dir, twilight offsets in minutes)

**`streaker_config.json`** (Streaker Detect dir) — detection parameters auto-saved by `StreakerDetect.py` GUI:
- `threshold` — MOG2 variance threshold (default 40; lower = more sensitive)
- `min_area` / `max_area` — contour size bounds in px² (120–1400)
- `min_aspect` — elongation filter (streaks are long/thin)
- `warmup` — frames to skip before detection starts (200)
- `pre_buffer` / `post_buffer` — event clip padding frames (30 each)
- `cloud_thresh` / `cloud_ratio` — adaptive cloud suppression (sigma + fraction of active pixels)
- `min_bright` / `min_travel` — per-detection brightness and displacement gates

## Core Detection Algorithm

`Full_Detect_counts.py` and `StreakerDetect.py` share the same logic:

1. MOG2 background model (history=500, varThreshold=40)
2. Optional binary mask (`bitwise_and` with foreground)
3. Morphological opening (3×3 ellipse) to remove noise
4. **Cloud suppression**: rolling 200-frame mean + sigma; suppress frame if >15% of pixels are active
5. Contour extraction → shape filtering (area, aspect ratio)
6. **TrackManager**: IoU matching (threshold=0.3), ghost-frame tolerance (3 frames), max track age (5 frames)
7. Brightness gate: peak pixel must exceed `min_bright`

## TrackManager

Both `Full_Detect_counts.py` and `StreakerDetect.py` implement `TrackManager` independently. It uses IoU-based matching, maintains a ghost-frame grace period (3 frames) so brief occlusions don't split tracks, and caps track length at `max_track` frames for output.

## Event Buffering (MKV mode)

A ring-buffer holds the last 30 frames (pre-buffer). When a detection fires it starts flushing: pre-buffer frames → live detection frames → 30 post-buffer frames. Only clips with detections are written to disk (saves space vs. raw frame dumps).

## GUI Architecture

All GUIs are Tkinter-based with no web UI. Key pattern: parameter controls (Scale/Entry widgets) update a shared config dict; a "Run" button launches a subprocess (`Full_Detect_counts.py`) or triggers in-process processing.

- **`Detect_overlay_mask_module_refactored_launch_v3.py`** — loads a PNG frame stack, shows edge/diff mask overlays, builds CLI args, launches detection
- **`Mask_editor_gui.py`** — polygon + rectangle drawing on frames; saves PNG mask
- **`star_gui.py`** — threshold/blur/dilate controls to auto-generate a star mask
- **`StreakerDetect.py`** — full pipeline GUI with thumbnail gallery and per-event preview
- **`StreakerPlayer.py`** — MKV timeline player, CUDA-accelerated via FFmpeg; color-coded event markers (meteor=cyan, plane=green, satellite=magenta, unreviewed=gray); saves JSON annotations per event

## Data Flow

```
RTSP stream
  → ffmpeg (15-min MKV chunks) → Record/MM-DD-YYYY/*.mkv
  → extract_monitor_advanced.py → Frame_HHMMSS/frame_*.png
  → stack_monitor_advanced.py (optional) → stacked composites
  → Full_Detect_counts.py / StreakerDetect.py
      → clips_dir/frame_*.jpg + overlay_*.jpg
  → StreakerPlayer.py → event_annotations.json
```

## Folder Naming Convention

Recording output follows `MM-DD-YYYY/` date folders. Extract/stack monitors watch for this pattern to auto-trigger downstream steps.
