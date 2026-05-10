import os
import sys
import json
import subprocess
import logging
import traceback
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
from tkinter import Tk
import cv2
import numpy as np
import threading
import queue
import time
import ctypes

_CONFIG_PATH = Path(os.path.dirname(os.path.abspath(sys.argv[0]))) / "dahua_display_config.json"
_EXAMPLE_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "dahua_display_config.example.json"

def _load_config():
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    print(f"WARNING: Config not found at {_CONFIG_PATH} — copy dahua_display_config.example.json and fill in your settings.")
    return {}

_cfg = _load_config()

RTSP_URL      = _cfg.get("rtsp_url", "rtsp://user:password@192.168.x.x:554/cam/realmonitor?channel=1&subtype=0")
WINDOW_WIDTH  = _cfg.get("window_width", 864)
WINDOW_HEIGHT = _cfg.get("window_height", 648)

_LOG_PATH = Path(__file__).parent / f"dahua_display_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("dahua")
QUEUE_SIZE    = 2   # only 2 frames needed for live display; 10 wasted ~120 MB

CURSOR_HIDDEN = False


def get_screen_size():
    root = Tk()
    root.withdraw()
    return root.winfo_screenwidth(), root.winfo_screenheight()


def frame_reader(cap, frame_queue, stop_event):
    while not stop_event.is_set():
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                log.warning("Failed to read frame from camera.")
                time.sleep(0.1)   # prevent tight busy-loop on stream failure
                continue

            if not frame_queue.full():
                frame_queue.put_nowait(frame.copy())
            else:
                try:
                    frame_queue.get_nowait()  # discard oldest
                    frame_queue.put_nowait(frame.copy())
                except queue.Empty:
                    pass

        except Exception as e:
            log.error("Exception in frame_reader: %s", e)
            time.sleep(0.1)


def resize_into(image, buf, win_w, win_h):
    """Resize image with letterbox into pre-allocated buf (win_h x win_w x 3)."""
    h, w = image.shape[:2]
    aspect = w / h
    try:
        if win_w / aspect <= win_h:
            new_w = max(win_w, 1)
            new_h = max(int(win_w / aspect), 1)
        else:
            new_h = max(win_h, 1)
            new_w = max(int(win_h * aspect), 1)

        resized  = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        buf[:] = 0
        y_off = (win_h - new_h) // 2
        x_off = (win_w - new_w) // 2
        buf[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return buf
    except Exception as e:
        log.error("Failed to resize frame: %s", e)
        return None


def main():
    import tkinter.messagebox as _mb
    global CURSOR_HIDDEN
    log.info("Starting Dahua Display")

    if RTSP_URL.startswith("rtsp://user:password"):
        _mb.showerror("Config Missing",
            f"No config file found.\n\nCreate dahua_display_config.json next to the exe:\n{_CONFIG_PATH}")
        return

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 1 frame internal buffer — we want latest, not stale

    if not cap.isOpened():
        log.error("Could not open RTSP stream: %s", RTSP_URL)
        _mb.showerror("Connection Failed", f"Could not open RTSP stream:\n{RTSP_URL}")
        return

    screen_width, screen_height = get_screen_size()
    x = screen_width - WINDOW_WIDTH
    y = (screen_height - WINDOW_HEIGHT) // 2

    cv2.namedWindow("Dahua Camera Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dahua Camera Feed", WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.moveWindow("Dahua Camera Feed", x, y)

    ret, frame = cap.read()
    if not ret:
        log.error("Could not read initial frame.")
        cap.release()
        return

    height, width = frame.shape[:2]
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Pre-allocate display buffer — reused every frame, no repeated heap allocation
    display_buf     = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    display_buf_key = (WINDOW_HEIGHT, WINDOW_WIDTH)

    frame_queue = queue.Queue(maxsize=QUEUE_SIZE)
    stop_event  = threading.Event()
    reader_thread = threading.Thread(target=frame_reader, args=(cap, frame_queue, stop_event),
                                     daemon=True)
    reader_thread.start()

    try:
        fullscreen_mode = False

        while True:
            if cv2.getWindowProperty("Dahua Camera Feed", cv2.WND_PROP_VISIBLE) < 1:
                log.info("Window closed by user.")
                break

            if fullscreen_mode:
                cv2.setWindowProperty("Dahua Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                if not CURSOR_HIDDEN and sys.platform == 'win32':
                    ctypes.windll.user32.ShowCursor(False)
                    CURSOR_HIDDEN = True
            else:
                cv2.setWindowProperty("Dahua Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                if CURSOR_HIDDEN and sys.platform == 'win32':
                    ctypes.windll.user32.ShowCursor(True)
                    CURSOR_HIDDEN = False

            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                frame = black_frame

            _, _, win_w, win_h = cv2.getWindowImageRect("Dahua Camera Feed")
            if win_w <= 0 or win_h <= 0:
                continue

            # Reallocate buffer only when window size changes
            if (win_h, win_w) != display_buf_key:
                display_buf     = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                display_buf_key = (win_h, win_w)

            result = resize_into(frame, display_buf, win_w, win_h)
            if result is None:
                continue

            cv2.imshow("Dahua Camera Feed", result)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                log.info("Quit key pressed.")
                break
            elif key & 0xFF == ord("f"):
                fullscreen_mode = not fullscreen_mode

    finally:
        stop_event.set()
        reader_thread.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()
        if CURSOR_HIDDEN and sys.platform == 'win32':
            ctypes.windll.user32.ShowCursor(True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.critical("Unhandled exception:\n%s", traceback.format_exc())
        sys.exit(1)
