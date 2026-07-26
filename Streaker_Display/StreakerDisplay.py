import os
import sys
import json
import logging
import shutil
import subprocess
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
import tkinter as tk
import cv2
import numpy as np
import threading
import queue
import time
import ctypes

_CONFIG_PATH  = Path(os.path.dirname(os.path.abspath(sys.argv[0]))) / "streaker_display_config.json"
_EXAMPLE_PATH = Path(os.path.dirname(os.path.abspath(__file__)))    / "streaker_display_config.example.json"

_DEFAULT_CAM = {
    "label": "CAM",
    "rtsp_url": "rtsp://user:password@192.168.x.x:554/cam/realmonitor?channel=1&subtype=0",
    "substream_url": "",
    "use_substream": False,
}

QUEUE_SIZE  = 4   # per-camera receive queue depth
SYNC_BUFFER = 4   # frames buffered per camera for timestamp matching


def _load_config():
    if not _CONFIG_PATH.exists():
        print(f"WARNING: Config not found at {_CONFIG_PATH} — copy streaker_display_config.example.json and fill in your settings.")
        return {}
    with open(_CONFIG_PATH) as f:
        raw = json.load(f)
    # Migrate old single-camera format
    if "rtsp_url" in raw and "cameras" not in raw:
        raw = {
            "cameras": [
                {
                    "label":         "CAM1",
                    "rtsp_url":      raw.get("rtsp_url", ""),
                    "substream_url": raw.get("substream_url", ""),
                    "use_substream": raw.get("use_substream", False),
                },
                dict(_DEFAULT_CAM, label="CAM2"),
            ],
            "window_width":  raw.get("window_width",  1728),
            "window_height": raw.get("window_height",  648),
        }
    return raw


_cfg = _load_config()

_LOG_PATH = Path(__file__).parent / f"streaker_display_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("streaker_display")

CURSOR_HIDDEN = False


def _save_config(cfg):
    with open(_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def show_settings_dialog(cfg):
    """Dual-camera settings dialog. Returns updated cfg dict or None if cancelled."""
    result = {}
    root = tk.Tk()
    root.title("Camera Settings")
    root.configure(bg="#1a1a2e")
    root.resizable(False, False)

    DARK, FG, ENTRY_BG = "#1a1a2e", "#e0e0e0", "#2a2a4a"

    cameras = list(cfg.get("cameras", []))
    while len(cameras) < 2:
        cameras.append(dict(_DEFAULT_CAM, label=f"CAM{len(cameras) + 1}"))

    frm = tk.Frame(root, bg=DARK, padx=16, pady=16)
    frm.pack(fill="both", expand=True)

    def lbl(text, row):
        tk.Label(frm, text=text, bg=DARK, fg=FG,
                 font=("Arial", 10)).grid(row=row, column=0, sticky="w", padx=12, pady=4)

    cam_vars = []
    row = 0
    for i, cam in enumerate(cameras):
        tk.Label(frm, text=f"── Camera {i + 1} ──", bg=DARK, fg="#8888cc",
                 font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2,
                                                   sticky="w", padx=12, pady=(10, 2))
        row += 1

        lbl("Label", row)
        lbl_var = tk.StringVar(value=cam.get("label", f"CAM{i + 1}"))
        tk.Entry(frm, textvariable=lbl_var, width=20,
                 bg=ENTRY_BG, fg=FG, insertbackground=FG, relief="flat"
                 ).grid(row=row, column=1, sticky="w", padx=8, pady=4)
        row += 1

        lbl("Main Stream URL", row)
        url_var = tk.StringVar(value=cam.get("rtsp_url", ""))
        tk.Entry(frm, textvariable=url_var, width=60,
                 bg=ENTRY_BG, fg=FG, insertbackground=FG, relief="flat"
                 ).grid(row=row, column=1, padx=8, pady=4)
        row += 1

        lbl("Sub-Stream URL", row)
        sub_var = tk.StringVar(value=cam.get("substream_url", ""))
        tk.Entry(frm, textvariable=sub_var, width=60,
                 bg=ENTRY_BG, fg=FG, insertbackground=FG, relief="flat"
                 ).grid(row=row, column=1, padx=8, pady=4)
        row += 1

        use_sub_var = tk.BooleanVar(value=cam.get("use_substream", False))
        tk.Checkbutton(frm, text="Use sub-stream", variable=use_sub_var,
                       bg=DARK, fg=FG, selectcolor=ENTRY_BG,
                       activebackground=DARK, activeforeground=FG
                       ).grid(row=row, column=1, sticky="w", padx=8, pady=2)
        row += 1

        cam_vars.append((lbl_var, url_var, sub_var, use_sub_var))

    tk.Frame(frm, bg="#3a3a5a", height=1).grid(
        row=row, column=0, columnspan=2, sticky="ew", pady=8, padx=4)
    row += 1

    lbl("Window Width", row)
    w_var = tk.StringVar(value=str(cfg.get("window_width", 960)))
    tk.Entry(frm, textvariable=w_var, width=10,
             bg=ENTRY_BG, fg=FG, insertbackground=FG, relief="flat"
             ).grid(row=row, column=1, sticky="w", padx=8, pady=4)
    row += 1

    lbl("Window Height", row)
    h_var = tk.StringVar(value=str(cfg.get("window_height", 1080)))
    tk.Entry(frm, textvariable=h_var, width=10,
             bg=ENTRY_BG, fg=FG, insertbackground=FG, relief="flat"
             ).grid(row=row, column=1, sticky="w", padx=8, pady=4)
    row += 1

    _RES_OPTIONS = ["Auto (1080p)", "Max (Native)", "2560x1440", "1920x1080",
                    "1280x720", "960x540", "640x360"]
    lbl("Display Resolution", row)
    res_var = tk.StringVar(value=cfg.get("display_resolution", "Auto (1080p)"))
    tk.OptionMenu(frm, res_var, *_RES_OPTIONS
                  ).grid(row=row, column=1, sticky="w", padx=8, pady=4)
    row += 1

    status_var = tk.StringVar()
    tk.Label(frm, textvariable=status_var, bg=DARK, fg="#ff6060",
             font=("Arial", 9)).grid(row=row, column=0, columnspan=2, pady=4)
    row += 1

    def on_connect():
        new_cameras = []
        for i, (lbl_v, url_v, sub_v, use_sub_v) in enumerate(cam_vars):
            url = url_v.get().strip()
            if not url.startswith("rtsp://"):
                status_var.set(f"Camera {i + 1} main URL must start with rtsp://")
                return
            sub = sub_v.get().strip()
            if sub and not sub.startswith("rtsp://"):
                status_var.set(f"Camera {i + 1} sub-stream URL must start with rtsp://")
                return
            new_cameras.append({
                "label":         lbl_v.get().strip() or f"CAM{i + 1}",
                "rtsp_url":      url,
                "substream_url": sub,
                "use_substream": use_sub_v.get(),
            })
        try:
            w = int(w_var.get()) if w_var.get().strip() else cfg.get("window_width", 960)
            h = int(h_var.get()) if h_var.get().strip() else cfg.get("window_height", 1080)
        except ValueError:
            status_var.set("Width and Height must be integers")
            return
        result["cameras"]            = new_cameras
        result["window_width"]        = w
        result["window_height"]       = h
        result["display_resolution"]  = res_var.get()
        root.destroy()

    def on_cancel():
        root.destroy()

    btn_frm = tk.Frame(frm, bg=DARK)
    btn_frm.grid(row=row, column=0, columnspan=2, pady=12)
    tk.Button(btn_frm, text="Connect", command=on_connect,
              bg="#1a3a2a", fg="white", font=("Arial", 10, "bold"),
              relief="flat", padx=16, pady=6).pack(side="left", padx=8)
    tk.Button(btn_frm, text="Cancel", command=on_cancel,
              bg="#3a1a1a", fg="white", font=("Arial", 10),
              relief="flat", padx=16, pady=6).pack(side="left", padx=8)

    root.bind("<Return>", lambda _: on_connect())
    root.bind("<Escape>", lambda _: on_cancel())
    root.protocol("WM_DELETE_WINDOW", on_cancel)

    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{(sw - root.winfo_width()) // 2}+{(sh - root.winfo_height()) // 2}")
    root.mainloop()
    return result if result else None


def get_screen_size():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def _find_ffmpeg():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundled = os.path.join(sys._MEIPASS, 'ffmpeg.exe')
        if os.path.isfile(bundled):
            return bundled
    found = shutil.which('ffmpeg')
    if found:
        return found
    fallback = (r"C:\Program Files\FFMPEG"
                r"\ffmpeg-2024-03-04-git-e30369bc1c-full_build\bin\ffmpeg.exe")
    return fallback if os.path.isfile(fallback) else 'ffmpeg'


def _find_ffprobe(ffmpeg):
    d = os.path.dirname(os.path.abspath(ffmpeg))
    candidate = os.path.join(d, 'ffprobe.exe' if sys.platform == 'win32' else 'ffprobe')
    if os.path.isfile(candidate):
        return candidate
    return shutil.which('ffprobe') or 'ffprobe'


_FFMPEG  = _find_ffmpeg()
_FFPROBE = _find_ffprobe(_FFMPEG)


def _probe_dimensions(url):
    try:
        r = subprocess.run(
            [_FFPROBE, '-v', 'quiet', '-select_streams', 'v:0',
             '-show_entries', 'stream=width,height', '-of', 'csv=p=0', url],
            capture_output=True, text=True, timeout=12,
        )
        parts = r.stdout.strip().split(',')
        if len(parts) == 2:
            log.info("ffprobe dimensions: %s x %s", parts[0], parts[1])
            return int(parts[0]), int(parts[1])
    except Exception as e:
        log.warning("ffprobe failed (%s) — falling back to cv2", e)
    try:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;8000000"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w and h:
                log.info("cv2 probe dimensions: %d x %d", w, h)
                return w, h
        cap.release()
    except Exception as e:
        log.error("cv2 probe also failed: %s", e)
    return None, None


def frame_reader(frame_queue, stop_event, display_url, reconnecting_event, force_reconnect_event,
                 display_resolution="Auto (1080p)"):
    reconnecting_event.set()  # show banner immediately while we set up

    if display_resolution == "Max (Native)":
        native_w, native_h = _probe_dimensions(display_url)
        PIPE_W = (native_w or 1920) & ~1
        PIPE_H = (native_h or 1080) & ~1
    elif display_resolution not in (None, "Auto (1080p)", "Native"):
        try:
            pw, ph = (int(x) for x in display_resolution.lower().split("x"))
            PIPE_W, PIPE_H = pw & ~1, ph & ~1
        except Exception:
            PIPE_W, PIPE_H = 1920, 1080
    else:
        PIPE_W, PIPE_H = 1920, 1080

    log.info("Pipe dimensions: %dx%d  (resolution=%s)", PIPE_W, PIPE_H, display_resolution)

    FRAME_BYTES = PIPE_W * PIPE_H * 3
    CHUNK       = max(65536, FRAME_BYTES // 8)

    def launch():
        cmd = [
            _FFMPEG, '-loglevel', 'error',
            '-rtsp_transport', 'tcp',
            '-i', display_url,
            '-vf', f'scale={PIPE_W}:{PIPE_H}',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-r', '24',
            'pipe:1',
        ]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=FRAME_BYTES * 2,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
        )

    proc = None
    try:
        while not stop_event.is_set():
            if force_reconnect_event.is_set():
                force_reconnect_event.clear()
                reconnecting_event.set()
                log.info("Force-reconnect requested.")
                if proc:
                    proc.kill()
                    proc.wait()
                    proc = None
                time.sleep(0.5)

            if proc is None:
                reconnecting_event.set()
                try:
                    proc = launch()
                    log.info("ffmpeg display started (PID %d).", proc.pid)
                except Exception as e:
                    log.error("Failed to launch ffmpeg: %s", e)
                    time.sleep(2.0)
                    continue

            # Read one frame in chunks so stop/reconnect events are checked frequently
            buf      = bytearray()
            pipe_ok  = True
            while len(buf) < FRAME_BYTES:
                if stop_event.is_set() or force_reconnect_event.is_set():
                    pipe_ok = False
                    break
                chunk = proc.stdout.read(min(CHUNK, FRAME_BYTES - len(buf)))
                if not chunk:
                    pipe_ok = False
                    break
                buf.extend(chunk)

            if not pipe_ok or stop_event.is_set():
                if proc and not force_reconnect_event.is_set():
                    log.warning("ffmpeg pipe closed — restarting.")
                    reconnecting_event.set()
                    proc.kill()
                    proc.wait()
                    proc = None
                    time.sleep(0.5)
                continue

            reconnecting_event.clear()
            frame = np.frombuffer(buf, dtype=np.uint8).reshape((PIPE_H, PIPE_W, 3)).copy()
            item  = (time.monotonic(), frame)
            if not frame_queue.full():
                frame_queue.put_nowait(item)
            else:
                try:
                    frame_queue.get_nowait()
                    frame_queue.put_nowait(item)
                except queue.Empty:
                    pass
    finally:
        if proc:
            proc.kill()
            proc.wait()


def resize_into(image, buf, win_w, win_h):
    h, w = image.shape[:2]
    aspect = w / h
    try:
        if win_w / aspect <= win_h:
            new_w = max(win_w, 1)
            new_h = max(int(win_w / aspect), 1)
        else:
            new_h = max(win_h, 1)
            new_w = max(int(win_h * aspect), 1)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        buf[:] = 0
        y_off = (win_h - new_h) // 2
        x_off = (win_w - new_w) // 2
        buf[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return buf
    except Exception as e:
        log.error("Failed to resize frame: %s", e)
        return None


def _camera_reachable(url, timeout=2.0):
    try:
        from urllib.parse import urlparse
        import socket
        p = urlparse(url)
        host, port = p.hostname, p.port or 554
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except Exception:
        return False


def _draw_label(buf, text):
    cv2.putText(buf, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),      3, cv2.LINE_AA)
    cv2.putText(buf, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_reconnecting(buf, pane_w, pane_h):
    msg = "RECONNECTING..."
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    tx = (pane_w - tw) // 2
    ty = (pane_h + th) // 2
    cv2.rectangle(buf, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10), (0, 0, 0), -1)
    cv2.putText(buf, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2, cv2.LINE_AA)


def _pick_sync_pair(bufs, last_good):
    """
    Return ([frame0, frame1], delta_ms).
    Anchors on the most recent frame from either camera, finds the closest
    matching timestamp in the other camera's buffer.
    Falls back to last_good when a buffer is empty.
    """
    b0, b1 = bufs[0], bufs[1]
    frames   = list(last_good)
    delta_ms = 0.0

    if b0 and b1:
        t0_last, _ = b0[-1]
        t1_last, _ = b1[-1]
        if t0_last >= t1_last:
            _, frames[0] = b0[-1]
            t_match, frames[1] = min(b1, key=lambda x: abs(x[0] - t0_last))
            delta_ms = abs(t0_last - t_match) * 1000
        else:
            _, frames[1] = b1[-1]
            t_match, frames[0] = min(b0, key=lambda x: abs(x[0] - t1_last))
            delta_ms = abs(t1_last - t_match) * 1000
    elif b0:
        _, frames[0] = b0[-1]
    elif b1:
        _, frames[1] = b1[-1]

    return frames, delta_ms


_HELP_LINES = [
    ("1", "CAM1 solo  (again to return)"),
    ("2", "CAM2 solo  (again to return)"),
    ("0", "dual view"),
    ("c", "compare/live solo toggle"),
    ("w", "fit window to frame size"),
    ("f", "fullscreen"),
    ("s", "settings"),
    ("r", "reconnect streams"),
    ("p", "screenshot"),
    ("h", "toggle this help"),
    ("q", "quit"),
]

def _draw_help(img):
    """Draw a semi-transparent keybinding overlay centred on img."""
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.55
    thickness = 1
    pad       = 14
    line_h    = 24

    col_w = max(cv2.getTextSize(k, font, scale, thickness)[0][0] for k, _ in _HELP_LINES)
    val_w = max(cv2.getTextSize(v, font, scale, thickness)[0][0] for _, v in _HELP_LINES)
    box_w = col_w + val_w + pad * 3
    box_h = len(_HELP_LINES) * line_h + pad * 2

    ih, iw = img.shape[:2]
    x0 = (iw - box_w) // 2
    y0 = (ih - box_h) // 2

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)

    for i, (key, desc) in enumerate(_HELP_LINES):
        ty = y0 + pad + (i + 1) * line_h - 4
        cv2.putText(img, key,  (x0 + pad,               ty), font, scale, (100, 220, 255), thickness, cv2.LINE_AA)
        cv2.putText(img, desc, (x0 + pad + col_w + pad, ty), font, scale, (220, 220, 220), thickness, cv2.LINE_AA)


def _pane_heights(win_h):
    th = win_h // 2
    return th, win_h - th


def main(force_settings=False):
    global CURSOR_HIDDEN
    log.info("Starting StreakerDisplay")

    cfg     = _load_config()
    cameras = cfg.get("cameras", [])

    needs_dialog = force_settings or len(cameras) < 2
    if not needs_dialog:
        for cam in cameras:
            url = cam.get("rtsp_url", "")
            if not url or url.startswith("rtsp://user:password") or not _camera_reachable(url):
                needs_dialog = True
                break

    if needs_dialog:
        new_cfg = show_settings_dialog(cfg)
        if new_cfg is None:
            log.info("Settings cancelled — exiting.")
            return
        _save_config(new_cfg)
        cfg = new_cfg

    cameras            = cfg["cameras"]
    window_width       = cfg["window_width"]
    window_height      = cfg["window_height"]
    display_resolution = cfg.get("display_resolution", "Native")
    labels             = [cam.get("label", f"CAM{i + 1}") for i, cam in enumerate(cameras)]

    display_urls = []
    for cam in cameras:
        use_sub = cam.get("use_substream", False)
        sub     = cam.get("substream_url", "").strip()
        url     = sub if (use_sub and sub) else cam["rtsp_url"]
        display_urls.append(url)
        log.info("%s → %s", cam.get("label"), "sub-stream" if (use_sub and sub) else "main stream")

    screen_width, screen_height = get_screen_size()
    win_x = cfg.get("window_x", (screen_width  - window_width)  // 2)
    win_y = cfg.get("window_y", (screen_height - window_height) // 2)
    cv2.namedWindow("Streaker Live Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Streaker Live Feed", window_width, window_height)
    cv2.moveWindow("Streaker Live Feed", win_x, win_y)
    _TITLE_HINTS = ("Streaker Live Feed  |  "
                    "1/2: solo  0: dual  c: compare/live  w: fit  "
                    "f: fullscreen  s: settings  r: reconnect  p: screenshot  h: help  q: quit")
    cv2.setWindowTitle("Streaker Live Feed", _TITLE_HINTS)

    queues          = [queue.Queue(maxsize=QUEUE_SIZE) for _ in cameras]
    reconn          = [threading.Event()               for _ in cameras]
    force_reconnect = [threading.Event()               for _ in cameras]
    cam_stop        = [threading.Event()               for _ in cameras]
    threads         = [None] * len(cameras)

    def _start_cam(i):
        cam_stop[i].clear()
        while not queues[i].empty():
            try: queues[i].get_nowait()
            except queue.Empty: break
        sync_bufs[i].clear()
        reconn[i].set()
        t = threading.Thread(
            target=frame_reader,
            args=(queues[i], cam_stop[i], display_urls[i], reconn[i], force_reconnect[i]),
            kwargs={'display_resolution': display_resolution},
            daemon=True,
        )
        t.start()
        return t

    def _stop_cam(i):
        cam_stop[i].set()
        if threads[i] and threads[i].is_alive():
            threads[i].join(timeout=3)
        threads[i] = None

    th, bh     = _pane_heights(window_height)
    pane_bufs  = [np.zeros((th, window_width, 3), dtype=np.uint8),
                  np.zeros((bh, window_width, 3), dtype=np.uint8)]
    combined   = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    buf_key    = (window_height, window_width)
    sync_bufs  = [deque(maxlen=SYNC_BUFFER), deque(maxlen=SYNC_BUFFER)]
    last_good  = [None, None]
    delta_ms   = 0.0

    for i in range(len(cameras)):
        threads[i] = _start_cam(i)

    screenshot_dir = Path(__file__).parent / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)

    last_display_time = 0.0
    frame_interval    = 1.0 / 24

    try:
        fullscreen_mode      = False
        _prev_fullscreen     = None   # track last applied state to avoid redundant setWindowProperty
        reopen_settings      = False
        solo         = 0      # 0=both, 1=CAM1 only, 2=CAM2 only
        compare_mode = True   # True=both streams run (instant switch); False=hidden cam pauses (saves CPU)
        show_help    = False

        def _apply_solo(new_solo):
            nonlocal solo
            solo = new_solo
            if compare_mode:
                return
            if new_solo == 0:
                # returning to dual — restart any paused cameras
                for i in range(len(cameras)):
                    if not threads[i] or not threads[i].is_alive():
                        threads[i] = _start_cam(i)
            else:
                # going solo — pause the hidden camera, ensure active is running
                hidden = new_solo % 2        # solo=1 → hidden idx 1; solo=2 → hidden idx 0
                active = 1 - hidden
                if threads[hidden] and threads[hidden].is_alive():
                    _stop_cam(hidden)
                if not threads[active] or not threads[active].is_alive():
                    threads[active] = _start_cam(active)

        while True:
            if reopen_settings:
                reopen_settings = False
                for i in range(len(cameras)):
                    cam_stop[i].set()
                for t in threads:
                    if t: t.join(timeout=2)
                cv2.destroyAllWindows()
                if CURSOR_HIDDEN and sys.platform == "win32":
                    ctypes.windll.user32.ShowCursor(True)
                    CURSOR_HIDDEN = False
                main(force_settings=True)
                return

            if cv2.getWindowProperty("Streaker Live Feed", cv2.WND_PROP_VISIBLE) < 1:
                log.info("Window closed by user.")
                break

            # Only apply fullscreen property when state actually changes — calling
            # setWindowProperty every frame floods the Windows message queue.
            if fullscreen_mode != _prev_fullscreen:
                if fullscreen_mode:
                    cv2.setWindowProperty("Streaker Live Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    if not CURSOR_HIDDEN and sys.platform == "win32":
                        ctypes.windll.user32.ShowCursor(False)
                        CURSOR_HIDDEN = True
                else:
                    cv2.setWindowProperty("Streaker Live Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    if CURSOR_HIDDEN and sys.platform == "win32":
                        ctypes.windll.user32.ShowCursor(True)
                        CURSOR_HIDDEN = False
                _prev_fullscreen = fullscreen_mode

            _, _, win_w, win_h = cv2.getWindowImageRect("Streaker Live Feed")
            if win_w <= 0 or win_h <= 0:
                cv2.waitKey(1)
                continue

            if (win_h, win_w) != buf_key:
                th, bh    = _pane_heights(win_h)
                pane_bufs = [np.zeros((th, win_w, 3), dtype=np.uint8),
                             np.zeros((bh, win_w, 3), dtype=np.uint8)]
                combined  = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                buf_key   = (win_h, win_w)

            # Drain incoming frames into sync buffers
            for i in range(len(cameras)):
                try:
                    while True:
                        sync_bufs[i].append(queues[i].get_nowait())
                except queue.Empty:
                    pass

            now       = time.monotonic()
            do_render = now - last_display_time >= frame_interval
            ph        = (th, bh)

            if do_render:
                frames, delta_ms = _pick_sync_pair(sync_bufs, last_good)
                for i, frame in enumerate(frames):
                    if frame is not None:
                        last_good[i] = frame

                if solo != 0:
                    # Single-camera fullscreen
                    idx = solo - 1
                    frame = frames[idx] if frames[idx] is not None else \
                            np.zeros((10, 10, 3), dtype=np.uint8)
                    res = resize_into(frame, combined, win_w, win_h)
                    if res is not None:
                        _draw_label(combined, labels[idx])
                        if reconn[idx].is_set():
                            _draw_reconnecting(combined, win_w, win_h)
                else:
                    # Dual stacked view
                    for i, frame in enumerate(frames):
                        draw_frame = frame if frame is not None else \
                                     np.zeros((10, 10, 3), dtype=np.uint8)
                        res = resize_into(draw_frame, pane_bufs[i], win_w, ph[i])
                        if res is not None:
                            _draw_label(pane_bufs[i], labels[i])
                            if reconn[i].is_set():
                                _draw_reconnecting(pane_bufs[i], win_w, ph[i])

                    combined[:th, :] = pane_bufs[0]
                    combined[th:, :] = pane_bufs[1]

                    # Horizontal divider + sync delta indicator
                    cv2.line(combined, (0, th), (win_w, th), (64, 64, 64), 2)
                    sync_text = f"{delta_ms:.0f}ms"
                    (tsw, tsh), _ = cv2.getTextSize(sync_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    tx = win_w - tsw - 8
                    ty = th - 4
                    sync_color = ((0, 220, 0)   if delta_ms < 50  else
                                  (0, 165, 255) if delta_ms < 200 else
                                  (0, 0, 220))
                    cv2.putText(combined, sync_text, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),  2, cv2.LINE_AA)
                    cv2.putText(combined, sync_text, (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, sync_color, 1, cv2.LINE_AA)

                # Mode indicator — always visible, top-right corner
                mode_txt   = "COMPARE" if compare_mode else "LIVE"
                mode_color = (0, 180, 255) if compare_mode else (0, 210, 0)
                (mw, mh), _ = cv2.getTextSize(mode_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                mx, my = win_w - mw - 8, 20
                cv2.putText(combined, mode_txt, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),    2, cv2.LINE_AA)
                cv2.putText(combined, mode_txt, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color,   1, cv2.LINE_AA)

                if show_help:
                    _draw_help(combined)

                if fullscreen_mode:
                    fs_txt = "f: exit fullscreen"
                    (fsw, fsh), _ = cv2.getTextSize(fs_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.putText(combined, fs_txt, (win_w - fsw - 8, win_h - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),       2, cv2.LINE_AA)
                    cv2.putText(combined, fs_txt, (win_w - fsw - 8, win_h - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

                cv2.imshow("Streaker Live Feed", combined)
                last_display_time = now

            # Sleep for remaining time until next frame — avoids spinning at ~1000fps
            # which wastes CPU and can starve the Windows message pump.
            elapsed  = time.monotonic() - last_display_time
            wait_ms  = max(1, int((frame_interval - elapsed) * 1000))
            key = cv2.waitKey(min(wait_ms, 16))
            if key & 0xFF == ord("q"):
                log.info("Quit key pressed.")
                break
            elif key & 0xFF == ord("f"):
                fullscreen_mode = not fullscreen_mode
            elif key & 0xFF == ord("s"):
                log.info("Settings key pressed.")
                reopen_settings = True
            elif key & 0xFF == ord("1"):
                new = 0 if solo == 1 else 1
                _apply_solo(new)
                log.info("Solo: %s", labels[0] if solo == 1 else "both")
            elif key & 0xFF == ord("2"):
                new = 0 if solo == 2 else 2
                _apply_solo(new)
                log.info("Solo: %s", labels[1] if solo == 2 else "both")
            elif key & 0xFF == ord("0"):
                _apply_solo(0)
                log.info("Solo: both")
            elif key & 0xFF == ord("c"):
                compare_mode = not compare_mode
                log.info("Solo mode: %s", "compare" if compare_mode else "live")
                if solo != 0:
                    hidden = solo % 2
                    if compare_mode:
                        # switched to compare — restart hidden cam if paused
                        if not threads[hidden] or not threads[hidden].is_alive():
                            threads[hidden] = _start_cam(hidden)
                    else:
                        # switched to live — pause hidden cam
                        if threads[hidden] and threads[hidden].is_alive():
                            _stop_cam(hidden)
            elif key & 0xFF == ord("h"):
                show_help = not show_help
            elif key & 0xFF == ord("w"):
                ref = last_good[solo - 1] if solo != 0 else next(
                    (f for f in last_good if f is not None), None)
                if ref is not None:
                    fh, fw = ref.shape[:2]
                    max_h  = screen_height - 80
                    if solo != 0:
                        new_w, new_h = fw, fh
                    else:
                        # dual stacked: ideal is fw x fh*2, but cap to screen height
                        ideal_h = fh * 2
                        if ideal_h <= max_h:
                            new_w, new_h = fw, ideal_h
                        else:
                            # scale down proportionally so it fits on one screen
                            scale  = max_h / ideal_h
                            new_w  = int(fw * scale) & ~1
                            new_h  = max_h
                    cv2.resizeWindow("Streaker Live Feed", new_w, new_h)
                    log.info("Fit window to frame: %dx%d", new_w, new_h)
            elif key & 0xFF == ord("r"):
                for i, ev in enumerate(force_reconnect):
                    if threads[i] and threads[i].is_alive():
                        ev.set()
                log.info("Force-reconnect all streams.")
            elif key & 0xFF == ord("p"):
                ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
                shot_path = screenshot_dir / f"screenshot_{ts}.jpg"
                cv2.imwrite(str(shot_path), combined)
                log.info("Screenshot saved: %s", shot_path)

    finally:
        # Save window position and size so next launch opens in the same place
        try:
            rx, ry, rw, rh = cv2.getWindowImageRect("Streaker Live Feed")
            if rw > 0 and rh > 0:
                saved = _load_config()
                saved["window_x"]      = rx
                saved["window_y"]      = ry
                saved["window_width"]  = rw
                saved["window_height"] = rh
                _save_config(saved)
        except Exception as e:
            log.warning("Could not save window position: %s", e)
        for i in range(len(cameras)):
            cam_stop[i].set()
        for t in threads:
            if t: t.join(timeout=3)
        cv2.destroyAllWindows()
        if CURSOR_HIDDEN and sys.platform == "win32":
            ctypes.windll.user32.ShowCursor(True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.critical("Unhandled exception:\n%s", traceback.format_exc())
        sys.exit(1)
