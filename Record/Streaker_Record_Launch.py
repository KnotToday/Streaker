import sys
import os
import shutil
import subprocess
import builtins
import traceback
from datetime import datetime, timedelta, date, timezone
from tkinter import (
    Tk, Label, Button, Entry, filedialog, StringVar,
    DoubleVar, IntVar, messagebox, Frame
)
from threading import Thread, Event
import time
import json
from astral import LocationInfo
from astral.sun import sun, elevation, dawn, dusk
import pytz


def _find_ffmpeg():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundled = os.path.join(sys._MEIPASS, 'ffmpeg.exe')
        if os.path.isfile(bundled):
            return bundled
    found = shutil.which('ffmpeg')
    return found if found else 'ffmpeg'

FFMPEG_PATH = _find_ffmpeg()

# === Logging Setup ===

# Determine log base directory (handles .py and .exe)
if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

# Log directory
log_folder = os.path.join(base_dir, "logs")
os.makedirs(log_folder, exist_ok=True)

# Clean logs older than 7 days
now = datetime.now()
for filename in os.listdir(log_folder):
    if filename.startswith("stream_capture_") and filename.endswith(".log"):
        filepath = os.path.join(log_folder, filename)
        try:
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if (now - file_time).days > 7:
                os.remove(filepath)
        except OSError as e:
            print(f"[LOG CLEANUP ERROR] Could not process {filename}: {e}")

# Today's log file
log_filename = now.strftime("stream_capture_%Y-%m-%d.log")
log_path = os.path.join(log_folder, log_filename)

# Redirect stdout and stderr
log_file = open(log_path, "a", buffering=1, encoding='utf-8')
sys.stdout = log_file
sys.stderr = log_file

VERSION = "v1.000"
print(f"[INFO] STREAKERrec {VERSION} starting up")

# Timestamped print
def timestamped_print(*args, **kwargs):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    builtins.print(timestamp, *args, **kwargs)

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(BASE_DIR, "stream_capture_config.json")


    
class StreamCapture:
    def __init__(self, root):
        self.root = root
        self.root.title("Stream Capture - Twilight Timing")
        self.output_dir = StringVar(value=os.path.join(os.path.expanduser("~"), "Dahua_MKV_Streaker"))
        self.rtsp_url = StringVar()
        self.chunk_minutes = IntVar(value=15)
        self.start_time = StringVar()
        self.end_time = StringVar()
        self.latitude = DoubleVar(value=40.7)
        self.longitude = DoubleVar(value=-74.0)
        self.timer_label = StringVar(value="Timer: 00:00:00")
        self.mode_label = StringVar(value="Mode: Not Started")
        self.stop_offset_label = StringVar(value="Stop Offset: 0 min")
        self.start_offset_label = StringVar(value="Start Offset: 0 min")
        self.session_start_date = datetime.now(timezone.utc).strftime("%m-%d-%Y")
        self.stop_time_offset = IntVar(value=0)
        self.start_time_offset = IntVar(value=0)
        self.load_config()
        self.recording_thread = None
        self.stop_event = Event()
        self.recording_process = None

        self.calculate_twilight_times()
        self.build_gui()

    def update_stop_offset_label(self):
        offset = self.stop_time_offset.get()
        sign = "+" if offset >= 0 else ""
        self.stop_offset_label.set(f"Stop Offset: {sign}{offset} min")
        self.save_config()

    def increment_stop_offset(self):
        if self.stop_time_offset.get() < 1439:
            self.stop_time_offset.set(self.stop_time_offset.get() + 1)
            self.calculate_twilight_times()
            self.update_stop_offset_label()

    def decrement_stop_offset(self):
        if self.stop_time_offset.get() > -1439:
            self.stop_time_offset.set(self.stop_time_offset.get() - 1)
            self.calculate_twilight_times()
            self.update_stop_offset_label()

    def reset_stop_offset(self):
        self.stop_time_offset.set(0)
        self.calculate_twilight_times()
        self.update_stop_offset_label()

    def update_start_offset_label(self):
        offset = self.start_time_offset.get()
        sign = "+" if offset >= 0 else ""
        self.start_offset_label.set(f"Start Offset: {sign}{offset} min")

    def increment_start_offset(self):
        if self.start_time_offset.get() < 1439:
            self.start_time_offset.set(self.start_time_offset.get() + 1)
            self.calculate_twilight_times()
            self.update_start_offset_label()

    def decrement_start_offset(self):
        if self.start_time_offset.get() > -1439:
            self.start_time_offset.set(self.start_time_offset.get() - 1)
            self.calculate_twilight_times()
            self.update_start_offset_label()

    def reset_start_offset(self):
        self.start_time_offset.set(0)
        self.calculate_twilight_times()
        self.update_start_offset_label()

    # Aliases for GUI compatibility
    increment_offset = increment_stop_offset
    decrement_offset = decrement_stop_offset
    reset_offset = reset_stop_offset

    def _start_recording(self, start_dt, end_dt):
        def record():
            self.timer_label.set("Recording...")
            chunk_seconds = self.chunk_minutes.get() * 60
            actual_start_time = datetime.now(timezone.utc)
            print(f"[DEBUG] Scheduled recording from {start_dt} to {end_dt} ({(end_dt - start_dt).total_seconds()} seconds)")
            print("[DEBUG] Entering recording loop")
            while True:
                now = datetime.now(timezone.utc)
                print(f"[DEBUG] Current Time: {now}, End Time: {end_dt}")
                if now >= end_dt or self.stop_event.is_set():
                    break
                timestamp = datetime.now(timezone.utc).strftime("%H-%M-%S")
                output_folder = self.create_date_folder(self.output_dir.get())
                filename = f"{self.session_start_date}_{timestamp}.mkv"
                filepath = os.path.join(output_folder, filename)

                cmd = [
                    FFMPEG_PATH, "-hide_banner", "-rtsp_transport", "tcp", "-rtbufsize", "400M",
                    "-i", self.rtsp_url.get(), "-map", "0:v:0", "-use_wallclock_as_timestamps", "1",
                    "-fflags", "+genpts", "-err_detect", "ignore_err", "-c", "copy",
                    "-avoid_negative_ts", "make_zero", "-t", str(chunk_seconds), filepath
                ]

                print(f"[DEBUG] Starting recording: {filepath}")
                self.recording_process = subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = self.recording_process.communicate()
                print(f"[DEBUG] ffmpeg exited with code {self.recording_process.returncode}")
                if self.recording_process.returncode not in (0, 1):
                    print(f"[ERROR] ffmpeg failed (code {self.recording_process.returncode}): {stderr.decode(errors='ignore').strip()}")

            self.timer_label.set("Recording Complete")

        self.stop_event.clear()
        Thread(target=record, daemon=True).start()

    def build_gui(self):
        Label(self.root, text="RTSP Stream URL:").grid(row=0, column=0, sticky="e")
        rtsp_entry = Entry(self.root, textvariable=self.rtsp_url, width=50)
        rtsp_entry.grid(row=0, column=1, columnspan=2)

        # Automatically save config when the user finishes editing the RTSP field
        rtsp_entry.bind("<FocusOut>", lambda e: self.save_config())
        rtsp_entry.bind("<Return>", lambda e: self.save_config())  # Optional: save on Enter


        Label(self.root, text="Output Directory:").grid(row=1, column=0, sticky="e")
        Entry(self.root, textvariable=self.output_dir, width=50).grid(row=1, column=1)
        Button(self.root, text="Browse", command=self.select_output_directory).grid(row=1, column=2)

        Label(self.root, text="Latitude:").grid(row=2, column=0, sticky="e")
        lat_entry = Entry(self.root, textvariable=self.latitude)
        lat_entry.grid(row=2, column=1)
        lat_entry.bind("<FocusOut>", lambda e: self.on_gps_change())
        lat_entry.bind("<Return>", lambda e: self.on_gps_change())

        Label(self.root, text="Longitude:").grid(row=3, column=0, sticky="e")
        lon_entry = Entry(self.root, textvariable=self.longitude)
        lon_entry.grid(row=3, column=1)
        lon_entry.bind("<FocusOut>", lambda e: self.on_gps_change())
        lon_entry.bind("<Return>", lambda e: self.on_gps_change())

        Label(self.root, text="UTC Start Time (HH:MM):").grid(row=4, column=0, sticky="e")
        Entry(self.root, textvariable=self.start_time).grid(row=4, column=1)
        Button(self.root, text="+1 min", command=self.increment_start_offset).grid(row=4, column=2)
        Button(self.root, text="-1 min", command=self.decrement_start_offset).grid(row=4, column=3)
        Button(self.root, text="Reset Offset", command=self.reset_start_offset).grid(row=4, column=4)


        Label(self.root, text="End Time (HH:MM):").grid(row=5, column=0, sticky="e")
        Entry(self.root, textvariable=self.end_time).grid(row=5, column=1)
        Button(self.root, text="+1 min", command=self.increment_offset).grid(row=5, column=2)
        Button(self.root, text="-1 min", command=self.decrement_offset).grid(row=5, column=3)
        Button(self.root, text="Reset Offset", command=self.reset_offset).grid(row=5, column=4)

        Label(self.root, textvariable=self.start_offset_label).grid(row=6, column=0)
        Label(self.root, textvariable=self.stop_offset_label).grid(row=6, column=1)


        Label(self.root, textvariable=self.timer_label, font=("Arial", 14), fg="red").grid(row=7, column=0, columnspan=3)
        Label(self.root, textvariable=self.mode_label, font=("Arial", 10), fg="blue").grid(row=8, column=0, columnspan=3)

        Button(self.root, text="Start Now", command=self.start_now).grid(row=9, column=0)
        Button(self.root, text="Start with Delay", command=self.start_with_delay).grid(row=9, column=1)
        Button(self.root, text="Force Stop", command=self.force_stop_capture).grid(row=9, column=2)

        self.display_twilight_times()

    def select_output_directory(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
            self.save_config()

    def load_config(self):
        """
        Load the configuration from a JSON file at CONFIG_FILE.

        Supported keys:
        - latitude (float): The latitude of the camera.
        - longitude (float): The longitude of the camera.
        - stop_time_offset (int): Offset in minutes from nautical dusk to end recording.
        - start_time_offset (int): Offset in minutes from nautical dawn to start recording.
        - rtsp_url (str): The RTSP URL of the camera.
        """
        print("[DEBUG] Loading config...")

        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    print(f"[DEBUG] Loaded config: {config}")

                    self.latitude.set(config.get("latitude", 40.7))
                    self.longitude.set(config.get("longitude", -74.0))
                    self.stop_time_offset.set(config.get("stop_time_offset", 0))
                    self.start_time_offset.set(config.get("start_time_offset", 0))

                    rtsp = config.get("rtsp_url", "")
                    if rtsp:
                        self.rtsp_url.set(rtsp)
                    else:
                        print("[WARNING] RTSP URL not found in config; leaving blank")

                    output_dir = config.get("output_dir", "")
                    if output_dir:
                        self.output_dir.set(output_dir)
                    else:
                        print("[WARNING] output_dir not found in config; using default")

            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to decode config JSON: {e}")
            except Exception as e:
                print(f"[ERROR] Unexpected error loading config: {e}")
        else:
            print("[INFO] No config file found; using defaults")

        self.update_stop_offset_label()
        self.update_start_offset_label()


    def save_config(self):
        """
        Save the current configuration to a JSON file at CONFIG_FILE.

        Supported keys:
        - latitude (float): The latitude of the camera.
        - longitude (float): The longitude of the camera.
        - stop_time_offset (int): Offset in minutes from nautical dusk to end recording.
        - start_time_offset (int): Offset in minutes from nautical dawn to start recording.
        - rtsp_url (str): The RTSP URL of the camera.
        """
        print("[DEBUG] Saving config... ")
        config = {
            "latitude": self.latitude.get(),
            "longitude": self.longitude.get(),
            "stop_time_offset": self.stop_time_offset.get(),
            "start_time_offset": self.start_time_offset.get(),
            "rtsp_url": self.rtsp_url.get(),
            "output_dir": self.output_dir.get()
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            print(f"[DEBUG] Config saved: {config}")
        except Exception as e:
            print(f"[ERROR] Failed to save config: {e}")


    def calculate_twilight_times(self):
        import datetime as _dt
        try:
            location = LocationInfo("Custom", "World", "UTC", self.latitude.get(), self.longitude.get())
            observer = location.observer
            date_today = date.today()
            tz = _dt.timezone.utc
            self.civil_dusk    = dusk(observer, date_today,  depression=6,  tzinfo=tz)
            self.civil_dawn    = dawn(observer, date_today,  depression=6,  tzinfo=tz)
            self.nautical_dusk = dusk(observer, date_today,  depression=12, tzinfo=tz)
            self.nautical_dawn = dawn(observer, date_today,  depression=12, tzinfo=tz)
            self.start_midpoint = self.civil_dusk + (self.nautical_dusk - self.civil_dusk) / 2
            self.start_midpoint += timedelta(minutes=self.start_time_offset.get())
            self.end_midpoint = self.nautical_dawn + (self.civil_dawn - self.nautical_dawn) / 2 + timedelta(minutes=self.stop_time_offset.get())
            self.start_time.set(self.start_midpoint.strftime("%H:%M"))
            self.end_time.set(self.end_midpoint.strftime("%H:%M"))
        except Exception as e:
            print(f"[ERROR] Twilight calculation failed: {e}")
            now_utc = datetime.now(_dt.timezone.utc)
            self.civil_dusk    = now_utc.replace(hour=20, minute=0,  second=0, microsecond=0)
            self.nautical_dusk = now_utc.replace(hour=20, minute=30, second=0, microsecond=0)
            self.civil_dawn    = now_utc.replace(hour=5,  minute=30, second=0, microsecond=0)
            self.nautical_dawn = now_utc.replace(hour=5,  minute=0,  second=0, microsecond=0)
            self.start_midpoint = self.civil_dusk
            self.end_midpoint   = self.civil_dawn
            self.start_time.set("20:00")
            self.end_time.set("05:30")
            messagebox.showwarning("Twilight Calculation Failed",
                f"Could not calculate twilight times:\n{e}\n\nFalling back to fixed defaults. "
                "Check your latitude/longitude in the config.")

    def display_twilight_times(self):
        local_tz = datetime.now().astimezone().tzinfo
        frame = Frame(self.root)
        frame.grid(row=10, column=0, columnspan=3, sticky="w")

        def fmt(dt):
            return dt.strftime("%H:%M") + " UTC / " + dt.astimezone(local_tz).strftime("%H:%M %Z")

        times = [
            ("Civil Dusk:", fmt(self.civil_dusk)),
            ("Nautical Dusk:", fmt(self.nautical_dusk)),
            ("Start Midpoint:", fmt(self.start_midpoint)),
            ("", ""),  # blank row for spacing
            ("Nautical Dawn:", fmt(self.nautical_dawn)),
            ("Civil Dawn:", fmt(self.civil_dawn)),
            ("End Midpoint:", fmt(self.end_midpoint)),
        ]

        for i, (label, value) in enumerate(times):
            if label == "":
                Label(frame, text="").grid(row=i, column=0, pady=5)  # spacer with extra vertical padding
                continue

            if "Dawn" in label:
                color = "#B8860B"  # Dark goldenrod
            elif "Midpoint" in label:
                color = "#2F4F4F"  # Dark slate gray
            else:
                color = "#333333"  # Dark gray for dusk

            font = ("Arial", 14, "bold") if "Midpoint" in label else ("Arial", 13)
            Label(frame, text=label, fg=color, font=font).grid(row=i, column=0, sticky="w")
            Label(frame, text=value, fg=color, font=font).grid(row=i, column=1, sticky="w")
  
    def start_now(self):
        self.mode_label.set("Mode: Immediate")
        self.calculate_twilight_times()
        start_dt = datetime.now(timezone.utc)
        end_dt = self.end_midpoint.astimezone(timezone.utc)

        while end_dt <= start_dt:
            end_dt += timedelta(days=1)

        self._start_recording(start_dt, end_dt)


    def start_with_delay(self):
        self.mode_label.set("Mode: Delayed")
        try:
            now = datetime.now(timezone.utc)
            today = now.date()
            # Parse input times
            start_clock = datetime.strptime(self.start_time.get(), "%H:%M").time()
            end_clock = datetime.strptime(self.end_time.get(), "%H:%M").time()
            # Combine with today's date
            start_dt = datetime.combine(today, start_clock, tzinfo=timezone.utc)
            end_dt = datetime.combine(today, end_clock, tzinfo=timezone.utc)
            # Only push start_dt to tomorrow if it has already passed today
            if start_dt < now:
                print(f"[DEBUG] Start time {start_dt} already passed (now = {now}), pushing to tomorrow")
                start_dt += timedelta(days=1)
            # If end_dt is earlier than start_dt, push it forward too
            if end_dt <= start_dt:
                print(f"[DEBUG] End time {end_dt} is earlier than start_dt {start_dt}, pushing end time to next day")
                end_dt += timedelta(days=1)
            print(f"[DEBUG] Final Scheduled Start: {start_dt}, End: {end_dt}")
            self._wait_and_start(start_dt, end_dt)
        except ValueError:
            messagebox.showerror("Time Format Error", "Start/End times must be in HH:MM format.")

    def _wait_and_start(self, start_dt, end_dt):
        def thread_func():
            now = datetime.now(timezone.utc)
            if now >= start_dt:
                self._start_recording(now, end_dt)  # use current time if start is past
                return

            while not self.stop_event.is_set():
                now = datetime.now(timezone.utc)
                if now >= start_dt:
                    break
                remaining = start_dt - now
                h, s = divmod(int(remaining.total_seconds()), 60)
                hours, minutes = divmod(h, 60)
                self.timer_label.set(f"Starts in: {hours:02}:{minutes:02}:{int(remaining.total_seconds()%60):02}")
                self.root.update()
                time.sleep(1)

            if not self.stop_event.is_set():
                self._start_recording(start_dt, end_dt)

        self.stop_event.clear()
        Thread(target=thread_func, daemon=True).start()

    def create_date_folder(self, output_dir):
        date_folder = os.path.join(output_dir, self.session_start_date)
        os.makedirs(date_folder, exist_ok=True)
        return date_folder

    def on_gps_change(self):
        try:
            self.latitude.get()
            self.longitude.get()
            self.calculate_twilight_times()
            self.display_twilight_times()
            self.save_config()
        except Exception as e:
            messagebox.showerror("Invalid Coordinates", f"Latitude/Longitude must be valid numbers.\n{e}")

    def find_twilight(self, observer, date_obj, angle, is_dawn):
        import datetime as _dt
        tz = _dt.timezone.utc
        base_time = sun(observer, date_obj, tzinfo=tz)["sunrise"] if is_dawn else sun(observer, date_obj, tzinfo=tz)["sunset"]
        start_time = base_time - timedelta(hours=3) if is_dawn else base_time
        end_time = base_time if is_dawn else base_time + timedelta(hours=3)
        step = timedelta(minutes=1)

        current_time = start_time
        last_elev = elevation(observer, current_time)

        while current_time <= end_time:
            current_time += step
            elev = elevation(observer, current_time)

            if is_dawn and last_elev < angle <= elev:
                return current_time
            if not is_dawn and last_elev > angle >= elev:
                return current_time

            last_elev = elev

        return base_time  # fallback

    def force_stop_capture(self):
        self.stop_event.set()
        self.timer_label.set("Force Stopped")
        if self.recording_process:
            self.recording_process.terminate()


if __name__ == "__main__":
    try:
        root = Tk()
        app = StreamCapture(root)
        root.mainloop()
    except Exception as e:
        print(f"[ERROR] Exception in main loop: {e}")
        traceback.print_exc()
        messagebox.showerror("STREAKERrec Error", f"Unexpected error:\n{e}\n\nSee log for details:\n{log_path}")
