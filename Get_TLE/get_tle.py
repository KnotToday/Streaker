"""
get_tle.py  —  Download TLE data from space-track.org
Replaces Update13_Get_TLE_FIXED_by_gpt.bat + Update6_Get_tle_sans_html.py
Cross-platform: Windows and Linux. No browser/Selenium needed.

First-time setup — create tle_config.json next to this script:
  {
    "username": "you@example.com",
    "password": "yourpassword",
    "save_dir": "TLEz",
    "wifi_prefix": "2.4"
  }

Schedule via Windows Task Scheduler or Linux cron to run once daily.
"""

import os
import sys
import json
import socket
import logging
import subprocess
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests is required:  pip install requests")

BASE_DIR    = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "tle_config.json"
LOG_FILE    = BASE_DIR / f"get_tle_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("get_tle")

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
TLE_URL   = (
    "https://www.space-track.org/basicspacedata/query/class/gp"
    "/EPOCH/%3Enow-30/orderby/NORAD_CAT_ID,EPOCH/format/3le"
)


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        sys.exit(
            f"Config file not found: {CONFIG_FILE}\n"
            'Create it with: {"username": "...", "password": "...", "save_dir": "TLEz"}'
        )
    with open(CONFIG_FILE) as f:
        return json.load(f)


def already_downloaded_today(save_dir: Path) -> bool:
    """True if a TLE file for today already exists that was saved after 20:00."""
    today   = datetime.now().strftime("%Y%m%d")
    cutoff  = datetime.now().replace(hour=20, minute=0, second=0, microsecond=0)
    for f in save_dir.glob(f"{today}_*.tle"):
        if datetime.fromtimestamp(f.stat().st_mtime) >= cutoff:
            log.info(f"Already downloaded today after 20:00: {f.name}")
            return True
    return False


def check_internet(host="8.8.8.8", port=53, timeout=5) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
        return True
    except OSError:
        return False


def try_connect_wifi(prefix: str) -> None:
    """Best-effort WiFi reconnect. Uses netsh on Windows, nmcli on Linux."""
    log.info("No internet — attempting WiFi reconnect...")
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["netsh", "wlan", "show", "profiles"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                if prefix in line and ":" in line:
                    name = line.split(":", 1)[1].strip()
                    subprocess.run(["netsh", "wlan", "connect", f"name={name}"],
                                   timeout=15, check=False)
                    log.info(f"Connected to {name}")
                    return
        else:
            result = subprocess.run(
                ["nmcli", "-t", "-f", "NAME", "connection", "show"],
                capture_output=True, text=True, timeout=10
            )
            for name in result.stdout.splitlines():
                if prefix in name:
                    subprocess.run(["nmcli", "connection", "up", name],
                                   timeout=15, check=False)
                    log.info(f"Connected to {name}")
                    return
        log.warning(f"No WiFi profile matching '{prefix}' found.")
    except Exception as e:
        log.warning(f"WiFi reconnect failed: {e}")


def download_tle(username: str, password: str, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        resp = session.post(
            LOGIN_URL,
            data={"identity": username, "password": password},
            timeout=30,
        )
        resp.raise_for_status()
        log.info("Logged in to space-track.org")

        resp = session.get(TLE_URL, timeout=60)
        resp.raise_for_status()
        content = resp.text

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = save_dir / f"{timestamp}.tle"
    out_path.write_text(content, encoding="utf-8")
    line_count = content.count("\n")
    log.info(f"Saved {line_count} lines -> {out_path.name}")
    return out_path


def main() -> None:
    now = datetime.now()
    log.info(f"get_tle.py starting — {now.strftime('%Y-%m-%d %H:%M:%S')}")

    config   = load_config()
    save_dir = BASE_DIR / config.get("save_dir", "TLEz")

    if now.hour >= 20 and already_downloaded_today(save_dir):
        log.info("Nothing to do.")
        return

    if not check_internet():
        try_connect_wifi(config.get("wifi_prefix", "2.4"))
        if not check_internet():
            log.error("No internet connection. Aborting.")
            sys.exit(1)

    download_tle(config["username"], config["password"], save_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
