"""Cross-platform helpers for FFmpeg discovery and OS-specific features."""

import os
import sys
import shutil
import subprocess


def find_ffmpeg():
    """Discover ffmpeg: FFMPEG_PATH env var → system PATH → known Windows location."""
    env = os.environ.get('FFMPEG_PATH') or os.environ.get('FFMPEG')
    if env and os.path.isfile(env):
        return env
    found = shutil.which('ffmpeg')
    if found:
        return found
    if sys.platform == 'win32':
        fallback = (
            r"C:\Program Files\FFMPEG"
            r"\ffmpeg-2024-03-04-git-e30369bc1c-full_build\bin\ffmpeg.exe"
        )
        if os.path.isfile(fallback):
            return fallback
    return 'ffmpeg'  # last resort: let subprocess resolve from PATH


def find_ffprobe(ffmpeg_path):
    """Derive ffprobe path from ffmpeg path, without assuming .exe extension."""
    directory = os.path.dirname(os.path.abspath(ffmpeg_path))
    binary = 'ffprobe.exe' if sys.platform == 'win32' else 'ffprobe'
    candidate = os.path.join(directory, binary)
    if os.path.isfile(candidate):
        return candidate
    found = shutil.which('ffprobe')
    return found if found else 'ffprobe'


def _detect_hwaccel(ffmpeg_path):
    """Return ['-hwaccel', 'cuda'] if CUDA is available, else []."""
    try:
        result = subprocess.run(
            [ffmpeg_path, '-hwaccels'],
            capture_output=True, text=True, timeout=5
        )
        if 'cuda' in result.stdout.lower():
            return ['-hwaccel', 'cuda']
    except Exception:
        pass
    return []


def play_completion_sound():
    """Play a system alert sound on Windows; no-op on other platforms."""
    if sys.platform == 'win32':
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            pass


FFMPEG_PATH = find_ffmpeg()
HWACCEL_ARGS = _detect_hwaccel(FFMPEG_PATH)
