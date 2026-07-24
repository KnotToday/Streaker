"""Cross-platform helpers for FFmpeg discovery and OS-specific features."""

import os
import sys
import shutil
import subprocess


def find_ffmpeg():
    """Discover ffmpeg: bundled exe → local bin/ → FFMPEG_PATH env var → known Windows location → system PATH."""
    # PyInstaller bundle: ffmpeg.exe is extracted alongside the app at runtime
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundled = os.path.join(sys._MEIPASS, 'ffmpeg.exe')
        if os.path.isfile(bundled):
            return bundled
    # bin/ subfolder next to this file (used when running from source)
    here = os.path.dirname(os.path.abspath(__file__))
    local_bin = os.path.join(here, 'bin', 'ffmpeg.exe')
    if os.path.isfile(local_bin):
        return local_bin
    env = os.environ.get('FFMPEG_PATH') or os.environ.get('FFMPEG')
    if env and os.path.isfile(env):
        return env
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
    """Return ['-hwaccel', 'cuda'] only if a CUDA device is actually usable, else []."""
    try:
        result = subprocess.run(
            [ffmpeg_path, '-hwaccels'],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        if 'cuda' not in result.stdout.lower():
            return []
        # Verify a CUDA device is actually present by attempting a null decode
        probe = subprocess.run(
            [ffmpeg_path, '-hwaccel', 'cuda', '-f', 'lavfi', '-i', 'nullsrc=s=64x64:d=1',
             '-f', 'null', '-'],
            capture_output=True, timeout=8,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        if probe.returncode == 0:
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


def find_python():
    """Return a real Python interpreter path, even when running as a frozen exe."""
    # Always prefer Environ1 venv — project dependencies (astropy, cv2, etc.) live there
    environ1 = os.path.join(os.path.expanduser('~'),
                            'PythonTrials', 'Environ1', 'Scripts', 'python.exe')
    if os.path.isfile(environ1):
        return environ1
    for name in ('python.exe', 'python3.exe', 'python', 'python3'):
        found = shutil.which(name)
        if found:
            return found
    if sys.platform == 'win32':
        try:
            import winreg
            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                for reg_path in (r'SOFTWARE\Python\PythonCore',
                                 r'SOFTWARE\WOW6432Node\Python\PythonCore'):
                    try:
                        with winreg.OpenKey(hive, reg_path) as key:
                            i = 0
                            while True:
                                try:
                                    version = winreg.EnumKey(key, i)
                                    with winreg.OpenKey(key, version + r'\InstallPath') as ikey:
                                        install_dir = winreg.QueryValueEx(ikey, '')[0]
                                        candidate = os.path.join(install_dir, 'python.exe')
                                        if os.path.isfile(candidate):
                                            return candidate
                                except OSError:
                                    break
                                i += 1
                    except OSError:
                        pass
        except ImportError:
            pass
    return sys.executable  # last resort


def find_companion_script(script_name):
    """Return the path to a companion .py script whether running frozen or from source."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, script_name)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, script_name)


def launch_companion(script_name, extra_args=None):
    """Launch a companion script as a no-console subprocess.

    Frozen exe: re-invokes sys.executable with --companion <script_name>.
    Development: runs the .py file with PYTHON_EXE.
    Either way there is no console window and the child inherits a clean env.
    """
    extra = extra_args or []
    if getattr(sys, 'frozen', False):
        cmd = [sys.executable, '--companion', script_name] + extra
    else:
        cmd = [PYTHON_EXE, find_companion_script(script_name)] + extra
    return subprocess.Popen(cmd, creationflags=NO_WINDOW, env=clean_env())


def clean_env():
    """Return an environment dict safe for launching a standalone Python GUI subprocess.

    When running as a PyInstaller frozen exe the bootloader injects TCL_LIBRARY,
    TK_LIBRARY, PYTHONPATH, and PYTHONHOME pointing at the _MEI* temp dir.
    Child processes inherit those, which breaks tkinter in the child.  Strip them.
    """
    env = os.environ.copy()
    for key in ('TCL_LIBRARY', 'TK_LIBRARY', 'TK_DATA', 'TCL_DATA',
                'PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    return env


FFMPEG_PATH = find_ffmpeg()
HWACCEL_ARGS = _detect_hwaccel(FFMPEG_PATH)
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
PYTHON_EXE = find_python()
