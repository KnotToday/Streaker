"""
ErrorGrab — press Ctrl+Alt+G over any window to copy its text to clipboard.
Works on standard Win32 dialogs. Run as administrator for best results.
"""
import win32gui
import win32clipboard
import keyboard


def _collect_texts(hwnd):
    texts = []
    seen = set()

    def _visit(h, _):
        t = win32gui.GetWindowText(h)
        if t.strip() and t not in seen:
            seen.add(t)
            texts.append(t.strip())

    title = win32gui.GetWindowText(hwnd)
    if title.strip():
        seen.add(title)
        texts.insert(0, title.strip())

    try:
        win32gui.EnumChildWindows(hwnd, _visit, None)
    except Exception:
        pass

    return texts


def _copy(text):
    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
    finally:
        win32clipboard.CloseClipboard()


def grab():
    hwnd = win32gui.GetForegroundWindow()
    texts = _collect_texts(hwnd)
    if not texts:
        print("No text found in active window.")
        return
    result = "\n".join(texts)
    _copy(result)
    print(f"Copied ({len(texts)} elements):\n{result}\n{'-'*40}")


print("ErrorGrab running — Ctrl+Alt+G copies active window text to clipboard.")
print("Press Ctrl+C to quit.\n")
keyboard.add_hotkey("ctrl+alt+g", grab)
keyboard.wait()
