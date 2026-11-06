"""Low-level Windows input synthesis via ctypes (zero external dependencies).

Provides DPI-aware cursor positioning, mouse buttons, wheel scrolling,
key combos and Unicode text typing through user32 SendInput — the lowest
latency path available to user-mode code.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import logging
import threading
import time

log = logging.getLogger(__name__)

user32 = ctypes.windll.user32

# ---- DPI awareness (must run before any coordinate query) -----------------
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
except Exception:
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass

# ---- SendInput structures ---------------------------------------------------
INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, KEYEVENTF_EXTENDEDKEY = 0x0002, 0x0004, 0x0001
MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP = 0x0020, 0x0040
MOUSEEVENTF_WHEEL, MOUSEEVENTF_HWHEEL = 0x0800, 0x1000

ULONG_PTR = ctypes.POINTER(ctypes.c_ulong)


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _INPUTUNION)]


def _send(*inputs: INPUT) -> None:
    arr = (INPUT * len(inputs))(*inputs)
    user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))


def _mouse_input(flags: int, data: int = 0) -> INPUT:
    inp = INPUT(type=INPUT_MOUSE)
    inp.u.mi = MOUSEINPUT(0, 0, data, flags, 0, None)
    return inp


def _key_input(vk: int = 0, scan: int = 0, flags: int = 0) -> INPUT:
    inp = INPUT(type=INPUT_KEYBOARD)
    inp.u.ki = KEYBDINPUT(vk, scan, flags, 0, None)
    return inp


# ---- Virtual key map --------------------------------------------------------
VK = {
    "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "return": 0x0D, "shift": 0x10,
    "ctrl": 0x11, "control": 0x11, "alt": 0x12, "pause": 0x13, "capslock": 0x14,
    "esc": 0x1B, "escape": 0x1B, "space": 0x20, "pageup": 0x21, "pagedown": 0x22,
    "end": 0x23, "home": 0x24, "left": 0x25, "up": 0x26, "right": 0x27,
    "down": 0x28, "printscreen": 0x2C, "insert": 0x2D, "delete": 0x2E,
    "win": 0x5B, "windows": 0x5B, "apps": 0x5D,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
    "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    "volume_mute": 0xAD, "volume_down": 0xAE, "volume_up": 0xAF,
    "media_next": 0xB0, "media_prev": 0xB1, "media_stop": 0xB2,
    "media_play_pause": 0xB3,
    "comma": 0xBC, "period": 0xBE, "minus": 0xBD, "plus": 0xBB,
}
for _c in "abcdefghijklmnopqrstuvwxyz":
    VK[_c] = ord(_c.upper())
for _d in "0123456789":
    VK[_d] = ord(_d)

_EXTENDED = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2C, 0x2D, 0x2E, 0x5B}

_lock = threading.Lock()


# ---- Public API -------------------------------------------------------------
def screen_size() -> tuple[int, int]:
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def cursor_pos() -> tuple[int, int]:
    pt = wt.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def move_to(x: float, y: float) -> None:
    user32.SetCursorPos(int(round(x)), int(round(y)))


def _button_flags(button: str) -> tuple[int, int]:
    return {
        "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
        "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
        "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
    }[button]


def click(button: str = "left") -> None:
    down, up = _button_flags(button)
    with _lock:
        _send(_mouse_input(down), _mouse_input(up))


def double_click(button: str = "left") -> None:
    click(button)
    time.sleep(0.04)
    click(button)


def mouse_down(button: str = "left") -> None:
    with _lock:
        _send(_mouse_input(_button_flags(button)[0]))


def mouse_up(button: str = "left") -> None:
    with _lock:
        _send(_mouse_input(_button_flags(button)[1]))


def scroll(clicks: float) -> None:
    """Positive scrolls up. One click = one wheel notch."""
    with _lock:
        _send(_mouse_input(MOUSEEVENTF_WHEEL, int(clicks * 120) & 0xFFFFFFFF))


def key_down(name: str) -> None:
    vk = VK[name.lower()]
    flags = KEYEVENTF_EXTENDEDKEY if vk in _EXTENDED else 0
    with _lock:
        _send(_key_input(vk=vk, flags=flags))


def key_up(name: str) -> None:
    vk = VK[name.lower()]
    flags = KEYEVENTF_KEYUP | (KEYEVENTF_EXTENDEDKEY if vk in _EXTENDED else 0)
    with _lock:
        _send(_key_input(vk=vk, flags=flags))


def key_combo(names: list[str], hold_s: float = 0.03) -> None:
    """Press keys in order, release in reverse: key_combo(['ctrl','tab'])."""
    try:
        for n in names:
            key_down(n)
        time.sleep(hold_s)
    finally:
        for n in reversed(names):
            key_up(n)


def press(name: str) -> None:
    key_combo([name])


def type_text(text: str, cps: float = 40.0) -> None:
    """Type arbitrary Unicode text (layout-independent)."""
    delay = 1.0 / max(cps, 1.0)
    for ch in text:
        if ch == "\n":
            press("enter")
        else:
            code = ord(ch)
            with _lock:
                _send(_key_input(scan=code, flags=KEYEVENTF_UNICODE),
                      _key_input(scan=code, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
        time.sleep(delay)


# ---- Window management ------------------------------------------------------
def list_windows() -> list[tuple[int, str]]:
    """Visible top-level windows as (hwnd, title)."""
    result: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def _cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if buf.value.strip():
                    result.append((hwnd, buf.value))
        return True

    user32.EnumWindows(_cb, 0)
    return result


def focus_window(title_substring: str) -> bool:
    """Bring the first window whose title contains the substring to front."""
    needle = title_substring.lower()
    for hwnd, title in list_windows():
        if needle in title.lower():
            SW_RESTORE = 9
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, SW_RESTORE)
            # ALT keypress unlocks SetForegroundWindow from background processes
            key_combo(["alt"], hold_s=0.01)
            user32.SetForegroundWindow(hwnd)
            return True
    return False
