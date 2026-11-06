"""Maps parsed intents (and head gestures) to concrete system actions.

Single dispatch point for everything that touches the OS, so the metrics
layer can log every action and the UI can show a live action feed.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Callable, Optional

from app.core.types import VoiceCommand, now
from app.events import bus
from app.system import apps, win_input
from app.voice.intent import Intent, normalize, parse

log = logging.getLogger(__name__)
_HELLO_RE = re.compile(r"^hello (one|won|wan|1)[.!?]?$")


class ActionExecutor:
    """Executes intents; also handles head-gesture actions (tab/app switching)."""

    def __init__(self, cfg, cursor=None) -> None:
        self._cfg = cfg
        self._cursor = cursor           # CursorController (for pause/resume)
        self.dictation = False
        self._pipeline_hooks: dict[str, Callable] = {}

    def register_hook(self, name: str, fn: Callable) -> None:
        """Pipeline registers callbacks: calibrate, set_center, set_neutral, report."""
        self._pipeline_hooks[name] = fn

    # ---- gesture actions --------------------------------------------------
    GESTURE_ACTIONS: dict[str, Callable[[], None]] = {
        "tab_next": lambda: win_input.key_combo(["ctrl", "tab"]),
        "tab_prev": lambda: win_input.key_combo(["ctrl", "shift", "tab"]),
        "app_next": lambda: win_input.key_combo(["alt", "tab"], hold_s=0.08),
        "app_prev": lambda: win_input.key_combo(["alt", "shift", "tab"], hold_s=0.08),
    }

    def run_gesture(self, gesture_name: str) -> str:
        action = self._cfg.get(f"headpose.mapping.{gesture_name}", "")
        fn = self.GESTURE_ACTIONS.get(action)
        if fn is None:
            return ""
        fn()
        bus.publish("action.executed", {"source": "gesture", "gesture": gesture_name,
                                        "action": action, "t": now()})
        return action

    # ---- voice ---------------------------------------------------------------
    def handle_transcript(self, text: str, engine: str) -> VoiceCommand:
        cmd = VoiceCommand(text=text, engine=engine, t=now())
        # "hello one" always opens the dashboard, regardless of wake_word/dictation
        # state — it's the one phrase that must never be swallowed.
        if _HELLO_RE.match(normalize(text)):
            cmd.intent = "open_dashboard"
            cmd.reply = self._execute(Intent(name="open_dashboard", args={}))
            cmd.handled = True
            bus.publish("voice.command", cmd)
            return cmd
        if self.dictation:
            low = text.lower().strip()
            if "stop dictation" in low or "stop dictating" in low:
                self.dictation = False
                cmd.intent, cmd.handled, cmd.reply = "stop_dictation", True, "Dictation off"
            else:
                win_input.type_text(text + " ")
                cmd.intent, cmd.handled, cmd.reply = "dictation", True, ""
            bus.publish("voice.command", cmd)
            return cmd

        wake = (self._cfg.get("voice.wake_word", "") or "").strip().lower()
        if wake:
            low = text.lower()
            if wake not in low:
                cmd.intent = "ignored"
                bus.publish("voice.command", cmd)
                return cmd
            text = low.split(wake, 1)[1].strip() or text

        intent = parse(text)
        if intent is None:
            cmd.intent, cmd.reply = "unknown", "Sorry, I didn't get that"
        else:
            cmd.intent = intent.name
            cmd.args = intent.args
            try:
                cmd.reply = self._execute(intent)
                cmd.handled = True
            except Exception as e:  # noqa: BLE001
                log.exception("action failed")
                cmd.reply = f"Failed: {e}"
        bus.publish("voice.command", cmd)
        return cmd

    # ---- intent dispatch -------------------------------------------------
    def _execute(self, it: Intent) -> str:  # noqa: C901 - a dispatch table is fine
        n, a = it.name, it.args
        W = win_input
        if n == "click":
            W.click("left"); return "Click"
        if n == "double_click":
            W.double_click(); return "Double click"
        if n == "right_click":
            W.click("right"); return "Right click"
        if n == "middle_click":
            W.click("middle"); return "Middle click"
        if n == "scroll":
            amount = a.get("clicks", 3)
            W.scroll(amount if a.get("dir") == "up" else -amount)
            return f"Scrolling {a.get('dir')}"
        if n == "page":
            W.press("pageup" if a.get("dir") == "up" else "pagedown")
            return f"Page {a.get('dir')}"
        if n == "top":
            W.key_combo(["ctrl", "home"]); return "Top"
        if n == "bottom":
            W.key_combo(["ctrl", "end"]); return "Bottom"
        if n == "back":
            W.key_combo(["alt", "left"]); return "Back"
        if n == "forward":
            W.key_combo(["alt", "right"]); return "Forward"
        if n == "refresh":
            W.press("f5"); return "Refreshing"
        if n == "zoom_in":
            W.key_combo(["ctrl", "plus"]); return "Zoom in"
        if n == "zoom_out":
            W.key_combo(["ctrl", "minus"]); return "Zoom out"
        if n == "new_tab":
            W.key_combo(["ctrl", "t"]); return "New tab"
        if n == "close_tab":
            W.key_combo(["ctrl", "w"]); return "Tab closed"
        if n == "reopen_tab":
            W.key_combo(["ctrl", "shift", "t"]); return "Tab reopened"
        if n == "tab_next":
            W.key_combo(["ctrl", "tab"]); return "Next tab"
        if n == "tab_prev":
            W.key_combo(["ctrl", "shift", "tab"]); return "Previous tab"
        if n == "app_next":
            W.key_combo(["alt", "tab"], hold_s=0.08); return "Next app"
        if n == "app_prev":
            W.key_combo(["alt", "shift", "tab"], hold_s=0.08); return "Previous app"
        if n == "minimize":
            W.key_combo(["win", "down"]); return "Minimized"
        if n == "maximize":
            W.key_combo(["win", "up"]); return "Maximized"
        if n == "show_desktop":
            W.key_combo(["win", "d"]); return "Desktop"
        if n == "close_window":
            W.key_combo(["alt", "f4"]); return "Window closed"
        if n == "snap_left":
            W.key_combo(["win", "left"]); return "Snapped left"
        if n == "snap_right":
            W.key_combo(["win", "right"]); return "Snapped right"
        if n == "switch_to":
            name = a.get("name", "")
            if W.focus_window(name):
                return f"Switched to {name}"
            ok, what = apps.launch(name)
            return f"Opening {what}" if ok else f"Couldn't find {name}"
        if n == "open_app":
            ok, what = apps.launch(a.get("name", ""))
            return f"Opening {what}" if ok else f"Couldn't open {a.get('name')}"
        if n == "open_url":
            apps.open_url(a.get("url", "")); return "Opening site"
        if n == "search_web":
            apps.search_web(a.get("q", "")); return f"Searching for {a.get('q')}"
        if n == "type_text":
            W.type_text(a.get("text", "")); return "Typed"
        if n == "press_key":
            key = a.get("key", "").replace(" ", "")
            if key.lower() in W.VK:
                W.press(key.lower()); return f"Pressed {key}"
            return f"Unknown key {key}"
        if n == "press_enter":
            W.press("enter"); return "Enter"
        if n == "press_escape":
            W.press("escape"); return "Escape"
        if n == "select_all":
            W.key_combo(["ctrl", "a"]); return "Selected all"
        if n == "copy":
            W.key_combo(["ctrl", "c"]); return "Copied"
        if n == "paste":
            W.key_combo(["ctrl", "v"]); return "Pasted"
        if n == "cut":
            W.key_combo(["ctrl", "x"]); return "Cut"
        if n == "undo":
            W.key_combo(["ctrl", "z"]); return "Undone"
        if n == "redo":
            W.key_combo(["ctrl", "y"]); return "Redone"
        if n == "save":
            W.key_combo(["ctrl", "s"]); return "Saved"
        if n == "delete":
            W.press("delete"); return "Deleted"
        if n == "volume_up":
            for _ in range(4): W.press("volume_up")
            return "Volume up"
        if n == "volume_down":
            for _ in range(4): W.press("volume_down")
            return "Volume down"
        if n == "mute":
            W.press("volume_mute"); return "Toggled mute"
        if n == "play_pause":
            W.press("media_play_pause"); return "Play/pause"
        if n == "media_next":
            W.press("media_next"); return "Next track"
        if n == "media_prev":
            W.press("media_prev"); return "Previous track"
        if n == "screenshot":
            W.key_combo(["win", "printscreen"]); return "Screenshot saved to Pictures"
        if n == "lock_screen":
            import ctypes
            ctypes.windll.user32.LockWorkStation(); return "Locked"
        if n == "start_dictation":
            self.dictation = True; return "Dictation on — everything you say will be typed"
        if n == "pause_control":
            if self._cursor: self._cursor.pause()
            bus.publish("control.paused", None); return "Eye control paused"
        if n == "resume_control":
            if self._cursor: self._cursor.resume()
            bus.publish("control.resumed", None); return "Eye control resumed"
        if n in ("calibrate", "set_center", "set_neutral", "generate_report"):
            hook = self._pipeline_hooks.get(n)
            if hook:
                threading.Thread(target=hook, daemon=True).start()
                return {"calibrate": "Starting calibration",
                        "set_center": "Center set",
                        "set_neutral": "Neutral pose set",
                        "generate_report": "Generating report"}[n]
            return "Not available"
        if n == "open_dashboard":
            hook = self._pipeline_hooks.get("open_dashboard")
            if hook:
                hook()
                return "Hello! Opening your HandsFree dashboard."
            return "Dashboard hook not available"
        if n == "help":
            return ("Try: open chrome, search for cats, click, scroll down, "
                    "next tab, switch to notepad, type hello, volume up, "
                    "start dictation, pause control")
        return "Unknown command"
