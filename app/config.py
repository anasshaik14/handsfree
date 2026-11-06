"""Thread-safe application configuration with JSON persistence.

Access values with dotted paths: cfg.get("gaze.smoothing.mincutoff").
Every runtime-tunable threshold lives here so the web UI can adjust the
system live and settings survive restarts.
"""
from __future__ import annotations

import copy
import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List

log = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "camera": {"index": 0, "width": 640, "height": 480, "fps": 30},
    "gaze": {
        "enabled": True,
        "provider": "custom",                # custom | iris_geometric | iris_polynomial | pupil_classical
        "custom": {"gain_x": 1.6, "gain_y": 1.9},   # gaze-angle -> screen scaling for the custom CNN
        "gain_x": 2.8,                       # geometric mapping sensitivity
        "gain_y": 3.4,
        "pose_assist": 0.35,                 # 0..1 how much head pose steers the geometric cursor
        "margin": 0.0,                       # screen edge margin (fraction)
        "expand": 1.2,                       # stretch around center so corners are reachable
        "smoothing": {
            "filter": "one_euro",            # one_euro | ema | none
            "mincutoff": 0.3,                # Hz - lower = steadier at fixation
            "beta": 0.003,                   # speed coeff (pixel units) - higher = faster response
            "dcutoff": 1.0,
            "ema_alpha": 0.25,
            "deadzone_px": 12.0,             # hysteresis radius that kills residual jitter
            "max_speed_px": 2000.0,          # slew limit (px/second)
        },
        "freeze_on_blink_ms": 180,           # hold cursor around blinks (kills blink jitter)
        "calibration": {
            "points": 16,                    # grid density: 9 | 16 | 25 | 36
            "samples_per_point": 60,         # frames collected & averaged per dot
            "validate": True,                # post-fit quality check on random targets
            "validation_points": 5,          # how many held-out targets to test
            "dynamic": True,                 # online correction on every click
            "dynamic_forget": 0.9995,        # RLS forgetting (closer to 1 = gentler)
            "dynamic_min_quality": 0.5,      # only learn from high-confidence fixations
        },
    },
    "blink": {
        "enabled": True,
        "provider": "custom",                # custom | ear_standard | ear_adaptive | ela
        "ear_threshold": 0.20,               # fixed threshold (standard EAR)
        "adaptive_ratio": 0.70,              # threshold = ratio * per-user open baseline
        "ela_ratio": 0.55,                   # lid-angle closed ratio vs baseline
        "hold_click_ms": 420,                # intentional blink: >= this fires left click
        "hold_right_ms": 1000,               # >= this fires right click
        "hold_double_ms": 1700,              # >= this fires double click
        "cancel_ms": 3000,                   # >= this cancels (eye rest, no action)
        "refractory_ms": 400,                # dead time after an event
        "sound_feedback": True,
    },
    "headpose": {
        "enabled": True,
        "provider": "custom",                # custom | solvepnp | mp_geometry | landmark_heuristic
        "yaw_on_deg": 30.0,                  # gesture trigger threshold (full head turn)
        "yaw_off_deg": 12.0,                 # return-to-neutral threshold
        "pitch_on_deg": 22.0,
        "pitch_off_deg": 9.0,
        "dwell_ms": 650,                     # must hold the pose this long (deliberate)
        "cooldown_ms": 1200,                 # dead time between gestures
        "repeat": False,                     # hold pose to auto-repeat
        "repeat_interval_ms": 900,
        "mapping": {                         # gesture -> action
            "head_left": "app_prev",
            "head_right": "app_next",
            "head_up": "tab_next",
            "head_down": "tab_prev",
        },
        "guard_cursor": True,                # freeze gaze cursor during head gestures
        "nod_shake": {                       # nod = yes/confirm, shake = no/cancel
            "enabled": True,
            "window_s": 1.4,                  # oscillation must complete within this window
            "amplitude_deg": 14.0,            # min swing to count as deliberate (not jitter)
            "min_crossings": 3,               # sign flips required — a full back-and-forth
            "cooldown_s": 1.5,
        },
    },
    "voice": {
        "enabled": True,
        "engine": "custom",                  # custom | vosk | whisper | webspeech
        "custom": {"base_engine": "vosk"},   # ASR used for words the KWS model doesn't cover
        "vosk_model": "models/vosk-model-small-en-us-0.15",
        "whisper_model": "base.en",
        "device": None,                      # sounddevice input device index (None = default)
        "tts_feedback": True,                # dashboard speaks confirmations
        "wake_word": "",                     # empty = always listening
        "webspeech_lang": "en-IN",           # browser recognition locale (accent)
        "autocomplete": True,                # phrase-completion suggestions while dictating
    },
    "hands": {
        "enabled": True,                     # open palm / thumbs up / thumbs down / peace
        "frame_stride": 3,                    # run the hand detector every Nth frame (it costs
                                              # as much as the face mesh; gestures are held anyway)
        "debounce_frames": 3,                 # consecutive frames before a gesture counts
        "repeat_interval_ms": 600,             # thumbs up/down auto-repeat while held
        "quit_dwell_ms": 700,                  # peace sign must hold this long before it quits
    },
    "fatigue": {
        "enabled": True,
        "blink_rate_low": 8,                  # blinks/min below this = fatigue signal
        "blink_rate_high": 30,                # blinks/min above this = fatigue signal
        "yawn_mar": 0.55,                     # mouth-aspect-ratio threshold for a yawn
        "droop_pitch_deg": -18.0,             # sustained downward pitch = head drooping
        "droop_s": 4.0,
        "jitter_threshold_px": 20.0,          # elevated gaze jitter = instability signal
        "score_threshold": 0.6,               # 0..1 combined fatigue score that trips an alert
        "hold_s": 3.0,                        # score must stay above threshold this long
        "cooldown_s": 300.0,                  # minimum gap between "take a break" alerts
    },
    "objects": {
        "enabled": False,                     # environment object detection (COCO-80 classes).
                                              # OFF by default: ~105 ms/scan is the single most
                                              # expensive stage and it's informational, not a
                                              # control input. Toggle on if you want desk labels.
        "interval_s": 2.5,                    # throttle: full detection is expensive per-frame
        "confidence": 0.5,
    },
    "safety": {
        "mouse_override": True,              # physical mouse movement pauses gaze control
        "override_pause_s": 2.0,
        "start_paused": False,
    },
    "audio": {
        "keepalive": True,                   # silent stream keeps Bluetooth outputs awake
    },
    "server": {"host": "127.0.0.1", "port": 8765, "open_browser": True},
    "metrics": {"log_sessions": True, "snapshot_hz": 1.0},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config:
    def __init__(self, path: str | Path = "config.json") -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._listeners: List[Callable[[str, Any], None]] = []
        self._data = copy.deepcopy(DEFAULTS)
        self.load()

    # -- persistence ------------------------------------------------------
    def load(self) -> None:
        with self._lock:
            if self._path.exists():
                try:
                    user = json.loads(self._path.read_text(encoding="utf-8"))
                    self._data = _deep_merge(DEFAULTS, user)
                except Exception:
                    log.exception("config file unreadable; using defaults")

    def save(self) -> None:
        with self._lock:
            self._path.write_text(
                json.dumps(self._data, indent=2), encoding="utf-8"
            )

    # -- access -----------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        with self._lock:
            node: Any = self._data
            for part in dotted.split("."):
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return copy.deepcopy(node)

    def set(self, dotted: str, value: Any, save: bool = True) -> None:
        with self._lock:
            node = self._data
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
            if save:
                self.save()
        for fn in list(self._listeners):
            try:
                fn(dotted, value)
            except Exception:
                log.exception("config listener failed")

    def data(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._data)

    def replace(self, data: dict, save: bool = True) -> None:
        with self._lock:
            self._data = _deep_merge(DEFAULTS, data)
            if save:
                self.save()
        for fn in list(self._listeners):
            try:
                fn("*", None)
            except Exception:
                log.exception("config listener failed")

    def subscribe(self, fn: Callable[[str, Any], None]) -> None:
        self._listeners.append(fn)
