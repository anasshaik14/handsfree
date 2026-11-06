"""Turns stabilized gaze samples into actual OS cursor movement.

Responsibilities:
* map normalized gaze -> physical pixels (with configurable screen margin)
* run the stabilization chain (One Euro -> deadzone -> slew limit)
* freeze the cursor during blinks and head gestures (their landmark motion
  would otherwise fling the cursor)
* physical-mouse override: if the user grabs the real mouse, gaze control
  pauses automatically for a moment (safety + mixed-mode use)
* expose jitter/latency measurements to the metrics engine
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Optional

from app.core.types import GazeSample
from app.gaze.smoothing import Stabilizer
from app.system import win_input


class CursorController:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self.screen_w, self.screen_h = win_input.screen_size()
        self.stab = Stabilizer(cfg.get("gaze.smoothing", {}))
        self._lock = threading.Lock()
        self.enabled = True
        self.paused = bool(cfg.get("safety.start_paused", False))
        self._freeze_until = 0.0
        self._override_until = 0.0
        self._last_cmd: Optional[tuple[float, float]] = None
        self._trail: deque = deque(maxlen=120)   # (t, x, y) for jitter metrics
        self.x = self.screen_w / 2.0
        self.y = self.screen_h / 2.0
        cfg.subscribe(self._on_cfg)

    def _on_cfg(self, path: str, _val) -> None:
        if path == "*" or path.startswith("gaze.smoothing"):
            self.stab.retune(self._cfg.get("gaze.smoothing", {}))

    # -- external control hooks -------------------------------------------
    def freeze(self, duration_s: float) -> None:
        """Hold position (blink in progress / head gesture active)."""
        self._freeze_until = max(self._freeze_until, time.perf_counter() + duration_s)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False
        self.stab.reset()
        self._last_cmd = None

    # -- main entry ---------------------------------------------------------
    def update(self, sample: GazeSample) -> Optional[tuple[float, float]]:
        """Feed one gaze sample; moves the OS cursor. Returns commanded (x, y)."""
        t = time.perf_counter()
        with self._lock:
            if not self.enabled or self.paused or not sample.valid:
                return None
            # Physical mouse override: user moved the real mouse since our
            # last command -> back off.
            if self._cfg.get("safety.mouse_override", True) and self._last_cmd:
                cx, cy = win_input.cursor_pos()
                if math.hypot(cx - self._last_cmd[0], cy - self._last_cmd[1]) > 30:
                    self._override_until = t + float(
                        self._cfg.get("safety.override_pause_s", 2.0))
                    self._last_cmd = None
                    self.stab.reset()
            if t < self._override_until or t < self._freeze_until:
                return None

            # Expansion around center: calibration dots sit ~8% inside the
            # screen edges, so raw predictions top out before the corners.
            # Stretching by ~1.2 makes the true corners reachable.
            expand = float(self._cfg.get("gaze.expand", 1.2))
            margin = float(self._cfg.get("gaze.margin", 0.0))
            nx = 0.5 + (sample.nx - 0.5) * expand
            ny = 0.5 + (sample.ny - 0.5) * expand
            nx = min(max(nx, 0.0), 1.0)
            ny = min(max(ny, 0.0), 1.0)
            px = (margin + nx * (1 - 2 * margin)) * (self.screen_w - 1)
            py = (margin + ny * (1 - 2 * margin)) * (self.screen_h - 1)
            fx, fy = self.stab.process(px, py, sample.t)
            self.x, self.y = fx, fy
            win_input.move_to(fx, fy)
            self._last_cmd = (round(fx), round(fy))
            self._trail.append((t, fx, fy))
            return fx, fy

    # -- metrics -------------------------------------------------------------
    def jitter_px(self, window_s: float = 1.0, speed_thresh: float = 60.0) -> float:
        """RMS deviation of the cursor around its mean while fixating.

        Only counts windows where average speed is below `speed_thresh` px/s,
        so intentional movement doesn't register as jitter.
        """
        with self._lock:
            t_now = time.perf_counter()
            pts = [(x, y) for (t, x, y) in self._trail if t_now - t <= window_s]
        if len(pts) < 8:
            return 0.0
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        path = sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
                   for i in range(1, n))
        if path / window_s > speed_thresh:
            return -1.0  # moving: not a fixation window
        return math.sqrt(sum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in pts) / n)

    def state(self) -> dict:
        return {
            "x": round(self.x, 1), "y": round(self.y, 1),
            "paused": self.paused,
            "override": time.perf_counter() < self._override_until,
            "frozen": time.perf_counter() < self._freeze_until,
            "jitter_px": round(self.jitter_px(), 2),
        }
