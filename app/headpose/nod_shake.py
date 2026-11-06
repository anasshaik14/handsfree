"""Nod (yes) / shake (no) detection from head pose.

Different shape of motion than app.headpose.gesture_fsm: tab/app switching
fires on a single deliberate HOLD past a threshold. Nod/shake are repeated
back-and-forth motions, so this tracks oscillation instead: the pitch (nod)
or yaw (shake) signal, detrended against its own short-term mean, must cross
zero at least twice with sufficient amplitude within a rolling time window.

Runs independently of GestureFSM off the same HeadPose stream — the two
rarely conflict in practice because oscillation doesn't dwell long enough on
one side to also satisfy GestureFSM's dwell_ms hold-and-return pattern.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Optional

from app.core.types import HeadPose


class _AxisOscillation:
    def __init__(self) -> None:
        self._buf: deque = deque()   # (t, value)
        self._last_sign = 0

    def update(self, value: float, t: float, window_s: float,
               amplitude_deg: float, min_crossings: int = 3) -> bool:
        self._buf.append((t, value))
        while self._buf and t - self._buf[0][0] > window_s:
            self._buf.popleft()
        if len(self._buf) < 4:
            return False
        mean = sum(v for _, v in self._buf) / len(self._buf)
        # Only count a sign flip as a real crossing once the signal has swung
        # a good fraction of the required amplitude past the mean — tiny
        # dither around neutral must not rack up crossings and trip a gesture.
        gate = amplitude_deg * 0.4
        crossings = 0
        peak = 0.0
        last_sign = 0
        for _, v in self._buf:
            d = v - mean
            peak = max(peak, abs(d))
            sign = 1 if d > gate else (-1 if d < -gate else 0)
            if sign != 0 and last_sign != 0 and sign != last_sign:
                crossings += 1
            if sign != 0:
                last_sign = sign
        return crossings >= min_crossings and peak >= amplitude_deg

    def clear(self) -> None:
        self._buf.clear()


class NodShakeDetector:
    def __init__(self, cfg_get,
                 on_gesture: Optional[Callable[[str], None]] = None) -> None:
        self._cfg = cfg_get
        self._on_gesture = on_gesture
        self._pitch = _AxisOscillation()
        self._yaw = _AxisOscillation()
        self._cooldown_until = 0.0

    def update(self, pose: Optional[HeadPose]) -> Optional[str]:
        if pose is None or not pose.valid:
            return None
        t = pose.t
        if t < self._cooldown_until:
            return None
        window_s = float(self._cfg("headpose.nod_shake.window_s", 1.4))
        amp = float(self._cfg("headpose.nod_shake.amplitude_deg", 14.0))
        min_cross = int(self._cfg("headpose.nod_shake.min_crossings", 3))
        nodding = self._pitch.update(pose.pitch, t, window_s, amp, min_cross)
        shaking = self._yaw.update(pose.yaw, t, window_s, amp, min_cross)

        result = None
        if nodding and not shaking:
            result = "nod"
        elif shaking and not nodding:
            result = "shake"
        elif nodding and shaking:
            # ambiguous frame — let it resolve on the next update
            return None

        if result:
            self._pitch.clear()
            self._yaw.clear()
            self._cooldown_until = t + float(
                self._cfg("headpose.nod_shake.cooldown_s", 1.2))
            if self._on_gesture:
                self._on_gesture(result)
        return result

    def reset(self) -> None:
        self._pitch.clear()
        self._yaw.clear()
        self._cooldown_until = 0.0
