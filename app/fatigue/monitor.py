"""Fatigue detection: combines signals already computed elsewhere in the
pipeline (blink events, head pose, gaze jitter) plus one new one (yawning,
via mouth-aspect-ratio on the shared face landmarks) into a single score.

Deliberately a weighted heuristic, not a trained classifier — there's no
labeled drowsiness dataset in this project to train one against. Every
sub-signal is independently plausible (these are the same signals real
drowsiness-detection literature uses); the weights are a reasonable starting
point meant to be tuned in config, not a calibrated model.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Optional

import numpy as np

from app.core.types import BlinkEvent, FaceResult, HeadPose
from app.vision import landmarks as L


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    upper = pts[L.MOUTH_UPPER_INNER][:2]
    lower = pts[L.MOUTH_LOWER_INNER][:2]
    right = pts[L.MOUTH_CORNER_RIGHT][:2]
    left = pts[L.MOUTH_CORNER_LEFT][:2]
    width = float(np.linalg.norm(left - right))
    if width < 1e-6:
        return 0.0
    return float(np.linalg.norm(upper - lower)) / width


class FatigueMonitor:
    BLINK_WINDOW_S = 60.0
    YAWN_WINDOW_S = 300.0

    def __init__(self, cfg_get,
                 on_alert: Optional[Callable[[float], None]] = None) -> None:
        self._cfg = cfg_get
        self._on_alert = on_alert
        self._blink_times: deque = deque()
        self._blink_durations: deque = deque(maxlen=30)
        self._yawn_times: deque = deque()
        self._yawn_active_since: Optional[float] = None
        self._droop_since: Optional[float] = None
        self._score_high_since: Optional[float] = None
        self._last_alert = float("-inf")
        self.last_score = 0.0

    # ---- fed from the blink FSM (bus topic "blink.event") -----------------
    def on_blink_event(self, ev: BlinkEvent) -> None:
        if ev.kind in ("natural", "cancelled"):
            self._blink_times.append(ev.t)
            self._blink_durations.append(ev.duration)

    # ---- fed once per frame from the pipeline loop ------------------------
    def update(self, pose: Optional[HeadPose], jitter_px: float,
               face: Optional[FaceResult], t: float) -> float:
        while self._blink_times and t - self._blink_times[0] > self.BLINK_WINDOW_S:
            self._blink_times.popleft()
        while self._yawn_times and t - self._yawn_times[0] > self.YAWN_WINDOW_S:
            self._yawn_times.popleft()

        yawn_mar = float(self._cfg("fatigue.yawn_mar", 0.55))
        if face is not None and face.ok:
            mar = mouth_aspect_ratio(face.landmarks)
            if mar > yawn_mar:
                if self._yawn_active_since is None:
                    self._yawn_active_since = t
                elif t - self._yawn_active_since > 0.6 and \
                        (not self._yawn_times or t - self._yawn_times[-1] > 2.0):
                    self._yawn_times.append(t)
            else:
                self._yawn_active_since = None

        droop_deg = float(self._cfg("fatigue.droop_pitch_deg", -18.0))
        droop_s = float(self._cfg("fatigue.droop_s", 4.0))
        drooping = False
        if pose is not None and pose.valid and pose.pitch <= droop_deg:
            if self._droop_since is None:
                self._droop_since = t
            drooping = (t - self._droop_since) >= droop_s
        else:
            self._droop_since = None

        blink_rate = len(self._blink_times) / (self.BLINK_WINDOW_S / 60.0)
        avg_dur = sum(self._blink_durations) / len(self._blink_durations) \
            if self._blink_durations else 0.0

        score = 0.0
        lo, hi = float(self._cfg("fatigue.blink_rate_low", 8)), \
            float(self._cfg("fatigue.blink_rate_high", 30))
        if self._blink_times and (blink_rate < lo or blink_rate > hi):
            score += 0.2
        if avg_dur > 0.35:
            score += 0.15
        if self._yawn_times:
            score += min(0.35, 0.15 * len(self._yawn_times))
        if drooping:
            score += 0.25
        jitter_threshold = float(self._cfg("fatigue.jitter_threshold_px", 20.0))
        if jitter_px > jitter_threshold:
            score += 0.1
        score = min(1.0, score)
        self.last_score = score

        threshold = float(self._cfg("fatigue.score_threshold", 0.6))
        hold_s = float(self._cfg("fatigue.hold_s", 3.0))
        cooldown_s = float(self._cfg("fatigue.cooldown_s", 300.0))
        if score >= threshold:
            if self._score_high_since is None:
                self._score_high_since = t
            elif (t - self._score_high_since) >= hold_s and \
                    (t - self._last_alert) >= cooldown_s:
                self._last_alert = t
                if self._on_alert:
                    self._on_alert(score)
        else:
            self._score_high_since = None

        return score
