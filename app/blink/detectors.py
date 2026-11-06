"""Blink detection providers: Standard EAR, Adaptive EAR, and ELA (lid angle).

All three output the same BlinkFrame so the click state machine is agnostic
to how "closed" was decided.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional, Tuple

import numpy as np

from app.core.interfaces import BlinkProvider
from app.core.types import BlinkFrame, FaceResult
from app.vision import landmarks as L


def eye_aspect_ratio(pts: np.ndarray, idx: list[int]) -> float:
    """Soukupova & Cech EAR: (|p2-p6| + |p3-p5|) / (2 |p1-p4|)."""
    p1, p2, p3, p4, p5, p6 = (pts[i][:2] for i in idx)
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    return float((v1 + v2) / (2.0 * h)) if h > 1e-6 else 0.0


def lid_angle_deg(pts: np.ndarray, lid: dict) -> float:
    """Eye Lid Angle: angle at the outer corner between the upper-lid and
    lower-lid chords. Wide open ~ 25-40 deg, closed ~ < 8 deg."""
    corner = pts[lid["corner"]][:2]
    upper = (pts[lid["upper"][0]][:2] + pts[lid["upper"][1]][:2]) / 2.0
    lower = (pts[lid["lower"][0]][:2] + pts[lid["lower"][1]][:2]) / 2.0
    v1, v2 = upper - corner, lower - corner
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cosang))


class _AdaptiveBaseline:
    """Rolling per-eye open baseline: median of recent samples classified open.

    Adapts to eye shape, glasses, lighting and camera angle — the reason
    adaptive thresholds beat one-size-fits-all constants.
    """

    def __init__(self, warmup: int = 20, maxlen: int = 120) -> None:
        self._buf: deque = deque(maxlen=maxlen)
        self._warmup = warmup

    def update(self, value: float, currently_closed: bool) -> float:
        if not currently_closed or len(self._buf) < self._warmup:
            self._buf.append(value)
        buf = sorted(self._buf)
        return buf[len(buf) // 2] if buf else value

    @property
    def warmed(self) -> bool:
        return len(self._buf) >= self._warmup


class StandardEARBlink(BlinkProvider):
    """Fixed-threshold EAR — the literature-standard baseline."""

    name = "ear_standard"

    def __init__(self, cfg_get) -> None:
        self._cfg = cfg_get

    def process(self, face: FaceResult) -> Optional[BlinkFrame]:
        if not face.ok:
            return None
        pts = face.landmarks
        thr = float(self._cfg("blink.ear_threshold", 0.20))
        left = eye_aspect_ratio(pts, L.LEFT_EYE_EAR)
        right = eye_aspect_ratio(pts, L.RIGHT_EYE_EAR)
        return BlinkFrame(t=face.t, left=left, right=right,
                          closed_left=left < thr, closed_right=right < thr,
                          threshold_left=thr, threshold_right=thr, measure="ear")


class AdaptiveEARBlink(BlinkProvider):
    """EAR with a per-user, per-eye rolling baseline threshold."""

    name = "ear_adaptive"

    def __init__(self, cfg_get) -> None:
        self._cfg = cfg_get
        self._bl = _AdaptiveBaseline()
        self._br = _AdaptiveBaseline()
        self._closed_l = False
        self._closed_r = False

    def process(self, face: FaceResult) -> Optional[BlinkFrame]:
        if not face.ok:
            return None
        pts = face.landmarks
        ratio = float(self._cfg("blink.adaptive_ratio", 0.70))
        left = eye_aspect_ratio(pts, L.LEFT_EYE_EAR)
        right = eye_aspect_ratio(pts, L.RIGHT_EYE_EAR)
        base_l = self._bl.update(left, self._closed_l)
        base_r = self._br.update(right, self._closed_r)
        thr_l, thr_r = base_l * ratio, base_r * ratio
        if not (self._bl.warmed and self._br.warmed):
            self._closed_l = self._closed_r = False
        else:
            self._closed_l = left < thr_l
            self._closed_r = right < thr_r
        return BlinkFrame(t=face.t, left=left, right=right,
                          closed_left=self._closed_l, closed_right=self._closed_r,
                          threshold_left=thr_l, threshold_right=thr_r, measure="ear")

    def reset(self) -> None:
        self.__init__(self._cfg)


class ELABlink(BlinkProvider):
    """Eye Lid Angle with an adaptive per-user baseline."""

    name = "ela"

    def __init__(self, cfg_get) -> None:
        self._cfg = cfg_get
        self._bl = _AdaptiveBaseline()
        self._br = _AdaptiveBaseline()
        self._closed_l = False
        self._closed_r = False

    def process(self, face: FaceResult) -> Optional[BlinkFrame]:
        if not face.ok:
            return None
        pts = face.landmarks
        ratio = float(self._cfg("blink.ela_ratio", 0.55))
        left = lid_angle_deg(pts, L.LEFT_LID)
        right = lid_angle_deg(pts, L.RIGHT_LID)
        base_l = self._bl.update(left, self._closed_l)
        base_r = self._br.update(right, self._closed_r)
        thr_l, thr_r = base_l * ratio, base_r * ratio
        if not (self._bl.warmed and self._br.warmed):
            self._closed_l = self._closed_r = False
        else:
            self._closed_l = left < thr_l
            self._closed_r = right < thr_r
        return BlinkFrame(t=face.t, left=left, right=right,
                          closed_left=self._closed_l, closed_right=self._closed_r,
                          threshold_left=thr_l, threshold_right=thr_r, measure="angle")

    def reset(self) -> None:
        self.__init__(self._cfg)
