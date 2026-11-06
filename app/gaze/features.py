"""Shared eye-feature extraction from FaceMesh landmarks.

All iris-based gaze providers consume the same normalized features:
  hx in ~[0..1] — iris center between the eye corners (0 = outer/inner edge).
  hy in ~[-0.15..0.15] — iris vertical offset from the eye-corner line,
  normalized by eye width. Corners are the reference because they are the
  only landmarks that do NOT move with gaze: the eyelids follow the iris
  (the upper lid retracts as you look up), so an eyelid-relative measure
  compresses and can even invert vertical gaze.
  Both features are averaged over both eyes, weighted by eye openness
  (a closing eye gives garbage).

Image convention (non-mirrored frame): the subject's RIGHT eye appears on the
image LEFT; looking toward the user's right moves the iris toward image left.
The mapping layers handle that inversion.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from app.gaze.smoothing import OneEuro2D
from app.vision import landmarks as L


class FeatureSmoother:
    """One Euro on the raw (hx, hy) iris features.

    Smoothing here — BEFORE the mapping layer multiplies by gain ~3x or a
    polynomial — is far more effective per unit of lag than smoothing the
    screen-space output alone. Landmark noise of ±0.01 in hy becomes ±30+px
    of cursor jitter after amplification; killed at the source it never
    reaches the mapper.
    """

    def __init__(self, mincutoff: float = 0.35, beta: float = 2.0,
                 dcutoff: float = 1.0) -> None:
        # Feature units are ~[0..1] across the whole eye, so speeds are tiny:
        # noise ~0.05/s, a real saccade ~1/s. mincutoff 0.35 pins the cursor
        # during fixation; beta 2.0 opens the cutoff wide the instant the
        # iris genuinely moves, so tracking stays snappy.
        self._f = OneEuro2D(mincutoff, beta, dcutoff)

    def apply(self, hx: float, hy: float, t: float) -> Tuple[float, float]:
        return self._f.filter(hx, hy, t)

    def reset(self) -> None:
        self._f.reset()


def eye_openness(pts: np.ndarray, box: dict) -> float:
    """Vertical/horizontal eye box ratio, ~0.3 open, ~0.05 closed."""
    w = abs(pts[box["outer"]][0] - pts[box["inner"]][0])
    h = abs(pts[box["bottom"]][1] - pts[box["top"]][1])
    return float(h / w) if w > 1e-6 else 0.0


def _iris_in_box(pts: np.ndarray, box: dict, iris_idx: int) -> Optional[Tuple[float, float]]:
    outer, inner = pts[box["outer"]], pts[box["inner"]]
    iris = pts[iris_idx]
    ex0, ex1 = min(outer[0], inner[0]), max(outer[0], inner[0])
    w = ex1 - ex0
    if w < 4:
        return None
    hx = (iris[0] - ex0) / w
    # Vertical relative to the gaze-stable corner line, NOT the eyelids
    # (image y grows downward: looking up -> hy decreases).
    hy = (iris[1] - (outer[1] + inner[1]) / 2.0) / w
    return float(hx), float(hy)


def _eye_in_frame(pts: np.ndarray, box: dict, iris_idx: int,
                  width: int, height: int, margin: float = 8.0) -> bool:
    """True if every landmark this eye's features depend on lies safely
    inside the frame. MediaPipe EXTRAPOLATES landmarks for a partially
    visible face (coordinates beyond the image bounds), which yields
    plausible-looking but garbage iris positions that fling the cursor
    into a screen corner — reject such eyes outright."""
    for i in (box["outer"], box["inner"], box["top"], box["bottom"], iris_idx):
        x, y = pts[i][0], pts[i][1]
        if not (margin <= x <= width - margin and margin <= y <= height - margin):
            return False
    return True


def iris_features(pts: np.ndarray, width: int = 0,
                  height: int = 0) -> Optional[Tuple[float, float, float]]:
    """Returns (hx, hy, quality) averaged over both eyes, or None.

    When frame dimensions are given, eyes whose landmarks touch the frame
    border are discarded (see _eye_in_frame)."""
    r = _iris_in_box(pts, L.RIGHT_EYE_BOX, L.RIGHT_IRIS_CENTER)
    l = _iris_in_box(pts, L.LEFT_EYE_BOX, L.LEFT_IRIS_CENTER)
    if width and height:
        if r is not None and not _eye_in_frame(pts, L.RIGHT_EYE_BOX,
                                               L.RIGHT_IRIS_CENTER, width, height):
            r = None
        if l is not None and not _eye_in_frame(pts, L.LEFT_EYE_BOX,
                                               L.LEFT_IRIS_CENTER, width, height):
            l = None
    if r is None and l is None:
        return None
    open_r = eye_openness(pts, L.RIGHT_EYE_BOX)
    open_l = eye_openness(pts, L.LEFT_EYE_BOX)
    # Weight each eye by openness; a nearly closed eye contributes ~nothing.
    wr = max(open_r - 0.08, 0.0)
    wl = max(open_l - 0.08, 0.0)
    if r is None:
        wr = 0.0
        r = (0.0, 0.0)
    if l is None:
        wl = 0.0
        l = (0.0, 0.0)
    total = wr + wl
    if total < 1e-6:
        return None
    hx = (r[0] * wr + l[0] * wl) / total
    hy = (r[1] * wr + l[1] * wl) / total
    quality = min(1.0, total / 0.35)
    return hx, hy, quality
