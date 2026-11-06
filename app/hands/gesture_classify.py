"""Classifies one MediaPipe hand (21 landmarks) into a static gesture.

Geometry-only (no extra model): a finger counts as EXTENDED when its
fingertip sits farther from the wrist than its own PIP joint does, scaled by
hand size so the test is resolution/distance independent. The thumb is
tested against the wrist too, since it doesn't fold the same way as the
other four fingers.

Recognizes: open_palm, thumbs_up, thumbs_down, peace. Anything else (fist,
partial hand, no hand) is "" — the caller treats that as "no gesture".
"""
from __future__ import annotations

import numpy as np

WRIST = 0
THUMB_MCP, THUMB_TIP = 2, 4
INDEX_PIP, INDEX_TIP = 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20
MIDDLE_MCP = 9


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def classify(landmarks: np.ndarray | None) -> str:
    if landmarks is None or len(landmarks) < 21:
        return ""
    p = landmarks
    wrist = p[WRIST]
    scale = _dist(wrist, p[MIDDLE_MCP])
    if scale < 1e-6:
        return ""

    def extended(pip_i: int, tip_i: int, margin: float = 1.05) -> bool:
        return _dist(wrist, p[tip_i]) > _dist(wrist, p[pip_i]) * margin

    index_ext = extended(INDEX_PIP, INDEX_TIP)
    middle_ext = extended(MIDDLE_PIP, MIDDLE_TIP)
    ring_ext = extended(RING_PIP, RING_TIP)
    pinky_ext = extended(PINKY_PIP, PINKY_TIP)
    thumb_ext = _dist(wrist, p[THUMB_TIP]) > _dist(wrist, p[THUMB_MCP]) * 1.3

    four_fingers_up = index_ext and middle_ext and ring_ext and pinky_ext
    four_fingers_curled = not (index_ext or middle_ext or ring_ext or pinky_ext)

    if four_fingers_up and thumb_ext:
        return "open_palm"

    if four_fingers_curled and thumb_ext:
        # thumb sticking out of a fist: direction relative to the wrist
        # decides up vs down (image y grows downward)
        dy = p[THUMB_TIP][1] - wrist[1]
        if dy < -scale * 0.35:
            return "thumbs_up"
        if dy > scale * 0.35:
            return "thumbs_down"
        return ""

    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "peace"

    return ""
