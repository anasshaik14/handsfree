"""Blink provider "custom": our own CNN, trained from scratch on the MRL Eye
Dataset (open/closed eye images), classifying each eye directly from its
image crop — no EAR threshold, no dependence on the classical detectors.

Falls back to AdaptiveEARBlink only when the model can't run (torch/weights
absent, or no usable eye crop this frame). That's a pure availability
fallback, not a blend: when the CNN runs, the decision is entirely the CNN's.

Train it with:  python scripts/train_blink.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from app.blink.detectors import AdaptiveEARBlink
from app.core.interfaces import BlinkProvider
from app.core.types import BlinkFrame, FaceResult
from app.ml import preprocess, runtime

log = logging.getLogger(__name__)

WEIGHTS_PATH = runtime.MODELS_DIR / "blink_cnn.pt"
# Class index convention shared with scripts/train_blink.py.
OPEN, CLOSED = 0, 1


def _softmax2(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


class CustomBlinkCNN(BlinkProvider):
    name = "custom"

    def __init__(self, cfg_get) -> None:
        self._cfg = cfg_get
        self._fallback = AdaptiveEARBlink(cfg_get)
        self._model = runtime.load_scripted(WEIGHTS_PATH)

    @property
    def available(self) -> bool:
        return self._model is not None

    def _predict_eye(self, frame_bgr, pts, eye: str) -> Optional[float]:
        crop = preprocess.crop_eye_gray(frame_bgr, pts, eye)
        if crop is None:
            return None
        x = preprocess.norm_gray(crop)[None, :, :]   # (1, H, W)
        logits = runtime.infer(self._model, x)
        return float(_softmax2(logits)[CLOSED])

    def process(self, face: FaceResult) -> Optional[BlinkFrame]:
        if self._model is None or face.frame is None or not face.ok:
            return self._fallback.process(face)
        p_closed_l = self._predict_eye(face.frame, face.landmarks, "left")
        p_closed_r = self._predict_eye(face.frame, face.landmarks, "right")
        if p_closed_l is None or p_closed_r is None:
            return self._fallback.process(face)   # bad crop -> classical for this frame
        return BlinkFrame(
            t=face.t, left=1.0 - p_closed_l, right=1.0 - p_closed_r,
            closed_left=p_closed_l > 0.5, closed_right=p_closed_r > 0.5,
            threshold_left=0.5, threshold_right=0.5, measure="custom_cnn")

    def reset(self) -> None:
        self._fallback.reset()
