"""Head-pose provider "custom": our own CNN, trained from scratch on 300W-LP
(evaluated on AFLW2000), regressing yaw/pitch/roll directly from an RGB face
crop — no solvePnP, no landmark heuristics.

The model predicts ABSOLUTE head pose; the app works in neutral-relative
angles, so a per-user neutral offset (auto-captured, re-settable) is
subtracted, reusing the same _NeutralOffset used by the classical providers.
Falls back to SolvePnPHeadPose when the model can't run.

NOTE on sign convention: the trainer (scripts/train_headpose.py) emits
yaw/pitch/roll in the app's convention (yaw>0 = turned user's LEFT, pitch>0 =
up). If your trained model steers gestures the wrong way, flip AXIS_SIGN
below — it's the one thing that depends on the dataset's raw label signs.

Train it with:  python scripts/train_headpose.py
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from app.core.interfaces import HeadPoseProvider
from app.core.types import FaceResult, HeadPose
from app.headpose.providers import SolvePnPHeadPose, _NeutralOffset
from app.ml import preprocess, runtime

log = logging.getLogger(__name__)

WEIGHTS_PATH = runtime.MODELS_DIR / "headpose_cnn.pt"
AXIS_SIGN = (1.0, 1.0, 1.0)   # (yaw, pitch, roll) — flip an entry to -1 if inverted


class CustomHeadPoseCNN(HeadPoseProvider):
    name = "custom"

    def __init__(self, cfg_get=None) -> None:
        self._fallback = SolvePnPHeadPose()
        self._neutral = _NeutralOffset()
        self._model = runtime.load_scripted(WEIGHTS_PATH)
        self._last_abs = (0.0, 0.0, 0.0)

    @property
    def available(self) -> bool:
        return self._model is not None

    def _predict_abs(self, face: FaceResult) -> Optional[tuple[float, float, float]]:
        if face.frame is None or not face.ok:
            return None
        crop = preprocess.crop_face_rgb(face.frame, face.landmarks)
        if crop is None:
            return None
        x = preprocess.norm_rgb(crop).transpose(2, 0, 1)   # (3, H, W)
        out = runtime.infer(self._model, x)
        return (float(out[0]) * AXIS_SIGN[0], float(out[1]) * AXIS_SIGN[1],
                float(out[2]) * AXIS_SIGN[2])

    def process(self, face: FaceResult) -> Optional[HeadPose]:
        if self._model is None:
            return self._fallback.process(face)
        angles = self._predict_abs(face)
        if angles is None:
            return self._fallback.process(face)
        self._last_abs = angles
        yaw, pitch, roll = self._neutral.apply(*angles)
        return HeadPose(t=face.t, yaw=yaw, pitch=pitch, roll=roll,
                        valid=self._neutral.offset is not None)

    def set_neutral(self, face: FaceResult) -> bool:
        angles = self._predict_abs(face)
        if angles is None:
            return self._fallback.set_neutral(face)
        self._neutral.set_from(*angles)
        return True

    def reset(self) -> None:
        self._neutral.reset()
        self._fallback.reset()
