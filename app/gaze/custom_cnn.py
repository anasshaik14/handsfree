"""Gaze provider "custom": our own appearance-based gaze CNN, trained from
scratch on MPIIGaze, predicting gaze direction (pitch, yaw in radians)
directly from a grayscale eye patch — no iris-geometry mapping.

Gaze *direction* still has to become a *screen point*, which depends on the
user's screen geometry and distance, so a light per-user affine map turns
angles into normalized screen coords. `set_center` captures "looking at
screen center", and gaze.custom.gain_{x,y} scale angle->screen (the
calibration knob the physical setup needs). Falls back to IrisGeometricGaze
when the model can't run.

Train it with:  python scripts/train_gaze.py
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from app.core.interfaces import GazeProvider
from app.core.types import FaceResult, GazeSample, HeadPose
from app.gaze.features import iris_features
from app.gaze.iris_geometric import IrisGeometricGaze
from app.gaze.smoothing import OneEuro2D
from app.ml import preprocess, runtime

log = logging.getLogger(__name__)

WEIGHTS_PATH = runtime.MODELS_DIR / "gaze_cnn.pt"


class CustomGazeCNN(GazeProvider):
    name = "custom"
    needs_calibration = False

    def __init__(self, cfg_get) -> None:
        self._cfg = cfg_get
        self._fallback = IrisGeometricGaze(cfg_get)
        self._model = runtime.load_scripted(WEIGHTS_PATH)
        self._center = None                 # (pitch, yaw) captured as screen-center
        self._recent = []                   # for auto-center on first samples
        self._smooth = OneEuro2D(0.35, 2.0, 1.0)   # smooth the gaze ANGLES

    @property
    def available(self) -> bool:
        return self._model is not None

    def _predict_angles(self, face: FaceResult) -> Optional[tuple[float, float]]:
        if face.frame is None or not face.ok:
            return None
        outs = []
        for eye in ("left", "right"):
            crop = preprocess.crop_eye_gray(face.frame, face.landmarks, eye,
                                            size=preprocess.GAZE_EYE_SIZE)
            if crop is None:
                continue
            x = preprocess.norm_gray(crop)[None, :, :]   # (1, H, W)
            out = runtime.infer(self._model, x)
            outs.append((float(out[0]), float(out[1])))  # (pitch, yaw) rad
        if not outs:
            return None
        return tuple(np.mean(outs, axis=0))

    def process(self, face: FaceResult, pose: Optional[HeadPose],
                frame=None) -> Optional[GazeSample]:
        if self._model is None:
            return self._fallback.process(face, pose, frame)
        angles = self._predict_angles(face)
        if angles is None:
            return self._fallback.process(face, pose, frame)
        pitch, yaw = self._smooth.filter(angles[0], angles[1], face.t)

        if self._center is None:
            self._recent.append((pitch, yaw))
            if len(self._recent) >= 30:
                self._center = tuple(np.median(self._recent, axis=0))
            return GazeSample(t=face.t, nx=0.5, ny=0.5, quality=0.0,
                              raw_nx=yaw, raw_ny=pitch)   # holding center while settling

        cp, cy = self._center
        gx = float(self._cfg("gaze.custom.gain_x", 1.6))
        gy = float(self._cfg("gaze.custom.gain_y", 1.9))
        nx = 0.5 + gx * (yaw - cy)      # look right (yaw+) -> cursor right
        ny = 0.5 - gy * (pitch - cp)    # look up (pitch+) -> cursor up

        feats = iris_features(face.landmarks, face.width, face.height)
        quality = feats[2] if feats is not None else 0.5
        return GazeSample(t=face.t, nx=nx, ny=ny, quality=quality,
                          raw_nx=yaw, raw_ny=pitch)

    def set_center(self) -> bool:
        if not self._recent:
            return self._fallback.set_center()
        self._center = tuple(np.median(self._recent[-30:], axis=0))
        return True

    def reset(self) -> None:
        self._center = None
        self._recent = []
        self._smooth.reset()
        self._fallback.reset()
