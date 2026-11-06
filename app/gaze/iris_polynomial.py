"""Gaze provider 2: MediaPipe Iris + polynomial calibration.

The most accurate mode: a 9-point calibration fits a per-user polynomial from
(iris features + head pose) to screen coordinates. Until calibrated it
gracefully falls back to the geometric mapping so the cursor always works.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from app.core.interfaces import GazeProvider
from app.core.types import FaceResult, GazeSample, HeadPose
from app.gaze.calibration import PolynomialModel
from app.gaze.features import FeatureSmoother, iris_features
from app.gaze.iris_geometric import GeometricMapper

CALIB_PATH = Path("calibration/gaze_polynomial.json")


class IrisPolynomialGaze(GazeProvider):
    name = "iris_polynomial"
    needs_calibration = True

    def __init__(self, cfg_get) -> None:
        self._cfg = cfg_get
        self._fallback = GeometricMapper(cfg_get)
        self.smoother = FeatureSmoother()
        self._last_saved = 0.0
        self.model: Optional[PolynomialModel] = PolynomialModel.load(CALIB_PATH)
        # A calibration fit with a different head-pose provider feeds the
        # model mis-scaled yaw/pitch — silently wrong cursor. Require refit.
        if self.model is not None and self.model.pose_provider:
            current = cfg_get("headpose.provider", "")
            if current and self.model.pose_provider != current:
                self.model = None

    @property
    def calibrated(self) -> bool:
        return self.model is not None and self.model.ready

    def set_model(self, model: PolynomialModel) -> None:
        self.model = model
        model.save(CALIB_PATH)
        self._last_saved = time.time()

    def online_update(self, hx: float, hy: float, yaw: float, pitch: float,
                      tx: float, ty: float) -> None:
        """Dynamic calibration (Upgrade 4): refine the model toward one fresh
        (features -> screen target) pair, e.g. captured when the user clicks.
        Persists at most every 10 s so drift survives a restart without
        hammering the disk on every click."""
        if not self.calibrated:
            return
        self.model.online_update(hx, hy, yaw, pitch, tx, ty,
                                 forget=float(self._cfg(
                                     "gaze.calibration.dynamic_forget", 0.9995)))
        if time.time() - self._last_saved > 10.0:
            self.model.save(CALIB_PATH)
            self._last_saved = time.time()

    def process(self, face: FaceResult, pose: Optional[HeadPose],
                frame=None) -> Optional[GazeSample]:
        if not face.ok:
            return None
        feats = iris_features(face.landmarks, face.width, face.height)
        if feats is None:
            return None
        hx, hy, quality = feats
        hx, hy = self.smoother.apply(hx, hy, face.t)
        yaw = pose.yaw if pose and pose.valid else 0.0
        pitch = pose.pitch if pose and pose.valid else 0.0
        if self.calibrated:
            nx, ny = self.model.predict(hx, hy, yaw, pitch)
        else:
            # Note: quality must stay un-degraded here — the pipeline's
            # hysteretic gate reopens at 0.45, and scaling quality down
            # (old *0.6 "uncalibrated flag") latched the gate shut for
            # users whose raw quality sits below 0.75. Uncalibrated state
            # is reported via `calibrated` in the snapshot instead.
            nx, ny = self._fallback.map(hx, hy, pose)
        return GazeSample(t=face.t, nx=nx, ny=ny, quality=quality,
                          raw_nx=hx, raw_ny=hy)

    def set_center(self) -> bool:
        return self._fallback.set_center()

    def reset(self) -> None:
        self._fallback.reset()
        self.smoother.reset()
