"""Gaze provider 1: MediaPipe Iris + geometric mapping (no calibration).

Normalized iris-in-eye-box position is mapped linearly around a per-user
center point with configurable gains, optionally assisted by head pose
(where you point your head steers the cursor; the iris refines it).
Works instantly, no calibration required.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from app.core.interfaces import GazeProvider
from app.core.types import FaceResult, GazeSample, HeadPose
from app.gaze.features import FeatureSmoother, iris_features


class GeometricMapper:
    """Shared center+gain mapping used by the geometric and classical providers."""

    # Per-sample EMA rate pulling the center toward the running gaze mean
    # (tau ~17 s @ 30 fps). The startup auto-center is captured from the
    # first second of samples; if the user wasn't looking at screen center
    # right then, the reference is skewed and the cursor pins to an edge.
    # Gaze averages near screen center over tens of seconds, so a slow
    # drift toward the mean self-heals a bad capture without noticeably
    # fighting intentional fixations.
    RECENTER_RATE = 0.002

    def __init__(self, cfg_get) -> None:
        self._cfg = cfg_get                       # callable: dotted path -> value
        self._center_buf: deque = deque(maxlen=45)
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None

    def map(self, hx: float, hy: float, pose: Optional[HeadPose]) -> tuple[float, float]:
        self._center_buf.append((hx, hy))
        if self._cx is None and len(self._center_buf) >= 30:
            self.set_center()
        if self._cx is None:
            # No center reference yet (first ~1 s of face samples): mapping
            # against a guessed center flings the cursor — hold screen center.
            # A guess can't work here anyway: providers feed differently
            # scaled features (iris ~[-0.15..1], classical pupil ~[0..1]).
            return 0.5, 0.5
        self._cx += (hx - self._cx) * self.RECENTER_RATE
        self._cy += (hy - self._cy) * self.RECENTER_RATE
        cx, cy = self._cx, self._cy
        gx = float(self._cfg("gaze.gain_x", 2.8))
        gy = float(self._cfg("gaze.gain_y", 3.4))
        # Looking right (user's right) moves the iris toward image-left -> invert x.
        nx = 0.5 - gx * (hx - cx)
        ny = 0.5 + gy * (hy - cy)
        if pose is not None and pose.valid:
            k = float(self._cfg("gaze.pose_assist", 0.35))
            # yaw > 0 = turned left = look toward screen left; pitch > 0 = up.
            nx += -pose.yaw / 45.0 * k
            ny += -pose.pitch / 35.0 * k
        return nx, ny

    def set_center(self) -> bool:
        """Capture 'user is looking at screen center' from recent samples."""
        if not self._center_buf:
            return False
        xs = sorted(p[0] for p in self._center_buf)
        ys = sorted(p[1] for p in self._center_buf)
        self._cx = xs[len(xs) // 2]
        self._cy = ys[len(ys) // 2]
        return True

    def reset(self) -> None:
        self._center_buf.clear()
        self._cx = self._cy = None


class IrisGeometricGaze(GazeProvider):
    name = "iris_geometric"
    needs_calibration = False

    def __init__(self, cfg_get) -> None:
        self.mapper = GeometricMapper(cfg_get)
        self.smoother = FeatureSmoother()

    def process(self, face: FaceResult, pose: Optional[HeadPose],
                frame=None) -> Optional[GazeSample]:
        if not face.ok:
            return None
        feats = iris_features(face.landmarks, face.width, face.height)
        if feats is None:
            return None
        hx, hy, quality = feats
        hx, hy = self.smoother.apply(hx, hy, face.t)
        nx, ny = self.mapper.map(hx, hy, pose)
        return GazeSample(t=face.t, nx=nx, ny=ny, quality=quality,
                          raw_nx=hx, raw_ny=hy)

    def set_center(self) -> bool:
        return self.mapper.set_center()

    def reset(self) -> None:
        self.mapper.reset()
        self.smoother.reset()
