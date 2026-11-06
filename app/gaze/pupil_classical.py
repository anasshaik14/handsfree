"""Gaze provider 3: GazeTracking-style classical computer vision.

Reimplements the pupil-detection approach of the popular GazeTracking library
(antoinelame/GazeTracking): locate the eye region, then find the pupil as the
darkest blob via blurring + intensity thresholding + contour centroid — pure
classical CV, no neural iris landmarks.

The original library uses Dlib's 68-point detector only to locate the eye
region; building Dlib on Windows is notoriously fragile, so the eye region
here comes from the already-running FaceMesh, while pupil localization is
faithful to the classical method being compared.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from app.core.interfaces import GazeProvider
from app.core.types import FaceResult, GazeSample, HeadPose
from app.gaze.features import eye_openness
from app.gaze.iris_geometric import GeometricMapper
from app.vision import landmarks as L


def _pupil_in_roi(gray_roi: np.ndarray) -> Optional[Tuple[float, float]]:
    """Darkest-blob pupil detection (GazeTracking's isolate/threshold approach)."""
    if gray_roi.size == 0 or gray_roi.shape[0] < 6 or gray_roi.shape[1] < 6:
        return None
    blur = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    blur = cv2.erode(blur, np.ones((3, 3), np.uint8), iterations=2)
    # Threshold at a low intensity percentile -> mask of the darkest region.
    thresh_val = float(np.percentile(blur, 8))
    _, mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    m = cv2.moments(largest)
    if m["m00"] < 1e-3:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


class PupilClassicalGaze(GazeProvider):
    name = "pupil_classical"
    needs_calibration = False

    def __init__(self, cfg_get) -> None:
        self.mapper = GeometricMapper(cfg_get)

    def _eye_feature(self, pts: np.ndarray, box: dict,
                     gray: np.ndarray) -> Optional[Tuple[float, float, float]]:
        outer, inner = pts[box["outer"]], pts[box["inner"]]
        top, bottom = pts[box["top"]], pts[box["bottom"]]
        x0 = int(max(min(outer[0], inner[0]) - 4, 0))
        x1 = int(min(max(outer[0], inner[0]) + 4, gray.shape[1]))
        y0 = int(max(min(top[1], bottom[1]) - 6, 0))
        y1 = int(min(max(top[1], bottom[1]) + 6, gray.shape[0]))
        if x1 - x0 < 8 or y1 - y0 < 6:
            return None
        pupil = _pupil_in_roi(gray[y0:y1, x0:x1])
        if pupil is None:
            return None
        w, h = x1 - x0, y1 - y0
        return pupil[0] / w, pupil[1] / h, eye_openness(pts, box)

    def process(self, face: FaceResult, pose: Optional[HeadPose],
                frame=None) -> Optional[GazeSample]:
        if not face.ok or frame is None:
            return None
        pts = face.landmarks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        r = self._eye_feature(pts, L.RIGHT_EYE_BOX, gray)
        l = self._eye_feature(pts, L.LEFT_EYE_BOX, gray)
        feats = [f for f in (r, l) if f is not None and f[2] > 0.12]
        if not feats:
            return None
        hx = sum(f[0] for f in feats) / len(feats)
        hy = sum(f[1] for f in feats) / len(feats)
        quality = min(1.0, sum(f[2] for f in feats) / len(feats) / 0.3) * 0.9
        nx, ny = self.mapper.map(hx, hy, pose)
        return GazeSample(t=face.t, nx=nx, ny=ny, quality=quality,
                          raw_nx=hx, raw_ny=hy)

    def set_center(self) -> bool:
        return self.mapper.set_center()

    def reset(self) -> None:
        self.mapper.reset()
