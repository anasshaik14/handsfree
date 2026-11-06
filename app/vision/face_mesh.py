"""Shared MediaPipe FaceLandmarker stage (Tasks API, MediaPipe >= 0.10.2x).

One inference per frame feeds ALL modalities (gaze, blink, head pose), which
is what makes running every input simultaneously cheap: the expensive neural
step happens exactly once. The model outputs 478 landmarks including the
iris ring (indices 468..477).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.types import FaceResult, now

log = logging.getLogger(__name__)

MODEL_PATH = Path("models/face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")


def _ensure_model(path: Path = MODEL_PATH) -> Path:
    if path.exists():
        return path
    log.info("downloading face landmarker model to %s …", path)
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, path)  # noqa: S310 - fixed https URL
    return path


class FaceMeshService:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lm = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, frame_bgr: np.ndarray, t: Optional[float] = None) -> FaceResult:
        import cv2
        t = t if t is not None else now()
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires strictly increasing timestamps
        ts_ms = int(t * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        try:
            res = self._lm.detect_for_video(image, ts_ms)
        except Exception:
            log.exception("face landmarker inference failed")
            return FaceResult(t=t, landmarks=None, width=w, height=h)
        if not res.face_landmarks:
            return FaceResult(t=t, landmarks=None, width=w, height=h)
        lm = res.face_landmarks[0]
        # Pixel coordinates; z is scaled by width so all axes share one unit.
        pts = np.array([(p.x * w, p.y * h, p.z * w) for p in lm], dtype=np.float64)
        return FaceResult(t=t, landmarks=pts, width=w, height=h)

    def close(self) -> None:
        try:
            self._lm.close()
        except Exception:
            pass
