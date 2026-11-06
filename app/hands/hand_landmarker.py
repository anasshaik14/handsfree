"""MediaPipe HandLandmarker stage — one hand, 21 landmarks per frame.

Same VIDEO-mode pattern as app.vision.face_mesh.FaceMeshService (its own
monotonic timestamp track, since it's a separate detector instance).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.types import now
from app.system.model_download import ensure_model

log = logging.getLogger(__name__)

MODEL_PATH = Path("models/hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")


@dataclass
class HandResult:
    t: float
    landmarks: Optional[np.ndarray] = None   # (21, 3) pixel coords, or None
    handedness: str = ""                     # 'Left' | 'Right'

    @property
    def ok(self) -> bool:
        return self.landmarks is not None


class HandLandmarkerService:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path, MODEL_URL))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lm = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, frame_bgr: np.ndarray, t: Optional[float] = None) -> HandResult:
        import cv2
        t = t if t is not None else now()
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(t * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        try:
            res = self._lm.detect_for_video(image, ts_ms)
        except Exception:
            log.exception("hand landmarker inference failed")
            return HandResult(t=t)
        if not res.hand_landmarks:
            return HandResult(t=t)
        lm = res.hand_landmarks[0]
        pts = np.array([(p.x * w, p.y * h, p.z * w) for p in lm], dtype=np.float64)
        handedness = ""
        if res.handedness and res.handedness[0]:
            handedness = res.handedness[0][0].category_name
        return HandResult(t=t, landmarks=pts, handedness=handedness)

    def close(self) -> None:
        try:
            self._lm.close()
        except Exception:
            pass
