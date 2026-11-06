"""MediaPipe ObjectDetector (EfficientDet-Lite0, COCO-80 classes) stage.

Run at a low, throttled cadence from the pipeline (object detection is much
more expensive per-frame than face/hand landmarks) — see `objects.interval_s`.

ponytail: EfficientDet-Lite0 is a COCO-80 detector, so it never emits "pen" —
COCO has no pen class. It DOES cover cell phone / book / cup / laptop, which
is the actual overlap with the requested list. A pen detector would need a
custom-trained model; out of scope until asked for specifically.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from app.system.model_download import ensure_model

log = logging.getLogger(__name__)

MODEL_PATH = Path("models/efficientdet_lite0.tflite")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/object_detector/"
             "efficientdet_lite0/float32/1/efficientdet_lite0.tflite")


@dataclass
class Detection:
    label: str
    score: float
    bbox: tuple[int, int, int, int]   # x, y, w, h in pixels


class ObjectDetectorService:
    def __init__(self, model_path: Path = MODEL_PATH,
                 confidence: float = 0.5, max_results: int = 8) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path, MODEL_URL))),
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=confidence,
            max_results=max_results,
        )
        self._det = vision.ObjectDetector.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        import cv2
        # EfficientDet-Lite0's input is 320x320 — feeding it the full 640x480
        # frame just makes its preprocessing chew 4x the pixels (measured
        # 0.5-1.6 s/scan under load). Downscale first, scale boxes back up.
        h, w = frame_bgr.shape[:2]
        scale = 320.0 / w if w > 320 else 1.0
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA) if scale < 1.0 else frame_bgr
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        try:
            res = self._det.detect(image)
        except Exception:
            log.exception("object detector inference failed")
            return []
        out = []
        inv = 1.0 / scale
        for d in res.detections:
            if not d.categories:
                continue
            cat = d.categories[0]
            bb = d.bounding_box
            out.append(Detection(label=cat.category_name, score=round(cat.score, 2),
                                 bbox=(int(bb.origin_x * inv), int(bb.origin_y * inv),
                                       int(bb.width * inv), int(bb.height * inv))))
        return out

    def close(self) -> None:
        try:
            self._det.close()
        except Exception:
            pass
