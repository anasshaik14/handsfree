"""Shared preprocessing for the custom CNN models — torch-free (numpy/cv2
only) so it imports fine even when torch isn't installed. The same crop /
resize / normalize used here at RUNTIME (cropping from the live camera frame
via FaceMesh landmarks) must match how the TRAINING scripts prepare the
public-dataset images, or the model sees a different input distribution than
it trained on (train/serve skew).

Input sizes are the conventions of the datasets each model trains on:
  EYE_SIZE      32x32   grayscale eye crop      (MRL Eye Dataset, blink)
  GAZE_EYE_SIZE 36x60   grayscale eye patch     (MPIIGaze, gaze)
  FACE_SIZE     64x64   RGB face crop           (300W-LP, head pose)
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.vision import landmarks as L

EYE_SIZE = (32, 32)          # (h, w)
GAZE_EYE_SIZE = (36, 60)     # (h, w) — MPIIGaze normalized eye patch
FACE_SIZE = (64, 64)         # (h, w)


def _bbox_from_indices(pts: np.ndarray, idx: list[int], pad_frac: float,
                       w: int, h: int) -> Optional[tuple[int, int, int, int]]:
    xs = pts[idx, 0]
    ys = pts[idx, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    bw, bh = x1 - x0, y1 - y0
    if bw < 2 or bh < 2:
        return None
    px, py = bw * pad_frac, bh * pad_frac
    ix0 = int(max(x0 - px, 0))
    iy0 = int(max(y0 - py, 0))
    ix1 = int(min(x1 + px, w))
    iy1 = int(min(y1 + py, h))
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        return None
    return ix0, iy0, ix1, iy1


def crop_eye_gray(frame_bgr: np.ndarray, pts: np.ndarray, eye: str,
                  size: tuple[int, int] = EYE_SIZE,
                  pad_frac: float = 0.35) -> Optional[np.ndarray]:
    """Grayscale eye crop, resized to `size`, values in [0,1]. `eye` is
    'left'/'right' from the subject's point of view."""
    box = L.LEFT_EYE_BOX if eye == "left" else L.RIGHT_EYE_BOX
    idx = [box["outer"], box["inner"], box["top"], box["bottom"]]
    h, w = frame_bgr.shape[:2]
    bb = _bbox_from_indices(pts, idx, pad_frac, w, h)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    crop = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return crop.astype(np.float32) / 255.0


def crop_face_rgb(frame_bgr: np.ndarray, pts: np.ndarray,
                  size: tuple[int, int] = FACE_SIZE,
                  pad_frac: float = 0.15) -> Optional[np.ndarray]:
    """RGB face crop (all face landmarks' bounding box), resized to `size`,
    values in [0,1]."""
    h, w = frame_bgr.shape[:2]
    idx = list(range(min(468, len(pts))))
    bb = _bbox_from_indices(pts, idx, pad_frac, w, h)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    crop = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return crop.astype(np.float32) / 255.0


# ImageNet-ish normalization applied on top of the [0,1] crops above; the
# training scripts apply the identical constants.
GRAY_MEAN, GRAY_STD = 0.5, 0.5
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def norm_gray(img: np.ndarray) -> np.ndarray:
    return (img - GRAY_MEAN) / GRAY_STD


def norm_rgb(img: np.ndarray) -> np.ndarray:
    return (img - np.array(RGB_MEAN, np.float32)) / np.array(RGB_STD, np.float32)
