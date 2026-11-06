"""Evaluate the pretrained 6DRepNet head-pose model on AFLW2000, producing the
SAME regression metric suite used for our custom head-pose CNN, so the two are
directly comparable.

- 6DRepNet is trained on 300W-LP (same as our custom model) and AFLW2000 is the
  standard held-out test set for both — a fair comparison.
- Each face is cropped from the full-resolution image using the landmark
  bounding box (15% pad), matching how our own eval frames the face, but kept at
  native resolution (6DRepNet resizes to 224 internally). Angles filtered to
  |angle| <= 99 deg, matching our custom eval's 1969-image set.
- Output reordered to (yaw, pitch, roll) to match our convention.

Writes reports/tuning/pretrained/headpose_6drepnet.metrics.json
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AFLW = ROOT / "AFLW2000_DATASET"
OUT = ROOT / "reports" / "tuning" / "pretrained"
DEG = 180.0 / np.pi


def face_crop(img, m):
    """Native-res face crop from landmark bbox (15% pad), same framing as our eval."""
    p = m.get("pt3d_68")
    if p is not None and p.shape[0] >= 2 and p.shape[1] == 68:
        pts = p[:2]
    else:
        pts = m.get("pt2d")
    if pts is None or pts.shape[0] != 2:
        return img
    xs, ys = pts[0], pts[1]
    h, w = img.shape[:2]
    x0, x1 = max(int(xs.min()), 0), min(int(xs.max()), w)
    y0, y1 = max(int(ys.min()), 0), min(int(ys.max()), h)
    pad = int(0.15 * max(x1 - x0, y1 - y0))
    crop = img[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]
    return crop if crop.size else img


def main():
    from sixdrepnet import SixDRepNet
    model = SixDRepNet(gpu_id=-1,
                       dict_path=str(ROOT / "models/pretrained/6drepnet/6DRepNet_300W_LP_AFLW2000.pth"))

    gt, pred = [], []
    files = sorted(glob.glob(str(AFLW / "*.jpg")))
    t0 = time.time()
    for i, ip in enumerate(files):
        mp = ip[:-4] + ".mat"
        if not os.path.exists(mp):
            continue
        m = loadmat(mp, variable_names=["Pose_Para", "pt2d", "pt3d_68"])
        pitch, yaw, roll = m["Pose_Para"][0][:3] * DEG
        if max(abs(yaw), abs(pitch), abs(roll)) > 99.0:
            continue
        img = cv2.imread(ip)
        if img is None:
            continue
        crop = face_crop(img, m)
        p, y, r = model.predict(crop)                 # 6DRepNet -> (pitch, yaw, roll)
        gt.append([yaw, pitch, roll])                 # our order: yaw, pitch, roll
        pred.append([float(y[0]), float(p[0]), float(r[0])])
        if (i + 1) % 400 == 0:
            print(f"  {i + 1}/{len(files)}  ({time.time() - t0:.0f}s)", flush=True)

    gt = np.asarray(gt, np.float64)
    pred = np.asarray(pred, np.float64)
    full = metrics.regression_metrics(gt, pred, ["yaw", "pitch", "roll"])
    full["model"] = "6DRepNet (pretrained, 300W-LP)"
    full["test_set"] = "AFLW2000"
    full["output_order"] = ["yaw", "pitch", "roll"]

    OUT.mkdir(parents=True, exist_ok=True)
    # save raw arrays so report/Excel use the SAME canonical reg_metrics()
    np.savez(OUT / "headpose_6drepnet.arrays.npz", true_deg=gt, pred_deg=pred)
    (OUT / "headpose_6drepnet.metrics.json").write_text(json.dumps(full, indent=2))
    print(f"\n6DRepNet on AFLW2000 ({len(gt)} imgs): "
          f"MAE={full['mae_deg']:.3f} deg  RMSE={full['rmse_deg']:.3f}  "
          f"(custom CNN was ~7.4 deg)")
    print(f"wrote {OUT / 'headpose_6drepnet.metrics.json'}")


if __name__ == "__main__":
    main()
