"""Evaluate the pretrained L2CS-Net gaze model on MPIIGaze, producing the SAME
regression/angular metric suite used for our custom gaze CNN.

Honest caveats (stated in the output JSON):
- The available L2CS weights are GAZE360-trained, so this is a CROSS-DATASET
  test on MPIIGaze (out of domain) — expect worse than its in-domain 3.9 deg.
- L2CS needs a full-face crop; we take the FaceMesh face bounding box from the
  MPIIGaze Original image (the app's real front end), resize to 224, ImageNet-
  normalize — the same FaceMesh path used to score every other gaze provider.
- Gaze360 and MPIIGaze can differ in pitch/yaw SIGN convention. We pick the
  sign alignment (of 4) that minimises angular error and report it — a
  convention fix, not per-sample fitting, mirroring how the classical head-pose
  providers are aligned in eval_pretrained.py.

Writes reports/tuning/pretrained/gaze_l2cs.metrics.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import metrics  # noqa: E402
from eval_pretrained import _face_landmarker, _mpii_truth  # noqa: E402

OUT = ROOT / "reports" / "tuning" / "pretrained"
WEIGHTS = ROOT / "models/pretrained/l2cs/L2CSNet_gaze360.pkl"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_l2cs():
    import torch
    import torchvision.models.resnet as R
    spec = importlib.util.spec_from_file_location("l2cs_model", str(ROOT / "app/gaze/l2cs_model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = mod.L2CS(R.Bottleneck, [3, 4, 6, 3], 90)
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=False))
    net.eval()
    return net


def decode(net, crop_rgb224):
    """crop_rgb224: (224,224,3) uint8 RGB -> (pitch_deg, yaw_deg)."""
    import torch
    import torch.nn.functional as F
    x = (crop_rgb224.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1)[None]).float()
    with torch.no_grad():
        yaw_l, pitch_l = net(x)
        idx = torch.arange(90, dtype=torch.float32)
        pitch = float((F.softmax(pitch_l, 1)[0] * idx).sum() * 4 - 180)
        yaw = float((F.softmax(yaw_l, 1)[0] * idx).sum() * 4 - 180)
    return pitch, yaw


def main(per_participant=200):
    import cv2
    from app.core.types import FaceResult

    net = load_l2cs()
    mp, lm = _face_landmarker(confidence=0.15)

    orig = ROOT / "MPIIGaze_DATASET" / "Data" / "Original"
    subset = ROOT / "MPIIGaze_DATASET" / "Evaluation Subset" / "annotation for face image"

    gt, pred = [], []
    n_total = n_det = 0
    t0 = time.time()
    for plist in sorted(subset.glob("p*.txt")):
        pid = plist.stem
        want, seen = [], set()
        for l in plist.read_text().split("\n"):
            if l.strip():
                rel = l.split()[0]
                if rel not in seen:
                    seen.add(rel); want.append(rel)
        if per_participant and len(want) > per_participant:
            step = len(want) / per_participant
            want = [want[int(i * step)] for i in range(per_participant)]
        ann_cache = {}
        for rel in want:
            day, fname = rel.split("/")
            if day not in ann_cache:
                p = orig / pid / day / "annotation.txt"
                ann_cache[day] = p.read_text().splitlines() if p.exists() else []
            idx = int(Path(fname).stem) - 1
            lines = ann_cache[day]
            if not (0 <= idx < len(lines)):
                continue
            truth = _mpii_truth(lines[idx])
            if truth is None:
                continue
            n_total += 1
            img = cv2.imread(str(orig / pid / day / fname))
            if img is None:
                continue
            h, w = img.shape[:2]
            rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.face_landmarks:
                continue
            pts = np.array([(q.x * w, q.y * h) for q in res.face_landmarks[0]])
            x0, y0 = pts[:, 0].min(), pts[:, 1].min()
            x1, y1 = pts[:, 0].max(), pts[:, 1].max()
            padx, pady = 0.2 * (x1 - x0), 0.2 * (y1 - y0)
            cx0, cy0 = max(int(x0 - padx), 0), max(int(y0 - pady), 0)
            cx1, cy1 = min(int(x1 + padx), w), min(int(y1 + pady), h)
            crop = rgb[cy0:cy1, cx0:cx1]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (224, 224))
            n_det += 1
            pitch_p, yaw_p = decode(net, crop)
            gt.append(truth)                       # (pitch_deg, yaw_deg)
            pred.append((pitch_p, yaw_p))
        print(f"  {pid}: {n_det} usable so far ({time.time()-t0:.0f}s)", flush=True)

    gt = np.asarray(gt, np.float64)
    pred = np.asarray(pred, np.float64)

    # convention alignment: pick sign combo AND optional pitch<->yaw axis swap
    # minimising mean angular error (Gaze360 vs MPIIGaze can differ in both).
    best = None
    for swap in (False, True):
        base = pred[:, ::-1] if swap else pred
        for sp in (1, -1):
            for sy in (1, -1):
                pr = base * np.array([sp, sy])
                ang = metrics.gaze_angular_error_deg(np.radians(gt), np.radians(pr))["angular_error_deg_mean"]
                if best is None or ang < best[0]:
                    best = (ang, sp, sy, swap, pr)
    _, sp, sy, swap, pred_aligned = best
    print(f"  best alignment: swap_pitch_yaw={swap} sign_pitch={sp} sign_yaw={sy}", flush=True)

    full = metrics.regression_metrics(gt, pred_aligned, ["pitch", "yaw"])
    full.update(metrics.gaze_angular_error_deg(np.radians(gt), np.radians(pred_aligned)))
    full["model"] = "L2CS-Net (pretrained, Gaze360)"
    full["test_set"] = "MPIIGaze (Original, FaceMesh crop) — CROSS-DATASET"
    full["faces_detected"] = f"{n_det}/{n_total}"
    full["sign_alignment"] = {"pitch": sp, "yaw": sy, "swap_pitch_yaw": bool(swap)}
    full["output_order"] = ["pitch", "yaw"]
    full["note"] = ("Gaze360-trained L2CS evaluated cross-dataset on MPIIGaze via FaceMesh "
                    "face crop; pitch/yaw sign convention aligned globally (not per-sample).")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gaze_l2cs.metrics.json").write_text(json.dumps(full, indent=2))
    print(f"\nL2CS on MPIIGaze ({n_det} imgs): angular={full['angular_error_deg_mean']:.3f} deg "
          f"(custom CNN was ~6.1 deg on normalized patches)")
    print(f"wrote {OUT / 'gaze_l2cs.metrics.json'}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    main(per_participant=n)
