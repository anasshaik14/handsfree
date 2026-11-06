"""Full-resolution matched gaze comparison on the custom model's HELD-OUT participants.

Why this exists
---------------
eval_pretrained_gaze_mpiigaze.py was run with per_participant=2000, so the pretrained
prediction arrays hold 30,000 rows (2,000 x 15 people). gen_matched_arrays.py mirrors
that cap to stay aligned, and restricting to the custom model's held-out participants
then leaves only 4,000 rows -- 6.2% of the 64,172 samples those participants actually
contain. The cap was a runtime bound on the pretrained sweep, not a methodological
choice, and it should not bound the headline comparison.

This script re-runs BOTH models over EVERY held-out sample, with each model receiving
its own native preprocessing:
  * pretrained ResNet-preact : ToTensor(/255) + 2D head pose, right eyes mirrored into
    the left-eye convention and their predicted yaw flipped back afterwards.
  * custom CNN               : raw (unmirrored) patch through preprocess.norm_gray.
Ground truth is computed once and shared, so alignment is exact by construction.

Writes reports/tuning/pretrained/matched_gaze_heldout.npz (overwrites the 4,000-row
capped version) and prints both models' angular error.

Usage:  python scripts/eval_gaze_heldout_full.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

torch.set_num_threads(4)

from app.ml import preprocess  # noqa: E402
import tuning  # noqa: E402
from eval_pretrained_gaze_mpiigaze import (  # noqa: E402
    convert_gaze, convert_pose, load_model,
)

OUT = ROOT / "reports" / "tuning" / "pretrained"
NORM = ROOT / "MPIIGaze_DATASET" / "Data" / "Normalized"
CACHE = ROOT / "data" / "cache" / "mpiigaze.npz"


def held_out_participants(seed: int = 0) -> list[str]:
    parts = np.load(CACHE)["parts"]
    _, _, te = tuning.group_split(parts, (0.7, 0.15, 0.15), seed=seed)
    return sorted({str(p) for p in parts[te]})


def angular_deg(true_deg: np.ndarray, pred_deg: np.ndarray) -> np.ndarray:
    a, b = np.radians(true_deg), np.radians(pred_deg)

    def vec(py):
        p, y = py[:, 0], py[:, 1]
        return np.stack([-np.cos(p) * np.sin(y), -np.sin(p), -np.cos(p) * np.cos(y)], 1)

    return np.degrees(np.arccos(np.clip((vec(a) * vec(b)).sum(1), -1, 1)))


def main() -> None:
    import scipy.io as sio

    test_ids = held_out_participants()
    print(f"held-out participants: {test_ids}")

    net_pt = load_model()
    net_cu = torch.jit.load(str(ROOT / "models/custom/gaze_cnn.pt"), map_location="cpu").eval()

    raw_all, mir_all, pose_all, gaze_all, right_all, pid_all = [], [], [], [], [], []
    t0 = time.time()
    for pid in test_ids:
        pdir = NORM / pid
        for mat in sorted(pdir.glob("day*.mat")):
            d = sio.loadmat(str(mat), struct_as_record=False, squeeze_me=True)["data"]
            for side, mirror in (("left", False), ("right", True)):
                s = getattr(d, side)
                im = np.asarray(s.image, np.uint8).reshape(-1, 36, 60)
                hp = convert_pose(np.asarray(s.pose, np.float64).reshape(-1, 3))
                gz = convert_gaze(np.asarray(s.gaze, np.float64).reshape(-1, 3))
                raw_all.append(im)                       # custom: raw, never mirrored
                if mirror:                               # pretrained: left-eye convention
                    mir_all.append(im[:, :, ::-1])
                    pose_all.append(hp * np.array([1, -1]))
                else:
                    mir_all.append(im)
                    pose_all.append(hp)
                gaze_all.append(gz)
                right_all.append(np.full(len(im), mirror))
                pid_all.append(np.full(len(im), pid))
        print(f"  loaded {pid} ({time.time() - t0:.0f}s)", flush=True)

    raw = np.concatenate(raw_all)
    mir = np.ascontiguousarray(np.concatenate(mir_all))
    pose = np.concatenate(pose_all).astype(np.float32)
    true = np.degrees(np.concatenate(gaze_all))
    rmask = np.concatenate(right_all)
    pid = np.concatenate(pid_all)
    print(f"total held-out eyes: {len(true)}")

    # ---- pretrained: /255 ToTensor + head pose
    x = torch.from_numpy(np.ascontiguousarray(mir.astype(np.float32) / 255.0))[:, None]
    y = torch.from_numpy(pose)
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), 4096):
            outs.append(net_pt(x[i:i + 4096], y[i:i + 4096]).numpy())
    pred_pt = np.concatenate(outs)
    pred_pt[rmask, 1] *= -1                              # undo the mirror on yaw
    pred_pt = np.degrees(pred_pt)

    # ---- custom: its own norm_gray on the raw patch
    xc = preprocess.norm_gray(raw.astype(np.float32) / 255.0)[:, None]
    xc = torch.from_numpy(np.ascontiguousarray(xc))
    outs = []
    with torch.no_grad():
        for i in range(0, len(xc), 4096):
            outs.append(net_cu(xc[i:i + 4096]).numpy())
    pred_cu = np.degrees(np.concatenate(outs))

    np.savez(OUT / "matched_gaze_heldout.npz", true=true, pred_custom=pred_cu,
             pred_pt=pred_pt, axes=np.array(["pitch", "yaw"]), pid=pid)

    ac, ap = angular_deg(true, pred_cu).mean(), angular_deg(true, pred_pt).mean()
    print(f"\nN={len(true)} held-out eyes ({time.time() - t0:.0f}s)")
    print(f"  custom     angular={ac:.3f}  MAE={np.abs(true - pred_cu).mean():.3f}")
    print(f"  pretrained angular={ap:.3f}  MAE={np.abs(true - pred_pt).mean():.3f}")
    print(f"wrote {OUT / 'matched_gaze_heldout.npz'}")


if __name__ == "__main__":
    main()
