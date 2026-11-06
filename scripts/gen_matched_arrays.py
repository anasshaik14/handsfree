"""Regenerate MATCHED per-sample predictions: custom CNN vs pretrained NN, on the
SAME samples in the SAME order, each model with its OWN native preprocessing.

Loads the already-saved pretrained per-sample arrays (gaze_mpiigaze / headpose_6drepnet),
re-runs ONLY the tiny custom TorchScript models over the identical sample stream, and
asserts the recomputed ground truth matches the saved pretrained ground truth (alignment
proof). Writes reports/tuning/pretrained/matched_{gaze,headpose}.npz with keys:
    true (N,D)  pred_pt (N,D)  pred_custom (N,D)   [degrees]

Fairness: custom gets norm_gray/norm_rgb (its training pipeline); pretrained preds are
loaded as-is. Same inputs, same GT, each its own preprocessing. No model is handicapped.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from app.ml import preprocess  # norm_gray, norm_rgb, GAZE_EYE_SIZE, FACE_SIZE
import torch
torch.set_num_threads(1)

OUT = ROOT / "reports" / "tuning" / "pretrained"
NORM = ROOT / "MPIIGaze_DATASET" / "Data" / "Normalized"
AFLW = ROOT / "AFLW2000_DATASET"
DEG = 180.0 / np.pi


def convert_gaze(vecs):
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    return np.stack([np.arcsin(-y), np.arctan2(-x, -z)], 1)


# ----------------------------------------------------------------------------- gaze
def gaze():
    import scipy.io as sio
    saved = np.load(OUT / "gaze_mpiigaze.arrays.npz")
    true_saved, pred_pt = saved["true_deg"], saved["pred_deg"]      # (30000,2) pitch,yaw
    net = torch.jit.load(str(ROOT / "models/custom/gaze_cnn.pt"), map_location="cpu").eval()

    per_participant = 2000
    raw_all, true_all, pid_all = [], [], []
    t0 = time.time()
    for pdir in sorted(NORM.glob("p*")):
        raws, gzs = [], []
        for mat in sorted(pdir.glob("day*.mat")):
            d = sio.loadmat(str(mat), struct_as_record=False, squeeze_me=True)["data"]
            for side in ("left", "right"):                          # SAME order as pretrained eval
                s = getattr(d, side)
                im = np.asarray(s.image, np.uint8).reshape(-1, 36, 60)   # RAW (no mirror) for custom
                gz = convert_gaze(np.asarray(s.gaze, np.float64).reshape(-1, 3))
                raws.append(im); gzs.append(gz)
        im = np.concatenate(raws); gz = np.concatenate(gzs)
        if len(im) > per_participant:
            sel = np.linspace(0, len(im) - 1, per_participant).astype(int)
            im, gz = im[sel], gz[sel]
        raw_all.append(im); true_all.append(gz)
        pid_all.append(np.full(len(im), pdir.name))
    raw = np.concatenate(raw_all)
    true = np.degrees(np.concatenate(true_all))
    pid = np.concatenate(pid_all)   # participant of every row -- lets downstream
                                    # code score ONLY the custom model's held-out
                                    # participants (the full 15-participant set
                                    # includes its own training people).

    assert true.shape == true_saved.shape, (true.shape, true_saved.shape)
    d = np.abs(true - true_saved).max()
    assert d < 1e-3, f"gaze GT misaligned (max diff {d})"
    print(f"[gaze] alignment OK (max GT diff {d:.2e}), N={len(true)}", flush=True)

    x = preprocess.norm_gray(raw.astype(np.float32) / 255.0)[:, None]  # (N,1,36,60) custom pipeline
    x = torch.from_numpy(np.ascontiguousarray(x))
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), 4096):
            outs.append(net(x[i:i + 4096]).numpy())
    pred_custom = np.degrees(np.concatenate(outs))                 # rad -> deg, (N,2) pitch,yaw

    np.savez(OUT / "matched_gaze.npz", true=true, pred_pt=pred_pt, pred_custom=pred_custom,
             axes=np.array(["pitch", "yaw"]), pid=pid)
    ac = np.degrees(np.arccos(np.clip(_cos_gaze(true, pred_custom), -1, 1))).mean()
    ap = np.degrees(np.arccos(np.clip(_cos_gaze(true, pred_pt), -1, 1))).mean()
    print(f"[gaze] custom angular={ac:.3f}  pretrained angular={ap:.3f}  ({time.time()-t0:.0f}s)", flush=True)


def _cos_gaze(a_deg, b_deg):
    a, b = np.radians(a_deg), np.radians(b_deg)
    def vec(py):
        p, y = py[:, 0], py[:, 1]
        return np.stack([-np.cos(p) * np.sin(y), -np.sin(p), -np.cos(p) * np.cos(y)], 1)
    va, vb = vec(a), vec(b)
    return (va * vb).sum(1)


# ------------------------------------------------------------------------- headpose
def headpose():
    import cv2, glob, os
    from scipy.io import loadmat
    from eval_pretrained_headpose_6drepnet import face_crop
    saved = np.load(OUT / "headpose_6drepnet.arrays.npz")
    true_saved, pred_pt = saved["true_deg"], saved["pred_deg"]      # (N,3) yaw,pitch,roll
    net = torch.jit.load(str(ROOT / "models/custom/headpose_cnn.pt"), map_location="cpu").eval()

    H, W = preprocess.FACE_SIZE
    true_all, cust = [], []
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
        c = cv2.cvtColor(cv2.resize(crop, (W, H)), cv2.COLOR_BGR2RGB)
        c = preprocess.norm_rgb(c.astype(np.float32) / 255.0).transpose(2, 0, 1)
        with torch.no_grad():
            p = net(torch.from_numpy(c[None]).contiguous()).numpy()[0]   # (yaw,pitch,roll) deg
        cust.append(p); true_all.append([yaw, pitch, roll])
    true = np.asarray(true_all, np.float64)
    pred_custom = np.asarray(cust, np.float64)

    assert true.shape == true_saved.shape, (true.shape, true_saved.shape)
    d = np.abs(true - true_saved).max()
    assert d < 1e-3, f"headpose GT misaligned (max diff {d})"
    print(f"[headpose] alignment OK (max GT diff {d:.2e}), N={len(true)}", flush=True)

    np.savez(OUT / "matched_headpose.npz", true=true, pred_pt=pred_pt, pred_custom=pred_custom,
             axes=np.array(["yaw", "pitch", "roll"]))
    mc = np.abs(true - pred_custom).mean(); mp2 = np.abs(true - pred_pt).mean()
    print(f"[headpose] custom MAE={mc:.3f}  pretrained MAE={mp2:.3f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "gaze"):
        gaze()
    if which in ("all", "headpose"):
        headpose()
    print("done")
