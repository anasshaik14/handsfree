"""Train the custom head-pose CNN on 300W-LP (evaluate on AFLW2000).

Datasets (download once, unzip):
  300W-LP   http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
  AFLW2000  (same page) — used only for held-out TEST.
Each image has a matching .mat with 'Pose_Para' (pitch, yaw, roll in radians)
and 'pt2d'/'pt3d_68' (landmarks, for the face crop).

SPLIT PROTOCOL: 300W-LP is train+val, AFLW2000 is the untouched test set (the
standard head-pose protocol). 300W-LP synthesizes ~34 profile views of each of
its 3,837 base faces, so val is carved off by FACE IDENTITY (the filename minus
the view/_Flip suffix): a random split would put other views of the same face
in both train and val and score memorisation. Hyperparameters are picked on the
300W-LP val faces; AFLW2000 is read once for the final report.

Usage:
  python scripts/train_headpose.py --data path/to/300W_LP --eval path/to/AFLW2000
  python scripts/train_headpose.py --data ... --search 20
  python scripts/train_headpose.py --data ... --eval ... --test
  python scripts/train_headpose.py --smoke

Saves models/custom/headpose_cnn.pt (TorchScript) + headpose_cnn.metrics.json.
Output order is (yaw, pitch, roll) in degrees — matches app.headpose.custom_cnn.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from app.headpose.custom_cnn import WEIGHTS_PATH  # noqa: E402
from app.ml import preprocess  # noqa: E402
from app.ml.torch_models import HeadPoseCNN  # noqa: E402
import metrics  # noqa: E402
import tuning  # noqa: E402

DEG = 180.0 / math.pi
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

SPACE = {
    "opt": ["adam", "adamw", "sgd"],
    "lr": ("loguniform", 1e-4, 3e-3),
    "weight_decay": ("loguniform", 1e-6, 1e-3),
    "dropout": ("uniform", 0.1, 0.5),
    "width": [16, 32],
    "fc": [64, 128],
    "sched": ["step", "cosine", "none"],
    "aug": [0, 1],
    "batch": [32, 64, 128],
}
BASELINE = {"opt": "adam", "lr": 1e-3, "weight_decay": 0.0, "dropout": 0.3,
            "width": 16, "fc": 128, "sched": "step", "aug": 0, "batch": 64}


def _landmarks(m):
    p = m.get("pt3d_68")
    if p is not None and p.shape[0] >= 2 and p.shape[1] == 68:
        return p[:2]
    p = m.get("pt2d")
    if p is not None and p.shape[0] == 2:
        return p
    return None


def _face_id(img_path: str) -> str:
    """Base face identity from a 300W-LP filename: drop the trailing view index
    and the _Flip tag, so all synthesized views of one face share a key."""
    n = os.path.splitext(os.path.basename(img_path))[0]
    return re.sub(r"_\d+$", "", n).replace("_Flip", "")


def _load_pairs(root, limit, max_angle=99.0, tag=""):
    """Load (crop uint8 HWC RGB, [yaw,pitch,roll] deg, face_id) from a dir tree.
    Cached to .npz per root so the loadmat+crop pass runs once."""
    from scipy.io import loadmat
    cache = CACHE_DIR / f"headpose_{tag or Path(root).name}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        crops, tgts, ids = z["crops"], z["targets"], z["ids"]
    else:
        pairs = []
        for img in sorted(glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)):
            mat = os.path.splitext(img)[0] + ".mat"
            if os.path.exists(mat):
                pairs.append((img, mat))
        if not pairs:
            raise SystemExit(f"no image/.mat pairs under {root!r}")
        # Preallocate and fill in place instead of a python list + np.stack at
        # the end: stacking ~120k HWC uint8 crops needs the whole list AND the
        # freshly-stacked array in memory at once (~2x peak, plus another copy
        # from a needless .astype()) — on this box's tight RAM that triggered
        # swap thrashing that stalled the whole decode. One preallocated array,
        # written in place, keeps peak memory to the final size.
        n_max = len(pairs)
        crops = np.empty((n_max, *preprocess.FACE_SIZE, 3), dtype=np.uint8)
        tgts = np.empty((n_max, 3), dtype=np.float32)
        ids = np.empty(n_max, dtype=object)
        n = 0
        for k, (img_path, mat_path) in enumerate(pairs):
            m = loadmat(mat_path, variable_names=["Pose_Para", "pt2d", "pt3d_68"])
            pitch, yaw, roll = (m["Pose_Para"][0][:3] * DEG)
            if max(abs(yaw), abs(pitch), abs(roll)) > max_angle:
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            pts = _landmarks(m)
            if pts is not None:
                xs, ys = pts[0], pts[1]
                h, w = img.shape[:2]
                x0, x1 = max(int(xs.min()), 0), min(int(xs.max()), w)
                y0, y1 = max(int(ys.min()), 0), min(int(ys.max()), h)
                pad = int(0.15 * max(x1 - x0, y1 - y0))
                crop = img[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]
            else:
                crop = img
            if crop.size == 0:
                continue
            crop = cv2.cvtColor(cv2.resize(crop, (preprocess.FACE_SIZE[1], preprocess.FACE_SIZE[0])),
                                cv2.COLOR_BGR2RGB)
            crops[n] = crop
            tgts[n] = (yaw, pitch, roll)
            ids[n] = _face_id(img_path)
            n += 1
            if (k + 1) % 20000 == 0:
                print(f"  loaded {k + 1}/{len(pairs)} ({n} kept)", flush=True)
        if not n:
            raise SystemExit(f"no usable samples under {root!r}")
        crops, tgts, ids = crops[:n], tgts[:n], ids[:n].astype(str)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, crops=crops, targets=tgts, ids=ids)
        print(f"cached -> {cache}", flush=True)
    if limit and limit < len(tgts):
        keep = np.linspace(0, len(tgts) - 1, limit).astype(int)
        crops, tgts, ids = crops[keep], tgts[keep], ids[keep]
    return crops, tgts, ids


class HeadPoseSet(torch.utils.data.Dataset):
    def __init__(self, crops, targets, ids):
        self.crops, self.targets, self.ids = crops, targets, ids
        self.aug = False

    def __len__(self): return len(self.targets)

    def __getitem__(self, i):
        crop = self.crops[i]
        yaw, pitch, roll = self.targets[i]
        if self.aug and random.random() < 0.5:
            # Horizontal flip -> yaw and roll change sign (a mirror reverses
            # left/right turn and tilt direction); pitch is unchanged.
            crop = crop[:, ::-1]
            yaw, roll = -yaw, -roll
        if self.aug and random.random() < 0.5:
            crop = np.clip(crop.astype(np.float32) * random.uniform(0.8, 1.2)
                           + random.uniform(-25, 25), 0, 255).astype(np.uint8)
        crop = preprocess.norm_rgb(np.ascontiguousarray(crop).astype(np.float32) / 255.0).transpose(2, 0, 1)
        return (torch.from_numpy(crop.copy()),
                torch.tensor([yaw, pitch, roll], dtype=torch.float32))


class _SmokeHead(HeadPoseSet):
    def __init__(self, n: int = 256) -> None:
        rng = np.random.default_rng(0)
        crops = rng.integers(0, 255, (n, *preprocess.FACE_SIZE, 3), np.uint8)
        tgts = rng.uniform(-40, 40, (n, 3)).astype(np.float32)
        ids = np.array([f"f{i % 8}" for i in range(n)])
        super().__init__(crops, tgts, ids)


def mae(model, view, device) -> float:
    dl = DataLoader(view, batch_size=256)
    yt, yp = metrics.collect_regression(model.to(device), dl, device)
    return float(np.abs(yp - yt).mean())


def train_one(cfg, ds, idx_tr, idx_va, scale, device, epochs, seed=0, patience=6, verbose=False):
    tuning.seed_all(seed)
    train_dl = DataLoader(Subset(ds, idx_tr), batch_size=int(cfg["batch"]), shuffle=True)
    val_view = Subset(ds, idx_va)
    scale_t = torch.tensor(scale, dtype=torch.float32, device=device)

    model = HeadPoseCNN(width=int(cfg["width"]), fc=int(cfg["fc"]),
                        dropout=float(cfg["dropout"])).to(device)
    opt = tuning.make_optimizer(cfg["opt"], model.parameters(), float(cfg["lr"]),
                                float(cfg["weight_decay"]))
    sched = tuning.make_sched(cfg["sched"], opt, epochs)

    # Per-axis std-scaled MSE: yaw spans +-90 while pitch/roll are narrower, so
    # plain summed MSE lets yaw dominate and pitch/roll never learn. Keep it.
    def loss_fn(pred, y):
        return (((pred - y) / scale_t) ** 2).mean()

    best_mae, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        ds.aug = bool(cfg["aug"])
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss_fn(model(x.to(device)), y.to(device)).backward()
            opt.step()
        if sched:
            sched.step()
        ds.aug = False
        m = mae(model, val_view, device)
        if verbose:
            print(f"epoch {epoch + 1}/{epochs}  val_MAE={m:.3f} deg", flush=True)
        if m < best_mae:
            best_mae, bad = m, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_mae


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--eval", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--search", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--patience", type=int, default=6, help="early-stopping patience")
    args = ap.parse_args()

    tuning.cpu_threads()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        ds = _SmokeHead()
    else:
        crops, tgts, ids = _load_pairs(args.data, args.limit, tag="300w_lp")
        ds = HeadPoseSet(crops, tgts, ids)
    # Val carved off 300W-LP by FACE IDENTITY (test = AFLW2000, loaded below).
    idx_tr, idx_va, _ = tuning.group_split(ds.ids, (0.85, 0.15, 0.0), seed=args.seed)
    scale = np.std(ds.targets, axis=0).clip(min=1.0)
    print(f"300W-LP {len(ds)} imgs / {len(set(ds.ids))} faces -> "
          f"train {len(idx_tr)} | val {len(idx_va)} (face-disjoint); test = AFLW2000", flush=True)

    if args.search:
        rng = random.Random(args.seed)
        best = None
        for t in range(args.search):
            cfg = BASELINE.copy() if t == 0 else tuning.sample_cfg(rng, SPACE)
            _, m = train_one(cfg, ds, idx_tr, idx_va, scale, device, args.epochs, seed=args.seed)
            tuning.log_trial("headpose", {"trial": t, **cfg}, {"val_mae_deg": round(m, 4)})
            print(f"trial {t + 1}/{args.search} val_MAE={m:.3f}  {cfg}", flush=True)
            if best is None or m < best[1]:
                best = (cfg, m)
        print(f"\nBEST val_MAE={best[1]:.3f} deg\n{json.dumps(best[0])}")
        return

    cfg = json.loads(args.config) if args.config else BASELINE.copy()
    model, val_mae = train_one(cfg, ds, idx_tr, idx_va, scale, device, args.epochs,
                               seed=args.seed, patience=args.patience, verbose=True)

    # Report on AFLW2000 (test) when --test + --eval given, else 300W-LP val.
    if args.test and args.eval:
        ec, et, ei = _load_pairs(args.eval, 0, tag="aflw2000")
        report_ds = HeadPoseSet(ec, et, ei)
        report_view = Subset(report_ds, list(range(len(report_ds))))
        split = "test(AFLW2000)"
    else:
        report_ds, report_view, split = ds, Subset(ds, idx_va), "val(300W-LP)"
    report_ds.aug = False
    dl = DataLoader(report_view, batch_size=256)
    y_true, y_pred = metrics.collect_regression(model.to(device), dl, device)
    full_metrics = metrics.regression_metrics(y_true, y_pred, ["yaw", "pitch", "roll"])
    full_metrics["val_mae_deg"] = val_mae
    full_metrics["output_order"] = ["yaw", "pitch", "roll"]
    full_metrics["split"] = split
    full_metrics["protocol"] = "train/val = 300W-LP face-disjoint; test = AFLW2000 (untouched)"
    full_metrics["hyperparameters"] = cfg

    out = tuning.out_path(WEIGHTS_PATH, args.smoke)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.zeros(1, 3, *preprocess.FACE_SIZE)
    torch.jit.trace(model.cpu(), example).save(str(out))
    out.with_suffix(".metrics.json").write_text(json.dumps(full_metrics, indent=2))
    print(f"saved {out}  ({split} MAE={full_metrics['mae_deg']:.3f} deg)")


if __name__ == "__main__":
    main()
