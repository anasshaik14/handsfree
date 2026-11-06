"""Train the custom gaze CNN on MPIIGaze (normalized eye patches).

Dataset: MPIIGaze — https://www.mpi-inf.mpg.de/mpiigaze
Use the "Data/Normalized" part: per-person p00.mat..p14.mat, each holding
left/right eye images (36x60 grayscale) and 3D gaze vectors. This script
reads them via scipy.io.loadmat, converts each gaze vector to (pitch, yaw)
radians, and trains GazeCNN to regress them from the eye patch.

SPLIT PROTOCOL: participant-disjoint over the 15 people (p00..p14),
70/15/15 train/val/test. MPIIGaze has thousands of frames per participant, so
a random row split leaks a person's appearance across the split; splitting on
the participant makes the val/test numbers cross-subject (the real use case:
a new user the model never saw).

WARNING (train/serve gap): this normalized-patch score does NOT transfer to the
live app, which crops eyes from the raw frame via FaceMesh. Report the
app-pipeline gaze number from scripts/eval_pretrained.py separately.

Usage:
  python scripts/train_gaze.py --data path/to/MPIIGaze/Data/Normalized
  python scripts/train_gaze.py --data ... --search 24
  python scripts/train_gaze.py --data ... --test
  python scripts/train_gaze.py --smoke

Saves models/custom/gaze_cnn.pt (TorchScript) + gaze_cnn.metrics.json.
Output order is (pitch, yaw) in radians — matches app.gaze.custom_cnn.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset, Subset  # noqa: E402

from app.gaze.custom_cnn import WEIGHTS_PATH  # noqa: E402
from app.ml import preprocess  # noqa: E402
from app.ml.torch_models import GazeCNN  # noqa: E402
import metrics  # noqa: E402
import tuning  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "mpiigaze.npz"

SPACE = {
    "opt": ["adam", "adamw", "sgd"],
    "lr": ("loguniform", 1e-4, 3e-3),
    "weight_decay": ("loguniform", 1e-6, 1e-3),
    "dropout": ("uniform", 0.1, 0.5),
    "width": [16, 32],
    "fc": [64, 128],
    "sched": ["none", "step", "cosine"],
    "aug": [0, 1],
    "batch": [128, 256],
}
BASELINE = {"opt": "adam", "lr": 1e-3, "weight_decay": 0.0, "dropout": 0.3,
            "width": 16, "fc": 128, "sched": "none", "aug": 0, "batch": 256}


def _vec_to_pitch_yaw(g: np.ndarray) -> tuple[float, float]:
    g = g / (np.linalg.norm(g) + 1e-9)
    pitch = math.asin(-float(g[1]))
    yaw = math.atan2(-float(g[0]), -float(g[2]))
    return pitch, yaw


def _iter_eyes(mat):
    """Yield (image 36x60 uint8, gaze 3-vector) for every sample/eye in one
    MPIIGaze normalized .mat."""
    data = mat["data"]
    for side in ("left", "right"):
        try:
            node = data[side][0, 0] if data.dtype.names else data[0, 0][side][0, 0]
            images = np.asarray(node["image"][0, 0])
            gazes = np.asarray(node["gaze"][0, 0])
        except Exception:
            continue
        if images.ndim == 2:
            images, gazes = images[None], gazes[None]
        for img, gz in zip(images, gazes):
            yield np.asarray(img, np.uint8), np.asarray(gz, np.float64).reshape(-1)[:3]


def _participant(path: str) -> str:
    """p00..p14 from the .mat path — the group key."""
    m = os.path.basename(path)
    return m.split(".")[0][:3] if m[0] == "p" else os.path.dirname(path).split(os.sep)[-1]


class MPIIGaze(Dataset):
    """Preloaded normalized eye patches + (pitch,yaw) + participant id, cached
    to .npz so the scipy loadmat pass (slow) runs once."""

    def __init__(self, root: str, limit: int = 0) -> None:
        if CACHE.exists():
            z = np.load(CACHE)
            self.imgs, self.targets, self.parts = z["imgs"], z["targets"], z["parts"]
        else:
            from scipy.io import loadmat
            import cv2
            files = sorted(glob.glob(os.path.join(root, "**", "*.mat"), recursive=True))
            if not files:
                raise SystemExit(f"no .mat under {root!r}")
            imgs, tgts, parts = [], [], []
            for mp in files:
                pid = _participant(mp)
                for img, gz in _iter_eyes(loadmat(mp)):
                    if img.shape != preprocess.GAZE_EYE_SIZE:
                        img = cv2.resize(img, (preprocess.GAZE_EYE_SIZE[1], preprocess.GAZE_EYE_SIZE[0]))
                    imgs.append(img.astype(np.uint8))
                    tgts.append(_vec_to_pitch_yaw(gz))
                    parts.append(pid)
                print(f"  loaded {pid}: {len(imgs)} samples so far", flush=True)
            self.imgs = np.stack(imgs)                      # (N,36,60) uint8
            self.targets = np.asarray(tgts, np.float32)     # (N,2) pitch,yaw rad
            self.parts = np.asarray(parts)
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(CACHE, imgs=self.imgs, targets=self.targets, parts=self.parts)
            print(f"cached -> {CACHE}", flush=True)
        if limit and limit < len(self.targets):
            keep = np.linspace(0, len(self.targets) - 1, limit).astype(int)
            self.imgs, self.targets, self.parts = self.imgs[keep], self.targets[keep], self.parts[keep]
        self.aug = False

    def __len__(self) -> int:
        return len(self.targets)

    def imgs_tensor(self, i: int) -> torch.Tensor:
        """Normalized patch as a (1,H,W) tensor, no augmentation — used by the
        report to build the test batch."""
        img = preprocess.norm_gray(self.imgs[i].astype(np.float32) / 255.0)
        return torch.from_numpy(img[None, :, :].copy())

    def __getitem__(self, i: int):
        img = self.imgs[i].copy()
        pitch, yaw = self.targets[i]
        if self.aug and random.random() < 0.8:
            # Horizontal flip mirrors the eye left<->right, so the yaw sign
            # must flip too; pitch is unchanged. Getting this wrong teaches the
            # model the opposite direction on half the data.
            if random.random() < 0.5:
                # ascontiguousarray: the ::-1 view has negative strides, which
                # every cv2 call below rejects ("Layout of the output array is
                # incompatible"). Materialise it once here.
                img = np.ascontiguousarray(img[:, ::-1])
                yaw = -yaw

            # Gaussian blur (simulate low-resolution webcam sensor noise)
            if random.random() < 0.3:
                img = cv2.GaussianBlur(img, (5, 5), 0)

            # Brightness jitter (simulate lighting changes: indoor/outdoor/shadows)
            if random.random() < 0.4:
                brightness = random.uniform(0.7, 1.3)
                img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

            # Contrast jitter (simulate exposure variations)
            if random.random() < 0.4:
                contrast = random.uniform(0.8, 1.2)
                img = np.clip((img.astype(np.float32) - 128) * contrast + 128, 0, 255).astype(np.uint8)

            # Small rotation (simulate head tilt in webcam, ±8 degrees)
            if random.random() < 0.3:
                angle = random.uniform(-8, 8)
                h, w = img.shape
                M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        img = preprocess.norm_gray(img.astype(np.float32) / 255.0)
        return (torch.from_numpy(img[None, :, :].copy()),
                torch.tensor([pitch, yaw], dtype=torch.float32))


class _SmokeGaze(Dataset):
    def __init__(self, n: int = 512) -> None:
        rng = np.random.default_rng(0)
        self.imgs = rng.integers(0, 255, (n, *preprocess.GAZE_EYE_SIZE), np.uint8)
        self.targets = rng.uniform(-0.4, 0.4, (n, 2)).astype(np.float32)
        self.parts = np.array([f"p{i % 6:02d}" for i in range(n)])
        self.aug = False

    def __len__(self): return len(self.targets)

    def __getitem__(self, i):
        img = preprocess.norm_gray(self.imgs[i].astype(np.float32) / 255.0)
        return torch.from_numpy(img[None].copy()), torch.from_numpy(self.targets[i])


def _angular(model, view, device):
    """(pitch,yaw) true/pred rad + angular error deg over a Subset view."""
    dl = DataLoader(view, batch_size=256)
    yt, yp = metrics.collect_regression(model.to(device), dl, device)
    ang = metrics.gaze_angular_error_deg(yt, yp)["angular_error_deg_mean"]
    return yt, yp, ang


def train_one(cfg, ds, idx_tr, idx_va, device, epochs, seed=0, patience=6, verbose=False):
    """Train one config; select best epoch on VAL angular error. Returns (model, val_ang)."""
    tuning.seed_all(seed)
    train_dl = DataLoader(Subset(ds, idx_tr), batch_size=int(cfg["batch"]), shuffle=True)
    val_view = Subset(ds, idx_va)

    model = GazeCNN(width=int(cfg["width"]), fc=int(cfg["fc"]),
                    dropout=float(cfg["dropout"])).to(device)
    opt = tuning.make_optimizer(cfg["opt"], model.parameters(), float(cfg["lr"]),
                                float(cfg["weight_decay"]))
    sched = tuning.make_sched(cfg["sched"], opt, epochs)
    loss_fn = nn.MSELoss()
    best_ang, best_state, bad = float("inf"), None, 0
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
        _, _, ang = _angular(model, val_view, device)
        if verbose:
            print(f"epoch {epoch + 1}/{epochs}  val_angular={ang:.3f} deg", flush=True)
        if ang < best_ang:
            best_ang, bad = ang, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ang


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--search", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--patience", type=int, default=6, help="early-stopping patience")
    args = ap.parse_args()

    tuning.cpu_threads()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = _SmokeGaze() if args.smoke else MPIIGaze(args.data, args.limit)
    idx_tr, idx_va, idx_te = tuning.group_split(ds.parts, (0.7, 0.15, 0.15), seed=args.seed)
    print(f"{len(ds)} samples / {len(set(ds.parts))} participants -> "
          f"train {len(idx_tr)} | val {len(idx_va)} | test {len(idx_te)} (participant-disjoint)", flush=True)

    if args.search:
        rng = random.Random(args.seed)
        best = None
        for t in range(args.search):
            cfg = BASELINE.copy() if t == 0 else tuning.sample_cfg(rng, SPACE)
            _, ang = train_one(cfg, ds, idx_tr, idx_va, device, args.epochs, seed=args.seed)
            tuning.log_trial("gaze", {"trial": t, **cfg}, {"val_angular_deg": round(ang, 4)})
            print(f"trial {t + 1}/{args.search} val_angular={ang:.3f}  {cfg}", flush=True)
            if best is None or ang < best[1]:
                best = (cfg, ang)
        print(f"\nBEST val_angular={best[1]:.3f} deg\n{json.dumps(best[0])}")
        return

    cfg = json.loads(args.config) if args.config else BASELINE.copy()
    model, val_ang = train_one(cfg, ds, idx_tr, idx_va, device, args.epochs,
                               seed=args.seed, patience=args.patience, verbose=True)

    ds.aug = False
    report_idx = idx_te if args.test else idx_va
    yt, yp, _ = _angular(model, Subset(ds, report_idx), device)
    full_metrics = metrics.regression_metrics(np.degrees(yt), np.degrees(yp), ["pitch", "yaw"])
    full_metrics.update(metrics.gaze_angular_error_deg(yt, yp))
    full_metrics["output_order"] = ["pitch", "yaw"]
    full_metrics["split"] = "test" if args.test else "val"
    full_metrics["protocol"] = "participant-disjoint 70/15/15 over MPIIGaze p00..p14 (normalized patch)"
    full_metrics["val_angular_deg"] = val_ang
    full_metrics["hyperparameters"] = cfg

    out = tuning.out_path(WEIGHTS_PATH, args.smoke)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.zeros(1, 1, *preprocess.GAZE_EYE_SIZE)
    torch.jit.trace(model.cpu(), example).save(str(out))
    out.with_suffix(".metrics.json").write_text(json.dumps(full_metrics, indent=2))
    print(f"saved {out}  ({full_metrics['split']} MAE={full_metrics['mae_deg']:.3f} deg, "
          f"angular={full_metrics['angular_error_deg_mean']:.3f} deg)")


if __name__ == "__main__":
    main()
