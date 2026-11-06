"""Train the custom blink CNN on the MRL Eye Dataset.

Dataset: MRL Eye Dataset — ~85k infrared/greyscale eye crops labeled
open/closed. http://mrl.cs.vsb.cz/eyedataset
Filenames encode attributes as  sID_imgID_gender_glasses_EYESTATE_refl_light_sensor.png
where EYESTATE (field index 4) is 1=open, 0=closed.

SPLIT PROTOCOL: subject-disjoint 70/15/15 (train/val/test) over the 37 sIDs.
The dataset holds ~2.3k frames of the same eye per subject, so the old random
row split had near-duplicate frames of one subject on both sides and scored
memorisation, not generalisation. Hyperparameters are selected on VAL; TEST is
read once for the final report.

Usage:
  python scripts/train_blink.py --data mrlEyes_2018_01              # train once
  python scripts/train_blink.py --data ... --search 24              # random search on val
  python scripts/train_blink.py --data ... --test                   # final: train + test read
  python scripts/train_blink.py --smoke

Saves models/custom/blink_cnn.pt (TorchScript) + blink_cnn.metrics.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset, Subset  # noqa: E402

import cv2  # noqa: E402

from app.blink.custom_cnn import CLOSED, OPEN, WEIGHTS_PATH  # noqa: E402
from app.ml import preprocess  # noqa: E402
from app.ml.torch_models import BlinkCNN  # noqa: E402
import metrics  # noqa: E402
import tuning  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "mrl_eyes.npz"

# Random-search space. lr/weight_decay log-scale, the rest small discrete sets.
SPACE = {
    "opt": ["adam", "adamw", "sgd"],
    "lr": ("loguniform", 1e-4, 3e-3),
    "weight_decay": ("loguniform", 1e-6, 1e-3),
    "dropout": ("uniform", 0.1, 0.5),
    "width": [16, 32],
    "fc": [64, 128],
    "sched": ["none", "step", "cosine"],
    "aug": [0, 1],
    "batch": [64, 128, 256],
}
BASELINE = {"opt": "adam", "lr": 1e-3, "weight_decay": 0.0, "dropout": 0.3,
            "width": 16, "fc": 64, "sched": "none", "aug": 0, "batch": 128}


class MRLEyes(Dataset):
    """Preloaded MRL eye crops. Images are cached to an .npz (~87MB uint8) —
    decoding 85k PNGs takes minutes and every search trial would redo it."""

    def __init__(self, root: str, limit: int = 0) -> None:
        if CACHE.exists():
            z = np.load(CACHE)
            self.imgs, self.labels, self.subjects = z["imgs"], z["labels"], z["subjects"]
        else:
            paths = sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True))
            if not paths:
                raise SystemExit(f"no .png images found under {root!r}")
            imgs, labels, subs = [], [], []
            for i, p in enumerate(paths):
                name = os.path.basename(p)
                fields = name.split("_")
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                imgs.append(cv2.resize(img, (preprocess.EYE_SIZE[1], preprocess.EYE_SIZE[0]),
                                       interpolation=cv2.INTER_AREA))
                labels.append(OPEN if int(fields[4]) == 1 else CLOSED)
                subs.append(fields[0])                      # sID — the group key
                if (i + 1) % 20000 == 0:
                    print(f"  decoded {i + 1}/{len(paths)}", flush=True)
            self.imgs = np.stack(imgs)                      # (N,32,32) uint8
            self.labels = np.asarray(labels, np.int64)
            self.subjects = np.asarray(subs)
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(CACHE, imgs=self.imgs, labels=self.labels, subjects=self.subjects)
            print(f"cached -> {CACHE}", flush=True)
        if limit and limit < len(self.labels):
            keep = np.linspace(0, len(self.labels) - 1, limit).astype(int)
            self.imgs, self.labels, self.subjects = self.imgs[keep], self.labels[keep], self.subjects[keep]
        self.aug = False

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        img = self.imgs[i]
        if self.aug:
            img = _augment(img)
        img = preprocess.norm_gray(img.astype(np.float32) / 255.0)
        return torch.from_numpy(img[None, :, :].copy()), int(self.labels[i])


def _augment(img: np.ndarray) -> np.ndarray:
    """Flip / brightness / contrast / small rotation. A horizontal flip is
    label-safe here: eye-open vs closed is unchanged by mirroring (unlike gaze
    or head pose, where flipping must negate yaw)."""
    if random.random() < 0.5:
        img = img[:, ::-1]
    if random.random() < 0.5:
        alpha = random.uniform(0.8, 1.2)                    # contrast
        beta = random.uniform(-25, 25)                      # brightness
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        ang = random.uniform(-10, 10)
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return np.ascontiguousarray(img)


class _SmokeEyes(Dataset):
    """Random but linearly-separable stand-in so the train loop is runnable
    without the 2GB download."""
    def __init__(self, n: int = 512) -> None:
        rng = np.random.default_rng(0)
        self.imgs = np.zeros((n, *preprocess.EYE_SIZE), np.uint8)
        self.labels = np.zeros(n, np.int64)
        for i in range(n):
            label = int(rng.integers(0, 2))
            base = 50 if label == CLOSED else 200
            self.imgs[i] = np.clip(rng.normal(base, 25, preprocess.EYE_SIZE), 0, 255)
            self.labels[i] = label
        self.subjects = np.array([f"s{i % 8:04d}" for i in range(n)])
        self.aug = False

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        img = _augment(self.imgs[i]) if self.aug else self.imgs[i]
        img = preprocess.norm_gray(img.astype(np.float32) / 255.0)
        return torch.from_numpy(img[None, :, :].copy()), int(self.labels[i])


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / max(1, total)


def train_one(cfg: dict, ds, idx_tr, idx_va, device, epochs: int, seed: int = 0,
              patience: int = 4, verbose: bool = False):
    """Train one config; select the best epoch on VAL. Returns (model, val_acc)."""
    tuning.seed_all(seed)
    train_view = Subset(ds, idx_tr)
    val_view = Subset(ds, idx_va)
    train_dl = DataLoader(train_view, batch_size=int(cfg["batch"]), shuffle=True)
    val_dl = DataLoader(val_view, batch_size=256)

    model = BlinkCNN(width=int(cfg["width"]), fc=int(cfg["fc"]),
                     dropout=float(cfg["dropout"])).to(device)
    opt = tuning.make_optimizer(cfg["opt"], model.parameters(), float(cfg["lr"]),
                                float(cfg["weight_decay"]))
    sched = tuning.make_sched(cfg["sched"], opt, epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state, bad = 0.0, None, 0
    for epoch in range(epochs):
        ds.aug = bool(cfg["aug"])       # train with aug ...
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss_fn(model(x.to(device)), y.to(device)).backward()
            opt.step()
        if sched:
            sched.step()
        ds.aug = False                  # ... evaluate without
        acc = evaluate(model, val_dl, device)
        if verbose:
            print(f"epoch {epoch + 1}/{epochs}  val_acc={acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--search", type=int, default=0, help="run N random-search trials on val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", action="store_true",
                    help="final run: report metrics on the held-out TEST subjects")
    ap.add_argument("--config", type=str, default="", help="JSON hyperparameter dict")
    ap.add_argument("--patience", type=int, default=4, help="early-stopping patience")
    args = ap.parse_args()

    tuning.cpu_threads()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = _SmokeEyes() if args.smoke else MRLEyes(args.data, args.limit)
    idx_tr, idx_va, idx_te = tuning.group_split(ds.subjects, (0.7, 0.15, 0.15), seed=args.seed)
    print(f"{len(ds)} imgs / {len(set(ds.subjects))} subjects -> "
          f"train {len(idx_tr)} | val {len(idx_va)} | test {len(idx_te)} (subject-disjoint)", flush=True)

    if args.search:
        rng = random.Random(args.seed)
        best = None
        for t in range(args.search):
            cfg = BASELINE.copy() if t == 0 else tuning.sample_cfg(rng, SPACE)
            _, acc = train_one(cfg, ds, idx_tr, idx_va, device, args.epochs, seed=args.seed)
            tuning.log_trial("blink", {"trial": t, **cfg}, {"val_acc": round(acc, 5)})
            print(f"trial {t + 1}/{args.search} val_acc={acc:.4f}  {cfg}", flush=True)
            if best is None or acc > best[1]:
                best = (cfg, acc)
        print(f"\nBEST val_acc={best[1]:.4f}\n{json.dumps(best[0])}")
        return

    cfg = json.loads(args.config) if args.config else BASELINE.copy()
    model, val_acc = train_one(cfg, ds, idx_tr, idx_va, device, args.epochs,
                               seed=args.seed, patience=args.patience, verbose=True)

    # TEST is touched only here, once, for the chosen config.
    ds.aug = False
    report_idx = idx_te if args.test else idx_va
    report_dl = DataLoader(Subset(ds, report_idx), batch_size=256)
    y_true, y_pred, y_prob = metrics.collect_classification(model, report_dl, device)
    full_metrics = metrics.classification_metrics(y_true, y_pred, y_prob)
    full_metrics["classes"] = {"open": OPEN, "closed": CLOSED}
    full_metrics["split"] = "test" if args.test else "val"
    full_metrics["protocol"] = "subject-disjoint 70/15/15 over MRL sIDs"
    full_metrics["val_accuracy"] = val_acc
    full_metrics["hyperparameters"] = cfg

    out = tuning.out_path(WEIGHTS_PATH, args.smoke)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.zeros(1, 1, *preprocess.EYE_SIZE)
    torch.jit.trace(model.cpu(), example).save(str(out))
    (out.with_suffix(".metrics.json")).write_text(json.dumps(full_metrics, indent=2))
    print(f"saved {out}  ({full_metrics['split']}_acc={full_metrics['accuracy']:.4f})")


if __name__ == "__main__":
    main()
