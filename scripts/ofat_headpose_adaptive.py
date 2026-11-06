"""Adaptive OFAT hyperparameter sweep for the head-pose CNN — 10 parameters.

Same method as ofat_blink_adaptive.py: ordered parameters (Learning Rate,
Dropout, Weight Decay, Batch Size, Epochs, Conv Width, FC Width) run
adaptively — minimum 10 trials, stop 3 trials after the best (lowest) MAE
is confirmed, hard cap 25. Categorical parameters (Optimizer, LR Schedule,
Augmentation) use a fixed minimum-10-trial seed-repeated count, since they
have no ordering to walk.

Train/val is carved face-disjoint from 300W-LP only (matching
train_headpose.py's own protocol); AFLW2000 stays untouched, never used
here — this is a validation-only search.

Usage:
  python scripts/ofat_headpose_adaptive.py --data 300W_LP_DATASET --limit 20000 --epochs 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

import train_headpose  # noqa: E402
import metrics  # noqa: E402
import tuning  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "tuning" / "ofat_headpose_adaptive"
SCALAR_KEYS = [
    "mae_deg", "rmse_deg", "median_ae_deg", "max_ae_deg", "p90_ae_deg", "p95_ae_deg",
    "mae_deg_yaw", "rmse_deg_yaw", "bias_deg_yaw", "std_deg_yaw", "r2_yaw", "corr_yaw",
    "mae_deg_pitch", "rmse_deg_pitch", "bias_deg_pitch", "std_deg_pitch", "r2_pitch", "corr_pitch",
    "mae_deg_roll", "rmse_deg_roll", "bias_deg_roll", "std_deg_roll", "r2_roll", "corr_roll",
]

START = dict(opt="adam", lr=1e-3, weight_decay=0.0, dropout=0.3,
             width=16, fc=128, sched="step", aug=0, batch=64)

MIN_TRIALS = 10
CONFIRM_AFTER_PEAK = 3
MAX_TRIALS = 25


def gen_lr():
    v = 0.0001
    while True:
        yield round(v, 8)
        v *= 1.35


def gen_dropout():
    v = 0.05
    while v < 0.95:
        yield round(v, 4)
        v += 0.05


def gen_weight_decay():
    v = 1e-7
    while True:
        yield v
        v *= 2.5


def gen_batch():
    v = 16.0
    while True:
        yield max(4, int(round(v)))
        v *= 1.3


def gen_epochs():
    v = 2
    while True:
        yield v
        v += 1


def gen_width():
    v = 8.0
    while True:
        yield max(4, int(round(v)))
        v *= 1.3


def gen_fc():
    v = 16.0
    while True:
        yield max(8, int(round(v)))
        v *= 1.3


def run_trial(cfg: dict, epochs: int, seed: int, ds, idx_tr, idx_va, scale, device):
    """LOWER MAE (deg) is better."""
    model, _ = train_headpose.train_one(cfg, ds, idx_tr, idx_va, scale, device, epochs,
                                        seed=seed, patience=max(3, epochs), verbose=False)
    ds.aug = False
    y_true, y_pred = metrics.collect_regression(model.to(device), _val_dl(ds, idx_va), device)
    full = metrics.regression_metrics(y_true, y_pred, ["yaw", "pitch", "roll"])
    return full, full["mae_deg"]


def _val_dl(ds, idx_va):
    return DataLoader(Subset(ds, idx_va), batch_size=256)


def run_adaptive_table(name, param_key, value_gen, base_cfg, base_epochs, ds, idx_tr, idx_va,
                       scale, device, is_epochs_param=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val, best_mae = None, float("inf")
    trials_since_best = 0
    n = 0
    stopped_by_decline = False
    for v in value_gen:
        n += 1
        cfg = {**base_cfg, param_key: v} if not is_epochs_param else base_cfg
        ep = v if is_epochs_param else base_epochs
        t0 = time.time()
        full, mae = run_trial(cfg, ep, 0, ds, idx_tr, idx_va, scale, device)
        dt = time.time() - t0
        row = {"trial": n, param_key: v, "epochs": ep,
              **{k: cfg[k] for k in cfg if k != param_key},
              **{k: round(full[k], 6) for k in SCALAR_KEYS if k in full}, "seconds": round(dt, 1)}
        tuning.log_trial(f"ofat_headpose_adaptive_{name}", {"trial": n, param_key: v, "epochs": ep, **cfg},
                         {k: row[k] for k in SCALAR_KEYS if k in row})
        marker = ""
        if mae < best_mae:
            best_val, best_mae = v, mae
            trials_since_best = 0
            marker = "  <- new best (lowest)"
        else:
            trials_since_best += 1
        print(f"  [{name}] trial {n}  {param_key}={v}  MAE={mae:.3f}deg  ({dt:.0f}s){marker}", flush=True)
        if n >= MIN_TRIALS and trials_since_best >= CONFIRM_AFTER_PEAK:
            stopped_by_decline = True
            break
        if n >= MAX_TRIALS:
            print(f"  [{name}] hit MAX_TRIALS={MAX_TRIALS} without a confirmed rise — "
                  f"stopping anyway; best so far stands as the winner.", flush=True)
            break
    print(f"  [{name}] BEST {param_key}={best_val}  MAE={best_mae:.3f}deg  "
          f"({n} trials, {'rise confirmed' if stopped_by_decline else 'safety cap hit'})\n", flush=True)
    new_cfg = dict(base_cfg)
    if not is_epochs_param:
        new_cfg[param_key] = best_val
    new_epochs = best_val if is_epochs_param else base_epochs
    return best_val, new_cfg, new_epochs


def run_categorical_table(name, param_key, choices, seeds_per_choice, base_cfg, base_epochs,
                          ds, idx_tr, idx_va, scale, device):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best = None
    n = 0
    for choice in choices:
        for seed in range(1, seeds_per_choice + 1):
            n += 1
            cfg = {**base_cfg, param_key: choice}
            t0 = time.time()
            full, mae = run_trial(cfg, base_epochs, seed, ds, idx_tr, idx_va, scale, device)
            dt = time.time() - t0
            row = {"trial": n, param_key: choice, "seed": seed, "epochs": base_epochs,
                  **{k: cfg[k] for k in cfg if k != param_key},
                  **{k: round(full[k], 6) for k in SCALAR_KEYS if k in full}, "seconds": round(dt, 1)}
            tuning.log_trial(f"ofat_headpose_adaptive_{name}", {"trial": n, param_key: choice, "seed": seed,
                                                                 "epochs": base_epochs, **cfg},
                             {k: row[k] for k in SCALAR_KEYS if k in row})
            print(f"  [{name}] trial {n}  {param_key}={choice} seed={seed}  MAE={mae:.3f}deg  ({dt:.0f}s)", flush=True)
            if best is None or mae < best[1]:
                best = (choice, mae)
    print(f"  [{name}] BEST {param_key}={best[0]}  MAE={best[1]:.3f}deg  "
          f"({n} trials — categorical, no peak/decline concept)\n", flush=True)
    new_cfg = {**base_cfg, param_key: best[0]}
    return best[0], new_cfg, base_epochs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()

    tuning.cpu_threads()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    crops, tgts, ids = train_headpose._load_pairs(args.data, args.limit, tag="300w_lp")
    ds = train_headpose.HeadPoseSet(crops, tgts, ids)
    idx_tr, idx_va, _ = tuning.group_split(ds.ids, (0.85, 0.15, 0.0), seed=0)
    scale = np.std(ds.targets, axis=0).clip(min=1.0)
    print(f"300W-LP {len(ds)} imgs / {len(set(ds.ids))} faces -> "
          f"train {len(idx_tr)} | val {len(idx_va)} (face-disjoint) | base_epochs={args.epochs}\n"
          f"Adaptive tables: min {MIN_TRIALS} trials, stop {CONFIRM_AFTER_PEAK} trials after a "
          f"confirmed rise, hard cap {MAX_TRIALS}.\n", flush=True)

    cfg = dict(START)
    epochs = args.epochs

    _, cfg, epochs = run_adaptive_table("P1_learning_rate", "lr", gen_lr(), cfg, epochs, ds, idx_tr, idx_va, scale, device)
    _, cfg, epochs = run_categorical_table("P2_optimizer", "opt", ["adam", "adamw", "sgd"], 4,
                                           cfg, epochs, ds, idx_tr, idx_va, scale, device)
    _, cfg, epochs = run_adaptive_table("P3_dropout", "dropout", gen_dropout(), cfg, epochs, ds, idx_tr, idx_va, scale, device)
    _, cfg, epochs = run_categorical_table("P4_augmentation", "aug", [0, 1], 5,
                                           cfg, epochs, ds, idx_tr, idx_va, scale, device)
    _, cfg, epochs = run_adaptive_table("P5_batch_size", "batch", gen_batch(), cfg, epochs, ds, idx_tr, idx_va, scale, device)
    _, cfg, epochs = run_categorical_table("P6_lr_schedule", "sched", ["none", "step", "cosine"], 4,
                                           cfg, epochs, ds, idx_tr, idx_va, scale, device)
    _, cfg, epochs = run_adaptive_table("P7_weight_decay", "weight_decay", gen_weight_decay(), cfg, epochs,
                                        ds, idx_tr, idx_va, scale, device)
    _, cfg, epochs = run_adaptive_table("P8_epochs", "epochs", gen_epochs(), cfg, epochs, ds, idx_tr, idx_va,
                                        scale, device, is_epochs_param=True)
    _, cfg, epochs = run_adaptive_table("P9_conv_width", "width", gen_width(), cfg, epochs, ds, idx_tr, idx_va, scale, device)
    _, cfg, epochs = run_adaptive_table("P10_fc_width", "fc", gen_fc(), cfg, epochs, ds, idx_tr, idx_va, scale, device)

    print("=" * 60)
    print("FINAL LOCKED-IN CONFIG:")
    print(json.dumps({**cfg, "epochs": epochs}, indent=2))
    (OUT_DIR / "final_config.json").write_text(json.dumps({**cfg, "epochs": epochs}, indent=2))
    print("OFAT ADAPTIVE SWEEP COMPLETE")


if __name__ == "__main__":
    main()
