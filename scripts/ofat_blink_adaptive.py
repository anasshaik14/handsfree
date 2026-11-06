"""Adaptive OFAT hyperparameter sweep for the blink CNN — 10 parameters.

Unlike ofat_blink.py (fixed 15 trials/table), this version runs each
ORDERED parameter (one where "increasing the value" is a real direction —
Learning Rate, Dropout, Weight Decay, Batch Size, Epochs, Conv Width, FC
Width) until it finds a genuine peak-then-decline: minimum 10 trials, then
keeps going until 3 more trials in a row fail to beat the best score seen so
far (confirms it's a real decline, not one noisy trial), then stops. A hard
cap of 25 trials/table prevents a runaway search if no clean decline ever
appears — if the cap is hit, that's reported plainly, not faked.

CATEGORICAL parameters (Optimizer: adam/adamw/sgd, LR Schedule: none/step/
cosine, Augmentation: off/on) have no "increasing value" direction, so
peak-then-decline doesn't apply — "does SGD outperform Adam" isn't a curve
you walk along. These three tables use a fixed minimum-10-trial count
(seed-repeated per choice) instead, and this exemption is called out
explicitly rather than silently treating them the same as the ordered ones.

Usage:
  python scripts/ofat_blink_adaptive.py --data mrlEyes_2018_01_DATASET --limit 20000 --epochs 6
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

import train_blink  # noqa: E402
import metrics  # noqa: E402
import tuning  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "tuning" / "ofat_blink_adaptive"
SCALAR_KEYS = [
    "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted", "mcc", "cohen_kappa",
    "false_positive_rate", "false_negative_rate", "roc_auc", "pr_auc",
    "brier_score", "log_loss",
]

START = dict(opt="adam", lr=1e-3, weight_decay=0.0, dropout=0.3,
             width=16, fc=64, sched="none", aug=0, batch=128)

MIN_TRIALS = 10
CONFIRM_AFTER_PEAK = 3   # trials in a row worse than the peak before we call it "declined"
MAX_TRIALS = 25          # safety cap so a table that never declines can't run forever


# ---- expanding candidate generators for the 7 ORDERED parameters -----------
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


def run_trial(cfg: dict, epochs: int, seed: int, ds, idx_tr, idx_va, device):
    model, _ = train_blink.train_one(cfg, ds, idx_tr, idx_va, device, epochs,
                                     seed=seed, patience=max(3, epochs), verbose=False)
    ds.aug = False
    dl = DataLoader(Subset(ds, idx_va), batch_size=256)
    y_true, y_pred, y_prob = metrics.collect_classification(model, dl, device)
    full = metrics.classification_metrics(y_true, y_pred, y_prob)
    return full, full["accuracy"]


def run_adaptive_table(name: str, param_key: str, value_gen, base_cfg: dict, base_epochs: int,
                       ds, idx_tr, idx_va, device, is_epochs_param: bool = False):
    """Run trials with an ever-increasing parameter value until we see a peak
    followed by CONFIRM_AFTER_PEAK trials that all fail to beat it (min
    MIN_TRIALS, hard cap MAX_TRIALS). Returns (winning_value, updated_cfg,
    updated_epochs, stopped_by_decline: bool)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val, best_acc, best_full = None, -1.0, None
    trials_since_peak = 0
    n = 0
    stopped_by_decline = False
    for v in value_gen:
        n += 1
        cfg = {**base_cfg, param_key: v} if not is_epochs_param else base_cfg
        ep = v if is_epochs_param else base_epochs
        t0 = time.time()
        full, acc = run_trial(cfg, ep, 0, ds, idx_tr, idx_va, device)
        dt = time.time() - t0
        row = {"trial": n, param_key: v, "epochs": ep,
              **{k: cfg[k] for k in cfg if k != param_key},
              **{k: round(full[k], 6) for k in SCALAR_KEYS if k in full}, "seconds": round(dt, 1)}
        tuning.log_trial(f"ofat_blink_adaptive_{name}", {"trial": n, param_key: v, "epochs": ep, **cfg},
                         {k: row[k] for k in SCALAR_KEYS if k in row})
        marker = ""
        if acc > best_acc:
            best_val, best_acc, best_full = v, acc, full
            trials_since_peak = 0
            marker = "  <- new peak"
        else:
            trials_since_peak += 1
        print(f"  [{name}] trial {n}  {param_key}={v}  acc={acc:.4f}  ({dt:.0f}s){marker}", flush=True)
        if n >= MIN_TRIALS and trials_since_peak >= CONFIRM_AFTER_PEAK:
            stopped_by_decline = True
            break
        if n >= MAX_TRIALS:
            print(f"  [{name}] hit MAX_TRIALS={MAX_TRIALS} without a confirmed decline — "
                  f"stopping anyway; peak so far stands as the winner.", flush=True)
            break
    print(f"  [{name}] BEST {param_key}={best_val}  acc={best_acc:.4f}  "
          f"({n} trials, {'decline confirmed' if stopped_by_decline else 'safety cap hit'})\n", flush=True)
    new_cfg = dict(base_cfg)
    if not is_epochs_param:
        new_cfg[param_key] = best_val
    new_epochs = best_val if is_epochs_param else base_epochs
    return best_val, new_cfg, new_epochs


def run_categorical_table(name: str, param_key: str, choices: list, seeds_per_choice: int,
                          base_cfg: dict, base_epochs: int, ds, idx_tr, idx_va, device):
    """Fixed minimum-MIN_TRIALS trial count, seed-repeated per choice. No
    peak-then-decline logic — categorical choices have no ordering to walk."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best = None
    n = 0
    for choice in choices:
        for seed in range(1, seeds_per_choice + 1):
            n += 1
            cfg = {**base_cfg, param_key: choice}
            t0 = time.time()
            full, acc = run_trial(cfg, base_epochs, seed, ds, idx_tr, idx_va, device)
            dt = time.time() - t0
            row = {"trial": n, param_key: choice, "seed": seed, "epochs": base_epochs,
                  **{k: cfg[k] for k in cfg if k != param_key},
                  **{k: round(full[k], 6) for k in SCALAR_KEYS if k in full}, "seconds": round(dt, 1)}
            tuning.log_trial(f"ofat_blink_adaptive_{name}", {"trial": n, param_key: choice, "seed": seed,
                                                              "epochs": base_epochs, **cfg},
                             {k: row[k] for k in SCALAR_KEYS if k in row})
            print(f"  [{name}] trial {n}  {param_key}={choice} seed={seed}  acc={acc:.4f}  ({dt:.0f}s)", flush=True)
            if best is None or acc > best[1]:
                best = (choice, acc)
    print(f"  [{name}] BEST {param_key}={best[0]}  acc={best[1]:.4f}  "
          f"({n} trials — categorical, no peak/decline concept)\n", flush=True)
    new_cfg = {**base_cfg, param_key: best[0]}
    return best[0], new_cfg, base_epochs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=6, help="base epoch count for non-epoch tables")
    args = ap.parse_args()

    tuning.cpu_threads()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = train_blink.MRLEyes(args.data, args.limit)
    idx_tr, idx_va, idx_te = tuning.group_split(ds.subjects, (0.7, 0.15, 0.15), seed=0)
    print(f"{len(ds)} imgs / {len(set(ds.subjects))} subjects -> "
          f"train {len(idx_tr)} | val {len(idx_va)} (subject-disjoint) | base_epochs={args.epochs}\n"
          f"Adaptive tables: min {MIN_TRIALS} trials, stop {CONFIRM_AFTER_PEAK} trials after a "
          f"confirmed peak, hard cap {MAX_TRIALS}.\n", flush=True)

    cfg = dict(START)
    epochs = args.epochs

    _, cfg, epochs = run_adaptive_table("P1_learning_rate", "lr", gen_lr(), cfg, epochs, ds, idx_tr, idx_va, device)
    _, cfg, epochs = run_categorical_table("P2_optimizer", "opt", ["adam", "adamw", "sgd"], 4,
                                           cfg, epochs, ds, idx_tr, idx_va, device)
    _, cfg, epochs = run_adaptive_table("P3_dropout", "dropout", gen_dropout(), cfg, epochs, ds, idx_tr, idx_va, device)
    _, cfg, epochs = run_categorical_table("P4_augmentation", "aug", [0, 1], 5,
                                           cfg, epochs, ds, idx_tr, idx_va, device)
    _, cfg, epochs = run_adaptive_table("P5_batch_size", "batch", gen_batch(), cfg, epochs, ds, idx_tr, idx_va, device)
    _, cfg, epochs = run_categorical_table("P6_lr_schedule", "sched", ["none", "step", "cosine"], 4,
                                           cfg, epochs, ds, idx_tr, idx_va, device)
    _, cfg, epochs = run_adaptive_table("P7_weight_decay", "weight_decay", gen_weight_decay(), cfg, epochs,
                                        ds, idx_tr, idx_va, device)
    _, cfg, epochs = run_adaptive_table("P8_epochs", "epochs", gen_epochs(), cfg, epochs, ds, idx_tr, idx_va,
                                        device, is_epochs_param=True)
    _, cfg, epochs = run_adaptive_table("P9_conv_width", "width", gen_width(), cfg, epochs, ds, idx_tr, idx_va, device)
    _, cfg, epochs = run_adaptive_table("P10_fc_width", "fc", gen_fc(), cfg, epochs, ds, idx_tr, idx_va, device)

    print("=" * 60)
    print("FINAL LOCKED-IN CONFIG:")
    print(json.dumps({**cfg, "epochs": epochs}, indent=2))
    (OUT_DIR / "final_config.json").write_text(json.dumps({**cfg, "epochs": epochs}, indent=2))
    print("OFAT ADAPTIVE SWEEP COMPLETE")


if __name__ == "__main__":
    main()
