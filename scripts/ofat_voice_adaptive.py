"""Adaptive OFAT hyperparameter sweep for the voice keyword-spotting CNN —
10 parameters, FULL dataset (no subset cap — every training/validation wav
for the app's 6-keyword vocabulary: up, down, stop, go, yes, no).

Same method as ofat_blink_adaptive.py: ordered parameters (Learning Rate,
Dropout, Weight Decay, Batch Size, Epochs, Conv Width, FC Width) run
adaptively — minimum 10 trials, stop 3 trials after the best (highest)
accuracy is confirmed, hard cap 25. Categorical parameters (Optimizer, LR
Schedule, Augmentation) use a fixed minimum-10-trial seed-repeated count.

Usage:
  python scripts/ofat_voice_adaptive.py --root data/speech_commands --epochs 6
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train_voice  # noqa: E402
import metrics  # noqa: E402
import tuning  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "tuning" / "ofat_voice_adaptive"
SCALAR_KEYS = [
    "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted", "mcc", "cohen_kappa",
    "roc_auc_ovr_macro", "top5_accuracy", "log_loss",
]

START = dict(opt="adam", lr=1e-3, weight_decay=0.0, dropout=0.3,
             width=16, fc=64, sched="step", aug=0, batch=128)

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


def run_trial(cfg: dict, epochs: int, seed: int, train_base, val_dl, n_classes, device):
    model, _ = train_voice.train_one(cfg, train_base, val_dl, n_classes, device, epochs,
                                     seed=seed, patience=max(3, epochs), verbose=False)
    y_true, y_pred, y_prob = metrics.collect_classification(model, val_dl, device)
    full = metrics.classification_metrics(y_true, y_pred, y_prob)
    return full, full["accuracy"]


def run_adaptive_table(name, param_key, value_gen, base_cfg, base_epochs, train_base, val_dl,
                       n_classes, device, is_epochs_param=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val, best_acc = None, -1.0
    trials_since_peak = 0
    n = 0
    stopped_by_decline = False
    for v in value_gen:
        n += 1
        cfg = {**base_cfg, param_key: v} if not is_epochs_param else base_cfg
        ep = v if is_epochs_param else base_epochs
        t0 = time.time()
        full, acc = run_trial(cfg, ep, 0, train_base, val_dl, n_classes, device)
        dt = time.time() - t0
        row = {"trial": n, param_key: v, "epochs": ep,
              **{k: cfg[k] for k in cfg if k != param_key},
              **{k: round(full[k], 6) for k in SCALAR_KEYS if k in full}, "seconds": round(dt, 1)}
        tuning.log_trial(f"ofat_voice_adaptive_{name}", {"trial": n, param_key: v, "epochs": ep, **cfg},
                         {k: row[k] for k in SCALAR_KEYS if k in row})
        marker = ""
        if acc > best_acc:
            best_val, best_acc = v, acc
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


def run_categorical_table(name, param_key, choices, seeds_per_choice, base_cfg, base_epochs,
                          train_base, val_dl, n_classes, device):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best = None
    n = 0
    for choice in choices:
        for seed in range(1, seeds_per_choice + 1):
            n += 1
            cfg = {**base_cfg, param_key: choice}
            t0 = time.time()
            full, acc = run_trial(cfg, base_epochs, seed, train_base, val_dl, n_classes, device)
            dt = time.time() - t0
            row = {"trial": n, param_key: choice, "seed": seed, "epochs": base_epochs,
                  **{k: cfg[k] for k in cfg if k != param_key},
                  **{k: round(full[k], 6) for k in SCALAR_KEYS if k in full}, "seconds": round(dt, 1)}
            tuning.log_trial(f"ofat_voice_adaptive_{name}", {"trial": n, param_key: choice, "seed": seed,
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
    ap.add_argument("--root", type=str, default="data/speech_commands")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--keywords", type=str, default="up,down,stop,go,yes,no")
    args = ap.parse_args()

    tuning.cpu_threads()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class _Args:
        root = args.root
        keywords = args.keywords
        unknown = False
        limit = 0   # no cap — full dataset for every keyword class
        test = False
    train_base, val_base, _te, labels = train_voice._build_splits(_Args(), random.Random(0))
    val_dl = DataLoader(val_base, batch_size=256)
    n_classes = len(labels)
    print(f"FULL dataset -> train {len(train_base)} | val {len(val_base)} "
          f"({n_classes} classes: {labels}) | base_epochs={args.epochs}\n"
          f"Adaptive tables: min {MIN_TRIALS} trials, stop {CONFIRM_AFTER_PEAK} trials after a "
          f"confirmed decline, hard cap {MAX_TRIALS}.\n", flush=True)

    cfg = dict(START)
    epochs = args.epochs

    _, cfg, epochs = run_adaptive_table("P1_learning_rate", "lr", gen_lr(), cfg, epochs, train_base, val_dl, n_classes, device)
    _, cfg, epochs = run_categorical_table("P2_optimizer", "opt", ["adam", "adamw", "sgd"], 4,
                                           cfg, epochs, train_base, val_dl, n_classes, device)
    _, cfg, epochs = run_adaptive_table("P3_dropout", "dropout", gen_dropout(), cfg, epochs, train_base, val_dl, n_classes, device)
    _, cfg, epochs = run_categorical_table("P4_augmentation", "aug", [0, 1], 5,
                                           cfg, epochs, train_base, val_dl, n_classes, device)
    _, cfg, epochs = run_adaptive_table("P5_batch_size", "batch", gen_batch(), cfg, epochs, train_base, val_dl, n_classes, device)
    _, cfg, epochs = run_categorical_table("P6_lr_schedule", "sched", ["none", "step", "cosine"], 4,
                                           cfg, epochs, train_base, val_dl, n_classes, device)
    _, cfg, epochs = run_adaptive_table("P7_weight_decay", "weight_decay", gen_weight_decay(), cfg, epochs,
                                        train_base, val_dl, n_classes, device)
    _, cfg, epochs = run_adaptive_table("P8_epochs", "epochs", gen_epochs(), cfg, epochs, train_base, val_dl,
                                        n_classes, device, is_epochs_param=True)
    _, cfg, epochs = run_adaptive_table("P9_conv_width", "width", gen_width(), cfg, epochs, train_base, val_dl, n_classes, device)
    _, cfg, epochs = run_adaptive_table("P10_fc_width", "fc", gen_fc(), cfg, epochs, train_base, val_dl, n_classes, device)

    print("=" * 60)
    print("FINAL LOCKED-IN CONFIG:")
    print(json.dumps({**cfg, "epochs": epochs}, indent=2))
    (OUT_DIR / "final_config.json").write_text(json.dumps({**cfg, "epochs": epochs}, indent=2))
    print("OFAT ADAPTIVE SWEEP COMPLETE")


if __name__ == "__main__":
    main()
