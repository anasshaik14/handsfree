"""One-factor-at-a-time (OFAT) hyperparameter sweep for the gaze CNN.

Same 8-parameter, 15-trial-per-table methodology as ofat_blink.py, adapted
for regression: the objective is angular error in degrees (LOWER is better),
not accuracy, and every trial records the full MPIIGaze regression metric
suite (MAE/RMSE/R2/correlation per axis, plus the true 3D angular error).

Usage:
  python scripts/ofat_gaze.py --data MPIIGaze_DATASET/Data/Normalized --limit 20000 --epochs 6
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
from torch.utils.data import Subset  # noqa: E402

import train_gaze  # noqa: E402
import metrics  # noqa: E402
import tuning  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "tuning" / "ofat_gaze"
SCALAR_KEYS = [
    "mae_deg", "rmse_deg", "median_ae_deg", "max_ae_deg", "p90_ae_deg", "p95_ae_deg",
    "mae_deg_pitch", "rmse_deg_pitch", "bias_deg_pitch", "std_deg_pitch", "r2_pitch", "corr_pitch",
    "mae_deg_yaw", "rmse_deg_yaw", "bias_deg_yaw", "std_deg_yaw", "r2_yaw", "corr_yaw",
    "angular_error_deg_mean", "angular_error_deg_median", "angular_error_deg_p90",
]

START = dict(opt="adam", lr=1e-3, weight_decay=0.0, dropout=0.3,
             width=16, fc=128, sched="none", aug=0, batch=256)


def lin(lo, hi, n):
    return [float(x) for x in np.linspace(lo, hi, n)]


def logspace(lo, hi, n):
    return [float(x) for x in np.exp(np.linspace(np.log(lo), np.log(hi), n))]


def run_trial(cfg: dict, epochs: int, seed: int, ds, idx_tr, idx_va, device):
    """Train one model from scratch, return (full_metrics_dict, angular_error_deg)."""
    model, _ = train_gaze.train_one(cfg, ds, idx_tr, idx_va, device, epochs,
                                     seed=seed, patience=max(3, epochs), verbose=False)
    ds.aug = False
    yt, yp, _ = train_gaze._angular(model, Subset(ds, idx_va), device)
    full = metrics.regression_metrics(np.degrees(yt), np.degrees(yp), ["pitch", "yaw"])
    full.update(metrics.gaze_angular_error_deg(yt, yp))
    return full, full["angular_error_deg_mean"]


def run_table(name: str, param_key: str, trials: list, base_cfg: dict, base_epochs: int,
             ds, idx_tr, idx_va, device):
    """trials: list of (cfg_overrides, seed, epochs_override) tuples. LOWER angular
    error wins. Returns (winning_value, updated_cfg, updated_epochs)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best = None
    for i, (overrides, seed, ep_override) in enumerate(trials):
        cfg = {**base_cfg, **overrides}
        ep = ep_override if ep_override is not None else base_epochs
        t0 = time.time()
        full, ang = run_trial(cfg, ep, seed, ds, idx_tr, idx_va, device)
        dt = time.time() - t0
        row = {"trial": i + 1, param_key: overrides.get(param_key, cfg.get(param_key)),
              "seed": seed, "epochs": ep, **{k: cfg[k] for k in cfg if k != param_key},
              **{k: round(full[k], 6) for k in SCALAR_KEYS if k in full},
              "seconds": round(dt, 1)}
        tuning.log_trial(f"ofat_gaze_{name}", {"trial": i + 1, param_key: row[param_key],
                                                "seed": seed, "epochs": ep, **cfg},
                         {k: row[k] for k in SCALAR_KEYS if k in row})
        print(f"  [{name}] trial {i + 1}/{len(trials)}  {param_key}={row[param_key]}"
              f"{' seed=' + str(seed) if seed else ''}  angular={ang:.3f}deg  ({dt:.0f}s)", flush=True)
        if best is None or ang < best[1]:
            best = (overrides, ang, ep)
    print(f"  [{name}] BEST {param_key}={best[0].get(param_key)}  angular={best[1]:.3f}deg\n", flush=True)
    new_cfg = {**base_cfg, **best[0]}
    new_epochs = best[2] if param_key == "epochs" else base_epochs
    return best[0].get(param_key), new_cfg, new_epochs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=6, help="base epoch count for P1-P7")
    args = ap.parse_args()

    tuning.cpu_threads()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = train_gaze.MPIIGaze(args.data, args.limit)
    idx_tr, idx_va, idx_te = tuning.group_split(ds.parts, (0.7, 0.15, 0.15), seed=0)
    print(f"{len(ds)} samples / {len(set(ds.parts))} participants -> "
          f"train {len(idx_tr)} | val {len(idx_va)} (participant-disjoint) | base_epochs={args.epochs}",
          flush=True)

    cfg = dict(START)
    epochs = args.epochs

    # P1: Learning Rate
    trials = [({"lr": v}, 0, None) for v in logspace(0.0001, 0.003, 15)]
    _, cfg, epochs = run_table("P1_learning_rate", "lr", trials, cfg, epochs, ds, idx_tr, idx_va, device)

    # P2: Optimizer — 3 choices x 5 seeds
    trials = [({"opt": o}, s, None) for o in ("adam", "adamw", "sgd") for s in range(1, 6)]
    _, cfg, epochs = run_table("P2_optimizer", "opt", trials, cfg, epochs, ds, idx_tr, idx_va, device)

    # P3: Dropout
    trials = [({"dropout": v}, 0, None) for v in lin(0.1, 0.5, 15)]
    _, cfg, epochs = run_table("P3_dropout", "dropout", trials, cfg, epochs, ds, idx_tr, idx_va, device)

    # P4: Augmentation — off x7, on x8, seeded
    trials = ([({"aug": 0}, s, None) for s in range(1, 8)] +
             [({"aug": 1}, s, None) for s in range(1, 9)])
    _, cfg, epochs = run_table("P4_augmentation", "aug", trials, cfg, epochs, ds, idx_tr, idx_va, device)

    # P5: Batch Size
    batch_vals = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 256]
    trials = [({"batch": v}, 0, None) for v in batch_vals]
    _, cfg, epochs = run_table("P5_batch_size", "batch", trials, cfg, epochs, ds, idx_tr, idx_va, device)

    # P6: LR Schedule — 3 choices x 5 seeds
    trials = [({"sched": s}, sd, None) for s in ("none", "step", "cosine") for sd in range(1, 6)]
    _, cfg, epochs = run_table("P6_lr_schedule", "sched", trials, cfg, epochs, ds, idx_tr, idx_va, device)

    # P7: Weight Decay
    trials = [({"weight_decay": v}, 0, None) for v in logspace(1e-6, 1e-3, 15)]
    _, cfg, epochs = run_table("P7_weight_decay", "weight_decay", trials, cfg, epochs, ds, idx_tr, idx_va, device)

    # P8: Epochs
    epoch_vals = list(range(2, 17))
    trials = [({"epochs": v}, 0, v) for v in epoch_vals]
    _, cfg, epochs = run_table("P8_epochs", "epochs", trials, cfg, epochs, ds, idx_tr, idx_va, device)

    print("=" * 60)
    print("FINAL LOCKED-IN CONFIG:")
    print(json.dumps({**cfg, "epochs": epochs}, indent=2))
    (OUT_DIR / "final_config.json").write_text(json.dumps({**cfg, "epochs": epochs}, indent=2))
    print("OFAT SWEEP COMPLETE")


if __name__ == "__main__":
    main()
