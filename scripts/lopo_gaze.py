"""Leave-one-person-out evaluation of the custom gaze CNN on MPIIGaze.

The standard person-independent protocol for MPIIGaze is LOPO over all fifteen
participants; the single 70/15/15 group split used elsewhere in this project is
weaker.  This script runs the full protocol with the DEPLOYED hyperparameters
(models/custom/gaze_cnn.metrics.json) so the result is comparable to the shipped
model rather than to a fresh search.

For each held-out participant p:
    test  = every sample of p
    val   = 2 participants drawn from the remaining 14 (for early stopping)
    train = the other 12

Results are appended to reports/tuning/lopo_gaze.json after every fold, so the
run is resumable: re-invoking skips folds already present.

Usage:
    python scripts/lopo_gaze.py                     # all 15 folds, 80k cap
    python scripts/lopo_gaze.py --folds p05,p07     # just these
    python scripts/lopo_gaze.py --limit 0           # no subsample (slow)
    python scripts/lopo_gaze.py --epochs 20 --limit 80000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

import metrics  # noqa: E402
import tuning  # noqa: E402
from train_gaze import MPIIGaze, train_one, _angular  # noqa: E402

OUT = ROOT / "reports" / "tuning" / "lopo_gaze.json"
CFG_PATH = ROOT / "models" / "custom" / "gaze_cnn.metrics.json"


def load_results() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"protocol": "leave-one-person-out over MPIIGaze p00..p14 (normalized patch)",
            "folds": {}}


def save_results(res: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")


def summarise(res: dict) -> None:
    f = res["folds"]
    if not f:
        return
    ang = np.array([v["angular_error_deg_mean"] for v in f.values()])
    mae = np.array([v["mae_deg"] for v in f.values()])
    n = len(ang)
    se = ang.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    print("\n" + "=" * 62)
    print(f"LOPO over {n}/15 participants")
    for p in sorted(f):
        print(f"   {p}  angular {f[p]['angular_error_deg_mean']:6.3f} deg   "
              f"MAE {f[p]['mae_deg']:5.3f}   (train {f[p]['n_train']}, test {f[p]['n_test']})")
    print(f"  mean angular  {ang.mean():.3f} deg   sd {ang.std(ddof=1):.3f}   "
          f"se {se:.3f}   95% CI [{ang.mean()-1.96*se:.3f}, {ang.mean()+1.96*se:.3f}]")
    print(f"  mean MAE      {mae.mean():.3f} deg   sd {mae.std(ddof=1):.3f}")
    print(f"  range         {ang.min():.3f} - {ang.max():.3f} deg")
    print("=" * 62)
    res["summary"] = {"n_folds": n, "angular_mean": float(ang.mean()),
                      "angular_sd": float(ang.std(ddof=1)), "angular_se": float(se),
                      "angular_min": float(ang.min()), "angular_max": float(ang.max()),
                      "mae_mean": float(mae.mean()), "mae_sd": float(mae.std(ddof=1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--limit", type=int, default=80000,
                    help="cap on TRAIN samples per fold (0 = use all); matches the Section 4 budget")
    ap.add_argument("--val-limit", type=int, default=6000, dest="val_limit",
                    help="cap on VAL samples per fold; only feeds early stopping, so a few "
                         "thousand suffices and it is evaluated every epoch (0 = use all)")
    ap.add_argument("--folds", type=str, default="", help="comma-separated participants, default all")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=str, default="", help="only needed if the .npz cache is absent")
    args = ap.parse_args()

    tuning.cpu_threads()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))["hyperparameters"]
    print(f"device={device}  config={cfg}", flush=True)

    ds = MPIIGaze(args.data)
    parts = np.asarray(ds.parts)
    people = sorted(set(parts.tolist()))
    todo = args.folds.split(",") if args.folds else people

    res = load_results()
    res["config"] = cfg
    res["epochs"] = args.epochs
    res["train_limit"] = args.limit
    res["val_limit"] = args.val_limit

    for held in todo:
        if held in res["folds"]:
            print(f"[{held}] already done, skipping", flush=True)
            continue
        if held not in people:
            print(f"[{held}] not a participant, skipping", flush=True)
            continue

        rng = np.random.default_rng(args.seed)
        rest = [p for p in people if p != held]
        val_people = set(rng.choice(rest, size=2, replace=False).tolist())
        tr_people = [p for p in rest if p not in val_people]

        idx_te = np.nonzero(parts == held)[0]
        idx_va = np.nonzero(np.isin(parts, list(val_people)))[0]
        idx_tr = np.nonzero(np.isin(parts, tr_people))[0]
        if args.limit and len(idx_tr) > args.limit:      # even stride keeps all people
            idx_tr = idx_tr[np.linspace(0, len(idx_tr) - 1, args.limit).astype(int)]
        if args.val_limit and len(idx_va) > args.val_limit:
            idx_va = idx_va[np.linspace(0, len(idx_va) - 1, args.val_limit).astype(int)]

        t0 = time.time()
        print(f"\n[{held}] train {len(idx_tr)} / val {len(idx_va)} ({sorted(val_people)}) "
              f"/ test {len(idx_te)}", flush=True)
        model, val_ang = train_one(cfg, ds, idx_tr, idx_va, device, args.epochs,
                                   seed=args.seed, patience=args.patience, verbose=True)

        from torch.utils.data import Subset
        yt, yp, ang = _angular(model, Subset(ds, idx_te), device)
        m = metrics.regression_metrics(np.degrees(yt), np.degrees(yp), ["pitch", "yaw"])
        m.update(metrics.gaze_angular_error_deg(yt, yp))
        m.update({"n_train": int(len(idx_tr)), "n_val": int(len(idx_va)),
                  "n_test": int(len(idx_te)), "val_people": sorted(val_people),
                  "val_angular_deg": float(val_ang), "minutes": round((time.time()-t0)/60, 1)})
        res["folds"][held] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                              for k, v in m.items()}
        summarise(res)
        save_results(res)
        print(f"[{held}] done in {m['minutes']} min -> {OUT}", flush=True)

    summarise(res)
    save_results(res)


if __name__ == "__main__":
    main()
