"""Resume headpose adaptive sweep at P9 + P10 only, with epochs CAPPED at 6.

Context: the full ofat_headpose_adaptive.py locked P8 to 18 epochs. Carried
into the two capacity tables (P9 conv width, P10 fc width), that made single
trials take 1-4 HOURS at large widths — a single capacity table would have
run ~10 hours, contrary to the intent of the sweep (demonstrate the
overfitting trend in a handful of trials, not exhaustively grind every large
width). So P9 and P10 are re-run here with epochs capped at 6, matching what
the blink/gaze capacity tables effectively used. The capacity/overfitting
TREND is what these tables demonstrate, and that trend is visible fine at 6
epochs; this cap is documented transparently rather than hidden. P1-P8 keep
their fully-tuned winners (including epochs=18) unchanged.

Uses the P1-P8 winning config from the completed run:
  opt=adam, lr=0.00366442, weight_decay=6.103515625e-05, dropout=0.05,
  sched=cosine, aug=1, batch=46, epochs=18(P8 winner, but capped to 6 here).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import train_headpose  # noqa: E402
import tuning  # noqa: E402
import ofat_headpose_adaptive as ofa  # noqa: E402

# P1-P8 winners from the completed portion of the run.
BASE = dict(opt="adam", lr=0.00366442, weight_decay=6.103515625e-05, dropout=0.05,
            width=16, fc=128, sched="cosine", aug=1, batch=46)
CAPPED_EPOCHS = 6   # capacity tables only; P8's true winner was 18


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="300W_LP_DATASET")
    ap.add_argument("--limit", type=int, default=20000)
    args = ap.parse_args()

    tuning.cpu_threads()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    crops, tgts, ids = train_headpose._load_pairs(args.data, args.limit, tag="300w_lp")
    ds = train_headpose.HeadPoseSet(crops, tgts, ids)
    idx_tr, idx_va, _ = tuning.group_split(ds.ids, (0.85, 0.15, 0.0), seed=0)
    scale = np.std(ds.targets, axis=0).clip(min=1.0)
    print(f"RESUME P9+P10 (epochs capped to {CAPPED_EPOCHS}) — 300W-LP "
          f"train {len(idx_tr)} | val {len(idx_va)} (face-disjoint)\n"
          f"base config (P1-P8 winners): {BASE}\n", flush=True)

    cfg = dict(BASE)
    _, cfg, _ = ofa.run_adaptive_table("P9_conv_width", "width", ofa.gen_width(), cfg,
                                       CAPPED_EPOCHS, ds, idx_tr, idx_va, scale, device)
    _, cfg, _ = ofa.run_adaptive_table("P10_fc_width", "fc", ofa.gen_fc(), cfg,
                                       CAPPED_EPOCHS, ds, idx_tr, idx_va, scale, device)

    print("=" * 60)
    final = {**cfg, "epochs": 18}   # report the true P8 winner as the config's epoch count
    print("FINAL LOCKED-IN CONFIG (P9/P10 capacity found at capped 6 epochs; "
          "deployed epochs=18 from P8):")
    print(json.dumps(final, indent=2))
    ofa.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ofa.OUT_DIR / "final_config.json").write_text(json.dumps(final, indent=2))
    print("OFAT ADAPTIVE SWEEP COMPLETE")


if __name__ == "__main__":
    main()
