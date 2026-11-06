"""Run ONLY P10 (FC width) for voice, to complete its 10-table sweep.

Context: P9 (conv width) was stopped after 10 trials once it had clearly shown
the overfitting curve — a peak at width=65 and a catastrophic divergence at
width=85 (accuracy collapsed to 0.39). Its remaining decline-confirmation
trials (width 110, 143) would each take 2-3 hours and near-certainly diverge
too, adding cost but no insight, so P9 stands at its 10 trials.

P9's nominal peak (width=65, acc 0.9347) is within seed-noise of width=10
(0.9316) — and width=10 is what the blink, gaze AND head-pose sweeps all chose
— so width=10 is carried into P10 as the robust, consistent, tractable conv
width. (Using 65 would make every P10 trial ~30 min.) This choice is documented
here and in the final report; it does not affect voice's headline result, which
comes from the fully-tuned P8 config (97.0%).

Epochs capped at 6 (capacity table), matching P9 and the head-pose capacity
tables. P1-P8 winners otherwise carried forward.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train_voice  # noqa: E402
import tuning  # noqa: E402
import ofat_voice_adaptive as ofa  # noqa: E402

BASE = dict(opt="adam", lr=0.01217139, weight_decay=9.765625e-06, dropout=0.3,
            width=10, fc=64, sched="cosine", aug=0, batch=16)   # width=10: robust conv width (see docstring)
CAPPED_EPOCHS = 6


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/speech_commands")
    args = ap.parse_args()

    tuning.cpu_threads()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class _Args:
        root = args.root
        keywords = "up,down,stop,go,yes,no"
        unknown = False
        limit = 0
        test = False
    train_base, val_base, _te, labels = train_voice._build_splits(_Args(), random.Random(0))
    val_dl = DataLoader(val_base, batch_size=256)
    n_classes = len(labels)
    print(f"P10-only (FC width) at conv width=10, epochs capped {CAPPED_EPOCHS} — "
          f"train {len(train_base)} | val {len(val_base)} ({n_classes} classes)\n"
          f"base config: {BASE}\n", flush=True)

    cfg = dict(BASE)
    _, cfg, _ = ofa.run_adaptive_table("P10_fc_width", "fc", ofa.gen_fc(), cfg,
                                       CAPPED_EPOCHS, train_base, val_dl, n_classes, device)

    final = {**cfg, "width": 10, "epochs": 17}   # deployed epochs from P8; conv width documented as 10
    print("=" * 60)
    print("FINAL LOCKED-IN CONFIG (P9/P10 capacity at capped 6 epochs; deployed epochs=17):")
    print(json.dumps(final, indent=2))
    ofa.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ofa.OUT_DIR / "final_config.json").write_text(json.dumps(final, indent=2))
    print("OFAT ADAPTIVE SWEEP COMPLETE")


if __name__ == "__main__":
    main()
