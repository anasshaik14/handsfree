"""Resume voice adaptive sweep at P9 + P10 only, with epochs CAPPED at 6.

Same reasoning as ofat_headpose_resume_p9p10.py: the full voice sweep locked
P8 to 17 epochs, which made the two capacity tables (P9 conv width, P10 fc
width) run at 17 epochs each — with a single large-width trial hitting ~1
hour under contention, projecting to many hours for the two tables. The
capacity/overfitting TREND those tables demonstrate is visible fine at 6
epochs, so P9 and P10 are re-run here capped at 6 epochs, matching how the
headpose capacity tables were handled. P1-P8 keep their fully-tuned winners
(including epochs=17) unchanged; the deployed config's epoch count is still 17.

P1-P8 winners (from the completed portion):
  lr=0.01217139, opt=adam, dropout=0.3, aug=0, batch=16, sched=cosine,
  weight_decay=9.765625e-06, epochs=17.
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
            width=16, fc=64, sched="cosine", aug=0, batch=16)
CAPPED_EPOCHS = 6   # capacity tables only; P8's true winner was 17


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
    print(f"RESUME P9+P10 (epochs capped to {CAPPED_EPOCHS}) — FULL dataset "
          f"train {len(train_base)} | val {len(val_base)} ({n_classes} classes)\n"
          f"base config (P1-P8 winners): {BASE}\n", flush=True)

    cfg = dict(BASE)
    _, cfg, _ = ofa.run_adaptive_table("P9_conv_width", "width", ofa.gen_width(), cfg,
                                       CAPPED_EPOCHS, train_base, val_dl, n_classes, device)
    _, cfg, _ = ofa.run_adaptive_table("P10_fc_width", "fc", ofa.gen_fc(), cfg,
                                       CAPPED_EPOCHS, train_base, val_dl, n_classes, device)

    final = {**cfg, "epochs": 17}
    print("=" * 60)
    print("FINAL LOCKED-IN CONFIG (P9/P10 capacity found at capped 6 epochs; "
          "deployed epochs=17 from P8):")
    print(json.dumps(final, indent=2))
    ofa.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ofa.OUT_DIR / "final_config.json").write_text(json.dumps(final, indent=2))
    print("OFAT ADAPTIVE SWEEP COMPLETE")


if __name__ == "__main__":
    main()
