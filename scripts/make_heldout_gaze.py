"""Restrict the matched gaze comparison to the custom model's HELD-OUT participants.

Why this exists
---------------
matched_gaze.npz scores 2,000 eyes from each of MPIIGaze's 15 participants. But the
custom CNN was trained participant-disjoint, so 9 of those 15 people are its own
TRAINING participants. Scoring the custom model over all 15 therefore blends
memorised identities with genuine generalisation and flatters it by ~1.9 deg -- the
same leakage class this project fixed inside the trainers.

This script recomputes the trainer's exact split (tuning.group_split over the cached
participant labels, seed 0) and writes matched_gaze_heldout.npz containing only the
TEST participants, so custom-vs-pretrained is measured on people the custom model
has never seen.

Note the pretrained baseline was itself trained on MPIIGaze and very likely saw these
participants, so its number here is a LOWER BOUND on its true held-out error. The
comparison is therefore generous to the pretrained model, not to ours.

Usage:  python scripts/make_heldout_gaze.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import tuning  # noqa: E402

PT = ROOT / "reports" / "tuning" / "pretrained"
CACHE = ROOT / "data" / "cache" / "mpiigaze.npz"


def held_out_participants(seed: int = 0) -> list[str]:
    """The TEST participants of the trainer's own group-disjoint split."""
    parts = np.load(CACHE)["parts"]
    _, _, te = tuning.group_split(parts, (0.7, 0.15, 0.15), seed=seed)
    return sorted({str(p) for p in parts[te]})


def angular_deg(true_deg: np.ndarray, pred_deg: np.ndarray) -> np.ndarray:
    a, b = np.radians(true_deg), np.radians(pred_deg)

    def vec(py):
        p, y = py[:, 0], py[:, 1]
        return np.stack([-np.cos(p) * np.sin(y), -np.sin(p), -np.cos(p) * np.cos(y)], 1)

    return np.degrees(np.arccos(np.clip((vec(a) * vec(b)).sum(1), -1, 1)))


def main() -> None:
    d = np.load(PT / "matched_gaze.npz")
    if "pid" not in d:
        raise SystemExit("matched_gaze.npz has no 'pid' -- re-run gen_matched_arrays.py gaze")

    pid = d["pid"].astype(str)
    test_ids = held_out_participants()
    mask = np.isin(pid, test_ids)
    if not mask.any():
        raise SystemExit(f"no rows for held-out participants {test_ids}")

    true, pc, pp = d["true"][mask], d["pred_custom"][mask], d["pred_pt"][mask]
    np.savez(PT / "matched_gaze_heldout.npz", true=true, pred_custom=pc, pred_pt=pp,
             axes=np.array(["pitch", "yaw"]), pid=pid[mask])

    ec_all = angular_deg(d["true"], d["pred_custom"]).mean()
    ep_all = angular_deg(d["true"], d["pred_pt"]).mean()
    ec, ep = angular_deg(true, pc).mean(), angular_deg(true, pp).mean()

    print(f"held-out participants : {test_ids}  (N={mask.sum()} of {len(pid)})")
    print(f"  all 15 participants : custom={ec_all:.3f}  pretrained={ep_all:.3f}  <- contaminated")
    print(f"  held-out only       : custom={ec:.3f}  pretrained={ep:.3f}  <- honest")
    print(f"  custom MAE held-out : {np.abs(true - pc).mean():.3f} deg")
    print(f"wrote {PT / 'matched_gaze_heldout.npz'}")


if __name__ == "__main__":
    main()
