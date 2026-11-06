"""Negative control for the second-order-leakage detector.

The detector flags leakage when the per-group error ratio r(g) = e_audited/e_reference
varies across groups beyond sampling noise.  Head pose is the control case: both
models train on 300W-LP and are scored on AFLW2000, so NEITHER has seen the
evaluation groups and r(g) must be homogeneous.  Gaze is the positive case.

Heterogeneity is tested with Cochran's Q on bootstrap variances of r(g), and
summarised as I^2 (share of observed variance not attributable to noise).

Usage:  python scripts/control_detector.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PT = ROOT / "reports" / "tuning" / "pretrained"
RNG = np.random.default_rng(0)
B = 2000          # bootstrap resamples per group


def gaze_errors():
    def vec(a):
        p, y = np.radians(a[:, 0]), np.radians(a[:, 1])
        return np.stack([-np.cos(p)*np.sin(y), -np.sin(p), -np.cos(p)*np.cos(y)], 1)
    z = np.load(PT / "matched_gaze.npz", allow_pickle=True)
    T, C, P = z["true"], z["pred_custom"].astype(float), z["pred_pt"].astype(float)
    def ang(Q):
        u, v = vec(Q), vec(T)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return np.degrees(np.arccos(np.clip((u*v).sum(1), -1, 1)))
    return ang(C), ang(P), z["pid"]


def headpose_errors():
    z = np.load(PT / "matched_headpose.npz", allow_pickle=True)
    T, C, P = z["true"].astype(float), z["pred_custom"].astype(float), z["pred_pt"].astype(float)
    return np.abs(C - T).mean(1), np.abs(P - T).mean(1), T[:, 0]   # + true yaw for binning


def heterogeneity(eA, eB, groups, label):
    """Cochran's Q / I^2 on r(g), with bootstrap within-group variances."""
    G = [g for g in np.unique(groups) if (groups == g).sum() >= 30]
    r, var = [], []
    for g in G:
        m = groups == g
        a, b = eA[m], eB[m]
        r.append(a.mean() / b.mean())
        idx = RNG.integers(0, m.sum(), (B, m.sum()))
        var.append((a[idx].mean(1) / b[idx].mean(1)).var(ddof=1))
    r, var = np.array(r), np.array(var)
    w = 1.0 / var
    rbar = (w * r).sum() / w.sum()
    Q = (w * (r - rbar) ** 2).sum()
    df = len(G) - 1
    p = 1 - stats.chi2.cdf(Q, df)
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    print(f"\n{label}")
    print(f"  groups {len(G)}   r range {r.min():.2f}-{r.max():.2f}   pooled r {rbar:.3f}   CV {r.std(ddof=1)/r.mean():.3f}")
    print(f"  Cochran Q = {Q:.1f} on {df} df,  p = {p:.3g},  I^2 = {I2:.1f}%")
    return r, rbar, Q, p, I2


def main() -> None:
    print("=" * 68)
    print("POSITIVE CASE - gaze: audited model trained on 9 of 15 participants")
    eA, eB, pid = gaze_errors()
    heterogeneity(eA, eB, pid, "gaze, grouped by participant")

    print("\n" + "=" * 68)
    print("NEGATIVE CONTROL - head pose: neither model saw AFLW2000")
    eA, eB, yaw = headpose_errors()

    # (a) pose bins: intrinsic difficulty varies ~4x across these, so this is the
    #     strong test of whether the reference model really prices difficulty out.
    edges = np.array([-100, -60, -40, -25, -12, 0, 12, 25, 40, 60, 100])
    heterogeneity(eA, eB, np.digitize(yaw, edges), "head pose, grouped by yaw band")

    # (b) random groups: a partition with no structure at all.
    heterogeneity(eA, eB, RNG.integers(0, 15, len(eA)), "head pose, 15 random groups")

    # (c) control on the control: gaze grouped at random should also be homogeneous.
    eA, eB, pid = gaze_errors()
    heterogeneity(eA, eB, RNG.integers(0, 15, len(eA)), "gaze, 15 random groups (structure destroyed)")


if __name__ == "__main__":
    main()
