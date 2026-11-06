"""The leakage-detector figure.

(a) per-group error of the audited model against a reference model that has no
    differential familiarity across the groups;
(b) the ratio r(g) separated by training membership;
(c) negative control - head pose, where NEITHER model saw the evaluation groups,
    so r(g) must be homogeneous even though intrinsic difficulty varies ~4x.

Greyscale, house style.   Usage:  python scripts/fig_protocol.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PT = ROOT / "reports" / "tuning" / "pretrained"
TRAIN = {"p00","p01","p02","p03","p06","p09","p10","p11","p12"}


def ang(P, T):
    def vec(a):
        p, y = np.radians(a[:, 0]), np.radians(a[:, 1])
        return np.stack([-np.cos(p)*np.sin(y), -np.sin(p), -np.cos(p)*np.cos(y)], 1)
    u, v = vec(P), vec(T)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((u*v).sum(1), -1, 1)))


z = np.load(PT / "matched_gaze.npz", allow_pickle=True)
T, C, P, pid = z["true"], z["pred_custom"].astype(float), z["pred_pt"].astype(float), z["pid"]
eC, eP = ang(C, T), ang(P, T)
G = sorted(set(pid.tolist()))
a = np.array([eC[pid == g].mean() for g in G])
b = np.array([eP[pid == g].mean() for g in G])
seen = np.array([g in TRAIN for g in G])
r_gaze = a / b

zh = np.load(PT / "matched_headpose.npz", allow_pickle=True)
Th, Ch, Ph = zh["true"].astype(float), zh["pred_custom"].astype(float), zh["pred_pt"].astype(float)
hA, hB = np.abs(Ch - Th).mean(1), np.abs(Ph - Th).mean(1)
bins = np.digitize(Th[:, 0], np.array([-100, -60, -40, -25, -12, 0, 12, 25, 40, 60, 100]))
r_hp = np.array([hA[bins == k].mean() / hB[bins == k].mean()
                 for k in np.unique(bins) if (bins == k).sum() >= 30])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.8))

# ---------------------------------------------------------------- panel (a)
lim = [0, max(a.max(), b.max())*1.12]
ax1.plot(lim, lim, color="0.55", lw=1, ls=(0, (4, 3)), zorder=1)
ax1.scatter(b[seen], a[seen], s=88, facecolor="0.82", edgecolor="black", lw=1.2, zorder=3,
            label="in audited model's training partition")
ax1.scatter(b[~seen], a[~seen], s=100, facecolor="0.18", edgecolor="black", lw=1.2,
            marker="s", zorder=3, label="never trained on")
OFF = {"p04": (-6, 9), "p13": (7, 6), "p05": (8, -9), "p00": (-4, -12),
       "p01": (8, 3), "p02": (7, -9), "p09": (8, 2), "p11": (-6, -12)}
for g, x, y in zip(G, b, a):
    ax1.annotate(g, (x, y), textcoords="offset points", xytext=OFF.get(g, (8, -3)),
                 fontsize=7.2, color="0.25")
ax1.set_xlim(lim); ax1.set_ylim(lim)
ax1.set_xlabel("reference model, error per participant (deg)")
ax1.set_ylabel("audited model, error (deg)")
ax1.set_title("(a)  Reference calibrates group difficulty", fontsize=10.5, pad=9)
ax1.legend(frameon=False, fontsize=8, loc="upper left")

# ---------------------------------------------------------------- panel (b)
for i, (mask, mk, fc) in enumerate([(seen, "o", "0.82"), (~seen, "s", "0.18")]):
    x = np.full(mask.sum(), i + 1) + np.random.default_rng(0).normal(0, .045, mask.sum())
    ax2.scatter(x, r_gaze[mask], s=92, facecolor=fc, edgecolor="black", lw=1.2, marker=mk, zorder=3)
    ax2.plot([i + .78, i + 1.22], [r_gaze[mask].mean()]*2, color="black", lw=2.2, zorder=4)
ax2.axhline(1.0, color="0.55", lw=1, ls=(0, (4, 3)))
ax2.set_xticks([1, 2]); ax2.set_xticklabels(["in training", "not in training"])
ax2.set_xlim(.55, 2.45)
ax2.set_ylabel(r"error ratio  $r_g$")
ax2.set_title("(b)  Ratio separates membership (AUC 0.889, p = 0.006)", fontsize=10.5, pad=9)

# ---------------------------------------------------------------- panel (c)
rng = np.random.default_rng(1)
sets = [(r_gaze / r_gaze.mean(), "gaze,\nby participant\n(leakage present)", "0.18", "s"),
        (r_hp / r_hp.mean(), "head pose,\nby yaw band\n(no leakage)", "0.82", "o")]
for i, (v, lab, fc, mk) in enumerate(sets):
    x = np.full(len(v), i + 1) + rng.normal(0, .05, len(v))
    ax3.scatter(x, v, s=92, facecolor=fc, edgecolor="black", lw=1.2, marker=mk, zorder=3)
    ax3.plot([i + .74, i + 1.26], [v.min()]*2, color="black", lw=1, ls=":")
    ax3.plot([i + .74, i + 1.26], [v.max()]*2, color="black", lw=1, ls=":")
ax3.axhline(1.0, color="0.55", lw=1, ls=(0, (4, 3)))
ax3.set_xticks([1, 2]); ax3.set_xticklabels([s[1] for s in sets], fontsize=8.6)
ax3.set_xlim(.5, 2.5)
ax3.set_ylabel(r"$r_g$ / pooled $r$")
ax3.set_title(r"(c)  Negative control:  $I^2$ = 99.8% vs 0.0%", fontsize=10.5, pad=9)
ax3.annotate("spread 3.0x", (1, r_gaze.max()/r_gaze.mean()), textcoords="offset points",
             xytext=(16, -4), fontsize=8.4)
ax3.annotate("spread 1.2x", (2, r_hp.max()/r_hp.mean()), textcoords="offset points",
             xytext=(16, -4), fontsize=8.4)

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=.22, lw=.6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.suptitle("Detecting comparison-protocol leakage without access to the split",
             fontsize=13, fontweight="bold", y=.985)
fig.tight_layout(rect=(0, 0, 1, .945))
out = ROOT / "reports" / "figures" / "protocol" / "P1_leakage_detector.png"
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
print("  gaze  r: %.2f-%.2f (pooled %.3f)" % (r_gaze.min(), r_gaze.max(), r_gaze.mean()))
print("  hpose r: %.2f-%.2f (pooled %.3f)" % (r_hp.min(), r_hp.max(), r_hp.mean()))
