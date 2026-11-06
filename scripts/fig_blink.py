"""7 fascinating, LINE-FREE blink figures — deep per-sample audit of the custom blink CNN.

(The app's blink alternative is a classical Eye-Aspect-Ratio maths rule, not a neural net,
and can't be scored per-sample on 32x32 eye crops — so blink is a custom-model deep-dive.)
Forms: beeswarm, marimekko mosaic, calibration constellation, outcome streamgraph, confusion
river, score-separation density, decision-margin ridgeline. No line charts.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as F
F.apply_style()

OUT = ROOT / "reports" / "figures" / "blink"
d = np.load(ROOT / "reports/tuning/pretrained/matched_blink.npz", allow_pickle=True)
prob, label = d["prob"], d["label"]
score = prob[:, 1]                     # P(closed)
y = (label == 1).astype(int)           # 1 = closed (the click event)
OPENc, CLOSEDc = F.PRETRAIN, F.CUSTOM


def b1_beeswarm():
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    F.beeswarm(ax, score[y == 0], 1.0, OPENc)
    F.beeswarm(ax, score[y == 1], 0.0, CLOSEDc)
    for cls, yy, col in [(0, 1.0, OPENc), (1, 0.0, CLOSEDc)]:
        med = np.median(score[y == cls])
        ax.scatter([med], [yy], s=160, color="white", edgecolor=col, linewidth=2.6, zorder=6)
    ax.axvline(0.5, color=F.MUTED, lw=1, ls=":")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true: closed", "true: open"])
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("model output  P(closed)")
    F.title(fig, "Every eye as a dot — the separation swarm",
            "Each test eye placed by its P(closed) score; the two clouds pull apart to opposite ends. Ringed dots mark the class medians.")
    F.caption(fig, "MRL Eye Dataset, subject-disjoint test split, N=12,927 (swarm subsampled for clarity).")
    F.savefig(fig, OUT / "B1_swarm.png")


def b2_mosaic():
    yh = (score >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    x = 0.0; gap = 0.015
    for cls, col in [(0, OPENc), (1, CLOSEDc)]:
        m = y == cls; n = m.sum(); w = n / len(y)
        corr = (yh[m] == cls).mean()
        ax.add_patch(Rectangle((x, 0), w, corr, color=col, alpha=0.9, lw=0))
        ax.add_patch(Rectangle((x, corr), w, 1 - corr, facecolor=col, alpha=0.28, lw=0, hatch="///", edgecolor="white"))
        ax.text(x + w / 2, corr / 2, f"correct\n{corr*100:.1f}%", ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")
        if 1 - corr > 0.03:
            ax.text(x + w / 2, corr + (1 - corr) / 2, f"missed\n{(1-corr)*100:.1f}%", ha="center", va="center",
                    color=F.INK2, fontsize=10)
        ax.text(x + w / 2, -0.05, f"{'open' if cls==0 else 'closed'}\n(n={n:,})", ha="center", va="top", fontsize=11, color=col, fontweight="bold")
        x += w + gap
    ax.set_xlim(-0.01, 1.03); ax.set_ylim(-0.14, 1.02); ax.axis("off")
    ax.text(-0.008, 1.0, "100%", ha="right", fontsize=9, color=F.MUTED)
    ax.text(-0.008, 0.0, "0%", ha="right", fontsize=9, color=F.MUTED)
    F.title(fig, "Confusion as proportional territory — the mosaic",
            "Column width = how common each true class is; solid = correctly classified, hatched = missed. Area is faithful to the real sample counts.")
    F.caption(fig, "MRL Eye Dataset test split, N=12,927. Threshold 0.5.")
    F.savefig(fig, OUT / "B2_mosaic.png")


def b3_constellation():
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(score, bins) - 1, 0, 9)
    conf = np.array([score[idx == k].mean() if (idx == k).any() else np.nan for k in range(10)])
    acc = np.array([y[idx == k].mean() if (idx == k).any() else np.nan for k in range(10)])
    cnt = np.array([(idx == k).sum() for k in range(10)])
    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    sc = F.constellation(ax, conf, acc, cnt)
    ece = np.nansum(cnt / cnt.sum() * np.abs(acc - conf))
    ax.text(0.05, 0.9, f"ECE = {ece:.4f}", fontsize=13, fontweight="bold", color=F.CUSTOM_D)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("over- ↔ under-confidence (accuracy − confidence)", color=F.INK2)
    ax.set_xlabel("predicted confidence  P(closed)"); ax.set_ylabel("actual fraction closed")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    F.title(fig, "Is the confidence honest? — the calibration constellation",
            "Each bubble is a confidence bin: on the grey diagonal band = well-calibrated; bubble size = how many samples; colour = the direction of miscalibration.")
    F.caption(fig, "MRL Eye Dataset test split, N=12,927. 10 confidence bins.")
    F.savefig(fig, OUT / "B3_constellation.png")


def b4_streamgraph():
    ts = np.linspace(0, 1, 120)
    P, N = (y == 1).sum(), (y == 0).sum()
    TN, FP, FN, TP = [], [], [], []
    for t in ts:
        yh = score >= t
        TP.append((yh & (y == 1)).sum() / len(y)); FN.append((~yh & (y == 1)).sum() / len(y))
        FP.append((yh & (y == 0)).sum() / len(y)); TN.append((~yh & (y == 0)).sum() / len(y))
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ax.stackplot(ts, TP, TN, FP, FN,
                 colors=[CLOSEDc, OPENc, "#f6b48f", "#9ec5f4"],
                 labels=["true positive (closed✓)", "true negative (open✓)",
                         "false positive", "false negative"], alpha=0.9)
    ax.axvline(0.5, color=F.INK, lw=1.2, ls=":")
    ax.text(0.5, 1.02, "0.5", ha="center", color=F.INK2, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("decision threshold on P(closed)"); ax.set_ylabel("share of all test samples")
    ax.legend(loc="center left", fontsize=9, framealpha=0)
    F.title(fig, "The whole operating spectrum — outcome streamgraph",
            "As the click threshold slides 0→1, the bands show how the four outcomes trade off. The error bands (pale) stay thin across a wide middle range.")
    F.caption(fig, "MRL Eye Dataset test split, N=12,927.")
    F.savefig(fig, OUT / "B4_streamgraph.png")


def b5_river():
    yh = (score >= 0.5).astype(int)
    M = np.zeros((2, 2))
    for i in (0, 1):
        for j in (0, 1):
            M[i, j] = ((y == i) & (yh == j)).sum()
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    F.sankey_ribbons(ax, M, ["open", "closed"], ["open", "closed"], [OPENc, CLOSEDc])
    acc = (y == yh).mean()
    fig.text(0.5, 0.5, f"{acc*100:.1f}%\ncorrect", ha="center", va="center", fontsize=13, color=F.INK, fontweight="bold")
    F.title(fig, "Where predictions flow — confusion as a river",
            "Ribbons carry test samples from their true state (left) to the prediction (right). Thin cross-ribbons are the mistakes.")
    F.caption(fig, "MRL Eye Dataset test split, N=12,927. Threshold 0.5.")
    F.savefig(fig, OUT / "B5_confusion_flow.png")


def b6_separation():
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    xs = np.linspace(0, 1, 240)
    for cls, col, lab in [(0, OPENc, "true: open"), (1, CLOSEDc, "true: closed")]:
        s = score[y == cls]
        kde = gaussian_kde(np.clip(s, 0, 1))(xs)
        ax.fill_between(xs, kde, color=col, alpha=0.5, lw=0, label=lab)
    ax.axvline(0.5, color=F.MUTED, lw=1, ls=":")
    ax.set_xlabel("model output  P(closed)"); ax.set_ylabel("density"); ax.set_xlim(0, 1)
    ax.set_yticks([]); ax.legend(loc="upper center")
    F.title(fig, "Two clean peaks — how separable are open vs closed?",
            "Filled distribution of P(closed) for genuinely-open (blue) and genuinely-closed (orange) eyes. Little overlap = a decisive model.")
    F.caption(fig, "MRL Eye Dataset test split, N=12,927.")
    F.savefig(fig, OUT / "B6_score_separation.png")


def b7_margin_ridge():
    margin = np.abs(2 * score - 1)                     # 0 = unsure, 1 = certain
    correct = (score >= 0.5).astype(int) == y
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    xs = np.linspace(0, 1, 220)
    for m, col, lab in [(margin[correct], F.GOOD, "correct"), (margin[~correct], F.CRIT, "wrong")]:
        kde = gaussian_kde(np.clip(m, 0, 1))(xs); kde /= kde.max()
        ax.fill_between(xs, kde, color=col, alpha=0.4, lw=0, label=f"{lab} ({len(m):,})")
    ax.set_xlabel("decision certainty  |2·P(closed) − 1|   (0 = unsure, 1 = certain)")
    ax.set_ylabel("relative density"); ax.set_xlim(0, 1); ax.set_yticks([])
    ax.legend(loc="upper center")
    F.title(fig, "Does the model hesitate on its mistakes?",
            "Certainty of correct (green) vs wrong (red) predictions. Wrong answers lean toward low certainty — the model half-flags its own errors.")
    F.caption(fig, "MRL Eye Dataset test split, N=12,927.")
    F.savefig(fig, OUT / "B7_margin_ridge.png")


if __name__ == "__main__":
    b1_beeswarm(); b2_mosaic(); b3_constellation(); b4_streamgraph(); b5_river(); b6_separation(); b7_margin_ridge()
    print("blink figures done")
