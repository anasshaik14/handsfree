"""7 fascinating head-pose figures: custom CNN vs pretrained 6DRepNet, matched per-sample."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as F
F.apply_style()

OUT = ROOT / "reports" / "figures" / "headpose"
d = np.load(ROOT / "reports/tuning/pretrained/matched_headpose.npz")
true, pc, pp = d["true"], d["pred_custom"], d["pred_pt"]     # (N,3) yaw,pitch,roll deg
AX = ["yaw", "pitch", "roll"]
ac = np.abs(pc - true); ap = np.abs(pp - true)               # per-axis abs error
mae_c = ac.mean(1); mae_p = ap.mean(1)                       # per-sample mean-abs-error


def h1_raincloud():
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.6))
    for k, ax in enumerate(axes):
        for e, col, lab, side in [(pp[:, k] - true[:, k], F.PRETRAIN, "pretrained", 1),
                                  (pc[:, k] - true[:, k], F.CUSTOM, "custom", -1)]:
            xs = np.linspace(-25, 25, 200)
            kde = gaussian_kde(np.clip(e, -25, 25))(xs); kde /= kde.max()
            ax.fill_betweenx(xs, 0, side * kde, color=col, alpha=0.55, lw=0)
            ax.plot(side * kde, xs, color=col, lw=1.5)
            q = np.percentile(e, [25, 50, 75])
            ax.scatter([side * 0.12], [q[1]], color=col, s=28, zorder=5, edgecolor="white", linewidth=1)
        ax.axhline(0, color=F.MUTED, lw=0.8, ls=":")
        ax.set_title(AX[k], color=F.INK, fontsize=12)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-25, 25); ax.set_xticks([])
        if k == 0:
            ax.set_ylabel("signed error: pred − true (deg)")
    axes[0].plot([], [], color=F.CUSTOM, lw=6, label=F.LABELS["custom"])
    axes[0].plot([], [], color=F.PRETRAIN, lw=6, label=F.LABELS["pretrained"])
    axes[0].legend(loc="upper left", fontsize=9)
    F.title(fig, "Where each axis drifts — signed-error violins",
            "Distribution of prediction minus truth per rotation axis. Centred & narrow = unbiased & precise; dots mark the medians.")
    F.caption(fig, "AFLW2000, N=1,969, matched per-sample. Custom left of each spine, 6DRepNet right.")
    F.savefig(fig, OUT / "H1_signed_violins.png")


def h2_fingerprint():
    cols = ["MAE", "RMSE", "Median", "P90"]
    def row(err, k):
        e = err[:, k]
        return [e.mean(), np.sqrt((e ** 2).mean()), np.median(e), np.percentile(e, 90)]
    Mc = np.array([row(ac, k) for k in range(3)])
    Mp = np.array([row(ap, k) for k in range(3)])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    vmax = max(Mc.max(), Mp.max())
    for ax, M, col, lab in [(axes[0], Mc, F.CUSTOM, "custom"), (axes[1], Mp, F.PRETRAIN, "pretrained")]:
        im = ax.imshow(M, cmap=F.SEQ, vmin=0, vmax=vmax, aspect="auto")
        for i in range(3):
            for j in range(4):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=12,
                        color="white" if M[i, j] > vmax * 0.55 else F.INK2, fontweight="bold")
        ax.set_xticks(range(4)); ax.set_xticklabels(cols)
        ax.set_yticks(range(3)); ax.set_yticklabels(AX)
        ax.set_title(F.LABELS[lab], color=col); ax.grid(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02); cb.set_label("degrees (lower=better)", color=F.INK2)
    F.title(fig, "Error fingerprint — every metric, every axis",
            "A heat-grid of four error measures across the three rotation axes. Read the two panels together: the custom side runs consistently warmer.")
    F.caption(fig, "AFLW2000, N=1,969, matched per-sample. Shared colour scale.")
    F.savefig(fig, OUT / "H2_fingerprint.png")


def h3_heat():
    yb = np.linspace(-95, 95, 22); pb = np.linspace(-60, 60, 18)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    for ax, err, col, lab in [(axes[0], mae_c, F.CUSTOM, "custom"), (axes[1], mae_p, F.PRETRAIN, "pretrained")]:
        H = np.full((len(pb) - 1, len(yb) - 1), np.nan)
        for i in range(len(pb) - 1):
            for j in range(len(yb) - 1):
                m = ((true[:, 1] >= pb[i]) & (true[:, 1] < pb[i + 1]) &
                     (true[:, 0] >= yb[j]) & (true[:, 0] < yb[j + 1]))
                if m.sum() >= 3:
                    H[i, j] = err[m].mean()
        im = ax.pcolormesh(yb, pb, H, cmap=F.SEQ, vmin=0, vmax=18, shading="flat")
        ax.set_title(F.LABELS[lab], color=col); ax.set_xlabel("true yaw (deg)"); ax.set_ylabel("true pitch (deg)")
        ax.grid(False); ax.set_aspect("auto")
    cb = fig.colorbar(im, ax=axes, fraction=0.024, pad=0.02); cb.set_label("mean abs error (deg)", color=F.INK2)
    F.title(fig, "Error across the pose sphere — heatmap by head orientation",
            "Average error at each yaw×pitch region. Dark = hard poses. The custom model lights up at extreme angles the pretrained net handles.")
    F.caption(fig, "AFLW2000, N=1,969. Cells with <3 samples left blank.")
    F.savefig(fig, OUT / "H3_pose_heatmap.png")


def h4_skyline():
    delta = np.sort(mae_c - mae_p); x = np.linspace(0, 100, len(delta))
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.fill_between(x, delta, 0, where=delta <= 0, color=F.CUSTOM, alpha=0.85, lw=0)
    ax.fill_between(x, delta, 0, where=delta > 0, color=F.PRETRAIN, alpha=0.85, lw=0)
    ax.axhline(0, color=F.AXIS, lw=1)
    win = (mae_c < mae_p).mean() * 100
    ax.text(2, delta.min() * 0.7, f"Custom better\non {win:.0f}%", color=F.CUSTOM_D, fontsize=11, fontweight="bold", va="center")
    ax.text(98, delta.max() * 0.7, f"Pretrained better\non {100-win:.0f}%", color=F.PRETRAIN_D, fontsize=11, fontweight="bold", ha="right", va="center")
    ax.set_xlim(0, 100); ax.set_xlabel("images, sorted by head-to-head margin (%)")
    ax.set_ylabel("Δ mean-abs-error (custom − pretrained, deg)")
    F.title(fig, "Head-to-head skyline — who wins each face",
            "Every test face sorted by its error gap. Orange below = custom wins; blue above = 6DRepNet wins.")
    F.caption(fig, "AFLW2000, N=1,969, same crop scored by both models.")
    F.savefig(fig, OUT / "H4_win_skyline.png")


def h5_yaw_ridge():
    absyaw = np.abs(true[:, 0]); qs = np.linspace(0, 1, 7)
    edges = np.quantile(absyaw, qs)
    fig, ax = plt.subplots(figsize=(10.0, 6.4)); gap = 1.0
    for k in range(6):
        m = (absyaw >= edges[k]) & (absyaw <= edges[k + 1] if k == 5 else absyaw < edges[k + 1])
        base = k * gap
        for err, col, side in [(mae_p[m], F.PRETRAIN, +1), (mae_c[m], F.CUSTOM, -1)]:
            xs = np.linspace(0, 22, 160)
            kde = gaussian_kde(np.clip(err, 0, 22))(xs); kde = kde / kde.max() * 0.92
            ax.fill_betweenx(xs, base, base + side * kde, color=col, alpha=0.6, lw=0)
        ax.text(base, -1.7, f"{edges[k]:.0f}–{edges[k+1]:.0f}°", ha="center", color=F.MUTED, fontsize=9)
    ax.set_ylim(0, 22); ax.set_xlim(-1, 6 * gap)
    ax.set_ylabel("mean-abs-error (deg)"); ax.set_xlabel("head yaw band  |yaw|  (frontal → full profile)")
    ax.set_xticks([])
    ax.fill_between([], [], color=F.CUSTOM, label=F.LABELS["custom"])
    ax.fill_between([], [], color=F.PRETRAIN, label=F.LABELS["pretrained"])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=F.CUSTOM, label=F.LABELS["custom"]), Patch(color=F.PRETRAIN, label=F.LABELS["pretrained"])],
              loc="upper right")
    F.title(fig, "Error swells toward profile — the distribution, not just the mean",
            "Error spread in each yaw band as the head turns to profile. Custom on the left of each spine, 6DRepNet on the right; watch the custom tail fatten.")
    F.caption(fig, "AFLW2000, N=1,969. Equal-count |yaw| sextiles.")
    F.savefig(fig, OUT / "H5_yaw_ridgeline.png")


def h6_truepred():
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 9.0))
    for r, (pred, col, lab) in enumerate([(pc, F.CUSTOM, "custom"), (pp, F.PRETRAIN, "pretrained")]):
        for c in range(3):
            ax = axes[r, c]; t, p = true[:, c], pred[:, c]
            ax.hexbin(t, p, gridsize=38, cmap=F.SEQ, mincnt=1, linewidths=0)
            lim = [t.min(), t.max()]; ax.plot(lim, lim, color=F.CRIT, lw=1.3, ls="--")
            r2 = 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
            ax.set_title(f"{F.LABELS[lab]} — {AX[c]}", color=col, fontsize=10.5)
            ax.set_xlabel(f"true {AX[c]} (deg)"); ax.set_ylabel(f"pred {AX[c]} (deg)")
            ax.text(0.04, 0.94, f"R²={r2:.3f}", transform=ax.transAxes, va="top", fontsize=10.5, fontweight="bold")
            ax.set_aspect("equal")
    F.title(fig, "Predicted vs true angle — all three axes",
            "Density of predictions against ground truth; red dashed = perfect. R² summarises each axis' fit.")
    F.caption(fig, "AFLW2000, N=1,969, matched per-sample.")
    F.savefig(fig, OUT / "H6_true_vs_pred.png")


def h7_winmap():
    delta = mae_p - mae_c                       # >0 custom better
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    order = np.argsort(np.abs(delta))
    sc = ax.scatter(true[order, 0], true[order, 1], c=np.clip(delta[order], -12, 12),
                    cmap=F.DIV.reversed(), s=14 + np.abs(delta[order]) * 2.2, vmin=-12, vmax=12,
                    edgecolor="white", linewidth=0.25)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("margin (deg):  ← pretrained better (blue)   |   custom better (orange) →", color=F.INK2)
    ax.set_xlabel("true yaw (deg)"); ax.set_ylabel("true pitch (deg)")
    F.title(fig, "Who wins where — the pose battleground",
            "Each test face placed by its true head angle, coloured by which model wins and sized by the margin. Orange islands = custom's territory.")
    F.caption(fig, "AFLW2000, N=1,969. Colour = per-face error gap (custom − pretrained), clipped at ±12°.")
    F.savefig(fig, OUT / "H7_win_map.png")


if __name__ == "__main__":
    h1_raincloud(); h2_fingerprint(); h3_heat(); h4_skyline(); h5_yaw_ridge(); h6_truepred(); h7_winmap()
    print("headpose figures done")
