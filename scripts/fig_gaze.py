"""7 fascinating gaze figures: custom CNN vs pretrained NN, matched per-sample."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as F
F.apply_style()

OUT = ROOT / "reports" / "figures" / "gaze"
# Held-out participants only: the full 15-participant matched file includes nine of
# the custom model's own TRAINING participants and flatters it by ~1.9 deg. Fall back
# to the full file only if the held-out subset has not been generated yet.
_heldout = ROOT / "reports/tuning/pretrained/matched_gaze_heldout.npz"
d = np.load(_heldout if _heldout.exists()
            else ROOT / "reports/tuning/pretrained/matched_gaze.npz")
true, pc, pp = d["true"], d["pred_custom"], d["pred_pt"]     # (N,2) pitch,yaw deg
ec = F.gaze_angular_err(true, pc)      # custom angular error
ep = F.gaze_angular_err(true, pp)      # pretrained angular error


def g1_sky():
    theta = np.arctan2(true[:, 1], true[:, 0])            # gaze direction
    ecc = np.sqrt(true[:, 0] ** 2 + true[:, 1] ** 2)
    nt, nr = 28, 7
    te = np.linspace(-np.pi, np.pi, nt + 1)
    re = np.linspace(0, np.quantile(ecc, 0.98), nr + 1)
    ti = np.clip(np.digitize(theta, te) - 1, 0, nt - 1)
    ri = np.clip(np.digitize(ecc, re) - 1, 0, nr - 1)
    fig = plt.figure(figsize=(13.2, 6.6))
    T, R = np.meshgrid(te, re)
    for s, (err, col, lab) in enumerate([(ec, F.CUSTOM, "custom"), (ep, F.PRETRAIN, "pretrained")]):
        ax = fig.add_subplot(1, 2, s + 1, projection="polar")
        ax.set_facecolor(F.SURFACE)
        G = np.full((nr, nt), np.nan)
        for a in range(nt):
            for b in range(nr):
                m = (ti == a) & (ri == b)
                if m.sum() >= 5:
                    G[b, a] = err[m].mean()
        pc = ax.pcolormesh(T, R, G, cmap=F.SEQ, vmin=0, vmax=9, shading="flat")
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_title(F.LABELS[lab], color=col, pad=18)
        ax.grid(color=F.GRID, alpha=0.4); ax.tick_params(colors=F.MUTED); ax.set_yticklabels([])
    cb = fig.colorbar(pc, ax=fig.axes, fraction=0.022, pad=0.04)
    cb.set_label("mean angular error (deg)", color=F.INK2)
    F.title(fig, "The error sky — a polar heatmap of gaze mistakes",
            "Angle = gaze direction, radius = how far off-centre; colour = error there. The custom map glows brighter, especially at the rim.")
    F.caption(fig, "MPIIGaze normalized eyes, N=30,000, matched per-sample. Cells with <5 samples blank.")
    F.savefig(fig, OUT / "G1_error_sky.png")


def g2_field():
    yb = np.linspace(-1.05, 1.05, 15) * 28
    pb = np.linspace(-1.05, 1.05, 13) * 22
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8))
    for ax, pred, col, lab in [(axes[0], pc, F.CUSTOM, F.LABELS["custom"]),
                               (axes[1], pp, F.PRETRAIN, F.LABELS["pretrained"])]:
        err = np.degrees(np.arccos(np.clip((F.gaze_vec(true) * F.gaze_vec(pred)).sum(1), -1, 1)))
        X, Y, U, V, C = [], [], [], [], []
        for i in range(len(pb) - 1):
            for j in range(len(yb) - 1):
                m = ((true[:, 0] >= pb[i]) & (true[:, 0] < pb[i + 1]) &
                     (true[:, 1] >= yb[j]) & (true[:, 1] < yb[j + 1]))
                if m.sum() < 8:
                    continue
                X.append((yb[j] + yb[j + 1]) / 2); Y.append((pb[i] + pb[i + 1]) / 2)
                U.append((pred[m, 1] - true[m, 1]).mean()); V.append((pred[m, 0] - true[m, 0]).mean())
                C.append(err[m].mean())
        q = ax.quiver(X, Y, U, V, C, cmap=F.SEQ, angles="xy", scale_units="xy", scale=1,
                      width=0.006, clim=(0, 9))
        ax.set_title(lab, color=col)
        ax.set_xlabel("gaze yaw (deg)"); ax.set_ylabel("gaze pitch (deg)")
        ax.set_xlim(-32, 32); ax.set_ylim(-26, 26); ax.set_aspect("equal")
    cb = fig.colorbar(q, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("mean angular error (deg)", color=F.INK2)
    F.title(fig, "Gaze displacement field — the direction of each model's mistake",
            "Arrows point from true gaze to the average prediction in each region; colour is error size. Longer arrows = stronger systematic bias.")
    F.caption(fig, "MPIIGaze, N=30,000. Grid cells with <8 samples omitted.")
    F.savefig(fig, OUT / "G2_displacement_field.png")


def g3_sunburst():
    from matplotlib.patches import Wedge
    bands = [(0, 5), (5, 10), (10, 15), (15, 999)]
    shades = ["#0d366b", "#256abf", "#6da7ec", "#cde2fb"]
    labels = ["≤5°", "5–10°", "10–15°", ">15°"]
    fig, ax = plt.subplots(figsize=(8.6, 7.6)); ax.set_aspect("equal"); ax.axis("off")
    rings = [("pretrained", ep, 0.32, 0.62), ("custom", ec, 0.66, 0.98)]
    for name, err, r0, r1 in rings:
        start = 90.0
        for (lo, hi), sh in zip(bands, shades):
            frac = ((err >= lo) & (err < hi)).mean()
            ext = frac * 360
            ax.add_patch(Wedge((0, 0), r1, start - ext, start, width=r1 - r0, facecolor=sh,
                               edgecolor="white", linewidth=2))
            if frac > 0.04:
                mid = np.radians(start - ext / 2); rm = (r0 + r1) / 2
                ax.text(rm * np.cos(mid), rm * np.sin(mid), f"{frac*100:.0f}%", ha="center", va="center",
                        color="white" if sh in shades[:2] else F.INK2, fontsize=10, fontweight="bold")
            start -= ext
        ax.text(0, (r0 + r1) / 2 * 0 + (r1 + 0.03), "", ha="center")
    ax.text(0, 0.5 * (0.32 + 0.62), "", ha="center")
    ax.text(0, 1.06, F.LABELS["custom"] + "  (outer ring)", ha="center", color=F.CUSTOM_D, fontsize=11, fontweight="bold")
    ax.text(0, -1.08, F.LABELS["pretrained"] + "  (inner ring)", ha="center", color=F.PRETRAIN_D, fontsize=11, fontweight="bold")
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.2, 1.18)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=s, label=l) for s, l in zip(shades, labels)],
              loc="center", bbox_to_anchor=(0.5, 0.5), ncol=1, fontsize=9, title="within")
    F.title(fig, "Accuracy sunburst — how the errors split by tolerance",
            "Each ring is one model; the coloured arcs show what share of predictions land within 5°, 10°, 15° or beyond. More dark-blue = more precise.")
    F.caption(fig, "MPIIGaze, N=30,000, matched per-sample. Arc length ∝ fraction of samples.")
    F.savefig(fig, OUT / "G3_tolerance_sunburst.png")


def g4_skyline():
    delta = ec - ep                                       # <0 custom better
    s = np.sort(delta)
    x = np.linspace(0, 100, len(s))
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.fill_between(x, s, 0, where=s <= 0, color=F.CUSTOM, alpha=0.85, lw=0)
    ax.fill_between(x, s, 0, where=s > 0, color=F.PRETRAIN, alpha=0.85, lw=0)
    ax.axhline(0, color=F.AXIS, lw=1)
    win = (delta < 0).mean() * 100
    ax.text(2, s.min() * 0.72, f"Custom better\non {win:.0f}% of samples", color=F.CUSTOM_D, fontsize=11, fontweight="bold", va="center")
    ax.text(98, s.max() * 0.72, f"Pretrained better\non {100-win:.0f}%", color=F.PRETRAIN_D, fontsize=11, fontweight="bold", ha="right", va="center")
    ax.set_xlim(0, 100); ax.set_xlabel("samples, sorted by head-to-head margin (%)")
    ax.set_ylabel("Δ angular error  (custom − pretrained, deg)")
    F.title(fig, "Head-to-head skyline — who wins each sample",
            "Every sample sorted by the per-sample error gap. Orange below = custom wins; blue above = pretrained wins; area shows by how much.")
    F.caption(fig, "MPIIGaze, N=30,000, same eye scored by both models.")
    F.savefig(fig, OUT / "G4_win_skyline.png")


def g5_bland():
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 9.2))
    axnames = ["pitch", "yaw"]
    for r, (pred, col, lab) in enumerate([(pc, F.CUSTOM, "custom"), (pp, F.PRETRAIN, "pretrained")]):
        for c in range(2):
            ax = axes[r, c]
            mean = (true[:, c] + pred[:, c]) / 2
            diff = pred[:, c] - true[:, c]
            hb = ax.hexbin(mean, diff, gridsize=45, cmap=F.SEQ, mincnt=1, linewidths=0)
            bias, sd = diff.mean(), diff.std()
            ax.axhline(bias, color=col, lw=2)
            ax.axhline(bias + 1.96 * sd, color=col, lw=1, ls="--")
            ax.axhline(bias - 1.96 * sd, color=col, lw=1, ls="--")
            ax.axhline(0, color=F.MUTED, lw=0.8, ls=":")
            ax.set_title(f"{F.LABELS[lab]} — {axnames[c]}", color=col, fontsize=11)
            ax.set_xlabel(f"mean of true & pred {axnames[c]} (deg)")
            ax.set_ylabel("pred − true (deg)")
            ax.text(0.02, 0.95, f"bias {bias:+.2f}°\n±1.96σ {1.96*sd:.2f}°", transform=ax.transAxes,
                    va="top", fontsize=9, color=F.INK2)
    F.title(fig, "Bland–Altman agreement — bias and spread per axis",
            "Prediction minus truth against their mean. A flat band on zero = unbiased; narrow band = precise. Dashed lines are 95% limits of agreement.")
    F.caption(fig, "MPIIGaze, N=30,000. Density shaded (darker = more samples).")
    F.savefig(fig, OUT / "G5_bland_altman.png")


def g6_ridgeline():
    ecc = np.sqrt(true[:, 0] ** 2 + true[:, 1] ** 2)
    qs = np.quantile(ecc, np.linspace(0, 1, 7))
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    gap = 1.0
    for k in range(6):
        m = (ecc >= qs[k]) & (ecc < qs[k + 1] if k < 5 else ecc <= qs[k + 1])
        base = k * gap
        for e, col, side in [(ep[m], F.PRETRAIN, +1), (ec[m], F.CUSTOM, -1)]:
            xs = np.linspace(0, 20, 160)
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(np.clip(e, 0, 20))(xs)
            kde = kde / kde.max() * 0.9
            ax.fill_betweenx(xs, base, base + side * kde, color=col, alpha=0.6, lw=0)
        ax.text(base, -1.4, f"{qs[k]:.0f}–{qs[k+1]:.0f}°", ha="center", color=F.MUTED, fontsize=9)
    ax.set_ylim(0, 20); ax.set_xlim(-1, 6 * gap)
    ax.set_ylabel("angular error (deg)")
    ax.set_xlabel("gaze eccentricity band (distance from screen centre)")
    ax.set_xticks([])
    ax.plot([], [], color=F.CUSTOM, lw=6, label=F.LABELS["custom"])
    ax.plot([], [], color=F.PRETRAIN, lw=6, label=F.LABELS["pretrained"])
    ax.legend(loc="upper right")
    F.title(fig, "Does error grow toward the screen edges?",
            "Error distribution (violin) split by how far the gaze is off-centre. Custom on the left of each band, pretrained on the right.")
    F.caption(fig, "MPIIGaze, N=30,000. Bands are equal-count eccentricity sextiles.")
    F.savefig(fig, OUT / "G6_eccentricity_ridgeline.png")


def g7_truepred():
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.4))
    names = ["pitch", "yaw"]
    for r, (pred, col, lab) in enumerate([(pc, F.CUSTOM, "custom"), (pp, F.PRETRAIN, "pretrained")]):
        for c in range(2):
            ax = axes[r, c]
            t, p = true[:, c], pred[:, c]
            ax.hexbin(t, p, gridsize=45, cmap=F.SEQ, mincnt=1, linewidths=0)
            lim = [min(t.min(), p.min()), max(t.max(), p.max())]
            ax.plot(lim, lim, color=F.CRIT, lw=1.4, ls="--")
            r2 = 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
            ax.set_title(f"{F.LABELS[lab]} — {names[c]}", color=col, fontsize=11)
            ax.set_xlabel(f"true {names[c]} (deg)"); ax.set_ylabel(f"predicted {names[c]} (deg)")
            ax.text(0.04, 0.94, f"R² = {r2:.3f}", transform=ax.transAxes, va="top",
                    fontsize=11, color=F.INK, fontweight="bold")
            ax.set_aspect("equal")
    F.title(fig, "Predicted vs true angle — density and fit",
            "Each panel: predictions against ground truth; the red dashed line is perfect. Tight cloud on the line = accurate; R² summarises the fit.")
    F.caption(fig, "MPIIGaze, N=30,000, matched per-sample.")
    F.savefig(fig, OUT / "G7_true_vs_pred.png")


if __name__ == "__main__":
    g1_sky(); g2_field(); g3_sunburst(); g4_skyline(); g5_bland(); g6_ridgeline(); g7_truepred()
    print("gaze figures done")
