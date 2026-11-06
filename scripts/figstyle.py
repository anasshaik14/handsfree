"""Shared design system for the custom-vs-pretrained comparison figures.

Validated dataviz palette (light publication surface). Entity colours are STABLE:
custom = orange (my model), pretrained = blue (reference). Diverging = those two
poles with a neutral-grey midpoint; sequential magnitude = blue ramp.
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ---- palette (dataviz reference instance, light mode) -----------------------
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK2      = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
AXIS      = "#c3c2b7"

CUSTOM    = "#eb6834"   # orange — "my model"
CUSTOM_D  = "#c14f22"
PRETRAIN  = "#2a78d6"   # blue — reference
PRETRAIN_D= "#1c5cab"
GOOD      = "#0ca30c"
CRIT      = "#d03b3b"

# sequential blue ramp (near-zero -> dark)
SEQ = LinearSegmentedColormap.from_list(
    "seqblue", ["#eef5fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])
# diverging: custom-orange (neg / custom better) <-> grey <-> pretrained-blue (pos)
DIV = LinearSegmentedColormap.from_list(
    "divcp", [CUSTOM, "#f6b48f", "#f0efec", "#9ec5f4", PRETRAIN])

LABELS = {"custom": "Custom CNN (mine)", "pretrained": "Pretrained NN"}

# categorical hues (validated order) for multi-class (voice keywords etc.)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948"]


def sankey_ribbons(ax, M, left_labels, right_labels, colors, gap=0.12):
    """Bipartite flow: rows=true (left), cols=pred (right). Ribbon width ∝ M[i,j]."""
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    M = np.asarray(M, float); n = len(left_labels)
    tot = M.sum()
    row = M.sum(1); col = M.sum(0)
    unit = (1 - gap) / tot
    # node y-extents (top→down)
    def extents(sizes):
        y = 1.0; out = []
        pad = gap / max(n - 1, 1)
        for s in sizes:
            h = s * unit; out.append((y, y - h)); y -= h + pad
        return out
    le, re = extents(row), extents(col)
    lcur = [t for t, b in le]; rcur = [t for t, b in re]
    for i in range(n):
        for j in range(n):
            w = M[i, j] * unit
            if w <= 0:
                continue
            y0t, y0b = lcur[i], lcur[i] - w; lcur[i] -= w
            y1t, y1b = rcur[j], rcur[j] - w; rcur[j] -= w
            x0, x1 = 0.08, 0.92
            cx = (x0 + x1) / 2
            verts = [(x0, y0t), (cx, y0t), (cx, y1t), (x1, y1t),
                     (x1, y1b), (cx, y1b), (cx, y0b), (x0, y0b), (x0, y0t)]
            codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                     MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
            alpha = 0.72 if i == j else 0.32
            ax.add_patch(PathPatch(MPath(verts, codes), facecolor=colors[i], edgecolor="none", alpha=alpha))
    for (t, b), lab, c in zip(le, left_labels, colors):
        ax.add_patch(plt.Rectangle((0.045, b), 0.035, t - b, color=c, lw=0))
        ax.text(0.03, (t + b) / 2, lab, ha="right", va="center", fontsize=10, color=INK)
    for (t, b), lab, c in zip(re, right_labels, colors):
        ax.add_patch(plt.Rectangle((0.92, b), 0.035, t - b, color=c, lw=0))
        ax.text(0.97, (t + b) / 2, lab, ha="left", va="center", fontsize=10, color=INK)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.04); ax.axis("off")
    ax.text(0.06, 1.06, "TRUE", fontsize=9, color=MUTED, ha="center", fontweight="bold")
    ax.text(0.94, 1.06, "PREDICTED", fontsize=9, color=MUTED, ha="center", fontweight="bold")


def beeswarm(ax, values, y0, color, width=0.42, s=6, maxn=4500, seed=0):
    """Density-jittered strip (SHAP-style) at row y0. Returns nothing."""
    v = np.asarray(values)
    if len(v) > maxn:
        v = v[np.random.default_rng(seed).choice(len(v), maxn, replace=False)]
    order = np.argsort(v); v = v[order]
    # local density -> jitter amplitude
    hist, edges = np.histogram(v, bins=60)
    dens = hist / hist.max()
    bi = np.clip(np.digitize(v, edges) - 1, 0, len(dens) - 1)
    rng = np.random.default_rng(seed)
    jit = (rng.random(len(v)) - 0.5) * 2 * width * (0.25 + 0.75 * dens[bi])
    ax.scatter(v, y0 + jit, s=s, color=color, alpha=0.45, linewidths=0, zorder=3)


def constellation(ax, conf, acc, cnt):
    """Calibration bubbles: (confidence, accuracy) sized by count, coloured by signed gap."""
    ax.fill_between([0, 1], [-0.06, 0.94], [0.06, 1.06], color=GRID, alpha=0.6, zorder=0)  # ±band around diagonal
    m = cnt > 0
    gap = acc - conf
    sc = ax.scatter(conf[m], acc[m], s=40 + cnt[m] / cnt.max() * 900, c=gap[m],
                    cmap=DIV.reversed(), vmin=-0.15, vmax=0.15, edgecolor="white",
                    linewidth=1.2, zorder=4)
    return sc


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 130, "savefig.dpi": 200,
        "axes.titlesize": 13, "axes.titleweight": "bold",
        "figure.titlesize": 15, "figure.titleweight": "bold",
    })


def title(fig, main, sub=None):
    fig.suptitle(main, x=0.012, y=1.055, ha="left", fontsize=15, fontweight="bold")
    if sub:
        fig.text(0.012, 1.005, sub, ha="left", fontsize=10.3, color=INK2)


def caption(fig, text):
    fig.text(0.012, -0.035, text, ha="left", fontsize=8.6, color=MUTED, style="italic")


def savefig(fig, path):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print("  wrote", path, flush=True)


# ---- shared math ------------------------------------------------------------
def gaze_vec(deg):
    """(N,2) [pitch,yaw] deg -> (N,3) unit gaze vectors (ptgaze convention)."""
    p, y = np.radians(deg[:, 0]), np.radians(deg[:, 1])
    return np.stack([-np.cos(p) * np.sin(y), -np.sin(p), -np.cos(p) * np.cos(y)], 1)


def gaze_angular_err(true, pred):
    c = (gaze_vec(true) * gaze_vec(pred)).sum(1)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))
