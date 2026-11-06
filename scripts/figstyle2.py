"""Signature dark/luminous aesthetic for the 16 showcase figures. Real data, artful form.

Helpers: glow() bloomed scatter, gradient_fill() luminous area, ridge_glow(), chord ribbons,
and custom nebula/ember/ion colormaps. Entity colours: custom = ember, pretrained = ion-cyan.
"""
from __future__ import annotations
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path as MPath

BG      = "#080b12"          # deep space
PANEL   = "#0b0f18"
INK     = "#eef3fa"
INK2    = "#aab4c5"
MUTED   = "#6b7689"
GRID    = "#182231"

CUSTOM   = "#ff7a3c"         # ember (mine)
CUSTOM_HI= "#ffd7a8"
PRETRAIN = "#38d6ea"         # ion cyan (pretrained)
PRETRAIN_HI = "#c9f7ff"
GOLD     = "#ffd24a"

NEBULA = LinearSegmentedColormap.from_list("nebula",
    ["#080b12", "#241a4d", "#6c2a8f", "#c83a86", "#ff8a4c", "#ffe6a8", "#ffffff"])
EMBER = LinearSegmentedColormap.from_list("ember", ["#080b12", "#5a1a12", "#c1431f", "#ff7a3c", "#ffd7a8"])
ION   = LinearSegmentedColormap.from_list("ion",   ["#080b12", "#0d3b52", "#12849c", "#38d6ea", "#c9f7ff"])
# who-wins diverging: cyan (pretrained) <-> ember (custom)
DIVE  = LinearSegmentedColormap.from_list("dive", [PRETRAIN, "#1a2740", "#0b0f18", "#402518", CUSTOM])
CAT6  = ["#38d6ea", "#ff7a3c", "#7ce38a", "#ffd24a", "#e879c9", "#8f9bff"]

LABELS = {"custom": "Custom CNN (mine)", "pretrained": "Pretrained NN"}


def apply():
    mpl.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "figure.dpi": 130, "savefig.dpi": 200,
    })


def title(fig, main, sub=None):
    fig.text(0.5, 1.02, main, ha="center", fontsize=19, fontweight="bold", color=INK)
    if sub:
        fig.text(0.5, 0.975, sub, ha="center", fontsize=11, color=INK2)


def caption(fig, text):
    fig.text(0.5, -0.035, text, ha="center", fontsize=8.6, color=MUTED, style="italic")


def save(fig, path):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3, facecolor=BG)
    plt.close(fig)
    print("  wrote", path, flush=True)


def glow(ax, x, y, color, s=26, layers=((7, .05), (3.6, .10), (1.8, .22)), core=.95, cores="white", zorder=3):
    """Bloomed scatter: soft halos + bright core."""
    for m, a in layers:
        ax.scatter(x, y, s=s * m, c=color, alpha=a, linewidths=0, zorder=zorder)
    ax.scatter(x, y, s=s, c=color, alpha=core, linewidths=0, zorder=zorder + 1)


def gradient_fill(ax, x, y, cmap, alpha=1.0, glowline=None, y0=0.0):
    """Fill under curve (x,y) with a vertical gradient, luminous top edge."""
    x = np.asarray(x); y = np.asarray(y)
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    im = ax.imshow(grad, extent=[x.min(), x.max(), y0, max(y.max(), y0 + 1e-6)],
                   origin="lower", aspect="auto", cmap=cmap, alpha=alpha, zorder=1)
    verts = np.vstack([np.column_stack([x, y]), [x[-1], y0], [x[0], y0]])
    clip = Polygon(verts, closed=True, transform=ax.transData)
    ax.add_patch(clip); clip.set_facecolor("none"); clip.set_edgecolor("none")
    im.set_clip_path(clip)
    if glowline is not None:
        for lw, a in ((5, .12), (2.6, .3), (1.2, .9)):
            ax.plot(x, y, color=glowline, lw=lw, alpha=a, zorder=2)


def ribbon(ax, p0, p1, color, width, alpha=0.5, bow=0.0):
    """Glowing bezier band from arc point p0 to p1 curving through centre."""
    x0, y0 = float(p0[0]), float(p0[1]); x1, y1 = float(p1[0]), float(p1[1])
    ctrl = (bow * (x0 + x1) / 2, bow * (y0 + y1) / 2)
    n = np.array([-(y1 - y0), (x1 - x0)]); n = n / (np.linalg.norm(n) + 1e-9) * width
    for scale, a in ((1.0, alpha * .45), (0.6, alpha)):
        nx, ny = n[0] * scale, n[1] * scale
        verts = [(x0 + nx, y0 + ny), ctrl, (x1 + nx, y1 + ny),
                 (x1 - nx, y1 - ny), ctrl, (x0 - nx, y0 - ny), (x0 + nx, y0 + ny)]
        codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", alpha=a))
