"""Two schematic figures for the MAPLE manuscript.

  P0_framework.png          - Fig. 1, the five-phase framework overview
  P2_exposure_mechanism.png - Fig. 3, why asymmetric exposure biases a comparison

Both are drawn in the same plain black-on-white register as the rest of the
manuscript figures (no accent colours), so they print correctly in greyscale.

Layout note: the framework figure places everything in figure coordinates on a
single full-bleed axes, and every vertical offset is derived from FIG_H so that
a font size in points maps to a known fraction. Hard-coded fractions drift the
moment the figure size changes, which is how the first draft overflowed its
boxes.

Usage:  python scripts/fig_framework.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "figures" / "protocol"

INK = "#111111"
MID = "#5a5a5a"
PALE = "#dcdcdc"
FAINT = "#f4f4f4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 0.7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

FIG_W, FIG_H = 10.0, 7.6          # inches


def pt2frac(pt: float) -> float:
    """Vertical points -> axis fraction, for this figure height."""
    return (pt / 72.0) / FIG_H


# --------------------------------------------------------------------- helpers
def card(ax, x, y_top, w, title, lines, fill="#ffffff", lw=1.0,
         ts=9.0, bs=7.4, pad_pt=7.0, ec=INK):
    """Draw a titled card whose height is derived from its content."""
    lead = pt2frac(bs * 1.45)
    h = pt2frac(pad_pt) + pt2frac(ts * 1.25) + len(lines) * lead + pt2frac(pad_pt)
    ax.add_patch(FancyBboxPatch(
        (x, y_top - h), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        linewidth=lw, edgecolor=ec, facecolor=fill, zorder=2))
    yy = y_top - pt2frac(pad_pt)
    ax.text(x + w / 2, yy, title, ha="center", va="top", fontsize=ts,
            fontweight="bold", color=INK, zorder=3)
    yy -= pt2frac(ts * 1.25)
    for ln in lines:
        ax.text(x + w / 2, yy, ln, ha="center", va="top", fontsize=bs,
                color=INK, zorder=3)
        yy -= lead
    return h


def banner(ax, x, y_top, w, title, lines, fill=PALE, lw=1.1, ts=9.0, bs=7.4):
    return card(ax, x, y_top, w, title, lines, fill=fill, lw=lw, ts=ts, bs=bs)


def arrow(ax, x, y0, y1, lw=1.1, color=INK):
    ax.add_patch(FancyArrowPatch(
        (x, y0), (x, y1), arrowstyle="-|>", mutation_scale=9,
        linewidth=lw, color=color, shrinkA=0.5, shrinkB=0.5, zorder=4))


def phase(ax, x, y, n, text, strong=False):
    ax.text(x, y, f"PHASE {n}", fontsize=8.0, fontweight="bold",
            color=INK if strong else MID, ha="left", va="bottom")
    ax.text(x + 0.062, y, text, fontsize=8.0,
            color=INK if strong else MID, ha="left", va="bottom",
            fontweight="bold" if strong else "normal")


# ------------------------------------------------------------------- figure 1
def framework() -> Path:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    L, R = 0.035, 0.965
    span = R - L
    gap = 0.016
    w4 = (span - 3 * gap) / 4
    xs = [L + i * (w4 + gap) for i in range(4)]

    y = 0.975

    # ---- Phase 1 ---------------------------------------------------------
    phase(ax, L, y, 1, "  Group-structured corpus preparation")
    y -= pt2frac(13)
    corpora = [
        ("MRL Eye", ["84,898 eye crops", "key: subject (37)"]),
        ("MPIIGaze", ["427,316 patches", "key: participant (15)"]),
        ("300W-LP", ["~122,450 renders", "key: face identity (3,837)"]),
        ("Speech Commands v2", ["18,657 clips", "key: speaker (official)"]),
    ]
    h = 0.0
    for x, (t, ls) in zip(xs, corpora):
        h = card(ax, x, y, w4, t, ls, ts=8.6, bs=7.2)
    y -= h
    for x in xs:
        arrow(ax, x + w4 / 2, y, y - 0.022, lw=0.9)
    y -= 0.024

    h = banner(ax, L, y, span, "Group-disjoint partitioning  (Algorithm 1, constraint 35)",
               ["greedy longest-first deal on the grouping key   →   "
                "γ(T) ∩ γ(V) = γ(V) ∩ γ(E) = γ(T) ∩ γ(E) = ∅"],
               ts=9.2, bs=7.6)
    y -= h
    arrow(ax, 0.5, y, y - 0.026, lw=1.3)
    y -= 0.030

    # ---- Phase 2 ---------------------------------------------------------
    phase(ax, L, y, 2, "  Compact task-specific training")
    y -= pt2frac(13)
    nets = [
        ("Blink CNN", ["1×32×32 → 2 logits", "136,066 parameters"]),
        ("Gaze CNN", ["1×36×60 → 2 angles", "551,810 parameters"]),
        ("Head-pose CNN", ["3×64×64 → 3 angles", "1,142,339 parameters"]),
        ("Voice CNN", ["1×40×81 → 6 logits", "101,766 parameters"]),
    ]
    for x, (t, ls) in zip(xs, nets):
        h = card(ax, x, y, w4, t, ls, ts=8.6, bs=7.2)
    y -= h
    for x in xs:
        arrow(ax, x + w4 / 2, y, y - 0.020, lw=0.8, color=MID)
    y -= 0.022

    h = card(ax, L, y, span, "Pinned-baseline random search  (Algorithm 3)",
             ["trial 0 = deployed configuration   ·   selection on validation only   ·   "
              "test partition read exactly once   ·   seeded"],
             fill=FAINT, lw=0.9, ts=8.8, bs=7.3)
    y -= h
    arrow(ax, 0.5, y, y - 0.026, lw=1.3)
    y -= 0.030

    # ---- Phase 3 (the contribution) --------------------------------------
    phase(ax, L, y, 3, "  Exposure audit of the COMPARISON  —  the novel component",
          strong=True)
    y -= pt2frac(13)
    outer_top = y

    w3 = (span - 2 * gap - 0.030) / 3
    x3 = [L + 0.015 + i * (w3 + gap) for i in range(3)]
    inner_top = y - 0.014
    for x, (t, ls) in zip(x3, [
        ("Preconditions (Def. 2)",
         ["1  overlapping development corpora",
          "2  group structure, ρ > 0",
          "3  sample spans a partition boundary"]),
        ("GERA statistic (Algorithm 2)",
         ["r(g) = e_A(g) / e_R(g)",
          "intrinsic difficulty cancels (Prop. 2)",
          "AUC_r, Mann–Whitney p, Q, I²"]),
        ("Negative control",
         ["head pose: exposure impossible",
          "by construction (300W-LP → AFLW2000)",
          "I² must return ≈ 0"]),
    ]):
        h3 = card(ax, x, inner_top, w3, t, ls, fill=FAINT, lw=0.9, ts=8.4, bs=7.0)
    yb = inner_top - h3
    for x in x3:
        arrow(ax, x + w3 / 2, yb, yb - 0.018, lw=0.8)
    yb -= 0.020

    wh = (span - 0.030 - gap) / 2
    hb = card(ax, L + 0.015, yb, wh, "Bias decomposition (Theorem 3)",
              ["I = Δ_R · (c_A/c_R) + Φ,    ϕ = Φ / I"],
              fill=PALE, lw=0.9, ts=8.4, bs=7.4)
    card(ax, L + 0.015 + wh + gap, yb, wh, "Exposure-disjoint protocol (Table 5)",
         ["E* = { i : γ(i) ∉ T_A ∪ T_R }"],
         fill=PALE, lw=0.9, ts=8.4, bs=7.4)
    yb -= hb

    ax.add_patch(FancyBboxPatch(
        (L, yb - 0.012), span, outer_top - yb + 0.012,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        linewidth=1.7, edgecolor=INK, facecolor="none", zorder=5))
    y = yb - 0.012
    arrow(ax, 0.5, y, y - 0.026, lw=1.3)
    y -= 0.030

    # ---- Phase 4 ---------------------------------------------------------
    phase(ax, L, y, 4, "  Decision-rule-aware evaluation")
    y -= pt2frac(13)
    rules = [
        ("Blink → click", ["closure ≥ 420 ms (≈13 frames)", "stat: run integrity"]),
        ("Gaze → cursor", ["calibration + 1€ + deadzone", "stat: residual after affine"]),
        ("Head pose → gesture", ["30°/12°, 650 ms dwell", "stat: per-axis bias"]),
        ("Voice → command", ["argmax, optional reject", "stat: per-class precision"]),
    ]
    for x, (t, ls) in zip(xs, rules):
        h = card(ax, x, y, w4, t, ls, ts=8.6, bs=7.2)
    y -= h
    for x in xs:
        arrow(ax, x + w4 / 2, y, y - 0.020, lw=0.8, color=MID)
    y -= 0.022

    # ---- Phase 5 ---------------------------------------------------------
    banner(ax, L, y, span,
           "PHASE 5   Mechanistic attribution on the deployed TorchScript modules  (Algorithm 7)",
           ["Integrated Gradients, 4×4 region grid, completeness  Σ_k Attr(R_k) = F(x) − F(x′)",
            "→  signed midline differential (gaze, head pose)  ·  fissure contrast polarity (blink)  "
            "·  gated high-band detector (voice)"],
           ts=9.0, bs=7.4)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "P0_framework.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------- figure 3
def exposure_mechanism() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))

    # ---------- (a) the partition picture ---------------------------------
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(a)  where the two partitions differ", fontsize=9.0, pad=8)

    # outer: the corpus group population
    ax.add_patch(Rectangle((0.05, 0.24), 0.90, 0.52, linewidth=1.1,
                           edgecolor=INK, facecolor="#ffffff"))
    ax.text(0.50, 0.885, "corpus group population  G   (15 participants)",
            ha="center", fontsize=7.6, color=INK)

    # the reference model saw every group
    ax.add_patch(Rectangle((0.07, 0.28), 0.86, 0.44, linewidth=1.0,
                           edgecolor=MID, facecolor=PALE, alpha=0.6))
    ax.text(0.50, 0.690, "T_R  —  reference trained on ALL groups",
            ha="center", va="top", fontsize=7.2, color=INK)

    # the audited model saw only the left block
    ax.add_patch(Rectangle((0.09, 0.31), 0.42, 0.34, linewidth=1.3,
                           edgecolor=INK, facecolor="#bcbcbc"))
    ax.text(0.30, 0.605, "T_A\n9 groups seen", ha="center", va="top",
            fontsize=7.4, color=INK, fontweight="bold")
    ax.text(0.72, 0.605, "outside T_A\n6 groups unseen", ha="center", va="top",
            fontsize=7.4, color=INK)

    # evaluation sample straddles the boundary, drawn as a lower band
    ax.add_patch(Rectangle((0.14, 0.335), 0.75, 0.135, linewidth=1.5,
                           edgecolor=INK, facecolor="none", linestyle=(0, (4, 2))))
    ax.text(0.515, 0.400, "evaluation sample  E", ha="center", va="center",
            fontsize=7.4, color=INK, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none"))
    ax.text(0.50, 0.195, "E spans A's boundary  →  precondition 3 of Definition 2",
            ha="center", fontsize=7.2, color=INK)

    # mark the boundary itself
    ax.annotate("", xy=(0.51, 0.31), xytext=(0.51, 0.755),
                arrowprops=dict(arrowstyle="-", lw=1.4, color=INK, ls=(0, (2, 1.6))))
    ax.text(0.51, 0.790, "A's partition boundary", fontsize=6.9, color=INK,
            ha="center", va="bottom",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

    # ---------- (b) the quota mixture -------------------------------------
    ax = axes[1]
    ax.set_title("(b)  reported error depends on the quota", fontsize=9.0, pad=8)
    Pi = np.linspace(0, 1, 200)
    E_in, E_out = 3.684, 6.494
    E_A = Pi * E_in + (1 - Pi) * E_out
    E_R = np.full_like(Pi, 3.05)

    ax.plot(Pi, E_A, color=INK, lw=1.9, label="audited model A")
    ax.plot(Pi, E_R, color=MID, lw=1.4, ls="--", label="reference model R")
    ax.fill_between(Pi, E_A, E_R, color=PALE, alpha=0.65, zorder=0)

    for pi in (0.0, 0.6, 1.0):
        ax.plot([pi], [pi * E_in + (1 - pi) * E_out], "o", ms=4.8,
                mfc="white", mec=INK, mew=1.3, zorder=5)
    ax.annotate("6.494°\nall held-out", xy=(0.0, E_out), xytext=(0.06, 6.30),
                fontsize=7.0, color=INK, va="top")
    ax.annotate("4.589°\nmatched sampling", xy=(0.6, 0.6 * E_in + 0.4 * E_out),
                xytext=(0.30, 4.30), fontsize=7.0, color=INK, va="top")
    ax.annotate("3.684°\nall seen", xy=(1.0, E_in), xytext=(0.68, 3.55),
                fontsize=7.0, color=INK, va="top")

    ax.set_xlabel("exposed sampling mass  Π", fontsize=8.0)
    ax.set_ylabel("reported mean error  (°)", fontsize=8.0)
    ax.set_ylim(2.6, 7.1)
    ax.set_xlim(-0.02, 1.02)
    ax.tick_params(labelsize=7.2, length=2.5)
    ax.legend(fontsize=7.0, frameon=False, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---------- (c) the ratio cancels difficulty ---------------------------
    ax = axes[2]
    ax.set_title("(c)  the ratio r(g) removes difficulty", fontsize=9.0, pad=8)
    rng = np.random.default_rng(7)
    d_in = rng.uniform(0.8, 2.2, 9)
    d_out = rng.uniform(0.8, 2.2, 6)
    cA, phi = 1.77, 0.339
    eA_in, eR_in = d_in * cA * (1 - phi), d_in
    eA_out, eR_out = d_out * cA, d_out

    ax.scatter(eR_in, eA_in, s=24, facecolor="#bcbcbc", edgecolor=INK,
               linewidth=0.7, zorder=3, label="g ∈ T_A   (seen)")
    ax.scatter(eR_out, eA_out, s=26, marker="^", facecolor="white",
               edgecolor=INK, linewidth=1.0, zorder=3, label="g ∉ T_A   (unseen)")

    xs_ = np.linspace(0.6, 2.4, 50)
    ax.plot(xs_, xs_, color=MID, lw=0.9, ls=":", zorder=1)
    ax.plot(xs_, xs_ * cA * (1 - phi), color=INK, lw=1.2, zorder=2)
    ax.plot(xs_, xs_ * cA, color=INK, lw=1.2, ls="--", zorder=2)
    ax.text(2.42, 2.42 * cA * (1 - phi), "  slope κ(1−φ_A)", fontsize=6.8,
            color=INK, ha="left", va="center")
    ax.text(2.00, 2.00 * cA + 0.22, "slope κ", fontsize=6.8, color=INK, ha="center")
    ax.text(2.40, 2.40, "  parity", fontsize=6.6, color=MID, ha="left", va="center")

    ax.set_xlabel("reference model error  e_R(g)", fontsize=8.0)
    ax.set_ylabel("audited model error  e_A(g)", fontsize=8.0)
    ax.tick_params(labelsize=7.2, length=2.5)
    ax.legend(fontsize=6.9, frameon=False, loc="upper left")
    ax.set_xlim(0.55, 3.05)
    ax.set_ylim(0.6, 4.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout(pad=0.8, w_pad=2.0)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "P2_exposure_mechanism.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


if __name__ == "__main__":
    for f in (framework, exposure_mechanism):
        try:
            print("wrote", f())
        except Exception as exc:            # noqa: BLE001
            print(f"FAILED {f.__name__}: {exc}", file=sys.stderr)
            raise
