"""Explainability figures, four per input (blink, gaze, head pose, voice) = 16 total.

Mirrors the four explainability views used in the reference manuscript, adapted from
tabular SHAP to networks that consume images and spectrograms. Tabular SHAP ranks named
columns; our inputs have none, so the input is divided into a grid of spatial regions
and each REGION plays the role of a feature.

Per input:
  A1  global summary   - regions ranked by mean |attribution|, swarm of per-sample values
  A2  waterfall        - two individual instances, base value -> prediction, region by region
  A3  similarity order - stacked force plot, instances ordered by explanation similarity
  A4  sample order     - stacked force plot, instances in dataset order

Attribution: Integrated Gradients (Sundararajan et al., 2017) against a zero baseline,
32 steps. IG is the Aumann-Shapley (continuous Shapley) attribution and satisfies
completeness, so region attributions sum exactly to f(x) - f(baseline); this is what
makes the waterfall and stacked force plots exact rather than indicative. Computed on
the deployed TorchScript modules, so what is explained is what runs.

All figures render greyscale for black-and-white reproduction.

Usage:  python scripts/fig_attribution.py [blink|gaze|headpose|voice|all]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import leaves_list, linkage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.ml import preprocess  # noqa: E402
import tuning  # noqa: E402

torch.set_num_threads(4)
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "reports" / "figures" / "attribution"
N_SUMMARY = 400          # samples for the global summary
N_FORCE = 120            # instances shown in the stacked force plots
IG_STEPS = 32
GRID = (4, 4)
TOP_K = 10

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.edgecolor": "0.2", "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# ------------------------------------------------------------------ attribution
def integrated_gradients(model, x, target_fn, steps=IG_STEPS):
    """IG against a zero baseline. x:(N,C,H,W) -> attribution of the same shape."""
    total = torch.zeros_like(x)
    for a in np.linspace(1.0 / steps, 1.0, steps):
        xi = (x * float(a)).clone().requires_grad_(True)
        scalar = target_fn(model(xi))
        grad, = torch.autograd.grad(scalar, xi)
        total += grad
    return (total / steps) * x


def region_scores(attr, grid=GRID):
    """(N,C,H,W) -> (N, gr*gc) signed per-region sums, plus region boxes."""
    n, _, h, w = attr.shape
    gr, gc = grid
    rs, cs = np.array_split(np.arange(h), gr), np.array_split(np.arange(w), gc)
    scores, boxes = [], []
    for ri in rs:
        for ci in cs:
            blk = attr[:, :, ri[0]:ri[-1] + 1, ci[0]:ci[-1] + 1]
            scores.append(blk.sum(axis=(1, 2, 3)))
            boxes.append((ci[0], ri[0], len(ci), len(ri)))
    return np.stack(scores, 1), boxes


def grays(n):
    """n distinguishable grey levels for stacked bands."""
    return [str(v) for v in np.linspace(0.18, 0.88, n)]


def _finish(fig, title, sub, outfile, top=0.845):
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    fig.text(0.5, top + 0.08, sub, ha="center", fontsize=8.8, color="0.25")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"  wrote {outfile.relative_to(ROOT)}", flush=True)


# ------------------------------------------------------------- A1 global summary
def fig_summary(M, outfile):
    fig = plt.figure(figsize=(11.6, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.28)
    origin = M["origin"]

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(M["mean_img"], cmap="gray", aspect="auto", interpolation="nearest", origin=origin)
    a = M["attr_map"] / (M["attr_map"].max() + 1e-12)
    ax.contour(a, levels=[0.35, 0.6, 0.85], colors="k", linewidths=[0.7, 1.0, 1.4],
               origin="image" if origin == "upper" else "lower")
    ax.imshow(a, cmap="gray", alpha=0.30, aspect="auto", interpolation="bilinear", origin=origin)

    sc, boxes = M["scores"], M["boxes"]
    order = np.argsort(-np.abs(sc).mean(0))
    H = M["mean_img"].shape[0]
    for rank, idx in enumerate(order[:3]):
        x0, y0, w, h = boxes[idx]
        yy = (H - y0 - h) if origin == "lower" else y0
        ax.add_patch(plt.Rectangle((x0 - .5, yy - .5), w, h, fill=False, ec="k", lw=1.6, ls="--"))
        ax.text(x0 + w / 2 - .5, yy + h / 2 - .5, str(rank + 1), ha="center", va="center",
                fontsize=11, fontweight="bold", color="k",
                bbox=dict(boxstyle="circle,pad=0.16", fc="white", ec="k", lw=1.0))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Mean input with attribution contours", fontsize=9.5, pad=6)

    ax2 = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(0)
    top = order[:TOP_K][::-1]
    denom = np.abs(sc).max() + 1e-12
    for row, idx in enumerate(top):
        s = sc[:, idx] / denom
        shade = np.abs(s) / (np.abs(s).max() + 1e-12)
        ax2.scatter(s, row + rng.uniform(-0.17, 0.17, len(s)), s=9, c=1.0 - shade,
                    cmap="gray", vmin=0, vmax=1, edgecolors="k", linewidths=0.18, alpha=0.85)
    ax2.axvline(0, color="k", lw=0.9)
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels([M["label"](boxes[i], i) for i in top], fontsize=8)
    ax2.set_xlabel("Integrated-gradients attribution  (normalised)", fontsize=9)
    ax2.set_ylabel("input region  (ranked by mean |attribution|)", fontsize=9)
    ax2.set_title("Per-sample attribution by region", fontsize=9.5, pad=6)
    ax2.grid(axis="x", color="0.85", lw=0.6); ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    fig.subplots_adjust(top=0.845, bottom=0.13, left=0.045, right=0.985)
    _finish(fig, f"{M['name']} — global attribution summary",
            M["sub_summary"], outfile)


# ----------------------------------------------------------------- A2 waterfall
def fig_waterfall(M, outfile):
    sc, boxes = M["scores"], M["boxes"]
    order = np.argsort(-np.abs(sc).mean(0))
    keep = order[:TOP_K]
    pick = [int(np.argmax(M["pred"])), int(np.argmin(M["pred"]))]   # extreme instances

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))
    for ax, inst in zip(axes, pick):
        base = float(M["base"][inst])
        v = sc[inst]
        contrib = list(v[keep])
        labels = [M["label"](boxes[i], i) for i in keep]
        rest = float(v.sum() - v[keep].sum())
        contrib.append(rest); labels.append("all other regions")

        idx = np.argsort(-np.abs(contrib))
        contrib = [contrib[i] for i in idx]; labels = [labels[i] for i in idx]

        # walk once to get the cumulative path, so x-limits (and therefore the
        # label offset) are sized to THIS panel rather than a global range
        run, path = base, [base]
        for c in contrib:
            run += c
            path.append(run)
        lo, hi = min(path), max(path)
        span = (hi - lo) or 1.0
        ax.set_xlim(lo - 0.20 * span, hi + 0.20 * span)
        pad = 0.015 * span

        run = base
        for k, (c, lab) in enumerate(zip(contrib, labels)):
            y = len(contrib) - 1 - k
            ax.barh(y, c, left=run, height=0.62,
                    color="0.25" if c >= 0 else "0.75", edgecolor="k", linewidth=0.6)
            end = run + c
            ax.text(end + (pad if c >= 0 else -pad), y, f"{c:+.3f}", va="center",
                    fontsize=7.2, ha="left" if c >= 0 else "right")
            run = end
        ax.axvline(base, color="k", lw=1.0, ls=":")
        ax.axvline(run, color="k", lw=1.4)
        ax.set_yticks(range(len(contrib)))
        ax.set_yticklabels(labels[::-1], fontsize=7.8)
        ax.set_xlabel(f"model output: {M['target']}", fontsize=8.8)
        ax.set_title(f"instance #{inst}   base {base:+.3f} → prediction {run:+.3f}",
                     fontsize=9.2, pad=6)
        ax.grid(axis="x", color="0.88", lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.subplots_adjust(top=0.815, bottom=0.115, left=0.105, right=0.975, wspace=0.42)
    _finish(fig, f"{M['name']} — waterfall attribution for two instances",
            "Cumulative region contributions carrying the model from its baseline output "
            "(dotted) to the prediction (solid). Dark bars push up, light bars push down.",
            outfile, top=0.815)


# -------------------------------------------------- A3 / A4 stacked force plots
def fig_force(M, outfile, ordering):
    sc, boxes = M["scores"], M["boxes"]
    n = min(N_FORCE, len(sc))
    S = sc[:n]
    order = np.argsort(-np.abs(sc).mean(0))
    keep = order[:TOP_K]

    if ordering == "similarity":
        Z = linkage(S, method="average", metric="euclidean")
        idx = leaves_list(Z)
        xlab = "instances ordered by similarity of their explanations"
    else:
        idx = np.arange(n)
        xlab = "instances in dataset order"
    S = S[idx]

    pos = np.clip(S[:, keep], 0, None)
    neg = np.clip(S[:, keep], None, 0)
    other_p = np.clip(S.sum(1) - S[:, keep].sum(1), 0, None)
    other_n = np.clip(S.sum(1) - S[:, keep].sum(1), None, 0)

    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    x = np.arange(len(S))
    cols = grays(len(keep))
    ax.stackplot(x, *[pos[:, k] for k in range(len(keep))], other_p,
                 baseline="zero", colors=cols + ["0.95"], linewidth=0.25, edgecolor="k")
    ax.stackplot(x, *[neg[:, k] for k in range(len(keep))], other_n,
                 baseline="zero", colors=cols + ["0.95"], linewidth=0.25, edgecolor="k")
    ax.plot(x, S.sum(1), color="k", lw=1.5, label="total attribution  (f(x) − base)")
    ax.axhline(0, color="k", lw=1.0, ls=":")
    ax.set_xlim(0, len(S) - 1)
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(f"attribution to {M['target']}", fontsize=9)
    ax.grid(axis="y", color="0.88", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="k", lw=0.4) for c in cols]
    labels = [M["label"](boxes[i], i) for i in keep]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="0.95", ec="k", lw=0.4))
    labels.append("other regions")
    ax.legend(handles + [plt.Line2D([], [], color="k", lw=1.5)],
              labels + ["total  f(x) − base"], fontsize=7.2, ncol=6,
              loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)

    fig.subplots_adjust(top=0.815, bottom=0.245, left=0.075, right=0.985)
    kind = ("grouping of instances by resemblance of their explanations"
            if ordering == "similarity" else
            "consecutive organisation of instances across the dataset")
    _finish(fig, f"{M['name']} — global explanation, {ordering} ordering",
            f"Stacked region attributions per instance, showing the {kind}.",
            outfile, top=0.815)


# --------------------------------------------------------------------- per input
def grid_label(box, idx):
    _, gc = GRID
    r, c = divmod(idx, gc)
    return f"R{r + 1}C{c + 1}"


def build(name, key):
    if key == "blink":
        z = np.load(CACHE / "mrl_eyes.npz")
        _, _, te = tuning.group_split(z["subjects"], (0.7, 0.15, 0.15), seed=0)
        sel = te[np.linspace(0, len(te) - 1, N_SUMMARY).astype(int)]
        raw = z["imgs"][sel]
        x = torch.from_numpy(np.ascontiguousarray(
            preprocess.norm_gray(raw.astype(np.float32) / 255.0)[:, None]))
        model = torch.jit.load(str(ROOT / "models/custom/blink_cnn.pt"), map_location="cpu").eval()
        tgt = lambda o: o[:, 1]
        mean_img, origin, target = raw.mean(0), "upper", "logit P(closed)"
        sub = ("Which regions of the 32×32 eye crop drive P(closed); "
               f"Integrated Gradients over {N_SUMMARY} held-out eyes.")
        label = grid_label

    elif key == "gaze":
        z = np.load(CACHE / "mpiigaze.npz")
        _, _, te = tuning.group_split(z["parts"], (0.7, 0.15, 0.15), seed=0)
        sel = te[np.linspace(0, len(te) - 1, N_SUMMARY).astype(int)]
        raw = z["imgs"][sel]
        x = torch.from_numpy(np.ascontiguousarray(
            preprocess.norm_gray(raw.astype(np.float32) / 255.0)[:, None]))
        model = torch.jit.load(str(ROOT / "models/custom/gaze_cnn.pt"), map_location="cpu").eval()
        tgt = lambda o: o[:, 1]
        mean_img, origin, target = raw.mean(0), "upper", "predicted yaw (rad)"
        sub = ("Which regions of the 36×60 eye patch drive predicted yaw; "
               f"Integrated Gradients over {N_SUMMARY} held-out eyes (p05, p07).")
        label = grid_label

    elif key == "headpose":
        z = np.load(CACHE / "headpose_aflw2000.npz")
        crops = z["crops"]
        sel = np.linspace(0, len(crops) - 1, N_SUMMARY).astype(int)
        raw = crops[sel]
        xn = np.stack([preprocess.norm_rgb(c.astype(np.float32) / 255.0).transpose(2, 0, 1)
                       for c in raw])
        x = torch.from_numpy(np.ascontiguousarray(xn))
        model = torch.jit.load(str(ROOT / "models/custom/headpose_cnn.pt"), map_location="cpu").eval()
        tgt = lambda o: o[:, 0]
        mean_img, origin, target = raw.mean(0).mean(2), "upper", "predicted yaw (deg)"
        sub = ("Which regions of the 64×64 face crop drive predicted yaw; "
               f"Integrated Gradients over {N_SUMMARY} AFLW2000 faces.")
        label = grid_label

    else:  # voice
        z = np.load(CACHE / "voice_kwudsgyn_test.npz")
        X, Y = z["X"], z["Y"]
        sel = np.linspace(0, len(X) - 1, N_SUMMARY).astype(int)
        x = torch.from_numpy(np.ascontiguousarray(X[sel]))
        yt = torch.from_numpy(Y[sel])
        model = torch.jit.load(str(ROOT / "models/custom/voice_kws.pt"), map_location="cpu").eval()
        tgt = lambda o: o.gather(1, yt[:, None]).squeeze(1)
        mean_img, origin, target = X[sel, 0].mean(0), "lower", "logit of spoken keyword"
        sub = ("Which mel-band × time regions of the 40×81 log-mel spectrogram drive the "
               f"predicted keyword; Integrated Gradients over {N_SUMMARY} test clips.")

        def label(box, idx):
            gr, gc = GRID
            r, c = divmod(idx, gc)
            return f"mel{gr - r}/t{c + 1}"

    # tgt returns a PER-SAMPLE vector; IG needs a scalar, so sum it there. The
    # baseline must stay per-sample because for voice the target class (and hence
    # the baseline logit) differs between clips.
    attr = integrated_gradients(model, x, lambda o: tgt(o).sum()).detach().numpy()
    scores, boxes = region_scores(attr)
    with torch.no_grad():
        base = tgt(model(torch.zeros_like(x))).detach().numpy().astype(np.float64)
    pred = scores.sum(1) + base
    return dict(name=name, mean_img=mean_img, origin=origin, target=target,
                attr_map=np.abs(attr).mean((0, 1)), scores=scores, boxes=boxes,
                base=base, pred=pred, label=label, sub_summary=sub)


NAMES = {"blink": "Blink", "gaze": "Gaze", "headpose": "Head pose", "voice": "Voice"}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for key, nice in NAMES.items():
        if which not in ("all", key):
            continue
        print(f"[{key}]", flush=True)
        M = build(nice, key)
        d = OUT / key
        fig_summary(M, d / "A1_summary.png")
        fig_waterfall(M, d / "A2_waterfall.png")
        fig_force(M, d / "A3_similarity.png", "similarity")
        fig_force(M, d / "A4_ordering.png", "sample")
    print("attribution figures done")
