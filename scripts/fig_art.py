"""16 signature figures (4 per input) — dark, luminous, unusual forms. Real matched data."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle2 as S
import figstyle as Fm   # math helpers only (gaze_vec / gaze_angular_err)
S.apply()
rng = np.random.default_rng(7)
PT = ROOT / "reports/tuning/pretrained"
ART = ROOT / "reports" / "figures" / "figures_art"   # alongside the other figures

# ---- data
_gz = PT / "matched_gaze_heldout.npz"      # held-out participants only: the full 15-participant
_gz = _gz if _gz.exists() else PT / "matched_gaze.npz"   # file contains 9 of the custom model's
g = np.load(_gz); gt, gpc, gpp = g["true"], g["pred_custom"], g["pred_pt"]   # own TRAINING people
gec, gep = Fm.gaze_angular_err(gt, gpc), Fm.gaze_angular_err(gt, gpp)
h = np.load(PT / "matched_headpose.npz"); ht, hpc, hpp = h["true"], h["pred_custom"], h["pred_pt"]
hmc, hmp = np.abs(hpc - ht).mean(1), np.abs(hpp - ht).mean(1)
b = np.load(PT / "matched_blink.npz", allow_pickle=True); bprob, blab = b["prob"], b["label"]
bscore = bprob[:, 1]; by = (blab == 1).astype(int)
v = np.load(PT / "matched_voice.npz", allow_pickle=True)
vprob, vlab = v["prob"], v["label"].astype(int); vnames = [str(x) for x in v["labels"]]
vpred = vprob.argmax(1)


def sub(n, k):
    return rng.choice(n, min(k, n), replace=False)


# ============================================================ GAZE
def g1_comet():
    idx = sub(len(gt), 1300)
    fig, ax = plt.subplots(figsize=(10.4, 8.2)); ax.set_facecolor(S.BG)
    for pred, col, hi, lab in [(gpp, S.PRETRAIN, S.PRETRAIN_HI, "pretrained"), (gpc, S.CUSTOM, S.CUSTOM_HI, "custom")]:
        segs = [[(gt[i, 1], gt[i, 0]), (pred[i, 1], pred[i, 0])] for i in idx]
        lc = LineCollection(segs, colors=col, linewidths=0.7, alpha=0.32)
        ax.add_collection(lc)
        S.glow(ax, pred[idx, 1], pred[idx, 0], col, s=10, layers=((6, .04), (3, .09)), core=.8)
    ax.scatter(gt[idx, 1], gt[idx, 0], s=2, c=S.MUTED, alpha=0.5, zorder=1)
    ax.set_xlim(-35, 35); ax.set_ylim(-30, 30); ax.set_aspect("equal")
    ax.set_xlabel("gaze yaw (deg)"); ax.set_ylabel("gaze pitch (deg)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=S.CUSTOM, marker="o", ls="", label=S.LABELS["custom"]),
                       Line2D([], [], color=S.PRETRAIN, marker="o", ls="", label=S.LABELS["pretrained"])],
              loc="upper right", facecolor=S.PANEL, edgecolor=S.GRID, labelcolor=S.INK)
    S.title(fig, "Comet field", "Each comet streaks from the true gaze (grey) to the model's prediction — length is the error, glow marks where it landed.")
    S.caption(fig, "MPIIGaze, matched per-sample (1,300 shown). Custom = ember, pretrained = ion-cyan.")
    S.save(fig, ART / "gaze" / "GA1_comet_field.png")


def g2_aurora():
    fig, ax = plt.subplots(figsize=(10.6, 6.4)); ax.set_facecolor(S.BG)
    xs = np.linspace(0, 20, 300)
    for err, cmap, edge, lab, sgn in [(gep, S.ION, S.PRETRAIN, "pretrained", 1), (gec, S.EMBER, S.CUSTOM, "custom", 1)]:
        k = gaussian_kde(np.clip(err, 0, 20))(xs)
        S.gradient_fill(ax, xs, k, cmap, alpha=0.9, glowline=edge)
    ax.set_xlim(0, 20); ax.set_ylim(0, None); ax.set_yticks([])
    ax.set_xlabel("angular error (deg)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=S.CUSTOM, label=S.LABELS["custom"]), Patch(color=S.PRETRAIN, label=S.LABELS["pretrained"])],
              loc="upper right", facecolor=S.PANEL, edgecolor=S.GRID, labelcolor=S.INK)
    S.title(fig, "Aurora of error", "The angular-error distributions as luminous curtains. The custom curtain glows wider and further right — larger errors.")
    S.caption(fig, "MPIIGaze, N=30,000, matched per-sample.")
    S.save(fig, ART / "gaze" / "GA2_aurora.png")


def g3_nebula():
    theta = np.arctan2(gt[:, 1], gt[:, 0]); ecc = np.sqrt((gt ** 2).sum(1))
    idx = sub(len(gt), 14000)
    fig = plt.figure(figsize=(13.4, 6.8))
    for s, (err, name, col) in enumerate([(gec, "custom", S.CUSTOM), (gep, "pretrained", S.PRETRAIN)]):
        ax = fig.add_subplot(1, 2, s + 1, projection="polar"); ax.set_facecolor(S.BG)
        c = np.clip(err[idx], 0, 10)
        for m, a in ((5, .05), (2.4, .12)):
            ax.scatter(theta[idx], ecc[idx], s=m * 4, c=c, cmap=S.NEBULA, alpha=a, linewidths=0, vmin=0, vmax=10)
        sc = ax.scatter(theta[idx], ecc[idx], s=3, c=c, cmap=S.NEBULA, alpha=0.9, linewidths=0, vmin=0, vmax=10)
        ax.set_title(S.LABELS[name], color=col, pad=16)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.grid(color=S.GRID, alpha=0.3); ax.set_yticklabels([]); ax.tick_params(colors=S.MUTED)
    cb = fig.colorbar(sc, ax=fig.axes, fraction=0.02, pad=0.05); cb.set_label("angular error (deg)", color=S.INK2)
    cb.ax.yaxis.set_tick_params(color=S.MUTED)
    S.title(fig, "Gaze nebula", "Every gaze as a glowing mote — angle = look direction, radius = eccentricity, brightness = error. Custom burns hotter at the rim.")
    S.caption(fig, "MPIIGaze, 14,000 motes per panel, matched per-sample.")
    S.save(fig, ART / "gaze" / "GA3_nebula.png")


def g4_islands():
    fig, ax = plt.subplots(figsize=(9.8, 8.4)); ax.set_facecolor(S.BG)
    gx = np.linspace(-14, 14, 200); gy = np.linspace(-14, 14, 200); MX, MY = np.meshgrid(gx, gy)
    grid = np.vstack([MX.ravel(), MY.ravel()])
    for pred, cmap, edge, lab in [(gpp, S.ION, S.PRETRAIN, "pretrained"), (gpc, S.EMBER, S.CUSTOM, "custom")]:
        d = (pred - gt); idx = sub(len(d), 6000)
        Z = gaussian_kde(d[idx].T)(grid).reshape(MX.shape)
        ax.contourf(MX, MY, Z, levels=8, cmap=cmap, alpha=0.55)
        ax.contour(MX, MY, Z, levels=6, colors=[edge], linewidths=0.7, alpha=0.6)
    ax.axhline(0, color=S.GRID, lw=1); ax.axvline(0, color=S.GRID, lw=1)
    ax.scatter([0], [0], s=60, c=S.GOLD, marker="+", zorder=5)
    ax.set_xlabel("yaw error (deg)"); ax.set_ylabel("pitch error (deg)"); ax.set_aspect("equal")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=S.CUSTOM, label=S.LABELS["custom"]), Patch(color=S.PRETRAIN, label=S.LABELS["pretrained"])],
              loc="upper right", facecolor=S.PANEL, edgecolor=S.GRID, labelcolor=S.INK)
    S.title(fig, "Error islands", "Topographic map of the mistakes: contours where predictions land relative to truth. A tight island on the gold cross = accurate & unbiased.")
    S.caption(fig, "MPIIGaze, 2-D error density (6,000 samples/model), matched per-sample.")
    S.save(fig, ART / "gaze" / "GA4_error_islands.png")


# ============================================================ HEAD POSE
def h1_atlas():
    delta = hmp - hmc          # >0 custom better
    order = np.argsort(np.abs(delta))
    fig, ax = plt.subplots(figsize=(11.0, 7.0)); ax.set_facecolor(S.BG)
    for gx in range(-90, 91, 30): ax.axvline(gx, color=S.GRID, lw=0.6, alpha=0.5)
    for gy in range(-60, 61, 30): ax.axhline(gy, color=S.GRID, lw=0.6, alpha=0.5)
    c = np.clip(delta[order], -10, 10)
    for m, a in ((6, .05), (3, .12)):
        ax.scatter(ht[order, 0], ht[order, 1], s=(14 + np.abs(delta[order]) * 3) * m, c=c, cmap=S.DIVE, vmin=-10, vmax=10, alpha=a, linewidths=0)
    sc = ax.scatter(ht[order, 0], ht[order, 1], s=14 + np.abs(delta[order]) * 3, c=c, cmap=S.DIVE, vmin=-10, vmax=10, alpha=0.95, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02); cb.set_label("← pretrained wins   ·   custom wins →", color=S.INK2)
    cb.ax.yaxis.set_tick_params(color=S.MUTED)
    ax.set_xlabel("head yaw (deg)"); ax.set_ylabel("head pitch (deg)"); ax.set_xlim(-95, 95); ax.set_ylim(-65, 65)
    S.title(fig, "Atlas of poses", "Every test face charted by its true head angle and lit by which model wins it — an ember-vs-ion map of the pose sphere.")
    S.caption(fig, "AFLW2000, N=1,969, matched per-sample. Size = win margin.")
    S.save(fig, ART / "headpose" / "HA1_atlas.png")


def h2_butterfly():
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 6.2))
    AX = ["yaw", "pitch", "roll"]
    for k, ax in enumerate(axes):
        ax.set_facecolor(S.BG)
        xs = np.linspace(-20, 20, 240)
        for err, cmap, edge, up in [(hpc[:, k] - ht[:, k], S.EMBER, S.CUSTOM, 1), (hpp[:, k] - ht[:, k], S.ION, S.PRETRAIN, -1)]:
            kd = gaussian_kde(np.clip(err, -20, 20))(xs)
            kd = kd / kd.max()
            y = up * kd
            ax.fill_between(xs, 0, y, color=edge, alpha=0.18, lw=0)
            for lw, a in ((6, .10), (2.6, .3), (1.2, .9)):
                ax.plot(xs, y, color=edge, lw=lw, alpha=a)
        ax.axhline(0, color=S.GRID, lw=1); ax.axvline(0, color=S.GRID, lw=0.6, alpha=0.5)
        ax.set_title(AX[k], color=S.INK); ax.set_xlim(-20, 20); ax.set_ylim(-1.15, 1.15)
        ax.set_yticks([]); ax.set_xlabel("signed error (deg)")
    axes[0].text(-18, 0.8, "custom", color=S.CUSTOM, fontsize=10, fontweight="bold")
    axes[0].text(-18, -0.9, "pretrained", color=S.PRETRAIN, fontsize=10, fontweight="bold")
    S.title(fig, "Butterfly of drift", "Each axis a pair of wings: custom above, pretrained below. Symmetric, slender wings pinned on zero = unbiased and precise.")
    S.caption(fig, "AFLW2000, N=1,969, matched per-sample.")
    S.save(fig, ART / "headpose" / "HA2_butterfly.png")


def h3_silk():
    o = np.argsort(ht[:, 0]); yaw = ht[o, 0]
    def smooth(e, w=90):
        e = e[o]; k = np.ones(w) / w
        return np.convolve(np.pad(e, w // 2, mode="edge"), k, "same")[w // 2:-w // 2 or None][:len(e)]
    fig, ax = plt.subplots(figsize=(11.2, 6.2)); ax.set_facecolor(S.BG)
    for err, cmap, edge in [(hmp, S.ION, S.PRETRAIN), (hmc, S.EMBER, S.CUSTOM)]:
        sm = smooth(err)
        S.gradient_fill(ax, yaw, sm, cmap, alpha=0.85, glowline=edge)
    ax.set_xlim(yaw.min(), yaw.max()); ax.set_ylim(0, None)
    ax.set_xlabel("head yaw (deg) — profile · frontal · profile"); ax.set_ylabel("error (deg)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=S.CUSTOM, label=S.LABELS["custom"]), Patch(color=S.PRETRAIN, label=S.LABELS["pretrained"])],
              loc="upper center", facecolor=S.PANEL, edgecolor=S.GRID, labelcolor=S.INK)
    S.title(fig, "Silk streams", "Error flowing as the head turns from profile to frontal and back. The custom stream swells into the profile extremes.")
    S.caption(fig, "AFLW2000, N=1,969, error smoothed across yaw.")
    S.save(fig, ART / "headpose" / "HA3_silk.png")


def h4_honeycomb():
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2))
    for ax, err, name, col in [(axes[0], hmc, "custom", S.CUSTOM), (axes[1], hmp, "pretrained", S.PRETRAIN)]:
        ax.set_facecolor(S.BG)
        hb = ax.hexbin(ht[:, 0], ht[:, 1], C=np.clip(err, 0, 20), gridsize=22, cmap=S.NEBULA,
                       reduce_C_function=np.mean, mincnt=1, edgecolors=S.BG, linewidths=0.6, vmin=0, vmax=16)
        ax.set_title(S.LABELS[name], color=col); ax.set_xlabel("yaw (deg)"); ax.set_ylabel("pitch (deg)")
        ax.set_xlim(-95, 95); ax.set_ylim(-65, 65)
    cb = fig.colorbar(hb, ax=axes, fraction=0.022, pad=0.02); cb.set_label("mean error (deg)", color=S.INK2)
    cb.ax.yaxis.set_tick_params(color=S.MUTED)
    S.title(fig, "Honeycomb of error", "The pose plane tiled into cells, each glowing by the error inside it. Bright combs at the edges are the hard, extreme poses.")
    S.caption(fig, "AFLW2000, N=1,969, matched per-sample.")
    S.save(fig, ART / "headpose" / "HA4_honeycomb.png")


# ============================================================ VOICE
def v1_chord():
    from sklearn.metrics import confusion_matrix
    C = confusion_matrix(vlab, vpred, labels=range(6)).astype(float)
    supp = C.sum(1); K = 6
    ang = np.linspace(90, 90 - 360, K, endpoint=False)
    pos = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in ang])
    fig, ax = plt.subplots(figsize=(9.2, 9.2)); ax.set_facecolor(S.BG); ax.set_aspect("equal"); ax.axis("off")
    mx = C.copy(); np.fill_diagonal(mx, 0); mmax = mx.max()
    for i in range(K):
        for j in range(K):
            if i != j and C[i, j] > 0:
                S.ribbon(ax, pos[i] * 0.92, pos[j] * 0.92, S.CAT6[i], width=0.02 + 0.12 * C[i, j] / mmax,
                         alpha=0.28 + 0.5 * C[i, j] / mmax, bow=0.28)
    for i in range(K):
        S.glow(ax, [pos[i, 0]], [pos[i, 1]], S.CAT6[i], s=120 + supp[i] / supp.max() * 480, layers=((3, .10), (1.8, .25)), core=.95)
        lab = pos[i] * 1.16
        ax.text(lab[0], lab[1], vnames[i], color=S.CAT6[i], ha="center", va="center", fontsize=13, fontweight="bold")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    S.title(fig, "Confusion constellation", "Words as stars (size = how often spoken); glowing arcs are the mis-hearings that flow between them. Few, faint arcs = a clean recogniser.")
    S.caption(fig, "Speech Commands test split, N=2,468. Only off-diagonal confusions drawn.")
    S.save(fig, ART / "voice" / "VA1_chord.png")


def v2_sunflower():
    conf = vprob.max(1); n = len(vlab)
    order = np.argsort(-conf)                # bright confident seeds toward centre
    gold = np.pi * (3 - np.sqrt(5))
    i = np.arange(n); r = np.sqrt(i / n); th = i * gold
    x, y = r * np.cos(th), r * np.sin(th)
    fig, ax = plt.subplots(figsize=(9.4, 9.4)); ax.set_facecolor(S.BG); ax.set_aspect("equal"); ax.axis("off")
    correct = (vpred == vlab)[order]
    cols = np.array(S.CAT6)[vpred[order]]
    sizes = 8 + conf[order] * 42
    for m, a in ((4, .06), (2, .16)):
        ax.scatter(x, y, s=sizes * m, c=cols, alpha=a, linewidths=0)
    ax.scatter(x, y, s=sizes, c=cols, alpha=0.95, linewidths=0)
    w = ~correct
    ax.scatter(x[w], y[w], s=sizes[w] + 22, facecolors="none", edgecolors="white", linewidths=0.8, alpha=0.8)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=S.CAT6[k], marker="o", ls="", label=vnames[k]) for k in range(6)],
              loc="upper right", facecolor=S.PANEL, edgecolor=S.GRID, labelcolor=S.INK, ncol=2, fontsize=9)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    S.title(fig, "Confidence sunflower", "Every clip is a seed — colour = predicted word, size & glow = confidence, centre = most certain. White-ringed seeds are the mistakes.")
    S.caption(fig, "Speech Commands test split, N=2,468. Phyllotaxis by confidence.")
    S.save(fig, ART / "voice" / "VA2_sunflower.png")


def v3_starmap():
    X = vprob - vprob.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(10.0, 8.4)); ax.set_facecolor(S.BG); ax.axis("off")
    for k in range(6):
        m = vlab == k
        S.glow(ax, P[m, 0], P[m, 1], S.CAT6[k], s=14, layers=((6, .04), (3, .10)), core=.85)
        cx, cy = P[m, 0].mean(), P[m, 1].mean()
        ax.text(cx, cy, vnames[k], color="white", fontsize=13, fontweight="bold", ha="center",
                path_effects=None)
    S.title(fig, "Word star-map", "Each clip's 6-way probability projected to 2-D. Clips of the same word cluster into their own constellation — tight, separated islands = confident classes.")
    S.caption(fig, "Speech Commands test split, N=2,468. PCA of the probability vectors.")
    S.save(fig, ART / "voice" / "VA3_starmap.png")


def v4_aurora():
    fig, ax = plt.subplots(figsize=(10.2, 7.0)); ax.set_facecolor(S.BG)
    xs = np.linspace(0, 1, 220)
    for k in range(6):
        p = vprob[vlab == k, k]
        kd = gaussian_kde(np.clip(p, 0, 1))(xs); kd = kd / kd.max() * 0.9
        base = (6 - 1 - k)
        ax.fill_between(xs, base, base + kd, color=S.CAT6[k], alpha=0.16, lw=0)
        for lw, a in ((6, .10), (2.4, .3), (1.1, .95)):
            ax.plot(xs, base + kd, color=S.CAT6[k], lw=lw, alpha=a)
        ax.text(-0.02, base + 0.12, vnames[k], color=S.CAT6[k], ha="right", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.12, 1.02); ax.set_yticks([]); ax.set_xlabel("probability given to the correct word")
    S.title(fig, "Aurora of the keywords", "For each word, the glow of confidence in the right answer. Curtains piled at 1.0 are the easy, decisively-recognised words.")
    S.caption(fig, "Speech Commands test split, N=2,468.")
    S.save(fig, ART / "voice" / "VA4_aurora.png")


# ============================================================ BLINK
def b1_stream():
    fig, ax = plt.subplots(figsize=(10.6, 6.6)); ax.set_facecolor(S.BG)
    xs = np.linspace(0, 1, 300)
    ko = gaussian_kde(np.clip(bscore[by == 0], 0, 1))(xs)
    kc = gaussian_kde(np.clip(bscore[by == 1], 0, 1))(xs)
    S.gradient_fill(ax, xs, kc, S.EMBER, alpha=0.9, glowline=S.CUSTOM)
    S.gradient_fill(ax, xs, -ko, S.ION, alpha=0.9, glowline=S.PRETRAIN, y0=0)
    ax.axvline(0.5, color=S.GOLD, lw=1, alpha=0.6, ls=(0, (4, 4)))
    ax.axhline(0, color=S.GRID, lw=1)
    ax.text(0.9, kc.max() * 0.6, "truly closed", color=S.CUSTOM, fontsize=11, fontweight="bold", ha="right")
    ax.text(0.1, -ko.max() * 0.6, "truly open", color=S.PRETRAIN, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xlabel("model output  P(closed)")
    S.title(fig, "The delta of decision", "Two facing rivers of score: closed eyes flow up, open eyes flow down. They part cleanly at the gold seam — the model rarely hesitates.")
    S.caption(fig, "MRL Eye Dataset test split, N=12,927.")
    S.save(fig, ART / "blink" / "BA1_delta_stream.png")


def b2_galaxy():
    idx = sub(len(bscore), 7000)
    o = idx[np.argsort(bscore[idx])]
    n = len(o); i = np.arange(n)
    r = np.sqrt(i / n); th = i * 0.0045 * n / 6 * (2 * np.pi) / n * 6   # a few sweeping arms
    th = i * (np.pi * (3 - np.sqrt(5)))
    x, y = r * np.cos(th), r * np.sin(th)
    col = np.where(by[o] == 1, S.CUSTOM, S.PRETRAIN)
    fig, ax = plt.subplots(figsize=(9.4, 9.4)); ax.set_facecolor(S.BG); ax.set_aspect("equal"); ax.axis("off")
    for m, a in ((5, .05), (2.4, .12)):
        ax.scatter(x, y, s=10 * m, c=col, alpha=a, linewidths=0)
    ax.scatter(x, y, s=7, c=col, alpha=0.9, linewidths=0)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=S.CUSTOM, marker="o", ls="", label="truly closed"),
                       Line2D([], [], color=S.PRETRAIN, marker="o", ls="", label="truly open")],
              loc="upper right", facecolor=S.PANEL, edgecolor=S.GRID, labelcolor=S.INK)
    S.title(fig, "Blink galaxy", "12,927 eyes spun onto a spiral, ordered by score from centre outward and coloured by their true state. The two colours settle into separate arms.")
    S.caption(fig, "MRL Eye Dataset test split (7,000 shown), phyllotaxis by P(closed).")
    S.save(fig, ART / "blink" / "BA2_galaxy.png")


def b3_threshold_aurora():
    ts = np.linspace(0, 1, 200)
    P, N = (by == 1).sum(), (by == 0).sum()
    TP = np.array([((bscore >= t) & (by == 1)).sum() for t in ts]) / len(by)
    TN = np.array([((bscore < t) & (by == 0)).sum() for t in ts]) / len(by)
    FP = np.array([((bscore >= t) & (by == 0)).sum() for t in ts]) / len(by)
    FN = np.array([((bscore < t) & (by == 1)).sum() for t in ts]) / len(by)
    fig, ax = plt.subplots(figsize=(10.6, 6.4)); ax.set_facecolor(S.BG)
    ax.stackplot(ts, TP, TN, FP, FN, colors=[S.CUSTOM, S.PRETRAIN, "#c83a86", "#7a3cff"],
                 labels=["closed ✓", "open ✓", "false positive", "false negative"], alpha=0.88)
    ax.axvline(0.5, color=S.GOLD, lw=1, alpha=0.7, ls=(0, (4, 4)))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("decision threshold"); ax.set_ylabel("share of samples")
    leg = ax.legend(loc="center left", facecolor=S.PANEL, edgecolor=S.GRID, labelcolor=S.INK, fontsize=9)
    S.title(fig, "Threshold aurora", "As the click threshold glides 0→1, the outcome bands shift like light. The dim error bands stay thin across a broad, safe middle.")
    S.caption(fig, "MRL Eye Dataset test split, N=12,927.")
    S.save(fig, ART / "blink" / "BA3_threshold_aurora.png")


def b4_arc():
    idx = sub(len(bscore), 6000)
    ang = np.pi * (1 - bscore[idx])          # score 0 -> pi (left), 1 -> 0 (right)
    rr = 1 + rng.normal(0, 0.05, len(idx))
    correct = (bscore[idx] >= 0.5).astype(int) == by[idx]
    x, y = rr * np.cos(ang), rr * np.sin(ang)
    fig, ax = plt.subplots(figsize=(11.0, 6.4)); ax.set_facecolor(S.BG); ax.set_aspect("equal"); ax.axis("off")
    for mask, col, lab in [(correct, "#7ce38a", "correct"), (~correct, "#ff4d5e", "wrong")]:
        for m, a in ((6, .05), (3, .12)):
            ax.scatter(x[mask], y[mask], s=9 * m, c=col, alpha=a, linewidths=0)
        ax.scatter(x[mask], y[mask], s=7, c=col, alpha=0.9, linewidths=0)
    for sc, tx in [(1, "P=1  closed"), (0.5, "0.5"), (0, "P=0  open")]:
        a = np.pi * (1 - sc); ax.text(1.24 * np.cos(a), 1.24 * np.sin(a), tx, color=S.MUTED, ha="center", fontsize=9)
    ax.text(0, -0.25, "certain →  outer edges", color=S.MUTED, ha="center", fontsize=9)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color="#7ce38a", marker="o", ls="", label="correct"),
                       Line2D([], [], color="#ff4d5e", marker="o", ls="", label="wrong")],
              loc="lower center", facecolor=S.PANEL, edgecolor=S.GRID, labelcolor=S.INK, ncol=2)
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-0.35, 1.4)
    S.title(fig, "Arc of certainty", "Each eye placed on a rainbow arc by its score — open at left, closed at right. Red mistakes cluster near the middle, where the model was unsure.")
    S.caption(fig, "MRL Eye Dataset test split (6,000 shown).")
    S.save(fig, ART / "blink" / "BA4_certainty_arc.png")


ALL = [g1_comet, g2_aurora, g3_nebula, g4_islands,
       h1_atlas, h2_butterfly, h3_silk, h4_honeycomb,
       v1_chord, v2_sunflower, v3_starmap, v4_aurora,
       b1_stream, b2_galaxy, b3_threshold_aurora, b4_arc]

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for fn in ALL:
        if which == "all" or fn.__name__.startswith(which):
            fn()
    print("art figures done")
