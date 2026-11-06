"""7 fascinating voice figures — custom keyword CNN, with a genuine macro-metric
comparison against pretrained Vosk (Vosk exposes summary metrics only, no per-sample
probabilities, so per-sample figures are the custom model's)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import (roc_curve, auc, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, matthews_corrcoef,
                             cohen_kappa_score, confusion_matrix)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as F
F.apply_style()

OUT = ROOT / "reports" / "figures" / "voice"
d = np.load(ROOT / "reports/tuning/pretrained/matched_voice.npz", allow_pickle=True)
prob, label = d["prob"], d["label"].astype(int)
names = [str(x) for x in d["labels"]]
K = len(names)
pred = prob.argmax(1)
vosk = json.load(open(ROOT / "reports/tuning/pretrained/voice_vosk.metrics.json"))
COL = F.CAT[:K]


def v1_confusion():
    C = confusion_matrix(label, pred, labels=range(K)).astype(float)
    Cn = C / C.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    im = ax.imshow(Cn, cmap=F.SEQ, vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            v = Cn[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=10,
                        color="white" if v > 0.5 else F.INK2, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(names); ax.set_yticklabels(names)
    ax.set_xlabel("predicted word"); ax.set_ylabel("true word"); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label("row %", color=F.INK2)
    F.title(fig, "Which words get confused? — custom CNN confusion matrix",
            "Row-normalised: each row shows where that spoken word's clips were sent. Strong diagonal = few mix-ups.")
    F.caption(fig, "Speech Commands v2, 6-keyword test split, N=2,468. Values are % of each true class.")
    F.savefig(fig, OUT / "V1_confusion.png")


def v2_nightingale():
    metrics = [
        ("Accuracy", (pred == label).mean(), vosk["accuracy"]),
        ("Balanced\nacc.", balanced_accuracy_score(label, pred), vosk["balanced_accuracy"]),
        ("Precision\n(macro)", precision_score(label, pred, average="macro", zero_division=0), vosk["precision_macro"]),
        ("Recall\n(macro)", recall_score(label, pred, average="macro", zero_division=0), vosk["recall_macro"]),
        ("F1\n(macro)", f1_score(label, pred, average="macro"), vosk["f1_macro"]),
        ("MCC", matthews_corrcoef(label, pred), vosk["mcc"]),
        ("Cohen\nκ", cohen_kappa_score(label, pred), vosk["cohen_kappa"]),
    ]
    n = len(metrics); base = np.linspace(0, 2 * np.pi, n, endpoint=False)
    lo = 0.90
    def rad(v): return (v - lo) / (1 - lo)
    fig = plt.figure(figsize=(8.8, 8.4)); ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor(F.SURFACE)
    w = 2 * np.pi / n * 0.34
    for k, (nm, c, v) in enumerate(metrics):
        ax.bar(base[k] - w / 1.7, rad(c), width=w, bottom=0, color=F.CUSTOM, alpha=0.9, edgecolor="white", linewidth=1)
        ax.bar(base[k] + w / 1.7, rad(v), width=w, bottom=0, color=F.PRETRAIN, alpha=0.9, edgecolor="white", linewidth=1)
        ax.text(base[k] - w / 1.7, rad(c) + 0.05, f"{c:.3f}", ha="center", fontsize=7.6, color=F.CUSTOM_D, rotation=0)
        ax.text(base[k] + w / 1.7, rad(v) + 0.05, f"{v:.3f}", ha="center", fontsize=7.6, color=F.PRETRAIN_D)
    ax.set_xticks(base); ax.set_xticklabels([m[0] for m in metrics], fontsize=10)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.2); ax.set_yticks([rad(x) for x in (0.92, 0.95, 0.98, 1.0)])
    ax.set_yticklabels(["0.92", "0.95", "0.98", "1.0"], fontsize=8, color=F.MUTED)
    ax.grid(color=F.GRID); ax.tick_params(colors=F.MUTED)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=F.CUSTOM, label=F.LABELS["custom"]),
                       Patch(color=F.PRETRAIN, label="Vosk (pretrained)")],
              loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    F.title(fig, "Custom CNN vs pretrained Vosk — the metric rose",
            "Seven summary scores as paired petals (orange = mine, blue = Vosk). Radius is stretched over 0.90–1.0 so small gaps are visible.")
    F.caption(fig, "Speech Commands v2, 6-keyword test split. Custom N=2,468; Vosk on its own recognised subset.")
    F.savefig(fig, OUT / "V2_metric_rose.png")


def v3_constellation():
    conf = prob.max(1); correct = (pred == label).astype(int)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
    acc = np.array([correct[idx == k].mean() if (idx == k).any() else np.nan for k in range(10)])
    cf = np.array([conf[idx == k].mean() if (idx == k).any() else np.nan for k in range(10)])
    cnt = np.array([(idx == k).sum() for k in range(10)])
    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    sc = F.constellation(ax, cf, acc, cnt)
    ece = np.nansum(cnt / cnt.sum() * np.abs(acc - cf))
    ax.text(0.05, 0.92, f"ECE = {ece:.4f}", fontsize=13, fontweight="bold", color=F.CUSTOM_D)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("over- ↔ under-confidence (accuracy − confidence)", color=F.INK2)
    ax.set_xlabel("predicted confidence (max softmax)"); ax.set_ylabel("actual accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    F.title(fig, "Does the model know when it's right? — calibration constellation",
            "Each bubble is a confidence bin: on the grey diagonal = trustworthy; size = sample count; colour = the direction of miscalibration.")
    F.caption(fig, "Speech Commands v2 test split, N=2,468. 10 confidence bins.")
    F.savefig(fig, OUT / "V3_constellation.png")


def v4_ridgeline():
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    xs = np.linspace(0, 1, 200)
    for k in range(K):
        p_true = prob[label == k, k]
        kde = gaussian_kde(np.clip(p_true, 0, 1))(xs); kde = kde / kde.max() * 0.85
        base = (K - 1 - k) * 1.0
        ax.fill_between(xs, base, base + kde, color=COL[k], alpha=0.7, lw=0)
        ax.plot(xs, base + kde, color=COL[k], lw=1.4)
        ax.text(-0.02, base + 0.1, names[k], ha="right", va="bottom", fontsize=11, color=COL[k], fontweight="bold")
        ax.text(1.02, base + 0.1, f"{p_true.mean():.2f}", ha="left", va="bottom", fontsize=9, color=F.MUTED)
    ax.set_xlim(0, 1.1); ax.set_yticks([]); ax.set_xlabel("probability the model assigned to the CORRECT word")
    F.title(fig, "Per-word confidence — which keywords are easy?",
            "For each true word, the spread of probability the model gave the right answer. Mass piled near 1.0 = an easy, confidently-recognised word.")
    F.caption(fig, "Speech Commands v2 test split, N=2,468. Number at right = mean correct-class probability.")
    F.savefig(fig, OUT / "V4_perword_ridgeline.png")


def v5_separation():
    """Per-word mirrored densities: model's score for the target word among true-word
    clips (up) vs all other clips (down). Wide separation ⇒ high AUC — shown with no lines."""
    yb = np.eye(K)[label]
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.2))
    xs = np.linspace(0, 1, 200)
    for k, ax in enumerate(axes.ravel()):
        fpr, tpr, _ = roc_curve(yb[:, k], prob[:, k]); a = auc(fpr, tpr)
        pos, neg = prob[label == k, k], prob[label != k, k]
        kp = gaussian_kde(np.clip(pos, 0, 1))(xs); kp /= kp.max()
        kn = gaussian_kde(np.clip(neg, 0, 1))(xs); kn /= kn.max()
        ax.fill_between(xs, 0, kp, color=COL[k], alpha=0.75, lw=0)
        ax.fill_between(xs, 0, -kn, color=F.MUTED, alpha=0.5, lw=0)
        ax.axhline(0, color=F.AXIS, lw=0.8)
        ax.set_title(f"'{names[k]}'  ·  AUC {a:.3f}", color=COL[k], fontsize=11.5)
        ax.set_xlim(0, 1); ax.set_ylim(-1.15, 1.15); ax.set_yticks([])
        if k >= 3: ax.set_xlabel(f"score for '{names[k]}'")
        ax.text(0.03, 0.8, "this word", color=COL[k], fontsize=8.5)
        ax.text(0.03, -0.95, "other words", color=F.MUTED, fontsize=8.5)
    F.title(fig, "One word against all others — separation densities",
            "For each keyword: the model's score on clips that ARE that word (coloured, up) vs everything else (grey, down). The cleaner the split, the higher the AUC.")
    F.caption(fig, "Speech Commands v2 test split, N=2,468. One-vs-rest score distributions.")
    F.savefig(fig, OUT / "V5_separation.png")


def v6_margin():
    srt = np.sort(prob, 1)
    margin = srt[:, -1] - srt[:, -2]
    correct = pred == label
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    xs = np.linspace(0, 1, 220)
    for m, col, lab in [(margin[correct], F.GOOD, "correct"), (margin[~correct], F.CRIT, "wrong")]:
        if len(m) > 5:
            kde = gaussian_kde(np.clip(m, 0, 1))(xs)
            ax.fill_between(xs, kde, color=col, alpha=0.45, lw=0, label=f"{lab} ({len(m)})")
    ax.set_xlabel("decision margin  (top prob − runner-up)"); ax.set_ylabel("density")
    ax.set_xlim(0, 1); ax.legend(loc="upper center")
    F.title(fig, "How decisive — and is hesitation a warning sign?",
            "Gap between the winning word and the runner-up. Wrong answers (red) cluster at low margins — the model's hesitation flags its own mistakes.")
    F.caption(fig, "Speech Commands v2 test split, N=2,468.")
    F.savefig(fig, OUT / "V6_decision_margin.png")


def v7_chord():
    C = confusion_matrix(label, pred, labels=range(K)).astype(float)
    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    F.sankey_ribbons(ax, C, names, names, COL)
    acc = (pred == label).mean()
    fig.text(0.5, 0.47, f"{acc*100:.1f}% correct", ha="center", fontsize=12, color=F.INK, fontweight="bold")
    F.title(fig, "The flow of every spoken word",
            "Each clip streams from its true word (left) to the predicted word (right). Cross-overs reveal the specific confusions (e.g. go↔no).")
    F.caption(fig, "Speech Commands v2 test split, N=2,468. Ribbon width ∝ number of clips.")
    F.savefig(fig, OUT / "V7_word_flow.png")


if __name__ == "__main__":
    v1_confusion(); v2_nightingale(); v3_constellation(); v4_ridgeline(); v5_separation(); v6_margin(); v7_chord()
    print("voice figures done")
