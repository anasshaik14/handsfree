"""Assemble all 28 comparison figures + aligned metric tables into ONE PDF:
reports/HandsFree_Comparison_Figures.pdf
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle as F
import build_metrics_table as BM
F.apply_style()

FIG = ROOT / "reports" / "figures"
data = json.loads((ROOT / "reports" / "metrics_aligned.json").read_text(encoding="utf-8"))
PAGE = (11.69, 8.27)   # landscape A4

INPUTS = [
    ("gaze", "Gaze", "Custom CNN  vs  pretrained MPIIGaze ResNet",
     "A true neural-net-vs-neural-net comparison: both models scored on the SAME 30,000 normalized eyes."),
    ("headpose", "Head pose", "Custom CNN  vs  pretrained 6DRepNet",
     "Matched per-sample on AFLW2000 (the standard test for 300W-LP models). Same crop scored by both."),
    ("voice", "Voice", "Custom keyword CNN  vs  pretrained Vosk",
     "Custom scored per-sample; Vosk gives a hard transcript, so it enters via summary metrics."),
    ("blink", "Blink", "Custom CNN — deep per-sample audit",
     "The app's blink alternative is a classical maths rule (not a neural net), so blink is a custom-model audit."),
]
FIGS = {k: [t[0] for t in v[2]] for k, v in BM.__dict__.get("_", {}).items()} if False else None
# figure filenames per input (in gallery order)
ORDER = {
 "gaze": ["G1_error_sky", "G2_displacement_field", "G3_tolerance_sunburst", "G4_win_skyline",
          "G5_bland_altman", "G6_eccentricity_ridgeline", "G7_true_vs_pred"],
 "headpose": ["H1_signed_violins", "H2_fingerprint", "H3_pose_heatmap", "H4_win_skyline",
              "H5_yaw_ridgeline", "H6_true_vs_pred", "H7_win_map"],
 "voice": ["V1_confusion", "V2_metric_rose", "V3_constellation", "V4_perword_ridgeline",
           "V5_separation", "V6_decision_margin", "V7_word_flow"],
 "blink": ["B1_swarm", "B2_mosaic", "B3_constellation", "B4_streamgraph", "B5_confusion_flow",
           "B6_score_separation", "B7_margin_ridge"],
}


def cover(pdf):
    fig = plt.figure(figsize=PAGE); fig.patch.set_facecolor(F.SURFACE)
    fig.text(0.06, 0.72, "HandsFree", fontsize=44, fontweight="bold", color=F.INK)
    fig.text(0.06, 0.63, "Custom CNNs vs Pretrained Models", fontsize=24, color=F.CUSTOM_D)
    fig.text(0.06, 0.55, "A per-sample visual comparison across all four inputs —\n"
                          "gaze, head pose, voice and blink.", fontsize=14, color=F.INK2)
    fig.text(0.06, 0.40, "28 figures (7 per input) + full aligned metric tables.\n"
                          "Every figure is drawn from real per-sample predictions; the pretrained\n"
                          "models are scored with the identical metric functions used for the custom models.",
             fontsize=11.5, color=F.INK2)
    fig.text(0.06, 0.20, "Work in progress: I am still refining the hyperparameter tuning for the gaze\n"
                          "and head-pose custom models to push their accuracy further, aiming to make them\n"
                          "as competitive with the pretrained models as the custom blink and voice models already are.",
             fontsize=11, color=F.PRETRAIN_D, style="italic")
    fig.text(0.06, 0.08, "HandsFree MSc project", fontsize=10, color=F.MUTED)
    pdf.savefig(fig); plt.close(fig)


def divider(pdf, name, sub, note):
    fig = plt.figure(figsize=PAGE); fig.patch.set_facecolor(F.SURFACE)
    fig.add_axes([0, 0, 0.02, 1]).axis("off")
    fig.patches.append(plt.Rectangle((0, 0), 0.025, 1, transform=fig.transFigure, color=F.CUSTOM))
    fig.text(0.08, 0.6, name, fontsize=40, fontweight="bold", color=F.INK)
    fig.text(0.08, 0.5, sub, fontsize=18, color=F.PRETRAIN_D)
    fig.text(0.08, 0.4, note, fontsize=12, color=F.INK2, wrap=True)
    pdf.savefig(fig); plt.close(fig)


def table_page(pdf, key):
    e = data[key]
    rows = BM.rows_for(e)
    fig = plt.figure(figsize=PAGE); fig.patch.set_facecolor(F.SURFACE)
    F.title(fig, f"{key.capitalize()} — full metric suite (identical formulas)",
            f"{e['custom_name']}  vs  {e['pretrained_name']}   ·   N = {e['N']:,}")
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.76]); ax.axis("off")
    cell = [[lbl, cv, pv] for lbl, cv, pv in rows]
    tbl = ax.table(cellText=cell, colLabels=["Metric", e["custom_name"], e["pretrained_name"]],
                   cellLoc="left", colLoc="center", loc="upper center",
                   colWidths=[0.42, 0.29, 0.29])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.2); tbl.scale(1, 1.28)
    for (r, c), cellobj in tbl.get_celld().items():
        cellobj.set_edgecolor("#e5e8ee")
        if r == 0:
            cellobj.set_facecolor("#2a78d6"); cellobj.set_text_props(color="white", fontweight="bold")
        elif cell[r - 1][0].startswith("—"):
            cellobj.set_facecolor("#f4f5f7"); cellobj.set_text_props(color="#999999", style="italic")
        elif r % 2 == 0:
            cellobj.set_facecolor("#fbfbfa")
        if c == 1 and r > 0 and not cell[r - 1][0].startswith("—"):
            cellobj.set_text_props(color=F.CUSTOM_D, fontweight="bold")
        if c == 2 and r > 0 and not cell[r - 1][0].startswith("—"):
            cellobj.set_text_props(color=F.PRETRAIN_D)
    F.caption(fig, e["note"])
    pdf.savefig(fig); plt.close(fig)


def image_page(pdf, key, stem, idx, total):
    path = FIG / key / f"{stem}.png"
    img = plt.imread(str(path))
    fig = plt.figure(figsize=PAGE); fig.patch.set_facecolor(F.SURFACE)
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.9]); ax.axis("off")
    ax.imshow(img)
    fig.text(0.97, 0.02, f"{key} · {idx}/{total}", ha="right", fontsize=8, color=F.MUTED)
    pdf.savefig(fig, dpi=200); plt.close(fig)


def main():
    out = ROOT / "reports" / "HandsFree_Comparison_Figures.pdf"
    with PdfPages(out) as pdf:
        cover(pdf)
        # figures only — section dividers and metric-table pages removed per request
        for key, name, sub, note in INPUTS:
            stems = ORDER[key]
            for i, stem in enumerate(stems, 1):
                image_page(pdf, key, stem, i, len(stems))
        d = pdf.infodict(); d["Title"] = "HandsFree — Custom vs Pretrained (28 figures + metrics)"
    size = out.stat().st_size / 1e6
    print(f"wrote {out}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
