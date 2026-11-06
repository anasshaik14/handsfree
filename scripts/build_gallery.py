"""Assemble the 28 comparison figures into one local HTML gallery."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"

CAPS = {
 "gaze": ("Gaze  ·  custom CNN vs pretrained MPIIGaze ResNet",
   "Matched per-sample on the custom model's HELD-OUT participants (p05, p07) only — the full 15-participant set contains nine of its own training people and flatters it by ~1.9°. Each model gets its own preprocessing. The pretrained baseline was trained on MPIIGaze and likely saw these participants, so the comparison is conservative towards ours.",
   [("G1_error_sky.png","Error sky","Polar heatmap of error by gaze direction × eccentricity — custom glows brighter."),
    ("G2_displacement_field.png","Displacement field","Arrows from true gaze to average prediction; longer = stronger systematic bias."),
    ("G3_tolerance_sunburst.png","Tolerance sunburst","Nested rings: share of predictions within 5/10/15° for each model."),
    ("G4_win_skyline.png","Head-to-head skyline","Every sample sorted by the per-sample error gap — who wins, and by how much."),
    ("G5_bland_altman.png","Bland–Altman density","Bias and 95% limits of agreement per axis, over a density field."),
    ("G6_eccentricity_ridgeline.png","Error vs eccentricity","Error distribution split by how far off-centre the gaze is."),
    ("G7_true_vs_pred.png","Predicted vs true","Density fit with R² for pitch and yaw.")]),
 "headpose": ("Head pose  ·  custom CNN vs pretrained 6DRepNet",
   "Matched per-sample on AFLW2000 (the standard test for 300W-LP-trained models). Same crop scored by both models — a true NN-vs-NN comparison.",
   [("H1_signed_violins.png","Signed-error violins","Per-axis bias and spread; custom left, 6DRepNet right of each spine."),
    ("H2_fingerprint.png","Error fingerprint","Heat-grid of four error measures across all three axes, per model."),
    ("H3_pose_heatmap.png","Pose-sphere heatmap","Where in yaw×pitch space each model struggles."),
    ("H4_win_skyline.png","Head-to-head skyline","Every face sorted by its error gap."),
    ("H5_yaw_ridgeline.png","Error vs profile","Error distribution per yaw band, frontal → full profile."),
    ("H6_true_vs_pred.png","Predicted vs true","Density fit with R² for yaw, pitch, roll."),
    ("H7_win_map.png","Pose battleground","Each face placed by true angle, coloured by which model wins.")]),
 "voice": ("Voice  ·  custom keyword CNN vs pretrained Vosk",
   "Custom CNN scored per-sample on the Speech Commands test split. Vosk gives a hard transcript (no probabilities), so it enters as summary metrics — the metric rose (V2) is the direct comparison; the rest are the custom model's per-sample audit.",
   [("V1_confusion.png","Confusion matrix","Row-normalised — where each spoken word's clips were sent."),
    ("V2_metric_rose.png","Metric rose (vs Vosk)","Seven summary scores as paired petals, custom vs pretrained Vosk."),
    ("V3_constellation.png","Calibration constellation","Confidence bins as bubbles — is the model's confidence honest?"),
    ("V4_perword_ridgeline.png","Per-word confidence","Which keywords are easy vs hard."),
    ("V5_separation.png","Separation densities","Per-keyword: score on target word (up) vs the rest (down); AUC shown."),
    ("V6_decision_margin.png","Decision margin","Wrong answers cluster at low margins — hesitation flags mistakes."),
    ("V7_word_flow.png","Word flow","Every clip streaming true→predicted; cross-overs are the confusions.")]),
 "blink": ("Blink  ·  custom CNN deep-dive",
   "The blink alternative in the app is a classical Eye-Aspect-Ratio maths rule (not a neural net) and cannot be scored per-sample on the MRL eye-crop test set. So blink is presented as a rigorous per-sample audit of the custom CNN rather than a NN-vs-NN race.",
   [("B1_swarm.png","Separation swarm","Every test eye as a dot by P(closed); the two clouds pull to opposite ends."),
    ("B2_mosaic.png","Confusion mosaic","Proportional territory — column width = class prevalence, solid = correct."),
    ("B3_constellation.png","Calibration constellation","Confidence bins as bubbles on the honesty diagonal."),
    ("B4_streamgraph.png","Outcome streamgraph","The four outcomes trading off as the click threshold slides 0→1."),
    ("B5_confusion_flow.png","Confusion as a river","Samples flowing true→predicted; thin cross-ribbons are mistakes."),
    ("B6_score_separation.png","Score separation","How cleanly the P(closed) score splits the two classes."),
    ("B7_margin_ridge.png","Decision certainty","Certainty of correct vs wrong predictions — does the model hesitate on mistakes?")]),
}

css = """
*{box-sizing:border-box} body{margin:0;background:#f9f9f7;color:#0b0b0b;
font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:48px 24px 90px}
h1{font-size:2.3rem;margin:0 0 6px;letter-spacing:-.01em}
.lede{color:#52514e;font-size:1.08rem;margin:0 0 8px;max-width:760px}
.note{background:#fff;border:1px solid #e5e8ee;border-left:3px solid #eb6834;border-radius:8px;
padding:12px 16px;margin:22px 0 8px;color:#52514e;font-size:.92rem;max-width:900px}
section{margin-top:56px}
h2{font-size:1.5rem;margin:0 0 4px}
.sub{color:#52514e;margin:0 0 6px;max-width:900px}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:26px;margin-top:22px}
.card{background:#fff;border:1px solid #e5e8ee;border-radius:12px;overflow:hidden}
.card img{width:100%;display:block;border-bottom:1px solid #eef0f4}
.cap{padding:12px 16px 16px}
.cap b{font-size:1rem}.cap p{margin:4px 0 0;color:#52514e;font-size:.88rem}
footer{margin-top:64px;padding-top:20px;border-top:1px solid #e5e8ee;color:#898781;font-size:.85rem}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
"""

def main():
    h = ['<!doctype html><meta charset="utf-8"><title>HandsFree — Custom vs Pretrained</title>',
         f"<style>{css}</style>", '<div class="wrap">',
         "<h1>Custom&nbsp;CNNs vs Pretrained&nbsp;Models</h1>",
         '<p class="lede">A visual, per-sample comparison across all four HandsFree inputs — no bar charts, '
         'no pie charts, no line graphs. Every model is scored on the same held-out data with the same '
         'metrics; each figure is drawn from real per-sample predictions.</p>',
         '<div class="note"><b>How to read the comparison honestly.</b> Gaze and head pose are true '
         'neural-net-vs-neural-net comparisons on matched samples. Voice compares the custom CNN against '
         'pretrained Vosk (summary metrics only). Blink’s alternative is a classical maths rule, not a neural '
         'net, so it is shown as a deep audit of the custom model.</div>',
         '<div class="note" style="border-left-color:#2a78d6"><b>Work in progress.</b> I am still '
         'refining the hyperparameter tuning for the gaze and head-pose custom models to push their '
         'accuracy further, with the goal of making them as competitive with the pretrained models as '
         'the custom blink and voice models already are.</div>']
    for key in ("gaze", "headpose", "voice", "blink"):
        title, sub, items = CAPS[key]
        h.append(f'<section><h2>{title}</h2><p class="sub">{sub}</p><div class="grid">')
        for fn, t, c in items:
            h.append(f'<div class="card"><img src="{key}/{fn}" alt="{t}">'
                     f'<div class="cap"><b>{t}</b><p>{c}</p></div></div>')
        h.append("</div></section>")
    h.append('<footer>HandsFree MSc project · 28 figures (7 per input) · generated from matched per-sample '
             'predictions in reports/tuning/pretrained/matched_*.npz</footer></div>')
    out = FIG / "index.html"
    out.write_text("\n".join(h), encoding="utf-8")
    print("wrote", out)

if __name__ == "__main__":
    main()
