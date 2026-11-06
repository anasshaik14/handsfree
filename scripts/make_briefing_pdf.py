"""Supervisor briefing: every figure (old 28 + new 16) as a thumbnail + a plain-English
explanation. Output: reports/HandsFree_Figure_Briefing.pdf"""
from __future__ import annotations
import sys, textwrap
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PAGE = (8.27, 11.69)   # portrait A4
W, H = PAGE
INK = "#14181f"; INK2 = "#4a5261"; MUTED = "#8a93a3"; LINE = "#e3e7ee"; BG = "#ffffff"
BLUE = "#2a78d6"; ORANGE = "#e0662f"

CLR = ROOT / "reports" / "figures"
ART = ROOT / "reports" / "figures_art"

# (image_path, title, set_tag, explanation)
def C(k, f):  # clear
    return CLR / k / f
def A(k, f):  # signature
    return ART / k / f

CONTENT = {
 "Gaze — where the eyes look": (
   "My model is off by about 5.3°, the pretrained by about 2.8°. A fair test on the same 30,000 eyes.", [
   (C("gaze","G1_error_sky.png"),"Error sky","Clear","A round colour-map of error by look-direction and distance from centre. Darker = more error; my model is darker, worst at the edges."),
   (C("gaze","G2_displacement_field.png"),"Displacement field","Clear","Arrows from the true gaze to the model's average guess. Longer arrows = bigger, systematic error; mine are longer."),
   (C("gaze","G3_tolerance_sunburst.png"),"Tolerance sunburst","Clear","Two rings showing how many gazes land within 5/10/15°. Pretrained hits 88% within 5°, mine 55%."),
   (C("gaze","G4_win_skyline.png"),"Win skyline","Clear","Every gaze sorted by which model guessed it better. Orange (mine) wins about 1 in 5; blue wins the rest."),
   (C("gaze","G5_bland_altman.png"),"Bland–Altman","Clear","Checks for a constant lean (always guessing a bit high or low). A band sitting on zero means no systematic bias."),
   (C("gaze","G6_eccentricity_ridgeline.png"),"Eccentricity ridgeline","Clear","Error split by how far the gaze is from the screen centre. Both get worse toward the edges; mine a little faster."),
   (C("gaze","G7_true_vs_pred.png"),"Predicted vs true","Clear","The predicted angle against the true angle. The tighter the cloud sits on the line (higher R²), the more accurate."),
   (A("gaze","GA1_comet_field.png"),"Comet field","Signature","Each 'comet' streaks from the true gaze to the model's guess. Short comets = good guesses."),
   (A("gaze","GA2_aurora.png"),"Aurora of error","Signature","The spread of errors drawn as glowing curtains. Mine glows wider and further right = larger errors."),
   (A("gaze","GA3_nebula.png"),"Gaze nebula","Signature","Every gaze is a glowing dot, brighter = more wrong. My cloud burns hotter, especially at the rim."),
   (A("gaze","GA4_error_islands.png"),"Error islands","Signature","A contour 'island' of where guesses land vs truth. A small island on the centre cross = accurate and unbiased."),
 ]),
 "Head pose — which way the head turns": (
   "My model is off by about 7.4°, the pretrained by about 3.5°. Same faces scored by both.", [
   (C("headpose","H1_signed_violins.png"),"Signed violins","Clear","The error shape for turn, nod and tilt. Narrow and centred = precise and unbiased; mine is wider."),
   (C("headpose","H2_fingerprint.png"),"Fingerprint","Clear","A number grid of four error measures across the three axes. My side runs warmer (higher error)."),
   (C("headpose","H3_pose_heatmap.png"),"Pose heatmap","Clear","Where in head-angle space each model struggles. Dark = hard poses; mine lights up at extreme angles."),
   (C("headpose","H4_win_skyline.png"),"Win skyline","Clear","Every face sorted by which model did better — same idea as the gaze skyline."),
   (C("headpose","H5_yaw_ridgeline.png"),"Yaw ridgeline","Clear","Error as the head turns from facing forward to full profile. Mine grows more toward the profile."),
   (C("headpose","H6_true_vs_pred.png"),"Predicted vs true","Clear","Guess against truth for all three axes, with R² for the fit."),
   (C("headpose","H7_win_map.png"),"Win map","Clear","Every face placed by its true angle, coloured by who won. Pretrained owns the easy middle; my wins are at the hard extremes."),
   (A("headpose","HA1_atlas.png"),"Atlas of poses","Signature","A glowing map of every face by its angle, lit ember (I win) vs cyan (pretrained wins)."),
   (A("headpose","HA2_butterfly.png"),"Butterfly of drift","Signature","Each axis as two glowing wings (mine on top, pretrained below). Slim, symmetric wings on zero = precise and unbiased."),
   (A("headpose","HA3_silk.png"),"Silk streams","Signature","Error 'flowing' as the head turns; my stream swells at the side-profile extremes."),
   (A("headpose","HA4_honeycomb.png"),"Honeycomb","Signature","The pose map tiled into glowing cells; bright edge-cells are the hard extreme poses."),
 ]),
 "Voice — spoken commands": (
   "My model is right 94.5% of the time, Vosk 95.5% — very close.", [
   (C("voice","V1_confusion.png"),"Confusion matrix","Clear","A grid of word-said vs word-heard. A strong diagonal = few mix-ups."),
   (C("voice","V2_metric_rose.png"),"Metric rose","Clear","Seven scores as petals, mine vs Vosk. The direct head-to-head; Vosk edges me slightly."),
   (C("voice","V3_constellation.png"),"Calibration bubbles","Clear","Checks if the model's confidence is honest. Bubbles on the diagonal = trustworthy."),
   (C("voice","V4_perword_ridgeline.png"),"Per-word confidence","Clear","For each word, how confident it was in the right answer. Piled near 1.0 = an easy word."),
   (C("voice","V5_separation.png"),"Separation densities","Clear","For each word, its score when it IS that word (up) vs everything else (down). A clean split = high AUC (0.99–1.0)."),
   (C("voice","V6_decision_margin.png"),"Decision margin","Clear","The gap between the top guess and the runner-up. Wrong answers bunch at small gaps."),
   (C("voice","V7_word_flow.png"),"Word flow","Clear","Clips streaming from the true word to the predicted word; the thin cross-overs are the confusions."),
   (A("voice","VA1_chord.png"),"Confusion constellation","Signature","Words as glowing stars; faint arcs are the mis-hearings between them. Few, faint arcs = a clean recogniser."),
   (A("voice","VA2_sunflower.png"),"Confidence sunflower","Signature","Every clip is a seed: colour = the word, brightness = confidence, white ring = a mistake (at the low-confidence edge)."),
   (A("voice","VA3_starmap.png"),"Word star-map","Signature","Each clip's 6-way guess squeezed into a 2-D sky; same-word clips form their own cluster. Tight clusters = confident."),
   (A("voice","VA4_aurora.png"),"Keyword aurora","Signature","Confidence for each word as glowing curtains; curtains at 1.0 are the easy words."),
 ]),
 "Blink — eye open vs closed (the click)": (
   "My model is right about 93.7% of the time. Its only rival is a maths rule, so these study my model.", [
   (C("blink","B1_swarm.png"),"Separation swarm","Clear","Every eye is a dot by its 'closed' score; open and closed pull to opposite ends. Dots in the middle are hard cases."),
   (C("blink","B2_mosaic.png"),"Confusion mosaic","Clear","Two blocks sized by how common open/closed are; solid = correct, hatched = missed."),
   (C("blink","B3_constellation.png"),"Calibration bubbles","Clear","Checks if the confidence is honest. Bubbles on the diagonal = trustworthy."),
   (C("blink","B4_streamgraph.png"),"Outcome streamgraph","Clear","Sliding the click-threshold, coloured bands show right/wrong outcomes. Error bands stay thin over a wide safe range."),
   (C("blink","B5_confusion_flow.png"),"Confusion river","Clear","Eyes flowing from true state to prediction; thin cross-ribbons are the mistakes."),
   (C("blink","B6_score_separation.png"),"Score separation","Clear","The 'closed' score for truly-open vs truly-closed eyes. Little overlap = a decisive model."),
   (C("blink","B7_margin_ridge.png"),"Decision certainty","Clear","How sure it was on right vs wrong answers. Wrong ones lean unsure — it half-warns itself."),
   (A("blink","BA1_delta_stream.png"),"Delta stream","Signature","Two facing glowing rivers: closed eyes flow up, open flow down, parting at the threshold."),
   (A("blink","BA2_galaxy.png"),"Blink galaxy","Signature","12,927 eyes spun onto a spiral by score; open settles in the core, closed in the outer ring — they separate on their own."),
   (A("blink","BA3_threshold_aurora.png"),"Threshold aurora","Signature","The threshold trade-off as a glowing light-show; dim error bands stay thin across the safe middle."),
   (A("blink","BA4_certainty_arc.png"),"Arc of certainty","Signature","Each eye on a rainbow arc by its score; red mistakes bunch in the middle where it was unsure."),
 ]),
}


def new_page(pdf, prev):
    if prev is not None:
        pdf.savefig(prev, facecolor=BG); plt.close(prev)
    fig = plt.figure(figsize=PAGE); fig.patch.set_facecolor(BG)
    return fig


def place_image(fig, path, x_left, y_center, max_fw, max_fh):
    img = plt.imread(str(path))
    ih, iw = img.shape[0], img.shape[1]; aim = iw / ih
    if (max_fw * W) / (max_fh * H) > aim:      # height-limited
        fh = max_fh; fw = aim * fh * H / W
    else:                                       # width-limited
        fw = max_fw; fh = fw * W / (aim * H)
    ax = fig.add_axes([x_left, y_center - fh / 2, fw, fh]); ax.axis("off")
    ax.imshow(img)
    # thin frame
    for s in ax.spines.values():
        s.set_visible(False)


def cover(pdf):
    fig = plt.figure(figsize=PAGE); fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.68, "HandsFree", ha="center", fontsize=40, fontweight="bold", color=INK)
    fig.text(0.5, 0.61, "A Guide to the Comparison Figures", ha="center", fontsize=18, color=ORANGE)
    fig.text(0.5, 0.55, "Plain-English explanations of every graph — for the supervisor meeting", ha="center", fontsize=12, color=INK2)
    lines = [
        "What is being compared:",
        "   • My Custom CNN  (orange / ember)     vs     a big Pretrained model  (blue / cyan)",
        "",
        "Two sets of figures:",
        "   • Clear — easy-to-read charts for the write-up",
        "   • Signature — the striking, visual versions for the paper",
        "",
        "How to read the comparison fairly:",
        "   • Gaze & head pose  — a fair fight, both models on the exact same samples",
        "   • Voice  — my model vs Vosk (Vosk gives only final answers, so one chart is the head-to-head)",
        "   • Blink  — a deep study of my model (its only rival is a maths rule, not a neural net)",
        "",
        "Headline results:",
        "   • Gaze:  5.3°  vs  2.8°          • Head pose:  7.4°  vs  3.5°",
        "   • Voice:  94.5%  vs  95.5%      • Blink:  ~93.7%",
        "   • I am still tuning gaze & head pose to close the gap.",
    ]
    y = 0.45
    for ln in lines:
        fig.text(0.1, y, ln, ha="left", fontsize=11, color=INK if ln.endswith(":") else INK2,
                 fontweight="bold" if ln.endswith(":") else "normal")
        y -= 0.028
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


def main():
    from matplotlib.backends.backend_pdf import PdfPages
    out = ROOT / "reports" / "HandsFree_Figure_Briefing.pdf"
    rows_per_page = 3
    with PdfPages(out) as pdf:
        cover(pdf)
        for section, (headline, items) in CONTENT.items():
            for pi in range(0, len(items), rows_per_page):
                chunk = items[pi:pi + rows_per_page]
                fig = plt.figure(figsize=PAGE); fig.patch.set_facecolor(BG)
                first = pi == 0
                fig.text(0.06, 0.955, section + ("" if first else "  (continued)"),
                         fontsize=17, fontweight="bold", color=INK)
                if first:
                    fig.text(0.06, 0.93, headline, fontsize=10.5, color=INK2, style="italic")
                fig.add_artist(plt.Line2D([0.06, 0.94], [0.915, 0.915], color=LINE, lw=1, transform=fig.transFigure))
                top = 0.87
                for r, (path, name, tag, expl) in enumerate(chunk):
                    rc = top - r * 0.275 - 0.11        # row centre
                    place_image(fig, path, 0.06, rc, 0.31, 0.235)
                    tx = 0.42
                    fig.text(tx, rc + 0.075, name, fontsize=13, fontweight="bold", color=INK)
                    tagcol = BLUE if tag == "Clear" else ORANGE
                    fig.text(tx, rc + 0.045, tag.upper(), fontsize=8, fontweight="bold", color=tagcol)
                    wrapped = textwrap.wrap(expl, width=52)
                    for li, wl in enumerate(wrapped):
                        fig.text(tx, rc + 0.012 - li * 0.026, wl, fontsize=10.5, color=INK2)
                pdf.savefig(fig, facecolor=BG); plt.close(fig)
        d = pdf.infodict(); d["Title"] = "HandsFree — Figure Briefing"
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
