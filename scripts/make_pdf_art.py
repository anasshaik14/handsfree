"""Assemble the 16 signature figures into one dark PDF: reports/HandsFree_Signature_Figures.pdf"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle2 as S
S.apply()

ART = ROOT / "reports" / "figures_art"
PAGE = (11.69, 8.27)
ORDER = [
    ("gaze", "Gaze", ["GA1_comet_field", "GA2_aurora", "GA3_nebula", "GA4_error_islands"]),
    ("headpose", "Head pose", ["HA1_atlas", "HA2_butterfly", "HA3_silk", "HA4_honeycomb"]),
    ("voice", "Voice", ["VA1_chord", "VA2_sunflower", "VA3_starmap", "VA4_aurora"]),
    ("blink", "Blink", ["BA1_delta_stream", "BA2_galaxy", "BA3_threshold_aurora", "BA4_certainty_arc"]),
]


def cover(pdf):
    fig = plt.figure(figsize=PAGE); fig.patch.set_facecolor(S.BG)
    fig.text(0.5, 0.66, "HandsFree", fontsize=52, fontweight="bold", color=S.INK, ha="center")
    fig.text(0.5, 0.565, "Signature Figures", fontsize=27, color=S.CUSTOM, ha="center")
    fig.text(0.5, 0.47, "Sixteen visual studies of the custom-vs-pretrained comparison —\n"
                        "four for each input: gaze, head pose, voice and blink.",
             fontsize=14, color=S.INK2, ha="center")
    fig.text(0.5, 0.30, "Comet fields · nebulae · error islands · butterflies · honeycombs ·\n"
                        "confusion constellations · sunflowers · star-maps · galaxies.",
             fontsize=12, color=S.PRETRAIN, ha="center", style="italic")
    fig.text(0.5, 0.12, "Every figure is drawn from the real matched per-sample predictions.",
             fontsize=10.5, color=S.MUTED, ha="center")
    pdf.savefig(fig, facecolor=S.BG); plt.close(fig)


def img_page(pdf, key, stem, tag):
    img = plt.imread(str(ART / key / f"{stem}.png"))
    fig = plt.figure(figsize=PAGE); fig.patch.set_facecolor(S.BG)
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.92]); ax.axis("off")
    ax.imshow(img)
    fig.text(0.975, 0.02, tag, ha="right", fontsize=8, color=S.MUTED)
    pdf.savefig(fig, facecolor=S.BG, dpi=200); plt.close(fig)


def main():
    out = ROOT / "reports" / "HandsFree_Signature_Figures.pdf"
    with PdfPages(out) as pdf:
        cover(pdf)
        for key, name, stems in ORDER:
            for i, stem in enumerate(stems, 1):
                img_page(pdf, key, stem, f"{name} · {i}/4")
        d = pdf.infodict(); d["Title"] = "HandsFree — Signature Figures (16)"
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
