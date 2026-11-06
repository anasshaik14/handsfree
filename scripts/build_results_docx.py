"""Assemble the results section + all 28 comparison figures into one .docx.

Reads reports/RESULTS_SECTION.md (markdown: headings, paragraphs, **bold**,
bullet/numbered lists, pipe tables) and the figure captions used by the HTML
gallery, and writes reports/HandsFree_Results.docx.

Usage:  python scripts/build_results_docx.py [out.docx] [in.md] [--no-appendix]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_gallery import CAPS  # reuse the gallery's figure captions verbatim
from refs import BIB          # bibliography for [@key] citations

MD = ROOT / "reports" / "RESULTS_SECTION.md"
FIG = ROOT / "reports" / "figures"
ART = FIG / "figures_art"
ORDER = ["gaze", "headpose", "blink", "voice"]

# Titles/blurbs copied from the S.title(...) calls in fig_art.py so the two agree.
ART_CAPS = {
 "gaze": ("Gaze", [
   ("GA1_comet_field.png", "Comet field", "Each comet streaks from the true gaze to the model's prediction — length is the error."),
   ("GA2_aurora.png", "Aurora of error", "Angular-error distributions as luminous curtains; the custom curtain sits further right."),
   ("GA3_nebula.png", "Gaze nebula", "Every gaze a mote — angle = direction, radius = eccentricity, brightness = error."),
   ("GA4_error_islands.png", "Error islands", "Contours of where predictions land relative to truth; a tight island on the cross is accurate and unbiased.")]),
 "headpose": ("Head pose", [
   ("HA1_atlas.png", "Atlas of poses", "Every test face charted by true head angle and lit by which model wins it."),
   ("HA2_butterfly.png", "Butterfly of drift", "Each axis a pair of wings, custom above and pretrained below; slender wings pinned on zero are unbiased."),
   ("HA3_silk.png", "Silk streams", "Error flowing as the head turns from profile to frontal; the custom stream swells at the extremes."),
   ("HA4_honeycomb.png", "Honeycomb of error", "The pose plane tiled into cells glowing by the error inside them.")]),
 "blink": ("Blink", [
   ("BA1_delta_stream.png", "The delta of decision", "Two facing rivers of score: closed eyes flow up, open eyes down, parting at the threshold."),
   ("BA2_galaxy.png", "Blink galaxy", "12,927 eyes spun onto a spiral, ordered by score and coloured by true state."),
   ("BA3_threshold_aurora.png", "Threshold aurora", "Outcome bands shifting as the click threshold glides 0 to 1."),
   ("BA4_certainty_arc.png", "Arc of certainty", "Each eye placed by score; mistakes cluster near the middle, where the model was unsure.")]),
 "voice": ("Voice", [
   ("VA1_chord.png", "Confusion constellation", "Words as stars sized by frequency; arcs are the mis-hearings flowing between them."),
   ("VA2_sunflower.png", "Confidence sunflower", "Every clip a seed — colour = predicted word, size = confidence; ringed seeds are mistakes."),
   ("VA3_starmap.png", "Word star-map", "Each clip's 6-way probability projected to 2-D; tight islands mean confident classes."),
   ("VA4_aurora.png", "Aurora of the keywords", "Per word, the glow of confidence in the right answer.")]),
}

ATTR = FIG / "attribution"
ATTR_MODS = [("blink", "Blink"), ("gaze", "Gaze"),
             ("headpose", "Head pose"), ("voice", "Voice")]
ATTR_VIEWS = [
    ("A1_summary.png", "Global attribution summary",
     "Input regions ranked by mean absolute attribution, with the per-sample distribution "
     "for each region. Left: the mean input overlaid with attribution contours, the three "
     "strongest regions outlined."),
    ("A2_waterfall.png", "Waterfall attribution for two instances",
     "Cumulative region contributions carrying the network from its baseline output to the "
     "prediction, for the two most extreme instances. Dark bars push the output up, light "
     "bars push it down."),
    ("A3_similarity.png", "Global explanation, similarity ordering",
     "Stacked region attributions with instances ordered by resemblance of their "
     "explanations, exposing groups of inputs the network treats alike."),
    ("A4_ordering.png", "Global explanation, sample ordering",
     "The same stacked attributions with instances in dataset order, exposing trends in "
     "the distribution of explanations across the evaluation set."),
]

BLACK = RGBColor(0x00, 0x00, 0x00)   # plain black-and-white throughout: no accent colours


# --------------------------------------------------------------------- helpers
def add_runs(par, text: str) -> None:
    """Write text into a paragraph, honouring **bold**, *italic* and `code`."""
    for part in re.split(r"(\*\*.+?\*\*|(?<!\*)\*[^*]+?\*(?!\*)|`[^`]+?`)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
        elif part.startswith("*") and part.endswith("*"):
            par.add_run(part[1:-1]).italic = True
        else:
            par.add_run(part)


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_sep(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s:|-]+\|", line.strip()))


def add_table(doc: Document, rows: list[list[str]]) -> None:
    """Render a markdown pipe table as a real Word table."""
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Table Grid"   # plain black borders, no colour fill
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = t.cell(i, j)
            cell.text = ""
            par = cell.paragraphs[0]
            txt = row[j] if j < len(row) else ""
            add_runs(par, txt)
            for r in par.runs:
                r.font.size = Pt(9)
                if i == 0:
                    r.bold = True
    doc.add_paragraph()


def caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = BLACK


# ---------------------------------------------------------------- citations
def resolve_citations(md: str) -> tuple[str, list[str]]:
    """Replace [@key] with IEEE numbers assigned in order of first appearance.

    Adjacent citations collapse: [@a][@b] -> [1], [2].  Unknown keys are left
    alone and reported, so a typo is visible rather than silently dropped.
    """
    order: list[str] = []
    missing: set[str] = set()

    def num(m):
        key = m.group(1)
        if key not in BIB:
            missing.add(key)
            return m.group(0)
        if key not in order:
            order.append(key)
        return f"[{order.index(key) + 1}]"

    out = re.sub(r"\[@([A-Za-z0-9_]+)\]", num, md)
    out = re.sub(r"\]\[(?=\d)", "], [", out)         # [1][2] -> [1], [2]
    for k in sorted(missing):
        print(f"  ! unknown citation key: {k}")
    return out, [f"[{i}] {BIB[k]}" for i, k in enumerate(order, 1)]


# ------------------------------------------------------------------ md -> docx
def render_markdown(doc: Document, md: str) -> None:
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()

        if not s:
            i += 1
            continue

        # table block
        if s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not is_sep(lines[i]):
                    block.append(split_row(lines[i]))
                i += 1
            if block:
                add_table(doc, block)
            continue

        # inline figure:  !FIG <path under reports/figures> | <caption>
        if s.startswith("!FIG"):
            rel, _, cap = s[4:].partition("|")
            path = FIG / rel.strip()
            if path.exists():
                doc.add_picture(str(path), width=Inches(6.3))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                caption(doc, cap.strip())
            else:
                print(f"  ! missing figure: {path}")
            i += 1
            continue

        # headings
        m = re.match(r"^(#{1,4})\s+(.*)", s)
        if m:
            level = len(m.group(1))
            text = m.group(2)
            if level == 1:
                h = doc.add_heading(level=1)
                add_runs(h, text)
            else:
                h = doc.add_heading(level=min(level, 4))
                add_runs(h, text)
            i += 1
            continue

        # bullet / numbered list
        if re.match(r"^[-*]\s+", s):
            p = doc.add_paragraph(style="List Bullet")
            add_runs(p, re.sub(r"^[-*]\s+", "", s))
            i += 1
            continue
        if re.match(r"^\d+\.\s+", s):
            p = doc.add_paragraph(style="List Number")
            add_runs(p, re.sub(r"^\d+\.\s+", "", s))
            i += 1
            continue

        # plain prose: join the hard-wrapped source lines back into one paragraph
        buf = []
        while i < len(lines):
            t = lines[i].strip()
            if not t or t.startswith(("|", "#", "!FIG")) or re.match(r"^([-*]\s+|\d+\.\s+)", t):
                break
            buf.append(t)
            i += 1
        p = doc.add_paragraph()
        add_runs(p, " ".join(buf))


# --------------------------------------------------------------------- figures
def add_figures(doc: Document) -> int:
    doc.add_page_break()
    doc.add_heading("Appendix A — Comparison Figures", level=1)
    p = doc.add_paragraph()
    add_runs(p, "All figures are generated from the matched per-sample arrays; "
                "custom and comparison models are scored on identical samples "
                "with each model receiving its own native preprocessing.")

    n = 0
    for key in ORDER:
        if key not in CAPS:
            continue
        title, blurb, items = CAPS[key]
        doc.add_heading(title, level=2)
        p = doc.add_paragraph()
        r = p.add_run(blurb)
        r.italic = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = BLACK

        for fname, short, desc in items:
            path = FIG / key / fname
            if not path.exists():
                continue
            doc.add_picture(str(path), width=Inches(6.3))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            n += 1
            caption(doc, f"Figure A{n}. {short} — {desc}")
        doc.add_page_break()
    return n


def add_attribution_figures(doc: Document) -> int:
    """Appendix B - explainability, four views per input."""
    doc.add_heading("Appendix B — Explainability Figures", level=1)
    p = doc.add_paragraph()
    add_runs(p, "Attribution is computed by Integrated Gradients against a zero baseline "
                "(32 steps) directly on the deployed TorchScript modules, so the explanation "
                "describes the network that runs. Integrated Gradients is the Aumann–Shapley "
                "attribution and satisfies completeness, so region contributions sum to the "
                "change in model output between baseline and input; this is what makes the "
                "waterfall and stacked plots exact rather than indicative. Because these "
                "networks consume images and spectrograms rather than named columns, the "
                "input is divided into a 4×4 grid and each region takes the role of a feature.")
    n = 0
    for key, nice in ATTR_MODS:
        doc.add_heading(nice, level=2)
        for fname, short, desc in ATTR_VIEWS:
            path = ATTR / key / fname
            if not path.exists():
                continue
            doc.add_picture(str(path), width=Inches(6.3))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            n += 1
            caption(doc, f"Figure B{n}. {nice} — {short}. {desc}")
        doc.add_page_break()
    return n


def add_art_figures(doc: Document, start: int) -> int:
    """Appendix B - the 16 signature/presentation renderings of the same data."""
    doc.add_heading("Appendix C — Signature Visualisations", level=1)
    p = doc.add_paragraph()
    add_runs(p, "Alternative renderings of the same matched per-sample data, intended "
                "for presentation. They carry no metric beyond Appendix A; gaze figures "
                "use the held-out participants only, as elsewhere.")
    n = start
    for key in ORDER:
        if key not in ART_CAPS:
            continue
        title, items = ART_CAPS[key]
        doc.add_heading(title, level=2)
        for fname, short, desc in items:
            path = ART / key / fname
            if not path.exists():
                continue
            doc.add_picture(str(path), width=Inches(6.3))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            n += 1
            caption(doc, f"Figure C{n - start}. {short} — {desc}")
        doc.add_page_break()
    return n - start


# ------------------------------------------------------------------------ main
def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    out = Path(args[0]) if args else ROOT / "reports" / "HandsFree_Results.docx"
    md = Path(args[1]) if len(args) > 1 else MD
    doc = Document()

    # base styles -- serif body, black only (journal manuscript convention)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.font.color.rgb = BLACK
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    # Word's built-in Heading styles are blue and italic by default; force them to
    # plain black so the manuscript prints correctly in greyscale.
    for lvl in range(1, 5):
        try:
            st = doc.styles[f"Heading {lvl}"]
        except KeyError:
            continue
        st.font.name = "Times New Roman"
        st.font.color.rgb = BLACK
        st.font.bold = True
        st.font.italic = False
        st.font.size = Pt({1: 14, 2: 12, 3: 11, 4: 11}[lvl])

    # title block
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("HandsFree: Compact Task-Specific CNNs for\nReal-Time Multimodal Hands-Free Interaction")
    r.bold = True
    r.font.size = Pt(19)
    r.font.color.rgb = BLACK

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("Results and Evaluation")
    r.font.size = Pt(12.5)
    r.font.color.rgb = BLACK

    doc.add_paragraph()

    body, references = resolve_citations(md.read_text(encoding="utf-8"))
    render_markdown(doc, body)
    if references:
        doc.add_heading("References", level=1)
        for ref in references:
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Inches(0.35)
            par.paragraph_format.first_line_indent = Inches(-0.35)
            add_runs(par, ref)
    if "--no-appendix" in flags:      # figures are already inline in the markdown
        nfig = nattr = nart = 0
    else:
        nfig = add_figures(doc)
        doc.add_page_break()
        nattr = add_attribution_figures(doc)
        nart = add_art_figures(doc, nfig)

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    print(f"wrote {out}  ({nfig} comparison + {nattr} explainability + "
          f"{nart} signature = {nfig + nattr + nart} figures)")


if __name__ == "__main__":
    main()
