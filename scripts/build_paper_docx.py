"""Assemble the MAPLE manuscript parts into one .docx.

Extends the markdown subset understood by build_results_docx.py with the two
constructs a methods-heavy paper needs:

  display equations      $$
                         <math, one or more lines>
                         $$ (17)
      -> centred Cambria Math paragraph, equation number flush right on a tab
         stop, so the numbering column lines up down the whole manuscript.

  algorithm blocks       ```algorithm
                         Algorithm 1 - Title
                         Require: ...
                          1: ...
                         ```
      -> single-cell bordered table, Consolas, leading indentation preserved
         (leading spaces become non-breaking spaces; Word otherwise eats them).

Everything else - headings, **bold**/*italic*/`code`, pipe tables, bullet and
numbered lists, `!FIG path | caption`, and [@key] citations resolved against
scripts/refs.py in IEEE order of first appearance - behaves as in the results
builder.

Usage:  python scripts/build_paper_docx.py [out.docx]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from refs import BIB  # noqa: E402

PARTS_DIR = ROOT / "reports" / "paper"
PARTS = [
    "part1_intro.md",
    "part2_method_a.md",
    "part3_method_b.md",
    "part4_experimental.md",
    "part5a_results_leak_gaze.md",
    "part5b_results_pose_voice.md",
    "part5c_results_blink_synth.md",
    "part6_conclusion.md",
]
FIGDIR = ROOT / "reports" / "figures"

BLACK = RGBColor(0x00, 0x00, 0x00)
BODY_FONT = "Times New Roman"
MATH_FONT = "Cambria Math"
MONO_FONT = "Consolas"

# Usable text width, in inches. Set from the document's own section geometry
# by set_page_geometry(); the module default is only a fallback. python-docx's
# stock template uses 1.25" side margins (6.0" of text), so a hard-coded 6.5
# put every equation number 0.5" into the right margin.
TEXT_WIDTH_IN = 6.5
NBSP = " "

# Title-page byline: (text, point size, italic)
AUTHOR_BLOCK = [
    ("Mohammad Anas Shaik", 12.0, False),
    ("MSc Data Science, York St John University, London, United Kingdom", 10.5, True),
    ("mohammad.shaik@yorksj.ac.uk", 10.5, False),
]


def set_page_geometry(doc: Document, margin_in: float = 1.0) -> float:
    """Pin uniform margins and return the resulting usable text width (inches).

    Equation numbers are placed on a right tab stop at exactly this width, so
    the two must be derived from one source or they drift apart silently.
    """
    for sec in doc.sections:
        sec.left_margin = Inches(margin_in)
        sec.right_margin = Inches(margin_in)
        sec.top_margin = Inches(margin_in)
        sec.bottom_margin = Inches(margin_in)
    sec = doc.sections[0]
    return (sec.page_width - sec.left_margin - sec.right_margin) / 914400


# --------------------------------------------------------------------- inline
def add_runs(par, text: str) -> None:
    """Write text into a paragraph, honouring **bold**, *italic* and `code`."""
    for part in re.split(r"(\*\*.+?\*\*|(?<!\*)\*[^*]+?\*(?!\*)|`[^`]+?`)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = MONO_FONT
            r.font.size = Pt(9.5)
        elif part.startswith("*") and part.endswith("*"):
            par.add_run(part[1:-1]).italic = True
        else:
            par.add_run(part)


# --------------------------------------------------------------------- tables
def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_sep(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s:|-]+\|", line.strip()))


def add_table(doc: Document, rows: list[list[str]], stats: dict | None = None) -> None:
    # A literal "|" inside a cell silently adds columns and shifts every cell
    # after it. Table 7 shipped that way once (O(|v|) in a complexity column),
    # so a ragged block is now reported rather than rendered.
    widths = {len(r) for r in rows}
    if len(widths) > 1 and stats is not None:
        stats["ragged_tables"].append((rows[0][:2], sorted(widths)))
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = t.cell(i, j)
            cell.text = ""
            par = cell.paragraphs[0]
            par.paragraph_format.space_after = Pt(1)
            add_runs(par, row[j] if j < len(row) else "")
            for r in par.runs:
                r.font.size = Pt(8.5)
                if i == 0:
                    r.bold = True
    doc.add_paragraph()


def caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = BLACK


# ------------------------------------------------------------------ equations
def add_equation(doc: Document, lines: list[str], number: str | None) -> None:
    """Centred math lines; the number, if any, sits on a right tab stop."""
    for k, line in enumerate(lines):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(4 if k == 0 else 1)
        pf.space_after = Pt(4 if k == len(lines) - 1 else 1)
        pf.tab_stops.add_tab_stop(Inches(TEXT_WIDTH_IN / 2), WD_TAB_ALIGNMENT.CENTER)
        pf.tab_stops.add_tab_stop(Inches(TEXT_WIDTH_IN), WD_TAB_ALIGNMENT.RIGHT)
        r = p.add_run("\t" + line)
        r.font.name = MATH_FONT
        r.font.size = Pt(10.5)
        r.font.color.rgb = BLACK
        if number and k == len(lines) - 1:
            rn = p.add_run("\t" + number)
            rn.font.name = BODY_FONT
            rn.font.size = Pt(10.5)
            rn.font.color.rgb = BLACK


# ----------------------------------------------------------------- algorithms
def _tight_cell_margins(table, twips: int = 58) -> None:
    """Shrink the cell padding so a long pseudocode line still fits on one row.

    python-docx has no API for this, so the tblCellMar element is written
    directly. 58 twips is 0.04 in, against Word's 0.08 in default.
    """
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    tblPr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(twips))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblPr.append(mar)


def add_algorithm(doc: Document, lines: list[str]) -> None:
    """Boxed pseudocode: one-cell table, monospace, indentation preserved.

    The font is 7 pt because the longest pseudocode line here runs to 113
    characters; at 8 pt those lines wrap and the right-hand equation
    references detach from the step they annotate.
    """
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    _tight_cell_margins(t)
    cell = t.cell(0, 0)
    cell.text = ""
    first = True
    for line in lines:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        pf = p.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = 1.0
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        r = p.add_run(NBSP * indent + stripped)
        r.font.name = MONO_FONT
        r.font.size = Pt(7.0)
        r.font.color.rgb = BLACK
        # the header line ("Algorithm N - Title") is set in bold
        if line.strip().lower().startswith("algorithm "):
            r.bold = True
    doc.add_paragraph()


# ------------------------------------------------------------------ citations
def resolve_citations(md: str) -> tuple[str, list[str], list[str]]:
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
    out = re.sub(r"\]\[(?=\d)", "], [", out)
    return out, [f"[{i}] {BIB[k]}" for i, k in enumerate(order, 1)], sorted(missing)


# --------------------------------------------------------------- typography
def bind_units(md: str) -> str:
    """Tie a number to its unit with a non-breaking space.

    The manuscript writes "56 %", "9.81 pp", "2.7 ms" with a normal space (SI
    convention). Word will happily break the line between the two, leaving a
    stranded "%" at the start of a line. A U+00A0 keeps them together without
    changing how the text reads.
    """
    return re.sub(r"(\d)\s(%|pp\b|ms\b|MB\b|Hz\b|px\b|s\b)", "\\1\u00a0\\2", md)


# --------------------------------------------------------------- md -> docx
def render_markdown(doc: Document, md: str) -> dict:
    lines = md.splitlines()
    stats = {"eq": 0, "alg": 0, "fig": 0, "tbl": 0, "missing_fig": [],
             "ragged_tables": []}
    eq_numbers: list[int] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()

        if not s or s == "---":
            i += 1
            continue

        # ---- display equation --------------------------------------------
        if s == "$$":
            i += 1
            buf = []
            number = None
            while i < len(lines):
                t = lines[i].strip()
                m = re.fullmatch(r"\$\$(?:\s*(\(\d+[a-z]?\)))?", t)
                if m:
                    number = m.group(1)
                    i += 1
                    break
                buf.append(lines[i].strip())
                i += 1
            add_equation(doc, buf, number)
            stats["eq"] += 1
            if number:
                d = re.fullmatch(r"\((\d+)[a-z]?\)", number)
                if d:
                    eq_numbers.append(int(d.group(1)))
            continue

        # ---- algorithm block ---------------------------------------------
        if s.startswith("```algorithm"):
            i += 1
            buf = []
            while i < len(lines) and lines[i].strip() != "```":
                buf.append(lines[i].rstrip())
                i += 1
            i += 1                                    # consume closing fence
            add_algorithm(doc, buf)
            stats["alg"] += 1
            continue

        # ---- pipe table ---------------------------------------------------
        if s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not is_sep(lines[i]):
                    block.append(split_row(lines[i]))
                i += 1
            if block:
                add_table(doc, block, stats)
                stats["tbl"] += 1
            continue

        # ---- inline figure -------------------------------------------------
        if s.startswith("!FIG"):
            rel, _, cap = s[4:].partition("|")
            path = FIGDIR / rel.strip()
            if path.exists():
                doc.add_picture(str(path), width=Inches(6.3))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                caption(doc, cap.strip())
                stats["fig"] += 1
            else:
                stats["missing_fig"].append(str(path))
            i += 1
            continue

        # ---- headings ------------------------------------------------------
        m = re.match(r"^(#{1,5})\s+(.*)", s)
        if m:
            # the H1 paper title is lifted into the title block, so "## 1."
            # numbered sections are the document's Heading 1.
            level, text = max(1, len(m.group(1)) - 1), m.group(2)
            h = doc.add_heading(level=min(level, 4))
            add_runs(h, text)
            i += 1
            continue

        # ---- lists ----------------------------------------------------------
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

        # ---- prose: rejoin the hard-wrapped source lines --------------------
        buf = []
        while i < len(lines):
            t = lines[i].strip()
            if (not t or t == "---" or t == "$$" or t.startswith(("|", "#", "!FIG", "```"))
                    or re.match(r"^([-*]\s+|\d+\.\s+)", t)):
                break
            buf.append(t)
            i += 1
        p = doc.add_paragraph()
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_runs(p, " ".join(buf))

    stats["eq_numbers"] = eq_numbers
    return stats


# ---------------------------------------------------------------- title block
def title_block(doc: Document, title: str, keywords: str) -> None:
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_after = Pt(10)
    r = t.add_run(title)
    r.bold = True
    r.font.size = Pt(18)
    r.font.color.rgb = BLACK

    for line, size, italic in AUTHOR_BLOCK:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        rr = p.add_run(line)
        rr.font.size = Pt(size)
        rr.italic = italic
        rr.font.color.rgb = BLACK
    doc.add_paragraph()

    if keywords:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(10)
        add_runs(p, keywords)


# ------------------------------------------------------------------------ main
def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "HANDSFREE AI PAPER 1.docx"

    md = "\n\n".join((PARTS_DIR / p).read_text(encoding="utf-8") for p in PARTS)

    # lift the H1 title and the keywords line out of the body
    title = "MAPLE"
    m = re.search(r"^#\s+(.+)$", md, flags=re.M)
    if m:
        title = m.group(1).strip()
        md = md.replace(m.group(0), "", 1)
    keywords = ""
    mk = re.search(r"^\*\*Keywords:\*\*.*$", md, flags=re.M)
    if mk:
        keywords = mk.group(0)
        md = md.replace(keywords, "", 1)

    body, references, missing = resolve_citations(md)
    body = bind_units(body)

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = BLACK
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    for lvl, size in {1: 14, 2: 12, 3: 11, 4: 10.5}.items():
        try:
            st = doc.styles[f"Heading {lvl}"]
        except KeyError:
            continue
        st.font.name = BODY_FONT
        st.font.color.rgb = BLACK
        st.font.bold = True
        st.font.italic = False
        st.font.size = Pt(size)

    global TEXT_WIDTH_IN
    TEXT_WIDTH_IN = set_page_geometry(doc)

    title_block(doc, title, keywords)
    stats = render_markdown(doc, body)

    if references:
        doc.add_page_break()
        doc.add_heading("References", level=1)
        for ref in references:
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Inches(0.35)
            par.paragraph_format.first_line_indent = Inches(-0.35)
            par.paragraph_format.space_after = Pt(3)
            add_runs(par, ref)
            for r in par.runs:
                r.font.size = Pt(9.5)

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))

    # ---- report ----------------------------------------------------------
    nums = stats["eq_numbers"]
    gaps = [(a, b) for a, b in zip(nums, nums[1:]) if b != a + 1]
    print(f"wrote {out}")
    print(f"  equations : {stats['eq']}  (numbered {min(nums) if nums else 0}"
          f"-{max(nums) if nums else 0})")
    print(f"  algorithms: {stats['alg']}")
    print(f"  tables    : {stats['tbl']}")
    print(f"  figures   : {stats['fig']}")
    print(f"  references: {len(references)}")
    if gaps:
        print(f"  ! equation numbering breaks at: {gaps[:12]}")
    if missing:
        print(f"  ! unknown citation keys: {missing}")
    for f in stats["missing_fig"]:
        print(f"  ! missing figure: {f}")
    for head, widths in stats["ragged_tables"]:
        print(f"  ! ragged table (a cell contains '|'): {head} column counts {widths}")


if __name__ == "__main__":
    main()
