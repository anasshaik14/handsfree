"""Convert the MAPLE manuscript parts into an Overleaf-ready LaTeX project.

Produces  reports/overleaf/
    main.tex          elsarticle (Elsevier) preamble, front matter, \\input lines
    sections/         one .tex per paper section (see SECTION_MAP)
    refs.bib          BibTeX generated from scripts/refs.py
    highlights.tex    Elsevier Highlights, a standalone document
    figures/          only the figures the in-scope sections reference
    README.md         upload + compile instructions

Sections are written one at a time. `--only 1,2` converts Sections 1 and 2 and
writes the rest as stubs that still compile and keep the numbering correct, so
the paper is a valid document at every stage and adding a section is one re-run
with a wider `--only`. With no `--only`, every section is converted.

What is converted automatically
    headings          ## / ### / ####      -> \\section / \\subsection / \\subsubsection
    equations         $$ ... $$ (N)        -> equation environment + \\label{eq:N}
    in-text (N)       equation references  -> \\eqref{eq:N}
    figures           !FIG path | caption  -> figure environment + \\label{fig:N}
    in-text Fig. N                         -> Fig.~\\ref{fig:N}
    tables            pipe tables          -> tabularx + \\label{tab:N}
    in-text Table N                        -> Table~\\ref{tab:N}
    algorithms        ```algorithm blocks  -> algorithm + algorithmic
    citations         [@key]               -> \\cite{key}
    inline markup     **b** *i* `c`        -> \\textbf \\emph \\texttt
    Unicode maths     160 distinct glyphs  -> LaTeX commands (see SYM)

Because LaTeX numbers equations, figures, tables and algorithms itself, the
manual numbers in the markdown become labels and every cross-reference becomes
a \\ref. Inserting an equation later therefore renumbers the whole paper
correctly, which the .docx pipeline could not do.

CAVEAT: the source equations are ASCII/Unicode maths written for a plain-text
renderer, and a few carry English words inside them. Those are wrapped in
\\text{} heuristically. The output compiles, but the equations should be read
once before submission -- the script reports which ones contain prose.

Usage:  python scripts/build_paper_latex.py [--only 1,2] [--zip]
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from refs import BIB  # noqa: E402

PARTS_DIR = ROOT / "reports" / "paper"
PARTS = [
    "part1_intro.md", "part2_method_a.md", "part3_method_b.md",
    "part4_experimental.md", "part5a_results_leak_gaze.md",
    "part5b_results_pose_voice.md", "part5c_results_blink_synth.md",
    "part6_conclusion.md",
]
FIGDIR = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "overleaf"

TITLE = ("MAPLE: A leakage-audited and explainable framework for real-time "
         "multimodal hands-free interaction")
AUTHOR = "Mohammad Anas Shaik"
DEPT = "MSc Data Science"
INST = "York St John University"
CITY = "London"
COUNTRY = "United Kingdom"
EMAIL = "mohammad.shaik@yorksj.ac.uk"
SHORT_TITLE = "Leakage-audited multimodal hands-free interaction"
SHORT_AUTHOR = "M. A. Shaik"
CREDIT = ("Conceptualization, Methodology, Software, Validation, "
          "Formal analysis, Investigation, Data curation, "
          "Writing -- original draft, Writing -- review and editing, "
          "Visualization")

# Files lifted out of els-cas-templates.zip so the project compiles anywhere,
# including an Overleaf account whose TeX Live lacks the CAS bundle.
TEMPLATE_ZIP = ROOT / "els-cas-templates.zip"
TEMPLATE_FILES = ["cas-sc.cls", "cas-dc.cls", "cas-common.sty",
                  "cas-model2-names.bst"]

# --------------------------------------------------------------- symbol table
GREEK = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\varepsilon", "ϵ": r"\epsilon", "ζ": r"\zeta", "η": r"\eta",
    "θ": r"\theta", "ϑ": r"\vartheta", "κ": r"\kappa", "λ": r"\lambda",
    "μ": r"\mu", "ν": r"\nu", "π": r"\pi", "ρ": r"\rho", "ϱ": r"\varrho",
    "σ": r"\sigma", "τ": r"\tau", "φ": r"\varphi", "ϕ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Λ": r"\Lambda", "Π": r"\Pi",
    "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    "Ι": r"\mathrm{I}",          # capital iota: used as a plain roman I
}
BB = {"𝔼": r"\mathbb{E}", "ℙ": r"\mathbb{P}", "ℝ": r"\mathbb{R}",
      "ℕ": r"\mathbb{N}",
      # bbm's font descriptor (ubbm.fd) errors on Overleaf's TeX Live, and an
      # indicator function does not justify a whole extra font package.
      "𝟙": r"\mathbf{1}"}
CAL = {"𝒜": "A", "ℬ": "B", "𝒞": "C", "𝒟": "D", "ℰ": "E", "ℱ": "F", "𝒢": "G",
       "ℋ": "H", "𝒦": "K", "ℒ": "L", "ℳ": "M", "𝒪": "O", "𝒬": "Q", "ℛ": "R",
       "𝒮": "S", "𝒯": "T", "𝒰": "U", "𝒱": "V", "𝒲": "W", "𝒳": "X", "𝒴": "Y"}
BOLD = {"𝐚": "a", "𝐛": "b", "𝐜": "c", "𝐠": "g", "𝐤": "k", "𝐦": "m", "𝐨": "o",
        "𝐬": "s", "𝐮": "u", "𝐰": "w", "𝐱": "x", "𝐈": "I", "𝐊": "K", "𝐌": "M",
        "𝐏": "P", "𝐑": "R", "𝐒": "S", "𝐖": "W", "𝛗": r"\varphi",
        "𝚽": r"\Phi"}
ACCENT = {"Ē": r"\bar{E}", "ē": r"\bar{e}", "ā": r"\bar{a}",
          "ŝ": r"\hat{s}", "ĉ": r"\hat{c}", "ĝ": r"\hat{g}", "ê": r"\hat{e}",
          "Ĝ": r"\hat{G}", "ẽ": r"\tilde{e}", "ẋ": r"\dot{x}"}
SUPER = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
         "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+",
         "ⁿ": "n"}
SUB = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6",
       "₇": "7", "₈": "8", "₉": "9", "₋": "-", "ₜ": "t", "ᵢ": "i", "ⱼ": "j",
       "ₘ": "m", "ₖ": "k", "ₙ": "n", "ₐ": "a", "ᵣ": "r", "₌": "=", "₊": "+"}
OPS = {
    "−": "-", "–": "--", "—": "---", "·": r"\cdot ", "×": r"\times ",
    "±": r"\pm ", "≈": r"\approx ", "≠": r"\neq ", "≤": r"\leq ",
    "≥": r"\geq ", "≪": r"\ll ", "≫": r"\gg ", "≲": r"\lesssim ",
    "≼": r"\preceq ", "≡": r"\equiv ", "∼": r"\sim ", "∈": r"\in ",
    "∉": r"\notin ", "⊂": r"\subset ", "⊆": r"\subseteq ", "∩": r"\cap ",
    "∪": r"\cup ", "⊎": r"\uplus ", "∅": r"\emptyset ", "∀": r"\forall ",
    "∃": r"\exists ", "→": r"\to ", "←": r"\leftarrow ", "↦": r"\mapsto ",
    "⟹": r"\implies ", "⟺": r"\iff ", "∘": r"\circ ", "⊙": r"\odot ",
    "⊕": r"\oplus ", "√": r"\sqrt", "∫": r"\int ", "∂": r"\partial ",
    "∇": r"\nabla ", "∞": r"\infty ", "⋃": r"\bigcup ", "⌊": r"\lfloor ",
    "⌋": r"\rfloor ", "⌈": r"\lceil ", "⌉": r"\rceil ", "‖": r"\|",
    "⟨": r"\langle ", "⟩": r"\rangle ", "…": r"\dots ", "′": "'",
    "°": r"^{\circ}", "½": r"\tfrac{1}{2}", "¬": r"\neg ", "⊥": r"\bot ",
    "ℓ": r"\ell ", "ᵀ": r"^{\mathsf{T}}", "⏟": "", "▷": r"\triangleright ",
    "•": r"\bullet ", "€": r"\euro{}",
}
COMBINING = {"\u0302": "hat", "\u0304": "bar", "\u0303": "tilde", "\u0307": "dot"}

# multi-letter tokens that are operators/functions, not variable products
FUNCS = ["arg max", "arg min", "MaxPool", "AdaptiveAvgPool", "Deadzone",
         "SlewLimit", "ReLU", "Conv", "GAP", "Drop", "Lin", "BN", "Resize",
         "Warp", "Gray", "Rot", "Jitter", "Blur", "Flip", "vec", "clip",
         "exp", "log", "max", "min", "sgn", "diag", "trace", "mean", "median",
         "Var", "Cov", "MSE", "MAE", "RMSE", "Acc", "BalAcc", "Prec", "Rec",
         "ECE", "Brier", "LogLoss", "AUC", "ROC", "PR", "MCC", "TPR", "FPR",
         "FNR", "TP", "TN", "FP", "FN", "DEFF", "Bernoulli", "softmax",
         "triangular", "mel", "Attr", "IG", "Fire", "cov", "risk", "spread",
         "inf", "sup", "lim", "Crop", "GERA", "Deadzone",
         "margin", "contribution", "against", "otherwise", "if", "where"]


def unicode_to_tex(s: str) -> str:
    """Map the manuscript's Unicode maths onto LaTeX commands."""
    # base letter + combining accent -> \hat{x} etc.
    # Two passes: an accent can stack on an already-converted group, as in
    # x-with-dot followed by a combining circumflex (\hat{\dot{x}}).
    for _ in range(2):
        for comb, cmd in COMBINING.items():
            s = re.sub(r"(\\[a-z]+\{[^{}]*\}|\w)" + comb,
                       lambda m, c=cmd: "\\%s{%s}" % (c, m.group(1)), s)
    for table in (ACCENT, BB, GREEK):
        for k, v in table.items():
            s = s.replace(k, v + " " if v[-1].isalpha() else v)
    for k, v in CAL.items():
        s = s.replace(k, r"\mathcal{%s}" % v)
    for k, v in BOLD.items():
        s = s.replace(k, r"\mathbf{%s}" % v if len(v) == 1 else r"\boldsymbol{%s}" % v)
    # runs of Unicode super/subscripts collapse into one ^{...} / _{...}
    s = re.sub("([" + "".join(SUPER) + "]+)",
               lambda m: "^{" + "".join(SUPER[c] for c in m.group(1)) + "}", s)
    s = re.sub("([" + "".join(SUB) + "]+)",
               lambda m: "_{" + "".join(SUB[c] for c in m.group(1)) + "}", s)
    for k, v in OPS.items():
        # a command ending in a letter must not weld to the next token:
        # "\sqrt" + "n" became "\sqrtn", an undefined control sequence
        s = s.replace(k, v + " " if v[-1:].isalpha() else v)
    return s


def _balanced(s: str, i: int) -> int:
    """Index just past the parenthesis group starting at s[i] == '(' (-1 if none)."""
    if i >= len(s) or s[i] != "(":
        return -1
    d = 0
    for j in range(i, len(s)):
        if s[j] == "(":
            d += 1
        elif s[j] == ")":
            d -= 1
            if d == 0:
                return j + 1
    return -1


def fix_sqrt(s: str) -> str:
    r"""Give every \sqrt a braced radicand.

    The source writes √( ... ) and √n. Naive substitution yields "\sqrt( ... )",
    which typesets a literal paren under no radical, and "\sqrtn", which is an
    undefined control sequence that kills the compile.
    """
    out, i = [], 0
    while True:
        k = s.find(r"\sqrt", i)
        if k < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:k])
        j = k + len(r"\sqrt")
        while j < len(s) and s[j] == " ":
            j += 1
        if j < len(s) and s[j] == "{":                  # already braced
            out.append(s[k:j])
            i = j
            continue
        end = _balanced(s, j)
        if end > 0:                                     # \sqrt( expr )
            out.append(r"\sqrt{%s}" % s[j + 1:end - 1].strip())
            i = end
            continue
        m = re.match(r"\\[a-zA-Z]+|[A-Za-z0-9.]+(?:_\{[^{}]*\}|\^\{[^{}]*\})*", s[j:])
        if m:                                           # \sqrt n_{dwell}
            out.append(r"\sqrt{%s}" % m.group(0))
            i = j + m.end()
        else:
            out.append(r"\sqrt")
            i = j


def brace_scripts(s: str) -> str:
    r"""A script whose argument is a command needs its own group.

    "f_\hat{\vartheta}" makes TeX report "Missing { inserted"; it must be
    written "f_{\hat{\vartheta}}".
    """
    s = re.sub(r"([_^])(\\[a-zA-Z]+\{[^{}]*\})", r"\1{\2}", s)
    s = re.sub(r"([_^])(\\[a-zA-Z]+)(?!\s*\{)", r"\1{\2}", s)
    # "C^(l)" is a parenthesised layer index, and unbraced it superscripts
    # only the opening bracket
    s = re.sub(r"([_^])(\([^()]*\))", r"\1{\2}", s)
    # sqrt took only the symbol, leaving its subscript outside the radical
    return re.sub(r"(\\sqrt\{[^{}]*)\}(_\{[^{}]*\})", r"\1\2}", s)


TRIG = ("arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
        "sin", "cos", "tan")

FONT_CMDS = frozenset((
    "mathrm", "text", "mathcal", "mathbf", "mathbb", "mathsf", "mathtt",
    "mathit", "operatorname", "textbf", "textit", "emph", "label",
    "begin", "end", "caption", "ref", "eqref",
    "CITE", "EQREF", "FIGREF", "TABREF", "ALGREF", "FIGRANGE"))


def _group_end(s: str, i: int) -> int:
    """Index just past the balanced {...} group that starts at s[i]."""
    depth = 0
    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(s)


def _protected(s: str) -> list:
    r"""Mask the characters a word rewrite must not touch.

    Command names themselves, and the argument of any command whose argument
    is already set as words -- \mathrm{dwell} must stay exactly as it is.
    """
    mask = [False] * len(s)
    for m in re.finditer(r"\\([a-zA-Z]+)", s):
        for i in range(m.start(), m.end()):
            mask[i] = True
        if m.group(1) in FONT_CMDS:
            j = m.end()
            while j < len(s) and s[j] == " ":
                j += 1
            if j < len(s) and s[j] == "{":
                for i in range(j, _group_end(s, j)):
                    mask[i] = True
    return mask


def words_to_text(s: str) -> str:
    r"""Wrap English prose inside maths in \text{}.

    Several equations read "FP = 347 open eyes classified closed"; in maths
    mode that sets as a single italic run of kerned letters reading
    "openeyesclassifiedclosed". A span counts as prose when it is a sequence
    of alphabetic tokens of which at least one is three letters or more.
    """
    mask = _protected(s)
    # candidate words are searched inside each run of unprotected characters.
    # Searching the whole string instead lets a span such as "in conv" (where
    # the "in" is really \in) straddle a command and be thrown away, which
    # left 27 words bare.
    spans, run = [], None
    for i, prot in enumerate(mask + [True]):
        if prot:
            if run is not None:
                spans.append((run, i))
                run = None
        elif run is None:
            run = i
    out, last = [], 0
    pat = re.compile(r"[A-Za-z]{2,}(?: +[A-Za-z]+)*")
    for lo, hi in spans:
        for m in pat.finditer(s, lo, hi):
            a, b = m.span()
            toks = m.group(0).split()
            if not any(len(t) >= 3 for t in toks):
                continue
            before, after = s[:a].rstrip(), s[b:].lstrip()
            lead = "" if (not before or before[-1] in "({[") else r"\;"
            tail = "" if (not after or after[0] in ")}],.;") else r"\;"
            out.append(s[last:a])
            out.append("%s\\text{%s}%s" % (lead, " ".join(toks), tail))
            last = b
    out.append(s[last:])
    return "".join(out)


def _split_top(s: str, sep: str) -> list:
    """Split on `sep` only where brace/bracket/paren depth is zero."""
    parts, depth, start, i = [], 0, 0, 0
    while i < len(s):
        c = s[i]
        # a nested environment is one unit: the row separators inside a
        # bmatrix sit at brace depth zero but must not be split on
        if s.startswith("\\begin{", i):
            depth += 1
        elif s.startswith("\\end{", i):
            depth -= 1
        elif c in "{[(":
            depth += 1
        elif c in "}])":
            depth -= 1
        elif depth == 0 and s.startswith(sep, i):
            parts.append(s[start:i])
            start = i + len(sep)
            i += len(sep)
            continue
        i += 1
    parts.append(s[start:])
    return parts


def bracket_matrices(s: str) -> str:
    r"""[[1,0,0],[0,c,-s],[0,s,c]] is a matrix, not nested intervals.

    Set literally it is one 160pt-overfull line of brackets and commas;
    bmatrix sets it as three rows.
    """
    while True:
        i = s.find("[[")
        if i < 0:
            return s
        depth, j = 0, i
        while j < len(s):
            if s[j] == "[":
                depth += 1
            elif s[j] == "]":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        else:
            return s
        rows = _split_top(s[i + 1:j], ",")
        cells = []
        for r in rows:
            r = r.strip()
            if not (r.startswith("[") and r.endswith("]")):
                return s                       # not a matrix after all
            cells.append(" & ".join(c.strip()
                                    for c in _split_top(r[1:-1], ",")))
        body = " \\\\ ".join(cells)
        s = s[:i] + "\\begin{bmatrix} " + body + " \\end{bmatrix}" + s[j + 1:]


def wrap_long(body: str, limit: int = 96) -> str:
    """Break an over-wide one-line equation at its own clause boundaries."""
    if len(body) <= limit or len(_split_top(body, "\\\\")) > 1:
        return body        # already broken into rows by the source
    chunks = []
    for part in _split_top(body, ","):
        arrows = _split_top(part, "\\to")
        chunks.extend(a + " \\to" for a in arrows[:-1])
        chunks.append(arrows[-1] + ",")
    chunks[-1] = chunks[-1].rstrip(",")
    if len(chunks) < 2:
        return body
    lines, cur = [], ""
    for c in chunks:
        c = c.strip()
        if cur and len(cur) + len(c) > limit - 12:
            lines.append(cur)
            cur = c
        else:
            cur = (cur + " " + c).strip()
    if cur:
        lines.append(cur)
    return " \\\\\n".join(lines) if len(lines) > 1 else body


def tidy_math(s: str) -> str:
    """Post-process a converted expression into idiomatic LaTeX."""
    # multi-letter sub/superscripts are labels, not products
    s = re.sub(r"_([A-Za-z][A-Za-z0-9]+)", r"_{\\mathrm{\1}}", s)
    s = re.sub(r"\^([A-Za-z][A-Za-z0-9]+)", r"^{\\mathrm{\1}}", s)
    # trigonometry: "cos a" is an operator applied to an angle, not three
    # italic variables multiplied together
    for fn in TRIG:
        s = re.sub(r"(?<![\\A-Za-z}])" + fn + r"(?![A-Za-z])", "\\\\" + fn, s)
    # known function names upright
    for fn in sorted(FUNCS, key=len, reverse=True):
        s = re.sub(r"(?<![\\A-Za-z}])" + re.escape(fn) + r"(?![A-Za-z])",
                   r"\\mathrm{%s}" % fn.replace(" ", r"\,"), s)
    # '#' is TeX's macro-parameter character and is illegal in maths; the
    # source uses it for pushforward. '&' would start a new table column.
    s = s.replace("%", r"\%").replace("#", r"\#").replace("&", r"\&")
    s = fix_sqrt(s)
    s = brace_scripts(s)
    # Runs of spaces in the markdown are visual alignment, not quad skips.
    # Turning every one of them into \quad produced 794 of them and was the
    # main reason equations ran off the page; "\kappa \quad = 0.8608" is
    # wrong in itself, because maths mode already spaces a relation.
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\\mathrm\{(\\mathrm\{[^{}]*\})\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([A-Za-z]+)\}-\\mathrm\{([A-Za-z]+)\}",
               r"\\mathrm{\1\\text{-}\2}", s)
    # an en dash in maths mode sets as two minus signs
    s = re.sub(r"(?<=[0-9])--(?=[0-9])", r"\\text{--}", s)
    s = bracket_matrices(s)
    # a connective set as \mathrm{if} butts straight against the expression
    # beside it; \text gives it the surrounding font and real spaces
    s = re.sub(r"\\mathrm\{(if|otherwise|where|and|or|for|all|with|else"
               r"|when|given)\}", r"\\;\\text{\1}\\;", s)
    s = words_to_text(s)
    s = re.sub(r"(\;)(?:\s*\;)+", r"", s)
    # display maths must not carry a double script either
    return merge_scripts(s.strip())


def to_math(s: str) -> str:
    return tidy_math(unicode_to_tex(s))


# ---------------------------------------------------------------- text escape
def escape_text(s: str) -> str:
    """Escape LaTeX specials in prose, leaving inline maths conversion to callers."""
    s = s.replace("\\", r"\textbackslash{}")
    for ch in "&%$#_{}":
        s = s.replace(ch, "\\" + ch)
    s = s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    return s


def inline(s: str) -> str:
    """Markdown inline markup + citations + cross-references -> LaTeX."""
    out, i = [], 0
    for m in re.finditer(r"\*\*(.+?)\*\*|(?<!\*)\*([^*]+?)\*(?!\*)|`([^`]+?)`", s):
        out.append(prose(s[i:m.start()]))
        if m.group(1) is not None:
            out.append(r"\textbf{%s}" % prose(m.group(1)))
        elif m.group(2) is not None:
            out.append(r"\emph{%s}" % prose(m.group(2)))
        else:
            out.append(r"\texttt{%s}" % escape_text(m.group(3)))
        i = m.end()
    out.append(prose(s[i:]))
    return "".join(out)


SENT_OPEN, SENT_ARG, SENT_SEP, SENT_CLOSE = "\x01", "\x02", "\x04", "\x03"


_SCRIPT_PAYLOAD = (r"(?:\\\{.*?\\\}|\{[^{}]*\}|\$[^$]+\$|\([^()]*\)"
                   r"|(?:--|-)?\d+|[A-Za-z][A-Za-z0-9]*)")
_SCRIPT_PART = re.compile(r"(\\_|\\textasciicircum\{\})(" + _SCRIPT_PAYLOAD + ")")
_BARE_BASE = re.compile(r"(?<![A-Za-z0-9\\])([A-Za-z][A-Za-z0-9]*|\d+(?:\.\d+)?)"
                        r"(?=\\_|\\textasciicircum\{\})")


def _script_payload(kind: str, raw: str) -> str:
    """Render one captured sub/superscript payload as maths."""
    inner = raw
    if inner.startswith("$") and inner.endswith("$"):
        inner = inner[1:-1]
    elif inner.startswith("\\{") and inner.endswith("\\}"):
        inner = inner[2:-2]
    elif inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1]
    # the payload is about to become part of a maths span, so any maths
    # delimiters inside it would nest and break the span
    # a space, not nothing: dropping the delimiters from "$	imes$W" would
    # weld them into the undefined command 	imesW
    inner = re.sub(r" {2,}", " ", inner.replace("$", " ")).strip()
    # the prose escape turned a minus into an en dash; inside maths it must
    # go back to one minus sign, not two
    inner = re.sub(r"^--(?=\d)", "-", inner)
    if re.fullmatch(r"[A-Za-z]{2,}", inner):
        inner = r"\mathrm{%s}" % inner            # "in", "out", "run" are tags
    return ("^" if "circum" in kind else "_") + "{%s}" % inner


def merge_prose_scripts(s: str) -> str:
    r"""Pull escaped scripts back inside the maths they belong to.

    "E_A^out" in the markdown came out as "$\bar{E}_{A}$\textasciicircum{}out",
    which prints a literal caret next to the symbol, 47 times over. The same
    happened to subscripts via \_. Maths spans are located first and paired
    left to right, so a base is never mistaken for the gap between two spans.
    """
    out, last = [], 0
    for m in re.finditer(r"\$[^$]+\$", s):
        if m.start() < last:
            continue
        j, parts = m.end(), []
        while True:
            p = _SCRIPT_PART.match(s, j)
            if not p:
                break
            parts.append(_script_payload(p.group(1), p.group(2)))
            j = p.end()
        if not parts:
            continue
        out.append(s[last:m.start()])
        out.append("$" + m.group(0)[1:-1] + "".join(parts) + "$")
        last = j
    out.append(s[last:])
    s = "".join(out)

    # the same chain, but hanging off a bare identifier: n\_eff\^{}run
    def bare(m):
        j, parts = m.end(), []
        while True:
            p = _SCRIPT_PART.match(m.string, j)
            if not p:
                break
            parts.append(_script_payload(p.group(1), p.group(2)))
            j = p.end()
        if not parts:
            return m.group(0)
        bare.skip_to = j
        return "$" + m.group(1) + "".join(parts) + "$"

    res, pos = [], 0
    for m in _BARE_BASE.finditer(s):
        if m.start() < pos:
            continue
        bare.skip_to = m.end()
        rep = bare(m)
        if rep == m.group(0):
            continue
        res.append(s[pos:m.start()])
        res.append(rep)
        pos = bare.skip_to
    res.append(s[pos:])
    s = "".join(res)
    # pulling a script inside can leave two maths spans touching, and "$$"
    # is display maths, not two inline ones. Fuse them.
    while "$$" in s:
        s = s.replace("$$", "")
    return s


def prose(s: str) -> str:
    """Escape prose and convert its symbols and cross-references.

    Cross-reference macros are staged as control characters rather than as
    backslash commands. escape_text() rewrites every backslash and brace, so a
    macro inserted before escaping comes back as \\textbackslash{}CITE\\{key\\};
    sentinels pass through escaping untouched and become macros last.
    """
    def sent(name, *args):
        return SENT_OPEN + name + SENT_ARG + SENT_SEP.join(args) + SENT_CLOSE

    s = re.sub(r"\[@([A-Za-z0-9_]+)\]", lambda m: sent("CITE", m.group(1)), s)
    s = re.sub(r"\bFigs?\.\s*(\d+)\s*[–-]\s*(\d+)",
               lambda m: sent("FIGRANGE", m.group(1), m.group(2)), s)
    s = re.sub(r"\bFig\.\s*(\d+)", lambda m: sent("FIGREF", m.group(1)), s)
    s = re.sub(r"\bTable\s+(\d+)", lambda m: sent("TABREF", m.group(1)), s)
    s = re.sub(r"\bAlgorithm\s+(\d+)", lambda m: sent("ALGREF", m.group(1)), s)
    # An equation reference is never preceded by a subscript or superscript
    # marker: "n_(1) >= n_(2)" is order-statistic notation, and without the
    # _ and ^ in this guard it silently became "n_\eqref{eq:1}".
    s = re.sub(r"(?<![A-Za-z0-9_^])\((\d{1,3})\)",
               lambda m: sent("EQREF", m.group(1)), s)
    s = unicode_to_tex_prose(s)
    s = escape_text_keep_math(s)
    pat = SENT_OPEN + r"([A-Z]+)" + SENT_ARG + r"([^" + SENT_CLOSE + r"]*)" + SENT_CLOSE

    def expand(m):
        # escape_text() ran over the sentinel payload too, so a BibTeX key such
        # as mpiigaze_pami arrives as mpiigaze\_pami. Macro arguments are keys
        # and numbers, never prose, so the escaping is simply undone here.
        args = [a.replace("\\_", "_") for a in m.group(2).split(SENT_SEP)]
        return "\\%s%s" % (m.group(1), "".join("{%s}" % a for a in args))

    s = re.sub(pat, expand, s)
    # "theta_y" and "E_A^out" in running text came out with a literal
    # underscore and a literal caret beside the symbol. Pull them back in.
    return merge_prose_scripts(s)


def escape_text_keep_math(s: str) -> str:
    """Escape LaTeX specials outside $...$ spans, leaving maths untouched."""
    out, last = [], 0
    for m in re.finditer(r"\$[^$]*\$", s):
        out.append(escape_text(s[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(escape_text(s[last:]))
    return "".join(out)


TEXT_SYM = {
    # Anything emitting a backslash must sit inside $...$: escape_text_keep_math
    # skips maths spans but would otherwise rewrite "\dots{}" as
    # "\textbackslash{}dots{}".
    "−": "--", "–": "--", "—": "---", "…": r"$\dots$", "′": "'",
    "‘": "`", "’": "'", "“": "``", "”": "''",
    "°": r"$^{\circ}$", "²": r"$^{2}$", "³": r"$^{3}$", "¹": r"$^{1}$",
    "⁰": r"$^{0}$", "⁴": r"$^{4}$", "⁵": r"$^{5}$", "⁶": r"$^{6}$",
    "⁸": r"$^{8}$", "⁻": r"$^{-}$", "⁺": r"$^{+}$",
    "½": r"$\tfrac{1}{2}$", "±": r"$\pm$", "·": r"$\cdot$", "×": r"$\times$",
    "≈": r"$\approx$", "≤": r"$\leq$", "≥": r"$\geq$", "≫": r"$\gg$",
    "≪": r"$\ll$", "≠": r"$\neq$", "≡": r"$\equiv$", "∼": r"$\sim$",
    "→": r"$\to$", "←": r"$\leftarrow$", "↦": r"$\mapsto$",
    "⟹": r"$\implies$", "∈": r"$\in$", "∉": r"$\notin$", "∩": r"$\cap$",
    "∪": r"$\cup$", "∅": r"$\emptyset$", "∘": r"$\circ$", "⊆": r"$\subseteq$",
    "⊎": r"$\uplus$", "∞": r"$\infty$", "‖": r"$\|$", "√": r"$\surd$",
    "▷": r"$\triangleright$", "⏟": "", "✔": r"$\checkmark$", "✘": r"$\times$",
    "•": r"$\bullet$", "€": r"\texteuro{}", "⌈": r"$\lceil$", "⌉": r"$\rceil$",
    "⌊": r"$\lfloor$", "⌋": r"$\rfloor$", "⟨": r"$\langle$", "⟩": r"$\rangle$",
    "∎": r"$\blacksquare$", "″": "''", "↔": r"$\leftrightarrow$",
    "⊇": r"$\supseteq$", "∝": r"$\propto$", "⊂": r"$\subset$",
    "∀": r"$\forall$", "∃": r"$\exists$", "⊙": r"$\odot$", "⊥": r"$\bot$",
    "¬": r"$\neg$", "∂": r"$\partial$", "∇": r"$\nabla$", "∫": r"$\int$",
    "≲": r"$\lesssim$", "≼": r"$\preceq$", "⋃": r"$\bigcup$",
    "ŷ": r"$\hat{y}$", "ⁿ": r"$^{n}$", "₌": r"$_{=}$", "₊": r"$_{+}$",
}


def unicode_to_tex_prose(s: str) -> str:
    """Symbol pass for prose: text mode where possible, maths as a fallback.

    The fallback class must be [^\\x00-\\x7f]. An earlier [^\\x00-\\xff] skipped
    every Latin-1 glyph, so superscript-two and the degree sign survived raw.
    """
    # a combining accent sits *after* an ASCII base letter, so the fallback
    # below would wrap the accent alone and orphan its base. Pair them first.
    for _ in range(2):
        for comb, cmd in COMBINING.items():
            s = re.sub(r"(\$?\\[a-z]+\{[^{}]*\}\$?|[A-Za-z])" + comb,
                       lambda m, c=cmd: "$\\%s{%s}$" % (c, m.group(1).strip("$")), s)
    # A RUN of Unicode super/subscripts is one script, not several. Converting
    # them one character at a time turned 10^-6 into 10$^{-}$$^{6}$, which
    # merges to ^{-}^{6} -- "Double superscript" and a failed compile.
    s = re.sub("([" + "".join(SUPER) + "]+)",
               lambda m: "$^{" + "".join(SUPER[c] for c in m.group(1)) + "}$", s)
    s = re.sub("([" + "".join(SUB) + "]+)",
               lambda m: "$_{" + "".join(SUB[c] for c in m.group(1)) + "}$", s)
    for k, v in TEXT_SYM.items():
        s = s.replace(k, v)
    # straight ASCII quotes both render as a closing quote in LaTeX; pair them
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    s = re.sub(r"[^\x00-\x7f]+",
               lambda m: "$" + unicode_to_tex(m.group(0)).strip() + "$", s)
    # independently converted neighbours produce "$a$$b$"; merge the spans
    return merge_scripts(s.replace("$$", ""))


def merge_scripts(s: str) -> str:
    """Fold adjacent scripts: ^{a}^{b} -> ^{ab}. Two in a row is a LaTeX error.

    A catch-all safety net -- whatever route produced the adjacency, TeX will
    refuse "Double superscript" and abort the whole compile.
    """
    for _ in range(4):                       # a few rounds settles long runs
        new = re.sub(r"(\^|_)\{([^{}]*)\}\1\{([^{}]*)\}", r"\1{\2\3}", s)
        if new == s:
            break
        s = new
    return s


def algo_line(ln: str) -> str:
    """One pseudocode line -> ttfamily LaTeX with indentation and gaps kept."""
    stripped = ln.lstrip(" ")
    indent = len(ln) - len(stripped)
    body = escape_text_keep_math(unicode_to_tex_prose(stripped))
    # pseudocode carries the same E_A^out / theta_on^a notation as the
    # prose and needs the same repair
    body = merge_prose_scripts(body)
    body = re.sub(r"  +", lambda m: "~" * len(m.group(0)), body)
    return "~" * indent + body


# ------------------------------------------------------------------- builders
def _year_of(s: str) -> str:
    """Last plausible year in a reference string.

    Page ranges and arXiv identifiers contain four-digit numbers that read as
    years ("pp. 2144--2151" yielded 2151 once), so they are masked out first.
    """
    masked = re.sub(r"pp\.\s*[\d\u2013\-]+", " ", s)
    masked = re.sub(r"(vol|no)\.\s*\d+", " ", masked)
    masked = re.sub(r"arXiv:\d{4}\.\d{4,5}", " ", masked)
    yrs = re.findall(r"\b(1[89]\d{2}|20\d{2})\b", masked)
    return yrs[-1] if yrs else "n.d."


def _authors_of(raw: str) -> str:
    """'A. Smith, B. Jones, and C. Lee' -> 'A. Smith and B. Jones and C. Lee'."""
    a = raw.rstrip(", ").replace(", and ", ", ").replace(" and ", ", ")
    etal = "et al." in a
    a = a.replace(" et al.", "")
    names = [x.strip() for x in a.split(",") if x.strip()]
    return " and ".join(names) + (" and others" if etal else "")


def _tex(s: str) -> str:
    """Escape a BibTeX field value and convert markdown emphasis."""
    s = re.sub(r"\*(.+?)\*", r"\\emph{\1}", s)
    s = s.replace("&", r"\&").replace("–", "--").replace("—", "---")
    # strip trailing commas/space only: a trailing period usually belongs to an
    # abbreviation ("IEEE Trans. Pattern Anal. Mach. Intell.")
    s = re.sub(r'"([^"]*)"', r"``\1''", s).strip().rstrip(",").strip()
    return _bib_ascii(s)


# A .bib must be ASCII here. cas-sc loads the stix fonts, whose LS1 and LS2
# encodings carry no UTF-8 mapping ("no UTF-8 mapping file for font encoding
# LS1" in the log), so a raw accented character in a bibliography field is a
# compile error. It went unnoticed for four rounds because a filename mismatch
# meant BibTeX never ran, and an un-run bibliography cannot fail.
BIB_ACCENT = {
    "ö": r'{\"o}', "ä": r'{\"a}', "ü": r'{\"u}', "Ö": r'{\"O}',
    "Ä": r'{\"A}', "Ü": r'{\"U}', "ß": r"{\ss}",
    "á": r"{\'a}", "é": r"{\'e}", "í": r"{\'i}", "ó": r"{\'o}",
    "ú": r"{\'u}", "ý": r"{\'y}", "ć": r"{\'c}", "ś": r"{\'s}",
    "à": r"{\`a}", "è": r"{\`e}", "ì": r"{\`i}", "ò": r"{\`o}",
    "ù": r"{\`u}",
    "â": r"{\^a}", "ê": r"{\^e}", "î": r"{\^i}", "ô": r"{\^o}",
    "û": r"{\^u}",
    "ã": r"{\~a}", "ñ": r"{\~n}", "õ": r"{\~o}",
    "č": r"{\v{c}}", "Č": r"{\v{C}}", "š": r"{\v{s}}", "Š": r"{\v{S}}",
    "ž": r"{\v{z}}", "Ž": r"{\v{Z}}", "ř": r"{\v{r}}", "ě": r"{\v{e}}",
    "ç": r"{\c{c}}", "ł": r"{\l}", "ø": r"{\o}", "å": r"{\aa}",
    "€": r"{\texteuro}", "£": r"{\pounds}", "°": r"{\textdegree}",
    "–": "--", "—": "---", "’": "'", "‘": "`", "“": "``", "”": "''",
}


def _bib_ascii(s: str) -> str:
    """Replace every non-ASCII character in a bibliography field."""
    for ch, rep in BIB_ACCENT.items():
        s = s.replace(ch, rep)
    leftover = sorted({c for c in s if ord(c) > 127})
    if leftover:
        raise ValueError(
            "bibliography field has no ASCII form for %s in %r -- add it to "
            "BIB_ACCENT rather than letting it reach the .bib"
            % (", ".join("U+%04X" % ord(c) for c in leftover), s[:70]))
    return s


def make_bib(only: set[str] | None = None) -> str:
    """Structured BibTeX parsed from the formatted strings in refs.py.

    An author-year style (cas-model2-names, which the journal uses) needs real
    author and year fields; the previous @misc/note form only worked with a
    numeric style. Entry types are inferred from the citation's shape.
    """
    out = []
    for key, s in BIB.items():
        if only is not None and key not in only:
            continue
        mt = re.search(r'"([^"]+?),?"', s)
        if mt:                                    # quoted title: paper
            title, before, after = mt.group(1), s[:mt.start()], s[mt.end():]
        else:                                     # italic title: book
            mi = re.search(r"\*([^*]+)\*", s)
            title, before, after = mi.group(1), s[:mi.start()], s[mi.end():]

        author, year = _authors_of(before), _year_of(s)
        venue = (re.search(r"\*([^*]+)\*", after) or [None, ""])[1] \
            if re.search(r"\*([^*]+)\*", after) else ""
        vol = (re.search(r"vol\.\s*(\d+)", after) or [None, ""])[1] \
            if re.search(r"vol\.\s*(\d+)", after) else ""
        num = (re.search(r"no\.\s*(\d+)", after) or [None, ""])[1] \
            if re.search(r"no\.\s*(\d+)", after) else ""
        pages = (re.search(r"pp\.\s*([\d\u2013\-]+)", after) or [None, ""])[1] \
            if re.search(r"pp\.\s*([\d\u2013\-]+)", after) else ""
        arx = re.search(r"(arXiv:\d{4}\.\d{4,5})", s)
        url = re.search(r"(https?://\S+)", s)

        f = [("author", author), ("title", _tex(title)), ("year", year)]
        if arx:                                             # preprint
            kind = "misc"
            f += [("howpublished", arx.group(1)), ("note", "Preprint")]
        elif url:                                           # software / online
            kind = "misc"
            f += [("howpublished", r"\url{%s}" % url.group(1).rstrip(".")),
                  ("note", _tex(venue) if venue else "Online resource")]
        elif vol:                                           # journal article
            kind = "article"
            f += [("journal", _tex(venue)), ("volume", vol)]
            if num:
                f += [("number", num)]
            if pages:
                f += [("pages", pages.replace("\u2013", "--"))]
        elif venue:                                         # conference paper
            kind = "inproceedings"
            f += [("booktitle", _tex(venue))]
            if pages:
                f += [("pages", pages.replace("\u2013", "--"))]
        else:                                               # book
            kind = "book"
            pub = re.search(r":\s*([^,]+),\s*\d{4}", after)
            if pub:
                f += [("publisher", _tex(pub.group(1)))]
        # every field, not only the ones routed through _tex(): the author
        # field is the one that actually carried the accents
        body = ",\n".join("  %-12s = {%s}" % (k, _bib_ascii(v))
                          for k, v in f if v)
        out.append("@%s{%s,\n%s\n}\n" % (kind, key, body))
    return "\n".join(out)


def convert(md: str, stats: dict) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        raw, s = lines[i], lines[i].strip()
        if not s or s == "---":
            i += 1
            continue

        # ---- equation ----------------------------------------------------
        if s == "$$":
            i += 1
            buf, num = [], None
            while i < len(lines):
                t = lines[i].strip()
                m = re.fullmatch(r"\$\$(?:\s*\((\d+)\))?", t)
                if m:
                    num = m.group(1)
                    i += 1
                    break
                buf.append(t)
                i += 1
            body = r" \\" "\n".join([""] + [to_math(b) for b in buf])[3:] if len(buf) > 1 \
                else to_math(buf[0])
            if len(buf) > 1:
                body = (r" \\ " + "\n").join(to_math(b) for b in buf)
            env = "equation" if num else "equation*"
            lab = "\n\\label{eq:%s}" % num if num else ""
            if len(buf) > 1:
                out.append("\\begin{%s}%s\n\\begin{aligned}\n%s\n\\end{aligned}\n\\end{%s}"
                           % (env, lab, body, env))
            else:
                wrapped = wrap_long(body)
                if wrapped is not body and "\\\\" in wrapped:
                    out.append("\\begin{%s}%s\n\\begin{aligned}\n%s\n"
                               "\\end{aligned}\n\\end{%s}"
                               % (env, lab, wrapped, env))
                    stats["wrapped"] += 1
                else:
                    out.append("\\begin{%s}%s\n%s\n\\end{%s}"
                               % (env, lab, body, env))
            stats["eq"] += 1
            if buf and re.search(r"\b(against|otherwise|if|where|all|seen|held)\b",
                                 " ".join(buf)):
                stats["prose_eq"].append(num)
            continue

        # ---- algorithm ----------------------------------------------------
        if s.startswith("```algorithm"):
            i += 1
            buf = []
            while i < len(lines) and lines[i].strip() != "```":
                buf.append(lines[i].rstrip())
                i += 1
            i += 1
            m = re.match(r"Algorithm (\d+) — (.*)", buf[0])
            num, cap = (m.group(1), m.group(2)) if m else ("?", buf[0])
            body = (" \\\\\n").join(algo_line(ln) for ln in buf[1:] if ln.strip())
            out.append("\\begin{algorithm}[htbp]\n\\caption{%s}\n\\label{alg:%s}\n"
                       "\\hrule\\vspace{3pt}\n"
                       "{\\ttfamily\\scriptsize\\raggedright\n%s\n\\par}\n"
                       "\\vspace{3pt}\\hrule\n"
                       "\\end{algorithm}" % (inline(cap), num, body))
            stats["alg"] += 1
            continue

        # ---- table --------------------------------------------------------
        if s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not re.fullmatch(r"\|[\s:|-]+\|", lines[i].strip()):
                    block.append([c.strip() for c in
                                  lines[i].strip().strip("|").split("|")])
                i += 1
            cap = stats["pending_caption"]
            stats["pending_caption"] = None
            ncol = max(len(r) for r in block)
            # Wide tables (Table 2 has nine columns) do not fit a single-column
            # 6.5in block at footnotesize. Shrink the type and give any column
            # whose entries are all short a natural width instead of an equal
            # share, so the prose columns keep the room they need.
            # measure the visible text, not the markdown: a bold tick "**✔**"
            # is five source characters but one glyph wide.
            def cell_w(r, j):
                return len(re.sub(r"[*`]", "", r[j]).strip()) if j < len(r) else 0
            narrow = [all(cell_w(r, j) <= 4 for r in block) for j in range(ncol)]
            spec = "".join("c" if narrow[j] else "X" for j in range(ncol))
            size = "\\scriptsize" if ncol >= 8 else "\\footnotesize"
            head = " & ".join(r"\textbf{%s}" % inline(c) for c in block[0])
            body = " \\\\\n".join(" & ".join(inline(c) if j < len(r) else ""
                                             for j, c in enumerate(
                                                 list(r) + [""] * (ncol - len(r))))
                                  for r in block[1:])
            num = cap[0] if cap else None
            out.append(
                # plain table/figure, not the starred variants: the document is
                # single-column (elsarticle preprint), where a starred float is
                # page-wide and can only be placed at a page top.
                "\\begin{table}[pos=htbp]\n\\centering\n\\caption{%s}\n\\label{tab:%s}\n"
                "%s\n\\begin{tabularx}{\\textwidth}{%s}\n\\toprule\n"
                "%s \\\\\n\\midrule\n%s \\\\\n\\bottomrule\n\\end{tabularx}\n"
                "\\end{table}" % (cap[1] if cap else "", num or "x", size, spec,
                                  head, body))
            stats["tbl"] += 1
            continue

        # ---- figure -------------------------------------------------------
        if s.startswith("!FIG"):
            rel, _, cap = s[4:].partition("|")
            rel = rel.strip()
            m = re.match(r"\s*Fig\.\s*(\d+)\.\s*(.*)", cap)
            num, ctext = (m.group(1), m.group(2)) if m else ("x", cap.strip())
            stats["figs"].append(rel)
            out.append("\\begin{figure}[pos=htbp]\n\\centering\n"
                       "\\includegraphics[width=\\linewidth]{figures/%s}\n"
                       "\\caption{%s}\n\\label{fig:%s}\n\\end{figure}"
                       % (flat_figname(rel), inline(ctext), num))
            stats["fig"] += 1
            i += 1
            continue

        # ---- table caption (precedes the pipe block) -----------------------
        m = re.match(r"\*\*Table (\d+)\.\*\*\s*(.*)", s)
        if m:
            stats["pending_caption"] = (m.group(1), inline(m.group(2)))
            i += 1
            continue

        # ---- headings ------------------------------------------------------
        m = re.match(r"^(#{2,4})\s+(.*)", s)
        if m:
            lvl = len(m.group(1))
            txt = re.sub(r"^\d+(\.\d+)*\.?\s*", "", m.group(2))
            cmd = {2: "section", 3: "subsection", 4: "subsubsection"}[lvl]
            # Section 5 alone carries 40+ figures, so a barrier only at
            # \section would still let the queue overflow mid-section.
            if cmd in ("section", "subsection"):
                out.append(r"\FloatBarrier")
            out.append("\\%s{%s}" % (cmd, inline(txt)))
            i += 1
            continue

        # ---- lists ----------------------------------------------------------
        if re.match(r"^[-*]\s+", s) or re.match(r"^\d+\.\s+", s):
            ordered = bool(re.match(r"^\d+\.\s+", s))
            items = []
            while i < len(lines):
                t = lines[i].strip()
                if re.match(r"^[-*]\s+", t) or re.match(r"^\d+\.\s+", t):
                    items.append(re.sub(r"^([-*]|\d+\.)\s+", "", t))
                    i += 1
                elif not t:
                    break
                else:
                    items[-1] += " " + t
                    i += 1
            env = "enumerate" if ordered else "itemize"
            out.append("\\begin{%s}\n%s\n\\end{%s}"
                       % (env, "\n".join(r"\item %s" % inline(x) for x in items), env))
            continue

        # ---- prose ----------------------------------------------------------
        buf = []
        while i < len(lines):
            t = lines[i].strip()
            if (not t or t == "---" or t == "$$"
                    or t.startswith(("|", "#", "!FIG", "```", "**Table"))
                    or re.match(r"^([-*]\s+|\d+\.\s+)", t)):
                break
            buf.append(t)
            i += 1
        out.append(inline(" ".join(buf)))
    return "\n\n".join(out)


# A figure is set at \linewidth -- about 6.3 in. At 300 dpi that needs 1890 px;
# the sources run to 3040 px, so every extra pixel is data pdfLaTeX embeds and
# Overleaf's compile timer counts. 1900 keeps full print quality.
FIG_MAX_PX = 1400      # 6.5 in at ~215 dpi; fewer pixels is less decode


def _fast_path(im):
    """Put a figure in the one form pdfTeX embeds without decoding it.

    pdfTeX copies a PNG's compressed stream straight into the PDF -- the
    "(PNG copy)" note in the log -- only for 8-bit RGB or greyscale, no alpha,
    no interlacing, no colour profile. Anything else is decoded with libpng
    and recompressed, which for 63 figures is most of the compile time and is
    what pushes this document over Overleaf's limit. An earlier pass here
    quantised to a 256-colour palette: a third of the bytes, but palette PNGs
    are exactly the case that misses the fast path, so it made the compile
    slower. Size is not the constraint; decode time is.
    """
    im.info.pop("icc_profile", None)
    im.info.pop("transparency", None)
    return im


def copy_figure(src: Path, dst: Path) -> int:
    """Copy a figure, downsampling if it is wider than print needs.

    Returns the bytes saved. Falls back to a plain copy if Pillow is absent or
    the image cannot be read -- a slow compile beats a missing figure.
    """
    try:
        from PIL import Image
        with Image.open(src) as im:
            small = im
            if small.width > FIG_MAX_PX:
                h = round(small.height * FIG_MAX_PX / small.width)
                small = small.resize((FIG_MAX_PX, h), Image.LANCZOS)
            if small.mode == "RGBA":
                # figures sit on white; the alpha channel is a quarter of the
                # data and carries nothing once it is composited
                bg = Image.new("RGB", small.size, "white")
                bg.paste(small, mask=small.split()[3])
                small = bg
            elif small.mode != "RGB":
                small = small.convert("RGB")
            small = _fast_path(small)
            small.save(dst, optimize=True, compress_level=9)
            # Always keep the rewritten file, even when it is the larger of
            # the two. An earlier version kept whichever was smaller, which
            # quietly restored 26 originals in RGBA at full resolution -- the
            # exact form that misses pdfTeX's fast path.
            return src.stat().st_size - dst.stat().st_size
    except Exception:                                    # noqa: BLE001
        pass
    shutil.copy2(src, dst)
    return 0


def flat_figname(rel: str) -> str:
    """Flatten a figure's source path into a unique filename.

    The figure tree reuses basenames across modalities -- attribution/gaze/
    A1_summary.png and attribution/blink/A1_summary.png, and so on for all four
    modalities. Copying by basename alone silently overwrote 12 of the 63
    figures, so the paper showed the wrong image in twelve places.
    """
    return re.sub(r"[\\/]+", "_", rel.strip())


def tidy_body(body: str) -> str:
    """Post-passes applied to every converted section."""
    # blank-line-separated bullets each became their own list; rejoin them
    body = re.sub(r"\\end\{(itemize|enumerate)\}\s*\n\s*\n\\begin\{\1\}\n", "\n", body)
    # [@a][@b] -> one \cite so the style prints [2,3] rather than [2][3]
    while True:
        body, n = re.subn(r"\\CITE\{([^}]+)\}\\CITE\{([^}]+)\}", r"\\CITE{\1,\2}", body)
        if not n:
            break
    return body


PREAMBLE = r"""%% ---------------------------------------------------------------
%%  MAPLE -- generated by scripts/build_paper_latex.py
%%  Elsevier CAS template (els-cas-templates v2.4), single column.
%%  Target journal: Machine Learning with Applications.
%%
%%  Overleaf: New Project -> Upload Project -> this zip.
%%  Compiler: pdfLaTeX.  Recompile TWICE so citations resolve.
%%
%%  For the two-column layout the journal prints in, change cas-sc to
%%  cas-dc on the line below. Expect wide equations and the 9-column
%%  Table 2 to need manual attention if you do -- see README.md.
%% ---------------------------------------------------------------
%% If Overleaf still reports "compile timed out", uncomment the next
%% line. It replaces every figure with an outlined box of the same size
%% and never reads the 63 PNGs, which is the bulk of the compile time.
%% Pagination, citations and cross-references all come out identical, so
%% it is the quickest way to get the bibliography to resolve; comment it
%% out again for the version you send on.
%\PassOptionsToPackage{draft}{graphicx}

%% cas-sc loads hyperref itself, so its options must be pre-declared.
%% Without this every float logs "destination with the same identifier
%% has been already used" and its PDF links land on the wrong page.
\PassOptionsToPackage{hypertexnames=false}{hyperref}
\documentclass[a4paper,fleqn]{cas-sc}

%% cas-sc.cls already loads graphicx, amsmath, amsfonts, amssymb, booktabs,
%% array, makecell, multirow, colortbl, dcolumn, xcolor, setspace and
%% hyperref, and defines \newdefinition, \newproof, \printcredits and the
%% highlights environment. Only genuinely missing packages are added here.
\usepackage[authoryear,longnamesfirst]{natbib}
\usepackage{tabularx}
\usepackage{algorithm}
\usepackage{algpseudocode}

%% 63 figures and 13 tables all asking for [t] overflow LaTeX's 18-float
%% queue -- "Too many unprocessed floats", followed by hundreds of junk
%% errors as the queue corrupts. morefloats raises the ceiling; placeins
%% drains the queue at each section boundary.
\usepackage{morefloats}
\usepackage[section]{placeins}

\renewcommand{\tabularxcolumn}[1]{>{\raggedright\arraybackslash}p{#1}}

%% cross-reference helpers emitted by the converter.
%% \CITE maps to \citep: every citation in this manuscript is parenthetical
%% ("the MRL Eye Dataset (Fusek, 2018) holds..."), never a running author.
\newcommand{\CITE}[1]{\citep{#1}}
\newcommand{\EQREF}[1]{\eqref{eq:#1}}
\newcommand{\FIGREF}[1]{Fig.~\ref{fig:#1}}
\newcommand{\FIGRANGE}[2]{Figs.~\ref{fig:#1}--\ref{fig:#2}}
\newcommand{\TABREF}[1]{Table~\ref{tab:#1}}
\newcommand{\ALGREF}[1]{Algorithm~\ref{alg:#1}}

\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}
\newdefinition{definition}{Definition}
\newproof{proof}{Proof}

\begin{document}
\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}

\shorttitle{SHORTTITLE_HERE}
\shortauthors{SHORTAUTHOR_HERE}

\title[mode = title]{TITLE_HERE}

\author[1]{AUTHOR_HERE}
\cormark[1]
\ead{EMAIL_HERE}
\credit{CREDIT_HERE}

\affiliation[1]{organization={DEPT_HERE, INST_HERE},
                city={CITY_HERE},
                country={COUNTRY_HERE}}

\cortext[1]{Corresponding author}

\begin{abstract}
ABSTRACT_HERE
\end{abstract}

\begin{highlights}
HIGHLIGHTS_HERE
\end{highlights}

\begin{keywords}
KEYWORDS_HERE
\end{keywords}

\maketitle

INPUTS_HERE

\printcredits

\bibliographystyle{cas-model2-names}
\bibliography{refs}

\end{document}
"""

# Paper section -> (output file, markdown heading prefix, source parts).
# Routing is by heading text, not by source file: parts 2+3 are both Section 3
# and parts 5a/5b/5c are all Section 5, so a per-file mapping would be wrong.
#   (number, output stem, markdown heading, source parts, numbered?)
# Back matter (CRediT, competing interest, funding, data availability) must be
# unnumbered, or CRediT prints as "8" and the paper appears to have 8 sections.
SECTION_MAP = [
    (1, "01-introduction",   "1. Introduction",                 "part1_intro.md", True),
    (2, "02-related-work",   "2. Related work",                 "part1_intro.md", True),
    (3, "03-methodology",    "3. Methodology",                  "part2_method_a.md + part3_method_b.md", True),
    (4, "04-experimental",   "4. Experimental settings",        "part4_experimental.md", True),
    (5, "05-results",        "5. Results and discussion",       "part5a/5b/5c_*.md", True),
    (6, "06-threats",        "6. Threats to validity",          "part6_conclusion.md", True),
    (7, "07-conclusion",     "7. Conclusion and future work",   "part6_conclusion.md", True),
    (8, "08-backmatter",     "Declaration of competing interest", "part6_conclusion.md", False),
]
# The CRediT statement is deliberately absent: CAS builds it from the
# \credit{} in the author block via \printcredits, so keeping the hand-written
# section too would print the contribution list twice.
BACKMATTER_HEADS = ("Declaration of competing interest",
                    "Funding",
                    "Data and code availability")
DROP_HEADS = ("CRediT authorship contribution statement",)


def split_sections(md: str) -> dict[str, str]:
    """Split the concatenated markdown on top-level '## ' headings.

    Returns {heading text: body including its own heading line}. Back-matter
    headings are merged under the first of them so they land in one file.
    """
    parts: dict[str, str] = {}
    cuts = [(m.start(), m.group(1).strip())
            for m in re.finditer(r"^##\s+(.*)$", md, flags=re.M)]
    for i, (pos, head) in enumerate(cuts):
        end = cuts[i + 1][0] if i + 1 < len(cuts) else len(md)
        parts[head] = md[pos:end]
    for h in DROP_HEADS:
        parts.pop(h, None)
    # fold the remaining back-matter headings into one entry
    merged = "".join(parts.pop(h, "") for h in BACKMATTER_HEADS)
    if merged:
        parts[BACKMATTER_HEADS[0]] = merged
    return parts


def parse_scope(argv: list[str]) -> set[int]:
    """--only 1,2 -> {1,2}; absent -> every section."""
    for i, a in enumerate(argv):
        if a == "--only" and i + 1 < len(argv):
            return {int(x) for x in argv[i + 1].split(",") if x.strip().isdigit()}
        if a.startswith("--only="):
            return {int(x) for x in a.split("=", 1)[1].split(",") if x.strip().isdigit()}
    return {n for n, _, _, _, _ in SECTION_MAP}


def stub_tex(num: int, heading: str, source: str, scope: set[int],
             numbered: bool) -> str:
    """A placeholder section that still compiles and numbers correctly."""
    title = re.sub(r"^\d+\.\s*", "", heading)
    want = ",".join(str(n) for n in sorted(scope | {num}))
    label = "Section %d" % num if numbered else "Back matter"
    star = "" if numbered else "*"
    return (f"%% {label} -- not yet written in LaTeX.\n"
            f"%% Source: reports/paper/{source}\n"
            f"%% Regenerate with:  python scripts/build_paper_latex.py --only {want}\n"
            f"\\section{star}{{{title}}}\n"
            f"\\textit{{[This section is not yet transferred to \\LaTeX{{}}.]}}\n")


def main() -> None:
    scope = parse_scope(sys.argv)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "sections").mkdir(exist_ok=True)

    md = "\n\n".join((PARTS_DIR / p).read_text(encoding="utf-8") for p in PARTS)

    # lift front matter out of the body
    md = re.sub(r"^#\s+.*$", "", md, count=1, flags=re.M)
    kw = re.search(r"^\*\*Keywords:\*\*\s*(.*)$", md, flags=re.M)
    keywords = kw.group(1) if kw else ""
    md = md.replace(kw.group(0), "", 1) if kw else md

    hl = re.search(r"## Highlights\n(.*?)\n## Abstract", md, flags=re.S)
    highlights = hl.group(1) if hl else ""
    md = md.replace(hl.group(0), "## Abstract", 1) if hl else md

    ab = re.search(r"## Abstract\n(.*?)\n---\n", md, flags=re.S)
    abstract = ab.group(1).strip() if ab else ""
    md = md.replace(ab.group(0), "", 1) if ab else md

    stats = {"eq": 0, "alg": 0, "tbl": 0, "fig": 0, "wrapped": 0,
             "figs": [], "prose_eq": [], "pending_caption": None}

    # ---- one .tex per paper section -------------------------------------
    # --standalone: emit ONE main.tex holding only the in-scope sections, with
    # no placeholder files and no uncited references, so a partial draft looks
    # like a short paper rather than a whole project with most of it missing.
    standalone = "--standalone" in sys.argv
    chunks = split_sections(md)
    written, stubbed, bodies = [], [], []
    for num, fname, head, source, numbered in SECTION_MAP:
        target = OUT / "sections" / f"{fname}.tex"
        if num in scope and head in chunks:
            stats["pending_caption"] = None      # captions never span sections
            body = tidy_body(convert(chunks[head], stats))
            if not numbered:
                body = body.replace("\\section{", "\\section*{")
            bodies.append(body)
            if not standalone:
                target.write_text(
                    f"%% Generated from reports/paper/{source}\n"
                    f"%% Do not edit by hand: re-run scripts/build_paper_latex.py\n\n"
                    + body + "\n", encoding="utf-8")
            written.append(num)
        elif not standalone:
            target.write_text(stub_tex(num, head, source, scope, numbered),
                              encoding="utf-8")
            stubbed.append(num)

    if standalone:
        shutil.rmtree(OUT / "sections", ignore_errors=True)
        inputs = "\n\n".join(bodies)
    else:
        inputs = "\n".join(r"\input{sections/%s}" % f
                           for _, f, _, _, _ in SECTION_MAP)

    pre = PREAMBLE
    # Longest placeholder first: TITLE_HERE is a substring of SHORTTITLE_HERE
    # (and AUTHOR_HERE of SHORTAUTHOR_HERE), so replacing the short names first
    # leaves "SHORT" welded to the full title.
    pre = pre.replace("SHORTTITLE_HERE", SHORT_TITLE)
    pre = pre.replace("SHORTAUTHOR_HERE", SHORT_AUTHOR)
    pre = pre.replace("TITLE_HERE", inline(TITLE))
    pre = pre.replace("CREDIT_HERE", CREDIT)
    pre = pre.replace("AUTHOR_HERE", AUTHOR).replace("EMAIL_HERE", EMAIL)
    pre = (pre.replace("DEPT_HERE", DEPT).replace("INST_HERE", INST)
              .replace("CITY_HERE", CITY).replace("COUNTRY_HERE", COUNTRY))
    pre = pre.replace("ABSTRACT_HERE", inline(abstract))

    # CAS provides a highlights environment, so they sit in the paper itself
    hl_items = "\n".join(r"\item %s" % inline(x[2:])
                         for x in highlights.split("\n") if x.startswith("- "))
    pre = pre.replace("HIGHLIGHTS_HERE", hl_items)
    pre = pre.replace("KEYWORDS_HERE",
                      " \\sep\n".join(prose(k.strip()) for k in keywords.split(";")))
    tex = pre.replace("INPUTS_HERE", inputs)
    (OUT / "main.tex").write_text(tex, encoding="utf-8")

    # a standalone draft ships only the references it actually cites, so the
    # bibliography matches the pages in front of the reader
    cited = set()
    for m in re.finditer(r"\\CITE\{([^}]+)\}", tex):
        cited.update(k.strip() for k in m.group(1).split(","))
    (OUT / "refs.bib").write_text(
        make_bib(cited if standalone else None), encoding="utf-8")

    # ship the CAS class files so the project compiles without relying on the
    # host TeX installation carrying the bundle
    tmpl = 0
    if TEMPLATE_ZIP.exists():
        with zipfile.ZipFile(TEMPLATE_ZIP) as tz:
            names = {Path(n).name: n for n in tz.namelist()}
            for want in TEMPLATE_FILES:
                if want in names:
                    (OUT / want).write_bytes(tz.read(names[want]))
                    tmpl += 1
    else:
        print(f"  ! template bundle not found: {TEMPLATE_ZIP}")

    # only the figures the in-scope sections actually reference get copied,
    # so a text-only slice ships as a ~40 KB upload rather than 17.7 MB
    for stale in (OUT / "figures").glob("*.png"):
        stale.unlink()
    copied, saved = 0, 0
    for rel in stats["figs"]:
        src = FIGDIR / rel
        if src.exists():
            dst = OUT / "figures" / flat_figname(rel)
            saved += copy_figure(src, dst)
            copied += 1
        else:
            print("  ! missing figure:", src)
    if saved:
        print(f"  downsampled: {saved/1e6:.1f} MB saved "
              f"(cap {FIG_MAX_PX} px = {FIG_MAX_PX/300:.1f} in at 300 dpi)")
    n_files = len(list((OUT / "figures").glob("*.png")))
    if copied and n_files != len(set(stats["figs"])):
        print(f"  ! figure name collision: {len(set(stats['figs']))} sources "
              f"-> {n_files} files on disk")
    if not copied and standalone:
        shutil.rmtree(OUT / "figures", ignore_errors=True)   # nothing to carry
    elif not copied:
        (OUT / "figures" / "README.txt").write_text(
            "Figures are copied here by scripts/build_paper_latex.py for whichever\n"
            "sections are in scope. The sections built so far reference none.\n",
            encoding="utf-8")

    done = ", ".join(str(n) for n in written) or "none"
    todo = ", ".join(str(n) for n in stubbed) or "none"
    if standalone:
        (OUT / "README.md").write_text(
            "# MAPLE -- partial draft\n\n"
            f"Contains the front matter and section(s) **{done}** only. This is a\n"
            "self-contained short paper: one `main.tex`, no placeholder sections,\n"
            "and a `refs.bib` trimmed to just the works these pages cite.\n\n"
            "## Upload to Overleaf\n\n"
            "1. Overleaf -> **New Project** -> **Upload Project** -> this zip.\n"
            "2. Menu -> **Compiler: pdfLaTeX**, **Main document: main.tex**.\n"
            "3. **Recompile twice** -- the first pass records the citations, the\n"
            "   second prints them. `[?]` in between is normal.\n\n"
            "## Files\n\n"
            "```\n"
            "main.tex        everything: preamble, front matter, the section text\n"
            "refs.bib        only the references cited on these pages\n"
            "highlights.tex  Elsevier Highlights; compile separately if needed\n"
            "```\n\n"
            "Write directly in `main.tex`. Later sections are added by pasting\n"
            "their text in before `\\bibliographystyle`.\n",
            encoding="utf-8")
    else:
        (OUT / "README.md").write_text(
        "# MAPLE -- Elsevier CAS manuscript\n\n"
        "Target journal: **Machine Learning with Applications**.\n"
        "Class: `cas-sc` (Elsevier CAS bundle v2.4), single column.\n\n"
        "## Upload\n\n"
        "1. Zip this folder (or use `reports/MAPLE_overleaf.zip`).\n"
        "2. Overleaf -> **New Project** -> **Upload Project** -> select the zip.\n"
        "3. Menu -> **Compiler: pdfLaTeX**, **Main document: main.tex**.\n"
        "4. Press **Recompile twice** -- the first pass writes the cross-reference\n"
        "   and citation files, the second resolves them. Until then references\n"
        "   show as `[?]`, which is normal.\n\n"
        "## If the compile times out\n\n"
        "This is 116 pages with 63 figures and 306 equations, and Overleaf's\n"
        "free plan caps compile time. In order of what to try:\n\n"
        "1. **Recompile dropdown -> Fast [draft]**. It runs LaTeX once instead\n"
        "   of the full loop. Once a normal compile has got as far as running\n"
        "   BibTeX, `.aux` and `.bbl` are cached in the project, so a single\n"
        "   pass is enough to produce a PDF with every citation resolved.\n"
        "2. **Uncomment the `\\PassOptionsToPackage{draft}{graphicx}` line** at\n"
        "   the top of `main.tex`. Every figure becomes an outlined box of the\n"
        "   same size and none of the PNGs is read, which is the bulk of the\n"
        "   compile. Pagination and references are unchanged. Comment it back\n"
        "   out for the version you send on.\n"
        "3. Compile one section at a time by commenting out `\\input` lines in\n"
        "   `main.tex`. Section numbering stays correct.\n\n"
        "The figures are deliberately plain 8-bit RGB PNGs, capped at 1400 px.\n"
        "That is the one form pdfTeX copies straight into the PDF without\n"
        "decoding it (the log says `(PNG copy)`). Palette or RGBA PNGs are\n"
        "smaller on disk but are decoded and recompressed one by one, which\n"
        "costs far more compile time than their size saves.\n\n"
        "## The ten BibTeX `empty pages` warnings\n\n"
        "Expected, and not worth fixing. They are ICLR papers, CVPR and ASRU\n"
        "workshop papers and arXiv preprints, none of which is paginated.\n"
        "The Elsevier style prefers a `pages` field and says so; the entries\n"
        "have no page numbers to give. They are warnings, not errors, and the\n"
        "bibliography renders correctly without them.\n\n"
        "## Layout\n\n"
        "```\n"
        "main.tex               preamble, CAS front matter, \\input lines\n"
        "sections/              one file per paper section\n"
        "refs.bib               structured BibTeX (author-year needs real fields)\n"
        "figures/               all figures\n"
        "cas-sc.cls, cas-dc.cls, cas-common.sty, cas-model2-names.bst\n"
        "                       the Elsevier CAS bundle, included so the\n"
        "                       project compiles without relying on the host\n"
        "```\n\n"
        "## Single vs two column\n\n"
        "This builds with `cas-sc` (single column). Elsevier accepts single\n"
        "column for submission and typesets the two-column journal layout at\n"
        "production. To preview the printed look, change `cas-sc` to `cas-dc`\n"
        "on the `\\documentclass` line -- but expect the wider equations and the\n"
        "nine-column Table 2 to overflow a 3-inch column and need manual work.\n\n"
        "## Citations\n\n"
        "Author-year (`cas-model2-names`), as the journal uses. `\\CITE{key}`\n"
        "maps to `\\citep`, giving \"(Fusek, 2018)\". For a running citation --\n"
        "\"Fusek (2018) showed\" -- use `\\citet{key}` instead.\n\n"
        f"Written in LaTeX so far: **section(s) {done}**. "
        f"Still placeholders: **{todo}**.\n\n"
        "## Adding the next section\n\n"
        "Everything is generated from the markdown in `reports/paper/`, so edit\n"
        "there and re-run rather than editing the `.tex` by hand -- a hand edit is\n"
        "lost on the next build. To add Section 3:\n\n"
        "```\n"
        "python scripts/build_paper_latex.py --only 1,2,3 --zip\n"
        "```\n\n"
        "`main.tex` already `\\input`s every section, so nothing else changes.\n\n"
        "## Switching venue\n\n"
        "The class is `elsarticle` (Elsevier). For a conference, swap the\n"
        "`\\documentclass` line for `\\documentclass[conference]{IEEEtran}` or the\n"
        "ACM `acmart` class, and delete the `\\journal{...}` line.\n\n"
        "`refs.bib` uses `@misc{key, note={...}}` entries carrying the formatted\n"
        "IEEE strings, so the reference list matches the .docx exactly. Convert\n"
        "them to typed BibTeX entries if the venue requires structured metadata.\n",
        encoding="utf-8")

    print(f"wrote {OUT}")
    print(f"  scope      : sections {sorted(scope)}")
    print(f"  written    : {written}")
    print(f"  stubbed    : {stubbed}")
    n_inputs = 0 if standalone else len(SECTION_MAP)
    print(f"  layout     : {'standalone (single main.tex)' if standalone else 'modular'}")
    print(f"  main.tex   : {len(tex):,} chars, {n_inputs} \\input lines")
    print(f"  refs.bib   : {len(cited) if standalone else len(BIB)} entries"
          f"{' (cited only)' if standalone else ''}")
    print(f"  figures    : {copied}/{len(stats['figs'])} copied")
    print(f"  equations  : {stats['eq']}")
    print(f"  algorithms : {stats['alg']}")
    print(f"  tables     : {stats['tbl']}")
    print(f"  figure envs: {stats['fig']}")
    if stats["prose_eq"]:
        print(f"  ! {len(stats['prose_eq'])} equations contain English words and "
              f"should be read once: {stats['prose_eq'][:20]}")
    # every \input must resolve or the compile dies at that line
    # (a standalone build has no \input at all -- the body is inlined)
    if not standalone:
        for _, fname, _, _, _ in SECTION_MAP:
            if not (OUT / "sections" / f"{fname}.tex").exists():
                print(f"  ! missing section file: sections/{fname}.tex")

    if "--zip" in sys.argv:
        z = ROOT / "reports" / "MAPLE_overleaf.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(OUT.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(OUT))
        print(f"  zipped -> {z}  ({z.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
