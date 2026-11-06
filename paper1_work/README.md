# MAPLE — Elsevier CAS manuscript

**MAPLE: A leakage-audited and explainable framework for real-time multimodal
hands-free interaction**

Mohammad Anas Shaik, Vaibhav Agarwal, Gayathri Karthick, Sahar Ahmadzadeh
Department of Computer and Data Science, York St John University, London

Target journal: **Machine Learning with Applications** (Elsevier).
Class: `cas-sc` (Elsevier CAS bundle v2.4), single column.

## Compiling

- Compiler: **pdfLaTeX**. Main document: **main.tex**.
- **Recompile twice.** The first pass records the citations and cross-references,
  the second resolves them. Until then references show as `?`, which is normal.
- Bibliography: `ref.bib`, style `cas-model2-names.bst`, 69 cited entries.

### If the compile times out

116 pages with 63 figures and 306 equations is heavy for Overleaf's free plan.
In order of what to try:

1. **Recompile dropdown → Fast [draft].** Runs LaTeX once instead of the full
   loop. Once a normal compile has reached BibTeX, `.aux` and `.bbl` are cached
   in the project, so one pass produces a PDF with every citation resolved.
2. **Uncomment `\PassOptionsToPackage{draft}{graphicx}`** at the top of
   `main.tex`. Every figure becomes an outlined box of the same size and none of
   the PNGs is read, which is most of the compile time. Pagination and
   references are unchanged. Comment it out again for the version you send on.
3. Comment out `\input` lines in `main.tex` to build one section at a time.
   Section numbering stays correct.

## Layout

```
main.tex          preamble, CAS front matter, author block, \input lines
sections/         one file per paper section, edited by hand
ref.bib           bibliography, 70 entries, ASCII only
figures/          63 figures
cas-sc.cls        Elsevier CAS single-column class
cas-dc.cls        Elsevier CAS double-column class (not used)
cas-common.sty    shared CAS macros
cas-model2-names.bst   author-year bibliography style
```

## Three things that have been changed from the stock template

1. **`cas-common.sty` line 1989.** The first-page footer is hard-coded in the
   class as "Preprint submitted to Elsevier"; it now reads "Preprint submitted
   to Machine Learning with Applications". If you ever re-download the CAS
   bundle, this edit has to be redone.
2. **`cas-model2-names.bst` line 1177.** `format.pages "pages" output.check`
   is now `format.pages output`. The style demanded a `pages` field on every
   `@inproceedings` entry and warned ten times when it was absent -- but the
   ten entries concerned are five ICLR main-conference papers and five
   workshop papers, and **none of those venues paginates at all**, so there
   was nothing to supply. `output` and `output.check` render identically; the
   only difference is the warning, and the 22 conference entries that do carry
   page numbers still print them. Four lines above, the style's own author had
   already used the non-warning form in the crossref branch.
3. **`hypertexnames=false` is passed to hyperref** in `main.tex` before
   `\documentclass`. Without it every float logs "destination with the same
   identifier has been already used" and its PDF links point at the wrong page.

## Things to keep in mind when editing

- **`ref.bib` must stay ASCII.** The class loads the `stix` fonts, whose LS1 and
  LS2 encodings have no UTF-8 mapping, so a raw accented character in the
  bibliography is a compile error. Use LaTeX accents: `K{\"o}stinger`,
  `Soukupov{\'a}`, `{\v{C}}ech`, `1{\texteuro} filter`.
- **Figures are plain 8-bit RGB PNGs capped at 1400 px.** That is the one form
  pdfTeX copies straight into the PDF without decoding it — the log says
  `(PNG copy)`. Palette or RGBA PNGs are smaller on disk but get decoded and
  recompressed one by one, costing far more compile time than their size saves.
- **Floats use `[pos=htbp]`, not `[htbp]`.** The CAS classes redefine the
  `figure` and `table` environments so the optional argument is a keyval list.
  A plain `[htbp]` is parsed as four unknown keys and silently yields an *empty*
  placement, which strands every float until the next `\FloatBarrier`.
- **Do not add page numbers to the ICLR or workshop entries.** Those venues do
  not paginate. The style used to warn about the missing field; that demand
  has been removed from the `.bst` instead (see above). Inventing a page
  range to quiet a warning would put a false claim in the bibliography.
- **One `Underfull \hbox (badness 1137)` on the keywords block is expected.**
  Badness runs to 10,000 and LaTeX reports anything over 1,000, so this is one
  keyword line setting very slightly loose. It is not visible in the PDF.

## Before submitting

- [ ] All four authors have agreed to authorship and approved the manuscript.
- [ ] CRediT roles confirmed with each co-author (`\credit{}` in `main.tex`).
- [ ] Co-author email addresses added if wanted (`\ead{}`); only the
      corresponding author's is required.
- [ ] ORCID added for each author who has one.
- [ ] Department name confirmed as the formal unit the university uses.
- [ ] Funding and competing-interest declarations still accurate.
