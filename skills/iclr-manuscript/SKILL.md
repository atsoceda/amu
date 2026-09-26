---
name: iclr-manuscript
description: >-
  Writes, renders and checks the AMU ICLR 2027 manuscript (paper.qmd, manuscript/sections,
  manuscript/figures/make_iclr_figures.py): Quarto-to-LaTeX pitfalls, figures drawn at printed
  size from committed results, float tables, pending-result markers, page counting and visual
  checks. Use when editing, extending or rendering the paper, or when adding a figure or table.
license: MIT
compatibility: >-
  Repo-local Quarto 1.9 and latexmk via bin/render-paper; /Users/anthony/miniconda3/bin/python
  with matplotlib for figures; poppler (pdfinfo, pdftotext, pdftoppm) for checks.
metadata:
  version: "1.0"
  updated: "2026-09-26"
---

# ICLR manuscript workflow

The live source is the ICLR 2027 extension; the NeurIPS workshop version (v35) is frozen
(see `AGENTS.md`). Never render the `neurips` target.

## Structure

- `paper.qmd` includes `manuscript/sections/01-introduction.qmd` ... `10-appendix.qmd`;
  title and abstract are in `manuscript/metadata.yml` (and the title again in `paper.qmd`).
- Pending results are marked in the text with `\pending{...}` (defined in `paper.qmd`,
  renders in red). Search for it to list open items; remove each marker when the result
  lands, and update the number everywhere it appears (abstract, Table 1, sections).
- Every number must come from `experiments/*/results` (or the experiment README). Re-read
  the JSON before quoting; state shares with the denominator they use (persistence of the
  same run).

## Figures

- `manuscript/figures/make_iclr_figures.py` builds every ICLR figure from committed results
  and skips pending runs, so re-run it after each sync:
  `/Users/anthony/miniconda3/bin/python manuscript/figures/make_iclr_figures.py`.
- Draw at printed size: ICLR text width is 5.5 in (`W` in the script). A figure drawn at
  7 in and scaled down shrinks 7 pt text to 5.5 pt; keep fonts at 6 to 7.5 pt at 5.5 in.
- Glyphs such as ⟨ ⟩ ↵ are missing from Arial: set `font.family` to the list
  `["Arial", "DejaVu Sans"]` (a `font.sans-serif` list does not trigger fallback), and
  never use Helvetica (`failed to load glyph`).
- One colour per path everywhere (Okabe-Ito; see the script header). Check every figure by
  opening the PNG: overlapping legends, labels and titles are the usual failure.

## Tables

- Quarto pipe tables become `longtable`s that split across pages and wrap narrow columns.
  Use a table float instead:

  ````
  ::: {#tbl-name}

  ```{=latex}
  \footnotesize
  \begin{tabular}{@{}lcc@{}} \toprule ... \bottomrule \end{tabular}
  ```

  Caption text.

  :::
  ````

  Cross-references (`@tbl-name`) keep working. Inside raw LaTeX use `$...$`, `\%` and
  `\textbf{}`.

## YAML pitfalls

- A `header-includes` entry containing `: ` (for example `[Pending: #1]`) parses as a
  mapping and the render fails with "failed to be a string": write `header-includes` as a
  literal block (`header-includes: |`).

## Render and check

1. `bin/render-paper iclr submission` (outputs in `dist/iclr-submission/`).
2. Pages: `pdfinfo dist/iclr-submission/paper.pdf`; find where the main text ends (the
   AI use statement follows the discussion) with `pdftotext -f N -l N`. ICLR allows 9 pages
   of main text at submission; report the count, do not cut text to fit (`AGENTS.md`).
3. Look at the pages: `pdftoppm -r 70 -png dist/iclr-submission/paper.pdf pg` and tile them
   into contact sheets; check figure legibility, float placement and table breaks.
4. Check unresolved references: `pdftotext ... | grep -n "??\|@fig\|@tbl\|@sec"`.
