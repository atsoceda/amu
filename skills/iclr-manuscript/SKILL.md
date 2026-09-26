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
  version: "1.9"
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

## Wording rules learned from review

Each path has exactly one name (fixed 2026-09-26 after "relay" had come to mean two
things and the paper needed "early relay" and glosses of "carry"):

- **indirect path**: the umbrella for everything that reaches the target through
  intermediate positions (the quantity \(T\)). A group of positions that was not split
  further is "the indirect path through" that group.
- **stored copy**, **late lookup**, **relay**: the three position groups of the indirect
  path. **Relay** means only the path along the remaining generated text, the one a
  monitor could miss. Never write "early relay", and never use "relay" for the umbrella.
- **looked up**: direct retrieval, the stored copy or the late lookup.
- The abstract, introduction and contributions use one top-level trio: **emission,
  lookup, relay**. The measurement split (direct retrieval versus indirect path, and the
  indirect path's three parts) appears only where route accounting is described, with the
  sentence that ties the two: direct retrieval, the stored copy and the late lookup are
  the three forms of lookup; relay is one part of the indirect path. Mixing the trios
  (for example "emission, retrieval, relay" in the abstract and "emission, direct
  retrieval, indirect paths" in the contributions) made the user ask whether relay and
  indirect path are the same thing.
- Say that relay and every indirect path are **hidden-state** paths wherever they are
  first defined (abstract, introduction, method): "passed forward in the hidden states of
  the generated text, without being written". "Along the text" or "through an
  intermediate token" reads as "via the written tokens", which is emission, the opposite
  (user question, 2026-09-26).
- Relay is not "the part through the generated text": the late lookup also passes
  through generated positions (the last three before the target). Relay is the part that
  holds the information across the generated text; the three lookups all read it from a
  fixed prompt position. A path through prompt positions (variable-chain statements, the
  induction key) is an indirect path, never relay.
- **Retrieval** means only **direct retrieval**, the measured path \(D\) (write it in full),
  or a cited author's own term (the "late retrieval" and the circuit that "retrieves" in
  Hanna and Ameisen). The umbrella is **lookup**; the verb "look up" / "looked up" names
  the same defined category, so "retrieval-dominated" becomes "lookup-dominated".
- Do not put a share above 100% in the abstract or introduction; write "at least X%".
  Where a table or section reports one, say why (the other paths or the interaction are
  negative).
- **path**, not "route", except in the method name "route accounting".
- **carry** appears only in the title and the slogan "look it up, don't carry it", as
  plain English; never define it or use it as a path name. For a share of an effect
  write "accounts for X%", never "carries X%".
- Use one name per quantity in figure labels and captions (for example "share of
  necessity"), and do not add a colour bar when every shaded cell prints its value.
- Do not use "private" (retired in the glossary); say hidden or secret.

## YAML pitfalls

- A `header-includes` entry containing `: ` (for example `[Pending: #1]`) parses as a
  mapping and the render fails with "failed to be a string": write `header-includes` as a
  literal block (`header-includes: |`).

## Render and check

Render from the checkout that already has the real `.tools` directory, normally
`/Users/anthony/repos/amu`. The command is `bin/render-paper iclr submission`.
Quarto 1.9.38 is `.tools/quarto/bin/quarto` and latexmk is `.tools/bin/latexmk`.
Fonts are vendored in `_extensions/texmf`. Do not download Quarto, latexmk, or a
TeX tree, and do not `rm` `.tools`.

`.tools/` and `.home/` are gitignored local installs, so a new worktree does not
have them. Do not create a worktree just to render. If a worktree already exists,
link from inside that worktree only:

```bash
ln -s /Users/anthony/repos/amu/.tools .tools
ln -s /Users/anthony/repos/amu/.home .home
```

Never run those in the main checkout, and never copy the worktree's `.tools`
symlink back onto `/Users/anthony/repos/amu/.tools`. That replaces the install
with a symlink to itself. `.gitignore` lists `.tools/`, which ignores a directory
and does not ignore a symlink, so a broken symlink shows up as an untracked file:
do not commit it. If `.tools/quarto/bin/quarto` is not an executable file, stop
and tell the user.

1. `bin/render-paper iclr submission` (outputs in `dist/iclr-submission/`). Every render
   also writes a numbered copy, `paper_<N>.pdf` (N = highest existing number + 1; the user's
   rule, 2026-09-26: numbered versions are easier to distribute). Never overwrite or delete an
   earlier number; quote the number when sending a render to the user.
2. Pages: `pdfinfo dist/iclr-submission/paper.pdf`; find where the main text ends (the
   AI use statement follows the discussion) with `pdftotext -f N -l N`. ICLR allows 9 pages
   of main text at submission; report the count, do not cut text to fit (`AGENTS.md`)
   unless the user asks. When they do, move detail to the appendix (unlimited at ICLR)
   rather than delete it, and state each result once in the main text.
3. Look at the pages: `pdftoppm -r 70 -png dist/iclr-submission/paper.pdf pg` and tile them
   into contact sheets; check figure legibility, float placement and table breaks.
4. Check unresolved references: `pdftotext ... | grep -n "??\|@fig\|@tbl\|@sec"`.
