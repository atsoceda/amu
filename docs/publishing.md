# Paper versions and publishing

This repository uses Quarto as the manuscript controller and LaTeX as the PDF
backend. The root [README](../README.md) is the reader-facing version index;
`paper-versions.json` is the machine-readable rebuild registry. The live source
is `paper.qmd` plus `manuscript/`. Historical source is preserved by annotated
Git tags, not by keeping permanent research branches.

## Versions

- **NeurIPS 2026 workshop v35: submitted and frozen.** The exact PDF, generated
  TeX, original manifest and checksums are in
  `submissions/neurips-2026-workshop-v35/`. Source tag:
  `neurips-2026-workshop-v35`. The paper was accepted.
- **NeurIPS 2026 workshop camera-ready: in preparation.** Edited and rendered
  only on branch `paper/neurips-workshop` (from the v35 tag, never merged into
  `main`; see its `CAMERA-READY.md`). Preview it from `main` with
  `bin/build-paper-version neurips-2026-workshop-camera-ready`. When uploaded,
  tag it and archive the exact files under `submissions/` like v35.
- **ICLR 2027 deadline draft: not submitted and frozen.** Its last local PDF,
  generated TeX, original manifest and checksums are in
  `milestones/iclr-2027-deadline-unsubmitted/`. Source tag:
  `iclr-2027-deadline-unsubmitted`.
- **Working paper: venue undecided** (another conference or a journal; ICLR is
  out). The current source is on `main`, rendered with an internal ICLR-like
  layout but no claim of ICLR review. Check it against the committed results with
  `bin/verify-paper`.

Do not edit frozen archive files. The historical manifests are preserved as
original records, including any `git_dirty` status. The version builder checks
each archive's SHA-256 and compares the tagged-source rebuild's TeX hash with
the archived TeX; a rebuilt PDF may differ in embedded timestamps.

## Build

```bash
bin/check-paper-env
bin/build-paper-version neurips-2026-workshop-v35
bin/build-paper-version iclr-2027-deadline-unsubmitted
bin/build-paper-version working
```

`bin/build-paper-version` renders frozen source tags in isolated temporary Git
worktrees, without switching or modifying your current branch. It copies rebuilt
outputs to ignored `dist/version-builds/<version>/`. The `working` target renders
the live source to ignored `dist/working-draft/`. `bin/render-paper neurips` and
`bin/render-paper iclr` are intentionally blocked from the live source; the
tagged historical scripts still know how to render those profiles.

The working profile loads the unmodified ICLR 2027 style under
`_extensions/iclr/`, then replaces its visible "under review" language using
`_extensions/working-paper/header-draft.tex`. This preserves the familiar layout
but is not an official venue submission build. The fonts and bibliography style
are vendored under `_extensions/texmf/` and `_extensions/iclr/` respectively.
When a new venue is chosen, add or validate its own official target.

Each working build writes a PDF, generated TeX, manifest, and an additional
numbered PDF under `dist/working-draft/`. Frozen builds produce temporary
numbered PDFs only inside their isolated worktrees; the exact historical PDFs
remain in the archive folders.

## Local Tools

The development machine has Quarto 1.9.38 under `.tools/quarto` and latexmk 4.88
under `.tools/bin`; both are gitignored. On a fresh clone, install matching
Quarto and a TeX Live/MacTeX distribution with latexmk, pdflatex and bibtex,
then run `bin/check-paper-env`. The builder can use installed system commands
when `.tools/` is absent. The repository does not yet pin a full redistributable
TeX environment; see [open-source readiness](open-source-readiness.md).

Quarto is run with `HOME=.home` so its cache stays inside the checkout's ignored
local state rather than `~/Library/Caches`. Temporary tagged worktrees receive
their own `.home`; the builder removes them after a successful or failed render.

## Editing Rules

- Edit live `.qmd` source files and `manuscript/references.bib` only for the
  working paper. Historical changes require a new version, never archive edits.
- Do not edit generated `.tex` files.
- Keep all figure and table paths relative to the repository.
- Vendor official conference `.sty`, `.cls`, and `.bst` files under
  `_extensions/<venue>/` before relying on a target for submission.
- After a real submission, tag its clean source commit and archive the exact
  PDF, TeX, manifest and checksums under `submissions/`. For an unsubmitted
  checkpoint use `milestones/`. Update `paper-versions.json` and the root README.
