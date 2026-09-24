# Publishing Workflow

This repository uses Quarto as the manuscript controller and LaTeX as the PDF
backend. Source text lives in `paper.qmd` and `manuscript/sections/`; generated
LaTeX and PDFs are build artifacts.

## Versions

- **NeurIPS 2026 workshop submission (v35): frozen.** The submitted PDF,
  generated `.tex`, manifest and checksums are in
  `submissions/neurips-2026-workshop-v35/` (read-only). Its source is Git tag
  `neurips-2026-workshop-v35`. Do not render the `neurips` target from the live
  source; rebuild the workshop version only from the tag, in a separate worktree.
- **ICLR 2027 main track: active.** The live source is the ICLR extension.

## Build

```bash
bin/render-paper iclr submission
bin/render-paper iclr camera-ready
```

The `iclr` profiles use the official ICLR 2027 files vendored unmodified under
`_extensions/iclr/`. Fonts the style requires (Helvetica, Courier) come from TeX
Live packages vendored under `_extensions/texmf/fonts/`, which the render script
exposes via `TEXMFHOME`; the `.bst` is exposed to BibTeX as
`_extensions/texmf/bibtex/bst/iclr2027conference.bst` (a symlink, because Quarto
escapes underscores in `biblio-style`). The `icml` and `neurips` profiles remain
for reference and for rebuilding the tagged workshop version.

Each build writes a PDF, generated `.tex`, and `build-manifest.json` under
`dist/<target>-<mode>/`.

## Local Tools

The first setup uses a no-admin Quarto CLI under `.tools/quarto` and a vendored
CTAN `latexmk` script under `.tools/bin`. These paths are ignored by Git.

Quarto is run with `HOME=.home` so its cache stays inside the repository's
ignored local state rather than `~/Library/Caches`.

## Editing Rules

- Edit `.qmd` source files and `manuscript/references.bib`.
- Do not edit generated `.tex` files.
- Keep all figure and table paths relative to the repository.
- Vendor official conference `.sty`, `.cls`, and `.bst` files under
  `_extensions/<venue>/` before relying on a target for submission.
