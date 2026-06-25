# Publishing Workflow

This repository uses Quarto as the manuscript controller and LaTeX as the PDF
backend. Source text lives in `paper.qmd` and `manuscript/sections/`; generated
LaTeX and PDFs are build artifacts.

## Build

```bash
bin/render-paper neurips submission
bin/render-paper neurips camera-ready
bin/render-paper icml submission
bin/render-paper iclr submission
```

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
