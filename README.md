# AMU: causal routes in autoregressive models

This repository contains the manuscript, experiments, and figure-generation
code for a study of how future-relevant information reaches later model outputs.
The current paper is a **working draft with no selected submission venue**.
Its PDF uses an ICLR-like *internal layout*; it is not under ICLR review.

## Start here

| Paper version | Status | Read the paper | Source | Rebuild |
| --- | --- | --- | --- | --- |
| Current working paper | Active; venue undecided | Run `bin/build-paper-version working`, then open `dist/working-draft/paper.pdf` | `paper.qmd`, `manuscript/` on `main` | `bin/build-paper-version working` |
| ICLR 2027 deadline draft | Frozen **unsubmitted** milestone | [PDF](milestones/iclr-2027-deadline-unsubmitted/paper.pdf) and [record](milestones/iclr-2027-deadline-unsubmitted/README.md) | [Source tag](https://github.com/atsoceda/amu/tree/iclr-2027-deadline-unsubmitted) | `bin/build-paper-version iclr-2027-deadline-unsubmitted` |
| NeurIPS 2026 workshop v35 | Frozen **submitted** paper | [PDF](submissions/neurips-2026-workshop-v35/paper.pdf) and [record](submissions/neurips-2026-workshop-v35/README.md) | [Source tag](https://github.com/atsoceda/amu/tree/neurips-2026-workshop-v35) | `bin/build-paper-version neurips-2026-workshop-v35` |

A tag is a human-readable name for one frozen Git commit. Tags are not working
branches (`git tag --list` shows them); the table above makes them visible so
you never need to remember or check them out manually. Run
`bin/build-paper-version --list` to list buildable versions. The rebuild command
uses an isolated temporary checkout and does not switch your current branch.

## Rebuild a paper

Run `bin/check-paper-env` first. The build needs Quarto 1.9.38, a TeX
Live/MacTeX installation with `latexmk`, `pdflatex` and `bibtex`, Python 3, and
Git. The repository vendors the venue style files and paper-specific fonts,
but not model weights or the full TeX distribution. Existing repo-local tools
under `.tools/` are used automatically; system commands also work.

```bash
bin/build-paper-version neurips-2026-workshop-v35
bin/build-paper-version iclr-2027-deadline-unsubmitted
bin/build-paper-version working
```

Frozen builds verify the archived checksums and compare rebuilt TeX to the
archived SHA-256. The exact historical PDFs are tracked in their archive
folders; newly rendered PDFs may differ byte-for-byte because of build
timestamps. Rebuilt outputs go under ignored `dist/version-builds/`, while the
working draft goes under ignored `dist/working-draft/`.

## Research record

- `experiments/<name>/README.md` states the question, frozen design,
  deviations, results (including failures), and exact run commands.
- `experiments/<name>/results/report.md` explains the committed row-level
  outcomes and summaries. See [research-record.md](docs/research-record.md).
- `manuscript/figures/make_iclr_figures.py` regenerates manuscript figures from
  committed results; figure PNGs are also tracked, so paper builds do not need
  model inference.
- [Publishing workflow](docs/publishing.md) explains versioning and build
  provenance. [Open-source readiness](docs/open-source-readiness.md) lists the
  remaining checks before inviting external replication.

Model weights and large intermediate tensors are not committed; individual
experiment READMEs identify their model snapshots and data sources.
