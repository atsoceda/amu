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
| NeurIPS 2026 workshop camera-ready | **In preparation** (paper accepted) | Run `bin/build-paper-version neurips-2026-workshop-camera-ready`, then open `dist/version-builds/neurips-2026-workshop-camera-ready/paper.pdf` | Branch [`paper/neurips-workshop`](https://github.com/atsoceda/amu/tree/paper/neurips-workshop) (from the v35 tag; never merged into `main`) | `bin/build-paper-version neurips-2026-workshop-camera-ready` |
| NeurIPS 2026 workshop v35 | Frozen **submitted** paper (accepted) | [PDF](submissions/neurips-2026-workshop-v35/paper.pdf) and [record](submissions/neurips-2026-workshop-v35/README.md) | [Source tag](https://github.com/atsoceda/amu/tree/neurips-2026-workshop-v35) | `bin/build-paper-version neurips-2026-workshop-v35` |

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

Everything under `dist/` is disposable build output and is not tracked. The
submitted and milestone PDFs that matter are archived in `submissions/` and
`milestones/`; older numbered drafts (`paper_N.pdf`) are backed up on Google
Drive, so `dist/` can be deleted to free disk space and rebuilt on demand.

## Reproduce the findings

Three levels, from cheapest to complete:

1. **Rebuild a submitted paper exactly**: `bin/build-paper-version <version>`
   (above).
2. **Verify the working paper from the committed results** (no model weights,
   minutes, numpy and matplotlib only):

   ```bash
   bin/verify-paper
   ```

   It regenerates every figure into a temporary folder, requires the exact set
   of result files recorded in `manuscript/figures/figure_inputs.json`,
   compares each figure pixel by pixel with the committed PNG, and checks the
   headline numbers in the text against the summaries
   ([`manuscript/claims.py`](manuscript/claims.py)). Known discrepancies awaiting
   an author decision are reported as `OPEN`.
3. **Rerun the experiments**: install [`env/`](env/README.md) (pinned packages,
   exact model revisions) and follow each `experiments/<name>/README.md`. Data
   from Hanna & Ameisen's model-planning repository is vendored under
   [`third_party/`](third_party/hanna-ameisen-model-planning/SOURCE.md). Runs of
   8B and larger models need a large-memory machine
   ([`skills/mac-studio-remote`](skills/mac-studio-remote/SKILL.md)).

## Research record

- `experiments/<name>/README.md` states the question, frozen design,
  deviations, results (including failures), and exact run commands.
- `experiments/<name>/results/report.md` explains the committed row-level
  outcomes and summaries. See [research-record.md](docs/research-record.md).
- `manuscript/figures/make_iclr_figures.py` regenerates manuscript figures from
  committed results; figure PNGs are also tracked, so paper builds do not need
  model inference. When a figure's inputs change on purpose, run
  `bin/verify-paper --update-inputs` and commit the manifest with the figure.
- When an edit changes a headline number, update its entry in
  `manuscript/claims.py` in the same commit.
- [Publishing workflow](docs/publishing.md) explains versioning and build
  provenance. [Open-source readiness](docs/open-source-readiness.md) lists the
  remaining checks before inviting external replication.

Model weights and large intermediate tensors are not committed; individual
experiment READMEs identify their model snapshots and data sources.
