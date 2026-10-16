# Handoff: repository structure and reproducibility (2026-10-16)

Supersedes `2026-10-16-paper-versions-and-working-draft.md` (earlier today) for
paper status and repository workflow. Scientific state is unchanged since
`2026-09-26-overnight-scale-and-boundary.md`. Start with the root README.

## Paper lines

| Line | Status | Where |
| --- | --- | --- |
| NeurIPS 2026 workshop, v35 | Submitted and **accepted**; frozen as reviewed | `submissions/neurips-2026-workshop-v35/`, tag `neurips-2026-workshop-v35` |
| NeurIPS 2026 workshop, camera-ready | In preparation; content edits not started (the user wants structure first) | Branch `paper/neurips-workshop` (never merged into `main`); to-do list in its `CAMERA-READY.md` |
| ICLR 2027 deadline draft | Never submitted; ICLR is **out**, no resubmission | `milestones/iclr-2027-deadline-unsubmitted/`, tag of the same name |
| Working paper | Active; next venue (another conference or a journal) not chosen | `paper.qmd`, `manuscript/` on `main` |

- `bin/build-paper-version neurips-2026-workshop-camera-ready` builds the branch
  head in a temporary worktree (output in `dist/version-builds/`). When the
  camera-ready is uploaded: tag it `neurips-2026-workshop-camera-ready`, archive the
  exact PDF/TeX/manifest/checksums under `submissions/` on `main`, and fill in
  `source_tag`, `source_commit`, `archive` and `expected_tex_sha256` in
  `paper-versions.json`. The same pattern applies to the working paper's first
  submission to the next venue.
- The camera-ready branch fixes a bug in the tagged v35 source: its camera-ready
  header used the main-track style option (`[main,final]`); it now uses
  `[dblblindworkshop,final]` with the workshop title "Interpretability as a Science".
- Open camera-ready items: authors and affiliations (metadata still anonymous),
  acknowledgements, how to treat claims retracted since v35 (mediator-relative
  routing; S1 description), checklist answers, the workshop's camera-ready instructions.

## Reproducibility, three levels

1. **Exact rebuild**: `bin/build-paper-version <version>` (TeX SHA-256 checked for frozen versions).
2. **Verification without weights**: `bin/verify-paper` (numpy, matplotlib). It
   regenerates the eight figures into a temporary folder (all pixel-identical to the
   committed PNGs today), requires the exact input set in
   `manuscript/figures/figure_inputs.json` (113 read, 24 looked for but absent), and
   checks 26 headline numbers with `manuscript/claims.py` (all pass, including every cell of the appendix hidden-choice table).
3. **Full replication**: `env/` (pinned packages; Hugging Face revisions of the
   locally run models), vendored Hanna & Ameisen data in
   `third_party/hanna-ameisen-model-planning/` (4.3 MB, MIT, commit 993cff5, checksums),
   per-experiment READMEs.

## Open items for the authors

- **Corrected (render paper_66):** two hidden-choice shares had been divided after
  rounding to two decimals. Qwen3-32B animals is 7.5% (now 8%, was 7%) and Gemma 4 31B
  fruits is 81.4% (now 81%, was 82%), in the introduction table, Section 5 and the
  appendix. The table's "0–40%" range reproduces: it is pick specificity from
  `choice_outlist_summary` (fruits) and `choice_replicate_animals_summary`. The
  NeurIPS v35 text contains neither number.
- The figure script asks for 24 files that do not exist (for example
  `relay_boundary_summary.json` for several Qwen3-plain and small Gemma runs, and
  Qwen3.5-35B-A3B everywhere) and silently draws without them. The manifest freezes
  that state; whether those rows should exist is a scientific question.

## Later the same day (renders paper_67, paper_68)

- Section 3 and the pre-registration table said relay was measured by necessity in 17
  couplet runs; it is 18 (still at most 1%, max 0.96%). The free-generation check is
  now reported for all five models it was run in. `manuscript/claims.py`: 28 claims.
- `bin/freeze-paper-version` freezes a submission in one step (tested end to end in
  a scratch clone); `AGENTS.md`, "Submitting the working paper", documents the venue
  lifecycle; `bin/render-paper` allows a venue profile on `main` once registered.
- The 24 absent figure inputs: 6 are fallbacks to an older file that exists; 4 are
  Qwen3.5-35B-A3B, dropped before the ICLR deadline and never mentioned in the text;
  the rest are analyses run only on a subset of models (per-token boundary split in 9
  runs, couplet necessity in Gemma 3 4B, free generation in 3 models), which the text
  scopes correctly. The Mac Studio was unreachable over SSH (VPN up), so whether any of
  these exist only there is unchecked.

## Remaining engineering (not started)

- Summaries are computed inside model-loading runners; recomputing them from the
  committed rows needs pure summary functions and a torch-free `boot()`
  (`experiments/couplet_routes/step34_routes.py`).
- About 60 experiment configs hold absolute `model_snapshot` paths on the
  development Mac; replace with model id + revision from `env/models.json`.
- Record the Mac Studio's model and transcoder revisions and both remote
  environments (`env/README.md`, "Not yet recorded") when the VPN is up.
- Render numbering: the working paper's numbered PDFs continue from
  `dist/iclr-submission/` (paper_65) into `dist/working-draft/` (paper_66 onward);
  stray paper_1 and paper_2 there came from before the fix.
- Public release (licence, history scan) is deferred until after camera-ready or
  the conference. `dist/` (210 MB locally, untracked) is disposable; its numbered
  PDFs are backed up on Google Drive.
