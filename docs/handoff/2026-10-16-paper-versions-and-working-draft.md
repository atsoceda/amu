# Handoff: paper versions and venue-undecided working draft (2026-10-16)

Superseded by `2026-10-16-repo-structure-and-reproducibility.md` (same day).

Supersedes the publishing-status portions of
`2026-09-26-iclr-submission-state.md` and
`2026-09-26-overnight-scale-and-boundary.md`. The latter remains the historical
scientific handoff; no experiment outcomes were changed here. Start with the
root README and `docs/research-record.md`.

## Current paper state

- No venue is selected. The ICLR 2027 deadline passed without a submission.
- The live source remains `paper.qmd` and `manuscript/` on `main`; render it with
  `bin/build-paper-version working` to `dist/working-draft/`.
- The working profile keeps ICLR-like layout but visibly says "Working paper —
  venue to be determined" and "Unsubmitted research draft", not "under ICLR
  review". The official ICLR style is unchanged.
- The exact submitted NeurIPS workshop v35 paper remains in
  `submissions/neurips-2026-workshop-v35/` with source tag
  `neurips-2026-workshop-v35`.
- The last local ICLR deadline draft, **never submitted**, is preserved in
  `milestones/iclr-2027-deadline-unsubmitted/` with source tag of the same name.
  Its original manifest truthfully records a pre-commit dirty render; the tagged
  source has been checked by regenerating the archived TeX SHA-256.

## Version build and navigation

- `paper-versions.json` maps human-readable names to source tags, render
  profiles, archives and expected TeX hashes. The root README is the human
  version index. Tags are visible with `git tag --list` but are not branches.
- `bin/build-paper-version <version>` builds a frozen tag in a temporary worktree,
  verifies archived SHA-256s and rebuilt TeX, then copies rebuilt artifacts to
  ignored `dist/version-builds/<version>/`. The current checkout is not switched.
- `bin/check-paper-env` checks tools; a fresh clone still needs Quarto and TeX
  installed. The repository does not yet ship a fully pinned TeX container.
- Existing completed research branches have no commits absent from `main`.
  They can be removed as routine branch hygiene after the version-workflow
  milestone merges. A separate detached worktree at
  `/Users/anthony/repos/amu-wt-page-budget` was observed and left untouched.

## Public-release gate

`docs/open-source-readiness.md` lists unresolved licensing, fresh-clone,
privacy/history, and experiment-recomputation checks. The new version index and
build workflow make the papers legible now; they do not by themselves certify
the entire research environment as externally reproducible.
