# Agent guidance (AMU)

This repository uses **[Agent Skills](https://agentskills.io/specification)**-style skill folders under [`skills/`](skills/) so coding agents can reliably execute the three-phase [`circuit-tracer`](https://github.com/decoderesearch/circuit-tracer) workflow:

1. **Attribution** — compute the attribution graph (features, errors, tokens, logits).
2. **Graph file creation** — prune and export **viewer JSON** plus `graph-metadata.json`.
3. **Local server** — serve the visualization UI in a browser.

Validated commands and defaults live in each skill's `SKILL.md` and `references/REFERENCE.md` files. Narrative notes from the original setup are in [`update-1.md`](update-1.md).

## Skill registry

| Phase | Skill directory | Activate when |
| --- | --- | --- |
| Attribution | [`skills/circuit-attribution/`](skills/circuit-attribution/) | User asks for an attribution/circuit graph, `.pt` export, or “run attribution” only. |
| Graph JSON export | [`skills/circuit-graph-export/`](skills/circuit-graph-export/) | User has a `.pt` graph and needs browser-ready JSON under `./graph_files/`. |
| Local viewer | [`skills/circuit-graph-viewer/`](skills/circuit-graph-viewer/) | User wants to **open** / **see** the graph in a browser or start the UI server. |
| Large-file streaming (supporting) | [`skills/hf-range-streaming/`](skills/hf-range-streaming/) | A job needs parts of very large Hugging Face files (transcoders, feature cards) that do not fit on disk, or range downloads are slow or return HTTP 429. |
| State-edit experiments (supporting) | [`skills/state-edit-experiments/`](skills/state-edit-experiments/) | Writing or running route-accounting experiments that patch hidden states (couplets, derived value, positive control, hidden choice), including batching and known pitfalls. |
| Remote compute (supporting) | [`skills/mac-studio-remote/`](skills/mac-studio-remote/) | An experiment needs more memory than the local M1, or the user asks to use the Mac Studio. Never copy the codebase there; send only allowlisted scripts and data. |
| Manuscript (supporting) | [`skills/working-manuscript/`](skills/working-manuscript/) | Editing, extending or rendering the active venue-undecided paper, or adding a figure or table to it. |

Each skill’s **`SKILL.md`** follows the Agent Skills format (YAML frontmatter + Markdown body). Optional detail lives under `references/` inside each skill.

## Research record

Follow [`docs/research-record.md`](docs/research-record.md) for every experiment:
a self-describing README per experiment (question, design frozen before results,
dated deviations, dated results including nulls), committed rows and summaries
(re-derivable intermediates gitignored), remote results fetched and committed the
same day, one branch per workstream merged into `main` at each milestone, and a
new dated snapshot in `docs/handoff/` at each milestone. Start a new session by
reading the latest file in `docs/handoff/`.

## Communication preferences

When writing math in chat or docs:

- Use `\(...\)` for inline math because it renders properly in this chat UI.
- Use `$$...$$` on separate lines for block equations.
- Avoid `$...$` for inline math in chat; it may show raw delimiters instead of rendering.
- Do not put math inside fenced code blocks like ` ```latex `.

## Naming models and experiments

With many models and experiments in play, refer to each one as
**model name & generation - model size - experiment class - experiment name**, for example
"Gemma 3 - 27B - relay distance - pilot" or "Qwen3 - 14B - couplet routes - position split
(plain)". Experiment classes: couplet routes, derived value, hidden choice, relay positive
control, recurrent carry, relay distance.

## Paper versions and venue targets

The root [`README.md`](README.md) and [`paper-versions.json`](paper-versions.json)
index the paper versions. Tags name frozen source commits; archive folders hold
exact historical artifacts. Working branches are not version archives.

| Version | Status | Where it lives |
| --- | --- | --- |
| **NeurIPS 2026 workshop submission (v35)** | **Submitted and frozen.** Never edit or replace the archive. | [`submissions/neurips-2026-workshop-v35/`](submissions/neurips-2026-workshop-v35/) (exact submitted PDF, generated `.tex`, manifest, checksums). Source tag: `neurips-2026-workshop-v35`. |
| **NeurIPS 2026 workshop camera-ready** | **In preparation; the paper was accepted.** | Branch `paper/neurips-workshop`, created from the v35 tag and never merged into `main`. Edit and render only there (`bin/render-paper neurips camera-ready`; see its `CAMERA-READY.md`). Preview from `main` with `bin/build-paper-version neurips-2026-workshop-camera-ready`. When uploaded: tag `neurips-2026-workshop-camera-ready` and archive the exact PDF under `submissions/` on `main`. |
| **ICLR 2027 deadline draft** | **Not submitted; frozen milestone.** Never edit or replace the archive. | [`milestones/iclr-2027-deadline-unsubmitted/`](milestones/iclr-2027-deadline-unsubmitted/) (last local ICLR-formatted draft). Source tag: `iclr-2027-deadline-unsubmitted`. |
| **Working paper** | **Active; venue undecided** (another conference or a journal; ICLR is out). | Live [`paper.qmd`](paper.qmd) and [`manuscript/`](manuscript/) on `main`. Render with `bin/build-paper-version working`. |

Rules:

- Treat an unqualified request to update, extend, or render "the paper" as work
  on the venue-undecided manuscript. Do not infer a new target venue.
- Do **not** run `bin/render-paper neurips ...` or `bin/render-paper iclr ...`
  from the live source. To rebuild either frozen version without switching the
  current checkout, use `bin/build-paper-version <version>`; it renders the
  tagged source in a temporary worktree and verifies the archived TeX hash.
- Do not edit `dist/neurips-submission/`. Its versioned PDFs (`...-v14.pdf`
  through `...-v35.pdf`) are historical drafts; v35 is the submitted one.
- Do not shorten, condense, or delete manuscript text just to meet a page limit.
  Write corrections at the length the content needs, and report any page-limit
  overflow to the user.

## Working-paper drafting and publishing

When the user gives a high-level request such as "generate an updated draft of
our results as a complete paper," treat it as a manuscript editing plus render
task for the venue-undecided working paper, not as a DOCX or Google Docs task.

Use the Quarto publishing workflow:

1. Edit the manuscript source only:
   - Main entry point: [`paper.qmd`](paper.qmd)
   - Body sections: [`manuscript/sections/`](manuscript/sections/)
   - Bibliography: [`manuscript/references.bib`](manuscript/references.bib)
   - Figures/assets: use repository-relative paths. Figures are tracked in Git,
     so source tags preserve their historical versions.
2. Do **not** edit generated files under `dist/` or root-level generated
   `paper.tex` files. Generated `.tex` is an output artifact, not the source of truth.
3. Render with:

   ```bash
   bin/build-paper-version working
   ```

   Outputs go to `dist/working-draft/` (`paper.pdf`, `paper.tex`,
   `build-manifest.json`).
4. The working profile reuses ICLR 2027 geometry and bibliography but replaces
   its visible review-status claims. It is **not an official submission
   template**. The official files under [`_extensions/iclr/`](_extensions/iclr/)
   stay unmodified for rebuilding the ICLR deadline milestone. Fonts are
   vendored in `_extensions/texmf/`. Report the main-text page count, but do
   not apply a venue page limit until the next venue is selected.
5. When a venue is selected, add or validate its official style, checklist,
   anonymity and page-limit requirements. Do not retroactively alter frozen
   versions.
6. Before reporting completion, verify that the working render succeeds and
   mention any remaining manuscript TODOs.

The helper script configures repo-local Quarto, `latexmk`, and TeX search paths.
Do not install a new TeX environment or create a Python virtual environment
unless the render fails for a reason that cannot be solved with the vendored
workflow. Additional publishing notes live in [`docs/publishing.md`](docs/publishing.md).
The vendored NeurIPS 2026 assets remain under [`_extensions/neurips/`](_extensions/neurips/)
so that its tagged workshop version can be rebuilt.

## Submitting the working paper (journal or conference)

The working paper on `main` goes through submission, review and revision with the
same three tools: `bin/verify-paper`, `bin/freeze-paper-version` and
`bin/build-paper-version`. Use version names `<venue>-<year>-v1` for the first
submission and `<venue>-<year>-r1`, `-r2` for revised resubmissions.

1. **Venue chosen.** Vendor its official style files under `_extensions/<venue>/`
   (never edit them), add `_quarto-<venue>-submission.yml`, and register the version
   in `paper-versions.json` as in preparation: `{"status": "in preparation",
   "render_target": "<venue>", "render_mode": "submission"}` (plus `"derived_from"`
   for a revision). `bin/render-paper <venue> submission` then works from `main`,
   and its numbered PDFs continue the working paper's series. Check the venue's
   anonymity, length, checklist, data and code-availability rules and record them in
   the entry's `note`. Report length overflow to the user; never cut text to fit.
2. **Editing.** Edit `paper.qmd` and `manuscript/` on `main`, commit, render a new
   numbered PDF, and keep `bin/verify-paper` passing (update `manuscript/claims.py`
   when a headline number changes). New experiments follow `docs/research-record.md`.
3. **Submitting.** With everything committed, run `bin/freeze-paper-version
   <venue>-<year>-v1`. It verifies, renders clean `HEAD`, archives the exact PDF, TeX,
   manifest and checksums under `submissions/`, tags the source, records it in
   `paper-versions.json`, and proves the rebuild. Upload exactly the archived PDF.
   Add a row to the version tables in `README.md` and this file, then
   `git push origin main <version>`.
4. **Under review.** Keep editing on `main`. Keep reviews and responses under
   `docs/reviews/<version>/` (reviews as received; the response letter as Markdown).
5. **Revision.** Register `<venue>-<year>-r1` with `"derived_from":
   "<venue>-<year>-v1"`, edit, then freeze it the same way. For a marked-up diff,
   rebuild both versions with `bin/build-paper-version` and compare their `paper.tex`.
6. **Accepted.** If `main` has moved on to other work, create
   `paper/<venue>` from the accepted tag for the camera-ready, as with
   `paper/neurips-workshop`; otherwise finish it on `main` and freeze it as
   `<venue>-<year>-camera-ready`.

Never move or delete a version tag, and never edit an archive folder: a correction
after submission is a new version.

## Google Drive / `.gdoc` workflow

`drive-sync-amu/*.gdoc` files are Google Drive pointer files. Do **not** create or edit them manually as document contents.

When creating a Google Doc for this repo:

1. Prefer creating/importing directly into the Drive folder `drive-sync-amu`.
2. If connector import tools create a file in Drive root, move that resulting Google Doc into the `drive-sync-amu` Drive folder before relying on the local sync folder.
3. Avoid manually adding `.gdoc` pointer files unless the target Drive doc already exists and you have confirmed this will not create a duplicate synced copy.

## DOCX + LaTeX equation workflow

For longer technical documents with equations, prefer a local `.docx` workflow over direct Google Docs editing when native Google Docs plugins are not required.

When the document contains LaTeX math that should render as equations:

1. Use Pandoc as the conversion path. Keep the editable source as Markdown with `\(...\)` inline math and `$$...$$` block math.
2. If starting from an existing `.docx`, first extract it with media:

   ```bash
   pandoc input.docx -t markdown+tex_math_single_backslash --extract-media=<work-dir>/media -o <work-dir>/source.md
   ```

3. Normalize any equation-image links or escaped math delimiters into real Markdown math before exporting.
4. Export back to `.docx` with Pandoc, preserving media and optionally using the original document as a style reference:

   ```bash
   pandoc source.md --from markdown+tex_math_single_backslash+tex_math_dollars --to docx --resource-path=. --reference-doc=input.docx -o output.docx
   ```

5. Verify the result is a real equation document, not just visually similar:
   - `word/document.xml` should contain Word math tags such as `<m:oMath>`.
   - The `.docx` should not contain raw `\(...\)` delimiters, `$$...$$` delimiters, or old equation-image service links such as CodeCogs.
   - Embedded screenshots and figures should still exist under `word/media/`.
6. If LibreOffice/`soffice` is available, render the `.docx` to page images and visually inspect it before delivery. If `soffice` is unavailable, state that full visual rendering was not completed.

## High-level user commands (orchestration)

When the user gives a **single high-level request** such as:

- “Show me the attribution graph for `The capital of France is`.”
- “Visualize the circuit for prompt …”

the agent should:

1. **Read** the three `skills/*/SKILL.md` files (or at minimum the attribution + viewer skills, plus graph-export if not using `--server` one-shot).
2. **Use the project environment** — use `/Users/anthony/miniconda3/bin/python` and `/Users/anthony/miniconda3/bin/circuit-tracer`. Do **not** create a venv for this repo.
3. **Choose a run directory** — e.g. `runs/<YYYY-MM-DD>_<short-slug>/` under the repo root (or a path the user specifies). Create it if missing.
4. **Pick identifiers**
   - **`slug`**: short filesystem-safe id (lowercase, hyphens), e.g. `france-capital` or derived from the prompt hash/date.
   - **`.pt` path**: e.g. `./graph.pt` or `./<slug>.pt` inside the run directory.
5. **Execute phase 1 (Attribution)** per [`skills/circuit-attribution/SKILL.md`](skills/circuit-attribution/SKILL.md): produce a `.pt` **unless** the user explicitly wants the one-shot CLI path (`--slug` + `--graph_file_dir` + `--server`), which combines attribution + JSON + server in one process.
   - First try the normal `circuit-tracer attribute` command from the project conda env.
   - If it fails on Hugging Face DNS, offline mode, or `repo_info` metadata even though the artifacts are cached, immediately use the local-cache fallback in [`skills/circuit-attribution/references/REFERENCE.md`](skills/circuit-attribution/references/REFERENCE.md). The local cache currently contains the Gemma model snapshot and Gemma Scope 2 transcoder weights needed for the default stack.
6. **Execute phase 2 (Graph export)** when needed per [`skills/circuit-graph-export/SKILL.md`](skills/circuit-graph-export/SKILL.md): call `create_graph_files(...)` so `./graph_files/<slug>.json` and `./graph_files/graph-metadata.json` exist. Always pass **`output_path` as `./graph_files`** (leading `./`) to avoid path assertion failures documented in `update-1.md`.
7. **Execute phase 3 (Viewer)** per [`skills/circuit-graph-viewer/SKILL.md`](skills/circuit-graph-viewer/SKILL.md): `circuit-tracer start-server --graph_file_dir ./graph_files --port <port>`. Prefer running the server in the **background**; tell the user to open `http://localhost:<port>/` (default **8041** if unchanged).
8. **Summarize** what ran: prompt, model, transcoder, backend, paths to `.pt` and JSON, server URL, and how to stop the server (Ctrl+C in the server terminal).

### Defaults (validated on this project)

Unless the user overrides:

- **Model:** `google/gemma-3-270m` (pretrained; preferred for a/an hypothesis tests — Latent Planning Appendix J found base slightly better than IT on a/an)
- **Transcoder:** `mwhanna/gemma-scope-2-270m-pt/clt/width_262k_l0_medium_affine`
- **Backend:** `nnsight`
- **Environment:** conda base at `/Users/anthony/miniconda3` (`/Users/anthony/miniconda3/bin/python`, `/Users/anthony/miniconda3/bin/circuit-tracer`)
- **Package:** PyPI `circuit-tracer` already installed in that environment

Always pair PT with PT (or IT with IT). Do not mix `gemma-3-270m` with `gemma-scope-2-270m-it`, or vice versa.

### Local cache details

The default run can be completed without re-downloading model weights once cached:

- Model snapshot: `/Users/anthony/.cache/huggingface/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`
- Transcoder config (repo-local, circuit-tracer format): [`experiments/lib/transcoder_configs/gemma-scope-2-270m-pt-width_262k_l0_medium_affine.yaml`](experiments/lib/transcoder_configs/gemma-scope-2-270m-pt-width_262k_l0_medium_affine.yaml)
- Transcoder weight snapshot: `/Users/anthony/.cache/huggingface/hub/models--google--gemma-scope-2-270m-pt/snapshots/b218cd5d69dc2fa71cff448b68d625e6c9702d49/clt/width_262k_l0_medium_affine`
- Note: Paper experiments use **PT + affine**. IT was evaluated and rejected as the primary stack for this hypothesis (weaker a/an task fit; Hub IT `medium_affine` layer-0 is also empty).

`circuit-tracer` may still perform live Hugging Face metadata checks even when these files are cached. If network is unavailable or sandboxed, use the documented local-cache fallback instead of installing packages or creating a new environment.

### Prerequisites reminder

- Hugging Face **account**, **model license acceptance**, and **authentication** (`HF_TOKEN` or `huggingface-cli login`) for downloads.
- Enough **RAM** for local attribution; first runs download weights and can spike memory (see `update-1.md`).

### Optional validation

Check every skill against the [Agent Skills specification](https://agentskills.io/specification)
after adding or editing one. `bin/validate-skills` is a local checker for the
specification's frontmatter rules (name format and directory match, field
lengths, string-valued metadata) plus broken file references and non-executable
scripts; it exits non-zero on errors. It needs PyYAML, so run it with the project
Python:

```bash
/Users/anthony/miniconda3/bin/python bin/validate-skills
```

If [skills-ref](https://github.com/agentskills/agentskills/tree/main/skills-ref) is installed, also run it against each skill folder:

```bash
skills-ref validate ./skills/circuit-attribution
skills-ref validate ./skills/circuit-graph-export
skills-ref validate ./skills/circuit-graph-viewer
skills-ref validate ./skills/hf-range-streaming
skills-ref validate ./skills/mac-studio-remote
skills-ref validate ./skills/working-manuscript
skills-ref validate ./skills/state-edit-experiments
```
