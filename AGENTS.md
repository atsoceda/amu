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
| Manuscript (supporting) | [`skills/iclr-manuscript/`](skills/iclr-manuscript/) | Editing, extending or rendering the ICLR 2027 paper, or adding a figure or table to it. |

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

There are two versions of the paper. Keep them separate.

| Version | Status | Where it lives |
| --- | --- | --- |
| **NeurIPS 2026 workshop submission (v35)** | **Submitted and frozen.** Never edit, re-render, or replace it. | [`submissions/neurips-2026-workshop-v35/`](submissions/neurips-2026-workshop-v35/) (exact submitted PDF, generated `.tex`, manifest, `SHA256SUMS`, all read-only). Source: Git tag `neurips-2026-workshop-v35` (commit `e7e1372`). |
| **ICLR 2027 main-track extension** | **Active target.** All new manuscript work goes here. | The live source: [`paper.qmd`](paper.qmd), [`manuscript/sections/`](manuscript/sections/), [`manuscript/references.bib`](manuscript/references.bib). Render target `iclr`. |

Rules:

- Treat any request to update, extend, or render "the paper" as ICLR 2027 work
  unless the user explicitly names the NeurIPS workshop version.
- Do **not** run `bin/render-paper neurips ...`. The live source is now the ICLR
  extension, so a NeurIPS render would write ICLR content into
  `dist/neurips-submission/` and blur the two versions. To inspect the submitted
  version, open the frozen folder. To rebuild it, check out the tag in a separate
  worktree and render there; that reproduces the frozen `paper.tex` byte for byte.
  Link `.tools` and `.home` only from inside that worktree, never over the real
  directories (`skills/iclr-manuscript`).
- Do not edit `dist/neurips-submission/`. Its versioned PDFs (`...-v14.pdf`
  through `...-v35.pdf`) are historical drafts; v35 is the submitted one.
- Do not shorten, condense, or delete manuscript text just to meet a page limit.
  Write corrections at the length the content needs, and report any page-limit
  overflow to the user.

## ICLR paper drafting and publishing

When the user gives a high-level request such as "generate an updated draft of
our results as a complete paper," treat it as a manuscript editing plus render
task for the ICLR 2027 main track, not as a DOCX or Google Docs task.

Use the Quarto publishing workflow:

1. Edit the manuscript source only:
   - Main entry point: [`paper.qmd`](paper.qmd)
   - Body sections: [`manuscript/sections/`](manuscript/sections/)
   - Bibliography: [`manuscript/references.bib`](manuscript/references.bib)
   - Figures/assets: use repository-relative paths. Figures are tracked in Git,
     so the tag preserves the NeurIPS versions; updating a figure for ICLR is fine.
2. Do **not** edit generated files under `dist/` or root-level generated
   `paper.tex` files. Generated `.tex` is an output artifact, not the source of truth.
3. Render with:

   ```bash
   bin/render-paper iclr submission
   ```

   Outputs go to `dist/iclr-submission/` (`paper.pdf`, `paper.tex`,
   `build-manifest.json`); camera-ready builds use `iclr camera-ready`.
4. The `iclr` profiles use the official, unmodified ICLR 2027 style files
   vendored under [`_extensions/iclr/`](_extensions/iclr/) (see its README).
   The Helvetica and Courier fonts the style needs are vendored in the
   repo-local TeX tree `_extensions/texmf/`, which `bin/render-paper` sets as
   `TEXMFHOME`. ICLR 2027 allows **9 pages of main text** at submission (10 for
   rebuttal/camera-ready), with unlimited pages for references and appendix.
   Report the main-text page count to the user; do not trim text to fit.
5. The manuscript still contains NeurIPS-specific material (for example the
   NeurIPS paper checklist and checklist notes in the appendix). Adapt these to
   ICLR requirements as part of the extension, never by editing the frozen copy.
6. Before reporting completion, verify that the render command succeeds and
   mention any remaining manuscript TODOs.

The helper script configures repo-local Quarto, `latexmk`, and TeX search paths.
Do not install a new TeX environment or create a Python virtual environment
unless the render fails for a reason that cannot be solved with the vendored
workflow. Additional publishing notes live in [`docs/publishing.md`](docs/publishing.md).
The vendored NeurIPS 2026 assets remain under [`_extensions/neurips/`](_extensions/neurips/)
only so that the tagged workshop version can be rebuilt.

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
scripts; it exits non-zero on errors:

```bash
bin/validate-skills
```

If [skills-ref](https://github.com/agentskills/agentskills/tree/main/skills-ref) is installed, also run it against each skill folder:

```bash
skills-ref validate ./skills/circuit-attribution
skills-ref validate ./skills/circuit-graph-export
skills-ref validate ./skills/circuit-graph-viewer
skills-ref validate ./skills/hf-range-streaming
skills-ref validate ./skills/mac-studio-remote
skills-ref validate ./skills/iclr-manuscript
skills-ref validate ./skills/state-edit-experiments
```
