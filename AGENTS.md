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

Each skill’s **`SKILL.md`** follows the Agent Skills format (YAML frontmatter + Markdown body). Optional detail lives under `references/` inside each skill.

## Communication preferences

When writing math in chat or docs:

- Use `\(...\)` for inline math because it renders properly in this chat UI.
- Use `$$...$$` on separate lines for block equations.
- Avoid `$...$` for inline math in chat; it may show raw delimiters instead of rendering.
- Do not put math inside fenced code blocks like ` ```latex `.

## NeurIPS paper drafting and publishing

When the user gives a high-level request such as “generate an updated draft of
our results as a complete paper in NeurIPS format,” treat this as a manuscript
editing plus render task, not as a DOCX or Google Docs task.

Use the Quarto publishing workflow:

1. Edit the manuscript source only:
   - Main entry point: [`paper.qmd`](paper.qmd)
   - Body sections: [`manuscript/sections/`](manuscript/sections/)
   - Bibliography: [`manuscript/references.bib`](manuscript/references.bib)
   - Figures/assets: use repository-relative paths.
2. Do **not** edit generated files under `dist/` or root-level generated
   `paper.tex` files. Generated `.tex` is an output artifact for inspection or
   venue upload, not the source of truth.
3. Use the vendored NeurIPS files under [`_extensions/neurips/`](_extensions/neurips/).
   The current target is NeurIPS 2026. Official style loading and local TeX
   compatibility notes live in [`_extensions/neurips/README.md`](_extensions/neurips/README.md).
4. Render with:

   ```bash
   bin/render-paper neurips submission
   ```

   For camera-ready output, use:

   ```bash
   bin/render-paper neurips camera-ready
   ```

5. Expected outputs:
   - `dist/neurips-submission/paper.pdf`
   - `dist/neurips-submission/paper.tex`
   - `dist/neurips-submission/build-manifest.json`
   - analogous files under `dist/neurips-camera-ready/` for camera-ready builds.
6. Before reporting completion, verify that the render command succeeds and
   mention any remaining manuscript TODOs, especially NeurIPS checklist items.

The helper script configures repo-local Quarto, `latexmk`, and TeX search paths.
Do not install a new TeX environment or create a Python virtual environment
unless the render fails for a reason that cannot be solved with the vendored
workflow. Additional publishing notes live in [`docs/publishing.md`](docs/publishing.md).

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
2. **Use the project environment** — use `/Users/anthony/miniconda3/envs/pyclean/bin/python` and `/Users/anthony/miniconda3/envs/pyclean/bin/circuit-tracer`. Do **not** create a venv for this repo.
3. **Choose a run directory** — e.g. `runs/<YYYY-MM-DD>_<short-slug>/` under the repo root (or a path the user specifies). Create it if missing.
4. **Pick identifiers**
   - **`slug`**: short filesystem-safe id (lowercase, hyphens), e.g. `france-capital` or derived from the prompt hash/date.
   - **`.pt` path**: e.g. `./graph.pt` or `./<slug>.pt` inside the run directory.
5. **Execute phase 1 (Attribution)** per [`skills/circuit-attribution/SKILL.md`](skills/circuit-attribution/SKILL.md): produce a `.pt` **unless** the user explicitly wants the one-shot CLI path (`--slug` + `--graph_file_dir` + `--server`), which combines attribution + JSON + server in one process.
   - First try the normal `circuit-tracer attribute` command from the `pyclean` env.
   - If it fails on Hugging Face DNS, offline mode, or `repo_info` metadata even though the artifacts are cached, immediately use the local-cache fallback in [`skills/circuit-attribution/references/REFERENCE.md`](skills/circuit-attribution/references/REFERENCE.md). The local cache currently contains the Gemma model snapshot and Gemma Scope 2 transcoder weights needed for the default stack.
6. **Execute phase 2 (Graph export)** when needed per [`skills/circuit-graph-export/SKILL.md`](skills/circuit-graph-export/SKILL.md): call `create_graph_files(...)` so `./graph_files/<slug>.json` and `./graph_files/graph-metadata.json` exist. Always pass **`output_path` as `./graph_files`** (leading `./`) to avoid path assertion failures documented in `update-1.md`.
7. **Execute phase 3 (Viewer)** per [`skills/circuit-graph-viewer/SKILL.md`](skills/circuit-graph-viewer/SKILL.md): `circuit-tracer start-server --graph_file_dir ./graph_files --port <port>`. Prefer running the server in the **background**; tell the user to open `http://localhost:<port>/` (default **8041** if unchanged).
8. **Summarize** what ran: prompt, model, transcoder, backend, paths to `.pt` and JSON, server URL, and how to stop the server (Ctrl+C in the server terminal).

### Defaults (validated on this project)

Unless the user overrides:

- **Model:** `google/gemma-3-270m`
- **Transcoder:** `mwhanna/gemma-scope-2-270m-pt/clt/width_262k_l0_medium_affine`
- **Backend:** `nnsight`
- **Environment:** conda env `pyclean` at `/Users/anthony/miniconda3/envs/pyclean`
- **Package:** PyPI `circuit-tracer` already installed in `pyclean`

### Local cache details

The default run can be completed without re-downloading model weights:

- Base model snapshot: `/Users/anthony/.cache/huggingface/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`
- Transcoder config snapshot: `/Users/anthony/.cache/huggingface/hub/models--mwhanna--gemma-scope-2-270m-pt/snapshots/fada11860ac1d337c1e41e9da308798405b94c8e/clt/width_262k_l0_medium_affine/config.yaml`
- Transcoder weight snapshot: `/Users/anthony/.cache/huggingface/hub/models--google--gemma-scope-2-270m-pt/snapshots/b218cd5d69dc2fa71cff448b68d625e6c9702d49/clt/width_262k_l0_medium_affine`

`circuit-tracer` may still perform live Hugging Face metadata checks even when these files are cached. If network is unavailable or sandboxed, use the documented local-cache fallback instead of installing packages or creating a new environment.

### Prerequisites reminder

- Hugging Face **account**, **model license acceptance**, and **authentication** (`HF_TOKEN` or `huggingface-cli login`) for downloads.
- Enough **RAM** for local attribution; first runs download weights and can spike memory (see `update-1.md`).

### Optional validation

Use [skills-ref validate](https://github.com/agentskills/agentskills/tree/main/skills-ref) against each skill folder if installed:

```bash
skills-ref validate ./skills/circuit-attribution
skills-ref validate ./skills/circuit-graph-export
skills-ref validate ./skills/circuit-graph-viewer
```
