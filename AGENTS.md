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

## Google Drive / `.gdoc` workflow

`drive-sync-amu/*.gdoc` files are Google Drive pointer files. Do **not** create or edit them manually as document contents.

When creating a Google Doc for this repo:

1. Prefer creating/importing directly into the Drive folder `drive-sync-amu`.
2. If connector import tools create a file in Drive root, move that resulting Google Doc into the `drive-sync-amu` Drive folder before relying on the local sync folder.
3. Avoid manually adding `.gdoc` pointer files unless the target Drive doc already exists and you have confirmed this will not create a duplicate synced copy.

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
