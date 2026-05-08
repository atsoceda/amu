# Agent guidance (AMU)

This repository uses **[Agent Skills](https://agentskills.io/specification)**-style skill folders under [`skills/`](skills/) so coding agents can reliably execute the three-phase [`circuit-tracer`](https://github.com/decoderesearch/circuit-tracer) workflow:

1. **Attribution** — compute the attribution graph (features, errors, tokens, logits).
2. **Graph file creation** — prune and export **viewer JSON** plus `graph-metadata.json`.
3. **Local server** — serve the visualization UI in a browser.

Validated commands and defaults are also summarized in [`CIRCUIT_TRACER_LOCAL_WORKFLOW.md`](CIRCUIT_TRACER_LOCAL_WORKFLOW.md) and narrative notes in [`update-1.md`](update-1.md).

## Skill registry

| Phase | Skill directory | Activate when |
| --- | --- | --- |
| Attribution | [`skills/circuit-attribution/`](skills/circuit-attribution/) | User asks for an attribution/circuit graph, `.pt` export, or “run attribution” only. |
| Graph JSON export | [`skills/circuit-graph-export/`](skills/circuit-graph-export/) | User has a `.pt` graph and needs browser-ready JSON under `./graph_files/`. |
| Local viewer | [`skills/circuit-graph-viewer/`](skills/circuit-graph-viewer/) | User wants to **open** / **see** the graph in a browser or start the UI server. |

Each skill’s **`SKILL.md`** follows the Agent Skills format (YAML frontmatter + Markdown body). Optional detail lives under `references/` inside each skill.

## High-level user commands (orchestration)

When the user gives a **single high-level request** such as:

- “Show me the attribution graph for `The capital of France is`.”
- “Visualize the circuit for prompt …”

the agent should:

1. **Read** the three `skills/*/SKILL.md` files (or at minimum the attribution + viewer skills, plus graph-export if not using `--server` one-shot).
2. **Choose a run directory** — e.g. `runs/<YYYY-MM-DD>_<short-slug>/` under the repo root (or a path the user specifies). Create it if missing.
3. **Pick identifiers**
   - **`slug`**: short filesystem-safe id (lowercase, hyphens), e.g. `france-capital` or derived from the prompt hash/date.
   - **`.pt` path**: e.g. `./graph.pt` or `./<slug>.pt` inside the run directory.
4. **Execute phase 1 (Attribution)** per [`skills/circuit-attribution/SKILL.md`](skills/circuit-attribution/SKILL.md): produce a `.pt` **unless** the user explicitly wants the one-shot CLI path (`--slug` + `--graph_file_dir` + `--server`), which combines attribution + JSON + server in one process (see workflow doc).
5. **Execute phase 2 (Graph export)** when needed per [`skills/circuit-graph-export/SKILL.md`](skills/circuit-graph-export/SKILL.md): call `create_graph_files(...)` so `./graph_files/<slug>.json` and `./graph_files/graph-metadata.json` exist. Always pass **`output_path` as `./graph_files`** (leading `./`) to avoid path assertion failures documented in `update-1.md`.
6. **Execute phase 3 (Viewer)** per [`skills/circuit-graph-viewer/SKILL.md`](skills/circuit-graph-viewer/SKILL.md): `circuit-tracer start-server --graph_file_dir ./graph_files --port <port>`. Prefer running the server in the **background**; tell the user to open `http://localhost:<port>/` (default **8041** if unchanged).
7. **Summarize** what ran: prompt, model, transcoder, backend, paths to `.pt` and JSON, server URL, and how to stop the server (Ctrl+C in the server terminal).

### Defaults (validated on this project)

Unless the user overrides:

- **Model:** `google/gemma-3-270m`
- **Transcoder:** `mwhanna/gemma-scope-2-270m-pt/clt/width_262k_l0_medium_affine`
- **Backend:** `nnsight`
- **Package:** PyPI `circuit-tracer` (pin version in env requirements when possible)

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
