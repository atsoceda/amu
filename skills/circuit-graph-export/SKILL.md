---
name: circuit-graph-export
description: >-
  Converts a raw circuit-tracer attribution graph (.pt) into pruned viewer JSON files (slug.json and graph-metadata.json)
  suitable for the local web UI. Use when the user already has graph.pt or attribution output on disk and needs browser-ready graph_files.
license: MIT
compatibility: >-
  Python 3.10+, circuit-tracer installed in this repo's pyclean conda environment, write access to run directory.
metadata:
  upstream: "https://github.com/decoderesearch/circuit-tracer"
  version: "1.0"
---

# Graph file creation (phase 2)

This skill maps to **Graph File Creation** in the [circuit-tracer README](https://github.com/decoderesearch/circuit-tracer): **prune** the graph and **export JSON** for visualization.

## Preconditions

- A **`Graph`** serialized via **`graph.to_pt(...)`** exists at a known path (typically `*.pt`), **or** you are re-exporting from an existing `.pt` the user points to.

## When you are done

- `./graph_files/<slug>.json` exists.
- `./graph_files/graph-metadata.json` exists and lists the `slug` (metadata merges/replaces by slug).
- You used **`./graph_files`** (with `./`) as `output_path`, not the bare directory name `graph_files` (see `references/REFERENCE.md` and `update-1.md` — avoids empty `dirname` assertion failures).

## Procedure

1. Use the project conda environment: `/Users/anthony/miniconda3/envs/pyclean/bin/python`. Do **not** create a venv.
2. `cd` to the directory containing the `.pt` **or** pass an absolute path into `create_graph_files`.
3. Run Python calling **`circuit_tracer.utils.create_graph_files.create_graph_files`** with:
   - `graph_or_path`: path to `.pt` **or** an in-memory `Graph`.
   - `slug`: stable id for this graph (matches viewer expectations).
   - `output_path`: **`./graph_files`** (create parent beforehand if needed — the helper also mkdirs in many cases).

Example one-liner:

```bash
/Users/anthony/miniconda3/envs/pyclean/bin/python -c 'from circuit_tracer.utils.create_graph_files import create_graph_files; create_graph_files("graph.pt", slug="my-run-slug", output_path="./graph_files")'
```

4. Confirm files landed under `./graph_files/` before starting the server skill.

## Edge cases

- **Path assertion errors**: Pass `output_path="./graph_files"` with the leading `./`, not bare `graph_files`.
- **`scan` / transcoder identity**: Usually inferred from the graph; if export errors mention missing scan metadata, revisit attribution configuration per upstream docs rather than guessing.

## Further detail

See [`references/REFERENCE.md`](references/REFERENCE.md).
