---
name: circuit-graph-viewer
description: >-
  Starts circuit-tracer local-server to visualize and interact with attribution graphs in a browser from graph JSON files.
  Use when the user wants to view, open, browse, or inspect a graph UI locally after graph_files exist.
license: MIT
compatibility: >-
  circuit-tracer CLI available; graph_files directory containing slug.json and graph-metadata.json; localhost browser access;
  choose a free TCP port (default 8041).
metadata:
  upstream: "https://github.com/decoderesearch/circuit-tracer"
  version: "1.0"
---

# Local visualization server (phase 3)

This skill maps to **Local Server** in the [circuit-tracer README](https://github.com/decoderesearch/circuit-tracer): serve the packaged frontend against **`graph_files/`**.

## Preconditions

- `./graph_files/` contains **`graph-metadata.json`** and at least one **`<slug>.json`** produced by phase 2 (`circuit-graph-export`) or by `circuit-tracer attribute` when both `--slug` and `--graph_file_dir` were provided.

## When you are done

- A **`circuit-tracer start-server`** process is running (typically **background** for agents).
- User knows the **URL** `http://localhost:<port>/` (default port **8041** unless changed).
- User knows how to **stop** the server (interrupt the terminal job / Ctrl+C).

## Procedure

1. `cd` to the run directory that contains **`./graph_files`** (the directory holding the JSON exports).
2. Start the server:

```bash
/Users/anthony/miniconda3/envs/pyclean/bin/circuit-tracer start-server --graph_file_dir ./graph_files --port 8041
```

3. If port is busy, retry with another `--port` and tell the user the updated URL.

## Agent conventions

- Prefer **long-running servers in the background**; do not block the session indefinitely unless the user asks for foreground logs.
- Confirm the server responds (e.g., HTTP 200 to `/`) when troubleshooting blank pages.
- If socket binding fails with `PermissionError: [Errno 1] Operation not permitted`, rerun the same server command with the agent/tooling's escalation mechanism. This is a local server permission issue, not a graph export issue.

## Edge cases

- **Firewall / remote machines**: User may need SSH port forwarding if running on a remote host (mentioned upstream).

## Further detail

See [`references/REFERENCE.md`](references/REFERENCE.md).
