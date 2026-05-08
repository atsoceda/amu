# circuit-graph-viewer — CLI reference

## Command

```bash
/Users/anthony/miniconda3/envs/pyclean/bin/circuit-tracer start-server --graph_file_dir ./graph_files --port 8041
```

## URLs

- Default: [http://localhost:8041/](http://localhost:8041/)

## Combined shortcut (from attribution)

If `circuit-tracer attribute ... --slug SLUG --graph_file_dir ./graph_files --server` was used, phase 3 starts automatically after export; still communicate the listening port and stop semantics.

## Local permission note

If `start-server` prints `PermissionError: [Errno 1] Operation not permitted` while binding the socket, rerun the same command with the agent/tooling's escalation mechanism. Then verify with:

```bash
curl -I http://127.0.0.1:8041/
```
