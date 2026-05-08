# circuit-attribution — CLI reference

Upstream: [decoderesearch/circuit-tracer](https://github.com/decoderesearch/circuit-tracer) · PyPI: [circuit-tracer](https://pypi.org/project/circuit-tracer/).

## Validated default stack (this repo)

| Input | Value |
| --- | --- |
| Model | `google/gemma-3-270m` |
| Transcoder | `mwhanna/gemma-scope-2-270m-pt/clt/width_262k_l0_medium_affine` |
| Backend | `nnsight` |

## Pattern A — `.pt` only (then use `circuit-graph-export`)

```bash
circuit-tracer attribute \
  --prompt "The capital of France is" \
  --model "google/gemma-3-270m" \
  --transcoder_set "mwhanna/gemma-scope-2-270m-pt/clt/width_262k_l0_medium_affine" \
  --backend nnsight \
  --graph_output_path ./graph.pt \
  --verbose
```

Replace `--prompt` with the user’s text (quote safely for the shell).

## Pattern B — JSON in one step (still attribution phase; may skip export skill)

```bash
circuit-tracer attribute \
  --prompt "The capital of France is" \
  --model "google/gemma-3-270m" \
  --transcoder_set "mwhanna/gemma-scope-2-270m-pt/clt/width_262k_l0_medium_affine" \
  --backend nnsight \
  --slug my-run-slug \
  --graph_file_dir ./graph_files \
  --verbose
```

Add `--server` to chain into the viewer (see `circuit-graph-viewer` skill).

## Useful optional flags (when needed)

- `--node_threshold`, `--edge_threshold` — pruning thresholds for JSON creation when `--slug`/`--graph_file_dir` are set.
- `--dtype`, `--batch_size`, `--offload` — performance / memory tuning per upstream docs.
