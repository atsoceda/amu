# circuit-graph-export — implementation notes

## API

```python
from circuit_tracer.utils.create_graph_files import create_graph_files

create_graph_files(
    graph_or_path="graph.pt",   # or Graph instance
    slug="my-run-slug",
    output_path="./graph_files",
    node_threshold=0.8,        # defaults match CLI unless overridden
    edge_threshold=0.98,
)
```

## Outputs

| File | Role |
| --- | --- |
| `./graph_files/<slug>.json` | Pruned graph model for the UI |
| `./graph_files/graph-metadata.json` | Index of available graphs (slug keyed) |

## Path pitfall (macOS / relative paths)

Pass **`./graph_files`** for `output_path` when using a relative directory string so internal metadata helpers resolve a non-empty parent directory.

## Command

Use the project conda environment:

```bash
/Users/anthony/miniconda3/envs/pyclean/bin/python -c 'from circuit_tracer.utils.create_graph_files import create_graph_files; create_graph_files("graph.pt", slug="my-run-slug", output_path="./graph_files")'
```
