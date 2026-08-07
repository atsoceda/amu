# circuit-attribution — CLI reference

Upstream: [decoderesearch/circuit-tracer](https://github.com/decoderesearch/circuit-tracer) · PyPI: [circuit-tracer](https://pypi.org/project/circuit-tracer/).

## Validated default stack (this repo)

| Input | Value |
| --- | --- |
| Model | `google/gemma-3-270m` |
| Transcoder | `mwhanna/gemma-scope-2-270m-pt/clt/width_262k_l0_medium_affine` |
| Backend | `nnsight` |
| Python/CLI | `/Users/anthony/miniconda3/bin/python`, `/Users/anthony/miniconda3/bin/circuit-tracer` |

Do not create a venv for this repo. Use the project conda env (`/Users/anthony/miniconda3`).

Pairing rule: always match PT/IT across the LM and Gemma Scope 2 transcoder. Do not mix
`gemma-3-270m` with `…-it` transcoders or `gemma-3-270m-it` with `…-pt` transcoders.

Paper experiments default to **PT + affine** (stronger a/an instrument; prior clincher signal).

## Pattern A — `.pt` only (then use `circuit-graph-export`)

```bash
/Users/anthony/miniconda3/bin/circuit-tracer attribute \
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
/Users/anthony/miniconda3/bin/circuit-tracer attribute \
  --prompt "The capital of France is" \
  --model "google/gemma-3-270m" \
  --transcoder_set "mwhanna/gemma-scope-2-270m-pt/clt/width_262k_l0_medium_affine" \
  --backend nnsight \
  --slug my-run-slug \
  --graph_file_dir ./graph_files \
  --verbose
```

Add `--server` to chain into the viewer (see `circuit-graph-viewer` skill).

## Pattern C — local-cache fallback when HF metadata is blocked

Prefer the experiment loader (handles the Hub offline URI map and the repo-local CLT yaml):

```bash
cd /path/to/run_dir
/Users/anthony/miniconda3/bin/python - <<'PY'
import logging, sys
from pathlib import Path
sys.path.insert(0, "/Users/anthony/repos/amu")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
from experiments.lib.core import load_replacement_model
from circuit_tracer import attribute
from circuit_tracer.utils.create_graph_files import create_graph_files

config = {
    "model": "google/gemma-3-270m",
    "model_snapshot": "/Users/anthony/.cache/huggingface/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
    "transcoder_set": "mwhanna/gemma-scope-2-270m-pt/clt/width_262k_l0_medium_affine",
    "transcoder_config_snapshot": "/Users/anthony/repos/amu/experiments/lib/transcoder_configs/gemma-scope-2-270m-pt-width_262k_l0_medium_affine.yaml",
    "transcoder_weight_snapshot": "/Users/anthony/.cache/huggingface/hub/models--google--gemma-scope-2-270m-pt/snapshots/b218cd5d69dc2fa71cff448b68d625e6c9702d49",
    "backend": "nnsight",
    "dtype": "bfloat16",
    "stream_transcoder_load": True,
}
prompt = "Someone who handles financial records is"
model = load_replacement_model(config)
graph = attribute(prompt=prompt, model=model, batch_size=32, verbose=True, offload=None, max_feature_nodes=1200)
Path("./graph.pt").parent.mkdir(parents=True, exist_ok=True)
graph.to_pt("./graph.pt")
create_graph_files(graph_or_path="./graph.pt", slug="my-run-slug", output_path="./graph_files")
print(Path("./graph.pt").resolve())
PY
```

Why this fallback exists:

- `circuit-tracer` may still call Hugging Face `repo_info(...)` before resolving cached `hf://` weight URIs; `experiments.lib.core.patch_hf_cache` maps those URIs to local files.
- Passing the model as a local snapshot path avoids a separate Transformers/nnsight metadata check for the model id.
- Repo-local CLT yaml: `experiments/lib/transcoder_configs/gemma-scope-2-270m-pt-width_262k_l0_medium_affine.yaml`.

## Useful optional flags (when needed)

- `--node_threshold`, `--edge_threshold` — pruning thresholds for JSON creation when `--slug`/`--graph_file_dir` are set.
- `--dtype`, `--batch_size`, `--offload` — performance / memory tuning per upstream docs.
