# circuit-attribution — CLI reference

Upstream: [decoderesearch/circuit-tracer](https://github.com/decoderesearch/circuit-tracer) · PyPI: [circuit-tracer](https://pypi.org/project/circuit-tracer/).

## Validated default stack (this repo)

| Input | Value |
| --- | --- |
| Model | `google/gemma-3-270m` |
| Transcoder | `mwhanna/gemma-scope-2-270m-pt/clt/width_262k_l0_medium_affine` |
| Backend | `nnsight` |
| Python/CLI | `/Users/anthony/miniconda3/envs/pyclean/bin/python`, `/Users/anthony/miniconda3/envs/pyclean/bin/circuit-tracer` |

Do not create a venv for this repo. Use the `pyclean` conda env.

## Pattern A — `.pt` only (then use `circuit-graph-export`)

```bash
/Users/anthony/miniconda3/envs/pyclean/bin/circuit-tracer attribute \
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
/Users/anthony/miniconda3/envs/pyclean/bin/circuit-tracer attribute \
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

Use this when the normal CLI reaches the cached files but fails with Hugging Face DNS errors, `HF_HUB_OFFLINE` errors, or a live `repo_info` metadata check. This happened even though the base model and transcoder artifacts were already present in `~/.cache/huggingface/hub`.

Run from the run directory and replace `PROMPT_TEXT` and output filename as needed:

```bash
/Users/anthony/miniconda3/envs/pyclean/bin/python - <<'PY'
from pathlib import Path
import logging
import os
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import circuit_tracer.utils.hf_utils as h

MODEL_SNAPSHOT = "/Users/anthony/.cache/huggingface/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
TRANSCODER_REF = "mwhanna/gemma-scope-2-270m-pt/clt/width_262k_l0_medium_affine"
TRANSCODER_WEIGHTS = Path("/Users/anthony/.cache/huggingface/hub/models--google--gemma-scope-2-270m-pt/snapshots/b218cd5d69dc2fa71cff448b68d625e6c9702d49")

PROMPT_TEXT = "The capital of France is"
GRAPH_OUTPUT = Path("./graph.pt")

def local_download_hf_uris(uris, max_workers=8):
    mapping = {}
    for uri in uris:
        info = h.parse_hf_uri(uri)
        if info.repo_id != "google/gemma-scope-2-270m-pt":
            raise ValueError(f"No local mapping for {uri}")
        path = TRANSCODER_WEIGHTS / info.file_path
        if not path.exists():
            raise FileNotFoundError(path)
        mapping[uri] = str(path)
    return mapping

h.download_hf_uris = local_download_hf_uris

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

logging.info("Loading cached transcoder and local model snapshot")
transcoder, config = load_transcoder_from_hub(
    TRANSCODER_REF,
    dtype=torch.float32,
    lazy_encoder=False,
    lazy_decoder=False,
)
model = ReplacementModel.from_pretrained_and_transcoders(
    MODEL_SNAPSHOT,
    transcoder,
    dtype=torch.float32,
    backend="nnsight",
)

logging.info("Running attribution")
graph = attribute(
    prompt=PROMPT_TEXT,
    model=model,
    max_n_logits=10,
    desired_logit_prob=0.95,
    batch_size=256,
    verbose=True,
    offload=None,
    max_feature_nodes=None,
)

logging.info("Saving graph to %s", GRAPH_OUTPUT)
graph.to_pt(str(GRAPH_OUTPUT))
print(GRAPH_OUTPUT.resolve())
PY
```

Why this fallback exists:

- `mwhanna/gemma-scope-2-270m-pt/.../config.yaml` is cached locally and points its `transcoders:` entries at `hf://google/gemma-scope-2-270m-pt/.../params_layer_*.safetensors`.
- The actual layer weights are cached under the `google/gemma-scope-2-270m-pt` snapshot path above.
- `circuit-tracer` still calls Hugging Face `repo_info(...)` before resolving those cached weight files; the fallback maps those `hf://` URIs directly to cached files.
- Passing the base model as a local snapshot path avoids a separate Transformers/nnsight metadata check for the model id.

## Useful optional flags (when needed)

- `--node_threshold`, `--edge_threshold` — pruning thresholds for JSON creation when `--slug`/`--graph_file_dir` are set.
- `--dtype`, `--batch_size`, `--offload` — performance / memory tuning per upstream docs.
