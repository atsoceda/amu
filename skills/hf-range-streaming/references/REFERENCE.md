# hf-range-streaming — reference

Implementation: [`experiments/qwen3_planning_six_cell/hf_stream.py`](../../../experiments/qwen3_planning_six_cell/hf_stream.py).
First used by the Qwen3 planning-feature pipeline
([`select_features.py`](../../../experiments/qwen3_planning_six_cell/select_features.py)).

## API

```python
import sys; sys.path.insert(0, "experiments/qwen3_planning_six_cell")
from hf_stream import SafetensorsRemote, fetch_feature_records, load_feature_index, fetch_range, url

st = SafetensorsRemote("mwhanna/qwen3-4b-transcoders", "layer_12.safetensors", cache_dir)
w_enc = st.tensor("W_enc")                    # whole tensor, e.g. [163840, 2560] bf16
rows  = st.rows("W_dec", [17, 40961, 90000])  # {row_index: tensor}

index = load_feature_index(path_to_local_index_json_gz)   # download features/index.json.gz once
cards = fetch_feature_records("mwhanna/qwen3-4b-transcoders", index, 12, feature_ids)  # {feature: dict | None}
```

- `fetch_range(u, start, end_inclusive)`: resolves `u` once to its signed CDN
  URL, caches it until 120 s before its `Expires`, retries up to 12 times.
  - HTTP 403/410: re-resolve the CDN URL.
  - HTTP 429: honour `Retry-After`, otherwise back off up to 300 s.
- `fetch_feature_records(..., max_gap_bytes=65536, workers=32)`: merges records
  closer than 64 KB into one request and fetches the groups concurrently.
- Card binary format (same as the authors' `load_feature_from_binary.py`): for
  each feature, a 4-byte little-endian length, then that many bytes of
  gzip-compressed JSON.

## How the fix was found

| Measurement | Result |
| --- | --- |
| Card fetch through `/resolve/`, 4 workers | 4-5 min per 1.7B layer, about 120 KB/s |
| Response headers | `HTTP/2 302` to `us.aws.cdn.hf.co/...&Expires=<epoch>`; `ratelimit: "resolvers";r=2994;t=300`; `ratelimit-policy: "fixed window";"resolvers";q=3000;w=300` |
| Single small request via `/resolve/` | about 0.9 s (connect 0.15 s, time to first byte 0.69 s) |
| Same request direct to CDN | about 0.64 s |
| 32 parallel direct CDN requests | 1.17 s in total |
| Signed CDN URL lifetime | 3600 s |
| After the fix, 32 workers direct to CDN | 329-feature layer in 8.3 s; 1.7B layers about 30-40 s; output byte-identical to the earlier cache |
| Bulk download, 1 vs 8 connections | 80 MB in 17 s vs 18 s (bandwidth-bound) |

## Transcoder layouts seen so far

| Repo | Files | Tensors per layer | Activation |
| --- | --- | --- | --- |
| `mwhanna/qwen3-0.6b-transcoders-lowl0` | 28 x 671 MB `layer_<n>.safetensors`; `features/` 33.6 GB | `W_enc [163840, 1024]`, `W_dec [163840, 1024]`, `b_enc`, `b_dec` (bf16) | ReLU, no skip |
| `mwhanna/qwen3-1.7b-transcoders-lowl0` | 28 layers; 71 GB total | d_model 2048 | ReLU |
| `mwhanna/qwen3-4b-transcoders` | 36 layers; 104 GB total; `features/` 44 GB | d_model 2560 | ReLU |
| `google/gemma-scope-2-1b-pt` (CLT) | 26 layers | `w_enc [1152, 10080]`, `w_dec [10080, 26, 1152]`, `b_enc`, `threshold`, `affine_skip_connection`, `b_dec` | JumpReLU |

Qwen3 transcoder features read `post_attention_layernorm` output (circuit-tracer
`mlp.hook_in`) and write to the MLP output (`mlp.hook_out`). Encoder and decoder
rows are contiguous per feature in the Qwen3 files, so single features can be
fetched cheaply by row.

## Checklist for a new large-file job

1. Read the safetensors header first (8-byte length plus JSON) to learn dtype,
   shape and offsets; cache it.
2. Stream per layer: fetch, compute, write a small result, free memory.
3. Fetch sparse records concurrently, direct to the CDN.
4. Keep one fetching job per repository at a time unless requests go direct to
   the CDN.
5. Verify a sample against a known-good copy before a long run.
