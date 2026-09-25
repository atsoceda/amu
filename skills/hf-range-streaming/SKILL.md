---
name: hf-range-streaming
description: >-
  Streams parts of large Hugging Face files (per-layer transcoder safetensors, feature-card binaries) by HTTP
  range request instead of downloading whole repositories, and avoids the Hugging Face rate limit and redirect
  latency that make naive range requests slow or fail. Use when transcoders or feature data are too large for
  local disk or RAM (for example Qwen3 transcoders, 52-104 GB per model), when only some tensors, rows or
  feature cards are needed, or when range downloads are slow or return HTTP 429.
license: MIT
compatibility: >-
  Python 3.10+ standard library (urllib, concurrent.futures) plus torch for tensor decoding; network access to
  huggingface.co and its CDN; public or authorized repositories. Implementation lives in
  experiments/qwen3_planning_six_cell/hf_stream.py; use the project conda environment.
metadata:
  implementation: "experiments/qwen3_planning_six_cell/hf_stream.py"
  version: "1.0"
---

# Hugging Face range streaming

Fetch only the bytes you need from very large Hugging Face files: one layer's
encoder at a time, selected decoder rows, or the feature cards of active
features. Nothing is stored whole, so 50-100 GB transcoder repositories fit a
machine with about 20 GB of free disk.

## When you are done

- The needed tensors, rows or feature records are loaded, with per-layer results
  checkpointed so an interrupted run resumes.
- No request went through the rate-limited `huggingface.co/.../resolve/...`
  endpoint more than once per file per hour.
- You reported throughput (seconds per layer) to the user.

## Procedure

1. Import the helpers from `experiments/qwen3_planning_six_cell/hf_stream.py`
   (add that directory to `sys.path`). Do not write a new downloader.
2. For safetensors files use `SafetensorsRemote(repo, path, cache_dir)`:
   `.tensor(name)` fetches one tensor; `.rows(name, idx)` fetches selected rows of
   a 2-D tensor, merging nearby rows. The parsed header is cached on disk.
3. For feature-card binaries (`features/index.json.gz` + `features/layer_<n>.bin`)
   download the small index once, then call
   `fetch_feature_records(repo, index, layer, feature_ids)`.
4. **Resolve once, then go direct to the CDN.** `fetch_range` resolves each
   `.../resolve/main/<file>` URL once to its signed CDN URL, caches it, and sends
   range requests straight to the CDN. It re-resolves shortly before the URL's
   `Expires` time or on HTTP 403/410. Keep this behaviour; it is the difference
   between minutes and seconds per layer.
5. Use high concurrency only against the CDN (32 workers for card fetches). Never
   point many parallel workers at `huggingface.co/.../resolve/...`.
6. Checkpoint per layer (one output file per layer) and make every stage skip
   work already on disk, so a failure costs at most one layer.
7. Before trusting a changed fetch path, re-fetch one already-cached layer and
   check it is byte-identical to the cache.

## Pitfalls (measured 2026-09-25 on this machine)

- **Rate limit.** Every `/resolve/` request is a 302 redirect that counts against
  Hugging Face's limit of 3000 requests per 300 s (`ratelimit-policy:
  "fixed window";"resolvers";q=3000;w=300`). Two jobs with 16 workers each hit
  HTTP 429 and aborted.
- **Redirect latency.** Going through `/resolve/` costs about 0.9 s per small
  request; direct to the CDN about 0.64 s, and 32 parallel CDN requests finish in
  about 1.2 s. Card fetching went from 4-5 min per layer to about 30 s.
- **Merging ranges does not help for scattered features.** Active features are
  spread across the whole 1.1 GB card file, so cutting the request count means
  downloading most of the file. At a 64 KB merge gap a 1.7B layer needs 847
  requests (8.8 MB); at a 4 MB gap, 24 requests but 972 MB.
- **Bulk bandwidth is the hard ceiling** for large contiguous tensors (about
  4.5-7 MB/s here); parallel connections did not raise it. A 4B encoder layer
  (about 840 MB) takes about 2 minutes.

Details, timings and the transcoder layouts are in
[`references/REFERENCE.md`](references/REFERENCE.md).
