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
  Python 3.10+ standard library; torch only for decoding tensors and rows. Network access to huggingface.co
  and its CDN; public or authorized repositories. In this repo use the project conda environment.
metadata:
  version: "1.2"
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

1. Use the bundled script [`scripts/hf_stream.py`](scripts/hf_stream.py), as a
   library (import it) or from the command line. Do not write a new downloader.
   In this repo, experiment code imports it through
   `experiments/qwen3_planning_six_cell/hf_stream.py`, a thin re-export; edit the
   skill copy only.
2. Size the job before fetching anything:

   ```bash
   python scripts/hf_stream.py sizes <repo> [--prefix features/]
   python scripts/hf_stream.py header <repo> <file.safetensors>
   ```

3. If requests are slow or fail with HTTP 429, diagnose before changing code:

   ```bash
   python scripts/hf_stream.py probe <repo> <file>
   ```

   It prints the redirect, the rate-limit headers, the CDN URL lifetime, and
   timings through `/resolve/` vs direct to the CDN.
4. For safetensors use `SafetensorsRemote(repo, path, cache_dir)`: `.tensor(name)`
   fetches one tensor and `.rows(name, idx)` fetches selected rows of a 2-D tensor
   (nearby rows merged, requests run concurrently, 32 workers by default).
5. For feature cards, download `features/index.json.gz` once, check the cost
   offline, then fetch:

   ```bash
   python scripts/hf_stream.py plan <index.json.gz> <layer> <ids>       # requests vs bytes
   python scripts/hf_stream.py cards <repo> <index.json.gz> <layer> <ids> --out cards.json
   ```

   `<ids>` is `3,17,100-120` or `@file` with one id per line. In code, call
   `fetch_feature_records(repo, index, layer, ids)`.
6. Keep the resolve-once behaviour of `fetch_range`: it resolves each
   `.../resolve/main/<file>` URL once to its signed CDN URL, caches it, sends range
   requests straight to the CDN, and re-resolves before `Expires` or on HTTP
   403/410. Client errors (401, 404, 416) fail at once; 429 backs off.
7. Use high concurrency only against the CDN (32 workers for cards and rows).
   Any loop of many small range requests must run concurrently. Never point
   many parallel workers at `huggingface.co/.../resolve/...`.
8. Checkpoint per layer and make every stage skip work already on disk.
9. Before trusting a changed fetch path, re-fetch a sample and compare:
   `cards ... --verify cards.json` exits 1 unless the cards are identical.

## Pitfalls (measured 2026-09-25 on this machine)

- **Rate limit.** Every `/resolve/` request is a 302 redirect that counts against
  Hugging Face's limit of 3000 requests per 300 s (`ratelimit-policy:
  "fixed window";"resolvers";q=3000;w=300`). Two jobs with 16 workers each hit
  HTTP 429 and aborted.
- **Redirect latency.** Going through `/resolve/` costs about 0.9 s per small
  request; direct to the CDN about 0.64 s, and 32 parallel CDN requests finish in
  about 1.2 s. Card fetching went from 4-5 min per layer to about 30 s.
- **Sequential small requests are slow even without the rate limit.** Row fetches
  used to run one request at a time: the 4B el/la decoder step needed about 2,700
  requests at about 0.7 s each (about 30 min, CPU near idle). Concurrent row
  fetching finished it in about 2 min. If CPU is low and a stage is slow, check
  for a serial request loop first.
- **Merging ranges does not help for scattered features.** Active features are
  spread across the whole 1.1 GB card file, so cutting the request count means
  downloading most of the file. At a 64 KB merge gap a 1.7B layer needs 847
  requests (8.8 MB); at a 4 MB gap, 24 requests but 972 MB.
- **Bulk bandwidth is the hard ceiling** for large contiguous tensors (about
  4.5-7 MB/s here); parallel connections did not raise it. A 4B encoder layer
  (about 840 MB) takes about 2 minutes.

Details, timings and the transcoder layouts are in
[`references/REFERENCE.md`](references/REFERENCE.md).
