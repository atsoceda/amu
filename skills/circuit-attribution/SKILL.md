---
name: circuit-attribution
description: >-
  Runs circuit-tracer attribution to compute an attribution graph (direct effects among transcoder features,
  error nodes, tokens, and logits) and saves a raw PyTorch graph (.pt). Use when the user asks for attribution,
  a circuit graph, mechanistic tracing, or exporting graph.pt before visualization.
license: MIT
compatibility: >-
  Python 3.10+, PyPI package circuit-tracer, Hugging Face auth for model/transcoder downloads, network on first run,
  sufficient RAM for chosen model; use this repo's pyclean conda environment.
metadata:
  upstream: "https://github.com/decoderesearch/circuit-tracer"
  version: "1.0"
---

# Circuit attribution (phase 1)

This skill maps to **Attribution** in the [circuit-tracer README](https://github.com/decoderesearch/circuit-tracer): run the attribution algorithm and persist the **raw graph** for later pruning/export.

## When you are done

- A **`.pt`** file exists at the path chosen for `--graph_output_path` **or** JSON has been written via combined `--slug` + `--graph_file_dir` (that path skips the separate graph-export skill).
- You recorded **prompt**, **model**, **transcoder**, **backend**, and **output paths** for the user.

## Procedure

1. Use the project conda environment: `/Users/anthony/miniconda3/envs/pyclean/bin/python` and `/Users/anthony/miniconda3/envs/pyclean/bin/circuit-tracer`. Do **not** create a venv.
2. Ensure Hugging Face access is configured for the **base model** and **transcoder** repos (token / CLI login; license acceptance on the hub) if downloads or live metadata checks are needed.
3. `cd` into the **run directory** (create `runs/<date>_<slug>/` if the user did not specify one).
4. Run **`circuit-tracer attribute`** with **either**:
   - **`.pt` output** (recommended default for splitting phases): `--graph_output_path ./<name>.pt` **and** omit `--slug`/`--graph_file_dir` unless both are provided for JSON creation in the same invocation.
   - **One-shot JSON + optional server**: provide **both** `--slug` and `--graph_file_dir ./graph_files` per CLI rules; add `--server` only if the user wants the viewer started immediately.
5. If the normal CLI fails on Hugging Face DNS, offline mode, or `repo_info` metadata while default artifacts are already cached, use the local-cache fallback in [`references/REFERENCE.md`](references/REFERENCE.md). This fallback patches only HF URI resolution for the cached Gemma Scope 2 files and passes the base model as a local snapshot path so `nnsight`/Transformers do not perform a live model-id metadata check.

Use **validated defaults** from [`references/REFERENCE.md`](references/REFERENCE.md) unless the user supplies alternatives.

## CLI guardrails

- You **must** specify at least one of: (`--slug` **and** `--graph_file_dir`) **or** `--graph_output_path`. Otherwise the CLI errors.
- `--server` requires **both** `--slug` and `--graph_file_dir`.
- Prefer **`--verbose`** when debugging slow runs.

## Edge cases

- **Cached but network-blocked default run**: Do not reinstall packages or create another environment. Use the local-cache fallback command in [`references/REFERENCE.md`](references/REFERENCE.md).
- **Memory pressure / freezes**: Prefer smaller models and `nnsight` for HF-only models; close other apps; rerun after caches warm (see `update-1.md`).
- **Llama / large Gemma shortcuts**: Often heavy downloads; confirm user intent before using preset transcoder shortcuts like `llama` or `gemma`.

## Further detail

See [`references/REFERENCE.md`](references/REFERENCE.md) for example commands and flags.
