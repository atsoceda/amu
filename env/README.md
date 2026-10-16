# Environment

What a researcher needs to rerun the experiments from a fresh clone.

| File | Contents |
| --- | --- |
| [`requirements-research.txt`](requirements-research.txt) | Python packages, pinned to the versions on the development Mac (Python 3.13.13, macOS, Apple Silicon). |
| [`models.json`](models.json) | Every Hugging Face model and transcoder the experiments load, with the exact revision (commit hash) used. |

Set up a fresh environment (any tool; for example):

```bash
python3.13 -m venv .venv && .venv/bin/python -m pip install -r env/requirements-research.txt
```

Code loads models at the recorded revision, so a later upload to the Hub cannot
change a result. The route-accounting experiments go through
`experiments/couplet_routes/models.py` (`load`, `pinned`); the older experiments'
configs name models and transcoders as `hf:<org>/<name>@<revision>[/<subfolder>]`
and repository files by repository-relative paths, resolved by
[`experiments/lib/model_refs.py`](../experiments/lib/model_refs.py). Recorded model
repositories are downloaded on first use; transcoder repositories hold many large
weight sets and are never downloaded automatically (fetch the subfolder you need, for
example with `skills/hf-range-streaming`). Result files written before 2026-10-16 may
still record absolute paths from the development Mac; the revision hash in such a
path is the one recorded here.

Known quirk: with transformers 4.57.3, loading a tokenizer by repository name makes
one metadata request to the Hub (a Mistral-tokenizer check) even when the files are
cached, so `HF_HUB_OFFLINE=1` makes the load fail. Run online; the weights still come
from the cache.

## Mac Studio (large-memory machine)

Read from the machine on 2026-10-16:

- [`requirements-remote.txt`](requirements-remote.txt): the main environment (Python 3.9.6,
  torch 2.8.0, transformers 4.57.6), which ran the Qwen3 and Gemma 3 jobs.
- [`requirements-remote-next.txt`](requirements-remote-next.txt): the second environment
  (Python 3.12.14, built with `uv`; torch 2.14.0, transformers 5.17.0), which ran Qwen3.5
  and Gemma 4.
- The revisions of every model run there are in `models.json`. The Qwen3.5 and Gemma 4
  weights had been removed from that machine's cache; its records still name the revision
  used. No transcoder repository is in that cache: the feature-card step streamed only the
  parts it needed (`skills/hf-range-streaming`).

To account for remote work again (results not yet fetched, failed jobs, revisions), use
`inventory.sh` and `compare_inventory.py` in `skills/mac-studio-remote/scripts/`.
