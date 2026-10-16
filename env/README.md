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

Load models at the recorded revision, for example
`AutoModelForCausalLM.from_pretrained(name, revision=models["models"][name])`, so a
later upload to the Hub cannot change a result. Several experiment configs still
name a local snapshot folder on the development Mac (`model_snapshot`); the
revision hash in that path is the same one recorded here.

## Not yet recorded

- **Models run on the large-memory machine** (Qwen3-8B to 32B, Qwen3.5, Gemma 3
  12B and 27B, Gemma 4 31B) and their transcoders: revisions are marked as
  pending in `models.json` until they are read from that machine's cache.
- **That machine's two environments** (Python 3.9 with transformers 4.57, and
  Python 3.12 with transformers 5.x for Qwen3.5 and Gemma 4; see
  [`skills/mac-studio-remote`](../skills/mac-studio-remote/SKILL.md)): record
  their `pip freeze` here as `requirements-remote.txt` and
  `requirements-remote-next.txt`.

## Verifying without models

`bin/verify-paper` needs only numpy and matplotlib: it regenerates the working
paper's figures from the committed results and checks its headline numbers. See
the root README.
