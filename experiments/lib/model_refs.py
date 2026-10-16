"""Portable references to models, transcoders and repository files.

Experiment configs name what they load in a machine-independent form:

- ``hf:<org>/<name>@<revision>`` or ``hf:<org>/<name>@<revision>/<subfolder>``: a
  Hugging Face repository at an exact revision (commit hash);
- a repository-relative path such as ``experiments/lib/transcoder_configs/x.yaml``.

``local(ref)`` turns either into a path on this machine. A Hugging Face reference maps to
this machine's cache folder for that revision (``$HF_HUB_CACHE``, by default
``~/.cache/huggingface/hub``). A model repository listed under "models" in
``env/models.json`` is downloaded on first use if it is missing; transcoder repositories are
never downloaded here, because they hold many large weight sets: fetch only the subfolder
you need (for example with ``skills/hf-range-streaming``) or use the experiment's streaming
option. Absolute paths written on another machine's cache
(``.../models--<org>--<name>/snapshots/<rev>/...``) are read as the same reference, so old
configs and result files stay usable. Torch-free and Python 3.9 compatible.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_JSON = ROOT / "env" / "models.json"
_CACHE_PATH = re.compile(r".*/models--([^/]+?)--([^/]+)/snapshots/([0-9a-f]{40})(?:/(.+))?/?$")
_REF = re.compile(r"^hf:([^/@]+)/([^/@]+)@([0-9a-f]{40})(?:/(.+))?$")


def recorded() -> dict:
    """{"models": {repo: revision}, "transcoders": {repo: revision}} from env/models.json."""
    data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    return {"models": data.get("models", {}), "transcoders": data.get("transcoders", {})}


def revision(repo_id: str):
    """Recorded revision of a model or transcoder repository, or None if not recorded."""
    rec = recorded()
    return rec["models"].get(repo_id) or rec["transcoders"].get(repo_id)


def parse(ref: str):
    """(repo_id, revision, subfolder or None) for an hf: reference or a cache path; else None."""
    m = _REF.match(ref) or _CACHE_PATH.match(ref)
    if not m:
        return None
    org, name, rev, sub = m.groups()
    return f"{org}/{name}", rev, (sub.strip("/") or None) if sub else None


def to_ref(path: str) -> str:
    """Portable form of a path: hf: reference for a cache path, repository-relative for a repo file."""
    p = parse(path)
    if p:
        repo, rev, sub = p
        return f"hf:{repo}@{rev}" + (f"/{sub}" if sub else "")
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return path


def hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def local(ref):
    """Path on this machine for a portable reference (see the module docstring)."""
    if not ref:
        return ref
    s = str(ref)
    p = parse(s)
    if p is None:
        if os.path.isabs(s):
            return s
        return str(ROOT / s) if (ROOT / s).exists() or not Path(s).exists() else s
    repo, rev, sub = p
    if not s.startswith("hf:") and Path(s).exists():
        return s  # an existing cache path on this machine
    org, name = repo.split("/")
    snap = hub_cache() / f"models--{org}--{name}" / "snapshots" / rev
    if not snap.exists() and repo in recorded()["models"]:
        from huggingface_hub import snapshot_download

        snap = Path(snapshot_download(repo_id=repo, revision=rev))
    return str(snap / sub) if sub else str(snap)
