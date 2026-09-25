"""Byte-range access to Hugging Face files, so transcoders never need to be stored whole.

Per-layer transcoder safetensors and feature-visualization binaries are read by
HTTP Range requests: only the encoder of the current layer, the visualization
records of active features, and the decoder rows of selected features are fetched.
"""
from __future__ import annotations

import gzip
import json
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch

HF = "https://huggingface.co"


def url(repo: str, path: str) -> str:
    return f"{HF}/{repo}/resolve/main/{path}"


import re as _re
import threading as _threading

_CDN: dict[str, tuple[str, float]] = {}
_CDN_LOCK = _threading.Lock()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401 - stop at the 302
        return None


def _resolve(u: str) -> str:
    """huggingface.co/.../resolve/... answers with a 302 to a signed CDN URL; each
    resolve counts against Hugging Face's 3000-per-5-minutes limit, so we resolve
    once per file and send range requests straight to the CDN until it expires."""
    with _CDN_LOCK:
        hit = _CDN.get(u)
        if hit and hit[1] - 120 > time.time():
            return hit[0]
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            opener.open(urllib.request.Request(u, headers={"Range": "bytes=0-0"}), timeout=60)
            cdn, expires = u, time.time() + 3600  # no redirect: use as is
        except urllib.error.HTTPError as e:
            if e.code not in (301, 302, 303, 307, 308):
                raise
            cdn = e.headers["Location"]
            m = _re.search(r"Expires=(\d+)", cdn)
            expires = float(m.group(1)) if m else time.time() + 600
        _CDN[u] = (cdn, expires)
        return cdn


def fetch_range(u: str, start: int, end_inclusive: int, retries: int = 12) -> bytes:
    """Fetch bytes [start, end_inclusive] with retries.

    HTTP 429 (Hugging Face rate limit) honours Retry-After, else backs off up to 5 minutes.
    """
    import urllib.error
    want = end_inclusive - start + 1
    for attempt in range(retries):
        try:
            target = _resolve(u) if "/resolve/" in u else u
            req = urllib.request.Request(target, headers={"Range": f"bytes={start}-{end_inclusive}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != want:
                raise IOError(f"short read {len(data)} != {want}")
            return data
        except urllib.error.HTTPError as e:
            if attempt == retries - 1:
                raise
            if e.code in (403, 410):  # signed CDN URL expired or rejected: resolve again
                with _CDN_LOCK:
                    _CDN.pop(u, None)
                continue
            if e.code == 429:
                wait = e.headers.get("Retry-After")
                time.sleep(float(wait) if wait and wait.isdigit() else min(300, 15 * 2 ** attempt))
            else:
                time.sleep(2 ** attempt)
        except Exception:  # noqa: BLE001 - other network errors are retried
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


class SafetensorsRemote:
    """Lazy remote safetensors file: parses the header, fetches tensors or rows on demand."""

    DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    SIZES = {"BF16": 2, "F16": 2, "F32": 4}

    def __init__(self, repo: str, path: str, cache_dir: Path | None = None):
        self.u = url(repo, path)
        cache = cache_dir / (path.replace("/", "__") + ".header.json") if cache_dir else None
        if cache and cache.exists():
            meta = json.loads(cache.read_text())
        else:
            n = struct.unpack("<Q", fetch_range(self.u, 0, 7))[0]
            meta = {"n": n, "header": json.loads(fetch_range(self.u, 8, 7 + n))}
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(meta))
        self.base = 8 + meta["n"]
        self.header = {k: v for k, v in meta["header"].items() if k != "__metadata__"}

    def tensor(self, name: str) -> torch.Tensor:
        h = self.header[name]
        a, b = h["data_offsets"]
        raw = bytearray(fetch_range(self.u, self.base + a, self.base + b - 1))
        return torch.frombuffer(raw, dtype=self.DTYPES[h["dtype"]]).reshape(h["shape"])

    def rows(self, name: str, idx: list[int], max_gap_rows: int = 64) -> dict[int, torch.Tensor]:
        """Fetch selected rows of a 2-D tensor, merging nearby rows into one request."""
        h = self.header[name]
        n_rows, width = h["shape"]
        row_bytes = width * self.SIZES[h["dtype"]]
        out: dict[int, torch.Tensor] = {}
        idx = sorted(set(idx))
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and idx[j + 1] - idx[j] <= max_gap_rows:
                j += 1
            lo, hi = idx[i], idx[j]
            a = self.base + h["data_offsets"][0] + lo * row_bytes
            raw = bytearray(fetch_range(self.u, a, a + (hi - lo + 1) * row_bytes - 1))
            block = torch.frombuffer(raw, dtype=self.DTYPES[h["dtype"]]).reshape(hi - lo + 1, width)
            for k in idx[i : j + 1]:
                out[k] = block[k - lo].clone()
            i = j + 1
        return out


def load_feature_index(index_path: Path) -> dict:
    with gzip.open(index_path, "rt") as f:
        return json.load(f)


def fetch_feature_records(repo: str, index: dict, layer: int, feats: list[int],
                          max_gap_bytes: int = 1 << 16, workers: int = 32) -> dict[int, dict | None]:
    """Fetch and decode visualization records (same binary format as the authors' loader).

    Nearby records are merged into one range request; groups are fetched concurrently.
    """
    from concurrent.futures import ThreadPoolExecutor
    layer_info = index[str(layer)]
    offsets, fname = layer_info["offsets"], layer_info["filename"]
    u = url(repo, f"features/{fname}")
    feats = sorted(set(feats))
    groups, i = [], 0
    while i < len(feats):
        j = i
        while j + 1 < len(feats) and offsets[feats[j + 1]] - offsets[feats[j] + 1] <= max_gap_bytes:
            j += 1
        groups.append(feats[i : j + 1])
        i = j + 1

    def get(group):
        lo, hi = offsets[group[0]], offsets[group[-1] + 1]
        blob = fetch_range(u, lo, hi - 1) if hi > lo else b""
        res = {}
        for f in group:
            a, b = offsets[f] - lo, offsets[f + 1] - lo
            if a == b:
                res[f] = None
                continue
            chunk = blob[a:b]
            n = struct.unpack("<I", chunk[:4])[0]
            res[f] = json.loads(gzip.decompress(chunk[4 : 4 + n]).decode("utf-8"))
        return res

    out: dict[int, dict | None] = {}
    with ThreadPoolExecutor(workers) as ex:
        for res in ex.map(get, groups):
            out.update(res)
    return out
