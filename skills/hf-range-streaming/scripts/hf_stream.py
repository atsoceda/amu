#!/usr/bin/env python3
"""Byte-range access to large Hugging Face files (library + command-line tool).

Library: SafetensorsRemote (headers, tensors, rows), fetch_feature_records (feature
cards), fetch_range (rate-limit-safe range requests that resolve each file once and
then go straight to the signed CDN URL).

Command line (run with --help for options):
  sizes   REPO                       list files and total sizes
  header  REPO FILE                  print a remote safetensors header, no download
  probe   REPO FILE                  redirect, rate-limit headers, CDN expiry, timings
  plan    INDEX LAYER FEATURES       requests vs bytes for several merge gaps (offline)
  cards   REPO INDEX LAYER FEATURES  fetch feature cards to JSON (--verify against a cache)

Dependencies: Python 3.10+ standard library; torch only for tensor/row decoding.
"""
from __future__ import annotations

import gzip
import json
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

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
            if 400 <= e.code < 500 and e.code != 429:
                raise  # client errors (401 gated, 404 missing, 416 bad range) do not improve with retries
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

    SIZES = {"BF16": 2, "F16": 2, "F32": 4}

    @staticmethod
    def _dtype(name: str):
        import torch
        return {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[name]

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

    def tensor(self, name: str):
        import torch
        h = self.header[name]
        a, b = h["data_offsets"]
        raw = bytearray(fetch_range(self.u, self.base + a, self.base + b - 1))
        return torch.frombuffer(raw, dtype=self._dtype(h["dtype"])).reshape(h["shape"])

    def rows(self, name: str, idx: list[int], max_gap_rows: int = 64) -> dict:
        import torch
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
            block = torch.frombuffer(raw, dtype=self._dtype(h["dtype"])).reshape(hi - lo + 1, width)
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


# ----------------------------------------------------------------------------- CLI


def _parse_features(arg: str) -> list[int]:
    """Comma list and ranges ("3,17,100-120"), or @file with one integer per line."""
    if arg.startswith("@"):
        return sorted({int(x) for x in Path(arg[1:]).read_text().split()})
    out = set()
    for part in arg.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def _cmd_sizes(a) -> None:
    api = f"{HF}/api/models/{a.repo}/tree/main?recursive=true"
    with urllib.request.urlopen(api, timeout=60) as r:
        tree = json.load(r)
    files = [x for x in tree if x.get("type") == "file"]
    if a.prefix:
        files = [x for x in files if x["path"].startswith(a.prefix)]
    for x in sorted(files, key=lambda x: x["path"])[: a.limit]:
        print(f"{x.get('size', 0) / 1e6:12.1f} MB  {x['path']}")
    print(f"{len(files)} files, {sum(x.get('size', 0) for x in files) / 1e9:.2f} GB total")


def _cmd_header(a) -> None:
    st = SafetensorsRemote(a.repo, a.file)
    for k, h in st.header.items():
        lo, hi = h["data_offsets"]
        print(f"{k:32s} {h['dtype']:5s} {str(h['shape']):22s} {(hi - lo) / 1e6:10.1f} MB  bytes {st.base + lo}-{st.base + hi - 1}")


def _cmd_probe(a) -> None:
    u = url(a.repo, a.file)
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        opener.open(urllib.request.Request(u, headers={"Range": "bytes=0-0"}), timeout=60)
        print("no redirect (served directly)")
        cdn = u
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} redirect")
        for h in ("ratelimit", "ratelimit-policy", "retry-after"):
            if e.headers.get(h):
                print(f"  {h}: {e.headers[h]}")
        cdn = e.headers.get("Location", u)
        m = _re.search(r"Expires=(\d+)", cdn)
        if m:
            print(f"  signed CDN URL expires in {int(m.group(1)) - int(time.time())} s")
    for label, target in (("via /resolve/", u), ("direct to CDN", cdn)):
        ts = []
        for i in range(a.n):
            t = time.time()
            req = urllib.request.Request(target, headers={"Range": f"bytes={i * 1000000}-{i * 1000000 + 8191}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                r.read()
            ts.append(time.time() - t)
        print(f"{label:14s}: mean {sum(ts) / len(ts):.2f} s per 8 KB request over {a.n}")


def _groups(offsets, feats, gap):
    groups, i = [], 0
    while i < len(feats):
        j = i
        while j + 1 < len(feats) and offsets[feats[j + 1]] - offsets[feats[j] + 1] <= gap:
            j += 1
        groups.append((offsets[feats[i]], offsets[feats[j] + 1]))
        i = j + 1
    return groups


def _cmd_plan(a) -> None:
    offsets = load_feature_index(Path(a.index))[str(a.layer)]["offsets"]
    feats = _parse_features(a.features)
    need = sum(offsets[f + 1] - offsets[f] for f in feats)
    print(f"layer {a.layer}: {len(feats)} features, {need / 1e6:.1f} MB of card data, file {offsets[-1] / 1e6:.0f} MB")
    for gap in (1 << 16, 1 << 18, 1 << 20, 4 << 20, 16 << 20):
        g = _groups(offsets, feats, gap)
        print(f"  gap {gap >> 10:6d} KB: {len(g):6d} requests, {sum(b - x for x, b in g) / 1e6:8.1f} MB")


def _cmd_cards(a) -> None:
    index = load_feature_index(Path(a.index))
    feats = _parse_features(a.features)
    t = time.time()
    got = fetch_feature_records(a.repo, index, a.layer, feats, workers=a.workers)
    print(f"fetched {len(got)} cards in {time.time() - t:.1f} s", flush=True)
    if a.verify:
        ref = json.loads(Path(a.verify).read_text())
        shared = [f for f in feats if str(f) in ref]
        bad = [f for f in shared if ref[str(f)] != got[f]]
        print(f"verify: {len(shared) - len(bad)}/{len(shared)} cards identical to {a.verify}"
              + (f"; differing: {bad[:10]}" if bad else ""))
        if bad:
            raise SystemExit(1)
    if a.out:
        Path(a.out).write_text(json.dumps({str(k): v for k, v in got.items()}))
        print(f"wrote {a.out}")


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Byte-range tools for large Hugging Face files.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sizes", help="list repository files and sizes")
    p.add_argument("repo"); p.add_argument("--prefix", default=""); p.add_argument("--limit", type=int, default=50)
    p = sub.add_parser("header", help="print a remote safetensors header")
    p.add_argument("repo"); p.add_argument("file")
    p = sub.add_parser("probe", help="redirect, rate-limit headers and request timings")
    p.add_argument("repo"); p.add_argument("file"); p.add_argument("-n", type=int, default=3)
    p = sub.add_parser("plan", help="requests vs bytes for merge gaps (offline)")
    p.add_argument("index", help="local features/index.json.gz"); p.add_argument("layer", type=int)
    p.add_argument("features", help='e.g. "3,17,100-120" or @ids.txt')
    p = sub.add_parser("cards", help="fetch feature cards")
    p.add_argument("repo"); p.add_argument("index"); p.add_argument("layer", type=int); p.add_argument("features")
    p.add_argument("--workers", type=int, default=32); p.add_argument("--out"); p.add_argument("--verify", help="JSON written earlier by --out; exit 1 unless identical")
    a = ap.parse_args(argv)
    try:
        {"sizes": _cmd_sizes, "header": _cmd_header, "probe": _cmd_probe, "plan": _cmd_plan, "cards": _cmd_cards}[a.cmd](a)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} for {e.url}: {e.reason}. 401/403: repository gated or private "
                         f"(authenticate or accept its license); 404: wrong repo or file path; 429: rate "
                         f"limited, wait and retry (fetch_range already backs off).")
    except FileNotFoundError as e:
        raise SystemExit(f"File not found: {e.filename}. For plan/cards, INDEX is a local copy of "
                         f"features/index.json.gz (download it once with a plain range fetch or curl).")
    except KeyError as e:
        raise SystemExit(f"Not found in header or index: {e}. Use 'header' to list tensors or check the layer number.")


if __name__ == "__main__":
    main()
