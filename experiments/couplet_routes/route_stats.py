"""Torch-free helpers shared by the route-accounting experiments.

`boot` is the percentile bootstrap used for every committed summary (10,000
resamples, seed 20260925). It lives here, not in step34_routes.py, so summaries can
be recomputed from committed rows without loading torch or a model.
"""
from __future__ import annotations

import random


def boot(xs, n=10000, seed=20260925):
    xs = [x for x in xs if x == x]
    if not xs:
        return None
    rng = random.Random(seed)
    bs = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return {"mean": sum(xs) / len(xs), "lo": bs[int(0.025 * n)], "hi": bs[int(0.975 * n) - 1], "n": len(xs)}


def strip_last_word(line: str) -> str | None:
    s = line.rstrip()
    s = s.rstrip(".,;:!?\"'")
    i = s.rstrip().rfind(" ")
    return s[:i] if i > 0 else None
