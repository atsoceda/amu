#!/usr/bin/env python3
"""Criterion 1 of the scale extension (README, frozen 2026-09-25): does the per-couplet
relay share rise with log parameter count across Qwen3 1.7B-32B?

Per couplet, share = relay / persistence (original line-2 words), for couplets with
|persistence| > 1 (as in make_report.py). Slope = OLS of share on log10(parameters,
billions), pooled over couplets. CI: 10,000 bootstrap resamples of couplets within
each size. Also reported: the same slope with shares clipped to [-1, 2] (robust to
ratio noise when persistence is small) and the slope of per-size medians.
Writes results/scale_trend.json.
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
from pathlib import Path

EXP = Path(__file__).resolve().parent
SIZES = {"Qwen3-1.7B": 1.7, "Qwen3-4B": 4.0, "Qwen3-8B": 8.0, "Qwen3-14B": 14.0, "Qwen3-32B": 32.0}


def slope(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def main() -> None:
    shares = {}
    for m in SIZES:
        rows = json.loads((EXP / "results" / m / "step34_rows.json").read_text())
        shares[m] = [r["d_relay"] / r["d_persistence_p0"] for r in rows if abs(r["d_persistence_p0"]) > 1]
    rng, out = random.Random(20260925), {}
    for name, f in {"raw": lambda s: s, "clipped": lambda s: min(max(s, -1.0), 2.0)}.items():
        pts = [(math.log10(SIZES[m]), f(s)) for m in SIZES for s in shares[m]]
        est = slope(*zip(*pts))
        boots = []
        for _ in range(10000):
            bp = [(math.log10(SIZES[m]), f(s)) for m in SIZES for s in rng.choices(shares[m], k=len(shares[m]))]
            boots.append(slope(*zip(*bp)))
        boots.sort()
        out[name] = {"slope_per_decade": est, "lo": boots[250], "hi": boots[9749]}
    med = [st.median(shares[m]) for m in SIZES]
    out["median_by_size"] = dict(zip(SIZES, med))
    out["slope_of_medians_per_decade"] = slope([math.log10(v) for v in SIZES.values()], med)
    out["n_by_size"] = {m: len(v) for m, v in shares.items()}
    (EXP / "results" / "scale_trend.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
