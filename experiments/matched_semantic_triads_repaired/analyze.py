#!/usr/bin/env python3
"""Paired primary and robustness summaries for repaired matched triads."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"
SEED = 20260902
RESAMPLES = 10_000


def interval(values, seed):
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(RESAMPLES))
    return {"n": n, "mean": sum(values) / n,
            "lo": draws[math.floor(.025 * (RESAMPLES - 1))],
            "hi": draws[math.ceil(.975 * (RESAMPLES - 1))],
            "method": "paired semantic-triad bootstrap", "resamples": RESAMPLES}


def main():
    cfg = json.loads((EXP / "config.json").read_text())
    rows = json.loads((RESULTS / "rows.json").read_text())
    out = {"experiment": cfg["experiment_name"], "settings": {}}
    for strength in cfg["strengths"]:
        out["settings"][str(strength)] = {}
        for tau in cfg["temperatures"]:
            paired = []
            for triad_id in sorted({r["triad_id"] for r in rows}):
                arms = {r["arm"]: r for r in rows if r["triad_id"] == triad_id and r["strength"] == strength}
                contrasts = {}
                for arm in ("within", "cross"):
                    cell = arms[arm]["stochastic"][str(tau)]
                    contrasts[arm] = cell["public"]["target_minus_source"] - cell["private"]["target_minus_source"]
                paired.append({"triad_id": triad_id, "within": contrasts["within"], "cross": contrasts["cross"],
                               "interaction": contrasts["cross"] - contrasts["within"]})
            out["settings"][str(strength)][str(tau)] = {
                "within_route": interval([p["within"] for p in paired], SEED + 1),
                "cross_route": interval([p["cross"] for p in paired], SEED + 2),
                "paired_interaction": interval([p["interaction"] for p in paired], SEED + 3),
                "positive_interaction_n": sum(p["interaction"] > 0 for p in paired),
                "full_double_dissociation_n": sum(p["cross"] > 0 and p["within"] < 0 for p in paired),
                "rows": paired,
            }
    (RESULTS / "analysis.json").write_text(json.dumps(out, indent=2) + "\n")
    primary = out["settings"][str(cfg["primary_strength"])][str(cfg["primary_temperature"])]
    print(json.dumps(primary, indent=2))


if __name__ == "__main__":
    main()
