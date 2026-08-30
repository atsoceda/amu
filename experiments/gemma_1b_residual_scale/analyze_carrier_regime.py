#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
from pathlib import Path

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"
ROWS = json.loads((RESULTS / "rows.json").read_text())
TEMPERATURES = ("0.1", "0.25", "0.5", "1.0")
STRENGTHS = (0.5, 1.0, 1.5)


def mean(values):
    return sum(values) / len(values)


def group(regime, strength):
    return [row for row in ROWS if row["regime"] == regime and float(row["strength"]) == strength]


def bootstrap_interaction(tau, metric, seed=20260830, resamples=10000):
    rng = random.Random(seed)
    cells = {(regime, strength): group(regime, strength) for regime in ("between", "within") for strength in (0.5, 1.5)}

    def value(sampled):
        cell_mean = {}
        for key, rows in sampled.items():
            cell_mean[key] = mean([row["stochastic"][tau][metric] for row in rows])
        return ((cell_mean[("between", 1.5)] - cell_mean[("between", 0.5)]) -
                (cell_mean[("within", 1.5)] - cell_mean[("within", 0.5)]))

    observed = value(cells)
    draws = []
    for _ in range(resamples):
        sampled = {key: [rows[rng.randrange(len(rows))] for _ in rows] for key, rows in cells.items()}
        draws.append(value(sampled))
    draws.sort()
    return {"estimate": observed, "lo": draws[math.floor(.025*(resamples-1))],
            "hi": draws[math.ceil(.975*(resamples-1))], "resamples": resamples}


def main():
    out = {"experiment": "gemma_1b_carrier_regime_interaction",
           "contrast": "(between strength 1.5 - 0.5) - (within strength 1.5 - 0.5)",
           "cells": {}, "interactions": {}}
    for regime in ("between", "within"):
        out["cells"][regime] = {}
        for strength in STRENGTHS:
            rows = group(regime, strength)
            block = {"n": len(rows), "greedy_article_change_rate": mean([float(row["article_changed"]) for row in rows]),
                     "target_delta_delta": mean([row["target_delta_delta"] for row in rows])}
            for tau in TEMPERATURES:
                block[tau] = {metric: mean([row["stochastic"][tau][metric] for row in rows])
                              for metric in ("public_tv", "private_tv", "total_tv", "off_article_mass", "on_article_mass")}
            out["cells"][regime][str(strength)] = block
    for temperature_index, tau in enumerate(TEMPERATURES):
        out["interactions"][tau] = {
            metric: bootstrap_interaction(tau, metric, seed=20260830 + temperature_index*10 + metric_index)
            for metric_index, metric in enumerate(("public_tv", "private_tv", "total_tv"))
        }
    (RESULTS / "carrier_regime_interaction.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = ["# Carrier-regime interaction", "", "Interaction: (between 1.5 - 0.5) - (within 1.5 - 0.5).", "",
             "| Temperature | Public TV interaction | 95% CI | Private TV interaction | 95% CI |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for tau, block in out["interactions"].items():
        public, private = block["public_tv"], block["private_tv"]
        lines.append(f"| {tau} | {public['estimate']:.3f} | [{public['lo']:.3f}, {public['hi']:.3f}] | "
                     f"{private['estimate']:.3f} | [{private['lo']:.3f}, {private['hi']:.3f}] |")
    (RESULTS / "carrier_regime_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
