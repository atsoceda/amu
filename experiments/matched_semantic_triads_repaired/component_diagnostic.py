#!/usr/bin/env python3
"""Artifact-only component and baseline-gap diagnostics for matched triads."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"
SEED = 20260912
RESAMPLES = 10_000


def load(path: Path):
    return json.loads(path.read_text())


def interval(values, seed):
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(RESAMPLES)
    )
    return {
        "n": n,
        "mean": sum(values) / n,
        "lo": draws[math.floor(0.025 * (RESAMPLES - 1))],
        "hi": draws[math.ceil(0.975 * (RESAMPLES - 1))],
        "positive_n": sum(value > 0 for value in values),
        "method": "paired semantic-triad bootstrap",
        "resamples": RESAMPLES,
    }


def correlation(left, right):
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered))
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    return numerator / denominator


def ranks(values):
    ordered = sorted((value, index) for index, value in enumerate(values))
    output = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][0] == ordered[position][0]:
            end += 1
        rank = (position + end - 1) / 2
        for _, index in ordered[position:end]:
            output[index] = rank
        position = end
    return output


def permutation_p(left, right, seed, permutations=100_000):
    rng = random.Random(seed)
    observed = abs(correlation(left, right))
    exceedances = 0
    shuffled = list(right)
    for _ in range(permutations):
        rng.shuffle(shuffled)
        exceedances += abs(correlation(left, shuffled)) >= observed
    return {
        "observed_absolute_pearson": observed,
        "two_sided_p": (exceedances + 1) / (permutations + 1),
        "permutations": permutations,
    }


def main():
    cfg = load(EXP / "config.json")
    rows = [
        row
        for row in load(RESULTS / "rows.json")
        if row["strength"] == cfg["primary_strength"]
    ]
    potency = load(RESULTS / "potency_audit.json")
    gaps = {
        row["triad_id"]: row["cross_minus_within"][
            "baseline_target_minus_source_gap"
        ]
        for row in potency["rows"]
    }
    by_triad = {}
    for row in rows:
        cell = row["stochastic"][str(cfg["primary_temperature"])]
        public = cell["public"]["target_minus_source"]
        private = cell["private"]["target_minus_source"]
        by_triad.setdefault(row["triad_id"], {})[row["arm"]] = {
            "public": public,
            "private": private,
            "route_contrast": public - private,
        }

    output_rows = []
    for triad_id in sorted(by_triad):
        within = by_triad[triad_id]["within"]
        cross = by_triad[triad_id]["cross"]
        output_rows.append(
            {
                "triad_id": triad_id,
                "within": within,
                "cross": cross,
                "cross_minus_within": {
                    "public": cross["public"] - within["public"],
                    "private": cross["private"] - within["private"],
                    "route_contrast": (
                        cross["route_contrast"] - within["route_contrast"]
                    ),
                    "baseline_target_minus_source_gap": gaps[triad_id],
                },
            }
        )

    summaries = {}
    metric_index = 0
    for arm in ("within", "cross"):
        for component in ("public", "private"):
            summaries[f"{arm}_{component}"] = interval(
                [row[arm][component] for row in output_rows],
                SEED + metric_index,
            )
            metric_index += 1
    for component in ("public", "private", "route_contrast"):
        summaries[f"paired_cross_minus_within_{component}"] = interval(
            [row["cross_minus_within"][component] for row in output_rows],
            SEED + metric_index,
        )
        metric_index += 1

    gap_values = [
        row["cross_minus_within"]["baseline_target_minus_source_gap"]
        for row in output_rows
    ]
    route_values = [
        row["cross_minus_within"]["route_contrast"] for row in output_rows
    ]
    pearson = correlation(gap_values, route_values)
    spearman = correlation(ranks(gap_values), ranks(route_values))
    leave_one_out = [
        correlation(
            gap_values[:index] + gap_values[index + 1 :],
            route_values[:index] + route_values[index + 1 :],
        )
        for index in range(len(output_rows))
    ]
    result = {
        "diagnostic": "matched-triad public/private components and baseline gap",
        "primary_strength": cfg["primary_strength"],
        "primary_temperature": cfg["primary_temperature"],
        "n_triads": len(output_rows),
        "rows": output_rows,
        "component_summaries": summaries,
        "baseline_gap_vs_paired_route_shift": {
            "pearson_r": pearson,
            "spearman_rho": spearman,
            "leave_one_out_pearson_min": min(leave_one_out),
            "leave_one_out_pearson_max": max(leave_one_out),
            "permutation": permutation_p(
                gap_values, route_values, SEED + metric_index
            ),
        },
    }
    (RESULTS / "component_diagnostic.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result["component_summaries"], indent=2))
    print(json.dumps(result["baseline_gap_vs_paired_route_shift"], indent=2))


if __name__ == "__main__":
    main()
