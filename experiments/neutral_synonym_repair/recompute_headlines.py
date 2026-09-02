#!/usr/bin/env python3
"""Independent artifact-only recomputation of every Figure 4 headline.

This file intentionally imports no experiment analysis or plotting code.
"""
from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
TOLERANCE = 1e-12
RESAMPLES = 10_000


def mean(values):
    return sum(values) / len(values)


def interval(values, seed):
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(RESAMPLES))
    return {
        "n": n,
        "mean": mean(values),
        "lo": draws[math.floor(.025 * (RESAMPLES - 1))],
        "hi": draws[math.ceil(.975 * (RESAMPLES - 1))],
        "resamples": RESAMPLES,
    }


def exact_sign_flip(values):
    observed = mean(values)
    null = [mean([s * v for s, v in zip(signs, values)]) for signs in itertools.product((-1, 1), repeat=len(values))]
    return {
        "observed": observed,
        "one_sided_p": sum(x >= observed - 1e-15 for x in null) / len(null),
        "two_sided_p": sum(abs(x) >= abs(observed) - 1e-15 for x in null) / len(null),
        "assignments": len(null),
    }


def exact_label_permutation(left, right):
    values = left + right
    observed = mean(left) - mean(right)
    null = []
    for indices in itertools.combinations(range(len(values)), len(left)):
        chosen = set(indices)
        a = [v for i, v in enumerate(values) if i in chosen]
        b = [v for i, v in enumerate(values) if i not in chosen]
        null.append(mean(a) - mean(b))
    return {
        "observed": observed,
        "two_sided_p": sum(abs(x) >= abs(observed) - 1e-12 for x in null) / len(null),
        "assignments": len(null),
    }


def assert_close(label, actual, expected, tolerance=TOLERANCE):
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: recomputed {actual!r}, frozen summary {expected!r}")


def main():
    rows = json.loads((RESULTS / "rows.json").read_text())
    screen = json.loads((RESULTS / "screen_rows.json").read_text())
    frozen_summary = json.loads((RESULTS / "summary.json").read_text())
    frozen_analysis = json.loads((RESULTS / "analysis.json").read_text())
    primary = [r for r in rows if r["strength"] == 1.0]
    if len(primary) != 14:
        raise AssertionError(f"Expected 14 primary rows, found {len(primary)}")

    output = {
        "runner": "independent artifact-only Figure 4 recomputation",
        "inputs": ["results/rows.json", "results/screen_rows.json"],
        "imports_experiment_code": False,
        "primary_setting": {"strength": 1.0, "temperature": 1.0},
        "groups": {},
    }
    route_values = {}
    for regime, expected_n in (("between", 6), ("within", 8)):
        group = [r for r in primary if r["regime"] == regime]
        if len(group) != expected_n:
            raise AssertionError(f"{regime}: expected {expected_n}, found {len(group)}")
        local = [r["target_branch_delta_delta"] for r in group]
        per_family = []
        for row in group:
            cell = row["stochastic"]["1.0"]
            public = cell["public"]["target_minus_source"]
            private = cell["private"]["target_minus_source"]
            per_family.append({
                "pair_id": row["pair_id"],
                "public_target_effect": public,
                "private_target_effect": private,
                "route_contrast_public_minus_private": public - private,
                "off_article_mass": cell["off_article_mass"],
                "on_article_mass": cell["on_article_mass"],
                "local_efficacy": row["target_branch_delta_delta"],
            })
        raw_route = [r["route_contrast_public_minus_private"] for r in per_family]
        signed_simple = raw_route if regime == "between" else [-v for v in raw_route]
        route_values[regime] = raw_route
        output["groups"][regime] = {
            "n": len(group),
            "local_efficacy_pass_n": sum(v > 0 for v in local),
            "local_efficacy": interval(local, 20260912),
            "mean_off_article_mass": mean([r["off_article_mass"] for r in per_family]),
            "minimum_off_article_mass": min(r["off_article_mass"] for r in per_family),
            "mean_on_article_mass": mean([r["on_article_mass"] for r in per_family]),
            "public_target_effect": mean([r["public_target_effect"] for r in per_family]),
            "private_target_effect": mean([r["private_target_effect"] for r in per_family]),
            "route_contrasts": raw_route,
            "predicted_sign_n": sum(v > 0 for v in signed_simple),
            "signed_simple_effect": interval(signed_simple, 20260911),
            "signed_simple_exact": exact_sign_flip(signed_simple),
            "families": per_family,
        }

        expected_condition = frozen_summary["conditions"][f"{regime}_1.0"]
        assert_close(f"{regime} local mean", mean(local), expected_condition["target_branch_delta_delta"]["mean"])
        expected_effect = frozen_analysis["effects"][regime]["target_aligned"]
        for field in ("mean", "lo", "hi"):
            assert_close(f"{regime} signed simple {field}", output["groups"][regime]["signed_simple_effect"][field], expected_effect["interval"][field])
        assert_close(f"{regime} sign-flip p", output["groups"][regime]["signed_simple_exact"]["one_sided_p"], expected_effect["randomization"]["one_sided_positive_p"])
        tau = expected_condition["temperatures"]["1.0"]
        assert_close(f"{regime} public target", output["groups"][regime]["public_target_effect"], tau["public_target_minus_source"]["mean"])
        assert_close(f"{regime} private target", output["groups"][regime]["private_target_effect"], tau["private_target_minus_source"]["mean"])

    interaction = [*route_values["between"]]
    within = route_values["within"]
    interaction_interval_draws = []
    rng = random.Random(20260902 + 500)
    for _ in range(RESAMPLES):
        interaction_interval_draws.append(
            mean([interaction[rng.randrange(len(interaction))] for _ in interaction])
            - mean([within[rng.randrange(len(within))] for _ in within])
        )
    interaction_interval_draws.sort()
    output["interaction"] = {
        "mean": mean(interaction) - mean(within),
        "lo": interaction_interval_draws[math.floor(.025 * (RESAMPLES - 1))],
        "hi": interaction_interval_draws[math.ceil(.975 * (RESAMPLES - 1))],
        "exact_permutation": exact_label_permutation(interaction, within),
    }
    expected_interaction = frozen_summary["interactions"]["strength_1.0"]["1.0"]
    for field in ("mean", "lo", "hi"):
        assert_close(f"interaction {field}", output["interaction"][field], expected_interaction["aligned_route_interaction"][field])
    assert_close("interaction exact p", output["interaction"]["exact_permutation"]["two_sided_p"], expected_interaction["exact_permutation"]["two_sided_p"])

    all_primary = [r["stochastic"]["1.0"] for r in primary]
    output["article_support"] = {
        "n": len(all_primary),
        "mean_off_mass": mean([r["off_article_mass"] for r in all_primary]),
        "minimum_off_mass": min(r["off_article_mass"] for r in all_primary),
        "mean_on_mass": mean([r["on_article_mass"] for r in all_primary]),
        "minimum_on_mass": min(r["on_article_mass"] for r in all_primary),
    }
    output["all_checks_passed"] = True
    (RESULTS / "independent_recomputation.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
