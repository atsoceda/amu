#!/usr/bin/env python3
"""Paired target-minus-random and 2x2 factorial summaries from stored rows.

No model forwards. Writes results/derived_controls.json used by plot.py and
the manuscript appendix.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


EXP_DIR = Path(__file__).resolve().parent
RESULTS = EXP_DIR / "results"
CONFIG = json.loads((EXP_DIR / "config.json").read_text())
ROWS = json.loads((RESULTS / "rows.json").read_text())
RANDOM_ROWS = json.loads((RESULTS / "random_rows.json").read_text())
SELECTED_LAYER = 12
RESAMPLES = 10000
SEED = 20260828


def interval(values: list[float], seed: int, resamples: int = RESAMPLES) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(values)
    boot = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    boot.sort()
    return {
        "n": n,
        "mean": sum(values) / n,
        "lo": boot[math.floor(0.025 * (len(boot) - 1))],
        "hi": boot[math.ceil(0.975 * (len(boot) - 1))],
        "method": "pair-level nonparametric bootstrap",
        "resamples": resamples,
    }


def rate_interval(flags: list[bool], seed: int) -> dict[str, Any]:
    values = [1.0 if flag else 0.0 for flag in flags]
    return interval(values, seed)


def main() -> None:
    test_pairs = [p for p in CONFIG["pairs"] if p["split"] == "test"]
    target_rows = {
        (r["pair_id"], r["article"]): r
        for r in ROWS
        if r["split"] == "test" and r["layer"] == SELECTED_LAYER
    }
    random_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in RANDOM_ROWS:
        random_by_pair.setdefault((r["pair_id"], r["article"]), []).append(r)

    pair_rows: list[dict[str, Any]] = []
    for pair in test_pairs:
        pid = pair["id"]
        a = target_rows[(pid, "a")]
        an = target_rows[(pid, "an")]
        random_a = [float(r["delta_delta"]) for r in random_by_pair[(pid, "a")]]
        random_an = [float(r["delta_delta"]) for r in random_by_pair[(pid, "an")]]
        random_gamma = [ran - ra for ra, ran in zip(random_an, random_a)]
        cell = {
            "pair_id": pid,
            "source_word": pair["source_word"],
            "target_word": pair["target_word"],
            "baseline_a": a["baseline"]["target_minus_source"],
            "baseline_an": an["baseline"]["target_minus_source"],
            "patched_a": a["patched"]["target_minus_source"],
            "patched_an": an["patched"]["target_minus_source"],
            "delta_delta_a": a["delta_delta"],
            "delta_delta_an": an["delta_delta"],
            "random_delta_delta_a_mean": sum(random_a) / len(random_a),
            "random_delta_delta_an_mean": sum(random_an) / len(random_an),
            "gamma_target": an["delta_delta"] - a["delta_delta"],
            "gamma_random_mean": sum(random_gamma) / len(random_gamma),
            "baseline_target_rank_a": a["baseline"]["target_rank"],
            "patched_target_rank_a": a["patched"]["target_rank"],
            "baseline_source_rank_a": a["baseline"]["source_rank"],
            "patched_source_rank_a": a["patched"]["source_rank"],
            "baseline_target_prob_a": a["baseline"]["target_prob"],
            "patched_target_prob_a": a["patched"]["target_prob"],
            "baseline_source_prob_a": a["baseline"]["source_prob"],
            "patched_source_prob_a": a["patched"]["source_prob"],
            "target_logit_change_a": (
                a["patched"]["target_logit"] - a["baseline"]["target_logit"]
            ),
            "source_logit_change_a": (
                a["patched"]["source_logit"] - a["baseline"]["source_logit"]
            ),
            "target_prob_change_a": (
                a["patched"]["target_prob"] - a["baseline"]["target_prob"]
            ),
            "source_prob_change_a": (
                a["patched"]["source_prob"] - a["baseline"]["source_prob"]
            ),
            "patch_tv_a": a["patch_tv"],
            "patch_tv_an": an["patch_tv"],
        }
        cell["paired_d_a"] = cell["delta_delta_a"] - cell["random_delta_delta_a_mean"]
        cell["paired_d_an"] = cell["delta_delta_an"] - cell["random_delta_delta_an_mean"]
        cell["paired_gamma"] = cell["gamma_target"] - cell["gamma_random_mean"]
        cell["public_off"] = cell["baseline_an"] - cell["baseline_a"]
        cell["public_on"] = cell["patched_an"] - cell["patched_a"]
        gap = abs(cell["baseline_a"])
        cell["gap_closed_a"] = cell["delta_delta_a"] / gap if gap else None
        cell["target_topk10_before"] = cell["baseline_target_rank_a"] <= 10
        cell["target_topk10_after"] = cell["patched_target_rank_a"] <= 10
        pair_rows.append(cell)

    summary = {
        "n_test_pairs": len(pair_rows),
        "selected_layer": SELECTED_LAYER,
        "paired_d_a": interval([r["paired_d_a"] for r in pair_rows], SEED + 301),
        "paired_d_an": interval([r["paired_d_an"] for r in pair_rows], SEED + 302),
        "paired_gamma": interval([r["paired_gamma"] for r in pair_rows], SEED + 303),
        "gamma_target": interval([r["gamma_target"] for r in pair_rows], SEED + 304),
        "gamma_random": interval([r["gamma_random_mean"] for r in pair_rows], SEED + 305),
        "cell_means": {
            "baseline_a": interval([r["baseline_a"] for r in pair_rows], SEED + 310),
            "baseline_an": interval([r["baseline_an"] for r in pair_rows], SEED + 311),
            "patched_a": interval([r["patched_a"] for r in pair_rows], SEED + 312),
            "patched_an": interval([r["patched_an"] for r in pair_rows], SEED + 313),
            "random_patched_a": interval(
                [r["baseline_a"] + r["random_delta_delta_a_mean"] for r in pair_rows],
                SEED + 314,
            ),
            "random_patched_an": interval(
                [r["baseline_an"] + r["random_delta_delta_an_mean"] for r in pair_rows],
                SEED + 315,
            ),
        },
        "lexical_gap_a": {
            "baseline_target_minus_source": interval(
                [r["baseline_a"] for r in pair_rows], SEED + 320
            ),
            "patched_target_minus_source": interval(
                [r["patched_a"] for r in pair_rows], SEED + 321
            ),
            "delta_delta": interval([r["delta_delta_a"] for r in pair_rows], SEED + 322),
            "fraction_of_gap_closed": interval(
                [r["gap_closed_a"] for r in pair_rows if r["gap_closed_a"] is not None],
                SEED + 323,
            ),
            "mean_target_rank_before": interval(
                [float(r["baseline_target_rank_a"]) for r in pair_rows], SEED + 324
            ),
            "mean_target_rank_after": interval(
                [float(r["patched_target_rank_a"]) for r in pair_rows], SEED + 325
            ),
            "target_top10_before": rate_interval(
                [r["target_topk10_before"] for r in pair_rows], SEED + 326
            ),
            "target_top10_after": rate_interval(
                [r["target_topk10_after"] for r in pair_rows], SEED + 327
            ),
            "target_top1_after": 0.0,
            "target_logit_change": interval(
                [r["target_logit_change_a"] for r in pair_rows], SEED + 328
            ),
            "source_logit_change": interval(
                [r["source_logit_change_a"] for r in pair_rows], SEED + 329
            ),
            "target_probability_change": interval(
                [r["target_prob_change_a"] for r in pair_rows], SEED + 330
            ),
            "source_probability_change": interval(
                [r["source_prob_change_a"] for r in pair_rows], SEED + 331
            ),
        },
        "pairs": pair_rows,
    }
    (RESULTS / "derived_controls.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
