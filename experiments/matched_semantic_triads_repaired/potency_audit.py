#!/usr/bin/env python3
"""Pairwise, route-outcome-blind potency audit for the repaired semantic triads."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch

from experiments.gemma_1b_residual_scale.run import ResidualModel, first_id
from experiments.lib.aan_protocol import token_id_for_text

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"


def load(path: Path):
    return json.loads(path.read_text())


def interval(values, seed=20260903, resamples=10000):
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return {
        "n": n,
        "mean": sum(values) / n,
        "lo": draws[math.floor(.025 * (resamples - 1))],
        "hi": draws[math.ceil(.975 * (resamples - 1))],
        "positive_n": sum(value > 0 for value in values),
        "method": "paired semantic-triad bootstrap",
        "resamples": resamples,
    }


def main():
    cfg = load(EXP / "config.json")
    screen = load(RESULTS / "screen_rows.json")
    rows = load(RESULTS / "rows.json")
    admissible = {row["triad_id"] for row in screen if row["admissible"]}
    screen_by_id = {row["triad_id"]: row for row in screen}
    primary = {
        (row["triad_id"], row["arm"]): row
        for row in rows
        if row["triad_id"] in admissible and row["strength"] == cfg["primary_strength"]
    }

    rm = ResidualModel(cfg["model_snapshot"], getattr(torch, cfg["dtype"]))
    tok = rm.tokenizer
    article_ids = {article: token_id_for_text(tok, f" {article}") for article in ("a", "an")}
    output_rows = []
    for triad in cfg["triads"]:
        if triad["id"] not in admissible:
            continue
        donor_states = {}
        for role in ("source", "within", "cross"):
            donor = cfg["donor_template"].format(word=triad[f"{role}_word"], definition=triad["definition"])
            position = len(tok(donor, add_special_tokens=True).input_ids) - 1
            donor_states[role] = rm.states(donor, position)[cfg["fixed_layer"]]
        neutral = cfg["neutral_template"].format(definition=triad["definition"])
        lexical_ids = {role: first_id(tok, triad[f"{role}_word"]) for role in ("source", "within", "cross")}
        arm_values = {}
        for arm in ("within", "cross"):
            delta = donor_states[arm] - donor_states["source"]
            target_article = triad[f"{arm}_article"]
            logits = rm.logits(neutral + tok.decode([article_ids[target_article]])).float()
            baseline_gap = float(logits[lexical_ids[arm]] - logits[lexical_ids["source"]])
            assay = primary[(triad["id"], arm)]
            stochastic = assay["stochastic"][str(cfg["primary_temperature"])]
            arm_values[arm] = {
                "direction_l2": float(torch.linalg.vector_norm(delta.float())),
                "baseline_target_minus_source_gap": baseline_gap,
                "fixed_target_article_efficacy": assay["fixed_target_article_effect"],
                "intervention_on_article_support": screen_by_id[triad["id"]]["arms"][arm]["on_support"]["article_mass"],
                "total_target_aligned_effect": stochastic["total"]["target_minus_source"],
                "total_tv": stochastic["total"]["tv"],
                "route_contrast": stochastic["public"]["target_minus_source"] - stochastic["private"]["target_minus_source"],
            }
        output_rows.append({
            "triad_id": triad["id"],
            "within": arm_values["within"],
            "cross": arm_values["cross"],
            "cross_minus_within": {
                key: arm_values["cross"][key] - arm_values["within"][key]
                for key in arm_values["within"]
            },
        })

    audit_metrics = (
        "direction_l2",
        "baseline_target_minus_source_gap",
        "fixed_target_article_efficacy",
        "intervention_on_article_support",
        "total_target_aligned_effect",
        "total_tv",
    )
    summaries = {}
    for metric in audit_metrics:
        within = [row["within"][metric] for row in output_rows]
        cross = [row["cross"][metric] for row in output_rows]
        differences = [row["cross_minus_within"][metric] for row in output_rows]
        summaries[metric] = {
            "within": interval(within),
            "cross": interval(cross),
            "paired_cross_minus_within": interval(differences),
        }
    result = {
        "audit": "matched-triad intervention potency and normalization",
        "model": cfg["model"],
        "fixed_layer": cfg["fixed_layer"],
        "primary_strength": cfg["primary_strength"],
        "primary_temperature": cfg["primary_temperature"],
        "route_outcomes_used_for_calibration": False,
        "n_triads": len(output_rows),
        "rows": output_rows,
        "summaries": summaries,
    }
    (RESULTS / "potency_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
