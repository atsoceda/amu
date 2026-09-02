#!/usr/bin/env python3
"""Recover and audit conditional article-policy movement for frozen matched triads."""
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
SEED = 20260911
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


def main():
    cfg = load(EXP / "config.json")
    rows = load(RESULTS / "rows.json")
    admissible = set(load(RESULTS / "screen_summary.json")["admissible_ids"])
    primary_rows = {
        (row["triad_id"], row["arm"]): row
        for row in rows
        if row["triad_id"] in admissible
        and row["strength"] == cfg["primary_strength"]
    }

    rm = ResidualModel(cfg["model_snapshot"], getattr(torch, cfg["dtype"]))
    tok = rm.tokenizer
    article_ids = {
        article: token_id_for_text(tok, f" {article}") for article in ("a", "an")
    }
    tau = cfg["primary_temperature"]
    output_rows = []
    max_public_reconstruction_error = 0.0

    for triad in cfg["triads"]:
        if triad["id"] not in admissible:
            continue
        neutral = cfg["neutral_template"].format(definition=triad["definition"])
        neutral_ids = tok(neutral, add_special_tokens=True).input_ids
        neutral_position = len(neutral_ids) - 1
        neutral_states = rm.states(neutral, neutral_position)
        donor_states = {}
        for role in ("source", "within", "cross"):
            donor = cfg["donor_template"].format(
                word=triad[f"{role}_word"], definition=triad["definition"]
            )
            donor_position = len(tok(donor, add_special_tokens=True).input_ids) - 1
            donor_states[role] = rm.states(donor, donor_position)[cfg["fixed_layer"]]

        off_article = rm.logits(neutral)
        off_branches = {
            article: rm.logits(neutral + tok.decode([article_id]))
            for article, article_id in article_ids.items()
        }
        q0 = torch.softmax(
            off_article[list(article_ids.values())].float() / tau, dim=-1
        )

        arm_values = {}
        for arm in ("within", "cross"):
            delta = donor_states[arm] - donor_states["source"]
            patch = (
                cfg["fixed_layer"],
                neutral_position,
                neutral_states[cfg["fixed_layer"]]
                + cfg["primary_strength"] * delta,
            )
            on_article = rm.logits(neutral, patch)
            q1 = torch.softmax(
                on_article[list(article_ids.values())].float() / tau, dim=-1
            )
            target_article = triad[f"{arm}_article"]
            target_index = 0 if target_article == "a" else 1
            target_policy_movement = float(q1[target_index] - q0[target_index])

            lexical_ids = {
                role: first_id(tok, triad[f"{role}_word"])
                for role in ("source", arm)
            }
            y0 = {
                article: torch.softmax(logits.float(), dim=-1)
                for article, logits in off_branches.items()
            }
            off = q0[0] * y0["a"] + q0[1] * y0["an"]
            public = q1[0] * y0["a"] + q1[1] * y0["an"]
            reconstructed = float(
                (public - off)[lexical_ids[arm]]
                - (public - off)[lexical_ids["source"]]
            )
            frozen = primary_rows[(triad["id"], arm)]["stochastic"][str(tau)][
                "public"
            ]["target_minus_source"]
            error = abs(reconstructed - frozen)
            max_public_reconstruction_error = max(
                max_public_reconstruction_error, error
            )
            arm_values[arm] = {
                "target_article": target_article,
                "q0_target_article": float(q0[target_index]),
                "q1_target_article": float(q1[target_index]),
                "target_article_policy_movement": target_policy_movement,
                "stored_public_target_minus_source": frozen,
                "reconstructed_public_target_minus_source": reconstructed,
                "absolute_reconstruction_error": error,
            }

        output_rows.append(
            {
                "triad_id": triad["id"],
                "within": arm_values["within"],
                "cross": arm_values["cross"],
                "cross_minus_within_policy_movement": (
                    arm_values["cross"]["target_article_policy_movement"]
                    - arm_values["within"]["target_article_policy_movement"]
                ),
            }
        )

    if max_public_reconstruction_error > 1e-6:
        raise RuntimeError(
            "Article-policy rerun does not reconstruct frozen public effects: "
            f"max error={max_public_reconstruction_error:.9g}"
        )

    within = [
        row["within"]["target_article_policy_movement"] for row in output_rows
    ]
    cross = [row["cross"]["target_article_policy_movement"] for row in output_rows]
    paired = [row["cross_minus_within_policy_movement"] for row in output_rows]
    result = {
        "audit": "matched-triad target-article policy movement",
        "model": cfg["model"],
        "fixed_layer": cfg["fixed_layer"],
        "primary_strength": cfg["primary_strength"],
        "primary_temperature": tau,
        "conditional_policy_support": ["a", "an"],
        "n_triads": len(output_rows),
        "max_public_effect_reconstruction_error": max_public_reconstruction_error,
        "rows": output_rows,
        "summaries": {
            "within_target_article_policy_movement": interval(within, SEED + 1),
            "cross_target_article_policy_movement": interval(cross, SEED + 2),
            "paired_cross_minus_within": interval(paired, SEED + 3),
        },
    }
    (RESULTS / "article_policy_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result["summaries"], indent=2))
    print(
        "max_public_effect_reconstruction_error=",
        f"{max_public_reconstruction_error:.9g}",
    )


if __name__ == "__main__":
    main()
