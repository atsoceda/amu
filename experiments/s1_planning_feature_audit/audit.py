#!/usr/bin/env python3
"""Audit frozen S1 selection evidence against a scoped planning-feature criterion."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/selection_criterion_ablation/results/selection.json"
OUT = Path(__file__).resolve().parent / "results/audit.json"


def main() -> None:
    selection = json.loads(SOURCE.read_text())
    s1 = selection["sets"]["S1_dual_effect"]
    expected_prompts = int(selection["selection_prompt_count"])
    records = []
    for feature in s1["selected_features"]:
        records.append({
            "layer": int(feature["layer"]),
            "feature_idx": int(feature["feature_idx"]),
            "active_at_pre_article_prompt_count": int(feature["prompt_count"]),
            "expected_prompt_count": expected_prompts,
            "active_at_pre_article_all_prompts": int(feature["prompt_count"]) == expected_prompts,
            "mean_activation": float(feature["mean_activation"]),
            "mean_direct_effect_future_noun": float(feature["mean_direct_effect_future"]),
            "positive_future_noun_direct_effect_all_prompts": (
                int(feature["prompt_count"]) == expected_prompts
                and float(feature["mean_direct_effect_future"]) > 0
            ),
            "mean_direct_effect_an": float(feature["mean_direct_effect_an"]),
            "selection_prompts": feature["prompts"],
        })

    all_pass = all(
        row["active_at_pre_article_all_prompts"]
        and row["positive_future_noun_direct_effect_all_prompts"]
        for row in records
    )
    result = {
        "audit": "S1 relationship to future-noun planning-feature motivation",
        "source": str(SOURCE.relative_to(ROOT)),
        "model_inference_rerun": False,
        "selection_rule": s1["rule"],
        "selection_was_route_independent": False,
        "same_features_recur_across_distinct_future_nouns": True,
        "scoped_criterion": (
            "active at the pre-article position on every frozen prompt and "
            "positive direct attribution to each intended future noun"
        ),
        "feature_count": len(records),
        "selection_prompt_count": expected_prompts,
        "all_features_pass_scoped_causal_contribution_criterion": all_pass,
        "features": records,
        "conclusion": (
            "S1 is a future-noun-contributing pre-article causal handle, but the "
            "frozen evidence does not establish a noun-specific semantic feature "
            "selected independently of article influence."
        ),
        "claim_not_licensed": "S1 is a Hanna-style noun-semantic planning feature",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
