#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.core import (
    dict_intervention_result,
    graph_features_for_target,
    load_replacement_model,
    logits_for_prompt,
    run_graph as shared_run_graph,
    serialize_feature,
    setup_file_logging,
    token_id_for_text,
)

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"


def setup_logging() -> None:
    setup_file_logging(RESULTS_DIR)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def logits_snapshot(model, prompt: str, target_ids: list[int]) -> dict[str, Any]:
    return logits_for_prompt(model, prompt, target_ids, top_k=15, return_activations=False)


def intervention_snapshot(
    model,
    prompt: str,
    interventions: list[dict[str, Any]],
    target_ids: list[int],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    return dict_intervention_result(
        model,
        prompt,
        interventions,
        target_ids,
        baseline,
        filter_to_prompt_length=True,
    )


def positive_features(features: list[Any], limit: int) -> list[Any]:
    selected = [feature for feature in features if feature.direct_effect > 0.0 and feature.activation > 0.0]
    return sorted(selected, key=lambda feature: feature.direct_effect, reverse=True)[:limit]


def intervention_item(feature, value: float) -> dict[str, Any]:
    return {
        "layer": int(feature.layer),
        "pos": int(feature.pos),
        "feature_idx": int(feature.feature_idx),
        "value": float(value),
    }


def build_interventions(
    paris_features: list[Any],
    stall_features: list[Any],
    group_sizes: list[int],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for size in group_sizes:
        paris_group = paris_features[:size]
        stall_group = stall_features[:size]
        if stall_group:
            specs.append(
                {
                    "name": f"stall_positive_top{size}_zero",
                    "kind": "stall_suppression",
                    "features": [serialize_feature(feature) for feature in stall_group],
                    "interventions": [intervention_item(feature, 0.0) for feature in stall_group],
                }
            )
        if paris_group:
            specs.append(
                {
                    "name": f"paris_positive_top{size}_double",
                    "kind": "paris_amplification",
                    "features": [serialize_feature(feature) for feature in paris_group],
                    "interventions": [
                        intervention_item(feature, float(feature.activation * 2.0))
                        for feature in paris_group
                    ],
                }
            )
        if paris_group and stall_group:
            specs.append(
                {
                    "name": f"commitment_combo_top{size}",
                    "kind": "stall_suppression_plus_paris_amplification",
                    "features": [serialize_feature(feature) for feature in stall_group + paris_group],
                    "interventions": [
                        *[intervention_item(feature, 0.0) for feature in stall_group],
                        *[
                            intervention_item(feature, float(feature.activation * 2.0))
                            for feature in paris_group
                        ],
                    ],
                }
            )
    return specs


def commitment_score(
    result: dict[str, Any],
    baseline: dict[str, Any],
    paris_id: int,
    stall_id: int,
    competitor_ids: list[int],
    fallback_ids: list[int],
) -> dict[str, Any]:
    targets = result["targets"]
    base_targets = baseline["targets"]
    paris = targets[str(paris_id)]
    stall = targets[str(stall_id)]
    base_margin = base_targets[str(paris_id)]["logit"] - base_targets[str(stall_id)]["logit"]
    margin = paris["logit"] - stall["logit"]
    competitor_delta = max(targets[str(tid)]["delta_logit"] for tid in competitor_ids)
    fallback_delta = max(targets[str(tid)]["delta_logit"] for tid in fallback_ids)
    return {
        "baseline_commitment_margin": float(base_margin),
        "commitment_margin": float(margin),
        "delta_commitment_margin": float(margin - base_margin),
        "paris_delta_logit": paris["delta_logit"],
        "stall_delta_logit": stall["delta_logit"],
        "paris_rank": paris["rank"],
        "stall_rank": stall["rank"],
        "paris_beats_stall": bool(paris["logit"] > stall["logit"]),
        "max_competitor_delta_logit": competitor_delta,
        "max_fallback_delta_logit": fallback_delta,
        "specificity_margin": paris["delta_logit"] - max(competitor_delta, fallback_delta),
    }


def summarize_by_family(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in sorted({row["prompt_family"] for row in rows}):
        family_rows = [row for row in rows if row["prompt_family"] == family]
        out[family] = {
            "n": len(family_rows),
            "mean_delta_commitment_margin": mean(
                row["score"]["delta_commitment_margin"] for row in family_rows
            ),
            "positive_commitment_count": sum(
                row["score"]["delta_commitment_margin"] > 0.0 for row in family_rows
            ),
            "paris_beats_stall_count": sum(row["score"]["paris_beats_stall"] for row in family_rows),
            "mean_specificity_margin": mean(row["score"]["specificity_margin"] for row in family_rows),
        }
    return out


def build_verdict(summary: dict[str, Any]) -> str:
    prompt_id = summary["config"]["intervention_prompt_id"]
    intervention_rows = [row for row in summary["rows"] if row["prompt_id"] == prompt_id]
    best = max(intervention_rows, key=lambda row: row["score"]["delta_commitment_margin"])
    best_score = best["score"]
    controls = [
        item for family, item in summary["family_summary"].items()
        if family in {"syntax_control", "country_competitor"}
    ]
    control_leak = any(item["mean_delta_commitment_margin"] > 0.25 for item in controls)

    if best_score["paris_beats_stall"] and best_score["delta_commitment_margin"] > 0.5 and not control_leak:
        return "The sprint found a candidate causal shift from stall/bridge continuation toward direct Paris commitment, with limited average control leakage."
    if best_score["delta_commitment_margin"] > 0.5:
        return "The sprint found a real commitment-margin handle, but validation suggests it may partly affect broad answer format rather than only France/Paris commitment."
    if best_score["delta_commitment_margin"] > 0.0:
        return "The sprint found directional movement toward Paris over the stall token, but the effect is modest and needs sharper feature selection."
    return "The tested interventions did not improve the Paris-over-stall commitment margin on the primary prompt."


def write_report(summary: dict[str, Any]) -> None:
    paris_id = str(summary["token_ids"]["semantic_target"])
    stall_id = str(summary["token_ids"]["stall_target"])
    primary_prompt_id = summary["config"]["intervention_prompt_id"]
    primary_rows = [row for row in summary["rows"] if row["prompt_id"] == primary_prompt_id]
    ranked = sorted(primary_rows, key=lambda row: row["score"]["delta_commitment_margin"], reverse=True)

    lines = [
        "# Stall Commitment Sprint Report",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['config']['model']}`",
        f"Primary prompt: `{summary['intervention_prompt']['text']}`",
        "",
        "## Verdict",
        "",
        summary["verdict"],
        "",
        "## Primary Baseline",
        "",
    ]
    base = summary["baselines"][primary_prompt_id]
    for item in base["top_tokens"][:10]:
        lines.append(
            f"- rank {item['rank']}: `{item['token']}` prob={item['prob']:.6f} logit={item['logit']:.3f}"
        )
    base_margin = base["targets"][paris_id]["logit"] - base["targets"][stall_id]["logit"]
    lines += [
        "",
        f"Baseline commitment margin `Paris - the`: {base_margin:.3f}",
        "",
        "## Best Primary Interventions",
        "",
    ]
    for row in ranked[:12]:
        score = row["score"]
        lines.append(
            f"- `{row['intervention_name']}` ({row['intervention_kind']}): "
            f"delta_margin={score['delta_commitment_margin']:.3f}; "
            f"Paris delta_logit={score['paris_delta_logit']:.3f}; "
            f"the delta_logit={score['stall_delta_logit']:.3f}; "
            f"Paris rank={score['paris_rank']}; the rank={score['stall_rank']}; "
            f"Paris beats the={score['paris_beats_stall']}"
        )

    best = ranked[0]
    best_score = best["score"]
    if best_score["delta_commitment_margin"] > 0.0:
        if abs(best_score["stall_delta_logit"]) > abs(best_score["paris_delta_logit"]):
            mechanism = "mostly by suppressing the stall token rather than amplifying `Paris`."
        else:
            mechanism = "mostly by changing the `Paris` logit rather than suppressing the stall token."
        lines += [
            "",
            "## Mechanism Note",
            "",
            f"The best primary intervention, `{best['intervention_name']}`, flips the margin {mechanism}",
        ]

    lines += ["", "## Family Summary", ""]
    for family, item in summary["family_summary"].items():
        lines.append(
            f"- `{family}` n={item['n']}; "
            f"mean delta_margin={item['mean_delta_commitment_margin']:.3f}; "
            f"positive={item['positive_commitment_count']}/{item['n']}; "
            f"Paris beats stall={item['paris_beats_stall_count']}/{item['n']}; "
            f"mean specificity_margin={item['mean_specificity_margin']:.3f}"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "This sprint treats the high `the` logit as a potential stall/bridge continuation rather than a lack of Paris knowledge. A publishable positive result would show that suppressing stall-supporting features or amplifying Paris-supporting features increases the `Paris - the` margin on France association prompts without doing the same on syntax and country-competitor controls.",
        "",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    setup_logging()
    started = time.time()
    config = load_config()
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading model")
    model = load_replacement_model(config)
    tokenizer = model.tokenizer

    target_text_to_id = {}
    target_ids = []
    for text in config["target_token_texts"]:
        token_id = token_id_for_text(tokenizer, text)
        target_text_to_id[text] = token_id
        if token_id not in target_ids:
            target_ids.append(token_id)

    paris_id = target_text_to_id[config["semantic_target_text"]]
    stall_id = target_text_to_id[config["stall_target_text"]]
    competitor_ids = [target_text_to_id[text] for text in config["competitor_target_texts"]]
    fallback_ids = [target_text_to_id[text] for text in config["fallback_target_texts"]]

    logging.info("Computing baselines")
    baselines = {
        prompt["id"]: logits_snapshot(model, prompt["text"], target_ids)
        for prompt in config["prompts"]
    }

    intervention_prompt = next(
        prompt for prompt in config["prompts"] if prompt["id"] == config["intervention_prompt_id"]
    )
    logging.info("Attributing primary prompt")
    graph = shared_run_graph(
        model,
        intervention_prompt["text"],
        intervention_prompt["id"],
        target_ids,
        config,
        GRAPHS_DIR,
    )

    paris_features = positive_features(
        graph_features_for_target(
            graph,
            tokenizer,
            intervention_prompt["id"],
            paris_id,
            int(config["top_features_per_target"]),
        ),
        int(config["positive_features_per_set"]),
    )
    stall_features = positive_features(
        graph_features_for_target(
            graph,
            tokenizer,
            intervention_prompt["id"],
            stall_id,
            int(config["top_features_per_target"]),
        ),
        int(config["positive_features_per_set"]),
    )
    specs = build_interventions(paris_features, stall_features, [int(size) for size in config["group_sizes"]])

    rows = []
    for spec in specs:
        logging.info("Testing intervention %s", spec["name"])
        for prompt in config["prompts"]:
            baseline = baselines[prompt["id"]]
            result = intervention_snapshot(model, prompt["text"], spec["interventions"], target_ids, baseline)
            rows.append(
                {
                    "intervention_name": spec["name"],
                    "intervention_kind": spec["kind"],
                    "prompt_id": prompt["id"],
                    "prompt_family": prompt["family"],
                    "prompt_text": prompt["text"],
                    "score": commitment_score(
                        result,
                        baseline,
                        paris_id,
                        stall_id,
                        competitor_ids,
                        fallback_ids,
                    ),
                    "result": result,
                }
            )

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": time.time() - started,
        "config": config,
        "target_text_to_id": target_text_to_id,
        "token_ids": {
            "semantic_target": paris_id,
            "stall_target": stall_id,
            "competitor_ids": competitor_ids,
            "fallback_ids": fallback_ids,
        },
        "intervention_prompt": intervention_prompt,
        "baselines": baselines,
        "top_features": {
            "paris_positive": [serialize_feature(feature) for feature in paris_features],
            "stall_positive": [serialize_feature(feature) for feature in stall_features],
        },
        "intervention_specs": specs,
        "rows": rows,
        "family_summary": summarize_by_family(rows),
    }
    summary["verdict"] = build_verdict(summary)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary)
    logging.info("Wrote %s", RESULTS_DIR / "summary.json")
    logging.info("Wrote %s", RESULTS_DIR / "report.md")


if __name__ == "__main__":
    main()
