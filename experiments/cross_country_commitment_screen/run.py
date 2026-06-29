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
    run_graph,
    serialize_feature,
    setup_file_logging,
    token_id_for_text,
)

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def logits_snapshot(model, prompt: str, target_ids: list[int]) -> dict[str, Any]:
    return logits_for_prompt(model, prompt, target_ids, top_k=15, return_activations=False)


def positive_features(features: list[Any], limit: int) -> list[Any]:
    selected = [
        feature for feature in features
        if feature.direct_effect > 0.0 and feature.activation > 0.0
    ]
    return sorted(selected, key=lambda feature: feature.direct_effect, reverse=True)[:limit]


def intervention_item(feature, value: float) -> dict[str, Any]:
    return {
        "layer": int(feature.layer),
        "pos": int(feature.pos),
        "feature_idx": int(feature.feature_idx),
        "value": float(value),
    }


def build_country_interventions(
    country: dict[str, Any],
    answer_features: list[Any],
    stall_features: list[Any],
    group_sizes: list[int],
    include_single_feature_screen: bool,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    country_id = country["id"]
    if include_single_feature_screen:
        for rank, feature in enumerate(stall_features, start=1):
            specs.append(
                {
                    "name": (
                        f"{country_id}_single_stall_r{rank}_"
                        f"l{feature.layer}_p{feature.pos}_f{feature.feature_idx}_zero"
                    ),
                    "source_country_id": country_id,
                    "kind": "single_stall_suppression",
                    "features": [serialize_feature(feature)],
                    "interventions": [intervention_item(feature, 0.0)],
                }
            )
        for rank, feature in enumerate(answer_features, start=1):
            specs.append(
                {
                    "name": (
                        f"{country_id}_single_answer_r{rank}_"
                        f"l{feature.layer}_p{feature.pos}_f{feature.feature_idx}_double"
                    ),
                    "source_country_id": country_id,
                    "kind": "single_answer_amplification",
                    "features": [serialize_feature(feature)],
                    "interventions": [intervention_item(feature, float(feature.activation * 2.0))],
                }
            )
    for size in group_sizes:
        answer_group = answer_features[:size]
        stall_group = stall_features[:size]
        if stall_group:
            specs.append(
                {
                    "name": f"{country_id}_stall_positive_top{size}_zero",
                    "source_country_id": country_id,
                    "kind": "stall_suppression",
                    "features": [serialize_feature(feature) for feature in stall_group],
                    "interventions": [intervention_item(feature, 0.0) for feature in stall_group],
                }
            )
        if answer_group:
            specs.append(
                {
                    "name": f"{country_id}_answer_positive_top{size}_double",
                    "source_country_id": country_id,
                    "kind": "answer_amplification",
                    "features": [serialize_feature(feature) for feature in answer_group],
                    "interventions": [
                        intervention_item(feature, float(feature.activation * 2.0))
                        for feature in answer_group
                    ],
                }
            )
        if answer_group and stall_group:
            specs.append(
                {
                    "name": f"{country_id}_commitment_combo_top{size}",
                    "source_country_id": country_id,
                    "kind": "stall_suppression_plus_answer_amplification",
                    "features": [serialize_feature(feature) for feature in stall_group + answer_group],
                    "interventions": [
                        *[intervention_item(feature, 0.0) for feature in stall_group],
                        *[
                            intervention_item(feature, float(feature.activation * 2.0))
                            for feature in answer_group
                        ],
                    ],
                }
            )
    return specs


def summarize_interventions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    country_ids = {
        row["prompt_country_id"] for row in rows
        if row["prompt_country_id"] != "control"
    }
    for name in sorted({row["intervention_name"] for row in rows}):
        group = [row for row in rows if row["intervention_name"] == name]
        applied = [row for row in group if row["applied_intervention_count"] > 0]
        if not applied:
            continue
        source_country_id = group[0]["source_country_id"]
        same = [
            row for row in applied
            if row["prompt_country_id"] == source_country_id
            and row["prompt_family"] == "ambiguous_association"
        ]
        cross = [
            row for row in applied
            if row["prompt_country_id"] in country_ids
            and row["prompt_country_id"] != source_country_id
        ]
        controls = [row for row in applied if row["prompt_country_id"] == "control"]
        if not same:
            continue
        same_delta = mean(row["score"]["delta_commitment_margin"] for row in same)
        same_specificity = mean(row["score"]["specificity_margin"] for row in same)
        cross_wrong = mean(row["score"]["max_wrong_answer_delta_logit"] for row in cross) if cross else 0.0
        control_delta = mean(row["score"]["delta_commitment_margin"] for row in controls) if controls else 0.0
        specificity_score = same_delta + same_specificity - max(0.0, cross_wrong) - max(0.0, control_delta)
        out.append(
            {
                "intervention_name": name,
                "intervention_kind": group[0]["intervention_kind"],
                "source_country_id": source_country_id,
                "n_rows": len(group),
                "n_applied": len(applied),
                "same_country_ambiguous_delta": same_delta,
                "same_country_ambiguous_specificity": same_specificity,
                "cross_country_wrong_delta": cross_wrong,
                "control_delta": control_delta,
                "specificity_score": specificity_score,
            }
        )
    return sorted(out, key=lambda item: item["specificity_score"], reverse=True)


def score_prompt(
    result: dict[str, Any],
    baseline: dict[str, Any],
    expected_id: int,
    stall_id: int,
    wrong_ids: list[int],
    fallback_ids: list[int],
) -> dict[str, Any]:
    targets = result["targets"]
    base_targets = baseline["targets"]
    expected = targets[str(expected_id)]
    stall = targets[str(stall_id)]
    base_margin = base_targets[str(expected_id)]["logit"] - base_targets[str(stall_id)]["logit"]
    margin = expected["logit"] - stall["logit"]
    wrong_delta = max(targets[str(tid)]["delta_logit"] for tid in wrong_ids)
    fallback_delta = max(targets[str(tid)]["delta_logit"] for tid in fallback_ids)
    return {
        "baseline_commitment_margin": float(base_margin),
        "commitment_margin": float(margin),
        "delta_commitment_margin": float(margin - base_margin),
        "expected_delta_logit": expected["delta_logit"],
        "stall_delta_logit": stall["delta_logit"],
        "expected_rank": expected["rank"],
        "stall_rank": stall["rank"],
        "expected_beats_stall": bool(expected["logit"] > stall["logit"]),
        "max_wrong_answer_delta_logit": wrong_delta,
        "max_fallback_delta_logit": fallback_delta,
        "specificity_margin": expected["delta_logit"] - max(wrong_delta, fallback_delta),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = sorted({(row["source_country_id"], row["prompt_country_id"], row["prompt_family"]) for row in rows})
    for source_country_id, prompt_country_id, family in keys:
        group = [
            row for row in rows
            if row["source_country_id"] == source_country_id
            and row["prompt_country_id"] == prompt_country_id
            and row["prompt_family"] == family
        ]
        applied_group = [row for row in group if row["applied_intervention_count"] > 0]
        scored_group = applied_group or group
        key = f"{source_country_id}__{prompt_country_id}__{family}"
        out[key] = {
            "source_country_id": source_country_id,
            "prompt_country_id": prompt_country_id,
            "prompt_family": family,
            "n": len(group),
            "n_applied": len(applied_group),
            "n_scored": len(scored_group),
            "mean_delta_commitment_margin": mean(
                row["score"]["delta_commitment_margin"] for row in scored_group
            ),
            "positive_commitment_count": sum(
                row["score"]["delta_commitment_margin"] > 0.0 for row in scored_group
            ),
            "expected_beats_stall_count": sum(
                row["score"]["expected_beats_stall"] for row in scored_group
            ),
            "mean_expected_delta_logit": mean(row["score"]["expected_delta_logit"] for row in scored_group),
            "mean_wrong_answer_delta_logit": mean(
                row["score"]["max_wrong_answer_delta_logit"] for row in scored_group
            ),
            "mean_specificity_margin": mean(row["score"]["specificity_margin"] for row in scored_group),
        }
    return out


def build_verdict(summary: dict[str, Any]) -> str:
    rows = summary["rows"]
    applied_rows = [row for row in rows if row["applied_intervention_count"] > 0]
    country_ids = {country["id"] for country in summary["config"]["countries"]}
    same_country = [
        row for row in applied_rows
        if row["source_country_id"] == row["prompt_country_id"]
        and row["prompt_family"] in {"ambiguous_association", "near_match"}
    ]
    cross_country = [
        row for row in applied_rows
        if row["prompt_country_id"] in country_ids
        and row["source_country_id"] != row["prompt_country_id"]
    ]
    controls = [row for row in applied_rows if row["prompt_country_id"] == "control"]

    same_delta = mean(row["score"]["delta_commitment_margin"] for row in same_country)
    cross_wrong = mean(row["score"]["max_wrong_answer_delta_logit"] for row in cross_country)
    control_delta = mean(row["score"]["delta_commitment_margin"] for row in controls)
    same_specific = mean(row["score"]["specificity_margin"] for row in same_country)

    if same_delta > 0.5 and same_specific > 0.0 and cross_wrong < 0.15 and control_delta < 0.25:
        return "The screen found a candidate commitment/stall pattern: same-country shifts are positive and comparatively specific, with limited wrong-city and control leakage."
    if same_delta > 0.5 and cross_wrong >= 0.15:
        return "The screen found strong same-country movement, but cross-country wrong-answer induction remains a serious confound."
    if same_delta > 0.5 and control_delta >= 0.25:
        return "The screen found strong same-country movement, but syntax or unrelated-country controls suggest a broad perturbation."
    if same_delta > 0.0:
        return "The screen found directional same-country movement, but not enough specificity for a publishable causal claim."
    return "The tested cross-country interventions did not produce reliable same-country commitment movement."


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Cross-Country Commitment Screen Report",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['config']['model']}`",
        "",
        "## Verdict",
        "",
        summary["verdict"],
        "",
        "## Baseline Snapshots",
        "",
    ]
    for prompt in summary["config"]["prompts"]:
        base = summary["baselines"][prompt["id"]]
        top = ", ".join(f"`{item['token']}`" for item in base["top_tokens"][:5])
        lines.append(f"- `{prompt['id']}` ({prompt['family']}): top5 {top}")

    lines += ["", "## Best Same-Country Effects", ""]
    same_rows = [
        row for row in summary["rows"]
        if row["source_country_id"] == row["prompt_country_id"]
    ]
    ranked = sorted(same_rows, key=lambda row: row["score"]["delta_commitment_margin"], reverse=True)
    for row in ranked[:16]:
        score = row["score"]
        lines.append(
            f"- `{row['intervention_name']}` on `{row['prompt_id']}`: "
            f"applied={row['applied_intervention_count']}/{row['intervention_count']}; "
            f"delta_margin={score['delta_commitment_margin']:.3f}; "
            f"expected_delta={score['expected_delta_logit']:.3f}; "
            f"stall_delta={score['stall_delta_logit']:.3f}; "
            f"wrong_delta={score['max_wrong_answer_delta_logit']:.3f}; "
            f"specificity={score['specificity_margin']:.3f}; "
            f"expected_beats_stall={score['expected_beats_stall']}"
        )

    lines += ["", "## Leakage Screen", ""]
    leakage_rows = [
        row for row in summary["rows"]
        if row["source_country_id"] != row["prompt_country_id"]
        or row["prompt_country_id"] == "control"
    ]
    leakage_ranked = sorted(
        leakage_rows,
        key=lambda row: (
            row["score"]["max_wrong_answer_delta_logit"],
            row["score"]["delta_commitment_margin"],
        ),
        reverse=True,
    )
    for row in leakage_ranked[:16]:
        score = row["score"]
        lines.append(
            f"- `{row['intervention_name']}` on `{row['prompt_id']}` "
            f"({row['prompt_country_id']}/{row['prompt_family']}): "
            f"applied={row['applied_intervention_count']}/{row['intervention_count']}; "
            f"delta_margin={score['delta_commitment_margin']:.3f}; "
            f"wrong_delta={score['max_wrong_answer_delta_logit']:.3f}; "
            f"specificity={score['specificity_margin']:.3f}"
        )

    lines += ["", "## Aggregate Screen", ""]
    for key, item in summary["screen_summary"].items():
        lines.append(
            f"- `{key}` n={item['n']}; "
            f"applied_rows={item['n_applied']}; "
            f"mean delta_margin={item['mean_delta_commitment_margin']:.3f}; "
            f"positive={item['positive_commitment_count']}/{item['n_scored']}; "
            f"expected beats stall={item['expected_beats_stall_count']}/{item['n_scored']}; "
            f"mean wrong_delta={item['mean_wrong_answer_delta_logit']:.3f}; "
            f"mean specificity={item['mean_specificity_margin']:.3f}"
        )

    lines += ["", "## Individual Feature Candidates", ""]
    individual = [
        item for item in summary["intervention_summary"]
        if item["intervention_kind"].startswith("single_")
    ]
    if individual:
        for item in individual[:16]:
            lines.append(
                f"- `{item['intervention_name']}` ({item['intervention_kind']}): "
                f"score={item['specificity_score']:.3f}; "
                f"same_delta={item['same_country_ambiguous_delta']:.3f}; "
                f"same_specificity={item['same_country_ambiguous_specificity']:.3f}; "
                f"cross_wrong={item['cross_country_wrong_delta']:.3f}; "
                f"control_delta={item['control_delta']:.3f}; "
                f"applied={item['n_applied']}/{item['n_rows']}"
            )
    else:
        lines.append("No individual feature candidates were evaluated.")

    lines += [
        "",
        "## Interpretation",
        "",
        "The high bar is not merely moving one city token or suppressing `the`. A meaningful result should show analogous same-country movement across country-answer prompts while failing the obvious confounds: wrong-city induction and syntax-control movement.",
        "",
        "Rows with `applied=0/...` are positional non-applications, not evidence that the representation is absent. This screen therefore treats them as a transfer-design limitation and bases the verdict on applied rows.",
        "",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    config = load_config()
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading model")
    model = load_replacement_model(config)
    tokenizer = model.tokenizer

    target_text_to_id: dict[str, int] = {}
    target_ids: list[int] = []
    for text in config["target_token_texts"]:
        token_id = token_id_for_text(tokenizer, text)
        target_text_to_id[text] = token_id
        if token_id not in target_ids:
            target_ids.append(token_id)

    country_by_id = {country["id"]: country for country in config["countries"]}
    prompt_by_id = {prompt["id"]: prompt for prompt in config["prompts"]}
    stall_id = target_text_to_id[config["stall_target_text"]]

    logging.info("Computing baselines")
    baselines = {
        prompt["id"]: logits_snapshot(model, prompt["text"], target_ids)
        for prompt in config["prompts"]
    }

    top_features: dict[str, Any] = {}
    specs: list[dict[str, Any]] = []
    for country in config["countries"]:
        prompt = prompt_by_id[country["discovery_prompt_id"]]
        answer_id = target_text_to_id[country["answer_text"]]
        logging.info("Attributing discovery prompt %s", prompt["id"])
        graph = run_graph(model, prompt["text"], prompt["id"], target_ids, config, GRAPHS_DIR)
        answer_features = positive_features(
            graph_features_for_target(
                graph,
                tokenizer,
                prompt["id"],
                answer_id,
                int(config["top_features_per_target"]),
            ),
            int(config["positive_features_per_set"]),
        )
        stall_features = positive_features(
            graph_features_for_target(
                graph,
                tokenizer,
                prompt["id"],
                stall_id,
                int(config["top_features_per_target"]),
            ),
            int(config["positive_features_per_set"]),
        )
        top_features[country["id"]] = {
            "answer_positive": [serialize_feature(feature) for feature in answer_features],
            "stall_positive": [serialize_feature(feature) for feature in stall_features],
        }
        specs.extend(
            build_country_interventions(
                country,
                answer_features,
                stall_features,
                [int(size) for size in config["group_sizes"]],
                bool(config.get("include_single_feature_screen", False)),
            )
        )

    rows: list[dict[str, Any]] = []
    for spec in specs:
        logging.info("Testing intervention %s", spec["name"])
        for prompt in config["prompts"]:
            prompt_country_id = prompt["country_id"]
            scoring_country = country_by_id.get(prompt_country_id, country_by_id[spec["source_country_id"]])
            expected_id = target_text_to_id[scoring_country["answer_text"]]
            wrong_ids = [target_text_to_id[text] for text in scoring_country["wrong_answer_texts"]]
            fallback_ids = [target_text_to_id[text] for text in scoring_country["fallback_target_texts"]]
            baseline = baselines[prompt["id"]]
            result = dict_intervention_result(
                model,
                prompt["text"],
                spec["interventions"],
                target_ids,
                baseline,
                filter_to_prompt_length=True,
            )
            rows.append(
                {
                    "intervention_name": spec["name"],
                    "intervention_kind": spec["kind"],
                    "source_country_id": spec["source_country_id"],
                    "prompt_id": prompt["id"],
                    "prompt_country_id": prompt_country_id,
                    "prompt_family": prompt["family"],
                    "prompt_text": prompt["text"],
                    "scoring_country_id": scoring_country["id"],
                    "intervention_count": result.get("intervention_count", len(spec["interventions"])),
                    "applied_intervention_count": result.get(
                        "applied_intervention_count",
                        len(spec["interventions"]),
                    ),
                    "score": score_prompt(result, baseline, expected_id, stall_id, wrong_ids, fallback_ids),
                    "result": result,
                }
            )

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": time.time() - started,
        "config": config,
        "target_text_to_id": target_text_to_id,
        "token_ids": {"stall_target": stall_id},
        "baselines": baselines,
        "top_features": top_features,
        "intervention_specs": specs,
        "rows": rows,
        "screen_summary": summarize_rows(rows),
        "intervention_summary": summarize_interventions(rows),
    }
    summary["verdict"] = build_verdict(summary)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary)
    logging.info("Wrote %s", RESULTS_DIR / "summary.json")
    logging.info("Wrote %s", RESULTS_DIR / "report.md")


if __name__ == "__main__":
    main()
