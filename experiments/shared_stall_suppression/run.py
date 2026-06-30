#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.core import (
    dict_intervention_result,
    load_replacement_model,
    logits_for_prompt,
    setup_file_logging,
    token_id_for_text,
)

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def logits_snapshot(model, prompt: str, target_ids: list[int]) -> dict[str, Any]:
    return logits_for_prompt(model, prompt, target_ids, top_k=15, return_activations=False)


def prompt_final_pos(model, prompt: str) -> int:
    return len(model.tokenizer(prompt, add_special_tokens=True).input_ids) - 1


def feature_coord(feature: dict[str, Any]) -> tuple[int, int, int]:
    return (int(feature["layer"]), int(feature["pos"]), int(feature["feature_idx"]))


def layer_feature_coord(feature: dict[str, Any]) -> tuple[int, int]:
    return (int(feature["layer"]), int(feature["feature_idx"]))


def load_shared_features(config: dict[str, Any]) -> list[dict[str, Any]]:
    source_path = (EXP_DIR / config["source_summary_path"]).resolve()
    source = json.loads(source_path.read_text())
    group_name = config["feature_source_group"]
    country_sets: dict[str, set[tuple[int, int, int]]] = {}
    by_coord: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)

    for country_id, groups in source["top_features"].items():
        features = groups[group_name]
        coords = {feature_coord(feature) for feature in features}
        country_sets[country_id] = coords
        for feature in features:
            by_coord[feature_coord(feature)].append(feature)

    shared_coords = set.intersection(*country_sets.values())
    shared_features: list[dict[str, Any]] = []
    for coord in sorted(shared_coords):
        instances = by_coord[coord]
        shared_features.append(
            {
                "layer": coord[0],
                "source_pos": coord[1],
                "feature_idx": coord[2],
                "source_prompt_ids": sorted({item["source_prompt_id"] for item in instances}),
                "mean_activation": mean(float(item["activation"]) for item in instances),
                "mean_direct_effect": mean(float(item["direct_effect"]) for item in instances),
                "country_count": len(instances),
            }
        )
    return sorted(shared_features, key=lambda item: item["mean_direct_effect"], reverse=True)


def build_intervention_specs(
    shared_features: list[dict[str, Any]],
    group_sizes: list[int],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for size in group_sizes:
        group = shared_features[:size]
        suffix = "" if size == len(shared_features) else f"_top{size}"
        specs.extend(
            [
                {
                    "name": f"shared_overlap{suffix}_fixed_pos",
                    "kind": "shared_fixed_absolute_position_suppression",
                    "description": "Suppresses shared features at the absolute token position where they were discovered.",
                    "features": group,
                },
                {
                    "name": f"shared_overlap{suffix}_final_token",
                    "kind": "shared_final_token_suppression",
                    "description": "Suppresses the same layer/feature coordinates at each prompt's final token position.",
                    "features": group,
                },
            ]
        )
    for feature in shared_features:
        specs.append(
            {
                "name": (
                    "shared_single_final_token_"
                    f"l{feature['layer']}_f{feature['feature_idx']}"
                ),
                "kind": "shared_single_final_token_suppression",
                "description": "Suppresses one shared layer/feature coordinate at each prompt's final token position.",
                "features": [feature],
            }
        )
    return specs


def interventions_for_prompt(
    model,
    prompt_text: str,
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    if spec["kind"] in {"shared_final_token_suppression", "shared_single_final_token_suppression"}:
        pos = prompt_final_pos(model, prompt_text)
        return [
            {
                "layer": int(feature["layer"]),
                "pos": pos,
                "feature_idx": int(feature["feature_idx"]),
                "value": 0.0,
            }
            for feature in spec["features"]
        ]
    return [
        {
            "layer": int(feature["layer"]),
            "pos": int(feature["source_pos"]),
            "feature_idx": int(feature["feature_idx"]),
            "value": 0.0,
        }
        for feature in spec["features"]
    ]


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
    wrong_delta = max(targets[str(token_id)]["delta_logit"] for token_id in wrong_ids)
    fallback_delta = max(targets[str(token_id)]["delta_logit"] for token_id in fallback_ids)
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


def summarize_by_spec(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for name in sorted({row["intervention_name"] for row in rows}):
        group = [row for row in rows if row["intervention_name"] == name]
        applied = [row for row in group if row["applied_intervention_count"] > 0]
        country = [
            row for row in applied
            if row["prompt_country_id"] != "control"
            and row["prompt_family"] in {"ambiguous_association", "near_match", "direct_capital"}
        ]
        ambiguous = [row for row in country if row["prompt_family"] == "ambiguous_association"]
        controls = [row for row in applied if row["prompt_country_id"] == "control"]
        scored_country = country or [row for row in group if row["prompt_country_id"] != "control"]
        scored_ambiguous = ambiguous or [
            row for row in group
            if row["prompt_country_id"] != "control"
            and row["prompt_family"] == "ambiguous_association"
        ]
        out.append(
            {
                "intervention_name": name,
                "intervention_kind": group[0]["intervention_kind"],
                "n_rows": len(group),
                "n_applied": len(applied),
                "country_mean_delta_margin": mean(
                    row["score"]["delta_commitment_margin"] for row in scored_country
                ),
                "ambiguous_mean_delta_margin": mean(
                    row["score"]["delta_commitment_margin"] for row in scored_ambiguous
                ),
                "ambiguous_positive_count": sum(
                    row["score"]["delta_commitment_margin"] > 0.0 for row in scored_ambiguous
                ),
                "ambiguous_expected_beats_stall_count": sum(
                    row["score"]["expected_beats_stall"] for row in scored_ambiguous
                ),
                "mean_expected_delta_logit": mean(row["score"]["expected_delta_logit"] for row in scored_country),
                "mean_stall_delta_logit": mean(row["score"]["stall_delta_logit"] for row in scored_country),
                "mean_wrong_answer_delta_logit": mean(
                    row["score"]["max_wrong_answer_delta_logit"] for row in scored_country
                ),
                "mean_specificity_margin": mean(row["score"]["specificity_margin"] for row in scored_country),
                "control_mean_delta_margin": mean(
                    row["score"]["delta_commitment_margin"] for row in controls
                ) if controls else 0.0,
            }
        )
    return sorted(out, key=lambda item: item["ambiguous_mean_delta_margin"], reverse=True)


def summarize_by_prompt(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "intervention_name": row["intervention_name"],
            "prompt_id": row["prompt_id"],
            "prompt_country_id": row["prompt_country_id"],
            "prompt_family": row["prompt_family"],
            "applied_intervention_count": row["applied_intervention_count"],
            "intervention_count": row["intervention_count"],
            "delta_commitment_margin": row["score"]["delta_commitment_margin"],
            "expected_delta_logit": row["score"]["expected_delta_logit"],
            "stall_delta_logit": row["score"]["stall_delta_logit"],
            "wrong_answer_delta_logit": row["score"]["max_wrong_answer_delta_logit"],
            "specificity_margin": row["score"]["specificity_margin"],
            "expected_beats_stall": row["score"]["expected_beats_stall"],
        }
        for row in rows
    ]


def build_verdict(summary: dict[str, Any]) -> str:
    best = summary["spec_summary"][0]
    if (
        best["ambiguous_mean_delta_margin"] > 0.5
        and best["ambiguous_positive_count"] >= 6
        and best["mean_specificity_margin"] > -0.5
        and best["control_mean_delta_margin"] < 0.5
    ):
        return "The shared suppression regimen is a plausible reusable anti-stall candidate, but it still needs held-out task-family validation."
    if best["ambiguous_mean_delta_margin"] > 0.0:
        return "The shared suppression regimen has directional evidence but is not yet strong enough as a reusable anti-stall tool."
    return "The shared suppression regimen did not improve commitment reliably in this screen."


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Shared Stall Suppression Report",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['config']['model']}`",
        "",
        "## Question",
        "",
        "This experiment asks whether the overlapping `the`-supporting features from the country screen can be used as one reusable suppression regimen, rather than discovering a separate intervention for each country.",
        "",
        "## Shared Features",
        "",
    ]
    for feature in summary["shared_features"]:
        lines.append(
            f"- layer={feature['layer']}; source_pos={feature['source_pos']}; "
            f"feature_idx={feature['feature_idx']}; "
            f"mean_direct_effect_on_the={feature['mean_direct_effect']:.3f}; "
            f"mean_activation={feature['mean_activation']:.3f}; "
            f"source_prompts={', '.join(feature['source_prompt_ids'])}"
        )

    lines += [
        "",
        "## Regimens Tested",
        "",
        "- `shared_overlap_fixed_pos`: suppresses those six features at the original absolute token position.",
        "- `shared_overlap_final_token`: suppresses those same layer/feature coordinates at the final token position for each prompt.",
        "",
        "The final-token version is the more handoff-relevant test because a user will not always use prompts with the same token length.",
        "",
        "## Verdict",
        "",
        summary["verdict"],
        "",
        "## What Happened",
        "",
        "The full six-feature shared suppression was not a reusable anti-stall intervention. It usually reduced `the`, but it also reduced the expected city token enough that the expected-city-over-`the` margin got worse on average.",
        "",
        "The subset screen was more informative. Two individual shared features showed weak directional promise when suppressed at the final token: `shared_single_final_token_l7_f89` and `shared_single_final_token_l10_f9037`. These did not create obvious average control movement, but the effect sizes are small and not yet publishable as a handoff tool.",
        "",
        "The strongest practical lesson is that overlap alone is not enough. Shared `the`-supporting features can be real and still be too entangled with useful prompt processing to suppress as a bundle.",
        "",
        "## Aggregate Results",
        "",
    ]
    for item in summary["spec_summary"]:
        lines.append(
            f"- `{item['intervention_name']}`: "
            f"ambiguous_mean_delta_margin={item['ambiguous_mean_delta_margin']:.3f}; "
            f"ambiguous_positive={item['ambiguous_positive_count']}; "
            f"ambiguous_expected_beats_stall={item['ambiguous_expected_beats_stall_count']}; "
            f"country_mean_delta_margin={item['country_mean_delta_margin']:.3f}; "
            f"mean_expected_delta={item['mean_expected_delta_logit']:.3f}; "
            f"mean_stall_delta={item['mean_stall_delta_logit']:.3f}; "
            f"mean_wrong_delta={item['mean_wrong_answer_delta_logit']:.3f}; "
            f"mean_specificity={item['mean_specificity_margin']:.3f}; "
            f"control_mean_delta={item['control_mean_delta_margin']:.3f}; "
            f"applied={item['n_applied']}/{item['n_rows']}"
        )

    lines += ["", "## Prompt-Level Results", ""]
    for row in summary["prompt_summary"]:
        lines.append(
            f"- `{row['intervention_name']}` on `{row['prompt_id']}` "
            f"({row['prompt_country_id']}/{row['prompt_family']}): "
            f"applied={row['applied_intervention_count']}/{row['intervention_count']}; "
            f"delta_margin={row['delta_commitment_margin']:.3f}; "
            f"expected_delta={row['expected_delta_logit']:.3f}; "
            f"stall_delta={row['stall_delta_logit']:.3f}; "
            f"wrong_delta={row['wrong_answer_delta_logit']:.3f}; "
            f"specificity={row['specificity_margin']:.3f}; "
            f"expected_beats_stall={row['expected_beats_stall']}"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "A positive result here would mean the same feature set moves several country prompts toward their own expected answer without injecting one particular city. A negative or mixed result would mean the overlap is real but not sufficient: the shared features may be general phrase-continuation machinery rather than a clean stalling circuit.",
        "",
        "The key readout is not whether `the` decreases in isolation. The key readout is whether expected-answer margin improves while wrong-answer movement and syntax-control movement stay low.",
        "",
        "## Regenerate",
        "",
        "Run `/Users/anthony/miniconda3/bin/python /Users/anthony/repos/amu/experiments/shared_stall_suppression/run.py`.",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    config = load_config()

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
    stall_id = target_text_to_id[config["stall_target_text"]]
    shared_features = load_shared_features(config)
    specs = build_intervention_specs(
        shared_features,
        [int(size) for size in config["shared_group_sizes"]],
    )

    logging.info("Computing baselines")
    baselines = {
        prompt["id"]: logits_snapshot(model, prompt["text"], target_ids)
        for prompt in config["prompts"]
    }

    rows: list[dict[str, Any]] = []
    for spec in specs:
        logging.info("Testing %s", spec["name"])
        for prompt in config["prompts"]:
            prompt_country_id = prompt["country_id"]
            scoring_country = country_by_id.get(prompt_country_id, country_by_id["france"])
            expected_id = target_text_to_id[scoring_country["answer_text"]]
            wrong_ids = [target_text_to_id[text] for text in scoring_country["wrong_answer_texts"]]
            fallback_ids = [target_text_to_id[text] for text in scoring_country["fallback_target_texts"]]
            baseline = baselines[prompt["id"]]
            interventions = interventions_for_prompt(model, prompt["text"], spec)
            result = dict_intervention_result(
                model,
                prompt["text"],
                interventions,
                target_ids,
                baseline,
                filter_to_prompt_length=True,
            )
            rows.append(
                {
                    "intervention_name": spec["name"],
                    "intervention_kind": spec["kind"],
                    "prompt_id": prompt["id"],
                    "prompt_country_id": prompt_country_id,
                    "prompt_family": prompt["family"],
                    "prompt_text": prompt["text"],
                    "prompt_final_pos": prompt_final_pos(model, prompt["text"]),
                    "scoring_country_id": scoring_country["id"],
                    "intervention_count": result.get("intervention_count", len(interventions)),
                    "applied_intervention_count": result.get(
                        "applied_intervention_count",
                        len(interventions),
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
        "shared_features": shared_features,
        "intervention_specs": specs,
        "baselines": baselines,
        "rows": rows,
        "spec_summary": summarize_by_spec(rows),
        "prompt_summary": summarize_by_prompt(rows),
    }
    summary["verdict"] = build_verdict(summary)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary)
    logging.info("Wrote %s", RESULTS_DIR / "summary.json")
    logging.info("Wrote %s", RESULTS_DIR / "report.md")


if __name__ == "__main__":
    main()
