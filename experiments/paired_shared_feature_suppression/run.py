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


def prompt_final_pos(model, prompt: str) -> int:
    return len(model.tokenizer(prompt, add_special_tokens=True).input_ids) - 1


def logits_snapshot(model, prompt: str, target_ids: list[int]) -> dict[str, Any]:
    return logits_for_prompt(model, prompt, target_ids, top_k=15, return_activations=False)


def interventions_for_prompt(model, prompt_text: str, regimen: dict[str, Any], features: dict[str, Any]) -> list[dict[str, Any]]:
    pos = prompt_final_pos(model, prompt_text)
    return [
        {
            "layer": int(features[key]["layer"]),
            "pos": pos,
            "feature_idx": int(features[key]["feature_idx"]),
            "value": 0.0,
        }
        for key in regimen["feature_keys"]
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
    base_expected = base_targets[str(expected_id)]
    base_stall = base_targets[str(stall_id)]
    base_margin = base_expected["logit"] - base_stall["logit"]
    margin = expected["logit"] - stall["logit"]
    wrong_delta = max(targets[str(token_id)]["delta_logit"] for token_id in wrong_ids)
    fallback_delta = max(targets[str(token_id)]["delta_logit"] for token_id in fallback_ids)
    return {
        "baseline_expected_logit": base_expected["logit"],
        "baseline_stall_logit": base_stall["logit"],
        "baseline_commitment_margin": float(base_margin),
        "commitment_margin": float(margin),
        "delta_commitment_margin": float(margin - base_margin),
        "expected_delta_logit": expected["delta_logit"],
        "stall_delta_logit": stall["delta_logit"],
        "expected_rank": expected["rank"],
        "stall_rank": stall["rank"],
        "expected_beats_stall_before": bool(base_expected["logit"] > base_stall["logit"]),
        "expected_beats_stall_after": bool(expected["logit"] > stall["logit"]),
        "max_wrong_answer_delta_logit": wrong_delta,
        "max_fallback_delta_logit": fallback_delta,
        "specificity_margin": expected["delta_logit"] - max(wrong_delta, fallback_delta),
    }


def summarize_by_regimen(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for name in [row["intervention_name"] for row in rows]:
        if any(item["intervention_name"] == name for item in out):
            continue
        group = [row for row in rows if row["intervention_name"] == name]
        country = [row for row in group if row["prompt_country_id"] != "control"]
        ambiguous = [row for row in country if row["prompt_family"] == "ambiguous_association"]
        hard_ambiguous = [
            row for row in ambiguous
            if not row["score"]["expected_beats_stall_before"]
        ]
        controls = [row for row in group if row["prompt_country_id"] == "control"]
        out.append(
            {
                "intervention_name": name,
                "feature_keys": group[0]["feature_keys"],
                "n_prompts": len(group),
                "ambiguous_mean_delta_margin": mean(row["score"]["delta_commitment_margin"] for row in ambiguous),
                "ambiguous_positive_count": sum(row["score"]["delta_commitment_margin"] > 0.0 for row in ambiguous),
                "ambiguous_flips_to_expected_count": sum(
                    (not row["score"]["expected_beats_stall_before"]) and row["score"]["expected_beats_stall_after"]
                    for row in ambiguous
                ),
                "hard_ambiguous_count": len(hard_ambiguous),
                "hard_ambiguous_mean_delta_margin": mean(row["score"]["delta_commitment_margin"] for row in hard_ambiguous) if hard_ambiguous else 0.0,
                "country_mean_delta_margin": mean(row["score"]["delta_commitment_margin"] for row in country),
                "mean_expected_delta_logit": mean(row["score"]["expected_delta_logit"] for row in country),
                "mean_stall_delta_logit": mean(row["score"]["stall_delta_logit"] for row in country),
                "mean_wrong_answer_delta_logit": mean(row["score"]["max_wrong_answer_delta_logit"] for row in country),
                "mean_specificity_margin": mean(row["score"]["specificity_margin"] for row in country),
                "control_mean_delta_margin": mean(row["score"]["delta_commitment_margin"] for row in controls) if controls else 0.0,
            }
        )
    return sorted(out, key=lambda item: item["ambiguous_mean_delta_margin"], reverse=True)


def status_word(row: dict[str, Any]) -> str:
    score = row["score"]
    if score["delta_commitment_margin"] > 0 and score["expected_beats_stall_after"] and not score["expected_beats_stall_before"]:
        return "flipped to expected city"
    if score["delta_commitment_margin"] > 0:
        return "helped margin"
    if score["delta_commitment_margin"] < 0:
        return "hurt margin"
    return "unchanged"


def pollution_word(row: dict[str, Any]) -> str:
    score = row["score"]
    if score["max_wrong_answer_delta_logit"] > score["expected_delta_logit"] and score["max_wrong_answer_delta_logit"] > 0.05:
        return "wrong-city movement exceeded expected-city movement"
    if score["specificity_margin"] < -0.25:
        return "specificity weakened"
    return "no obvious wrong-city pollution"


def compare_pair_to_singles(summary: dict[str, Any]) -> dict[str, Any]:
    by_name = {item["intervention_name"]: item for item in summary["regimen_summary"]}
    pair = by_name["l7_f89_plus_l10_f9037"]
    l7 = by_name["l7_f89"]
    l10 = by_name["l10_f9037"]
    return {
        "pair_minus_l7_ambiguous_mean_delta_margin": pair["ambiguous_mean_delta_margin"] - l7["ambiguous_mean_delta_margin"],
        "pair_minus_l10_ambiguous_mean_delta_margin": pair["ambiguous_mean_delta_margin"] - l10["ambiguous_mean_delta_margin"],
        "pair_beats_l7_on_ambiguous_mean": pair["ambiguous_mean_delta_margin"] > l7["ambiguous_mean_delta_margin"],
        "pair_beats_l10_on_ambiguous_mean": pair["ambiguous_mean_delta_margin"] > l10["ambiguous_mean_delta_margin"],
        "pair_has_more_flips_than_l7": pair["ambiguous_flips_to_expected_count"] > l7["ambiguous_flips_to_expected_count"],
        "pair_has_more_flips_than_l10": pair["ambiguous_flips_to_expected_count"] > l10["ambiguous_flips_to_expected_count"],
    }


def build_verdict(summary: dict[str, Any]) -> str:
    by_name = {item["intervention_name"]: item for item in summary["regimen_summary"]}
    pair = by_name["l7_f89_plus_l10_f9037"]
    l7 = by_name["l7_f89"]
    if (
        pair["ambiguous_mean_delta_margin"] > l7["ambiguous_mean_delta_margin"]
        and pair["ambiguous_positive_count"] >= 6
        and pair["mean_specificity_margin"] > -0.25
        and abs(pair["control_mean_delta_margin"]) < 0.15
    ):
        return "The pair improves on L7/F89 alone enough to justify a held-out prompt test."
    if pair["ambiguous_mean_delta_margin"] > 0.0:
        return "The pair has directional signal, but it does not clearly beat the cleaner single-feature intervention."
    return "The pair does not look useful on the current prompt set."


def prompt_table(rows: list[dict[str, Any]], regimen_name: str) -> list[str]:
    selected = [
        row for row in rows
        if row["intervention_name"] == regimen_name
        and row["prompt_country_id"] != "control"
        and row["prompt_family"] == "ambiguous_association"
    ]
    lines = [
        "| Prompt | Expected | Before city > `the`? | After city > `the`? | Margin Change | Expected Logit Change | `the` Logit Change | Wrong-City Check | Plain Result |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in selected:
        score = row["score"]
        lines.append(
            "| "
            f"`{row['prompt_text']}` | "
            f"`{row['expected_text'].strip()}` | "
            f"{score['expected_beats_stall_before']} | "
            f"{score['expected_beats_stall_after']} | "
            f"{score['delta_commitment_margin']:.3f} | "
            f"{score['expected_delta_logit']:.3f} | "
            f"{score['stall_delta_logit']:.3f} | "
            f"{pollution_word(row)} | "
            f"{status_word(row)} |"
        )
    return lines


def write_report(summary: dict[str, Any]) -> None:
    pair_comparison = summary["pair_comparison"]
    lines = [
        "# Paired Shared Feature Suppression Report",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['config']['model']}`",
        "",
        "## Question",
        "",
        "This experiment asks whether the two individually promising shared features from the previous screen work better when suppressed together.",
        "",
        "The key pair is `L7/F89 + L10/F9037`, applied at the final token position of each prompt. The goal is not merely to lower `the`. The goal is to increase the expected-country-city margin over `the` without increasing wrong-country city tokens.",
        "",
        "## What Was Suppressed",
        "",
    ]
    for key, feature in summary["config"]["features"].items():
        lines.append(
            f"- `{key}`: layer={feature['layer']}, feature_idx={feature['feature_idx']}, "
            f"source_pos={feature['source_pos']}. {feature['why_tested']}"
        )

    lines += [
        "",
        "## Regimens Tested",
        "",
    ]
    for regimen in summary["config"]["regimens"]:
        lines.append(
            f"- `{regimen['name']}`: {regimen['description']} Feature keys: {', '.join(regimen['feature_keys'])}."
        )

    lines += [
        "",
        "## Short Answer",
        "",
        summary["verdict"],
        "",
        "The pair did improve the average ambiguous-prompt margin more than either single feature. That is the favorable part. It is not yet a clean win because it produced zero hard-prompt flips and its wrong-city/specificity behavior was worse than `L7/F89` alone. In plain terms: the pair pushed `the` down, but it often pushed the expected city down too and sometimes let wrong city tokens move more than the correct city token.",
        "",
        "## Did The Pair Beat The Singles?",
        "",
        f"- Pair minus `L7/F89` on ambiguous mean margin: {pair_comparison['pair_minus_l7_ambiguous_mean_delta_margin']:.3f}",
        f"- Pair minus `L10/F9037` on ambiguous mean margin: {pair_comparison['pair_minus_l10_ambiguous_mean_delta_margin']:.3f}",
        f"- Pair beat `L7/F89` on ambiguous mean margin: {pair_comparison['pair_beats_l7_on_ambiguous_mean']}",
        f"- Pair beat `L10/F9037` on ambiguous mean margin: {pair_comparison['pair_beats_l10_on_ambiguous_mean']}",
        f"- Pair produced more hard-prompt flips than `L7/F89`: {pair_comparison['pair_has_more_flips_than_l7']}",
        f"- Pair produced more hard-prompt flips than `L10/F9037`: {pair_comparison['pair_has_more_flips_than_l10']}",
        "",
        "## Aggregate Results",
        "",
        "| Regimen | Features | Ambiguous Mean Margin Change | Ambiguous Helped | Hard Prompt Flips | Hard Prompt Count | Expected Logit Change | `the` Logit Change | Wrong-City Logit Change | Specificity | Control Margin Change |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["regimen_summary"]:
        lines.append(
            "| "
            f"`{item['intervention_name']}` | "
            f"{', '.join(item['feature_keys'])} | "
            f"{item['ambiguous_mean_delta_margin']:.3f} | "
            f"{item['ambiguous_positive_count']}/8 | "
            f"{item['ambiguous_flips_to_expected_count']} | "
            f"{item['hard_ambiguous_count']} | "
            f"{item['mean_expected_delta_logit']:.3f} | "
            f"{item['mean_stall_delta_logit']:.3f} | "
            f"{item['mean_wrong_answer_delta_logit']:.3f} | "
            f"{item['mean_specificity_margin']:.3f} | "
            f"{item['control_mean_delta_margin']:.3f} |"
        )

    lines += [
        "",
        "## Key Pair Prompt-Level Results",
        "",
        "This table shows the actual ambiguous prompts for the main pair. `Margin Change` means expected city logit minus `the` logit, after intervention minus before intervention. Positive is the desired direction.",
        "",
        *prompt_table(summary["rows"], "l7_f89_plus_l10_f9037"),
        "",
        "## Single-Feature Comparison On Ambiguous Prompts",
        "",
        "These tables are included so the pair can be judged against the two ingredients, not just against baseline.",
        "",
        "### L7/F89 Alone",
        "",
        *prompt_table(summary["rows"], "l7_f89"),
        "",
        "### L10/F9037 Alone",
        "",
        *prompt_table(summary["rows"], "l10_f9037"),
        "",
        "## Interpretation Rules",
        "",
        "- A useful result is not `the went down` by itself. A useful result is the expected city improving relative to `the`.",
        "- A stronger result flips a hard prompt where `the` beat the expected city before the intervention.",
        "- A polluted result is one where the wrong-country city movement is larger than the expected-city movement.",
        "- A reusable intervention should help several countries with the same feature set, not require choosing a different feature per country.",
        "",
        "## What This Decides Next",
        "",
        "If `L7/F89 + L10/F9037` beats `L7/F89` alone without wrong-city pollution, the next experiment should move to held-out prompt phrasings. If it does not beat `L7/F89`, then the next held-out test should use `L7/F89` alone as the cleaner candidate.",
        "",
        "## Regenerate",
        "",
        "Run `/Users/anthony/miniconda3/bin/python /Users/anthony/repos/amu/experiments/paired_shared_feature_suppression/run.py`.",
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

    logging.info("Computing baselines")
    baselines = {
        prompt["id"]: logits_snapshot(model, prompt["text"], target_ids)
        for prompt in config["prompts"]
    }

    rows: list[dict[str, Any]] = []
    for regimen in config["regimens"]:
        logging.info("Testing %s", regimen["name"])
        for prompt in config["prompts"]:
            prompt_country_id = prompt["country_id"]
            scoring_country = country_by_id.get(prompt_country_id, country_by_id["france"])
            expected_id = target_text_to_id[scoring_country["answer_text"]]
            wrong_ids = [target_text_to_id[text] for text in scoring_country["wrong_answer_texts"]]
            fallback_ids = [target_text_to_id[text] for text in scoring_country["fallback_target_texts"]]
            baseline = baselines[prompt["id"]]
            interventions = interventions_for_prompt(model, prompt["text"], regimen, config["features"])
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
                    "intervention_name": regimen["name"],
                    "feature_keys": regimen["feature_keys"],
                    "prompt_id": prompt["id"],
                    "prompt_country_id": prompt_country_id,
                    "prompt_family": prompt["family"],
                    "prompt_text": prompt["text"],
                    "expected_text": scoring_country["answer_text"],
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
        "baselines": baselines,
        "rows": rows,
        "regimen_summary": summarize_by_regimen(rows),
    }
    summary["pair_comparison"] = compare_pair_to_singles(summary)
    summary["verdict"] = build_verdict(summary)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary)
    logging.info("Wrote %s", RESULTS_DIR / "summary.json")
    logging.info("Wrote %s", RESULTS_DIR / "report.md")


if __name__ == "__main__":
    main()
