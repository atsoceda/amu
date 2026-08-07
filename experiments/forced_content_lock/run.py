#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.lib.aan_protocol import (
    activation_at,
    article_and_word,
    choose_control_features,
    summarize_condition,
    vowel_initial,
    write_json,
)
from experiments.lib.core import (
    dict_intervention_result,
    generate_with_interventions,
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


def load_set_features(config: dict[str, Any], set_id: str) -> list[dict[str, Any]]:
    selection = json.loads((EXP_DIR / config["e1_selection_path"]).resolve().read_text())
    feats = selection["sets"][set_id]["selected_features"]
    if not feats:
        raise RuntimeError(f"Empty feature set {set_id}")
    return feats


def pick_article_set(config: dict[str, Any]) -> str:
    summary_path = (EXP_DIR / config["e1_summary_path"]).resolve()
    preferred = str(config.get("article_set_id", "S1_dual_effect"))
    if not summary_path.exists():
        return preferred
    e1 = json.loads(summary_path.read_text())
    best_id = preferred
    best_score = -1e9
    for set_id, block in e1.get("set_results", {}).items():
        s = block.get("summary", {})
        score = abs(s.get("mean_delta_an_minus_a", 0.0)) + s.get(
            "generated_article_changed_rate", 0.0
        )
        if score > best_score:
            best_score = score
            best_id = set_id
    # Prefer S1 if it has any article movement; else best mover
    s1 = e1.get("set_results", {}).get("S1_dual_effect", {}).get("summary", {})
    if abs(s1.get("mean_delta_an_minus_a", 0.0)) >= 0.2 or s1.get(
        "generated_article_changed_rate", 0.0
    ) >= 0.15:
        return "S1_dual_effect"
    return best_id


def build_feature_interventions(
    model,
    prompt: str,
    position: int,
    features: list[dict[str, Any]],
    factor: float,
) -> list[dict[str, Any]]:
    interventions = []
    for feature in features:
        activation = activation_at(
            model,
            prompt,
            int(feature["layer"]),
            position,
            int(feature["feature_idx"]),
        )
        interventions.append(
            {
                "layer": int(feature["layer"]),
                "pos": position,
                "feature_idx": int(feature["feature_idx"]),
                "value": float(activation * factor),
            }
        )
    return interventions


def evaluate_rows(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    condition_name: str,
    article_features: list[dict[str, Any]] | None,
    content_features: list[dict[str, Any]] | None,
    article_factor: float,
    content_factor: float,
) -> list[dict[str, Any]]:
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    target_ids = [a_id, an_id]
    rows = []
    for index, example in enumerate(examples, start=1):
        prompt = f"{config['demonstration']} {example['sentence']}"
        position = len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1
        interventions: list[dict[str, Any]] = []
        if article_features:
            interventions.extend(
                build_feature_interventions(
                    model, prompt, position, article_features, article_factor
                )
            )
        if content_features:
            interventions.extend(
                build_feature_interventions(
                    model, prompt, position, content_features, content_factor
                )
            )
        baseline = logits_for_prompt(
            model, prompt, target_ids, top_k=10, return_activations=False
        )
        intervened = (
            dict_intervention_result(model, prompt, interventions, target_ids, baseline)
            if interventions
            else baseline
        )
        baseline_continuation = generate_with_interventions(
            model, prompt, [], max_new_tokens=int(config["max_new_tokens"])
        )
        intervention_continuation = generate_with_interventions(
            model,
            prompt,
            interventions,
            max_new_tokens=int(config["max_new_tokens"]),
        )
        baseline_gen_article, baseline_word = article_and_word(baseline_continuation)
        intervention_gen_article, intervention_word = article_and_word(
            intervention_continuation
        )
        if interventions:
            delta_a = intervened["targets"][str(a_id)]["delta_logit"]
            delta_an = intervened["targets"][str(an_id)]["delta_logit"]
        else:
            delta_a = 0.0
            delta_an = 0.0
        content_preserved = baseline_word == intervention_word and bool(baseline_word)
        class_shifted = (
            bool(baseline_word)
            and bool(intervention_word)
            and vowel_initial(baseline_word) != vowel_initial(intervention_word)
        )
        article_moved_toward_an = delta_an - delta_a > 0
        illicit_mismatch = False
        if intervention_gen_article == "an" and intervention_word:
            illicit_mismatch = not vowel_initial(intervention_word)
        elif intervention_gen_article == "a" and intervention_word:
            illicit_mismatch = vowel_initial(intervention_word)
        wrapper_like = (
            article_moved_toward_an
            and content_preserved
            and baseline_gen_article == "a"
            and intervention_gen_article == "an"
        )
        trajectory_like = (
            article_moved_toward_an and class_shifted and not content_preserved
        )
        rows.append(
            {
                "index": index,
                "condition": condition_name,
                "target_prompt": example["sentence"],
                "listed_word": example["listed_word"],
                "expected_article": example["expected_article"],
                "twin_word": example.get("twin_word", ""),
                "delta_a": delta_a,
                "delta_an": delta_an,
                "delta_an_minus_a": delta_an - delta_a,
                "baseline_continuation": baseline_continuation,
                "intervention_continuation": intervention_continuation,
                "baseline_generated_article": baseline_gen_article,
                "intervention_generated_article": intervention_gen_article,
                "baseline_generated_word": baseline_word,
                "intervention_generated_word": intervention_word,
                "content_preserved": content_preserved,
                "content_word_changed": baseline_word != intervention_word,
                "class_shifted": class_shifted,
                "article_moved_toward_an": article_moved_toward_an,
                "article_moved_toward_a": delta_a - delta_an > 0,
                "generated_article_changed": (
                    baseline_gen_article != intervention_gen_article
                ),
                "illicit_mismatch": illicit_mismatch,
                "wrapper_like": wrapper_like,
                "trajectory_like": trajectory_like,
                "matched_twin": (
                    bool(example.get("twin_word"))
                    and intervention_word == example["twin_word"].lower()
                ),
            }
        )
        logging.info(
            "%s %d/%d %s delta_an-a=%.3f content_preserved=%s",
            condition_name,
            index,
            len(examples),
            example["sentence"],
            delta_an - delta_a,
            content_preserved,
        )
    return rows


def decide_fork(condition_summaries: dict[str, Any]) -> tuple[str, str]:
    c1 = condition_summaries["C1_article_push"]
    c3 = condition_summaries["C3_dual"]
    c4 = condition_summaries.get("C4_illicit_dual")
    control = condition_summaries["C5_control_article"]
    # Outcome A: dual flips article with content preserve above controls and above C1
    a_hit = (
        c3["wrapper_like_rate"] >= 0.15
        and c3["content_preserved_rate"]
        >= max(c1["content_preserved_rate"], control["content_preserved_rate"]) + 0.05
        and c3["generated_article_changed_rate"] >= 0.15
    )
    if c4 is not None:
        a_hit = a_hit or (
            c4["wrapper_like_rate"] >= 0.15
            and c4["content_preserved_rate"] >= 0.2
            and c4["illicit_mismatch_rate"] < 0.2
        )
    if a_hit:
        return (
            "outcome_A_modular",
            "Dual (or illicit-dual) condition moved the article while locking content above C1 and controls.",
        )
    # Outcome B
    if (
        c1["trajectory_like_rate"] >= 0.15
        or (
            c1["class_shifted_rate"] >= 0.2
            and c1["content_preserved_rate"] < 0.5
        )
    ) and c3["illicit_mismatch_rate"] <= 0.1:
        return (
            "outcome_B_compiled_trajectory",
            "Article-push class-switches; dual re-bundles as a package; illicit mismatch stays low.",
        )
    return (
        "mixed_or_inconclusive",
        "Neither clean modular dissociation nor clear compiled-trajectory package pattern.",
    )


def partial_dissociation(c1: dict[str, Any], c3: dict[str, Any], thr: float) -> bool:
    return (
        (
            c3["content_preserved_rate"] >= thr
            and c3["generated_article_changed_rate"] >= 0.15
        )
        or (
            c3["content_preserved_rate"]
            > c1["content_preserved_rate"] + 0.1
            and c1["generated_article_changed_rate"] >= 0.15
            and c3["generated_article_changed_rate"]
            < c1["generated_article_changed_rate"]
        )
    )


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Forced Content-Lock / Dual Intervention (E3)",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        "",
        f"Article set: `{summary['article_set_id']}`",
        f"Content set: `{summary['content_set_id']}`",
        f"Decision: `{summary['decision']}`",
        "",
        summary["interpretation"],
        "",
        "| Condition | Mean Δ(an−a) | Wrapper-like | Trajectory-like | Content preserved | Class shifted | Illicit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cid, s in summary["condition_summaries"].items():
        lines.append(
            f"| {cid} | {s['mean_delta_an_minus_a']:.3f} | {s['wrapper_like_rate']:.3f} | "
            f"{s['trajectory_like_rate']:.3f} | {s['content_preserved_rate']:.3f} | "
            f"{s['class_shifted_rate']:.3f} | {s['illicit_mismatch_rate']:.3f} |"
        )
    lines += ["", f"C4 mode: `{summary['c4_mode']}`", ""]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    config = load_config()
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    article_set_id = pick_article_set(config)
    content_set_id = str(config.get("content_set_id", "S3_content_only"))
    article_features = load_set_features(config, article_set_id)
    content_features = load_set_features(config, content_set_id)
    factor = float(config["amplify_factor"])

    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    examples = config["test_examples"]
    first_prompt = f"{config['demonstration']} {examples[0]['sentence']}"
    first_pos = len(tokenizer(first_prompt, add_special_tokens=True).input_ids) - 1
    control_features = choose_control_features(
        model, first_prompt, first_pos, article_features, config
    )

    conditions: dict[str, list[dict[str, Any]]] = {}
    conditions["C0_baseline"] = evaluate_rows(
        model, tokenizer, examples, config, "C0_baseline", None, None, 0.0, 0.0
    )
    conditions["C1_article_push"] = evaluate_rows(
        model,
        tokenizer,
        examples,
        config,
        "C1_article_push",
        article_features,
        None,
        factor,
        0.0,
    )
    conditions["C2_content_lock"] = evaluate_rows(
        model,
        tokenizer,
        examples,
        config,
        "C2_content_lock",
        None,
        content_features,
        0.0,
        factor,
    )
    conditions["C3_dual"] = evaluate_rows(
        model,
        tokenizer,
        examples,
        config,
        "C3_dual",
        article_features,
        content_features,
        factor,
        factor,
    )
    conditions["C5_control_article"] = evaluate_rows(
        model,
        tokenizer,
        examples,
        config,
        "C5_control_article",
        control_features,
        None,
        factor,
        0.0,
    )
    conditions["C5_control_dual"] = evaluate_rows(
        model,
        tokenizer,
        examples,
        config,
        "C5_control_dual",
        control_features,
        choose_control_features(
            model, first_prompt, first_pos, content_features, {**config, "control_seed": 1}
        ),
        factor,
        factor,
    )

    summaries = {cid: summarize_condition(rows) for cid, rows in conditions.items()}
    thr = float(config.get("partial_dissociation_content_preserve_min", 0.2))
    do_full_c4 = partial_dissociation(
        summaries["C1_article_push"], summaries["C3_dual"], thr
    ) and bool(config.get("run_full_c4_if_partial_dissociation", True))

    # C4: push article toward an (article features) while locking consonant baselines
    # Use only expected-a examples for illicit attempt when short mode.
    consonant_examples = [
        ex for ex in examples if ex["expected_article"] == "a"
    ]
    if do_full_c4:
        c4_examples = examples
        c4_mode = "full"
    else:
        n = int(config.get("c4_short_prompt_count", 5))
        c4_examples = consonant_examples[:n] or examples[:n]
        c4_mode = "short_confirmation"

    conditions["C4_illicit_dual"] = evaluate_rows(
        model,
        tokenizer,
        c4_examples,
        config,
        "C4_illicit_dual",
        article_features,
        content_features,
        factor,
        factor,
    )
    # Note: C4 uses same dual ops as C3 on consonant-heavy subset; illicit score is in summary.
    summaries["C4_illicit_dual"] = summarize_condition(conditions["C4_illicit_dual"])

    decision, interpretation = decide_fork(summaries)
    payload = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "runtime_seconds": time.time() - started,
        "article_set_id": article_set_id,
        "content_set_id": content_set_id,
        "amplify_factor": factor,
        "c4_mode": c4_mode,
        "partial_dissociation": do_full_c4,
        "decision": decision,
        "interpretation": interpretation,
        "condition_summaries": summaries,
        "conditions": conditions,
        "article_features": article_features,
        "content_features": content_features,
        "control_features": control_features,
    }
    write_json(RESULTS_DIR / "summary.json", payload)
    write_report(payload)
    logging.info("E3 decision: %s", decision)


if __name__ == "__main__":
    main()
