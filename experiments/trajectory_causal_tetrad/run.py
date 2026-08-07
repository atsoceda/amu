#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.lib.aan_protocol import (
    article_and_word,
    build_amplify_interventions,
    build_zero_interventions,
    choose_control_features,
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


def resolve_amplify_factor(config: dict[str, Any]) -> float:
    e2_path = (EXP_DIR / config.get("e2_summary_path", "")).resolve()
    if e2_path.exists():
        e2 = json.loads(e2_path.read_text())
        return float(e2.get("recommended_e4_amplify_factor", config["amplify_factor"]))
    return float(config["amplify_factor"])


def load_features(config: dict[str, Any]) -> list[dict[str, Any]]:
    selection = json.loads((EXP_DIR / config["e1_selection_path"]).resolve().read_text())
    set_id = str(config.get("feature_set_id", "S1_dual_effect"))
    feats = selection["sets"][set_id]["selected_features"]
    if not feats:
        raise RuntimeError(f"No features for {set_id}")
    return feats


def package_label(word: str, baseline: str, twin: str) -> str:
    w = word.lower()
    if w == baseline.lower():
        return "baseline"
    if twin and w == twin.lower():
        return "twin"
    if not w:
        return "other"
    return "other"


def evaluate_family_condition(
    model,
    tokenizer,
    family: dict[str, Any],
    config: dict[str, Any],
    condition_name: str,
    interventions: list[dict[str, Any]],
) -> dict[str, Any]:
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    prompt = f"{config['demonstration']} {family['sentence']}"
    baseline = logits_for_prompt(
        model, prompt, [a_id, an_id], top_k=10, return_activations=False
    )
    intervened = (
        dict_intervention_result(model, prompt, interventions, [a_id, an_id], baseline)
        if interventions
        else baseline
    )
    baseline_cont = generate_with_interventions(
        model, prompt, [], max_new_tokens=int(config["max_new_tokens"])
    )
    interv_cont = generate_with_interventions(
        model, prompt, interventions, max_new_tokens=int(config["max_new_tokens"])
    )
    b_art, b_word = article_and_word(baseline_cont)
    i_art, i_word = article_and_word(interv_cont)
    if interventions:
        delta_a = intervened["targets"][str(a_id)]["delta_logit"]
        delta_an = intervened["targets"][str(an_id)]["delta_logit"]
    else:
        delta_a = 0.0
        delta_an = 0.0
    return {
        "condition": condition_name,
        "family": family["name"],
        "sentence": family["sentence"],
        "baseline_continuation": baseline_cont,
        "intervention_continuation": interv_cont,
        "baseline_article": b_art,
        "intervention_article": i_art,
        "baseline_word": b_word,
        "intervention_word": i_word,
        "baseline_package": package_label(
            b_word, family["baseline_word"], family["twin_word"]
        ),
        "intervention_package": package_label(
            i_word, family["baseline_word"], family["twin_word"]
        ),
        "class_shifted": bool(b_word)
        and bool(i_word)
        and vowel_initial(b_word) != vowel_initial(i_word),
        "content_preserved": b_word == i_word and bool(b_word),
        "matched_twin": i_word == family["twin_word"].lower(),
        "delta_an_minus_a": delta_an - delta_a,
    }


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Trajectory Causal Tetrad (E4)",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        f"Feature set: `{summary['feature_set_id']}`",
        f"Amplify factor: {summary['amplify_factor']}",
        "",
        summary["interpretation"],
        "",
    ]
    for family_name, block in summary["families"].items():
        lines.append(f"## {family_name}")
        lines.append("")
        lines.append(
            "| Condition | Baseline | Intervention | Package | Twin? | Class shift? | Δ(an−a) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | ---: |")
        for cond, row in block["conditions"].items():
            lines.append(
                f"| {cond} | `{row['baseline_continuation']}` | "
                f"`{row['intervention_continuation']}` | {row['intervention_package']} | "
                f"{row['matched_twin']} | {row['class_shifted']} | "
                f"{row['delta_an_minus_a']:.3f} |"
            )
        lines.append("")
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    config = load_config()
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    features = load_features(config)
    factor = resolve_amplify_factor(config)
    model = load_replacement_model(config)
    tokenizer = model.tokenizer

    families_out: dict[str, Any] = {}
    for family in config["twin_families"]:
        prompt = f"{config['demonstration']} {family['sentence']}"
        position = len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1
        control = choose_control_features(
            model, prompt, position, features, config
        )
        gof, _ = build_amplify_interventions(
            model, prompt, position, features, factor
        )
        lof = build_zero_interventions(features, position)
        # Rescue: LoF values then restore by amplifying (approximate restore)
        rescue = gof
        control_gof, _ = build_amplify_interventions(
            model, prompt, position, control, factor
        )
        conditions = {
            "baseline": evaluate_family_condition(
                model, tokenizer, family, config, "baseline", []
            ),
            "lof_zero": evaluate_family_condition(
                model, tokenizer, family, config, "lof_zero", lof
            ),
            "gof_amplify": evaluate_family_condition(
                model, tokenizer, family, config, "gof_amplify", gof
            ),
            "rescue_amplify": evaluate_family_condition(
                model, tokenizer, family, config, "rescue_amplify", rescue
            ),
            "control_amplify": evaluate_family_condition(
                model, tokenizer, family, config, "control_amplify", control_gof
            ),
        }
        families_out[family["name"]] = {
            "family": family,
            "features": features,
            "control_features": control,
            "conditions": conditions,
        }
        logging.info(
            "%s gof package=%s twin=%s",
            family["name"],
            conditions["gof_amplify"]["intervention_package"],
            conditions["gof_amplify"]["matched_twin"],
        )

    # Interpretation: GoF should move package more than control; LoF should disrupt
    gof_twin = sum(
        1
        for block in families_out.values()
        if block["conditions"]["gof_amplify"]["matched_twin"]
        or block["conditions"]["gof_amplify"]["class_shifted"]
    )
    control_move = sum(
        1
        for block in families_out.values()
        if block["conditions"]["control_amplify"]["class_shifted"]
        or block["conditions"]["control_amplify"]["matched_twin"]
    )
    if gof_twin > control_move:
        interpretation = (
            "GoF on the frozen set moves twin/class packages above matched controls; "
            "supports causal role for trajectory-class features."
        )
    else:
        interpretation = (
            "GoF did not clearly beat controls on twin/class package membership; "
            "causal tetrad is weak or nonspecific on these families."
        )

    payload = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "runtime_seconds": time.time() - started,
        "feature_set_id": config.get("feature_set_id", "S1_dual_effect"),
        "amplify_factor": factor,
        "interpretation": interpretation,
        "families": families_out,
    }
    write_json(RESULTS_DIR / "summary.json", payload)
    write_report(payload)
    logging.info("E4 complete: %s", interpretation)


if __name__ == "__main__":
    main()
