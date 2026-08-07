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
    path = (EXP_DIR / config["e1_selection_path"]).resolve()
    if not path.exists():
        return []
    selection = json.loads(path.read_text())
    return selection["sets"].get(set_id, {}).get("selected_features", [])


def run_condition(
    model,
    tokenizer,
    config: dict[str, Any],
    name: str,
    interventions: list[dict[str, Any]],
) -> dict[str, Any]:
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    prompt = f"{config['demonstration']} {config['prompt']}"
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
    content_preserved = b_word == i_word and bool(b_word)
    listed = str(config["listed_word"]).lower()
    listed_prefix = listed[: max(4, len(listed) // 2)]
    preserves_ophthalm = bool(i_word) and (
        i_word.startswith("ophthalm") or listed.startswith(i_word[:5])
    )
    wrapper_repair = (
        b_art == "a"
        and i_art == "an"
        and (content_preserved or preserves_ophthalm)
        and not (
            bool(b_word)
            and bool(i_word)
            and vowel_initial(b_word) != vowel_initial(i_word)
            and not preserves_ophthalm
        )
    )
    package_coincidence = (
        i_art == "an"
        and bool(i_word)
        and vowel_initial(i_word)
        and not preserves_ophthalm
    )
    return {
        "condition": name,
        "baseline_continuation": baseline_cont,
        "intervention_continuation": interv_cont,
        "baseline_article": b_art,
        "intervention_article": i_art,
        "baseline_word": b_word,
        "intervention_word": i_word,
        "delta_an_minus_a": delta_an - delta_a,
        "content_preserved": content_preserved,
        "preserves_ophthalm": preserves_ophthalm,
        "wrapper_repair": wrapper_repair,
        "package_coincidence": package_coincidence,
        "class_shifted": bool(b_word)
        and bool(i_word)
        and vowel_initial(b_word) != vowel_initial(i_word),
    }


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Ophthalmologist Reclassify (E6)",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        "",
        f"Classification: `{summary['classification']}`",
        "",
        summary["interpretation"],
        "",
        "| Condition | Baseline | Intervention | Wrapper repair? | Package coincidence? | Δ(an−a) |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for name, row in summary["conditions"].items():
        lines.append(
            f"| {name} | `{row['baseline_continuation']}` | "
            f"`{row['intervention_continuation']}` | {row['wrapper_repair']} | "
            f"{row['package_coincidence']} | {row['delta_an_minus_a']:.3f} |"
        )
    lines.append("")
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    config = load_config()
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    prompt = f"{config['demonstration']} {config['prompt']}"
    position = len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1
    factor = float(config["amplify_factor"])

    lof_pair = list(config.get("lof_pair", []))
    article_feats = load_set_features(
        config, str(config.get("article_set_id", "S1_dual_effect"))
    )
    content_feats = load_set_features(
        config, str(config.get("content_set_id", "S3_content_only"))
    )

    conditions: dict[str, Any] = {}
    conditions["baseline"] = run_condition(model, tokenizer, config, "baseline", [])
    if lof_pair:
        conditions["lof_pair"] = run_condition(
            model,
            tokenizer,
            config,
            "lof_pair",
            build_zero_interventions(lof_pair, position),
        )
    if article_feats:
        gof, _ = build_amplify_interventions(
            model, prompt, position, article_feats, factor
        )
        conditions["gof_s1"] = run_condition(
            model, tokenizer, config, "gof_s1", gof
        )
    if content_feats:
        lock, _ = build_amplify_interventions(
            model, prompt, position, content_feats, factor
        )
        conditions["content_lock"] = run_condition(
            model, tokenizer, config, "content_lock", lock
        )
    if article_feats and content_feats:
        dual_a, _ = build_amplify_interventions(
            model, prompt, position, article_feats, factor
        )
        dual_c, _ = build_amplify_interventions(
            model, prompt, position, content_feats, factor
        )
        conditions["dual"] = run_condition(
            model, tokenizer, config, "dual", dual_a + dual_c
        )

    wrapper_hits = [
        name for name, row in conditions.items() if row.get("wrapper_repair")
    ]
    package_hits = [
        name for name, row in conditions.items() if row.get("package_coincidence")
    ]
    if wrapper_hits and not package_hits:
        classification = "true_wrapper_repair"
        interpretation = (
            f"Conditions {wrapper_hits} look like content-preserving article repair "
            "on the ophthalmologist mismatch."
        )
    elif package_hits and not wrapper_hits:
        classification = "package_coincidence"
        interpretation = (
            f"Conditions {package_hits} flip toward an vowel-initial package without "
            "preserving ophthalmologist — coincidence with compiled trajectories, not a wrapper."
        )
    elif wrapper_hits and package_hits:
        classification = "mixed"
        interpretation = (
            f"Wrapper-like hits {wrapper_hits}; package-like hits {package_hits}. "
            "Source mismatch is ambiguous under the E3 protocol."
        )
    else:
        classification = "no_clear_repair"
        interpretation = (
            "No condition produced a clear wrapper repair or clean package coincidence "
            "on this prompt under the current feature sets."
        )

    payload = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "runtime_seconds": time.time() - started,
        "classification": classification,
        "interpretation": interpretation,
        "conditions": conditions,
        "lof_pair": lof_pair,
        "article_features": article_feats,
        "content_features": content_feats,
    }
    write_json(RESULTS_DIR / "summary.json", payload)
    write_report(payload)
    logging.info("E6 classification: %s", classification)


if __name__ == "__main__":
    main()
