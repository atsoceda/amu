"""Shared a/an occupation-completion protocol helpers for paper experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

from experiments.lib.core import (
    dict_intervention_result,
    generate_with_interventions,
    logits_for_prompt,
    token_id_for_text,
)


def verify_dataset(
    exp_dir: Path, config: dict[str, Any]
) -> dict[str, dict[str, str]]:
    path = (exp_dir / config["dataset_path"]).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != config["dataset_sha256"]:
        raise ValueError(f"Dataset checksum mismatch: {digest}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_sentence = {row["sentence"]: row for row in rows}
    for sentence in config["selection_sentences"]:
        if sentence not in by_sentence:
            raise ValueError(f"Missing selection sentence: {sentence}")
    for example in config["test_examples"]:
        if example["sentence"] in config["selection_sentences"]:
            raise ValueError(
                f"Test sentence overlaps selection set: {example['sentence']}"
            )
    return by_sentence


def slugify(sentence: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", sentence.lower()).strip("_")[:80]


def article_label(token_id: int, a_id: int, an_id: int) -> str:
    if token_id == a_id:
        return "a"
    if token_id == an_id:
        return "an"
    return "other"


def article_and_word(continuation: str) -> tuple[str, str]:
    match = re.match(r"^\s*(a|an)\s+([A-Za-z][A-Za-z-]*)", continuation)
    if not match:
        return "other", ""
    return match.group(1), match.group(2).lower()


def vowel_initial(word: str) -> bool:
    return bool(word) and word[0] in "aeiou"


def first_content_token_text(tokenizer, word: str) -> str:
    ids = tokenizer(f" {word}", add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"No tokens for word {word!r}")
    return tokenizer.decode([ids[0]])


def load_tokenizer(config: dict[str, Any]):
    from transformers import AutoTokenizer

    model_ref = (
        config["model_snapshot"]
        if Path(config["model_snapshot"]).exists()
        else config["model"]
    )
    return AutoTokenizer.from_pretrained(model_ref)


def activation_at(
    model,
    prompt: str,
    layer: int,
    pos: int,
    feature_idx: int,
) -> float:
    _, activations = model.feature_intervention(
        prompt,
        interventions=[],
        freeze_attention=False,
        sparse=False,
        return_activations=True,
    )
    if activations is None:
        return 0.0
    value = activations[layer, pos, feature_idx]
    return float(value.detach().float().cpu())


def build_amplify_interventions(
    model,
    prompt: str,
    position: int,
    features: list[dict[str, Any]],
    amplify_factor: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    interventions = []
    activation_rows = []
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
                "value": float(activation * amplify_factor),
            }
        )
        activation_rows.append(
            {
                "layer": int(feature["layer"]),
                "feature_idx": int(feature["feature_idx"]),
                "activation": activation,
                "value": float(activation * amplify_factor),
            }
        )
    return interventions, activation_rows


def build_zero_interventions(
    features: list[dict[str, Any]],
    position: int,
) -> list[dict[str, Any]]:
    return [
        {
            "layer": int(feature["layer"]),
            "pos": position,
            "feature_idx": int(feature["feature_idx"]),
            "value": 0.0,
        }
        for feature in features
    ]


def choose_control_features(
    model,
    prompt: str,
    position: int,
    content_features: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    _, activations = model.feature_intervention(
        prompt,
        interventions=[],
        freeze_attention=False,
        sparse=False,
        return_activations=True,
    )
    forbidden = {
        (int(item["layer"]), int(item["feature_idx"])) for item in content_features
    }
    layers = sorted({int(item["layer"]) for item in content_features}) or [0]
    target_activation = sum(
        float(item.get("mean_activation", item.get("activation", 0.0)))
        for item in content_features
    ) / max(len(content_features), 1)
    candidates = []
    for layer in layers:
        layer_acts = activations[layer, position].detach().float().cpu()
        nonzero = (layer_acts > 0).nonzero(as_tuple=False).view(-1).tolist()
        for feature_idx in nonzero:
            key = (layer, int(feature_idx))
            if key in forbidden:
                continue
            activation = float(layer_acts[int(feature_idx)])
            candidates.append(
                {
                    "layer": layer,
                    "feature_idx": int(feature_idx),
                    "activation": activation,
                    "activation_distance": abs(activation - target_activation),
                    "mean_activation": activation,
                    "label": f"`L{layer}/F{feature_idx}`",
                }
            )
    rng = random.Random(int(config["control_seed"]))
    candidates.sort(key=lambda item: (item["activation_distance"], item["feature_idx"]))
    pool = candidates[: max(20, int(config["control_feature_count"]) * 5)]
    if len(pool) <= int(config["control_feature_count"]):
        chosen = pool
    else:
        chosen = rng.sample(pool, int(config["control_feature_count"]))
    chosen.sort(key=lambda item: (item["layer"], item["feature_idx"]))
    return chosen


def evaluate_amplify_condition(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    features: list[dict[str, Any]],
    config: dict[str, Any],
    condition_name: str,
    amplify_factor: float | None = None,
) -> list[dict[str, Any]]:
    factor = (
        float(config["amplify_factor"])
        if amplify_factor is None
        else float(amplify_factor)
    )
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    target_ids = [a_id, an_id]
    rows = []
    for index, example in enumerate(examples, start=1):
        prompt = f"{config['demonstration']} {example['sentence']}"
        position = len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1
        interventions, activation_rows = build_amplify_interventions(
            model,
            prompt,
            position,
            features,
            factor,
        )
        baseline = logits_for_prompt(
            model, prompt, target_ids, top_k=10, return_activations=False
        )
        intervened = dict_intervention_result(
            model,
            prompt,
            interventions,
            target_ids,
            baseline,
        )
        baseline_continuation = generate_with_interventions(
            model,
            prompt,
            [],
            max_new_tokens=int(config["max_new_tokens"]),
        )
        intervention_continuation = generate_with_interventions(
            model,
            prompt,
            interventions,
            max_new_tokens=int(config["max_new_tokens"]),
        )
        baseline_article_token = article_label(
            baseline["top_tokens"][0]["token_id"], a_id, an_id
        )
        intervention_article_token = article_label(
            intervened["top_tokens"][0]["token_id"], a_id, an_id
        )
        baseline_gen_article, baseline_word = article_and_word(baseline_continuation)
        intervention_gen_article, intervention_word = article_and_word(
            intervention_continuation
        )
        delta_a = intervened["targets"][str(a_id)]["delta_logit"]
        delta_an = intervened["targets"][str(an_id)]["delta_logit"]
        content_preserved = baseline_word == intervention_word and bool(baseline_word)
        class_shifted = (
            bool(baseline_word)
            and bool(intervention_word)
            and vowel_initial(baseline_word) != vowel_initial(intervention_word)
        )
        article_moved_toward_an = delta_an - delta_a > 0
        article_moved_toward_a = delta_a - delta_an > 0
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
            article_moved_toward_an
            and class_shifted
            and not content_preserved
        )
        rows.append(
            {
                "index": index,
                "condition": condition_name,
                "amplify_factor": factor,
                "target_prompt": example["sentence"],
                "listed_word": example["listed_word"],
                "expected_article": example["expected_article"],
                "twin_word": example.get("twin_word", ""),
                "position": position,
                "feature_activations": activation_rows,
                "baseline_top_article": baseline_article_token,
                "intervention_top_article": intervention_article_token,
                "baseline_an_minus_a": (
                    baseline["targets"][str(an_id)]["logit"]
                    - baseline["targets"][str(a_id)]["logit"]
                ),
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
                "baseline_vowel_initial": vowel_initial(baseline_word),
                "intervention_vowel_initial": vowel_initial(intervention_word),
                "article_moved_toward_an": article_moved_toward_an,
                "article_moved_toward_a": article_moved_toward_a,
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
            "%s %d/%d %s delta_an-a=%.3f content_preserved=%s class_shifted=%s",
            condition_name,
            index,
            len(examples),
            example["sentence"],
            delta_an - delta_a,
            content_preserved,
            class_shifted,
        )
    return rows


def summarize_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)

    def rate(key: str) -> float:
        return sum(1 for row in rows if row[key]) / n if n else 0.0

    return {
        "n": n,
        "mean_delta_an_minus_a": (
            sum(row["delta_an_minus_a"] for row in rows) / n if n else 0.0
        ),
        "article_moved_toward_an_rate": rate("article_moved_toward_an"),
        "article_moved_toward_a_rate": rate("article_moved_toward_a"),
        "generated_article_changed_rate": rate("generated_article_changed"),
        "content_preserved_rate": rate("content_preserved"),
        "content_word_changed_rate": rate("content_word_changed"),
        "class_shifted_rate": rate("class_shifted"),
        "wrapper_like_rate": rate("wrapper_like"),
        "trajectory_like_rate": rate("trajectory_like"),
        "matched_twin_rate": rate("matched_twin"),
        "illicit_mismatch_rate": rate("illicit_mismatch"),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
