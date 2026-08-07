#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.lib.core import (
    dict_intervention_result,
    feature_effect_map,
    generate_with_interventions,
    load_replacement_model,
    logits_for_prompt,
    run_graph,
    setup_file_logging,
    token_id_for_text,
)

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
SELECTION_PATH = RESULTS_DIR / "selection.json"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-only",
        nargs=2,
        metavar=("PROMPT_ID", "GRAPH_NAME"),
        help="Internal isolated attribution phase: prompt_id article|future",
    )
    return parser.parse_args()


def verify_dataset(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    path = (EXP_DIR / config["dataset_path"]).resolve()
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


def ensure_selection_graphs(config: dict[str, Any], dataset: dict[str, dict[str, str]]) -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    demo = config["demonstration"]
    for sentence in config["selection_sentences"]:
        prompt_id = slugify(sentence)
        article_path = GRAPHS_DIR / f"{prompt_id}__article.pt"
        future_path = GRAPHS_DIR / f"{prompt_id}__future.pt"
        for graph_name, path in (("article", article_path), ("future", future_path)):
            if path.exists():
                logging.info("Reusing existing graph %s", path)
                continue
            logging.info("Starting isolated %s attribution for %s", graph_name, sentence)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--graph-only",
                    prompt_id,
                    graph_name,
                ],
                check=True,
            )
        # Touch metadata sidecar so selection can recover listed words without reloading CSV quirks.
        meta_path = GRAPHS_DIR / f"{prompt_id}__meta.json"
        if not meta_path.exists():
            row = dataset[sentence]
            meta_path.write_text(
                json.dumps(
                    {
                        "sentence": sentence,
                        "listed_word": row["word"].lower(),
                        "expected_article": row["article"],
                        "article_prompt": f"{demo} {sentence}",
                        "future_prompt": f"{demo} {sentence} a",
                    },
                    indent=2,
                )
                + "\n"
            )


def run_graph_phase(prompt_id: str, graph_name: str) -> None:
    config = load_config()
    setup_file_logging(RESULTS_DIR)
    dataset = verify_dataset(config)
    sentence = next(
        item for item in config["selection_sentences"] if slugify(item) == prompt_id
    )
    row = dataset[sentence]
    demo = config["demonstration"]
    article_prompt = f"{demo} {sentence}"
    future_prompt = f"{demo} {sentence} a"
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    if graph_name == "article":
        target_ids = [
            token_id_for_text(tokenizer, " a"),
            token_id_for_text(tokenizer, " an"),
        ]
        prompt = article_prompt
    else:
        content_text = first_content_token_text(tokenizer, row["word"])
        target_ids = [token_id_for_text(tokenizer, content_text)]
        prompt = future_prompt
        (GRAPHS_DIR / f"{prompt_id}__meta.json").write_text(
            json.dumps(
                {
                    "sentence": sentence,
                    "listed_word": row["word"].lower(),
                    "expected_article": row["article"],
                    "article_prompt": article_prompt,
                    "future_prompt": future_prompt,
                    "content_token_text": content_text,
                },
                indent=2,
            )
            + "\n"
        )
    run_graph(
        model,
        prompt,
        f"{prompt_id}__{graph_name}",
        target_ids,
        config,
        GRAPHS_DIR,
    )


def load_tokenizer(config: dict[str, Any]):
    from transformers import AutoTokenizer

    model_ref = (
        config["model_snapshot"]
        if Path(config["model_snapshot"]).exists()
        else config["model"]
    )
    return AutoTokenizer.from_pretrained(model_ref)


def select_content_features(config: dict[str, Any]) -> dict[str, Any]:
    from circuit_tracer.graph import Graph

    tokenizer = load_tokenizer(config)
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    prompt_records = []
    feature_stats: dict[tuple[int, int], dict[str, Any]] = {}

    for sentence in config["selection_sentences"]:
        prompt_id = slugify(sentence)
        meta = json.loads((GRAPHS_DIR / f"{prompt_id}__meta.json").read_text())
        article_graph = Graph.from_pt(str(GRAPHS_DIR / f"{prompt_id}__article.pt"))
        future_graph = Graph.from_pt(str(GRAPHS_DIR / f"{prompt_id}__future.pt"))
        content_text = meta.get("content_token_text") or first_content_token_text(
            tokenizer, meta["listed_word"]
        )
        content_id = token_id_for_text(tokenizer, content_text)
        pre_article_pos = (
            len(tokenizer(meta["article_prompt"], add_special_tokens=True).input_ids)
            - 1
        )
        an_effects = feature_effect_map(article_graph, an_id)
        a_effects = feature_effect_map(article_graph, a_id)
        future_effects = feature_effect_map(future_graph, content_id)
        dual_keys = []
        for key in set(an_effects) & set(future_effects) & set(a_effects):
            layer, pos, feature_idx = key
            if pos != pre_article_pos:
                continue
            if an_effects[key]["direct_effect"] <= 0:
                continue
            if future_effects[key]["direct_effect"] <= 0:
                continue
            dual_keys.append(key)
            feature_key = (layer, feature_idx)
            stats = feature_stats.setdefault(
                feature_key,
                {
                    "layer": layer,
                    "feature_idx": feature_idx,
                    "prompt_count": 0,
                    "prompts": [],
                    "mean_direct_effect_an": 0.0,
                    "mean_direct_effect_future": 0.0,
                    "mean_activation": 0.0,
                    "score_sum": 0.0,
                },
            )
            score = min(
                an_effects[key]["direct_effect"],
                future_effects[key]["direct_effect"],
            )
            stats["prompt_count"] += 1
            stats["prompts"].append(sentence)
            stats["mean_direct_effect_an"] += an_effects[key]["direct_effect"]
            stats["mean_direct_effect_future"] += future_effects[key]["direct_effect"]
            stats["mean_activation"] += an_effects[key]["activation"]
            stats["score_sum"] += score
        prompt_records.append(
            {
                "sentence": sentence,
                "listed_word": meta["listed_word"],
                "content_token_text": content_text,
                "pre_article_pos": pre_article_pos,
                "dual_feature_count": len(dual_keys),
            }
        )

    ranked = []
    for stats in feature_stats.values():
        count = stats["prompt_count"]
        ranked.append(
            {
                "layer": stats["layer"],
                "feature_idx": stats["feature_idx"],
                "prompt_count": count,
                "prompts": stats["prompts"],
                "mean_direct_effect_an": stats["mean_direct_effect_an"] / count,
                "mean_direct_effect_future": stats["mean_direct_effect_future"] / count,
                "mean_activation": stats["mean_activation"] / count,
                "mean_score": stats["score_sum"] / count,
                "label": f"`L{stats['layer']}/F{stats['feature_idx']}`",
            }
        )
    ranked.sort(
        key=lambda item: (
            item["prompt_count"],
            item["mean_score"],
            item["mean_direct_effect_future"],
        ),
        reverse=True,
    )
    selected = [
        item
        for item in ranked
        if item["prompt_count"] >= int(config["min_selection_prompt_count"])
    ][: int(config["top_content_features"])]

    # Fallback: if recurrence threshold yields nothing, keep top ranked dual-effect
    # features by mean score so the clincher still has a frozen content set.
    fallback_used = False
    if not selected:
        fallback_used = True
        selected = ranked[: int(config["top_content_features"])]

    selection = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_prompt_count": len(config["selection_sentences"]),
        "min_selection_prompt_count": int(config["min_selection_prompt_count"]),
        "fallback_used": fallback_used,
        "prompt_records": prompt_records,
        "ranked_dual_features": ranked[:50],
        "selected_features": selected,
    }
    SELECTION_PATH.write_text(json.dumps(selection, indent=2) + "\n")
    return selection


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
    layers = sorted({int(item["layer"]) for item in content_features})
    target_activation = sum(float(item["mean_activation"]) for item in content_features) / max(
        len(content_features), 1
    )
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
    # Keep a pool of near matches, then sample for reproducibility under seed.
    pool = candidates[: max(20, int(config["control_feature_count"]) * 5)]
    if len(pool) <= int(config["control_feature_count"]):
        chosen = pool
    else:
        chosen = rng.sample(pool, int(config["control_feature_count"]))
    chosen.sort(key=lambda item: (item["layer"], item["feature_idx"]))
    return chosen


def evaluate_condition(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    features: list[dict[str, Any]],
    config: dict[str, Any],
    condition_name: str,
) -> list[dict[str, Any]]:
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
            float(config["amplify_factor"]),
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
                "generated_article_changed": (
                    baseline_gen_article != intervention_gen_article
                ),
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
        "mean_delta_an_minus_a": sum(row["delta_an_minus_a"] for row in rows) / n if n else 0.0,
        "article_moved_toward_an_rate": rate("article_moved_toward_an"),
        "generated_article_changed_rate": rate("generated_article_changed"),
        "content_preserved_rate": rate("content_preserved"),
        "content_word_changed_rate": rate("content_word_changed"),
        "class_shifted_rate": rate("class_shifted"),
        "wrapper_like_rate": rate("wrapper_like"),
        "trajectory_like_rate": rate("trajectory_like"),
        "matched_twin_rate": rate("matched_twin"),
    }


def decide_clincher(
    content_summary: dict[str, Any],
    control_summary: dict[str, Any],
) -> tuple[str, str]:
    content_signal = (
        content_summary["article_moved_toward_an_rate"] >= 0.25
        or content_summary["mean_delta_an_minus_a"] >= 0.25
        or content_summary["generated_article_changed_rate"] >= 0.2
    )
    control_signal = (
        control_summary["article_moved_toward_an_rate"] >= 0.25
        or control_summary["mean_delta_an_minus_a"] >= 0.25
        or control_summary["generated_article_changed_rate"] >= 0.2
    )
    if not content_signal or (
        control_signal
        and content_summary["mean_delta_an_minus_a"]
        <= control_summary["mean_delta_an_minus_a"] + 0.125
        and content_summary["wrapper_like_rate"]
        <= control_summary["wrapper_like_rate"] + 1e-9
        and content_summary["trajectory_like_rate"]
        <= control_summary["trajectory_like_rate"] + 1e-9
    ):
        return (
            "inconclusive_or_nonspecific",
            "The content-feature amplification did not beat activation-matched random controls enough to support either a content-preserving planning effect or a clear trajectory-class effect. The stricter framework is unlikely to yield a clean result on this model/task without redesign.",
        )
    if (
        content_summary["wrapper_like_rate"] >= 0.15
        and content_summary["wrapper_like_rate"]
        > content_summary["trajectory_like_rate"]
        and content_summary["content_preserved_rate"]
        >= content_summary["content_word_changed_rate"]
    ):
        return (
            "framework_live_planning_path",
            "Amplifying frozen content-supporting features moved articles while preserving content more often than it class-switched, and more than controls. The stricter framework is live and the Outcome-1 path remains plausible.",
        )
    if (
        content_summary["trajectory_like_rate"] >= 0.15
        or (
            content_summary["class_shifted_rate"] >= 0.2
            and content_summary["content_preserved_rate"] < 0.5
        )
    ):
        return (
            "framework_live_trajectory_path",
            "Amplifying frozen content-supporting features changed behavior relative to controls, but mainly by changing content words / vowel-consonant class rather than preserving a fixed later word. The stricter framework is live and the Outcome-2 path is favored.",
        )
    return (
        "framework_live_mixed",
        "Amplifying frozen content-supporting features beat controls, but the pattern is mixed between content preservation and class switching. The framework can proceed, but the next experiment must separate these modes more cleanly.",
    )


def write_report(summary: dict[str, Any]) -> None:
    content = summary["content_summary"]
    control = summary["control_summary"]
    lines = [
        "# Planning Gain-of-Function Content Clincher",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        "",
        "## Question",
        "",
        "If we amplify a frozen set of content-supporting features in the style of *Latent Planning Emerges with Scale*, do held-out continuations show content-preserving article repair, trajectory-class switching, or nonspecific disruption relative to activation-matched random controls?",
        "",
        "## Design",
        "",
        f"- Demonstration: `{summary['demonstration']}`",
        f"- Selection prompts: {summary['selection_prompt_count']} expected-`an` occupation prompts (disjoint from the test set)",
        f"- Feature rule: recurring pre-article features with positive direct effect on both `an` and the listed-word first token",
        f"- Minimum prompt recurrence: {summary['min_selection_prompt_count']}",
        f"- Frozen content features amplified by {summary['amplify_factor']}× their prompt-specific activation",
        f"- Control: {summary['control_feature_count']} activation-matched random active features, same amplify factor",
        f"- Held-out test prompts: {summary['test_prompt_count']}",
        f"- Selection fallback used: {summary['fallback_used']}",
        "",
        "## Selected Content Features",
        "",
    ]
    if not summary["selected_features"]:
        lines.append("No content features were selected.")
    else:
        lines += [
            "| Feature | Prompt count | Mean score | Mean Δ-attr `an` | Mean Δ-attr future |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for item in summary["selected_features"]:
            lines.append(
                f"| {item['label']} | {item['prompt_count']} | "
                f"{item['mean_score']:.3f} | {item['mean_direct_effect_an']:.3f} | "
                f"{item['mean_direct_effect_future']:.3f} |"
            )
    lines += [
        "",
        "## Control Features",
        "",
        ", ".join(item["label"] for item in summary["control_features"]) or "None",
        "",
        "## Short Answer",
        "",
        summary["interpretation"],
        "",
        f"- Clincher decision: `{summary['decision']}`",
        "",
        "## Aggregate Scores",
        "",
        "| Metric | Content-feature amplify | Random control amplify |",
        "| --- | ---: | ---: |",
        f"| Mean Δ(`an`-`a`) | {content['mean_delta_an_minus_a']:.3f} | {control['mean_delta_an_minus_a']:.3f} |",
        f"| Article moved toward `an` | {content['article_moved_toward_an_rate']:.3f} | {control['article_moved_toward_an_rate']:.3f} |",
        f"| Generated article changed | {content['generated_article_changed_rate']:.3f} | {control['generated_article_changed_rate']:.3f} |",
        f"| Content preserved | {content['content_preserved_rate']:.3f} | {control['content_preserved_rate']:.3f} |",
        f"| Content word changed | {content['content_word_changed_rate']:.3f} | {control['content_word_changed_rate']:.3f} |",
        f"| Class shifted | {content['class_shifted_rate']:.3f} | {control['class_shifted_rate']:.3f} |",
        f"| Wrapper-like rate | {content['wrapper_like_rate']:.3f} | {control['wrapper_like_rate']:.3f} |",
        f"| Trajectory-like rate | {content['trajectory_like_rate']:.3f} | {control['trajectory_like_rate']:.3f} |",
        f"| Matched twin word | {content['matched_twin_rate']:.3f} | {control['matched_twin_rate']:.3f} |",
        "",
        "## Prompts With Generated Article Changes under Content Amplification",
        "",
    ]
    changed = [
        item
        for item in summary["content_examples"]
        if item["generated_article_changed"]
    ]
    if not changed:
        lines.append("No held-out prompt changed its generated article under content-feature amplification.")
    else:
        lines += [
            "| Prompt | Baseline | Intervention | Content preserved? | Class shifted? | Twin match? | Δ(`an`-`a`) |",
            "| --- | --- | --- | --- | --- | --- | ---: |",
        ]
        for item in changed:
            lines.append(
                f"| `{item['target_prompt']}` | "
                f"`{item['baseline_continuation']}` | "
                f"`{item['intervention_continuation']}` | "
                f"{item['content_preserved']} | {item['class_shifted']} | "
                f"{item['matched_twin']} | {item['delta_an_minus_a']:.3f} |"
            )
    lines += [
        "",
        "## Every Held-Out Prompt under Content Amplification",
        "",
        "| Prompt | Expected | Baseline continuation | Intervention continuation | Content preserved? | Class shifted? | Δ(`an`-`a`) |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for item in summary["content_examples"]:
        lines.append(
            f"| `{item['target_prompt']}` | `{item['expected_article']}` | "
            f"`{item['baseline_continuation']}` | `{item['intervention_continuation']}` | "
            f"{item['content_preserved']} | {item['class_shifted']} | "
            f"{item['delta_an_minus_a']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation Boundary",
        "",
        "This clincher asks whether Latent-Planning-style gain-of-function on frozen content-supporting features produces content-specific preparation effects on held-out prompts. A useful planning-supportive result requires article movement with content preservation above controls. A useful negative for content-specific planning is a control-beating effect that mainly class-switches content. If content features and random controls look alike, the framework should be redesigned before a larger study.",
        "",
        "## Source Artifacts",
        "",
        "- `results/selection.json`: recurring dual-effect feature ranking from the selection prompts",
        "- `results/graphs/`: per-selection-prompt article and future attribution graphs",
        "- `results/summary.json`: full machine-readable outputs",
        "",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.graph_only:
        run_graph_phase(args.graph_only[0], args.graph_only[1])
        return

    config = load_config()
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    dataset = verify_dataset(config)
    ensure_selection_graphs(config, dataset)
    selection = select_content_features(config)
    selected_features = selection["selected_features"]
    if not selected_features:
        raise RuntimeError("No content-supporting features available for the clincher")

    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    first_prompt = (
        f"{config['demonstration']} {config['test_examples'][0]['sentence']}"
    )
    first_pos = len(tokenizer(first_prompt, add_special_tokens=True).input_ids) - 1
    control_features = choose_control_features(
        model,
        first_prompt,
        first_pos,
        selected_features,
        config,
    )
    content_rows = evaluate_condition(
        model,
        tokenizer,
        config["test_examples"],
        selected_features,
        config,
        "content_amplify",
    )
    control_rows = evaluate_condition(
        model,
        tokenizer,
        config["test_examples"],
        control_features,
        config,
        "control_amplify",
    )
    content_summary = summarize_condition(content_rows)
    control_summary = summarize_condition(control_rows)
    decision, interpretation = decide_clincher(content_summary, control_summary)
    summary = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "runtime_seconds": time.time() - started,
        "demonstration": config["demonstration"],
        "amplify_factor": float(config["amplify_factor"]),
        "selection_prompt_count": len(config["selection_sentences"]),
        "min_selection_prompt_count": int(config["min_selection_prompt_count"]),
        "fallback_used": selection["fallback_used"],
        "test_prompt_count": len(config["test_examples"]),
        "control_feature_count": len(control_features),
        "selected_features": selected_features,
        "control_features": control_features,
        "content_summary": content_summary,
        "control_summary": control_summary,
        "decision": decision,
        "interpretation": interpretation,
        "content_examples": content_rows,
        "control_examples": control_rows,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary)
    logging.info("clincher decision: %s", decision)


if __name__ == "__main__":
    main()
