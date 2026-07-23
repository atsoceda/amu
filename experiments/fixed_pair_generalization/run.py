#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def load_examples(config: dict[str, Any]) -> list[dict[str, str]]:
    path = (EXP_DIR / config["dataset_path"]).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != config["dataset_sha256"]:
        raise ValueError(f"Dataset checksum mismatch: {digest}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row["sentence"].startswith("Someone who")
        and row["sentence"] != config["excluded_source_sentence"]
    ]


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


def grammatical_agreement(article: str, word: str) -> bool:
    if article not in {"a", "an"} or not word:
        return False
    return (article == "an") == (word[0] in "aeiou")


def write_report(summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    lines = [
        "# Fixed-Pair Generalization Report",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        "",
        "## Question",
        "",
        "Does the fixed `L13/F10304 + L14/F1949` suppression discovered on `ophthalmologist` improve article preparation on other occupations without corrupting consonant controls?",
        "",
        "## Design",
        "",
        f"- Demonstration: `{summary['demonstration']}`",
        f"- Held-out occupation prompts: {summary['example_count']}",
        f"- Expected `an` prompts: {counts['expected_an']}",
        f"- Expected `a` controls: {counts['expected_a']}",
        "- The ophthalmologist source sentence was excluded.",
        "- The same two feature identities were suppressed at each prompt's final token. No held-out feature selection was performed.",
        "",
        "## Short Answer",
        "",
        summary["interpretation"],
        "",
        "## Outcome Counts",
        "",
        "| Outcome | Count |",
        "| --- | ---: |",
        f"| Expected-`an` prompts corrected to generated `an` | {counts['an_corrections']} |",
        f"| Expected-`an` prompts regressed away from generated `an` | {counts['an_regressions']} |",
        f"| Expected-`a` controls incorrectly changed to generated `an` | {counts['a_false_flips']} |",
        f"| Expected-`a` controls preserved as generated `a` | {counts['a_preserved']} |",
        f"| All prompts whose top token changed | {counts['top_token_changes']} |",
        f"| All prompts whose generated article changed | {counts['generated_article_changes']} |",
        f"| Continuations whose first content word changed | {counts['content_word_changes']} |",
        f"| Realized article/word agreement repaired | {counts['grammar_repairs']} |",
        f"| Realized article/word agreement regressed | {counts['grammar_regressions']} |",
        f"| Exact listed-word completions before intervention | {counts['baseline_exact_word']} |",
        f"| Exact listed-word completions after intervention | {counts['intervention_exact_word']} |",
        "",
        "## Prompts Whose Top Article Changed",
        "",
    ]
    changed = [
        item for item in summary["examples"] if item["generated_article_changed"]
    ]
    if not changed:
        lines.append("The fixed pair did not change the top token on any held-out prompt.")
    else:
        lines += [
            "| Prompt | Expected | Baseline | Intervention | Baseline Continuation | Intervention Continuation | Content Changed? | Δ(`an-a`) |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: |",
        ]
        for item in changed:
            lines.append(
                f"| `{item['target_prompt']}` | `{item['expected_article']}` | "
                f"`{item['baseline_generated_article']}` | "
                f"`{item['intervention_generated_article']}` | "
                f"`{item['baseline_continuation']}` | "
                f"`{item['intervention_continuation']}` | "
                f"{item['content_word_changed']} | "
                f"{item['delta_an_minus_a']:.3f} |"
            )
    lines += [
        "",
        "## Every Held-Out Prompt",
        "",
        "| Prompt | Listed Word | Expected | Baseline Generated Article | Intervention Generated Article | Δ(`an-a`) | Baseline Continuation | Intervention Continuation |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for item in summary["examples"]:
        lines.append(
            f"| `{item['target_prompt']}` | `{item['listed_word']}` | "
            f"`{item['expected_article']}` | "
            f"`{item['baseline_generated_article']}` | "
            f"`{item['intervention_generated_article']}` | "
            f"{item['delta_an_minus_a']:.3f} | "
            f"`{item['baseline_continuation']}` | "
            f"`{item['intervention_continuation']}` |"
        )
    lines += [
        "",
        "## Interpretation Boundary",
        "",
        "A useful content-preserving intervention must repair multiple held-out expected-`an` cases while producing few or no target-label errors on expected-`a` controls and while keeping the intended content stable. A grammatically valid switch such as `a pilot` to `an aviator` is not a grammatical error, but it is evidence that the intervention changes answer selection rather than merely exposing a fixed future plan. This screen does not reselect features per prompt.",
        "",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    config = load_config()
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    examples = load_examples(config)
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    target_ids = [a_id, an_id]
    results = []

    for index, example in enumerate(examples, start=1):
        prompt = f"{config['demonstration']} {example['sentence']}"
        position = len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1
        interventions = [
            {
                "layer": item["layer"],
                "pos": position,
                "feature_idx": item["feature_idx"],
                "value": 0.0,
            }
            for item in config["fixed_features"]
        ]
        baseline = logits_for_prompt(
            model, prompt, target_ids, top_k=10, return_activations=False
        )
        intervention = dict_intervention_result(
            model,
            prompt,
            interventions,
            target_ids,
            baseline,
        )
        baseline_top_id = baseline["top_tokens"][0]["token_id"]
        intervention_top_id = intervention["top_tokens"][0]["token_id"]
        baseline_article = article_label(baseline_top_id, a_id, an_id)
        intervention_article = article_label(intervention_top_id, a_id, an_id)
        delta_a = intervention["targets"][str(a_id)]["delta_logit"]
        delta_an = intervention["targets"][str(an_id)]["delta_logit"]
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
        listed_word = example["word"].lower()
        baseline_realized_article, baseline_word = article_and_word(
            baseline_continuation
        )
        intervention_realized_article, intervention_word = article_and_word(
            intervention_continuation
        )
        results.append(
            {
                "index": index,
                "target_prompt": example["sentence"],
                "listed_word": listed_word,
                "expected_article": example["article"],
                "position": position,
                "baseline_article": baseline_article,
                "intervention_article": intervention_article,
                "baseline_top_token": baseline["top_tokens"][0]["token"],
                "intervention_top_token": intervention["top_tokens"][0]["token"],
                "baseline_an_minus_a": (
                    baseline["targets"][str(an_id)]["logit"]
                    - baseline["targets"][str(a_id)]["logit"]
                ),
                "delta_a": delta_a,
                "delta_an": delta_an,
                "delta_an_minus_a": delta_an - delta_a,
                "baseline_continuation": baseline_continuation,
                "intervention_continuation": intervention_continuation,
                "baseline_generated_word": baseline_word,
                "intervention_generated_word": intervention_word,
                "baseline_generated_article": baseline_realized_article,
                "intervention_generated_article": intervention_realized_article,
                "baseline_exact_word": listed_word == baseline_word,
                "intervention_exact_word": listed_word == intervention_word,
                "content_word_changed": baseline_word != intervention_word,
                "baseline_grammar_agreement": grammatical_agreement(
                    baseline_realized_article,
                    baseline_word,
                ),
                "intervention_grammar_agreement": grammatical_agreement(
                    intervention_realized_article,
                    intervention_word,
                ),
                "top_token_changed": baseline_top_id != intervention_top_id,
                "generated_article_changed": (
                    baseline_realized_article != intervention_realized_article
                ),
            }
        )
        if index % 10 == 0:
            logging.info("evaluated %d/%d prompts", index, len(examples))

    counts = Counter()
    for item in results:
        expected = item["expected_article"]
        counts[f"expected_{expected}"] += 1
        if expected == "an":
            if (
                item["baseline_generated_article"] != "an"
                and item["intervention_generated_article"] == "an"
            ):
                counts["an_corrections"] += 1
            if (
                item["baseline_generated_article"] == "an"
                and item["intervention_generated_article"] != "an"
            ):
                counts["an_regressions"] += 1
        else:
            if item["intervention_generated_article"] == "an":
                counts["a_false_flips"] += 1
            if item["intervention_generated_article"] == "a":
                counts["a_preserved"] += 1
        counts["top_token_changes"] += item["top_token_changed"]
        counts["generated_article_changes"] += item["generated_article_changed"]
        counts["content_word_changes"] += item["content_word_changed"]
        counts["grammar_repairs"] += (
            not item["baseline_grammar_agreement"]
            and item["intervention_grammar_agreement"]
        )
        counts["grammar_regressions"] += (
            item["baseline_grammar_agreement"]
            and not item["intervention_grammar_agreement"]
        )
        counts["baseline_exact_word"] += item["baseline_exact_word"]
        counts["intervention_exact_word"] += item["intervention_exact_word"]

    required_keys = [
        "expected_an",
        "expected_a",
        "an_corrections",
        "an_regressions",
        "a_false_flips",
        "a_preserved",
        "top_token_changes",
        "generated_article_changes",
        "content_word_changes",
        "grammar_repairs",
        "grammar_regressions",
        "baseline_exact_word",
        "intervention_exact_word",
    ]
    count_dict = {key: int(counts[key]) for key in required_keys}
    if (
        count_dict["an_corrections"] > 0
        and count_dict["a_false_flips"] == 0
        and count_dict["content_word_changes"] == 0
    ):
        interpretation = (
            "The fixed pair generalized behaviorally: it corrected at least one "
            "held-out expected-`an` prompt without causing false `an` flips on "
            "expected-`a` controls."
        )
    elif count_dict["top_token_changes"] == 0:
        interpretation = (
            "The fixed pair did not change the selected next token on any held-out "
            "occupation prompt. The source correction does not yet generalize "
            "behaviorally across this dataset."
        )
    else:
        interpretation = (
            "The fixed pair changed held-out behavior, but it did not generalize "
            "as a clean content-preserving preparation intervention. It repaired "
            "some expected-`an` labels while also changing many expected-`a` "
            "controls and frequently changing the answer word itself."
        )
    summary = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "runtime_seconds": time.time() - started,
        "demonstration": config["demonstration"],
        "excluded_source_sentence": config["excluded_source_sentence"],
        "fixed_features": config["fixed_features"],
        "example_count": len(results),
        "counts": count_dict,
        "interpretation": interpretation,
        "examples": results,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary)
    logging.info("completed held-out screen: %s", count_dict)


if __name__ == "__main__":
    main()
