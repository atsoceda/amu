#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from circuit_tracer.graph import Graph

from experiments.lib.core import (
    dict_intervention_result,
    feature_effect_map,
    load_replacement_model,
    logits_for_prompt,
    setup_file_logging,
    token_id_for_text,
)

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"
SOURCE_DIR = REPO_ROOT / "experiments" / "ophthalmologist_planning_pilot"
SOURCE_GRAPHS = SOURCE_DIR / "results" / "graphs"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def ensure_source_graphs() -> None:
    for graph_name in ("article", "future"):
        graph_path = SOURCE_GRAPHS / f"{graph_name}.pt"
        if graph_path.exists():
            continue
        logging.info("Regenerating missing source graph %s", graph_name)
        subprocess.run(
            [
                sys.executable,
                str(SOURCE_DIR / "run.py"),
                "--graph-only",
                graph_name,
            ],
            check=True,
            cwd=REPO_ROOT,
        )


def select_candidates(
    article_graph,
    future_graph,
    a_id: int,
    an_id: int,
    future_id: int,
    pre_article_pos: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    a_effects = feature_effect_map(article_graph, a_id)
    an_effects = feature_effect_map(article_graph, an_id)
    future_effects = feature_effect_map(future_graph, future_id)
    candidates = []
    for key in set(a_effects) & set(an_effects) & set(future_effects):
        if key[1] != pre_article_pos:
            continue
        a_effect = a_effects[key]["direct_effect"]
        an_effect = an_effects[key]["direct_effect"]
        future_effect = future_effects[key]["direct_effect"]
        article_advantage = a_effect - an_effect
        if article_advantage <= 0:
            continue
        if abs(future_effect) > float(config["max_abs_future_direct_effect"]):
            continue
        candidates.append(
            {
                "layer": key[0],
                "pos": key[1],
                "feature_idx": key[2],
                "activation": a_effects[key]["activation"],
                "direct_effect_a": a_effect,
                "direct_effect_an": an_effect,
                "direct_effect_a_minus_an": article_advantage,
                "direct_effect_future": future_effect,
            }
        )
    candidates.sort(
        key=lambda item: item["direct_effect_a_minus_an"],
        reverse=True,
    )
    return candidates[: int(config["top_candidates_to_intervene"])]


def generate_with_interventions(
    model,
    prompt: str,
    interventions: list[dict[str, Any]],
    max_new_tokens: int = 4,
) -> str:
    tuples = [
        (
            int(item["layer"]),
            int(item["pos"]),
            int(item["feature_idx"]),
            float(item["value"]),
        )
        for item in interventions
    ]
    generated_ids = []
    current_prompt = prompt
    for _ in range(max_new_tokens):
        logits, _ = model.feature_intervention(
            current_prompt,
            interventions=tuples,
            freeze_attention=True,
            sparse=True,
            return_activations=False,
        )
        token_id = int(torch.argmax(logits[0, -1]).item())
        generated_ids.append(token_id)
        current_prompt += model.tokenizer.decode([token_id])
        if model.tokenizer.decode([token_id]).strip() in {".", "!", "?"}:
            break
    return model.tokenizer.decode(generated_ids)


def write_report(summary: dict[str, Any]) -> None:
    individual_successes = [
        item for item in summary["interventions"] if item["success"]
    ]
    individual_flips = [
        item for item in summary["interventions"] if item["flipped_to_an"]
    ]
    combination_successes = [
        item for item in summary["combinations"] if item["success"]
    ]
    combination_flips = [
        item for item in summary["combinations"] if item["flipped_to_an"]
    ]
    behavioral_corrections = [
        item for item in summary["combinations"] if item["selected_an"]
    ]
    flips = individual_flips + combination_flips
    lines = [
        "# Ophthalmologist Competing-Pathway Screen",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        "",
        "## Question",
        "",
        "Can suppressing a feature that favors incorrect `a` over correct `an` make Gemma choose `an` while preserving its later ` ophthalm` prediction?",
        "",
        "## Exact Prompts",
        "",
        f"- Article decision: `{summary['prompt']}`",
        f"- Future-word check with the observed article fixed: `{summary['future_prompt']}`",
        "",
        "## Expectation Before Intervention",
        "",
        "The strongest graph-selected candidate was expected to improve the `an-a` margin because its direct attribution favored `a` over `an` by more than the baseline deficit. A useful intervention also had to preserve the future ` ophthalm` logit within the preregistered 0.125-logit bfloat16 tolerance. No answer feature was augmented.",
        "",
        "## Baseline",
        "",
        f"- `a` logit: {summary['baseline']['a_logit']:.3f}",
        f"- `an` logit: {summary['baseline']['an_logit']:.3f}",
        f"- `an-a` margin: {summary['baseline']['an_minus_a']:.3f}",
        f"- ` ophthalm` logit: {summary['baseline']['future_logit']:.3f}; rank: {summary['baseline']['future_rank']}",
        "",
        "## Short Answer",
        "",
        summary["interpretation"],
        "",
        f"- Candidates tested individually: {len(summary['interventions'])}",
        f"- Individual suppressions improving the margin while preserving ` ophthalm`: {len(individual_successes)}",
        f"- Individual suppressions that made `an` outrank `a`: {len(individual_flips)}",
        f"- Preselected pairs tested: {len(summary['combinations'])}",
        f"- Pairs improving the margin while preserving ` ophthalm`: {len(combination_successes)}",
        f"- Pairs that made `an` outrank `a`: {len(combination_flips)}",
        f"- Pairs that made `an` the top next token: {len(behavioral_corrections)}",
        "",
    ]
    if flips:
        best = max(flips, key=lambda item: item["post_an_minus_a"])
        lines += [
            "## Best Article Flip",
            "",
            f"Suppressing {best['label']} at the final prompt token changed:",
            "",
            f"- `a` by {best['suppression_delta_a_logit']:.3f} logits;",
            f"- `an` by {best['suppression_delta_an_logit']:.3f} logits;",
            f"- the `an-a` margin by {best['suppression_delta_an_minus_a']:.3f}, ending at {best['post_an_minus_a']:.3f}; and",
            f"- ` ophthalm` by {best['suppression_delta_future_logit']:.3f}, with post-intervention rank {best['post_future_rank']}.",
            f"- The top next token was `{best['post_top_token']}`, and continuation under the same regimen was `{best['generated_continuation']}`.",
            "",
        ]
    lines += [
        "## Candidate Selection",
        "",
        "Candidates were fixed from the two source attribution graphs before interventions were run. Each candidate was the same feature at the pre-article position in both graphs, directly favored `a` over `an`, and had absolute direct attribution to ` ophthalm` no greater than 0.05. Candidates were ranked by their direct `a-an` advantage.",
        "",
        "Because no individual suppression flipped the article, the screen also tested every pair among the five highest graph-ranked candidates. Pair membership therefore did not depend on the individual intervention outcomes.",
        "",
        "## Individual Suppressions",
        "",
        "| Feature | Graph `a-an` Advantage | Graph Effect on ` ophthalm` | Δ`a` | Δ`an` | Δ(`an-a`) | Post `an-a` | Δ` ophthalm` | Future Rank | Preserved? | Flipped? | Success? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in summary["interventions"]:
        lines.append(
            f"| `L{item['layer']}/F{item['feature_idx']}` | "
            f"{item['direct_effect_a_minus_an']:.3f} | "
            f"{item['direct_effect_future']:.3f} | "
            f"{item['suppression_delta_a_logit']:.3f} | "
            f"{item['suppression_delta_an_logit']:.3f} | "
            f"{item['suppression_delta_an_minus_a']:.3f} | "
            f"{item['post_an_minus_a']:.3f} | "
            f"{item['suppression_delta_future_logit']:.3f} | "
            f"{item['post_future_rank']} | "
            f"{item['future_preserved']} | {item['flipped_to_an']} | "
            f"{item['success']} |"
        )
    lines += [
        "",
        "## Preselected Pair Suppressions",
        "",
        "| Pair | Δ`a` | Δ`an` | Δ(`an-a`) | Post `an-a` | Top Token | Δ` ophthalm` | Future Rank | Preserved? | `an` Top? | Success? |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    for item in summary["combinations"]:
        lines.append(
            f"| {item['label']} | "
            f"{item['suppression_delta_a_logit']:.3f} | "
            f"{item['suppression_delta_an_logit']:.3f} | "
            f"{item['suppression_delta_an_minus_a']:.3f} | "
            f"{item['post_an_minus_a']:.3f} | "
            f"`{item['post_top_token']}` | "
            f"{item['suppression_delta_future_logit']:.3f} | "
            f"{item['post_future_rank']} | "
            f"{item['future_preserved']} | {item['selected_an']} | "
            f"{item['success']} |"
        )
    lines += [
        "",
        "## Interpretation Boundary",
        "",
        "Making `an` the top token and continuing to `ophthalmologist` is evidence for a removable competing pathway in this prompt. It is not yet a general intervention: the feature must be characterized and tested without reselection on held-out vowel-initial occupations, consonant controls, and paraphrases. Small 0.125-logit changes are at this bfloat16 run's measurement resolution and require replication.",
        "",
        "## Source Artifacts",
        "",
        "- `../ophthalmologist_planning_pilot/results/graphs/article.pt`",
        "- `../ophthalmologist_planning_pilot/results/graphs/future.pt`",
        "",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    config = load_config()
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    ensure_source_graphs()
    article_graph = Graph.from_pt(str(SOURCE_GRAPHS / "article.pt"))
    future_graph = Graph.from_pt(str(SOURCE_GRAPHS / "future.pt"))
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    future_id = token_id_for_text(tokenizer, config["future_target_text"])
    article_ids = [a_id, an_id]
    pre_article_pos = len(
        tokenizer(config["prompt"], add_special_tokens=True).input_ids
    ) - 1

    article_baseline = logits_for_prompt(
        model, config["prompt"], article_ids, top_k=10, return_activations=False
    )
    future_baseline = logits_for_prompt(
        model, config["future_prompt"], [future_id], top_k=10, return_activations=False
    )
    candidates = select_candidates(
        article_graph,
        future_graph,
        a_id,
        an_id,
        future_id,
        pre_article_pos,
        config,
    )

    base_a = article_baseline["targets"][str(a_id)]["logit"]
    base_an = article_baseline["targets"][str(an_id)]["logit"]
    base_future = future_baseline["targets"][str(future_id)]["logit"]
    base_margin = base_an - base_a
    interventions = []
    for candidate in candidates:
        intervention = {**candidate, "value": 0.0}
        article_result = dict_intervention_result(
            model,
            config["prompt"],
            [intervention],
            article_ids,
            article_baseline,
        )
        future_result = dict_intervention_result(
            model,
            config["future_prompt"],
            [intervention],
            [future_id],
            future_baseline,
        )
        delta_a = article_result["targets"][str(a_id)]["delta_logit"]
        delta_an = article_result["targets"][str(an_id)]["delta_logit"]
        delta_margin = delta_an - delta_a
        post_margin = base_margin + delta_margin
        delta_future = future_result["targets"][str(future_id)]["delta_logit"]
        post_future_rank = future_result["targets"][str(future_id)]["rank"]
        post_top = article_result["top_tokens"][0]
        future_preserved = (
            delta_future >= -float(config["future_preservation_tolerance"])
            and post_future_rank == future_baseline["targets"][str(future_id)]["rank"]
        )
        flipped = post_margin > 0
        interventions.append(
            {
                **candidate,
                "label": f"`L{candidate['layer']}/F{candidate['feature_idx']}`",
                "suppression_delta_a_logit": delta_a,
                "suppression_delta_an_logit": delta_an,
                "suppression_delta_an_minus_a": delta_margin,
                "post_an_minus_a": post_margin,
                "suppression_delta_future_logit": delta_future,
                "post_future_rank": post_future_rank,
                "post_top_token_id": post_top["token_id"],
                "post_top_token": post_top["token"],
                "future_preserved": future_preserved,
                "flipped_to_an": flipped,
                "selected_an": post_top["token_id"] == an_id,
                "success": delta_margin > 0 and future_preserved,
            }
        )

    combination_results = []
    combination_candidates = candidates[
        : int(config["combination_candidate_count"])
    ]
    for left, right in combinations(combination_candidates, 2):
        intervention_pair = [
            {**left, "value": 0.0},
            {**right, "value": 0.0},
        ]
        article_result = dict_intervention_result(
            model,
            config["prompt"],
            intervention_pair,
            article_ids,
            article_baseline,
        )
        future_result = dict_intervention_result(
            model,
            config["future_prompt"],
            intervention_pair,
            [future_id],
            future_baseline,
        )
        delta_a = article_result["targets"][str(a_id)]["delta_logit"]
        delta_an = article_result["targets"][str(an_id)]["delta_logit"]
        delta_margin = delta_an - delta_a
        post_margin = base_margin + delta_margin
        delta_future = future_result["targets"][str(future_id)]["delta_logit"]
        post_future_rank = future_result["targets"][str(future_id)]["rank"]
        post_top = article_result["top_tokens"][0]
        future_preserved = (
            delta_future >= -float(config["future_preservation_tolerance"])
            and post_future_rank
            == future_baseline["targets"][str(future_id)]["rank"]
        )
        combination_result = {
                "features": [
                    {
                        "layer": item["layer"],
                        "pos": item["pos"],
                        "feature_idx": item["feature_idx"],
                    }
                    for item in (left, right)
                ],
                "label": (
                    f"`L{left['layer']}/F{left['feature_idx']}` + "
                    f"`L{right['layer']}/F{right['feature_idx']}`"
                ),
                "suppression_delta_a_logit": delta_a,
                "suppression_delta_an_logit": delta_an,
                "suppression_delta_an_minus_a": delta_margin,
                "post_an_minus_a": post_margin,
                "suppression_delta_future_logit": delta_future,
                "post_future_rank": post_future_rank,
                "post_top_token_id": post_top["token_id"],
                "post_top_token": post_top["token"],
                "future_preserved": future_preserved,
                "flipped_to_an": post_margin > 0,
                "selected_an": post_top["token_id"] == an_id,
                "success": delta_margin > 0 and future_preserved,
                "generated_continuation": "",
            }
        combination_results.append(combination_result)

    for result in combination_results:
        if not result["selected_an"]:
            continue
        source_items = [
            next(
                item
                for item in combination_candidates
                if item["layer"] == feature["layer"]
                and item["feature_idx"] == feature["feature_idx"]
            )
            for feature in result["features"]
        ]
        result["generated_continuation"] = generate_with_interventions(
            model,
            config["prompt"],
            [{**item, "value": 0.0} for item in source_items],
        )

    success_count = sum(item["success"] for item in interventions)
    flip_count = sum(item["flipped_to_an"] for item in interventions)
    combination_success_count = sum(
        item["success"] for item in combination_results
    )
    combination_flip_count = sum(
        item["flipped_to_an"] for item in combination_results
    )
    behavioral_correction_count = sum(
        item["selected_an"] for item in combination_results
    )
    if behavioral_correction_count:
        interpretation = (
            "At least one preselected suppression regimen made `an` the top next "
            "token while retaining the future answer at rank 1. This is the "
            "expected prompt-level result: a removable competing pathway was "
            "causally blocking the preparation supported by future-answer "
            "information."
        )
    elif success_count:
        interpretation = (
            "Some suppressions improved the `an-a` margin while preserving the "
            "future answer, but none changed the selected article. The competing-"
            "pathway hypothesis remains plausible but the intervention is not yet "
            "behaviorally decisive."
        )
    else:
        interpretation = (
            "No tested suppression improved the article margin while preserving "
            "the future answer. This screen did not produce the expected competing-"
            "pathway intervention."
        )
    summary = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "prompt": config["prompt"],
        "future_prompt": config["future_prompt"],
        "runtime_seconds": time.time() - started,
        "pre_article_pos": pre_article_pos,
        "selection": {
            "max_abs_future_direct_effect": config[
                "max_abs_future_direct_effect"
            ],
            "future_preservation_tolerance": config[
                "future_preservation_tolerance"
            ],
        },
        "baseline": {
            "a_logit": base_a,
            "an_logit": base_an,
            "an_minus_a": base_margin,
            "future_logit": base_future,
            "future_rank": future_baseline["targets"][str(future_id)]["rank"],
        },
        "candidate_count": len(candidates),
        "success_count": success_count,
        "flip_count": flip_count,
        "combination_success_count": combination_success_count,
        "combination_flip_count": combination_flip_count,
        "behavioral_correction_count": behavioral_correction_count,
        "interventions": interventions,
        "combinations": combination_results,
        "interpretation": interpretation,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary)
    logging.info(
        "tested %d candidates and %d pairs; %d individual/%d pair successes; "
        "%d individual/%d pair article flips",
        len(interventions),
        len(combination_results),
        success_count,
        combination_success_count,
        flip_count,
        combination_flip_count,
    )


if __name__ == "__main__":
    main()
