#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.lib.core import (
    dict_intervention_result,
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


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-only",
        choices=("article", "future"),
        help="Internal isolated attribution phase.",
    )
    return parser.parse_args()


def ensure_graphs() -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    for graph_name in ("article", "future"):
        graph_path = GRAPHS_DIR / f"{graph_name}.pt"
        if graph_path.exists():
            logging.info("Reusing existing graph %s", graph_path)
            continue
        logging.info("Starting isolated %s attribution", graph_name)
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--graph-only", graph_name],
            check=True,
        )


def run_graph_phase(graph_name: str) -> None:
    config = load_config()
    setup_file_logging(RESULTS_DIR)
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    if graph_name == "article":
        target_ids = [
            token_id_for_text(tokenizer, " a"),
            token_id_for_text(tokenizer, " an"),
        ]
        prompt = config["prompt"]
    else:
        target_ids = [token_id_for_text(tokenizer, config["future_target_text"])]
        prompt = config["future_prompt"]
    run_graph(model, prompt, graph_name, target_ids, config, GRAPHS_DIR)


def feature_effect_map(graph, target_id: int) -> dict[tuple[int, int, int], dict[str, Any]]:
    target_ids = [int(target.vocab_idx) for target in graph.logit_targets]
    target_offset = target_ids.index(target_id)
    n_features = int(len(graph.selected_features))
    row = graph.adjacency_matrix[
        -len(target_ids) + target_offset, :n_features
    ].detach().float().cpu()
    output = {}
    for column, selected_idx_tensor in enumerate(graph.selected_features):
        selected_idx = int(selected_idx_tensor)
        layer, pos, feature_idx = [
            int(value) for value in graph.active_features[selected_idx].tolist()
        ]
        output[(layer, pos, feature_idx)] = {
            "layer": layer,
            "pos": pos,
            "feature_idx": feature_idx,
            "activation": float(
                graph.activation_values[selected_idx].detach().float().cpu()
            ),
            "direct_effect": float(row[column]),
        }
    return output


def write_report(summary: dict[str, Any]) -> None:
    dual_effects = [
        item for item in summary["interventions"] if item["dual_effect"]
    ]
    strongest = min(
        dual_effects,
        key=lambda item: item["suppression_delta_an_minus_a"],
        default=None,
    )
    lines = [
        "# Ophthalmologist Planning Pilot",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        "",
        "## Question",
        "",
        "Before Gemma predicts the incorrect article `a`, are there active features that also causally support the future answer prefix ` ophthalm`?",
        "",
        "## Exact Prompts",
        "",
        f"- Article decision: `{summary['prompt']}`",
        f"- Future-word decision with the observed article fixed: `{summary['future_prompt']}`",
        "",
        "`ophthalmologist` tokenizes as ` ophthalm` + `ologist`. This pilot attributes the first future-token prefix, following the paper's feature-selection rule.",
        "",
        "## Baseline",
        "",
        f"- Article logits: `a`={summary['baseline']['a_logit']:.3f}, `an`={summary['baseline']['an_logit']:.3f}; `an-a` margin={summary['baseline']['an_minus_a']:.3f}.",
        f"- Future prefix ` ophthalm`: rank={summary['baseline']['future_rank']}, probability={summary['baseline']['future_prob']:.4f}.",
        "",
        "## Short Answer",
        "",
        summary["interpretation"],
        "",
        f"- Shared pre-article features in both attribution graphs: {summary['shared_pre_article_feature_count']}",
        f"- Shared features with positive direct effects on both `an` and ` ophthalm`: {summary['positive_shared_candidate_count']}",
        f"- Suppression candidates tested: {len(summary['interventions'])}",
        f"- Candidates whose suppression reduced both the future prefix and the `an-a` margin: {summary['dual_effect_candidate_count']}",
        "",
    ]
    if strongest is not None:
        lines += [
            "## Strongest Individual Result",
            "",
            f"Suppressing `L{strongest['layer']}/F{strongest['feature_idx']}` at the final prompt token:",
            "",
            f"- changed the incorrect `a` logit by {strongest['suppression_delta_a_logit']:.3f};",
            f"- changed the losing `an` logit by {strongest['suppression_delta_an_logit']:.3f};",
            f"- therefore changed the `an-a` margin by {strongest['suppression_delta_an_minus_a']:.3f}; and",
            f"- changed the later ` ophthalm` logit by {strongest['suppression_delta_future_logit']:.3f}.",
            "",
            "This is the cleanest result because the same suppression selectively weakens the grammatically correct preparation and the future answer prefix while leaving the chosen `a` logit unchanged. It does not yet tell us what the feature represents.",
            "",
        ]
    lines += [
        "## How Candidates Were Selected",
        "",
        "The article and future-word attribution graphs were built separately. A candidate had to be the exact same circuit-tracer feature at the pre-article token position in both graphs, with a positive direct attribution to both `an` and ` ophthalm`. The 20 candidates with the largest minimum of those two direct effects were suppressed one at a time. No answer feature was augmented.",
        "",
        "## Candidate Interventions",
        "",
        "| Feature | Activation | Direct Effect on `a` | Direct Effect on `an` | Direct Effect on ` ophthalm` | Suppression Δ`a` | Suppression Δ`an` | Suppression Δ(`an-a`) | Suppression Δ` ophthalm` | Dual Effect? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summary["interventions"]:
        lines.append(
            f"| `L{item['layer']}/F{item['feature_idx']}` | "
            f"{item['article_activation']:.3f} | {item['direct_effect_a']:.3f} | "
            f"{item['direct_effect_an']:.3f} | {item['direct_effect_future']:.3f} | "
            f"{item['suppression_delta_a_logit']:.3f} | "
            f"{item['suppression_delta_an_logit']:.3f} | "
            f"{item['suppression_delta_an_minus_a']:.3f} | "
            f"{item['suppression_delta_future_logit']:.3f} | "
            f"{item['dual_effect']} |"
        )
    lines += [
        "",
        "## Interpretation Boundary",
        "",
        "A dual-effect feature is evidence that a representation active before the article contributes to both the losing grammatical preparation `an` and the later answer prefix. Feature semantics still require validation across activating examples and held-out prompts. This single prompt cannot establish a general concealed-planning mechanism.",
        "",
        "The model and transcoder ran in bfloat16 to fit the 16 GiB machine, so small changes are quantized in roughly 0.125-logit increments in this run. The strongest 1.125-logit article-margin effect is substantially larger than that resolution; the 0.125 effects require replication.",
        "",
        "## Artifacts",
        "",
        "- `results/graphs/article.pt`: attribution graph for `a` and `an`.",
        "- `results/graphs/future.pt`: attribution graph for ` ophthalm` after the observed `a`.",
        "- `results/summary.json`: machine-readable baselines, candidates, and interventions.",
        "",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.graph_only:
        run_graph_phase(args.graph_only)
        return

    config = load_config()
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    ensure_graphs()
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    future_id = token_id_for_text(tokenizer, config["future_target_text"])
    article_ids = [a_id, an_id]

    article_baseline = logits_for_prompt(
        model, config["prompt"], article_ids, top_k=10, return_activations=False
    )
    future_baseline = logits_for_prompt(
        model, config["future_prompt"], [future_id], top_k=10, return_activations=False
    )
    article_graph = run_graph(
        model,
        config["prompt"],
        "article",
        article_ids,
        config,
        GRAPHS_DIR,
    )
    future_graph = run_graph(
        model,
        config["future_prompt"],
        "future",
        [future_id],
        config,
        GRAPHS_DIR,
    )

    a_effects = feature_effect_map(article_graph, a_id)
    an_effects = feature_effect_map(article_graph, an_id)
    future_effects = feature_effect_map(future_graph, future_id)
    pre_article_pos = len(tokenizer(config["prompt"], add_special_tokens=True).input_ids) - 1
    shared_keys = {
        key
        for key in set(a_effects) & set(an_effects) & set(future_effects)
        if key[1] == pre_article_pos
    }
    candidates = []
    for key in shared_keys:
        if an_effects[key]["direct_effect"] <= 0 or future_effects[key]["direct_effect"] <= 0:
            continue
        candidates.append(
            {
                "key": key,
                "score": min(
                    an_effects[key]["direct_effect"],
                    future_effects[key]["direct_effect"],
                ),
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    candidates = candidates[: int(config["top_candidates_to_intervene"])]

    interventions = []
    base_a = article_baseline["targets"][str(a_id)]["logit"]
    base_an = article_baseline["targets"][str(an_id)]["logit"]
    base_future = future_baseline["targets"][str(future_id)]["logit"]
    for candidate in candidates:
        layer, pos, feature_idx = candidate["key"]
        intervention = {
            "layer": layer,
            "pos": pos,
            "feature_idx": feature_idx,
            "value": 0.0,
        }
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
        delta_future = future_result["targets"][str(future_id)]["delta_logit"]
        interventions.append(
            {
                "layer": layer,
                "pos": pos,
                "feature_idx": feature_idx,
                "article_activation": an_effects[candidate["key"]]["activation"],
                "direct_effect_a": a_effects[candidate["key"]]["direct_effect"],
                "direct_effect_an": an_effects[candidate["key"]]["direct_effect"],
                "direct_effect_future": future_effects[candidate["key"]]["direct_effect"],
                "suppression_delta_a_logit": delta_a,
                "suppression_delta_an_logit": delta_an,
                "suppression_delta_an_minus_a": delta_margin,
                "suppression_delta_future_logit": delta_future,
                "dual_effect": delta_margin < 0 and delta_future < 0,
            }
        )
    dual_count = sum(item["dual_effect"] for item in interventions)
    if dual_count:
        interpretation = (
            "The pilot found at least one feature active before the incorrect article whose "
            "suppression weakens both the losing `an` preparation and the later ` ophthalm` "
            "prefix. This is preliminary evidence of a future-answer pathway, not yet evidence "
            "that a competing `a` circuit conceals it."
        )
    else:
        interpretation = (
            "The pilot did not find a pre-article feature with the required causal effect on both "
            "`an` and the later ` ophthalm` prefix. The current case does not yet support the "
            "concealed-planning mechanism."
        )
    summary = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "prompt": config["prompt"],
        "future_prompt": config["future_prompt"],
        "runtime_seconds": time.time() - started,
        "pre_article_pos": pre_article_pos,
        "baseline": {
            "a_logit": base_a,
            "an_logit": base_an,
            "an_minus_a": base_an - base_a,
            "future_logit": base_future,
            "future_prob": future_baseline["targets"][str(future_id)]["prob"],
            "future_rank": future_baseline["targets"][str(future_id)]["rank"],
        },
        "shared_pre_article_feature_count": len(shared_keys),
        "positive_shared_candidate_count": sum(
            an_effects[key]["direct_effect"] > 0
            and future_effects[key]["direct_effect"] > 0
            for key in shared_keys
        ),
        "dual_effect_candidate_count": dual_count,
        "interventions": interventions,
        "interpretation": interpretation,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary)
    logging.info("found %d dual-effect candidates", dual_count)


if __name__ == "__main__":
    main()
