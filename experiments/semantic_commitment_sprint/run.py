#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.core import (
    CandidateFeature,
    graph_features_for_target,
    intervention_result,
    load_replacement_model,
    logits_for_prompt,
    run_graph as shared_run_graph,
    serialize_feature,
    setup_file_logging,
    token_id_for_text,
)

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def setup_logging() -> None:
    setup_file_logging(RESULTS_DIR)


def run_graph(model, prompt: str, prompt_id: str, target_ids: list[int], config: dict[str, Any]):
    return shared_run_graph(model, prompt, prompt_id, target_ids, config, GRAPHS_DIR)


def build_interventions(
    semantic_features: list[CandidateFeature],
    fallback_features: list[CandidateFeature],
    source_semantic_features: list[CandidateFeature],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    source_by_key = {
        (f.layer, f.pos, f.feature_idx): f.activation for f in source_semantic_features
    }

    for feature_set_name, features in [
        ("semantic_candidate", semantic_features),
        ("fallback_candidate", fallback_features),
    ]:
        for i, feature in enumerate(features):
            key = (feature.layer, feature.pos, feature.feature_idx)
            specs.append(
                {
                    "name": f"{feature_set_name}_{i+1}_zero",
                    "kind": "individual_zero",
                    "feature_set": feature_set_name,
                    "features": [serialize_feature(feature)],
                    "interventions": [(feature.layer, feature.pos, feature.feature_idx, 0.0)],
                }
            )
            specs.append(
                {
                    "name": f"{feature_set_name}_{i+1}_double",
                    "kind": "individual_double",
                    "feature_set": feature_set_name,
                    "features": [serialize_feature(feature)],
                    "interventions": [
                        (feature.layer, feature.pos, feature.feature_idx, float(feature.activation * 2.0))
                    ],
                }
            )
            if key in source_by_key:
                specs.append(
                    {
                        "name": f"{feature_set_name}_{i+1}_source_patch",
                        "kind": "individual_source_patch",
                        "feature_set": feature_set_name,
                        "features": [serialize_feature(feature)],
                        "interventions": [
                            (feature.layer, feature.pos, feature.feature_idx, float(source_by_key[key]))
                        ],
                    }
                )

    for size in config["group_sizes"]:
        for feature_set_name, features, mode in [
            ("semantic_candidate", semantic_features, "double"),
            ("fallback_candidate", fallback_features, "zero"),
        ]:
            group = features[: int(size)]
            if not group:
                continue
            interventions = []
            for feature in group:
                value = 0.0 if mode == "zero" else float(feature.activation * 2.0)
                interventions.append((feature.layer, feature.pos, feature.feature_idx, value))
            specs.append(
                {
                    "name": f"{feature_set_name}_top{size}_{mode}",
                    "kind": f"group_{mode}",
                    "feature_set": feature_set_name,
                    "features": [serialize_feature(f) for f in group],
                    "interventions": interventions,
                }
            )
    return specs


def write_report(summary: dict[str, Any]) -> None:
    semantic_id = str(summary["token_ids"]["semantic_target"])
    fallback_id = str(summary["token_ids"]["observed_fallback"])
    interventions = summary["interventions"]
    ranked = sorted(
        interventions,
        key=lambda r: (
            r["result"]["targets"][semantic_id]["delta_logit"]
            - r["result"]["targets"][fallback_id]["delta_logit"]
        ),
        reverse=True,
    )

    lines = [
        "# Semantic Commitment Sprint Report",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['config']['model']}`",
        f"Intervention prompt: `{summary['intervention_prompt']['text']}`",
        "",
        "## Baseline",
        "",
    ]
    base = summary["baselines"][summary["intervention_prompt"]["id"]]
    for item in base["top_tokens"][:10]:
        lines.append(
            f"- rank {item['rank']}: `{item['token']}` prob={item['prob']:.6f} logit={item['logit']:.3f}"
        )

    lines += ["", "## Best Interventions", ""]
    for item in ranked[:12]:
        result = item["result"]["targets"]
        semantic = result[semantic_id]
        fallback = result[fallback_id]
        score = semantic["delta_logit"] - fallback["delta_logit"]
        lines.append(
            f"- `{item['name']}` score={score:.3f}; "
            f"semantic `{semantic['token']}` delta_logit={semantic['delta_logit']:.3f}, "
            f"rank_delta={semantic['delta_rank']}; "
            f"fallback `{fallback['token']}` delta_logit={fallback['delta_logit']:.3f}, "
            f"rank_delta={fallback['delta_rank']}"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "This sprint is exploratory. Treat a large directional logit movement as a candidate causal handle, not a final representation label. The next validation step is to rerun the strongest interventions across a larger matched prompt family and test specificity against unrelated semantic targets.",
        "",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    setup_logging()
    config = load_config()
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    model = load_replacement_model(config)
    tokenizer = model.tokenizer

    target_ids = []
    target_text_to_id = {}
    for text in config["target_token_texts"]:
        token_id = token_id_for_text(tokenizer, text)
        target_text_to_id[text] = token_id
        if token_id not in target_ids:
            target_ids.append(token_id)

    logging.info("Target ids: %s", target_text_to_id)
    baselines = {}
    graphs = {}
    top_features = {}

    for prompt in config["prompts"]:
        prompt_id = prompt["id"]
        logging.info("Baseline logits for %s", prompt_id)
        baselines[prompt_id] = logits_for_prompt(model, prompt["text"], target_ids)
        graphs[prompt_id] = run_graph(model, prompt["text"], prompt_id, target_ids, config)
        top_features[prompt_id] = {}
        for token_id in target_ids:
            top_features[prompt_id][str(token_id)] = [
                serialize_feature(f)
                for f in graph_features_for_target(
                    graphs[prompt_id],
                    tokenizer,
                    prompt_id,
                    token_id,
                    int(config["top_features_per_target"]),
                )
            ]

    intervention_prompt = next(p for p in config["prompts"] if p["id"] == config["intervention_prompt_id"])
    semantic_target_id = target_text_to_id[config["semantic_target_text"]]
    fallback_ids = [target_text_to_id[t] for t in config["fallback_target_texts"]]
    base = baselines[intervention_prompt["id"]]
    observed_fallback_id = min(
        fallback_ids,
        key=lambda tid: base["targets"][str(tid)]["rank"],
    )

    semantic_features = [
        CandidateFeature(**f)
        for f in top_features[intervention_prompt["id"]][str(semantic_target_id)]
    ]
    fallback_features = [
        CandidateFeature(**f)
        for f in top_features[intervention_prompt["id"]][str(observed_fallback_id)]
    ]
    source_semantic_features = [
        CandidateFeature(**f)
        for f in top_features[config["semantic_source_prompt_id"]][str(semantic_target_id)]
    ]

    specs = build_interventions(semantic_features, fallback_features, source_semantic_features, config)
    intervention_results = []
    scoring_ids = sorted(set([semantic_target_id, observed_fallback_id] + fallback_ids))
    for spec in specs:
        logging.info("Intervention %s", spec["name"])
        result = intervention_result(
            model,
            intervention_prompt["text"],
            spec["interventions"],
            scoring_ids,
            base,
        )
        intervention_results.append(
            {
                "name": spec["name"],
                "kind": spec["kind"],
                "feature_set": spec["feature_set"],
                "features": spec["features"],
                "interventions": [
                    {"layer": int(l), "pos": int(p), "feature_idx": int(f), "value": float(v)}
                    for l, p, f, v in spec["interventions"]
                ],
                "result": result,
            }
        )

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": time.time() - started,
        "config": config,
        "target_text_to_id": target_text_to_id,
        "token_ids": {
            "semantic_target": semantic_target_id,
            "observed_fallback": observed_fallback_id,
            "fallback_ids": fallback_ids,
        },
        "intervention_prompt": intervention_prompt,
        "baselines": baselines,
        "top_features": top_features,
        "interventions": intervention_results,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary)
    logging.info("Wrote %s", RESULTS_DIR / "summary.json")
    logging.info("Wrote %s", RESULTS_DIR / "report.md")


if __name__ == "__main__":
    main()
