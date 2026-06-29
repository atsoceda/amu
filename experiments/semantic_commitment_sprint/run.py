#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"


@dataclass(frozen=True)
class CandidateFeature:
    source_prompt_id: str
    target_text: str
    target_token_id: int
    target_token_repr: str
    layer: int
    pos: int
    feature_idx: int
    activation: float
    direct_effect: float
    abs_direct_effect: float


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def setup_logging() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RESULTS_DIR / "run.log", mode="w"),
        ],
    )


def patch_hf_cache(config: dict[str, Any]) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import circuit_tracer.utils.hf_utils as h

    weight_root = Path(config["transcoder_weight_snapshot"])

    def local_download_hf_uris(uris, max_workers=8):
        mapping = {}
        for uri in uris:
            info = h.parse_hf_uri(uri)
            if info.repo_id != "google/gemma-scope-2-270m-pt":
                raise ValueError(f"No local mapping for {uri}")
            path = weight_root / info.file_path
            if not path.exists():
                raise FileNotFoundError(path)
            mapping[uri] = str(path)
        return mapping

    h.download_hf_uris = local_download_hf_uris


def load_replacement_model(config: dict[str, Any]):
    from circuit_tracer import ReplacementModel
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub, load_transcoders
    import yaml

    patch_hf_cache(config)
    model_ref = config["model_snapshot"] if Path(config["model_snapshot"]).exists() else config["model"]
    dtype = getattr(torch, config.get("dtype", "float32"))

    config_snapshot = Path(config.get("transcoder_config_snapshot", ""))
    if config_snapshot.exists():
        logging.info("Loading transcoder config %s", config_snapshot)
        transcoder_config = yaml.safe_load(config_snapshot.read_text())
        transcoder_config["repo_id"] = "mwhanna/gemma-scope-2-270m-pt"
        transcoder_config["revision"] = None
        transcoder_config["subfolder"] = "clt/width_262k_l0_medium_affine"
        transcoder_config["scan"] = "mwhanna/gemma-scope-2-270m-pt//clt/width_262k_l0_medium_affine"
        transcoder = load_transcoders(
            transcoder_config,
            dtype=dtype,
            lazy_encoder=False,
            lazy_decoder=False,
        )
    else:
        logging.info("Loading transcoder %s", config["transcoder_set"])
        transcoder, _ = load_transcoder_from_hub(
            config["transcoder_set"],
            dtype=dtype,
            lazy_encoder=False,
            lazy_decoder=False,
        )
    logging.info("Loading model %s", model_ref)
    return ReplacementModel.from_pretrained_and_transcoders(
        model_ref,
        transcoder,
        dtype=dtype,
        backend=config["backend"],
    )


def token_id_for_text(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"Target text {text!r} tokenized to {ids}; expected one token")
    return int(ids[0])


def decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)])


def logits_for_prompt(model, prompt: str, target_ids: list[int]) -> dict[str, Any]:
    logits, activations = model.feature_intervention(
        prompt,
        interventions=[],
        freeze_attention=False,
        sparse=True,
        return_activations=True,
    )
    next_logits = logits[0, -1].detach().float().cpu()
    probs = torch.softmax(next_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=20)
    target = {}
    for token_id in target_ids:
        target[str(token_id)] = {
            "token_id": token_id,
            "token": decode_token(model.tokenizer, token_id),
            "logit": float(next_logits[token_id]),
            "prob": float(probs[token_id]),
            "rank": int((next_logits > next_logits[token_id]).sum().item() + 1),
        }
    return {
        "top_tokens": [
            {
                "rank": i + 1,
                "token_id": int(tok),
                "token": decode_token(model.tokenizer, int(tok)),
                "prob": float(prob),
                "logit": float(next_logits[int(tok)]),
            }
            for i, (prob, tok) in enumerate(zip(top_probs, top_ids))
        ],
        "targets": target,
        "activation_nnz": int(activations._nnz()) if activations is not None and activations.is_sparse else None,
    }


def run_graph(model, prompt: str, prompt_id: str, target_ids: list[int], config: dict[str, Any]):
    from circuit_tracer import attribute

    graph_path = GRAPHS_DIR / f"{prompt_id}.pt"
    if graph_path.exists():
        from circuit_tracer.graph import Graph

        logging.info("Loading existing graph %s", graph_path)
        return Graph.from_pt(str(graph_path))

    logging.info("Attributing %s", prompt_id)
    graph = attribute(
        prompt=prompt,
        model=model,
        attribution_targets=torch.tensor(target_ids),
        batch_size=int(config["batch_size"]),
        max_feature_nodes=int(config["max_feature_nodes"]),
        verbose=True,
        offload=None,
    )
    graph.to_pt(str(graph_path))
    return graph


def graph_features_for_target(graph, tokenizer, prompt_id: str, target_id: int, top_n: int) -> list[CandidateFeature]:
    n_features = int(len(graph.selected_features))
    target_ids = [int(t.vocab_idx) for t in graph.logit_targets]
    if target_id not in target_ids:
        return []
    target_offset = target_ids.index(target_id)
    # Graph rows are sparse features first and logit targets last. Error/embed nodes
    # occupy columns but do not get rows in the dense attribution matrix.
    row = graph.adjacency_matrix[-len(target_ids) + target_offset, :n_features].detach().float().cpu()
    if row.numel() == 0:
        return []
    values, cols = torch.topk(row.abs(), k=min(top_n, row.numel()))
    out: list[CandidateFeature] = []
    for abs_value, col in zip(values, cols):
        selected_idx = int(graph.selected_features[int(col)])
        layer, pos, feature_idx = [int(x) for x in graph.active_features[selected_idx].tolist()]
        direct_effect = float(row[int(col)])
        activation = float(graph.activation_values[selected_idx].detach().float().cpu())
        out.append(
            CandidateFeature(
                source_prompt_id=prompt_id,
                target_text=decode_token(tokenizer, target_id),
                target_token_id=target_id,
                target_token_repr=decode_token(tokenizer, target_id),
                layer=layer,
                pos=pos,
                feature_idx=feature_idx,
                activation=activation,
                direct_effect=direct_effect,
                abs_direct_effect=float(abs_value),
            )
        )
    return out


def serialize_feature(feature: CandidateFeature) -> dict[str, Any]:
    return {
        "source_prompt_id": feature.source_prompt_id,
        "target_text": feature.target_text,
        "target_token_id": feature.target_token_id,
        "target_token_repr": feature.target_token_repr,
        "layer": feature.layer,
        "pos": feature.pos,
        "feature_idx": feature.feature_idx,
        "activation": feature.activation,
        "direct_effect": feature.direct_effect,
        "abs_direct_effect": feature.abs_direct_effect,
    }


def intervention_result(
    model,
    prompt: str,
    interventions: list[tuple[int, int, int, float]],
    target_ids: list[int],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    logits, _ = model.feature_intervention(
        prompt,
        interventions=interventions,
        freeze_attention=True,
        sparse=True,
        return_activations=False,
    )
    next_logits = logits[0, -1].detach().float().cpu()
    probs = torch.softmax(next_logits, dim=-1)
    target = {}
    for token_id in target_ids:
        base = baseline["targets"][str(token_id)]
        target[str(token_id)] = {
            "token_id": token_id,
            "token": decode_token(model.tokenizer, token_id),
            "logit": float(next_logits[token_id]),
            "prob": float(probs[token_id]),
            "rank": int((next_logits > next_logits[token_id]).sum().item() + 1),
            "delta_logit": float(next_logits[token_id] - base["logit"]),
            "delta_prob": float(probs[token_id] - base["prob"]),
            "delta_rank": int(base["rank"] - ((next_logits > next_logits[token_id]).sum().item() + 1)),
        }
    return {"targets": target}


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
