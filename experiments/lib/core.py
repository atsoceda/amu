from __future__ import annotations

import logging
import os
import sys
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


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


def ensure_repo_on_path(root: Path) -> None:
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def setup_file_logging(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(results_dir / "run.log", mode="w"),
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


def load_gemma_scope_2_streaming(
    paths: dict[int, str],
    *,
    feature_input_hook: str,
    feature_output_hook: str,
    scan: str | list[str] | None,
    dtype: torch.dtype,
):
    """Load Gemma Scope 2 one layer at a time to avoid two full float32 copies."""
    from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
    from circuit_tracer.utils import get_default_device
    from safetensors.torch import load_file

    # Match circuit-tracer's Gemma Scope 2 loader: the path list includes one
    # terminal entry beyond the model's 18 replacement layers.
    layer_ids = list(range(max(paths)))

    device = get_default_device()
    state_dict: dict[str, torch.Tensor] = {}
    encoders = []
    encoder_biases = []
    decoder_biases = []
    thresholds = []
    skip_connections = []

    logging.info(
        "Streaming %d Gemma Scope 2 layers as %s on %s",
        len(layer_ids),
        dtype,
        device,
    )
    for layer_idx in layer_ids:
        logging.info("Loading transcoder layer %d/%d", layer_idx + 1, len(layer_ids))
        params = load_file(paths[layer_idx], device=device.type)
        encoders.append(params["w_enc"].T.to(dtype=dtype).contiguous())
        encoder_biases.append(params["b_enc"].to(dtype=dtype))
        decoder_biases.append(params["b_dec"].to(dtype=dtype))
        thresholds.append(params["threshold"].to(dtype=dtype))
        state_dict[f"W_dec.{layer_idx}"] = (
            params["w_dec"][:, layer_idx:, :].to(dtype=dtype).contiguous()
        )
        if "affine_skip_connection" in params:
            skip_connections.append(
                params["affine_skip_connection"].to(dtype=dtype)
            )
        del params
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    state_dict["W_enc"] = torch.stack(encoders)
    state_dict["b_enc"] = torch.stack(encoder_biases)
    state_dict["b_dec"] = torch.stack(decoder_biases)
    state_dict["activation_function.threshold"] = torch.stack(thresholds).unsqueeze(1)
    if skip_connections:
        state_dict["W_skip"] = torch.stack(skip_connections)

    n_layers, d_transcoder, d_model = state_dict["W_enc"].shape
    with torch.device("meta"):
        transcoder = CrossLayerTranscoder(
            n_layers,
            d_transcoder,
            d_model,
            activation_function="jump_relu",
            skip_connection="W_skip" in state_dict,
            lazy_decoder=False,
            lazy_encoder=False,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
            scan=scan,
            dtype=dtype,
        )
    transcoder.load_state_dict(state_dict, assign=True)
    return transcoder


def load_replacement_model(config: dict[str, Any]):
    from circuit_tracer import ReplacementModel
    from circuit_tracer.utils.hf_utils import (
        load_transcoder_from_hub,
        load_transcoders,
        resolve_transcoder_paths,
    )
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
        if config.get("stream_transcoder_load", False):
            paths = resolve_transcoder_paths(transcoder_config)
            transcoder = load_gemma_scope_2_streaming(
                paths,
                feature_input_hook=transcoder_config["feature_input_hook"],
                feature_output_hook=transcoder_config["feature_output_hook"],
                scan=transcoder_config["scan"],
                dtype=dtype,
            )
        else:
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


def logits_for_prompt(
    model,
    prompt: str,
    target_ids: list[int],
    *,
    top_k: int = 20,
    return_activations: bool = True,
) -> dict[str, Any]:
    logits, activations = model.feature_intervention(
        prompt,
        interventions=[],
        freeze_attention=False,
        sparse=True,
        return_activations=return_activations,
    )
    next_logits = logits[0, -1].detach().float().cpu()
    probs = torch.softmax(next_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=top_k)
    targets = {}
    for token_id in target_ids:
        targets[str(token_id)] = {
            "token_id": int(token_id),
            "token": decode_token(model.tokenizer, token_id),
            "logit": float(next_logits[token_id]),
            "prob": float(probs[token_id]),
            "rank": int((next_logits > next_logits[token_id]).sum().item() + 1),
        }
    result = {
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
        "targets": targets,
    }
    if return_activations:
        result["activation_nnz"] = (
            int(activations._nnz()) if activations is not None and activations.is_sparse else None
        )
    return result


def run_graph(
    model,
    prompt: str,
    prompt_id: str,
    target_ids: list[int],
    config: dict[str, Any],
    graphs_dir: Path,
):
    from circuit_tracer import attribute

    graphs_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graphs_dir / f"{prompt_id}.pt"
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
        offload=config.get("offload"),
    )
    graph.to_pt(str(graph_path))
    return graph


def feature_effect_map(graph, target_id: int) -> dict[tuple[int, int, int], dict[str, Any]]:
    """Map selected (layer, position, feature) nodes to one target's direct effects."""
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


def graph_features_for_target(
    graph,
    tokenizer,
    prompt_id: str,
    target_id: int,
    top_n: int,
) -> list[CandidateFeature]:
    n_features = int(len(graph.selected_features))
    target_ids = [int(t.vocab_idx) for t in graph.logit_targets]
    if target_id not in target_ids:
        return []
    target_offset = target_ids.index(target_id)
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
    top_probs, top_ids = torch.topk(probs, k=min(10, probs.numel()))
    targets = {}
    for token_id in target_ids:
        base = baseline["targets"][str(token_id)]
        rank = int((next_logits > next_logits[token_id]).sum().item() + 1)
        targets[str(token_id)] = {
            "token_id": int(token_id),
            "token": decode_token(model.tokenizer, token_id),
            "logit": float(next_logits[token_id]),
            "prob": float(probs[token_id]),
            "rank": rank,
            "delta_logit": float(next_logits[token_id] - base["logit"]),
            "delta_prob": float(probs[token_id] - base["prob"]),
            "delta_rank": int(base["rank"] - rank),
        }
    return {
        "targets": targets,
        "top_tokens": [
            {
                "rank": rank,
                "token_id": int(token_id),
                "token": decode_token(model.tokenizer, int(token_id)),
                "logit": float(next_logits[int(token_id)]),
                "prob": float(prob),
            }
            for rank, (prob, token_id) in enumerate(
                zip(top_probs, top_ids),
                start=1,
            )
        ],
    }


def dict_intervention_result(
    model,
    prompt: str,
    interventions: list[dict[str, Any]],
    target_ids: list[int],
    baseline: dict[str, Any],
    *,
    filter_to_prompt_length: bool = False,
) -> dict[str, Any]:
    applied_interventions = interventions
    if filter_to_prompt_length:
        token_count = len(model.tokenizer(prompt, add_special_tokens=True).input_ids)
        applied_interventions = [
            item for item in interventions
            if 0 <= int(item["pos"]) < token_count
        ]
    tuples = [
        (int(item["layer"]), int(item["pos"]), int(item["feature_idx"]), float(item["value"]))
        for item in applied_interventions
    ]
    result = intervention_result(model, prompt, tuples, target_ids, baseline)
    if filter_to_prompt_length:
        result.update(
            {
                "intervention_count": len(interventions),
                "applied_intervention_count": len(applied_interventions),
                "skipped_intervention_count": len(interventions) - len(applied_interventions),
            }
        )
    return result
