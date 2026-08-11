#!/usr/bin/env python3
"""Trace S1-induced CLT activation changes into the fixed article position."""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.lib.aan_protocol import build_amplify_interventions, write_json
from experiments.lib.core import load_replacement_model, setup_file_logging, token_id_for_text


EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve(config: dict[str, Any], key: str) -> Path:
    return (EXP_DIR / str(config[key])).resolve()


def load_features(config: dict[str, Any]) -> list[dict[str, Any]]:
    selection = load_json(resolve(config, "e1_selection_path"))
    return [
        {
            "layer": int(item["layer"]),
            "feature_idx": int(item["feature_idx"]),
            "mean_activation": float(item.get("mean_activation", 0.0)),
            "label": str(item.get("label", "")),
        }
        for item in selection["sets"][config["feature_set"]]["selected_features"]
    ]


def tuples(interventions: list[dict[str, Any]]) -> list[tuple[int, int, int, float]]:
    return [
        (
            int(item["layer"]),
            int(item["pos"]),
            int(item["feature_idx"]),
            float(item["value"]),
        )
        for item in interventions
    ]


@torch.no_grad()
def run_with_activations(
    model,
    prompt: str,
    interventions: list[dict[str, Any]],
    *,
    freeze_attention: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, activations = model.feature_intervention(
        prompt,
        interventions=tuples(interventions),
        freeze_attention=freeze_attention,
        sparse=True,
        return_activations=True,
    )
    if activations is None:
        raise RuntimeError("Expected activation cache")
    return logits[0, -1].detach().float().cpu(), activations.detach().float().cpu()


def sparse_row(activations: torch.Tensor, layer: int, position: int) -> dict[int, float]:
    if activations.is_sparse:
        coalesced = activations.coalesce()
        indices = coalesced.indices()
        values = coalesced.values()
        mask = (indices[0] == layer) & (indices[1] == position)
        selected_indices = indices[2, mask].tolist()
        selected_values = values[mask].tolist()
        return {
            int(feature_idx): float(value)
            for feature_idx, value in zip(selected_indices, selected_values)
        }
    vector = activations[layer, position]
    nonzero = torch.nonzero(vector, as_tuple=False).view(-1)
    return {int(index): float(vector[index]) for index in nonzero.tolist()}


def row_delta(left: dict[int, float], right: dict[int, float]) -> dict[str, Any]:
    keys = set(left) | set(right)
    deltas = {key: right.get(key, 0.0) - left.get(key, 0.0) for key in keys}
    changed = [value for value in deltas.values() if value != 0.0]
    dot = sum(left.get(key, 0.0) * right.get(key, 0.0) for key in keys)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return {
        "off_nnz": len(left),
        "on_nnz": len(right),
        "changed_feature_count": len(changed),
        "delta_l1": sum(abs(value) for value in changed),
        "delta_l2": math.sqrt(sum(value * value for value in changed)),
        "off_l2": left_norm,
        "on_l2": right_norm,
        "cosine": (
            dot / (left_norm * right_norm)
            if left_norm > 0.0 and right_norm > 0.0
            else None
        ),
        "largest_deltas": [
            {"feature_idx": key, "delta": value}
            for key, value in sorted(
                deltas.items(),
                key=lambda item: -abs(item[1]),
            )[:10]
        ],
    }


def distribution_shift(on_logits: torch.Tensor, off_logits: torch.Tensor) -> dict[str, float]:
    p = torch.softmax(on_logits, dim=-1)
    q = torch.softmax(off_logits, dim=-1)
    midpoint = 0.5 * (p + q)
    log_p = torch.log(p.clamp_min(torch.finfo(p.dtype).tiny))
    log_q = torch.log(q.clamp_min(torch.finfo(q.dtype).tiny))
    log_m = torch.log(midpoint.clamp_min(torch.finfo(midpoint.dtype).tiny))
    js = 0.5 * torch.sum(p * (log_p - log_m)) + 0.5 * torch.sum(
        q * (log_q - log_m)
    )
    return {
        "js": float(js),
        "tv": float(0.5 * torch.sum(torch.abs(p - q))),
        "max_abs_logit_delta": float(torch.max(torch.abs(on_logits - off_logits))),
    }


def summarize(rows: list[dict[str, Any]], n_layers: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in sorted({row["attention_mode"] for row in rows}):
        mode_rows = [row for row in rows if row["attention_mode"] == mode]
        output[mode] = {
            "n": len(mode_rows),
            "mean_output_tv": sum(row["output_shift"]["tv"] for row in mode_rows)
            / len(mode_rows),
            "mean_output_js": sum(row["output_shift"]["js"] for row in mode_rows)
            / len(mode_rows),
            "layers": {},
        }
        for layer in range(n_layers):
            values = [row["article_position_layers"][str(layer)] for row in mode_rows]
            output[mode]["layers"][str(layer)] = {
                "mean_delta_l2": sum(item["delta_l2"] for item in values)
                / len(values),
                "mean_delta_l1": sum(item["delta_l1"] for item in values)
                / len(values),
                "mean_changed_feature_count": sum(
                    item["changed_feature_count"] for item in values
                )
                / len(values),
            }
    return output


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# S1 propagation diagnostics",
        "",
        f"Generated: {summary['generated_at']}",
        f"Runtime: {summary['elapsed_sec']:.1f}s",
        f"Forced article: `{summary['forced_article']}`",
        "",
        "S1 is applied at the pre-article position. Sparse CLT activation changes "
        "are measured one token later, at the fixed article position.",
        "",
    ]
    for mode, block in summary["analysis"].items():
        lines.extend(
            [
                f"## Attention {mode}",
                "",
                f"- Mean noun-distribution TV: {block['mean_output_tv']:.4f}",
                f"- Mean noun-distribution JS: {block['mean_output_js']:.6f}",
                "",
                "| Layer | Mean activation ΔL2 | Mean changed features |",
                "| ---: | ---: | ---: |",
            ]
        )
        for layer, values in block["layers"].items():
            lines.append(
                f"| {layer} | {values['mean_delta_l2']:.3f} | "
                f"{values['mean_changed_feature_count']:.1f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    config = load_json(CONFIG_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR)
    started = time.time()

    e1_config = load_json(resolve(config, "e1_config_path"))
    all_examples = list(e1_config["test_examples"])
    examples = [all_examples[int(index) - 1] for index in config["example_indices"]]
    features = load_features(config)
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    article_id = token_id_for_text(tokenizer, f" {config['forced_article']}")
    article_text = tokenizer.decode([article_id])
    n_layers = int(model.cfg.n_layers)

    rows: list[dict[str, Any]] = []
    for mode_config in config["attention_modes"]:
        mode = str(mode_config["id"])
        freeze_attention = bool(mode_config["freeze_attention"])
        for index, example in zip(config["example_indices"], examples):
            prompt = f"{e1_config['demonstration']} {example['sentence']}"
            intervention_position = len(
                tokenizer(prompt, add_special_tokens=True).input_ids
            ) - 1
            interventions, activation_rows = build_amplify_interventions(
                model,
                prompt,
                intervention_position,
                features,
                float(config["amplify_factor"]),
            )
            fixed_prompt = prompt + article_text
            article_position = len(
                tokenizer(fixed_prompt, add_special_tokens=True).input_ids
            ) - 1
            off_logits, off_activations = run_with_activations(
                model,
                fixed_prompt,
                [],
                freeze_attention=freeze_attention,
            )
            on_logits, on_activations = run_with_activations(
                model,
                fixed_prompt,
                interventions,
                freeze_attention=freeze_attention,
            )

            layer_deltas: dict[str, Any] = {}
            for layer in range(n_layers):
                off_row = sparse_row(off_activations, layer, article_position)
                on_row = sparse_row(on_activations, layer, article_position)
                delta = row_delta(off_row, on_row)
                delta["s1_feature_values"] = [
                    {
                        "feature_layer": feature["layer"],
                        "feature_idx": feature["feature_idx"],
                        "off": off_row.get(int(feature["feature_idx"]), 0.0)
                        if int(feature["layer"]) == layer
                        else None,
                        "on": on_row.get(int(feature["feature_idx"]), 0.0)
                        if int(feature["layer"]) == layer
                        else None,
                    }
                    for feature in features
                    if int(feature["layer"]) == layer
                ]
                layer_deltas[str(layer)] = delta

            rows.append(
                {
                    "example_index": int(index),
                    "sentence": example["sentence"],
                    "attention_mode": mode,
                    "freeze_attention": freeze_attention,
                    "intervention_position": intervention_position,
                    "article_position": article_position,
                    "interventions": activation_rows,
                    "output_shift": distribution_shift(on_logits, off_logits),
                    "article_position_layers": layer_deltas,
                }
            )
            logging.info(
                "%s example=%s TV=%.4f L12Δ=%.3f",
                mode,
                index,
                rows[-1]["output_shift"]["tv"],
                layer_deltas["12"]["delta_l2"],
            )
            del off_activations, on_activations

    summary = {
        "experiment": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - started,
        "model": config["model"],
        "forced_article": config["forced_article"],
        "features": features,
        "example_indices": config["example_indices"],
        "analysis": summarize(rows, n_layers),
    }
    write_json(RESULTS_DIR / "rows.json", rows)
    write_json(RESULTS_DIR / "summary.json", summary)
    write_report(summary, RESULTS_DIR / "report.md")


if __name__ == "__main__":
    main()
