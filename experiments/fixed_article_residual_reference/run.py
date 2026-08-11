#!/usr/bin/env python3
"""Held-out upstream positive control for fixed-article noun transmission.

For each same-sentence source/target pair, capture the decoder-layer output
residual at the fixed article-token position from a target-cued run. Replace
the source run's residual at one layer, sweep layers on a development split,
then evaluate the selected layer against matched-norm random directions on a
held-out test split.

This is a full-residual reference, not an oracle and not a sparse feature
intervention.
"""
from __future__ import annotations

import json
import logging
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from nnsight import save

from experiments.lib.aan_protocol import article_and_word, write_json
from experiments.lib.core import load_replacement_model, setup_file_logging, token_id_for_text


EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def word_token_ids(tokenizer, word: str) -> list[int]:
    return [int(x) for x in tokenizer(f" {word}", add_special_tokens=False).input_ids]


def first_token_id(tokenizer, word: str) -> int:
    ids = word_token_ids(tokenizer, word)
    if not ids:
        raise ValueError(f"No tokenization for {word!r}")
    return ids[0]


@torch.no_grad()
def raw_logits(model, prompt: str) -> torch.Tensor:
    with model.trace(prompt):
        logits = save(model.output.logits)
    return logits[0, -1].detach().float().cpu()


@torch.no_grad()
def capture_layer_residuals(
    model,
    prompt: str,
    *,
    position: int,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    activations: dict[int, torch.Tensor] = {}
    # NNSight's tracing context does not reliably preserve containers created
    # inside the context, so capture each requested layer in its own trace.
    for layer in layers:
        with model.trace(prompt):
            layer_loc = getattr(model.pre_logit_location, "layers")[layer]
            activation = save(layer_loc.output[0])
        activations[layer] = activation[0, position].detach().float().cpu()
    return activations


@torch.no_grad()
def patch_layer_residual(
    model,
    prompt: str,
    *,
    position: int,
    layer: int,
    replacement: torch.Tensor,
) -> torch.Tensor:
    with model.trace(prompt):
        layer_loc = getattr(model.pre_logit_location, "layers")[layer]
        value = replacement.to(
            device=layer_loc.output[0].device,
            dtype=layer_loc.output[0].dtype,
        )
        layer_loc.output[0][:, position, :] = value
        logits = save(model.output.logits)
    return logits[0, -1].detach().float().cpu()


def random_matched_replacement(
    source: torch.Tensor,
    target: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    delta = target - source
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_delta = torch.randn(delta.shape, generator=generator, dtype=delta.dtype)
    random_delta = random_delta / random_delta.norm().clamp_min(1e-12)
    return source + random_delta * delta.norm()


def target_source_stats(
    logits: torch.Tensor,
    source_id: int,
    target_id: int,
) -> dict[str, Any]:
    probs = torch.softmax(logits, dim=-1)
    top_id = int(torch.argmax(logits).item())
    return {
        "source_logit": float(logits[source_id]),
        "target_logit": float(logits[target_id]),
        "target_minus_source": float(logits[target_id] - logits[source_id]),
        "source_prob": float(probs[source_id]),
        "target_prob": float(probs[target_id]),
        "source_rank": int((logits > logits[source_id]).sum().item() + 1),
        "target_rank": int((logits > logits[target_id]).sum().item() + 1),
        "top_id": top_id,
    }


def percentile_interval(values: list[float], seed: int, n_resamples: int = 10000) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = random.Random(seed)
    n = len(values)
    boot = []
    for _ in range(n_resamples):
        boot.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    boot.sort()
    return {
        "n": n,
        "mean": sum(values) / n,
        "lo": boot[math.floor(0.025 * (len(boot) - 1))],
        "hi": boot[math.ceil(0.975 * (len(boot) - 1))],
        "method": "pair-level nonparametric bootstrap",
        "resamples": n_resamples,
    }


def simple_generation(model, prompt: str, max_new_tokens: int = 4) -> str:
    current = prompt
    ids: list[int] = []
    for _ in range(max_new_tokens):
        logits = raw_logits(model, current)
        token_id = int(torch.argmax(logits).item())
        ids.append(token_id)
        piece = model.tokenizer.decode([token_id])
        current += piece
        if piece.strip() in {".", "!", "?"}:
            break
    return model.tokenizer.decode(ids)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Fixed-article full-residual reference",
        "",
        f"Generated: {summary['generated_at']}",
        f"Runtime: {summary['elapsed_sec']:.1f}s",
        f"Development-selected layer: {summary['selected_layer']}",
        "",
        "This is an upstream full-residual reference intervention, not an oracle.",
        "",
        "## Development layer sweep",
        "",
        "| Layer | Mean target-specific ΔΔ |",
        "| ---: | ---: |",
    ]
    for layer, block in summary["dev_layer_summary"].items():
        lines.append(f"| {layer} | {block['mean']:.3f} |")
    target = summary["heldout_target_patch"]
    random_block = summary["heldout_random_controls"]
    lines.extend(
        [
            "",
            "## Held-out test",
            "",
            f"- Target patch mean ΔΔ: {target['mean']:.3f} "
            f"[{target['lo']:.3f}, {target['hi']:.3f}] over {target['n']} pairs.",
            f"- Random matched-norm mean ΔΔ: {random_block['mean']:.3f} "
            f"[{random_block['lo']:.3f}, {random_block['hi']:.3f}] after averaging "
            "seeds within each pair.",
            f"- Assay sensitivity validated: **{summary['assay_validated']}**.",
            "",
            "Validation requires a positive held-out target-specific effect exceeding "
            "the configured minimum and the mean matched-norm random effect.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    config = load_json(CONFIG_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR)
    started = time.time()

    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    article_id = token_id_for_text(tokenizer, f" {config['article']}")
    article_text = tokenizer.decode([article_id])
    layers = [int(x) for x in config["patch_layers"]]

    rows: list[dict[str, Any]] = []
    pair_cache: dict[str, dict[str, Any]] = {}
    for pair in config["pairs"]:
        source_prompt = config["paired_prompt_template"].format(
            cue=pair["source_word"],
            sentence=pair["sentence"],
        )
        target_prompt = config["paired_prompt_template"].format(
            cue=pair["target_word"],
            sentence=pair["sentence"],
        )
        source_prefix = source_prompt + article_text
        target_prefix = target_prompt + article_text
        source_position = len(
            tokenizer(source_prefix, add_special_tokens=True).input_ids
        ) - 1
        target_position = len(
            tokenizer(target_prefix, add_special_tokens=True).input_ids
        ) - 1
        source_id = first_token_id(tokenizer, pair["source_word"])
        target_id = first_token_id(tokenizer, pair["target_word"])

        baseline_logits = raw_logits(model, source_prefix)
        hinted_logits = raw_logits(model, target_prefix)
        baseline_free = simple_generation(model, source_prompt)
        target_hint_free = simple_generation(model, target_prompt)
        baseline_stats = target_source_stats(baseline_logits, source_id, target_id)
        hinted_stats = target_source_stats(hinted_logits, source_id, target_id)
        source_acts = capture_layer_residuals(
            model,
            source_prefix,
            position=source_position,
            layers=layers,
        )
        target_acts = capture_layer_residuals(
            model,
            target_prefix,
            position=target_position,
            layers=layers,
        )
        pair_cache[pair["id"]] = {
            "source_prefix": source_prefix,
            "source_position": source_position,
            "source_id": source_id,
            "target_id": target_id,
            "source_acts": source_acts,
            "target_acts": target_acts,
            "baseline_stats": baseline_stats,
        }

        for layer in layers:
            patched_logits = patch_layer_residual(
                model,
                source_prefix,
                position=source_position,
                layer=layer,
                replacement=target_acts[layer],
            )
            patched_stats = target_source_stats(patched_logits, source_id, target_id)
            row = {
                "pair_id": pair["id"],
                "split": pair["split"],
                "sentence": pair["sentence"],
                "source_word": pair["source_word"],
                "target_word": pair["target_word"],
                "source_token_ids": word_token_ids(tokenizer, pair["source_word"]),
                "target_token_ids": word_token_ids(tokenizer, pair["target_word"]),
                "layer": layer,
                "baseline": baseline_stats,
                "target_hinted": hinted_stats,
                "patched": patched_stats,
                "delta_delta": patched_stats["target_minus_source"]
                - baseline_stats["target_minus_source"],
                "target_logit_delta": patched_stats["target_logit"]
                - baseline_stats["target_logit"],
                "patch_norm": float(
                    (target_acts[layer] - source_acts[layer]).norm()
                ),
                "baseline_free": baseline_free,
                "target_hint_free": target_hint_free,
            }
            rows.append(row)
            logging.info(
                "%s %s layer=%d ΔΔ=%.3f",
                pair["split"],
                pair["id"],
                layer,
                row["delta_delta"],
            )

    dev_layer_summary: dict[str, dict[str, Any]] = {}
    for layer in layers:
        values = [
            float(row["delta_delta"])
            for row in rows
            if row["split"] == "dev" and row["layer"] == layer
        ]
        dev_layer_summary[str(layer)] = percentile_interval(
            values, seed=20260810 + layer
        )
    selected_layer = max(
        layers,
        key=lambda layer: float(dev_layer_summary[str(layer)]["mean"]),
    )

    target_test_rows = [
        row
        for row in rows
        if row["split"] == "test" and row["layer"] == selected_layer
    ]
    random_rows: list[dict[str, Any]] = []
    for pair in config["pairs"]:
        if pair["split"] != "test":
            continue
        cached = pair_cache[pair["id"]]
        baseline = cached["baseline_stats"]
        for seed in config["random_seeds"]:
            replacement = random_matched_replacement(
                cached["source_acts"][selected_layer],
                cached["target_acts"][selected_layer],
                int(seed),
            )
            random_logits = patch_layer_residual(
                model,
                cached["source_prefix"],
                position=cached["source_position"],
                layer=selected_layer,
                replacement=replacement,
            )
            stats = target_source_stats(
                random_logits,
                cached["source_id"],
                cached["target_id"],
            )
            random_rows.append(
                {
                    "pair_id": pair["id"],
                    "seed": int(seed),
                    "layer": selected_layer,
                    "delta_delta": stats["target_minus_source"]
                    - baseline["target_minus_source"],
                    "target_logit_delta": stats["target_logit"]
                    - baseline["target_logit"],
                }
            )

    random_pair_means = []
    for pair in config["pairs"]:
        if pair["split"] != "test":
            continue
        pair_values = [
            float(row["delta_delta"])
            for row in random_rows
            if row["pair_id"] == pair["id"]
        ]
        random_pair_means.append(sum(pair_values) / len(pair_values))

    target_summary = percentile_interval(
        [float(row["delta_delta"]) for row in target_test_rows],
        seed=20260811,
    )
    random_summary = percentile_interval(random_pair_means, seed=20260812)
    assay_validated = bool(
        target_summary["mean"] >= float(config["minimum_validation_delta_delta"])
        and target_summary["mean"] > random_summary["mean"]
    )
    summary = {
        "experiment": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - started,
        "model": config["model"],
        "article": config["article"],
        "patch_position": config["patch_position"],
        "patch_site": config["patch_site"],
        "selected_layer": selected_layer,
        "selection_rule": "maximum mean target-specific delta-delta on fixed dev pairs",
        "dev_layer_summary": dev_layer_summary,
        "heldout_target_patch": target_summary,
        "heldout_random_controls": random_summary,
        "assay_validated": assay_validated,
        "minimum_validation_delta_delta": config["minimum_validation_delta_delta"],
    }
    write_json(RESULTS_DIR / "rows.json", rows)
    write_json(RESULTS_DIR / "random_rows.json", random_rows)
    write_json(RESULTS_DIR / "summary.json", summary)
    write_report(summary, RESULTS_DIR / "report.md")
    logging.info(
        "Done selected_layer=%d target=%.3f random=%.3f validated=%s",
        selected_layer,
        target_summary["mean"],
        random_summary["mean"],
        assay_validated,
    )


if __name__ == "__main__":
    main()
