#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.lib.aan_protocol import token_id_for_text, write_json
from experiments.lib.core import load_replacement_model, setup_file_logging
from experiments.lib.mediation_estimands import total_variation_from_logits
from experiments.pre_article_public_private_factorial.run import (
    capture_residuals,
    first_token_id,
    patched_logits,
    raw_logits,
    stats,
)


EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def interval(values: list[float], seed: int, resamples: int) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(values)
    boot = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    boot.sort()
    return {
        "n": n,
        "mean": sum(values) / n,
        "lo": boot[math.floor(.025 * (len(boot) - 1))],
        "hi": boot[math.ceil(.975 * (len(boot) - 1))],
        "method": "pair-level nonparametric bootstrap",
        "resamples": resamples,
    }


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = left.norm() * right.norm()
    return float(torch.dot(left, right) / denom) if float(denom) else 0.0


def occupation_ids(tokenizer, path: Path) -> list[int]:
    ids = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["sentence"].startswith("Someone who"):
                ids.add(first_token_id(tokenizer, row["word"].strip().lower()))
    return sorted(ids)


def generation(model, prompt: str, position: int, layer: int | None, replacement: torch.Tensor | None):
    article_logits = raw_logits(model, prompt) if replacement is None else patched_logits(model, prompt, position, int(layer), replacement)
    article_id = int(torch.argmax(article_logits))
    piece = model.tokenizer.decode([article_id])
    noun_prompt = prompt + piece
    noun_logits = raw_logits(model, noun_prompt) if replacement is None else patched_logits(model, noun_prompt, position, int(layer), replacement)
    return {"article_id": article_id, "article": piece.strip(), "article_logits": article_logits, "noun_logits": noun_logits}


def main() -> None:
    config = load(EXP_DIR / "config.json")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    layers = [int(x) for x in config["patch_layers"]]
    demo = config["demonstration"]
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    occ_ids = occupation_ids(tokenizer, (EXP_DIR / config["dataset_path"]).resolve())

    dev_rows: list[dict[str, Any]] = []
    for pair in [p for p in config["pairs"] if p["split"] == "dev"]:
        source_prompt = f"{demo} {pair['source_sentence']}"
        target_prompt = f"{demo} {pair['target_sentence']}"
        source_pos = len(tokenizer(source_prompt, add_special_tokens=True).input_ids) - 1
        target_pos = len(tokenizer(target_prompt, add_special_tokens=True).input_ids) - 1
        source_states = capture_residuals(model, source_prompt, source_pos, layers)
        target_states = capture_residuals(model, target_prompt, target_pos, layers)
        source_id = first_token_id(tokenizer, pair["source_word"])
        target_id = first_token_id(tokenizer, pair["target_word"])
        article_id = a_id if pair["source_article"] == "a" else an_id
        prefix = source_prompt + tokenizer.decode([article_id])
        baseline = raw_logits(model, prefix)
        baseline_contrast = float(baseline[target_id] - baseline[source_id])
        for layer in layers:
            patched = patched_logits(model, prefix, source_pos, layer, target_states[layer])
            dev_rows.append({
                "pair_id": pair["id"],
                "regime": pair["regime"],
                "layer": layer,
                "delta_delta": float(patched[target_id] - patched[source_id]) - baseline_contrast,
            })

    layer_means = {
        layer: sum(r["delta_delta"] for r in dev_rows if r["layer"] == layer) / sum(r["layer"] == layer for r in dev_rows)
        for layer in layers
    }
    selected_layer = max(layers, key=lambda layer: layer_means[layer])

    test_pairs = [p for p in config["pairs"] if p["split"] == "test"]
    caches: dict[str, dict[str, Any]] = {}
    for pair in test_pairs:
        source_prompt = f"{demo} {pair['source_sentence']}"
        target_prompt = f"{demo} {pair['target_sentence']}"
        source_pos = len(tokenizer(source_prompt, add_special_tokens=True).input_ids) - 1
        target_pos = len(tokenizer(target_prompt, add_special_tokens=True).input_ids) - 1
        caches[pair["id"]] = {
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "source_pos": source_pos,
            "target_pos": target_pos,
            "source_state": capture_residuals(model, source_prompt, source_pos, [selected_layer])[selected_layer],
            "target_state": capture_residuals(model, target_prompt, target_pos, [selected_layer])[selected_layer],
            "source_id": first_token_id(tokenizer, pair["source_word"]),
            "target_id": first_token_id(tokenizer, pair["target_word"]),
        }

    wrong_target: dict[str, torch.Tensor] = {}
    for regime in ("between", "within"):
        group = [p for p in test_pairs if p["regime"] == regime]
        for idx, pair in enumerate(group):
            wrong_target[pair["id"]] = caches[group[(idx + 1) % len(group)]["id"]]["target_state"]

    rows: list[dict[str, Any]] = []
    for pair in test_pairs:
        cache = caches[pair["id"]]
        source_prompt = cache["source_prompt"]
        source_pos = cache["source_pos"]
        source_id = cache["source_id"]
        target_id = cache["target_id"]
        replacements = {
            "target": cache["target_state"],
            "wrong_target": wrong_target[pair["id"]],
            "sign_reversed": 2 * cache["source_state"] - cache["target_state"],
        }
        off_free = generation(model, source_prompt, source_pos, None, None)
        condition_free = {
            name: generation(model, source_prompt, source_pos, selected_layer, replacement)
            for name, replacement in replacements.items()
        }
        forced: dict[str, dict[str, torch.Tensor]] = {"off": {}}
        for article, article_id in (("a", a_id), ("an", an_id)):
            prefix = source_prompt + tokenizer.decode([article_id])
            forced["off"][article] = raw_logits(model, prefix)
            for name, replacement in replacements.items():
                forced.setdefault(name, {})[article] = patched_logits(model, prefix, source_pos, selected_layer, replacement)

        treated = condition_free["target"]
        treated_piece = tokenizer.decode([treated["article_id"]])
        replay_logits = raw_logits(model, source_prompt + treated_piece)
        total = torch.softmax(treated["noun_logits"], -1) - torch.softmax(off_free["noun_logits"], -1)
        mediator = torch.softmax(replay_logits, -1) - torch.softmax(off_free["noun_logits"], -1)
        residual = torch.softmax(treated["noun_logits"], -1) - torch.softmax(replay_logits, -1)
        source_article = pair["source_article"]
        baseline = forced["off"][source_article]
        base_stats = stats(baseline, source_id, target_id, tokenizer)
        target_stats = stats(forced["target"][source_article], source_id, target_id, tokenizer)
        control_stats = {
            name: stats(forced[name][source_article], source_id, target_id, tokenizer)
            for name in ("wrong_target", "sign_reversed")
        }
        baseline_probs = torch.softmax(baseline, -1)
        target_probs = torch.softmax(forced["target"][source_article], -1)

        target_prompt = cache["target_prompt"]
        target_article_id = a_id if pair["target_article"] == "a" else an_id
        target_prefix = target_prompt + tokenizer.decode([target_article_id])
        reverse_baseline = raw_logits(model, target_prefix)
        reverse_patched = patched_logits(model, target_prefix, cache["target_pos"], selected_layer, cache["source_state"])
        reverse_before = float(reverse_baseline[source_id] - reverse_baseline[target_id])
        reverse_after = float(reverse_patched[source_id] - reverse_patched[target_id])

        rows.append({
            "pair_id": pair["id"], "regime": pair["regime"],
            "source_word": pair["source_word"], "target_word": pair["target_word"],
            "source_article": pair["source_article"], "target_article": pair["target_article"],
            "off_free_article": off_free["article"], "target_free_article": treated["article"],
            "article_changed": off_free["article_id"] != treated["article_id"],
            "total_tv": float(.5 * total.abs().sum()),
            "mediator_tv": float(.5 * mediator.abs().sum()),
            "residual_tv": float(.5 * residual.abs().sum()),
            "mediator_total_cosine": cosine(mediator, total),
            "residual_total_cosine": cosine(residual, total),
            "reconstruction_l1": float((total - mediator - residual).abs().sum()),
            "baseline": base_stats, "target_patch": target_stats,
            "target_delta_delta": target_stats["target_minus_source"] - base_stats["target_minus_source"],
            "target_logit_change": target_stats["target_logit"] - base_stats["target_logit"],
            "source_logit_change": target_stats["source_logit"] - base_stats["source_logit"],
            "target_prob_change": target_stats["target_prob"] - base_stats["target_prob"],
            "source_prob_change": target_stats["source_prob"] - base_stats["source_prob"],
            "occupation_mass_change": float(target_probs[occ_ids].sum() - baseline_probs[occ_ids].sum()),
            "wrong_target_delta_delta": control_stats["wrong_target"]["target_minus_source"] - base_stats["target_minus_source"],
            "sign_reversed_delta_delta": control_stats["sign_reversed"]["target_minus_source"] - base_stats["target_minus_source"],
            "reverse_delta_delta": reverse_after - reverse_before,
            "fixed_a_tv": total_variation_from_logits(forced["target"]["a"], forced["off"]["a"]),
            "fixed_an_tv": total_variation_from_logits(forced["target"]["an"], forced["off"]["an"]),
        })

    seed = int(config["bootstrap_seed"])
    resamples = int(config["bootstrap_resamples"])
    metrics = (
        "total_tv", "mediator_tv", "residual_tv", "mediator_total_cosine", "residual_total_cosine",
        "target_delta_delta", "target_logit_change", "source_logit_change", "target_prob_change",
        "source_prob_change", "occupation_mass_change", "wrong_target_delta_delta",
        "sign_reversed_delta_delta", "reverse_delta_delta", "fixed_a_tv", "fixed_an_tv",
    )
    regime_summary: dict[str, Any] = {}
    for regime_index, regime in enumerate(("between", "within")):
        group = [r for r in rows if r["regime"] == regime]
        regime_summary[regime] = {
            metric: interval([float(r[metric]) for r in group], seed + regime_index * 100 + idx, resamples)
            for idx, metric in enumerate(metrics)
        }
        regime_summary[regime]["n_pairs"] = len(group)
        regime_summary[regime]["article_change_rate"] = sum(r["article_changed"] for r in group) / len(group)
        regime_summary[regime]["target_top10_before"] = sum(r["baseline"]["target_rank"] <= 10 for r in group) / len(group)
        regime_summary[regime]["target_top10_after"] = sum(r["target_patch"]["target_rank"] <= 10 for r in group) / len(group)

    summary = {
        "experiment": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - started,
        "selected_layer": selected_layer,
        "layer_selection": "maximum pooled mean target-minus-source change on eight disjoint development pairs",
        "dev_layer_means": {str(k): v for k, v in layer_means.items()},
        "regimes": regime_summary,
        "control_definitions": {
            "wrong_target": "cyclic natural target residual from another held-out pair in the same regime",
            "sign_reversed": "source residual minus the natural target-minus-source direction",
            "reverse": "source residual patched into the target prompt; positive values favor source over target",
        },
        "prompt_scope": "Natural occupation descriptions only; target nouns are never named in prompts.",
    }
    write_json(RESULTS_DIR / "dev_rows.json", dev_rows)
    write_json(RESULTS_DIR / "rows.json", rows)
    write_json(RESULTS_DIR / "summary.json", summary)
    lines = ["# Natural residual carrier regimes", "", f"Selected layer: {selected_layer}", "", "| Regime | N | Article change | Total TV | Mediator TV | Residual TV | Target ΔΔ | Wrong-target ΔΔ |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for regime in ("between", "within"):
        block = regime_summary[regime]
        lines.append(f"| {regime} | {block['n_pairs']} | {block['article_change_rate']:.2f} | {block['total_tv']['mean']:.3f} | {block['mediator_tv']['mean']:.3f} | {block['residual_tv']['mean']:.3f} | {block['target_delta_delta']['mean']:.3f} | {block['wrong_target_delta_delta']['mean']:.3f} |")
    lines.extend(["", "All target/source logit and probability changes, occupation-vocabulary mass, sign-reversed controls, reverse patches, and pair-bootstrap intervals are stored in `summary.json` and `rows.json`.", ""])
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
