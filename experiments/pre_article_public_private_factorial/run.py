#!/usr/bin/env python3
"""Cross a pre-article full-residual patch with public article identity."""
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

from experiments.lib.aan_protocol import write_json
from experiments.lib.core import load_replacement_model, setup_file_logging, token_id_for_text
from experiments.lib.mediation_estimands import total_variation_from_logits


EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def first_token_id(tokenizer, word: str) -> int:
    ids = tokenizer(f" {word}", add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"No tokenization for {word!r}")
    return int(ids[0])


@torch.no_grad()
def raw_logits(model, prompt: str) -> torch.Tensor:
    with model.trace(prompt):
        logits = save(model.output.logits)
    return logits[0, -1].detach().float().cpu()


@torch.no_grad()
def capture_residuals(model, prompt: str, position: int, layers: list[int]) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for layer in layers:
        with model.trace(prompt):
            loc = getattr(model.pre_logit_location, "layers")[layer]
            activation = save(loc.output[0])
        result[layer] = activation[0, position].detach().float().cpu()
    return result


@torch.no_grad()
def patched_logits(model, prompt: str, position: int, layer: int, replacement: torch.Tensor) -> torch.Tensor:
    with model.trace(prompt):
        loc = getattr(model.pre_logit_location, "layers")[layer]
        value = replacement.to(device=loc.output[0].device, dtype=loc.output[0].dtype)
        loc.output[0][:, position, :] = value
        logits = save(model.output.logits)
    return logits[0, -1].detach().float().cpu()


def matched_random(source: torch.Tensor, target: torch.Tensor, seed: int) -> torch.Tensor:
    delta = target - source
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_delta = torch.randn(delta.shape, generator=generator, dtype=delta.dtype)
    random_delta = random_delta / random_delta.norm().clamp_min(1e-12)
    return source + random_delta * delta.norm()


def stats(logits: torch.Tensor, source_id: int, target_id: int, tokenizer) -> dict[str, Any]:
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
        "top_token": tokenizer.decode([top_id]).strip(),
        "target_top1": top_id == target_id,
    }


def interval(values: list[float], seed: int, resamples: int) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = random.Random(seed)
    n = len(values)
    boot = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    boot.sort()
    return {
        "n": n,
        "mean": sum(values) / n,
        "lo": boot[math.floor(0.025 * (len(boot) - 1))],
        "hi": boot[math.ceil(0.975 * (len(boot) - 1))],
        "method": "pair-level nonparametric bootstrap",
        "resamples": resamples,
    }


def fmt(block: dict[str, Any]) -> str:
    return f"{block['mean']:.3f} [{block['lo']:.3f}, {block['hi']:.3f}]"


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Pre-article public/private factorial",
        "",
        f"Generated: {summary['generated_at']}",
        f"Runtime: {summary['elapsed_sec']:.1f}s",
        f"Development-selected layer: {summary['selected_layer']}",
        "",
        "| Inserted article | Target patch ΔΔ | Patch TV | Target top-1 | Random ΔΔ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for article in summary["articles"]:
        block = summary["heldout"][article]
        lines.append(
            f"| `{article}` | {fmt(block['target_delta_delta'])} | "
            f"{fmt(block['patch_tv'])} | {block['target_top1_rate']:.2f} | "
            f"{fmt(block['random_delta_delta'])} |"
        )
    lines.extend([
        "",
        f"- Public token effect without private patch: {fmt(summary['factorial']['public_effect_off'])}.",
        f"- Public token effect with private patch: {fmt(summary['factorial']['public_effect_on'])}.",
        f"- Public × private interaction ΔΔ: {fmt(summary['factorial']['interaction'])}.",
        f"- Held-out private-persistence criterion met: **{summary['private_persistence_validated']}**.",
        "",
        "The patch is a target-constructed full-residual reference at the original pre-article position; it is not evidence that S1 carries the same lexical state.",
        "",
    ])
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    config = load_json(CONFIG_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    layers = [int(x) for x in config["patch_layers"]]
    articles = [str(x) for x in config["articles"]]
    resamples = int(config["bootstrap_resamples"])

    caches: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for pair in config["pairs"]:
        source_prompt = config["paired_prompt_template"].format(cue=pair["source_word"], sentence=pair["sentence"])
        target_prompt = config["paired_prompt_template"].format(cue=pair["target_word"], sentence=pair["sentence"])
        source_position = len(tokenizer(source_prompt, add_special_tokens=True).input_ids) - 1
        target_position = len(tokenizer(target_prompt, add_special_tokens=True).input_ids) - 1
        source_acts = capture_residuals(model, source_prompt, source_position, layers)
        target_acts = capture_residuals(model, target_prompt, target_position, layers)
        source_id = first_token_id(tokenizer, pair["source_word"])
        target_id = first_token_id(tokenizer, pair["target_word"])
        baselines: dict[str, torch.Tensor] = {}
        for article in articles:
            article_id = token_id_for_text(tokenizer, f" {article}")
            prefix = source_prompt + tokenizer.decode([article_id])
            baselines[article] = raw_logits(model, prefix)
            for layer in layers:
                logits = patched_logits(model, prefix, source_position, layer, target_acts[layer])
                base_stats = stats(baselines[article], source_id, target_id, tokenizer)
                patch_stats = stats(logits, source_id, target_id, tokenizer)
                rows.append({
                    "pair_id": pair["id"], "split": pair["split"], "article": article,
                    "layer": layer, "source_word": pair["source_word"], "target_word": pair["target_word"],
                    "baseline": base_stats, "patched": patch_stats,
                    "delta_delta": patch_stats["target_minus_source"] - base_stats["target_minus_source"],
                    "patch_tv": total_variation_from_logits(logits, baselines[article]),
                    "patch_norm": float((target_acts[layer] - source_acts[layer]).norm()),
                })
        caches[pair["id"]] = {
            "source_prompt": source_prompt, "source_position": source_position,
            "source_id": source_id, "target_id": target_id, "source_acts": source_acts,
            "target_acts": target_acts, "baselines": baselines,
        }
        logging.info("captured %s", pair["id"])

    native = str(config["native_article"])
    dev_summary = {}
    for layer in layers:
        values = [float(r["delta_delta"]) for r in rows if r["split"] == "dev" and r["article"] == native and r["layer"] == layer]
        dev_summary[str(layer)] = interval(values, int(config["bootstrap_seed"]) + layer, resamples)
    selected_layer = max(layers, key=lambda layer: float(dev_summary[str(layer)]["mean"]))

    random_rows: list[dict[str, Any]] = []
    for pair in config["pairs"]:
        if pair["split"] != "test":
            continue
        cache = caches[pair["id"]]
        for article in articles:
            article_id = token_id_for_text(tokenizer, f" {article}")
            prefix = cache["source_prompt"] + tokenizer.decode([article_id])
            baseline = cache["baselines"][article]
            base_stats = stats(baseline, cache["source_id"], cache["target_id"], tokenizer)
            for seed in config["random_seeds"]:
                replacement = matched_random(cache["source_acts"][selected_layer], cache["target_acts"][selected_layer], int(seed))
                logits = patched_logits(model, prefix, cache["source_position"], selected_layer, replacement)
                random_stats = stats(logits, cache["source_id"], cache["target_id"], tokenizer)
                random_rows.append({
                    "pair_id": pair["id"], "article": article, "seed": int(seed), "layer": selected_layer,
                    "delta_delta": random_stats["target_minus_source"] - base_stats["target_minus_source"],
                    "patch_tv": total_variation_from_logits(logits, baseline),
                })

    heldout: dict[str, Any] = {}
    for article_index, article in enumerate(articles):
        selected = [r for r in rows if r["split"] == "test" and r["article"] == article and r["layer"] == selected_layer]
        random_pair_means = []
        for pair in config["pairs"]:
            if pair["split"] == "test":
                vals = [float(r["delta_delta"]) for r in random_rows if r["pair_id"] == pair["id"] and r["article"] == article]
                random_pair_means.append(sum(vals) / len(vals))
        seed = int(config["bootstrap_seed"]) + 100 + article_index * 10
        heldout[article] = {
            "target_delta_delta": interval([float(r["delta_delta"]) for r in selected], seed, resamples),
            "patch_tv": interval([float(r["patch_tv"]) for r in selected], seed + 1, resamples),
            "target_top1_rate": sum(bool(r["patched"]["target_top1"]) for r in selected) / max(len(selected), 1),
            "random_delta_delta": interval(random_pair_means, seed + 2, resamples),
        }

    test_pairs = [p for p in config["pairs"] if p["split"] == "test"]
    public_off, public_on, interactions = [], [], []
    for pair in test_pairs:
        cache = caches[pair["id"]]
        off_a = stats(cache["baselines"]["a"], cache["source_id"], cache["target_id"], tokenizer)["target_minus_source"]
        off_an = stats(cache["baselines"]["an"], cache["source_id"], cache["target_id"], tokenizer)["target_minus_source"]
        selected = {r["article"]: r for r in rows if r["pair_id"] == pair["id"] and r["split"] == "test" and r["layer"] == selected_layer}
        on_a = selected["a"]["patched"]["target_minus_source"]
        on_an = selected["an"]["patched"]["target_minus_source"]
        public_off.append(off_an - off_a)
        public_on.append(on_an - on_a)
        interactions.append((on_an - off_an) - (on_a - off_a))
    seed = int(config["bootstrap_seed"]) + 200
    factorial = {
        "public_effect_off": interval(public_off, seed, resamples),
        "public_effect_on": interval(public_on, seed + 1, resamples),
        "interaction": interval(interactions, seed + 2, resamples),
    }
    target_native = heldout[native]["target_delta_delta"]
    random_native = heldout[native]["random_delta_delta"]
    validated = bool(target_native["mean"] >= float(config["minimum_validation_delta_delta"]) and target_native["mean"] > random_native["mean"])
    summary = {
        "experiment": config["experiment_name"], "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - started, "model": config["model"], "patch_site": "decoder_layer_output_residual",
        "patch_position": "final_pre_article_prompt_token", "articles": articles, "native_article": native,
        "n_dev_pairs": sum(p["split"] == "dev" for p in config["pairs"]), "n_test_pairs": len(test_pairs),
        "selected_layer": selected_layer, "selection_rule": "maximum mean target-specific delta-delta on native-article development pairs",
        "dev_layer_summary": dev_summary, "heldout": heldout, "factorial": factorial,
        "private_persistence_validated": validated,
        "interpretation_scope": "Target-constructed full-residual capacity/reference intervention; not a natural sparse planning representation.",
    }
    write_json(RESULTS_DIR / "rows.json", rows)
    write_json(RESULTS_DIR / "random_rows.json", random_rows)
    write_json(RESULTS_DIR / "summary.json", summary)
    write_report(summary)
    logging.info("done layer=%d native_delta=%.3f random=%.3f validated=%s", selected_layer, target_native["mean"], random_native["mean"], validated)


if __name__ == "__main__":
    main()
