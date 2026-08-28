#!/usr/bin/env python3
from __future__ import annotations

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
from experiments.six_cell_family_sweep.run import activations_at_position, build_interventions, greedy_generate, next_logits, top1_word, word_token_ids


EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def interval(values: list[float], seed: int, resamples: int) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(values)
    boot = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    boot.sort()
    return {"n": n, "mean": sum(values) / n, "lo": boot[math.floor(.025*(len(boot)-1))], "hi": boot[math.ceil(.975*(len(boot)-1))], "method": "prompt-level nonparametric bootstrap", "resamples": resamples}


def legal(article: str, word: str) -> bool:
    return article in {"a", "an"} and bool(word) and ((article == "an") == (word[0] in "aeiou"))


def main() -> None:
    config = load(EXP_DIR / "config.json")
    e1_config = load((EXP_DIR / config["e1_config_path"]).resolve())
    selection = load((EXP_DIR / config["e1_selection_path"]).resolve())
    features = selection["sets"]["S2_article_only"]["selected_features"]
    conditions = [{"id": "S2_full", "features": features}]
    for feature in features:
        label = f"L{feature['layer']}_F{feature['feature_idx']}"
        conditions.append({"id": f"single_{label}", "features": [feature]})
        conditions.append({"id": f"loo_{label}", "features": [f for f in features if f is not feature]})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    an_id = token_id_for_text(tokenizer, " an")
    rows = []
    for index, example in enumerate(e1_config["test_examples"], start=1):
        prompt = f"{config['demonstration']} {example['sentence']}"
        position = len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1
        activations = activations_at_position(model, prompt, position)
        prefix = prompt + tokenizer.decode([an_id])
        baseline_logits = next_logits(model, prefix, [])
        listed_ids = word_token_ids(tokenizer, example.get("listed_word", ""))
        twin_ids = word_token_ids(tokenizer, example.get("twin_word", ""))
        for condition in conditions:
            interventions, activation_rows = build_interventions(activations, position, condition["features"], float(config["amplify_factor"]))
            free = greedy_generate(model, prompt, interventions, max_new_tokens=int(config["max_new_tokens"]), top_k=int(config["top_k"]))
            logits = next_logits(model, prefix, interventions)
            twin_delta = None
            if listed_ids and twin_ids:
                off = float(baseline_logits[twin_ids[0]] - baseline_logits[listed_ids[0]])
                on = float(logits[twin_ids[0]] - logits[listed_ids[0]])
                twin_delta = on - off
            rows.append({
                "index": index, "sentence": example["sentence"], "condition": condition["id"],
                "features": [{"layer": int(f["layer"]), "feature_idx": int(f["feature_idx"])} for f in condition["features"]],
                "feature_activations": activation_rows,
                "free_continuation": free["continuation"], "free_article": free["article"], "free_word": free["word"],
                "free_legal_article_noun": legal(free["article"], free["word"]),
                "forced_an_tv": total_variation_from_logits(logits, baseline_logits),
                "forced_an_top1_off": top1_word(tokenizer, baseline_logits), "forced_an_top1_on": top1_word(tokenizer, logits),
                "forced_an_top1_changed": int(torch.argmax(baseline_logits)) != int(torch.argmax(logits)),
                "twin_delta_delta_an": twin_delta,
            })
    analysis = {}
    resamples = int(config["bootstrap_resamples"])
    seed = int(config["bootstrap_seed"])
    for i, condition in enumerate(conditions):
        selected = [r for r in rows if r["condition"] == condition["id"]]
        twins = [float(r["twin_delta_delta_an"]) for r in selected if r["twin_delta_delta_an"] is not None]
        analysis[condition["id"]] = {
            "features": [{"layer": int(f["layer"]), "feature_idx": int(f["feature_idx"])} for f in condition["features"]],
            "legal_free_rate": sum(bool(r["free_legal_article_noun"]) for r in selected) / len(selected),
            "other_prefix_rate": sum(r["free_article"] == "other" for r in selected) / len(selected),
            "forced_an_top1_changed_rate": sum(bool(r["forced_an_top1_changed"]) for r in selected) / len(selected),
            "forced_an_tv": interval([float(r["forced_an_tv"]) for r in selected], seed + i * 10, resamples),
            "twin_delta_delta_an": interval(twins, seed + i * 10 + 1, resamples),
            "twin_positive_rate": sum(x > 0 for x in twins) / max(len(twins), 1),
        }
    summary = {"experiment": config["experiment_name"], "generated_at": datetime.now(timezone.utc).isoformat(), "elapsed_sec": time.time() - started, "n_prompts": len(e1_config["test_examples"]), "conditions": analysis}
    write_json(RESULTS_DIR / "rows.json", rows)
    write_json(RESULTS_DIR / "summary.json", summary)
    lines = ["# S2 feature decomposition", "", "| Condition | Legal free | Other prefix | Fixed-`an` TV | Twin ΔΔ | Positive twins |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for condition in conditions:
        block = analysis[condition["id"]]
        tv, twin = block["forced_an_tv"], block["twin_delta_delta_an"]
        lines.append(f"| `{condition['id']}` | {block['legal_free_rate']:.2f} | {block['other_prefix_rate']:.2f} | {tv['mean']:.3f} [{tv['lo']:.3f}, {tv['hi']:.3f}] | {twin['mean']:.3f} [{twin['lo']:.3f}, {twin['hi']:.3f}] | {block['twin_positive_rate']:.2f} |")
    (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
