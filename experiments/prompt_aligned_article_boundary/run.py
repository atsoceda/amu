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
from experiments.lib.mediation_estimands import effect_vector_decomposition, total_variation_from_logits
from experiments.six_cell_family_sweep.run import activations_at_position, build_interventions, next_logits


EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def interval(values: list[float], seed: int, resamples: int) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(values)
    boot = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    boot.sort()
    return {"n": n, "mean": sum(values)/n, "lo": boot[math.floor(.025*(len(boot)-1))], "hi": boot[math.ceil(.975*(len(boot)-1))], "method": "prompt-level nonparametric bootstrap", "resamples": resamples}


def main() -> None:
    config = load(EXP_DIR / "config.json")
    e1_config = load((EXP_DIR / config["e1_config_path"]).resolve())
    selection = load((EXP_DIR / config["e1_selection_path"]).resolve())
    features = selection["sets"]["S1_dual_effect"]["selected_features"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    rows = []
    for index, example in enumerate(e1_config["test_examples"], start=1):
        prompt = f"{config['demonstration']} {example['sentence']}"
        position = len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1
        activations = activations_at_position(model, prompt, position)

        def interventions(gain: float):
            return build_interventions(activations, position, features, gain)[0]

        def margin(gain: float) -> float:
            logits = next_logits(model, prompt, interventions(gain))
            return float(logits[an_id] - logits[a_id])

        low, high = float(config["gain_low"]), float(config["gain_high"])
        margin_low, margin_high = margin(low), margin(high)
        bracketed = margin_low <= 0 < margin_high
        if not bracketed:
            rows.append({"index": index, "sentence": example["sentence"], "bracketed": False, "gain_low": low, "gain_high": high, "margin_low": margin_low, "margin_high": margin_high})
            continue
        for _ in range(int(config["binary_search_steps"])):
            mid = (low + high) / 2
            margin_mid = margin(mid)
            if margin_mid > 0:
                high, margin_high = mid, margin_mid
            else:
                low, margin_low = mid, margin_mid
        low_i, high_i = interventions(low), interventions(high)
        cells: dict[str, dict[str, torch.Tensor]] = {"low": {}, "high": {}}
        for label, ints in (("low", low_i), ("high", high_i)):
            for article, token_id in (("a", a_id), ("an", an_id)):
                prefix = prompt + tokenizer.decode([token_id])
                cells[label][article] = next_logits(model, prefix, ints)
        vectors = effect_vector_decomposition(
            baseline_off=cells["low"]["a"],
            treated_article_off=cells["low"]["an"],
            treated_article_on=cells["high"]["an"],
        )
        fixed_a_tv = total_variation_from_logits(cells["high"]["a"], cells["low"]["a"])
        rows.append({
            "index": index, "sentence": example["sentence"], "bracketed": True,
            "gain_low": low, "gain_high": high, "gain_width": high-low,
            "margin_low": margin_low, "margin_high": margin_high,
            "total_tv": vectors["total_tv"], "token_substitution_tv": vectors["mediator_tv"],
            "fixed_an_residual_tv": vectors["residual_tv"], "fixed_a_residual_tv": fixed_a_tv,
            "cosine_token_total": vectors["cosine_mediator_total"],
            "reconstruction_l1": vectors["reconstruction_l1"],
        })
    selected = [r for r in rows if r["bracketed"]]
    seed = int(config["bootstrap_seed"])
    resamples = int(config["bootstrap_resamples"])
    def iv(key: str, offset: int):
        return interval([float(r[key]) for r in selected], seed + offset, resamples)
    summary = {
        "experiment": config["experiment_name"], "generated_at": datetime.now(timezone.utc).isoformat(), "elapsed_sec": time.time()-started,
        "n_prompts": len(rows), "n_bracketed": len(selected), "binary_search_steps": int(config["binary_search_steps"]),
        "gain_width": iv("gain_width", 0), "total_tv": iv("total_tv", 1), "token_substitution_tv": iv("token_substitution_tv", 2),
        "fixed_an_residual_tv": iv("fixed_an_residual_tv", 3), "fixed_a_residual_tv": iv("fixed_a_residual_tv", 4),
        "cosine_token_total": iv("cosine_token_total", 5), "reconstruction_l1": iv("reconstruction_l1", 6),
        "interpretation": "Prompt alignment isolates a discrete article-policy transition: the free-path noun jump is compared with smooth fixed-article changes over the same narrow gain interval."
    }
    write_json(RESULTS_DIR / "rows.json", rows)
    write_json(RESULTS_DIR / "summary.json", summary)
    def fmt(b): return f"{b['mean']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"
    report = f"""# Prompt-aligned article boundary

- Bracketed prompts: {summary['n_bracketed']}/{summary['n_prompts']}.
- Final gain-bracket width: {fmt(summary['gain_width'])}.
- Cross-boundary total TV: {fmt(summary['total_tv'])}.
- Token-substitution TV: {fmt(summary['token_substitution_tv'])}.
- Fixed-`an` local residual TV: {fmt(summary['fixed_an_residual_tv'])}.
- Fixed-`a` local residual TV: {fmt(summary['fixed_a_residual_tv'])}.
- Token/total cosine: {fmt(summary['cosine_token_total'])}.

{summary['interpretation']}
"""
    (RESULTS_DIR / "report.md").write_text(report)


if __name__ == "__main__":
    main()
