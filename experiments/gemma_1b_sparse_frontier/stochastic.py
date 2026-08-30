#!/usr/bin/env python3
"""Stochastic public/private decomposition for the frozen 1B frontier handle."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.aan_protocol import token_id_for_text, write_json
from experiments.lib.core import load_replacement_model, setup_file_logging
from experiments.gemma_1b_residual_scale.run import cosine, interval, mixture
from experiments.six_cell_family_sweep.run import activations_at_position, next_logits

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results/stochastic"


def load(path: Path):
    return json.loads(path.read_text())


def main():
    cfg = load(EXP / "config.json")
    source = load((EXP / cfg["e1_config_path"]).resolve())
    run = cfg["runs"][0]
    ranked = load((EXP / run["ranked_features_path"]).resolve())
    selected = [row for row in ranked if row["prompt_count"] >= 3][: int(run["feature_count"])]
    gain = float(run["amplify_factor"])
    temperatures = [float(value) for value in cfg["temperatures"]]
    RESULTS.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS)
    started = time.time()
    model = load_replacement_model(cfg)
    tok = model.tokenizer
    ids = {"a": token_id_for_text(tok, " a"), "an": token_id_for_text(tok, " an")}
    rows = []
    for prompt_index, example in enumerate(source["test_examples"], 1):
        prompt = f"{cfg['demonstration']} {example['sentence']}"
        position = len(tok(prompt, add_special_tokens=True).input_ids) - 1
        activations = activations_at_position(model, prompt, position)
        interventions = []
        for feature in selected:
            layer, feature_idx = int(feature["layer"]), int(feature["feature_idx"])
            activation = float(activations[layer, feature_idx].detach().float().cpu())
            interventions.append({"layer": layer, "pos": position, "feature_idx": feature_idx, "value": gain * activation})
        off_article = next_logits(model, prompt, [])
        on_article = next_logits(model, prompt, interventions)
        off_branches = {name: next_logits(model, prompt + tok.decode([token_id]), []) for name, token_id in ids.items()}
        on_branches = {name: next_logits(model, prompt + tok.decode([token_id]), interventions) for name, token_id in ids.items()}
        margin_off = float(off_article[ids["an"]] - off_article[ids["a"]])
        margin_on = float(on_article[ids["an"]] - on_article[ids["a"]])
        stochastic = {}
        for tau in temperatures:
            off_mix, off_mass = mixture(off_article, off_branches, ids, tau)
            on_mix, on_mass = mixture(on_article, on_branches, ids, tau)
            public_mix, _ = mixture(on_article, off_branches, ids, tau)
            total = on_mix - off_mix
            public = public_mix - off_mix
            private = on_mix - public_mix
            weights_off = torch.softmax(off_article[[ids["a"], ids["an"]]] / tau, -1)
            weights_on = torch.softmax(on_article[[ids["a"], ids["an"]]] / tau, -1)
            branch_leverage = float(.5 * (torch.softmax(off_branches["an"], -1) - torch.softmax(off_branches["a"], -1)).abs().sum())
            predicted_public = abs(float(weights_on[1] - weights_off[1])) * branch_leverage
            stochastic[str(tau)] = {
                "q_an_off": float(weights_off[1]), "q_an_on": float(weights_on[1]),
                "delta_q_an": float(weights_on[1] - weights_off[1]),
                "off_article_mass": off_mass, "on_article_mass": on_mass,
                "branch_leverage_tv": branch_leverage,
                "total_tv": float(.5 * total.abs().sum()),
                "public_tv": float(.5 * public.abs().sum()),
                "private_tv": float(.5 * private.abs().sum()),
                "public_total_cosine": cosine(public, total),
                "private_total_cosine": cosine(private, total),
                "gain_law_prediction": predicted_public,
                "gain_law_absolute_error": abs(float(.5 * public.abs().sum()) - predicted_public),
                "reconstruction_l1": float((total - public - private).abs().sum()),
            }
        rows.append({
            "prompt_index": prompt_index, "sentence": example["sentence"],
            "expected_article": example["expected_article"],
            "margin_off": margin_off, "margin_on": margin_on,
            "margin_movement": margin_on - margin_off,
            "greedy_article_off": "an" if int(off_article.argmax()) == ids["an"] else ("a" if int(off_article.argmax()) == ids["a"] else "other"),
            "greedy_article_on": "an" if int(on_article.argmax()) == ids["an"] else ("a" if int(on_article.argmax()) == ids["a"] else "other"),
            "stochastic": stochastic,
        })
        print(f"frontier stochastic prompt {prompt_index}/{len(source['test_examples'])}", flush=True)

    seed, resamples = int(cfg["bootstrap_seed"]), int(cfg["bootstrap_resamples"])
    summary = {
        "experiment": "gemma_1b_sparse_frontier_stochastic_decomposition",
        "generated_at": datetime.now(timezone.utc).isoformat(), "elapsed_sec": time.time() - started,
        "model": cfg["model"], "feature": selected[0], "gain": gain,
        "n_prompts": len(rows),
        "margin_movement": interval([r["margin_movement"] for r in rows], seed, resamples),
        "greedy_switch_count": sum(r["greedy_article_off"] != r["greedy_article_on"] for r in rows),
        "temperatures": {},
    }
    metrics = ("delta_q_an", "public_tv", "private_tv", "total_tv", "public_total_cosine", "private_total_cosine", "branch_leverage_tv", "gain_law_absolute_error", "off_article_mass", "on_article_mass")
    for t_index, tau in enumerate(temperatures):
        key = str(tau)
        summary["temperatures"][key] = {
            metric: interval([r["stochastic"][key][metric] for r in rows], seed + 100*t_index + i + 1, resamples)
            for i, metric in enumerate(metrics)
        }
    write_json(RESULTS / "rows.json", rows)
    write_json(RESULTS / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
