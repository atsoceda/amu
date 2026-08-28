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
from scipy.stats import rankdata, spearmanr

from circuit_tracer.graph import Graph
from experiments.lib.aan_protocol import first_content_token_text, slugify, token_id_for_text, write_json
from experiments.lib.core import feature_effect_map, load_replacement_model, setup_file_logging
from experiments.lib.mediation_estimands import total_variation_from_logits
from experiments.six_cell_family_sweep.run import activations_at_position, next_logits


EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def interval(values: list[float], seed: int, resamples: int) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(values)
    boot = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    boot.sort()
    return {"n": n, "mean": sum(values)/n, "lo": boot[math.floor(.025*(len(boot)-1))], "hi": boot[math.ceil(.975*(len(boot)-1))], "method": "feature-level nonparametric bootstrap", "resamples": resamples}


def select_features(config: dict[str, Any], tokenizer) -> list[dict[str, Any]]:
    e1 = load((EXP_DIR / config["e1_config_path"]).resolve())
    graphs = (EXP_DIR / config["graphs_path"]).resolve()
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    pool: dict[tuple[int, int], dict[str, Any]] = {}
    for sentence in e1["selection_sentences"]:
        prompt_id = slugify(sentence)
        meta = load(graphs / f"{prompt_id}__meta.json")
        article = Graph.from_pt(str(graphs / f"{prompt_id}__article.pt"))
        future = Graph.from_pt(str(graphs / f"{prompt_id}__future.pt"))
        content_id = token_id_for_text(tokenizer, meta.get("content_token_text") or first_content_token_text(tokenizer, meta["listed_word"]))
        pre_pos = len(tokenizer(meta["article_prompt"], add_special_tokens=True).input_ids) - 1
        a_map = feature_effect_map(article, a_id)
        an_map = feature_effect_map(article, an_id)
        future_map = feature_effect_map(future, content_id)
        for layer, pos, feature_idx in set(a_map) & set(an_map) & set(future_map):
            if pos != pre_pos:
                continue
            key = (layer, feature_idx)
            row = pool.setdefault(key, {"layer": layer, "feature_idx": feature_idx, "count": 0, "article_sum": 0.0, "future_sum": 0.0, "activation_sum": 0.0})
            row["count"] += 1
            row["article_sum"] += an_map[(layer,pos,feature_idx)]["direct_effect"] - a_map[(layer,pos,feature_idx)]["direct_effect"]
            row["future_sum"] += future_map[(layer,pos,feature_idx)]["direct_effect"]
            row["activation_sum"] += an_map[(layer,pos,feature_idx)]["activation"]
    candidates = []
    for row in pool.values():
        if row["count"] < int(config["minimum_selection_prompt_count"]):
            continue
        candidates.append({
            "layer": row["layer"], "feature_idx": row["feature_idx"], "selection_prompt_count": row["count"],
            "article_attribution": row["article_sum"] / row["count"],
            "future_attribution": row["future_sum"] / row["count"],
            "mean_selection_activation": row["activation_sum"] / row["count"],
        })
    article_values = [c["article_attribution"] for c in candidates]
    future_values = [c["future_attribution"] for c in candidates]
    article_rank = rankdata(article_values, method="average") / len(candidates)
    future_rank = rankdata(future_values, method="average") / len(candidates)
    for row, ar, fr in zip(candidates, article_rank, future_rank):
        row["article_rank"] = float(ar)
        row["future_rank"] = float(fr)
    scores = {
        "high_article_high_future": lambda r: r["article_rank"] + r["future_rank"],
        "high_article_low_future": lambda r: r["article_rank"] + (1-r["future_rank"]),
        "low_article_high_future": lambda r: (1-r["article_rank"]) + r["future_rank"],
        "low_article_low_future": lambda r: (1-r["article_rank"]) + (1-r["future_rank"]),
    }
    selected: list[dict[str, Any]] = []
    used: set[tuple[int, int]] = set()
    for stratum, score_fn in scores.items():
        ranked = sorted(candidates, key=lambda row: (score_fn(row), row["selection_prompt_count"]), reverse=True)
        count = 0
        for row in ranked:
            key = (row["layer"], row["feature_idx"])
            if key in used:
                continue
            item = dict(row)
            item["stratum"] = stratum
            item["selection_score"] = float(score_fn(row))
            selected.append(item)
            used.add(key)
            count += 1
            if count == int(config["features_per_stratum"]):
                break
    return selected


def main() -> None:
    config = load(EXP_DIR / "config.json")
    e1 = load((EXP_DIR / config["e1_config_path"]).resolve())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    features = select_features(config, tokenizer)
    write_json(RESULTS_DIR / "selection.json", {"selection_sentences": e1["selection_sentences"], "features": features})

    rows: list[dict[str, Any]] = []
    baseline_cache: dict[tuple[int, int], torch.Tensor] = {}
    for prompt_index, example in enumerate(e1["test_examples"], start=1):
        prompt = f"{config['demonstration']} {example['sentence']}"
        position = len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1
        activations = activations_at_position(model, prompt, position)
        off_article_logits = next_logits(model, prompt, [])
        off_article_id = int(torch.argmax(off_article_logits))
        off_noun = next_logits(model, prompt + tokenizer.decode([off_article_id]), [])
        off_forced = {"a": next_logits(model, prompt + tokenizer.decode([a_id]), []), "an": next_logits(model, prompt + tokenizer.decode([an_id]), [])}
        for feature_index, feature in enumerate(features):
            layer = int(feature["layer"]); feature_idx = int(feature["feature_idx"])
            activation = float(activations[layer, feature_idx].detach().float().cpu())
            interventions = [{"layer": layer, "pos": position, "feature_idx": feature_idx, "value": activation * float(config["amplify_factor"])}]
            tuples = [(layer, position, feature_idx, interventions[0]["value"])]
            on_article_logits = next_logits(model, prompt, interventions)
            on_article_id = int(torch.argmax(on_article_logits))
            on_noun = next_logits(model, prompt + tokenizer.decode([on_article_id]), interventions)
            key = (prompt_index, on_article_id)
            if key not in baseline_cache:
                baseline_cache[key] = next_logits(model, prompt + tokenizer.decode([on_article_id]), [])
            replay = baseline_cache[key]
            on_forced_a = next_logits(model, prompt + tokenizer.decode([a_id]), interventions)
            on_forced_an = next_logits(model, prompt + tokenizer.decode([an_id]), interventions)
            rows.append({
                "feature_index": feature_index, "prompt_index": prompt_index,
                "layer": layer, "feature_idx": feature_idx, "stratum": feature["stratum"],
                "activation": activation, "article_attribution": feature["article_attribution"],
                "future_attribution": feature["future_attribution"],
                "article_margin_effect": float((on_article_logits[an_id]-on_article_logits[a_id]) - (off_article_logits[an_id]-off_article_logits[a_id])),
                "article_changed": on_article_id != off_article_id,
                "total_tv": total_variation_from_logits(on_noun, off_noun),
                "mediator_tv": total_variation_from_logits(replay, off_noun),
                "residual_tv_treated": total_variation_from_logits(on_noun, replay),
                "fixed_a_tv": total_variation_from_logits(on_forced_a, off_forced["a"]),
                "fixed_an_tv": total_variation_from_logits(on_forced_an, off_forced["an"]),
            })

    feature_rows: list[dict[str, Any]] = []
    for index, feature in enumerate(features):
        group = [r for r in rows if r["feature_index"] == index]
        feature_rows.append({**feature,
            "active_prompt_rate": sum(r["activation"] > 0 for r in group)/len(group),
            "article_change_rate": sum(r["article_changed"] for r in group)/len(group),
            **{key: sum(float(r[key]) for r in group)/len(group) for key in ("article_margin_effect","total_tv","mediator_tv","residual_tv_treated","fixed_a_tv","fixed_an_tv")},
            "fixed_mean_tv": sum((r["fixed_a_tv"]+r["fixed_an_tv"])/2 for r in group)/len(group),
        })

    correlations: dict[str, Any] = {}
    for predictor in ("article_attribution", "future_attribution"):
        for outcome in ("article_margin_effect", "total_tv", "mediator_tv", "residual_tv_treated", "fixed_mean_tv"):
            statistic, pvalue = spearmanr([r[predictor] for r in feature_rows], [r[outcome] for r in feature_rows])
            correlations[f"{predictor}__{outcome}"] = {"spearman_rho": float(statistic), "two_sided_p": float(pvalue), "n_features": len(feature_rows)}
    seed = int(config["bootstrap_seed"]); resamples = int(config["bootstrap_resamples"])
    strata = {}
    for stratum_index, stratum in enumerate(sorted(set(r["stratum"] for r in feature_rows))):
        group = [r for r in feature_rows if r["stratum"] == stratum]
        strata[stratum] = {key: interval([float(r[key]) for r in group], seed + stratum_index*20 + idx, resamples) for idx,key in enumerate(("article_margin_effect","total_tv","mediator_tv","residual_tv_treated","fixed_mean_tv","article_change_rate"))}
    summary = {
        "experiment": config["experiment_name"], "generated_at": datetime.now(timezone.utc).isoformat(), "elapsed_sec": time.time()-started,
        "n_features": len(feature_rows), "n_prompts": len(e1["test_examples"]), "gain": config["amplify_factor"],
        "selection_scope": "All feature selection and strata use only eight selection occupations and their stored graphs.",
        "correlations": correlations, "strata": strata,
    }
    write_json(RESULTS_DIR / "rows.json", rows)
    write_json(RESULTS_DIR / "feature_rows.json", feature_rows)
    write_json(RESULTS_DIR / "summary.json", summary)
    lines = ["# Attribution-score calibration against causal channel type", "", f"Features: {len(feature_rows)}; held-out prompts: {len(e1['test_examples'])}.", "", "| Predictor | Outcome | Spearman rho | p |", "| --- | --- | ---: | ---: |"]
    for key, block in correlations.items():
        predictor, outcome = key.split("__")
        lines.append(f"| {predictor} | {outcome} | {block['spearman_rho']:.3f} | {block['two_sided_p']:.3g} |")
    (RESULTS_DIR / "report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
