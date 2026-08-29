#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from circuit_tracer.graph import Graph

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.aan_protocol import first_content_token_text, slugify, token_id_for_text, write_json
from experiments.lib.core import feature_effect_map, load_replacement_model, setup_file_logging
from experiments.lib.mediation_estimands import total_variation_from_logits
from experiments.six_cell_family_sweep.run import activations_at_position, next_logits

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    lm, rm = sum(left) / len(left), sum(right) / len(right)
    lc, rc = [x - lm for x in left], [x - rm for x in right]
    denominator = math.sqrt(sum(x*x for x in lc) * sum(x*x for x in rc))
    return sum(x*y for x, y in zip(lc, rc)) / denominator if denominator else 0.0


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(rankdata(left), rankdata(right))


def bootstrap_rho(left: list[float], right: list[float], seed: int, resamples: int) -> dict[str, Any]:
    rng, n = random.Random(seed), len(left)
    values = []
    for _ in range(resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        values.append(spearman([left[i] for i in indices], [right[i] for i in indices]))
    values.sort()
    return {"rho": spearman(left, right), "lo": values[math.floor(.025*(resamples-1))],
            "hi": values[math.ceil(.975*(resamples-1))], "n": n, "resamples": resamples}


def select_features(config: dict[str, Any], source: dict[str, Any], tokenizer) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graph_dir = (EXP / config["graphs_path"]).resolve()
    a_id, an_id = token_id_for_text(tokenizer, " a"), token_id_for_text(tokenizer, " an")
    pool: dict[tuple[int, int], dict[str, Any]] = {}
    cap_hits = 0
    active_counts = []
    for sentence in source["selection_sentences"]:
        slug = slugify(sentence)
        meta = load(graph_dir / f"{slug}__meta.json")
        article = Graph.from_pt(str(graph_dir / f"{slug}__article.pt"))
        future = Graph.from_pt(str(graph_dir / f"{slug}__future.pt"))
        content = meta.get("content_token_text") or first_content_token_text(tokenizer, meta["listed_word"])
        content_id = token_id_for_text(tokenizer, content)
        position = len(tokenizer(meta["article_prompt"], add_special_tokens=True).input_ids) - 1
        a_map, an_map = feature_effect_map(article, a_id), feature_effect_map(article, an_id)
        future_map = feature_effect_map(future, content_id)
        for graph in (article, future):
            count = int(len(graph.active_features))
            active_counts.append(count)
            cap_hits += int(len(graph.selected_features) >= 1200)
        for layer, pos, feature_idx in set(a_map) & set(an_map) & set(future_map):
            if pos != position:
                continue
            key = (layer, feature_idx)
            graph_key = (layer, pos, feature_idx)
            row = pool.setdefault(key, {"layer": layer, "feature_idx": feature_idx, "count": 0,
                                       "article_sum": 0.0, "future_sum": 0.0, "activation_sum": 0.0})
            row["count"] += 1
            row["article_sum"] += an_map[graph_key]["direct_effect"] - a_map[graph_key]["direct_effect"]
            row["future_sum"] += future_map[graph_key]["direct_effect"]
            row["activation_sum"] += an_map[graph_key]["activation"]
    candidates = []
    for row in pool.values():
        if row["count"] < int(config["minimum_selection_prompt_count"]):
            continue
        candidates.append({"layer": row["layer"], "feature_idx": row["feature_idx"],
                           "selection_prompt_count": row["count"],
                           "article_attribution": row["article_sum"] / row["count"],
                           "future_attribution": row["future_sum"] / row["count"],
                           "mean_selection_activation": row["activation_sum"] / row["count"]})
    ar, fr = rankdata([x["article_attribution"] for x in candidates]), rankdata([x["future_attribution"] for x in candidates])
    n = len(candidates)
    for row, article_rank, future_rank in zip(candidates, ar, fr):
        row["article_rank"], row["future_rank"] = article_rank/n, future_rank/n
    scores = {
        "high_article_high_future": lambda r: r["article_rank"] + r["future_rank"],
        "high_article_low_future": lambda r: r["article_rank"] + 1-r["future_rank"],
        "low_article_high_future": lambda r: 1-r["article_rank"] + r["future_rank"],
        "low_article_low_future": lambda r: 2-r["article_rank"]-r["future_rank"],
    }
    selected, used = [], set()
    for stratum, score in scores.items():
        for row in sorted(candidates, key=lambda r: (score(r), r["selection_prompt_count"]), reverse=True):
            key = (row["layer"], row["feature_idx"])
            if key in used:
                continue
            selected.append({**row, "stratum": stratum, "selection_score": score(row)})
            used.add(key)
            if sum(x["stratum"] == stratum for x in selected) == int(config["features_per_stratum"]):
                break
    diagnostics = {"eligible_candidates": n, "graph_count": len(active_counts), "graph_cap_hits": cap_hits,
                   "mean_active_features": sum(active_counts)/len(active_counts)}
    return selected, diagnostics


def main() -> None:
    config = load(EXP / "config.json")
    source = load((EXP / config["source_config_path"]).resolve())
    RESULTS.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS)
    started = time.time()
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    a_id, an_id = token_id_for_text(tokenizer, " a"), token_id_for_text(tokenizer, " an")
    article_ids = {a_id, an_id}
    features, diagnostics = select_features(config, source, tokenizer)
    write_json(RESULTS / "selection.json", {"features": features, "diagnostics": diagnostics})
    rows = []
    gain = float(config["amplify_factor"])
    for prompt_index, example in enumerate(source["test_examples"], 1):
        prompt = f"{config['demonstration']} {example['sentence']}"
        position = len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1
        activations = activations_at_position(model, prompt, position)
        off_article_logits = next_logits(model, prompt, [])
        off_article_id = int(torch.argmax(off_article_logits))
        off_noun = next_logits(model, prompt + tokenizer.decode([off_article_id]), [])
        off_forced = {a_id: next_logits(model, prompt + tokenizer.decode([a_id]), []),
                      an_id: next_logits(model, prompt + tokenizer.decode([an_id]), [])}
        for feature_index, feature in enumerate(features):
            layer, feature_idx = int(feature["layer"]), int(feature["feature_idx"])
            activation = float(activations[layer, feature_idx].detach().float().cpu())
            interventions = [{"layer": layer, "pos": position, "feature_idx": feature_idx, "value": activation*gain}]
            on_article_logits = next_logits(model, prompt, interventions)
            on_article_id = int(torch.argmax(on_article_logits))
            on_noun = next_logits(model, prompt + tokenizer.decode([on_article_id]), interventions)
            replay = next_logits(model, prompt + tokenizer.decode([on_article_id]), [])
            on_forced = {a_id: next_logits(model, prompt + tokenizer.decode([a_id]), interventions),
                         an_id: next_logits(model, prompt + tokenizer.decode([an_id]), interventions)}
            mediator_valid = off_article_id in article_ids and on_article_id in article_ids
            rows.append({"feature_index": feature_index, "prompt_index": prompt_index,
                         "layer": layer, "feature_idx": feature_idx, "stratum": feature["stratum"],
                         "activation": activation, "article_attribution": feature["article_attribution"],
                         "future_attribution": feature["future_attribution"],
                         "off_article_id": off_article_id, "on_article_id": on_article_id,
                         "mediator_valid": mediator_valid, "article_changed": on_article_id != off_article_id,
                         "article_margin_effect": float((on_article_logits[an_id]-on_article_logits[a_id]) -
                                                        (off_article_logits[an_id]-off_article_logits[a_id])),
                         "total_tv": total_variation_from_logits(on_noun, off_noun),
                         "mediator_tv": total_variation_from_logits(replay, off_noun) if mediator_valid else None,
                         "residual_tv_treated": total_variation_from_logits(on_noun, replay) if mediator_valid else None,
                         "fixed_a_tv": total_variation_from_logits(on_forced[a_id], off_forced[a_id]),
                         "fixed_an_tv": total_variation_from_logits(on_forced[an_id], off_forced[an_id])})
        print(f"completed prompt {prompt_index}/{len(source['test_examples'])}", flush=True)
    feature_rows = []
    for index, feature in enumerate(features):
        group = [r for r in rows if r["feature_index"] == index]
        valid = [r for r in group if r["mediator_valid"]]
        mean = lambda key, data=group: sum(float(r[key]) for r in data)/len(data)
        feature_rows.append({**feature, "active_prompt_rate": sum(r["activation"] > 0 for r in group)/len(group),
                             "mediator_valid_rate": len(valid)/len(group),
                             "article_change_rate": sum(r["article_changed"] for r in group)/len(group),
                             "article_margin_effect": mean("article_margin_effect"),
                             "fixed_mean_tv": sum((r["fixed_a_tv"]+r["fixed_an_tv"])/2 for r in group)/len(group),
                             "total_tv_valid": mean("total_tv", valid) if len(valid) == len(group) else None,
                             "mediator_tv_valid": mean("mediator_tv", valid) if len(valid) == len(group) else None,
                             "residual_tv_valid": mean("residual_tv_treated", valid) if len(valid) == len(group) else None})
    seed, resamples = int(config["bootstrap_seed"]), int(config["bootstrap_resamples"])
    correlations = {}
    outcomes = ["article_margin_effect", "fixed_mean_tv", "total_tv_valid", "mediator_tv_valid", "residual_tv_valid"]
    for pi, predictor in enumerate(("article_attribution", "future_attribution")):
        for oi, outcome in enumerate(outcomes):
            valid = [r for r in feature_rows if r[outcome] is not None]
            correlations[f"{predictor}__{outcome}"] = bootstrap_rho(
                [float(r[predictor]) for r in valid], [float(r[outcome]) for r in valid],
                seed + pi*100 + oi, resamples)
    summary = {"experiment": config["experiment_name"], "generated_at": datetime.now(timezone.utc).isoformat(),
               "elapsed_sec": time.time()-started, "n_features": len(features),
               "n_prompts": len(source["test_examples"]), "gain": gain,
               "identification_rule": "Channel outcomes are computed only where untreated and treated free next tokens are both a/an.",
               "diagnostics": diagnostics, "correlations": correlations}
    write_json(RESULTS / "rows.json", rows)
    write_json(RESULTS / "feature_rows.json", feature_rows)
    write_json(RESULTS / "summary.json", summary)
    lines = ["# Gemma 3 1B attribution-to-channel calibration", "",
             f"Features: {len(features)}; held-out prompts: {len(source['test_examples'])}; gain: {gain:g}x.", "",
             "| Predictor | Outcome | rho | 95% bootstrap CI | n |", "| --- | --- | ---: | ---: | ---: |"]
    for key, block in correlations.items():
        predictor, outcome = key.split("__")
        lines.append(f"| {predictor} | {outcome} | {block['rho']:.3f} | [{block['lo']:.3f}, {block['hi']:.3f}] | {block['n']} |")
    (RESULTS / "report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
