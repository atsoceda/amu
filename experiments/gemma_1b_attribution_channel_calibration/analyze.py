#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"


def load(path: Path):
    return json.loads(path.read_text())


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, cursor = [0.0] * len(values), 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2 + 1
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return np.asarray(ranks, dtype=float)


def rho(left, right):
    x, y = rankdata(left), rankdata(right)
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if denominator == 0 else float(np.dot(x, y) / denominator)


def bootstrap(left, right, seed, resamples=10000):
    rng, n, values = random.Random(seed), len(left), []
    for _ in range(resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        value = rho([left[i] for i in indices], [right[i] for i in indices])
        if value is not None:
            values.append(value)
    values.sort()
    return {"rho": rho(left, right), "lo": values[int(.025*(len(values)-1))],
            "hi": values[int(.975*(len(values)-1))], "n": n, "resamples": resamples}


def residualize_rank(values, controls):
    y = rankdata(values)
    design = np.column_stack([np.ones(len(values))] + [rankdata(control) for control in controls])
    return y - design @ np.linalg.lstsq(design, y, rcond=None)[0]


def auc(scores, labels):
    positive = [s for s, label in zip(scores, labels) if label]
    negative = [s for s, label in zip(scores, labels) if not label]
    if not positive or not negative:
        return None
    return sum((p > n) + .5*(p == n) for p in positive for n in negative) / (len(positive)*len(negative))


def decoder_norms(config, features):
    root = Path(config["transcoder_weight_snapshot"]) / "clt/width_262k_l0_medium_affine"
    by_layer = {}
    for feature in features:
        by_layer.setdefault(int(feature["layer"]), []).append(int(feature["feature_idx"]))
    norms = {}
    for layer, feature_ids in by_layer.items():
        with safe_open(root / f"params_layer_{layer}.safetensors", framework="pt", device="cpu") as tensors:
            decoder = tensors.get_tensor("w_dec")
            for feature_idx in feature_ids:
                norms[(layer, feature_idx)] = float(torch.linalg.vector_norm(decoder[feature_idx].float()))
    return norms


def main():
    config, selection, rows = load(EXP / "config.json"), load(RESULTS / "selection.json"), load(RESULTS / "rows.json")
    features = selection["features"]
    norms = decoder_norms(config, features)
    feature_rows = []
    for index, feature in enumerate(features):
        group = [row for row in rows if row["feature_index"] == index]
        valid = [row for row in group if row["mediator_valid"]]
        mean = lambda key, data=group: sum(float(row[key]) for row in data) / len(data)
        fully_valid = len(valid) == len(group)
        feature_rows.append({**feature, "decoder_norm": norms[(int(feature["layer"]), int(feature["feature_idx"]))],
                             "mean_heldout_activation": mean("activation"),
                             "active_prompt_rate": sum(row["activation"] > 0 for row in group)/len(group),
                             "mediator_valid_rate": len(valid)/len(group),
                             "article_change_rate": sum(row["article_changed"] for row in group)/len(group),
                             "article_margin_effect": mean("article_margin_effect"),
                             "fixed_mean_tv": sum((row["fixed_a_tv"]+row["fixed_an_tv"])/2 for row in group)/len(group),
                             "total_tv_valid": mean("total_tv", valid) if fully_valid else None,
                             "mediator_tv_valid": mean("mediator_tv", valid) if fully_valid else None,
                             "residual_tv_valid": mean("residual_tv_treated", valid) if fully_valid else None})
    analyses = {}
    for predictor_index, predictor in enumerate(("article_attribution", "future_attribution")):
        analyses[predictor] = {}
        for outcome_index, outcome in enumerate(("article_margin_effect", "fixed_mean_tv", "total_tv_valid", "mediator_tv_valid", "residual_tv_valid")):
            data = [row for row in feature_rows if row[outcome] is not None]
            x, y = [float(row[predictor]) for row in data], [float(row[outcome]) for row in data]
            result = bootstrap(x, y, 20260829 + predictor_index*100 + outcome_index)
            leave_one_out = [rho(x[:i]+x[i+1:], y[:i]+y[i+1:]) for i in range(len(x))]
            largest = max(range(len(y)), key=lambda i: abs(y[i]))
            controls = [[float(row[key]) for row in data] for key in ("layer", "mean_heldout_activation", "decoder_norm")]
            px, py = residualize_rank(x, controls), residualize_rank(y, controls)
            threshold = float(np.median(np.abs([row["article_margin_effect"] for row in data])))
            conditional = [i for i, row in enumerate(data) if abs(row["article_margin_effect"]) >= threshold]
            result.update({"loo_min": min(value for value in leave_one_out if value is not None),
                           "loo_max": max(value for value in leave_one_out if value is not None),
                           "largest_outcome_excluded_rho": rho(x[:largest]+x[largest+1:], y[:largest]+y[largest+1:]),
                           "partial_spearman_layer_activation_decoder_norm": rho(list(px), list(py)),
                           "conditional_on_above_median_abs_margin_rho": rho([x[i] for i in conditional], [y[i] for i in conditional]),
                           "conditional_n": len(conditional)})
            analyses[predictor][outcome] = result
        analyses[predictor]["article_boundary_crossing_auc"] = auc(
            [row[predictor] for row in feature_rows], [row["article_change_rate"] > 0 for row in feature_rows])
    diagnostics = selection["diagnostics"]
    graph_dir = (EXP / config["graphs_path"]).resolve()
    from circuit_tracer.graph import Graph
    graphs = [Graph.from_pt(str(path)) for path in sorted(graph_dir.glob("*.pt"))]
    diagnostics.update({"graph_count": len(graphs), "graph_cap_hits": sum(len(graph.selected_features) >= 1200 for graph in graphs),
                        "mean_active_features": sum(len(graph.active_features) for graph in graphs)/len(graphs),
                        "mean_selected_fraction_of_active": sum(len(graph.selected_features)/len(graph.active_features) for graph in graphs)/len(graphs)})
    selection["diagnostics"] = diagnostics
    summary = load(RESULTS / "summary.json")
    summary.update({"diagnostics": diagnostics, "fully_mediator_valid_features": sum(row["mediator_valid_rate"] == 1 for row in feature_rows),
                    "channel_feature_gate": "Feature must retain a/an top-1 support on all 20 held-out prompts.", "correlations": analyses})
    (RESULTS / "selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    (RESULTS / "feature_rows.json").write_text(json.dumps(feature_rows, indent=2)+"\n")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    lines = ["# Gemma 3 1B attribution-to-channel calibration", "",
             f"Features: {len(feature_rows)}; held-out prompts: 20; fully mediator-valid features: {summary['fully_mediator_valid_features']}.", "",
             "Channel correlations exclude any feature that leaves `a`/`an` top-1 support on any held-out prompt.", "",
             "| Predictor | Outcome | rho | 95% bootstrap CI | n |", "| --- | --- | ---: | ---: | ---: |"]
    for predictor, blocks in analyses.items():
        for outcome, block in blocks.items():
            if outcome == "article_boundary_crossing_auc":
                continue
            lines.append(f"| {predictor} | {outcome} | {block['rho']:.3f} | [{block['lo']:.3f}, {block['hi']:.3f}] | {block['n']} |")
    (RESULTS / "report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"diagnostics": diagnostics, "fully_mediator_valid_features": summary["fully_mediator_valid_features"], "analyses": analyses}, indent=2))


if __name__ == "__main__":
    main()
