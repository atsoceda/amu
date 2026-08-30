#!/usr/bin/env python3
"""Two-way held-out validation and baselines for the closed-loop gain model."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
TEMPERATURES = (0.1, 0.25, 0.5, 1.0)


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, value))))


def q_shift(row, dm, tau):
    return sigmoid((row["baseline_margin"] + dm) / tau) - sigmoid(row["baseline_margin"] / tau)


def ols(train, x_fn, y_fn):
    x = np.asarray([[1.0, float(x_fn(row))] for row in train])
    y = np.asarray([float(y_fn(row)) for row in train])
    return np.linalg.lstsq(x, y, rcond=None)[0]


def linear(beta, value):
    return float(beta[0] + beta[1] * value)


def enrich(rows, model, tau):
    out = []
    for row in rows:
        copy = dict(row)
        dm = float(row["article_margin_effect"])
        dq = q_shift(row, dm, tau)
        copy["observed_delta_q"] = dq
        copy["observed_public_tv"] = abs(dq) * float(row["branch_leverage_tv"])
        copy["activation_baseline"] = math.log1p(abs(float(row.get("activation", row.get("activation_archived_reconstruction", 0.0)))))
        copy["model"] = model
        out.append(copy)
    return out


def folds(rows, scheme):
    features = sorted({r["feature_index"] for r in rows})
    prompts = sorted({r["prompt_index"] for r in rows})
    if scheme == "feature":
        for feature in features:
            yield [r for r in rows if r["feature_index"] != feature], [r for r in rows if r["feature_index"] == feature]
    elif scheme == "prompt":
        for prompt in prompts:
            yield [r for r in rows if r["prompt_index"] != prompt], [r for r in rows if r["prompt_index"] == prompt]
    else:
        for feature in features:
            for prompt in prompts:
                train = [r for r in rows if r["feature_index"] != feature and r["prompt_index"] != prompt]
                test = [r for r in rows if r["feature_index"] == feature and r["prompt_index"] == prompt]
                yield train, test


def validate(rows, scheme, tau):
    predictions = []
    for train, test in folds(rows, scheme):
        dm_beta = ols(train, lambda r: r["article_attribution"], lambda r: r["article_margin_effect"])
        attr_y_beta = ols(train, lambda r: abs(r["article_attribution"]), lambda r: r["observed_public_tv"])
        susceptibility_beta = ols(train, lambda r: sigmoid(r["baseline_margin"]/tau)*(1-sigmoid(r["baseline_margin"]/tau))/tau,
                                  lambda r: r["observed_public_tv"])
        leverage_beta = ols(train, lambda r: r["branch_leverage_tv"], lambda r: r["observed_public_tv"])
        activation_beta = ols(train, lambda r: r["activation_baseline"], lambda r: r["observed_public_tv"])
        mean_y = float(np.mean([r["observed_public_tv"] for r in train]))
        mean_dm = float(np.mean([r["article_margin_effect"] for r in train]))
        mean_leverage = float(np.mean([r["branch_leverage_tv"] for r in train]))
        for row in test:
            predicted_dm = linear(dm_beta, row["article_attribution"])
            predicted_dq = q_shift(row, predicted_dm, tau)
            oracle_dq = row["observed_delta_q"]
            susceptibility = sigmoid(row["baseline_margin"]/tau)*(1-sigmoid(row["baseline_margin"]/tau))/tau
            pred = {
                "constant": mean_y,
                "attribution_only": max(0.0, linear(attr_y_beta, abs(row["article_attribution"]))),
                "susceptibility_only": max(0.0, linear(susceptibility_beta, susceptibility)),
                "branch_leverage_only": max(0.0, linear(leverage_beta, row["branch_leverage_tv"])),
                "activation_magnitude": max(0.0, linear(activation_beta, row["activation_baseline"])),
                "susceptibility_constant_shift": abs(q_shift(row, mean_dm, tau)) * mean_leverage,
                "attribution_susceptibility": abs(predicted_dq) * mean_leverage,
                "full_gain_model": abs(predicted_dq) * row["branch_leverage_tv"],
                "measured_margin_oracle": abs(oracle_dq) * row["branch_leverage_tv"],
            }
            predictions.append({"feature_index": row["feature_index"], "prompt_index": row["prompt_index"],
                                "observed": row["observed_public_tv"], "observed_delta_q": oracle_dq,
                                "predicted_delta_q": predicted_dq, "predictions": pred})
    y = np.asarray([r["observed"] for r in predictions])
    denom = float(((y-y.mean())**2).sum())
    metrics = {}
    for name in predictions[0]["predictions"]:
        p = np.asarray([r["predictions"][name] for r in predictions])
        slope = float(np.linalg.lstsq(np.asarray([[1.0, value] for value in p]), y, rcond=None)[0][1])
        metrics[name] = {"r2": 1-float(((y-p)**2).sum())/denom if denom else None,
                         "mae": float(np.mean(np.abs(y-p))), "calibration_slope": slope}
    nonzero = [r for r in predictions if r["observed_delta_q"] != 0 and r["predicted_delta_q"] != 0]
    metrics["full_gain_model"]["mean_vector_cosine"] = float(np.mean([1.0 if r["observed_delta_q"]*r["predicted_delta_q"] > 0 else -1.0 for r in nonzero]))
    metrics["full_gain_model"]["policy_direction_accuracy"] = float(np.mean([r["observed_delta_q"]*r["predicted_delta_q"] > 0 for r in nonzero]))
    return metrics, predictions


def main():
    sources = {
        "gemma_270m": ROOT / "experiments/attribution_channel_calibration/results/aligned_rows.json",
        "gemma_1b": ROOT / "experiments/gemma_1b_attribution_channel_calibration/results/aligned_rows.json",
    }
    summary = {"experiment": "closed_loop_gain_law_two_way_validation", "models": {}}
    all_predictions = []
    for model, path in sources.items():
        raw = json.loads(path.read_text()); summary["models"][model] = {}
        for tau in TEMPERATURES:
            rows = enrich(raw, model, tau); summary["models"][model][str(tau)] = {}
            for scheme in ("feature", "prompt", "feature_prompt"):
                metrics, predictions = validate(rows, scheme, tau)
                summary["models"][model][str(tau)][scheme] = metrics
                for row in predictions:
                    all_predictions.append({"model": model, "temperature": tau, "scheme": scheme, **row})
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RESULTS / "validation_predictions.json").write_text(json.dumps(all_predictions, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
