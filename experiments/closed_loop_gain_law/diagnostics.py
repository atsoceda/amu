#!/usr/bin/env python3
"""Transparent diagnostics for the held-out 270M public-gain prediction."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"
SEED = 20260902
RESAMPLES = 10_000


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranked = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranked


def spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.corrcoef(ranks(observed), ranks(predicted))[0, 1])


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    denom = float(np.sum((observed - observed.mean()) ** 2))
    return {
        "n": int(observed.size),
        "r2": 1.0 - float(np.sum((observed - predicted) ** 2)) / denom if denom else None,
        "mae": float(np.mean(np.abs(observed - predicted))),
        "median_absolute_error": float(np.median(np.abs(observed - predicted))),
        "spearman_rho": spearman(observed, predicted),
    }


def crossed_weights(feature_ids: np.ndarray, prompt_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    features = np.unique(feature_ids)
    prompts = np.unique(prompt_ids)
    feature_counts = np.bincount(rng.choice(features, size=features.size, replace=True), minlength=int(features.max()) + 1)
    prompt_counts = np.bincount(rng.choice(prompts, size=prompts.size, replace=True), minlength=int(prompts.max()) + 1)
    return feature_counts[feature_ids] * prompt_counts[prompt_ids]


def weighted_metrics(observed: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean = float(np.average(observed, weights=weights))
    denom = float(np.sum(weights * (observed - mean) ** 2))
    r2 = 1.0 - float(np.sum(weights * (observed - predicted) ** 2)) / denom if denom else np.nan
    mae = float(np.average(np.abs(observed - predicted), weights=weights))
    return r2, mae


def blocked_metric_intervals(observed: np.ndarray, predicted: np.ndarray,
                             feature_ids: np.ndarray, prompt_ids: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    r2_draws, mae_draws = [], []
    for _ in range(RESAMPLES):
        weights = crossed_weights(feature_ids, prompt_ids, rng)
        r2, mae = weighted_metrics(observed, predicted, weights)
        if np.isfinite(r2):
            r2_draws.append(r2)
        mae_draws.append(mae)
    return {
        "method": "crossed feature-and-prompt bootstrap",
        "resamples": RESAMPLES,
        "r2_lo": float(np.quantile(r2_draws, .025)),
        "r2_hi": float(np.quantile(r2_draws, .975)),
        "mae_lo": float(np.quantile(mae_draws, .025)),
        "mae_hi": float(np.quantile(mae_draws, .975)),
    }


def binned_calibration(observed: np.ndarray, predicted: np.ndarray,
                       feature_ids: np.ndarray, prompt_ids: np.ndarray) -> list[dict]:
    # Stable rank bins guarantee equal counts even when predictions tie.
    order = np.argsort(predicted, kind="stable")
    bins = np.array_split(order, 5)
    rng = np.random.default_rng(SEED)
    out = []
    for index, members in enumerate(bins, start=1):
        obs, pred = observed[members], predicted[members]
        obs_boot, pred_boot = [], []
        for _ in range(RESAMPLES):
            weights = crossed_weights(feature_ids, prompt_ids, rng)[members]
            if weights.sum() == 0:
                continue
            obs_boot.append(float(np.average(obs, weights=weights)))
            pred_boot.append(float(np.average(pred, weights=weights)))
        out.append({
            "bin": index,
            "n": int(members.size),
            "mean_predicted": float(pred.mean()),
            "mean_observed": float(obs.mean()),
            "predicted_lo": float(np.quantile(pred_boot, .025)),
            "predicted_hi": float(np.quantile(pred_boot, .975)),
            "observed_lo": float(np.quantile(obs_boot, .025)),
            "observed_hi": float(np.quantile(obs_boot, .975)),
            "predicted_min": float(pred.min()),
            "predicted_max": float(pred.max()),
        })
    return out


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, value))))


def predicted_margin_effect(raw_rows: list[dict], held_feature: int, held_prompt: int,
                            attribution: float) -> float:
    train = [r for r in raw_rows if r["feature_index"] != held_feature and r["prompt_index"] != held_prompt]
    x = np.asarray([[1.0, float(r["article_attribution"])] for r in train])
    y = np.asarray([float(r["article_margin_effect"]) for r in train])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return float(beta[0] + beta[1] * attribution)


def diagonal_audit(rows: list[dict], raw_rows: list[dict]) -> dict:
    raw = {(r["feature_index"], r["prompt_index"]): r for r in raw_rows}
    audited = []
    for row in rows:
        # This threshold selects every visually near-diagonal point above 0.5 TV.
        prediction = float(row["predictions"]["full_gain_model"])
        if row["observed"] < .5 or abs(row["observed"] - prediction) > .02:
            continue
        source = raw[(row["feature_index"], row["prompt_index"])]
        predicted_dm = predicted_margin_effect(
            raw_rows, row["feature_index"], row["prompt_index"], source["article_attribution"]
        )
        tau = .1
        base_scaled = source["baseline_margin"] / tau
        observed_scaled = (source["baseline_margin"] + source["article_margin_effect"]) / tau
        predicted_scaled = (source["baseline_margin"] + predicted_dm) / tau
        audited.append({
            "feature_index": row["feature_index"],
            "feature_id": {"layer": source["layer"], "feature_idx": source["feature_idx"]},
            "prompt_id": row["prompt_index"],
            "observed_margin_effect": source["article_margin_effect"],
            "predicted_margin_effect": predicted_dm,
            "observed_abs_delta_q": abs(float(row["observed_delta_q"])),
            "predicted_abs_delta_q": abs(float(row["predicted_delta_q"])),
            "continuation_leverage": source["branch_leverage_tv"],
            "predicted_public_tv": prediction,
            "observed_public_tv": row["observed"],
            "absolute_error": abs(row["observed"] - prediction),
            "observed_policy_high_movement": abs(row["observed_delta_q"]) >= .9,
            "predicted_policy_high_movement": abs(row["predicted_delta_q"]) >= .9,
            "observed_policy_near_saturated": abs(row["observed_delta_q"]) >= .99,
            "predicted_policy_near_saturated": abs(row["predicted_delta_q"]) >= .99,
            "sigmoid_input_clipped_at_80": any(abs(v) >= 80 for v in (base_scaled, observed_scaled, predicted_scaled)),
            "prediction_pipeline": "held-out attribution-to-margin calibration plus held-out prompt baseline margin and intervention-off branch leverage",
            "accounting_oracle_used": False,
            "measured_margin_oracle_public_tv": row["predictions"]["measured_margin_oracle"],
        })
    return {
        "selection_rule": "observed public TV >= 0.5 and absolute prediction error <= 0.02",
        "n": len(audited),
        "all_observed_and_predicted_policy_movement_at_least_0_9": all(
            r["observed_policy_high_movement"] and r["predicted_policy_high_movement"] for r in audited
        ),
        "any_sigmoid_input_clipped_at_80": any(r["sigmoid_input_clipped_at_80"] for r in audited),
        "cells": audited,
    }


def main() -> None:
    rows = json.loads((RESULTS / "validation_predictions.json").read_text())
    rows = [r for r in rows if r["model"] == "gemma_270m" and r["temperature"] == .1 and r["scheme"] == "feature_prompt"]
    observed = np.asarray([r["observed"] for r in rows], dtype=float)
    predicted = np.asarray([r["predictions"]["full_gain_model"] for r in rows], dtype=float)
    attribution = np.asarray([r["predictions"]["attribution_only"] for r in rows], dtype=float)
    feature_ids = np.asarray([r["feature_index"] for r in rows], dtype=int)
    prompt_ids = np.asarray([r["prompt_index"] for r in rows], dtype=int)
    raw_rows = json.loads((ROOT / "experiments/attribution_channel_calibration/results/aligned_rows.json").read_text())

    tail = {}
    for fraction in (0.0, 0.05, 0.10):
        if fraction == 0:
            keep = np.ones(observed.size, dtype=bool)
            threshold = None
        else:
            threshold = float(np.quantile(observed, 1.0 - fraction))
            keep = observed < threshold
        tail[f"exclude_top_{int(fraction * 100)}pct"] = {
            "observed_threshold": threshold,
            **metrics(observed[keep], predicted[keep]),
        }

    audit = diagonal_audit(rows, raw_rows)
    summary = {
        "experiment": "closed_loop_gain_law_diagnostics",
        "selection": {"model": "gemma_270m", "temperature": 0.1, "scheme": "feature_prompt"},
        "full_gain_model": metrics(observed, predicted),
        "blocked_uncertainty": blocked_metric_intervals(observed, predicted, feature_ids, prompt_ids),
        "attribution_only": metrics(observed, attribution),
        "tail_sensitivity": tail,
        "binned_calibration": binned_calibration(observed, predicted, feature_ids, prompt_ids),
        "near_diagonal_high_effect_audit": audit,
        "bootstrap": {
            "resamples": RESAMPLES,
            "seed": SEED,
            "unit": "crossed resampling of feature and prompt identities",
            "note": "Feature and prompt counts are sampled independently with replacement and crossed into cell weights.",
        },
    }
    (RESULTS / "diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RESULTS / "high_effect_diagonal_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
