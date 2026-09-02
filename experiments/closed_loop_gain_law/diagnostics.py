#!/usr/bin/env python3
"""Transparent diagnostics for the held-out 270M public-gain prediction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


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


def binned_calibration(observed: np.ndarray, predicted: np.ndarray) -> list[dict]:
    # Stable rank bins guarantee equal counts even when predictions tie.
    order = np.argsort(predicted, kind="stable")
    bins = np.array_split(order, 5)
    rng = np.random.default_rng(SEED)
    out = []
    for index, members in enumerate(bins, start=1):
        obs, pred = observed[members], predicted[members]
        draws = rng.integers(0, members.size, size=(RESAMPLES, members.size))
        obs_boot = obs[draws].mean(axis=1)
        pred_boot = pred[draws].mean(axis=1)
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


def main() -> None:
    rows = json.loads((RESULTS / "validation_predictions.json").read_text())
    rows = [r for r in rows if r["model"] == "gemma_270m" and r["temperature"] == .1 and r["scheme"] == "feature_prompt"]
    observed = np.asarray([r["observed"] for r in rows], dtype=float)
    predicted = np.asarray([r["predictions"]["full_gain_model"] for r in rows], dtype=float)
    attribution = np.asarray([r["predictions"]["attribution_only"] for r in rows], dtype=float)

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

    summary = {
        "experiment": "closed_loop_gain_law_diagnostics",
        "selection": {"model": "gemma_270m", "temperature": 0.1, "scheme": "feature_prompt"},
        "full_gain_model": metrics(observed, predicted),
        "attribution_only": metrics(observed, attribution),
        "tail_sensitivity": tail,
        "binned_calibration": binned_calibration(observed, predicted),
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "unit": "held-out feature-prompt cell"},
    }
    (RESULTS / "diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
