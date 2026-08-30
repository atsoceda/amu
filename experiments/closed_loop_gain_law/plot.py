#!/usr/bin/env python3
"""Plot local calibration, matched future estimands, and gain-law prediction."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/figures/fig_attribution_channel_calibration.png"


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    validation = load("experiments/closed_loop_gain_law/results/validation_summary.json")["models"]
    predictions = load("experiments/closed_loop_gain_law/results/validation_predictions.json")
    aligned = {
        "270M": load("experiments/attribution_channel_calibration/results/aligned_summary.json")["analyses"],
        "1B": load("experiments/gemma_1b_attribution_channel_calibration/results/aligned_summary.json")["analyses"],
    }
    colors = {"270M": "#2878B5", "1B": "#D95319"}
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    axes = axes.flat

    ax = axes[0]
    local = {"270M": (.530, .200, .772), "1B": (.432, .014, .729)}
    for i, model in enumerate(("270M", "1B")):
        v, lo, hi = local[model]
        ax.errorbar(i, v, yerr=[[v-lo], [hi-v]], fmt="o", ms=7, capsize=4, color=colors[model])
    ax.axhline(0, color="black", lw=.8); ax.set_xticks([0,1], ["270M","1B"])
    ax.set(ylabel="Spearman $\\rho$", title="A  Local margin calibration", ylim=(-.2,.9))

    ax = axes[1]
    metrics = [("signed_future_vs_signed_fixed_target", "Target logit"),
               ("signed_future_vs_signed_fixed_target_minus_source", "Target-source")]
    for j, (key, label) in enumerate(metrics):
        for offset, model in ((-.1,"270M"),(.1,"1B")):
            d = aligned[model][key]
            ax.errorbar(j+offset, d["rho"], yerr=[[d["rho"]-d["lo"]], [d["hi"]-d["rho"]]],
                        fmt="o", capsize=4, color=colors[model], label=model if j == 0 else None)
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(range(2), [x[1] for x in metrics])
    ax.set(title="B  Matched fixed-token estimands", ylim=(-.65,.75)); ax.legend(frameon=False)

    ax = axes[2]
    subset = [r for r in predictions if r["model"] == "gemma_270m" and r["temperature"] == .1 and r["scheme"] == "feature_prompt"]
    observed = np.asarray([r["observed"] for r in subset])
    for key, label, color, marker in (("attribution_only", "Attribution only", "#999999", "x"),
                                      ("full_gain_model", "Full composition", colors["270M"], "o")):
        predicted = np.asarray([r["predictions"][key] for r in subset])
        ax.scatter(predicted, observed, s=12, alpha=.35, color=color, marker=marker, label=label)
    limit = max(float(observed.max()), max(r["predictions"]["full_gain_model"] for r in subset))
    ax.plot([0, limit], [0, limit], color="black", lw=.8, ls="--")
    ax.set(xlabel="Predicted public TV", ylabel="Observed public TV", title="C  270M unseen feature + prompt")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[3]
    names = [("constant", "Constant"), ("attribution_only", "Attribution"),
             ("susceptibility_only", "Susceptibility"), ("full_gain_model", "Full")]
    x = np.arange(len(names)); width = .36
    for offset, (key, label) in zip((-.18,.18), (("gemma_270m","270M"),("gemma_1b","1B"))):
        vals = [validation[key]["0.1"]["feature_prompt"][name]["r2"] for name,_ in names]
        ax.bar(x+offset, vals, width, color=colors[label], label=label)
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(x, [label for _,label in names], rotation=18)
    ax.set(ylabel="Two-way held-out $R^2$", title="D  Baselines at $\\tau=0.1$", ylim=(-.65,1.0)); ax.legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.18)
    fig.tight_layout(); fig.savefig(OUT, dpi=220, bbox_inches="tight"); print(OUT)


if __name__ == "__main__": main()
