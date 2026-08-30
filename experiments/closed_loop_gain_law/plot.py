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
    gain = load("experiments/closed_loop_gain_law/results/summary.json")["models"]
    aligned = {
        "270M": load("experiments/attribution_channel_calibration/results/aligned_summary.json")["analyses"],
        "1B": load("experiments/gemma_1b_attribution_channel_calibration/results/aligned_summary.json")["analyses"],
    }
    colors = {"270M": "#2878B5", "1B": "#D95319"}
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.45))

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
    models = [("gemma_270m","270M"),("gemma_1b","1B")]
    xs = np.arange(4); temps = ["0.1","0.25","0.5","1.0"]
    for key, label in models:
        ax.plot(xs, [gain[key][t]["attribution_only"]["r2"] for t in temps], ls=":", marker="o",
                color=colors[label], alpha=.65, label=f"{label}: attribution")
        ax.plot(xs, [gain[key][t]["attribution_full"]["r2"] for t in temps], ls="-", marker="o",
                color=colors[label], label=f"{label}: + susceptibility + leverage")
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(xs, temps)
    ax.set(xlabel="Article temperature", ylabel="LOFO $R^2$", title="C  Public gain-law prediction", ylim=(-.7,1.0))
    ax.legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.18)
    fig.tight_layout(); fig.savefig(OUT, dpi=220, bbox_inches="tight"); print(OUT)


if __name__ == "__main__": main()
