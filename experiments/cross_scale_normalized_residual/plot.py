#!/usr/bin/env python3
"""Cross-scale residual-patch comparison at aligned relative depth and strength."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/figures/fig_cross_scale_normalized_residual.png"


def mean_group(rows, keys, value):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(float(value(row)))
    return {key: float(np.mean(vals)) for key, vals in groups.items()}


def main():
    rows = json.loads((Path(__file__).parent / "results/rows.json").read_text())
    dev270 = json.loads((ROOT / "experiments/natural_residual_carrier_regimes/results/dev_rows.json").read_text())
    dev1b = json.loads((ROOT / "experiments/gemma_1b_residual_scale/results/layer_rows.json").read_text())
    dev1b = [r for r in dev1b if r["split"] == "dev"]
    for r in dev270:
        r["relative_depth"] = r["layer"] / 17

    colors = {"gemma_270m": "#2878B5", "gemma_1b": "#D95319"}
    labels = {"gemma_270m": "Gemma 270M", "gemma_1b": "Gemma 1B"}
    styles = {"between": "-", "within": "--"}
    regime_labels = {"between": "cross-class", "within": "within-class"}
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))

    ax = axes[0, 0]
    for model, dev in (("gemma_270m", dev270), ("gemma_1b", dev1b)):
        means = mean_group(dev, ["relative_depth"], lambda r: r["delta_delta"])
        xs = sorted(k[0] for k in means)
        ax.plot(xs, [means[(x,)] for x in xs], color=colors[model], lw=2, label=labels[model])
    ax.set(xlabel="Relative depth", ylabel="Development target $\\Delta\\Delta$", title="A  Layer profile")
    ax.legend(frameon=False)

    panels = [
        (axes[0, 1], lambda r: r["fraction_gap_closed"], "Fraction of target gap closed", "B  Normalized target efficacy"),
        (axes[1, 0], lambda r: r["stochastic"]["0.1"]["public_tv"], "Public TV ($\\tau=0.1$)", "C  Public route"),
        (axes[1, 1], lambda r: r["stochastic"]["0.1"]["private_tv"], "Private TV ($\\tau=0.1$)", "D  Fixed-token route"),
    ]
    for ax, value, ylabel, title in panels:
        means = mean_group(rows, ["model", "regime", "strength"], value)
        for model in ("gemma_270m", "gemma_1b"):
            for regime in ("between", "within"):
                xs = sorted(k[2] for k in means if k[0] == model and k[1] == regime)
                ax.plot(xs, [means[(model, regime, x)] for x in xs], color=colors[model],
                        ls=styles[regime], marker="o", lw=2,
                        label=f"{labels[model]}, {regime_labels[regime]}")
        ax.set(xlabel="Patch strength", ylabel=ylabel, title=title)
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=2)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.18)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
