#!/usr/bin/env python3
"""Render the main mediation decomposition figure from saved JSON."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FACTORIAL = ROOT / "experiments/s1_article_factorial/results/summary.json"
REFERENCE = ROOT / "experiments/fixed_article_residual_reference/results/summary.json"
OUTPUT = ROOT / "manuscript/figures/fig_mediation_decomposition.png"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    factorial = load(FACTORIAL)
    reference = load(REFERENCE)
    colors = {"recomputed": "#2563a6", "frozen": "#d97706"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.65))

    ax = axes[0]
    metrics = [
        ("total_tv_full_vocab", "Total"),
        ("article_only_tv_full_vocab", "Article only"),
        ("residual_tv_full_vocab", "Same-prefix\nresidual"),
    ]
    x = np.arange(len(metrics))
    width = 0.34
    for offset, mode in [(-width / 2, "recomputed"), (width / 2, "frozen")]:
        block = factorial["analysis"][mode]["descriptive_article_switch_subset"]
        means = [block[key]["mean"] for key, _ in metrics]
        lows = [mean - block[key]["lo"] for mean, (key, _) in zip(means, metrics)]
        highs = [block[key]["hi"] - mean for mean, (key, _) in zip(means, metrics)]
        bars = ax.bar(
            x + offset,
            means,
            width,
            color=colors[mode],
            label=mode.capitalize(),
            yerr=np.array([lows, highs]),
            capsize=2.5,
            linewidth=0,
        )
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 0.035,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90 if mean < 0.1 else 0,
            )
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Full-vocabulary total variation")
    ax.set_title("A  Generated-token decomposition (N=19)", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    layers = sorted(int(layer) for layer in reference["dev_layer_summary"])
    dev = [reference["dev_layer_summary"][str(layer)]["mean"] for layer in layers]
    ax.plot(layers, dev, color="#4b5563", marker="o", markersize=3, linewidth=1.5)
    selected = int(reference["selected_layer"])
    ax.axvline(selected, color="#9ca3af", linestyle="--", linewidth=1)
    target = reference["heldout_target_patch"]
    random = reference["heldout_random_controls"]
    for xpos, block, color, label in [
        (selected - 0.23, target, "#059669", "Held-out target"),
        (selected + 0.23, random, "#dc2626", "Matched random"),
    ]:
        mean = block["mean"]
        ax.errorbar(
            [xpos],
            [mean],
            yerr=[[mean - block["lo"]], [block["hi"] - mean]],
            fmt="s",
            color=color,
            markersize=5,
            capsize=3,
            label=label,
            zorder=4,
        )
    ax.set_xticks(range(0, 18, 2))
    ax.set_xlim(-0.5, 17.5)
    ax.set_xlabel("Patched decoder layer")
    ax.set_ylabel(r"Target-specific $\Delta\Delta$ (logits)")
    ax.set_title("B  Full-residual sensitivity control", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
