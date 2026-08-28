#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "experiments/pre_article_public_private_factorial/results/summary.json"
OUTPUT = ROOT / "manuscript/figures/fig_pre_article_public_private.png"


def err(block):
    mean = float(block["mean"])
    return mean, [mean - float(block["lo"]), float(block["hi"]) - mean]


def main():
    summary = json.loads(SUMMARY.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7))
    articles = ["a", "an"]
    x = np.arange(2)
    width = 0.34
    target_means, target_err = [], np.zeros((2, 2))
    random_means, random_err = [], np.zeros((2, 2))
    for i, article in enumerate(articles):
        tm, te = err(summary["heldout"][article]["target_delta_delta"])
        rm, re = err(summary["heldout"][article]["random_delta_delta"])
        target_means.append(tm); target_err[:, i] = te
        random_means.append(rm); random_err[:, i] = re
    ax = axes[0]
    ax.bar(x - width/2, target_means, width, yerr=target_err, capsize=3, label="Target residual", color="#2563a6")
    ax.bar(x + width/2, random_means, width, yerr=random_err, capsize=3, label="Matched random", color="#9ca3af")
    ax.axhline(0, color="black", linewidth=.7)
    ax.set_xticks(x, ["Insert a", "Insert an"])
    ax.set_ylabel("Target-minus-source $\\Delta\\Delta$ (logits)")
    ax.set_title("A  Private-state effect from pre-article site", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=.2)

    ax = axes[1]
    labels = ["Public effect\nprivate off", "Public effect\nprivate on", "Interaction"]
    blocks = [summary["factorial"]["public_effect_off"], summary["factorial"]["public_effect_on"], summary["factorial"]["interaction"]]
    means, errors = [], np.zeros((2, 3))
    for i, block in enumerate(blocks):
        means.append(float(block["mean"]))
        errors[:, i] = [float(block["mean"])-float(block["lo"]), float(block["hi"])-float(block["mean"])]
    ax.bar(np.arange(3), means, yerr=errors, capsize=3, color=["#d97706", "#4c8dc9", "#7c3aed"])
    ax.axhline(0, color="black", linewidth=.7)
    ax.set_xticks(np.arange(3), labels, fontsize=8)
    ax.set_ylabel("Target-minus-source contrast (logits)")
    ax.set_title("B  Public-token × private-state factorial", loc="left", weight="bold")
    ax.grid(axis="y", alpha=.2)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
