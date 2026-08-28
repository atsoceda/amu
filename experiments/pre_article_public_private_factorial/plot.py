#!/usr/bin/env python3
"""Main-text Figure 3: private-route capacity and 2x2 interaction."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "experiments/pre_article_public_private_factorial/results/summary.json"
DERIVED = ROOT / "experiments/pre_article_public_private_factorial/results/derived_controls.json"
OUTPUT = ROOT / "manuscript/figures/fig_pre_article_public_private.png"


def err(block):
    mean = float(block["mean"])
    return mean, [mean - float(block["lo"]), float(block["hi"]) - mean]


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    ax.axhline(0, color="black", linewidth=0.7)


def main():
    summary = json.loads(SUMMARY.read_text())
    derived = json.loads(DERIVED.read_text())
    pairs = derived["pairs"]
    fig, axes = plt.subplots(1, 3, figsize=(10.9, 3.65))

    ax = axes[0]
    xs = np.arange(len(pairs))
    target = np.array([p["delta_delta_a"] for p in pairs])
    random = np.array([p["random_delta_delta_a_mean"] for p in pairs])
    ax.scatter(xs, target, color="#2563a6", s=28, zorder=3, label="Target residual")
    ax.scatter(xs, random, color="#9ca3af", s=28, zorder=3, label="Matched random")
    for i, (t, r) in enumerate(zip(target, random)):
        ax.plot([i, i], [r, t], color="#cbd5e1", linewidth=1.1, zorder=1)
    ax.axhline(0, color="black", linewidth=0.7)
    dmean = derived["paired_d_a"]["mean"]
    ax.set_xticks(xs, [str(i + 1) for i in xs], fontsize=7)
    ax.set_xlabel("Held-out pair")
    ax.set_ylabel(r"Target-minus-source $\Delta\Delta$ (logits)")
    ax.set_title("A  Paired private effects under $a$", loc="left", weight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=7.4, loc="upper right")
    ax.text(
        0.03,
        0.95,
        f"paired D = {dmean:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=7.2,
        color="#1f2937",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1]
    cells = derived["cell_means"]
    x = np.array([0, 1])
    off = [cells["baseline_a"]["mean"], cells["baseline_an"]["mean"]]
    on = [cells["patched_a"]["mean"], cells["patched_an"]["mean"]]
    rnd = [cells["random_patched_a"]["mean"], cells["random_patched_an"]["mean"]]
    off_lo = [cells["baseline_a"]["lo"], cells["baseline_an"]["lo"]]
    off_hi = [cells["baseline_a"]["hi"], cells["baseline_an"]["hi"]]
    on_lo = [cells["patched_a"]["lo"], cells["patched_an"]["lo"]]
    on_hi = [cells["patched_a"]["hi"], cells["patched_an"]["hi"]]
    ax.errorbar(
        x,
        off,
        yerr=np.array([[m - lo, hi - m] for m, lo, hi in zip(off, off_lo, off_hi)]).T,
        fmt="o-",
        color="#d97706",
        linewidth=1.8,
        markersize=6,
        capsize=3,
        label="Private patch off",
    )
    ax.errorbar(
        x,
        on,
        yerr=np.array([[m - lo, hi - m] for m, lo, hi in zip(on, on_lo, on_hi)]).T,
        fmt="s-",
        color="#2563a6",
        linewidth=1.8,
        markersize=6,
        capsize=3,
        label="Target residual on",
    )
    ax.plot(x, rnd, linestyle="--", color="#9ca3af", linewidth=1.4, marker="D", markersize=5, label="Matched random on")
    ax.set_xticks(x, ["Insert $a$", "Insert $an$"])
    ax.set_ylabel("Target-minus-source contrast (logits)")
    ax.set_title(r"B  Raw $2\times 2$ response", loc="left", weight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22)

    ax = axes[2]
    labels = [r"$\Gamma$ target", r"$\Gamma$ random", "paired difference"]
    blocks = [derived["gamma_target"], derived["gamma_random"], derived["paired_gamma"]]
    colors = ["#7c3aed", "#9ca3af", "#1f4e79"]
    means, errors = [], np.zeros((2, 3))
    for i, block in enumerate(blocks):
        mean, te = err(block)
        means.append(mean)
        errors[:, i] = te
    ax.bar(np.arange(3), means, yerr=errors, capsize=3, color=colors, width=0.72, linewidth=0)
    for i, mean in enumerate(means):
        va = "bottom" if mean >= 0 else "top"
        offset = 0.06 if mean >= 0 else -0.06
        ax.text(i, mean + offset, f"{mean:.3f}", ha="center", va=va, fontsize=8)
    ax.set_xticks(np.arange(3), labels, fontsize=7.6)
    ax.set_ylabel("Interaction on target-minus-source (logits)")
    ax.set_title("C  Interaction vs random", loc="left", weight="bold", fontsize=10)
    style_axis(ax)

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
