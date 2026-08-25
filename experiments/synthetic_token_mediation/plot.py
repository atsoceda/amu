#!/usr/bin/env python3
"""Plot synthetic mediation discriminator and k-token residual curves."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXP_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = EXP_DIR / "results" / "summary.json"
OUT_PATH = (
    EXP_DIR.parents[1] / "manuscript" / "figures" / "fig_synthetic_mediation_validation.png"
)


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)

    # Panel A: TE / article-only / residual TV
    ax = axes[0]
    metrics = ["total_tv", "article_only_tv", "residual_tv"]
    labels = ["Total", "Article-only", "Residual"]
    x = np.arange(len(labels))
    width = 0.35
    for i, mech in enumerate(("mediated", "direct")):
        means = [summary[mech][m]["mean"] for m in metrics]
        los = [summary[mech][m]["lo"] for m in metrics]
        his = [summary[mech][m]["hi"] for m in metrics]
        yerr = np.vstack(
            [
                np.maximum(0.0, np.array(means) - np.array(los)),
                np.maximum(0.0, np.array(his) - np.array(means)),
            ]
        )
        ax.bar(
            x + (i - 0.5) * width,
            means,
            width=width,
            yerr=yerr,
            capsize=3,
            label=mech,
            color="#4C78A8" if mech == "mediated" else "#F58518",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Noun-distribution TV")
    ax.set_title("A. Factorial decomposition")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: k-token residual curve
    ax = axes[1]
    for mech, color in (("mediated", "#4C78A8"), ("direct", "#F58518")):
        curve = summary[mech]["residual_control_curve"]
        ks = [pt["k"] for pt in curve]
        means = [pt["residual_tv"]["mean"] for pt in curve]
        los = [pt["residual_tv"]["lo"] for pt in curve]
        his = [pt["residual_tv"]["hi"] for pt in curve]
        ax.plot(ks, means, marker="o", color=color, label=mech)
        ax.fill_between(ks, los, his, color=color, alpha=0.2)
    ax.set_xlabel("Forced continuation length k under do(a)")
    ax.set_ylabel("Token-clamped residual TV")
    ax.set_title("B. Residual-control curve (force a)")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
