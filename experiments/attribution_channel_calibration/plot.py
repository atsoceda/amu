#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]


def main() -> None:
    rows = json.loads((EXP_DIR / "results/feature_rows.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    colors = {"high_article_high_future":"#6a3d9a","high_article_low_future":"#e66101","low_article_high_future":"#1f78b4","low_article_low_future":"#777777"}
    for row in rows:
        color = colors[row["stratum"]]
        axes[0].scatter(row["article_attribution"], row["article_margin_effect"], color=color, s=24, alpha=.8)
        axes[1].scatter(row["future_attribution"], row["mediator_tv"], color=color, s=24, alpha=.8)
        axes[2].scatter(row["future_attribution"], row["residual_tv_treated"], color=color, s=24, alpha=.8)
    axes[0].set(title="A  Article calibration", xlabel="Article attribution", ylabel="Article-margin effect")
    axes[1].set(title="B  Public relay", xlabel="Future-token attribution", ylabel="Mediator TV")
    axes[2].set(title="C  Fixed-token persistence", xlabel="Future-token attribution", ylabel="Residual TV")
    for ax in axes:
        ax.axhline(0, color="black", lw=.6)
        ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    out = ROOT / "manuscript/figures/fig_attribution_channel_calibration.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
