#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]


def main() -> None:
    summary = json.loads((EXP_DIR / "results/summary.json").read_text())
    regimes = ["between", "within"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.05))
    x = np.arange(2)
    axes[0].bar(x - .2, [summary["regimes"][r]["mediator_tv"]["mean"] for r in regimes], .4, label="Public-token relay")
    axes[0].bar(x + .2, [summary["regimes"][r]["residual_tv"]["mean"] for r in regimes], .4, label="Fixed-token residual")
    axes[0].set_xticks(x, ["Between class", "Within class"])
    axes[0].set_ylabel("TV-vector magnitude")
    axes[0].set_title("A  Same natural patch")
    axes[0].legend(frameon=False, fontsize=7)

    keys = ["target_delta_delta", "wrong_target_delta_delta", "sign_reversed_delta_delta"]
    labels = ["Target", "Wrong target", "Sign reversed"]
    width = .24
    for idx, (key, label) in enumerate(zip(keys, labels)):
        axes[1].bar(x + (idx - 1) * width, [summary["regimes"][r][key]["mean"] for r in regimes], width, label=label)
    axes[1].axhline(0, color="black", lw=.7)
    axes[1].set_xticks(x, ["Between", "Within"])
    axes[1].set_ylabel("Target-minus-source ΔΔ")
    axes[1].set_title("B  Natural controls")
    axes[1].legend(frameon=False, fontsize=7)

    width = .34
    axes[2].bar(x - width/2, [summary["regimes"][r]["target_logit_change"]["mean"] for r in regimes], width, label="Target logit")
    axes[2].bar(x + width/2, [summary["regimes"][r]["source_logit_change"]["mean"] for r in regimes], width, label="Source logit")
    axes[2].axhline(0, color="black", lw=.7)
    axes[2].set_xticks(x, ["Between", "Within"])
    axes[2].set_ylabel("Absolute logit change")
    axes[2].set_title("C  Enhancement vs suppression")
    axes[2].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = ROOT / "manuscript/figures/fig_natural_carrier_regimes.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
