#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
RESULTS = EXP_DIR / "results"


def main() -> None:
    dense = json.loads((RESULTS / "dense_grid.json").read_text())
    policy = json.loads((RESULTS / "policy_rows.json").read_text())
    summary = json.loads((RESULTS / "summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.15))

    for index in sorted({row["index"] for row in dense}):
        rows = [row for row in dense if row["index"] == index]
        axes[0].plot([r["offset"] for r in rows], [r["fp32_margin"] for r in rows], lw=1.5, alpha=.8)
        axes[0].step([r["offset"] for r in rows], [r["native_margin"] for r in rows], where="mid", lw=.8, alpha=.35, color="gray")
    axes[0].axhline(0, color="black", lw=.7)
    axes[0].set(title="A  Precision audit", xlabel="Gain offset from boundary", ylabel="an - a logit margin")

    temps = sorted({row["temperature"] for row in policy})
    for temperature in temps:
        rows = [row for row in policy if row["temperature"] == temperature]
        axes[1].plot([0, 1], [np.mean([r["pi_an_low"] for r in rows]), np.mean([r["pi_an_high"] for r in rows])], marker="o", label=f"tau={temperature:g}")
    axes[1].set_xticks([0, 1], ["below", "above"])
    axes[1].set_ylim(-.03, 1.03)
    axes[1].set(title="B  Stochastic article policy", ylabel="Conditional P(an)")
    axes[1].legend(frameon=False, fontsize=7)

    x = np.arange(len(temps))
    axes[2].bar(x - .18, [summary["policy"][str(t)]["policy_tv"]["mean"] for t in temps], width=.36, label="Policy relay")
    axes[2].bar(x + .18, [summary["policy"][str(t)]["fixed_tv"]["mean"] for t in temps], width=.36, label="Fixed token")
    axes[2].set_xticks(x, [str(t) for t in temps])
    axes[2].set(title="C  Channel magnitude", xlabel="Article temperature", ylabel="TV-vector magnitude")
    axes[2].legend(frameon=False, fontsize=7)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = ROOT / "manuscript/figures/fig_boundary_precision_policy.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
