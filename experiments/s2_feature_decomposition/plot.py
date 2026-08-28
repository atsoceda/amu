#!/usr/bin/env python3
"""Appendix figure: S2 single-feature and leave-one-out decomposition."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "experiments/s2_feature_decomposition/results/summary.json"
NULL = ROOT / "experiments/l5_fixed_token_matched_null/results/summary.json"
OUTPUT = ROOT / "manuscript/figures/fig_s2_decomposition.png"


def err(block):
    mean = float(block["mean"])
    return mean, [mean - float(block["lo"]), float(block["hi"]) - mean]


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    null = json.loads(NULL.read_text())
    conditions = summary["conditions"]
    order = [
        "S2_full",
        "single_L5_F383",
        "single_L14_F1949",
        "single_L13_F10231",
        "single_L11_F12690",
        "loo_L5_F383",
        "loo_L14_F1949",
        "loo_L13_F10231",
        "loo_L11_F12690",
    ]
    labels = [
        "S2 full",
        "L5/F383",
        "L14/F1949",
        "L13/F10231",
        "L11/F12690",
        "LOO F383",
        "LOO F1949",
        "LOO F10231",
        "LOO F12690",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7))

    ax = axes[0]
    xs = np.arange(len(order))
    tv_means, tv_err = [], np.zeros((2, len(order)))
    twin_means, twin_err = [], np.zeros((2, len(order)))
    for i, key in enumerate(order):
        m, e = err(conditions[key]["forced_an_tv"])
        tv_means.append(m)
        tv_err[:, i] = e
        m, e = err(conditions[key]["twin_delta_delta_an"])
        twin_means.append(m)
        twin_err[:, i] = e
    width = 0.38
    ax.bar(xs - width / 2, tv_means, width, yerr=tv_err, capsize=2, color="#4c8dc9", label="Fixed-$an$ TV", linewidth=0)
    ax.bar(xs + width / 2, twin_means, width, yerr=twin_err, capsize=2, color="#d97706", label=r"Twin $\Delta\Delta$", linewidth=0)
    ax.set_xticks(xs, labels, rotation=28, ha="right", fontsize=7.5)
    ax.set_ylabel("Effect under inserted $an$")
    ax.set_title("A  Single-feature and leave-one-out", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1]
    legal = [conditions[key]["legal_free_rate"] for key in order]
    ax.bar(xs, legal, color="#059669", linewidth=0)
    ax.set_xticks(xs, labels, rotation=28, ha="right", fontsize=7.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Legal free article--noun rate")
    ax.set_title("B  Continuation legality", loc="left", weight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    ax.axhline(1.0, color="#9ca3af", linestyle="--", linewidth=0.8)
    ctrl = null["control_tv_means"]["mean"]
    ax.text(
        0.02,
        0.08,
        f"Four matched L5 controls: mean TV {ctrl:.3f}; empirical $p=0.20$",
        transform=ax.transAxes,
        fontsize=7.3,
        color="#4b5563",
    )

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
