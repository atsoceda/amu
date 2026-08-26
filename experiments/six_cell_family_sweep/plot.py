#!/usr/bin/env python3
"""Figures for the six-cell family sweep and updated S1 decomposition."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "experiments/six_cell_family_sweep/results/summary.json"
REFERENCE = ROOT / "experiments/fixed_article_residual_reference/results/summary.json"
FIG = ROOT / "manuscript/figures"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def interval_yerr(block: dict) -> tuple[float, np.ndarray]:
    mean = float(block["mean"])
    return mean, np.array([[mean - float(block["lo"])], [float(block["hi"]) - mean]])


def plot_schematic() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.7))

    ax = axes[0]
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("What later effect is transmitted through")
    ax.set_ylabel("What is represented before the article")
    ax.set_xticks([0.2, 0.9], ["Generated-token identity", "Matched-prefix leftover"])
    ax.set_yticks([0.15, 0.55, 0.95], ["Phonological class", "Lexical token", "Semantic concept"])
    ax.scatter([0.2], [0.15], s=80, color="#2563a6", zorder=3)
    ax.annotate(
        "S1 5×\n(this paper)",
        (0.2, 0.15),
        textcoords="offset points",
        xytext=(12, 10),
        fontsize=8,
        color="#2563a6",
    )
    ax.set_title("A  Two independent questions", loc="left", weight="bold")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.set_axis_off()
    ax.set_title("B  Six-cell design", loc="left", weight="bold")
    columns = ["Free article", "Insert a", "Insert an"]
    rows = ["Intervention off", "Intervention on"]
    table = ax.table(
        cellText=[
            ["Y(0, B0)", "Y(0, a)", "Y(0, an)"],
            ["Y(1, B1)", "Y(1, a)", "Y(1, an)"],
        ],
        rowLabels=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.15, 2.0)
    ax.text(
        0.5,
        0.08,
        "Total: free on vs free off.  Token substitution: off + treated article vs off + baseline article.\n"
        "Matched-prefix leftover: on vs off under the same inserted article.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        transform=ax.transAxes,
    )

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_hypotheses_protocol.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_s1_and_family() -> None:
    sweep = load(SWEEP)
    analysis = sweep["analysis"]
    s1 = analysis["S1_5x"]
    colors = {
        "total": "#1f4e79",
        "mediator": "#4c8dc9",
        "residual": "#d97706",
        "article": "#6b7280",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.7))

    ax = axes[0]
    items = [
        ("Total free-generation", s1["decomposition_all_articled"]["total_tv"], colors["total"]),
        ("Token substitution", s1["decomposition_all_articled"]["mediator_tv"], colors["mediator"]),
        ("Matched-prefix leftover", s1["decomposition_all_articled"]["residual_tv"], colors["residual"]),
        ("Article prefix only\n(no intervention)", s1["generic_article_prefix_tv"], colors["article"]),
    ]
    xs = np.arange(len(items))
    means = []
    yerr = np.zeros((2, len(items)))
    for i, (_, block, _) in enumerate(items):
        mean, err = interval_yerr(block)
        means.append(mean)
        yerr[:, i] = err[:, 0]
    ax.bar(xs, means, color=[c for _, _, c in items], yerr=yerr, capsize=2.5, linewidth=0)
    for x, mean in zip(xs, means):
        ax.text(x, mean + 0.03, f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, [label for label, _, _ in items], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Full-vocabulary total variation")
    ax.set_title("A  S1 $5\\times$ vector split (N=20)", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    order = ["S1_5x", "S1_3x", "S1_1.5x", "S2_5x", "S3_5x", "S4_5x", "S1_random_5x"]
    labels = ["S1 5×", "S1 3×", "S1 1.5×", "S2", "S3", "S4", "Random"]
    x = np.arange(len(order))
    width = 0.36
    total_means, total_err = [], np.zeros((2, len(order)))
    res_means, res_err = [], np.zeros((2, len(order)))
    for i, key in enumerate(order):
        tmean, terr = interval_yerr(analysis[key]["decomposition_all_articled"]["total_tv"])
        rmean, rerr = interval_yerr(analysis[key]["matched_prefix"]["an"])
        total_means.append(tmean)
        res_means.append(rmean)
        total_err[:, i] = terr[:, 0]
        res_err[:, i] = rerr[:, 0]
    ax.bar(
        x - width / 2,
        total_means,
        width,
        color="#1f4e79",
        yerr=total_err,
        capsize=2,
        label="Total (articled free cells)",
        linewidth=0,
    )
    ax.bar(
        x + width / 2,
        res_means,
        width,
        color="#d97706",
        yerr=res_err,
        capsize=2,
        label="Leftover under inserted an",
        linewidth=0,
    )
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Full-vocabulary total variation")
    ax.set_title("B  Handles and S1 dose", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG / "fig_mediation_decomposition.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reference() -> None:
    if not REFERENCE.exists():
        return
    reference = load(REFERENCE)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
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
    ax.set_title("Full-residual sensitivity control", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_full_residual_reference.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_schematic()
    plot_s1_and_family()
    plot_reference()


if __name__ == "__main__":
    main()
