#!/usr/bin/env python3
"""Main-text Figures 1--2 and the family/dose appendix figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "experiments/six_cell_family_sweep/results/summary.json"
REPLAY = ROOT / "experiments/s1_replay_rescue/results/summary.json"
BOUNDARY = ROOT / "experiments/prompt_aligned_article_boundary/results/summary.json"
REFERENCE = ROOT / "experiments/fixed_article_residual_reference/results/summary.json"
FIG = ROOT / "manuscript/figures"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def interval_yerr(block: dict) -> tuple[float, np.ndarray]:
    mean = float(block["mean"])
    return mean, np.array([[mean - float(block["lo"])], [float(block["hi"]) - mean]])


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22)


def _box(ax, xy, w, h, text, facecolor, fontsize=8.5, edgecolor="#1f2937"):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.1,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
        zorder=3,
        wrap=True,
    )
    return patch


def _arrow(ax, start, end, color="#111827"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=color,
            zorder=1,
        )
    )


def plot_schematic() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.85), gridspec_kw={"width_ratios": [1.15, 0.95]})

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.set_axis_off()
    ax.set_title("A  Public relay and private contextual route", loc="left", weight="bold", fontsize=10)

    _box(ax, (0.25, 2.35), 2.15, 1.35, r"$I_t$" + "\nat pre-article $P$", "#dbeafe", 8.2)
    _box(ax, (3.55, 4.15), 2.35, 1.25, r"$B_{t+1}$" + "\nvisible article", "#fde68a", 8.2)
    _box(ax, (3.55, 0.55), 2.35, 1.35, r"$H_P$" + "\nprivate state from $P$", "#e9d5ff", 8.2)
    _box(ax, (7.35, 2.35), 2.35, 1.35, r"$Y_{t+2}$" + "\nnoun token", "#bbf7d0", 8.2)

    _arrow(ax, (2.45, 3.45), (3.50, 4.55), "#1d4ed8")
    _arrow(ax, (5.95, 4.75), (7.35, 3.55), "#1d4ed8")
    _arrow(ax, (2.45, 2.55), (3.50, 1.55), "#7c3aed")
    _arrow(ax, (5.95, 1.25), (7.35, 2.55), "#7c3aed")

    ax.text(3.15, 5.55, r"public: $\mathrm{do}(B)$ clamps token identity", fontsize=7.4, color="#1d4ed8", ha="left")
    ax.text(3.15, 0.12, "private: residual patch at $P$", fontsize=7.4, color="#7c3aed", ha="left")
    ax.text(
        5.0,
        3.05,
        "S1 is relay-dominant\nalong the public path",
        fontsize=7.1,
        color="#1f2937",
        ha="center",
        va="center",
        style="italic",
    )

    ax = axes[1]
    ax.set_axis_off()
    ax.set_title("B  Six-cell assay", loc="left", weight="bold", fontsize=10)
    columns = ["Free article", r"do($a$)", r"do($an$)"]
    rows = [r"$I$ off", r"$I$ on"]
    table = ax.table(
        cellText=[
            [r"$Y(0,B_0)$", r"$Y(0,a)$", r"$Y(0,an)$"],
            [r"$Y(1,B_1)$", r"$Y(1,a)$", r"$Y(1,an)$"],
        ],
        rowLabels=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.18, 2.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#e5e7eb")
            cell.set_text_props(weight="bold")
        elif col == -1:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(weight="bold")
    ax.text(
        0.5,
        0.08,
        r"$E_T=Y(1,B_1)-Y(0,B_0)$   $E_M=Y(0,B_1)-Y(0,B_0)$   $E_R=Y(1,B_1)-Y(0,B_1)$"
        "\nFree is a policy, not a treatment level of $B$.  Identity $E_T=E_M+E_R$ is exact.",
        ha="center",
        va="bottom",
        fontsize=7.4,
        transform=ax.transAxes,
    )

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_hypotheses_protocol.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_public_relay() -> None:
    sweep = load(SWEEP)
    s1 = sweep["analysis"]["S1_5x"]
    replay = load(REPLAY)
    boundary = load(BOUNDARY)
    colors = {
        "total": "#1f4e79",
        "mediator": "#4c8dc9",
        "residual": "#d97706",
        "fixed": "#9ca3af",
    }

    fig, axes = plt.subplots(1, 3, figsize=(10.9, 3.55), gridspec_kw={"width_ratios": [1.05, 0.95, 1.15]})

    ax = axes[0]
    items = [
        ("Total\nfree path", s1["decomposition_all_articled"]["total_tv"], colors["total"]),
        ("Token\nsubstitution", s1["decomposition_all_articled"]["mediator_tv"], colors["mediator"]),
        ("Matched-prefix\nleftover", s1["decomposition_all_articled"]["residual_tv"], colors["residual"]),
    ]
    xs = np.arange(len(items))
    means, yerr = [], np.zeros((2, len(items)))
    for i, (_, block, _) in enumerate(items):
        mean, err = interval_yerr(block)
        means.append(mean)
        yerr[:, i] = err[:, 0]
    ax.bar(xs, means, color=[c for _, _, c in items], yerr=yerr, capsize=2.5, linewidth=0, width=0.72)
    for x, mean in zip(xs, means):
        ax.text(x, mean + 0.035, f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, [label for label, _, _ in items], fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Full-vocabulary total variation")
    ax.set_title(r"A  S1 $5\times$ vector split", loc="left", weight="bold", fontsize=10)
    style_axis(ax)

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_axis_off()
    ax.set_title("B  Phenocopy and rescue", loc="left", weight="bold", fontsize=10)
    _box(ax, (0.4, 7.15), 4.15, 1.85, "S1 on, free $B_1$\ntreated top noun", "#dbeafe", 8)
    _box(ax, (5.45, 7.15), 4.15, 1.85, "S1 off, insert $B_1$\nsame top noun", "#fef3c7", 8)
    _arrow(ax, (4.55, 8.05), (5.40, 8.05), "#1d4ed8")
    ax.text(5.0, 9.35, "phenocopy 20/20", ha="center", fontsize=7.6, color="#1d4ed8")

    _box(ax, (0.4, 2.55), 4.15, 1.85, "S1 on, restore $B_0$\nbaseline top noun", "#dbeafe", 8)
    _box(ax, (5.45, 2.55), 4.15, 1.85, "S1 off, free $B_0$\nsame top noun", "#dcfce7", 8)
    _arrow(ax, (4.55, 3.45), (5.40, 3.45), "#059669")
    ax.text(5.0, 4.75, "rescue 20/20", ha="center", fontsize=7.6, color="#059669")
    leftover = replay["rescue_residual_tv"]["mean"]
    ax.text(
        5.0,
        0.55,
        f"restored-article leftover TV {leftover:.3f}",
        ha="center",
        fontsize=7.3,
        color="#4b5563",
    )

    ax = axes[2]
    items = [
        ("Free path", boundary["total_tv"], colors["total"]),
        ("Token\nsubstitution", boundary["token_substitution_tv"], colors["mediator"]),
        ("Fixed $a$", boundary["fixed_a_residual_tv"], colors["fixed"]),
        ("Fixed $an$", boundary["fixed_an_residual_tv"], colors["residual"]),
    ]
    xs = np.arange(len(items))
    means, yerr = [], np.zeros((2, len(items)))
    for i, (_, block, _) in enumerate(items):
        mean, err = interval_yerr(block)
        means.append(mean)
        yerr[:, i] = err[:, 0]
    ax.bar(xs, means, color=[c for _, _, c in items], yerr=yerr, capsize=2.5, linewidth=0, width=0.72)
    for x, mean in zip(xs, means):
        ax.text(x, mean + 0.03, f"{mean:.3f}", ha="center", va="bottom", fontsize=7.6)
    ax.set_xticks(xs, [label for label, _, _ in items], fontsize=7.6)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("TV across the gain bracket")
    ax.set_title("C  Prompt-aligned boundary", loc="left", weight="bold", fontsize=10)
    ax.text(
        0.5,
        0.92,
        "N=19; gains at most 0.0039 apart",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.2,
        color="#4b5563",
    )
    style_axis(ax)

    fig.tight_layout()
    fig.savefig(FIG / "fig_mediation_decomposition.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_family_appendix() -> None:
    sweep = load(SWEEP)
    analysis = sweep["analysis"]
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
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
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Full-vocabulary total variation")
    ax.set_title("Family and S1-dose six-cell leftovers", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_family_dose_appendix.png", dpi=220, bbox_inches="tight")
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
    ax.set_title("Article-position full-residual sensitivity control", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_full_residual_reference.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_schematic()
    plot_public_relay()
    plot_family_appendix()
    plot_reference()


if __name__ == "__main__":
    main()
