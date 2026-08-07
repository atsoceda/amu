#!/usr/bin/env python3
"""Regenerate figures 4–7 from experiment summary.json files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
FIGDIR = Path(__file__).resolve().parent / "figures"


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    e1 = json.loads((EXP / "selection_criterion_ablation/results/summary.json").read_text())
    e2 = json.loads((EXP / "planning_dose_response/results/summary.json").read_text())
    e3 = json.loads((EXP / "forced_content_lock/results/summary.json").read_text())
    e4 = json.loads((EXP / "trajectory_causal_tetrad/results/summary.json").read_text())

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 12,
        }
    )

    sets = [
        ("S1_dual_effect", "S1 Dual-effect"),
        ("S2_article_only", "S2 Article-only"),
        ("S3_content_only", "S3 Content-only"),
        ("S4_competing_a", "S4 Competing a"),
    ]
    labels = [x[1] for x in sets]
    wrap = [e1["set_results"][k]["summary"]["wrapper_like_rate"] for k, _ in sets]
    traj = [e1["set_results"][k]["summary"]["trajectory_like_rate"] for k, _ in sets]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.bar(x - w / 2, wrap, w, label="Wrapper-like rate", color="#4C72B0")
    ax.bar(
        x + w / 2,
        traj,
        w,
        label="Trajectory-like (chunking) rate",
        color="#C44E52",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate on 20 held-out prompts")
    ax.set_title("Feature-selection rule does not create content-preserving wrappers")
    ax.legend(frameon=False)
    for i, (wv, tv) in enumerate(zip(wrap, traj)):
        ax.text(i - w / 2, wv + 0.03, f"{wv:.2f}", ha="center", fontsize=10)
        ax.text(i + w / 2, tv + 0.03, f"{tv:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGDIR / "figure4_selection_ablation.png", dpi=160)
    plt.close()

    by = e2["set_results"]["S1_dual_effect"]["by_factor"]
    factors = sorted(by.keys(), key=float)
    traj_d = [by[f]["summary"]["trajectory_like_rate"] for f in factors]
    wrap_d = [by[f]["summary"]["wrapper_like_rate"] for f in factors]
    content_d = [by[f]["summary"]["content_preserved_rate"] for f in factors]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.plot(
        [float(f) for f in factors],
        traj_d,
        "o-",
        color="#C44E52",
        lw=2.5,
        ms=8,
        label="Trajectory-like (chunking)",
    )
    ax.plot(
        [float(f) for f in factors],
        content_d,
        "s--",
        color="#55A868",
        lw=2,
        ms=8,
        label="Content preserved",
    )
    ax.plot(
        [float(f) for f in factors],
        wrap_d,
        "^:",
        color="#4C72B0",
        lw=2,
        ms=8,
        label="Wrapper-like",
    )
    ax.set_xlabel("Amplify factor")
    ax.set_ylabel("Rate on 20 held-out prompts")
    ax.set_title("Dose–response for dual-effect features: no wrapper window")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False)
    ax.set_xticks([float(f) for f in factors])
    fig.tight_layout()
    fig.savefig(FIGDIR / "figure5_dose_response.png", dpi=160)
    plt.close()

    conds = [
        ("C1_article_push", "C1 Article-push"),
        ("C2_content_lock", "C2 Content-lock"),
        ("C3_dual", "C3 Dual"),
        ("C5_control_article", "Control"),
    ]
    labels = [c[1] for c in conds]
    traj = [e3["condition_summaries"][k]["trajectory_like_rate"] for k, _ in conds]
    content = [e3["condition_summaries"][k]["content_preserved_rate"] for k, _ in conds]
    illicit = [e3["condition_summaries"][k]["illicit_mismatch_rate"] for k, _ in conds]
    x = np.arange(len(labels))
    w = 0.28
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.bar(x - w, traj, w, label="Trajectory-like (chunking)", color="#C44E52")
    ax.bar(x, content, w, label="Content preserved", color="#55A868")
    ax.bar(x + w, illicit, w, label="Illicit mismatch", color="#8172B2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate")
    ax.set_title("Dual lock fails modular planning: package switching persists")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "figure6_dual_lock.png", dpi=160)
    plt.close()

    families = list(e4["families"].items())
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), sharey=True)
    for ax, (name, block) in zip(axes, families):
        order = ["baseline", "lof_zero", "gof_amplify", "control_amplify"]
        nice = ["Baseline", "LoF (zero)", "GoF (amplify)", "Control"]
        heights, colors, texts = [], [], []
        for cond in order:
            row = block["conditions"][cond]
            pkg = row["intervention_package"]
            word = row["intervention_word"] or "?"
            art = row["intervention_article"]
            texts.append(f"{art} {word}")
            if pkg == "twin":
                heights.append(1.0)
                colors.append("#C44E52")
            elif pkg == "baseline":
                heights.append(0.55)
                colors.append("#4C72B0")
            else:
                heights.append(0.35)
                colors.append("#999999")
        ax.bar(nice, heights, color=colors)
        for i, t in enumerate(texts):
            ax.text(i, heights[i] + 0.05, t, ha="center", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.35)
        ax.set_title(name.replace("_", " / "))
        ax.set_ylabel("Package outcome" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=20)
        ax.set_yticks([])
    fig.suptitle(
        "Causal tetrad: GoF selects twin packages; matched controls do not",
        y=1.02,
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "figure7_causal_tetrad.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Wrote figures 4–7 into", FIGDIR)


if __name__ == "__main__":
    main()
