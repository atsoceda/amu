#!/usr/bin/env python3
"""Regenerate collaborator-report figures 1–7 from experiment summary.json files.

Required experiment result artifacts (under experiments/*/results/summary.json):

  figure1  a_an_majority_baseline, a_an_full_dataset_screen
  figure2  ophthalmologist_competing_pathway_screen
  figure3  fixed_pair_generalization
  figure4  selection_criterion_ablation
  figure5  planning_dose_response
  figure6  forced_content_lock
  figure7  trajectory_causal_tetrad
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
FIGDIR = Path(__file__).resolve().parent / "figures"

REQUIRED = {
    "figure1_article_recall.png": [
        "a_an_majority_baseline",
        "a_an_full_dataset_screen",
    ],
    "figure2_source_intervention.png": ["ophthalmologist_competing_pathway_screen"],
    "figure3_generalization.png": ["fixed_pair_generalization"],
    "figure4_selection_ablation.png": ["selection_criterion_ablation"],
    "figure5_dose_response.png": ["planning_dose_response"],
    "figure6_dual_lock.png": ["forced_content_lock"],
    "figure7_causal_tetrad.png": ["trajectory_causal_tetrad"],
}


def load_summary(name: str) -> dict:
    path = EXP / name / "results" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing experiment summary for figure generation: {path}"
        )
    return json.loads(path.read_text())


def apply_style() -> None:
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


def figure1() -> None:
    pilot = load_summary("a_an_majority_baseline")
    full = load_summary("a_an_full_dataset_screen")
    values = [
        100.0 * float(pilot["recall_an"]),
        100.0 * float(full["article_an_recall"]),
        100.0 * float(pilot["recall_a"]),
    ]
    labels = [
        "Pilot an\nrecall",
        "Full-dataset an\nrecall",
        "a recall\n(control)",
    ]
    colors = ["#C44E52", "#DD8452", "#4C72B0"]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Recall (%)")
    ax.set_title('Gemma 3 270M strongly favors the majority article "a"')
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2.5,
            f"{val:.1f}%",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(FIGDIR / "figure1_article_recall.png", dpi=160)
    plt.close()


def figure2() -> None:
    screen = load_summary("ophthalmologist_competing_pathway_screen")
    baseline = float(screen["baseline"]["an_minus_a"])
    single = next(
        row
        for row in screen["interventions"]
        if row.get("layer") == 13 and row.get("feature_idx") == 10304
    )
    pair = next(
        row
        for row in screen["combinations"]
        if row.get("label") == "`L13/F10304` + `L14/F1949`"
    )
    margins = [
        baseline,
        float(single["post_an_minus_a"]),
        float(pair["post_an_minus_a"]),
    ]
    labels = ["Baseline", "Suppress L13/F10304", "Suppress fixed pair"]
    colors = ["#C44E52", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    x = np.arange(len(margins))
    ax.axhline(0.0, color="#555555", lw=1.5)
    bars = ax.bar(x, margins, color=colors, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Article logit margin: an − a")
    ax.set_title(
        "Suppression corrected the article decision on the source prompt"
    )
    y_lo = min(margins + [-0.2]) - 0.25
    y_hi = max(margins + [0.2]) + 0.35
    ax.set_ylim(y_lo, y_hi)
    for bar, val in zip(bars, margins):
        offset = 0.06 if val >= 0 else -0.12
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + offset,
            f"{val:+.3f}",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(FIGDIR / "figure2_source_intervention.png", dpi=160)
    plt.close()


def figure3() -> None:
    gen = load_summary("fixed_pair_generalization")
    counts = gen["counts"]
    values = [
        int(counts["an_corrections"]),
        int(counts["a_false_flips"]),
        int(counts["content_word_changes"]),
        int(counts["grammar_repairs"]),
    ]
    labels = [
        "Expected-an\nrepairs",
        "Expected-a\nchanged to an",
        "Content word\nchanged",
        "Grammar-only\nrepairs",
    ]
    colors = ["#55A868", "#C44E52", "#8172B2", "#4C72B0"]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    x = np.arange(len(values))
    ymax = max(values + [1])
    bars = ax.bar(x, values, color=colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_ylabel("Held-out prompts")
    ax.set_title("The fixed pair generalized as a response-class switch")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + ymax * 0.03,
            str(val),
            ha="center",
            fontsize=14,
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(FIGDIR / "figure3_generalization.png", dpi=160)
    plt.close()


def figure4(e1: dict) -> None:
    sets = [
        ("S1_dual_effect", "S1 Dual-effect"),
        ("S2_article_only", "S2 Article-only"),
        ("S3_content_only", "S3 Content-only"),
        ("S4_competing_a", "S4 Competing a"),
    ]
    labels = [x[1] for x in sets]
    wrap = [e1["set_results"][k]["summary"]["wrapper_like_rate"] for k, _ in sets]
    traj = [
        e1["set_results"][k]["summary"]["trajectory_like_rate"] for k, _ in sets
    ]
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


def figure5(e2: dict) -> None:
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


def figure6(e3: dict) -> None:
    conds = [
        ("C1_article_push", "C1 Article-push"),
        ("C2_content_lock", "C2 Content-lock"),
        ("C3_dual", "C3 Dual"),
        ("C5_control_article", "Control"),
    ]
    labels = [c[1] for c in conds]
    traj = [
        e3["condition_summaries"][k]["trajectory_like_rate"] for k, _ in conds
    ]
    content = [
        e3["condition_summaries"][k]["content_preserved_rate"] for k, _ in conds
    ]
    illicit = [
        e3["condition_summaries"][k]["illicit_mismatch_rate"] for k, _ in conds
    ]
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


def figure7(e4: dict) -> None:
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


def main() -> None:
    missing_exps = sorted(
        {
            name
            for deps in REQUIRED.values()
            for name in deps
            if not (EXP / name / "results" / "summary.json").exists()
        }
    )
    if missing_exps:
        print("Missing required experiment summaries:", file=sys.stderr)
        for name in missing_exps:
            print(f"  - experiments/{name}/results/summary.json", file=sys.stderr)
        raise SystemExit(1)

    FIGDIR.mkdir(exist_ok=True)
    apply_style()
    figure1()
    figure2()
    figure3()
    figure4(load_summary("selection_criterion_ablation"))
    figure5(load_summary("planning_dose_response"))
    figure6(load_summary("forced_content_lock"))
    figure7(load_summary("trajectory_causal_tetrad"))
    print("Wrote figures 1–7 into", FIGDIR)


if __name__ == "__main__":
    main()
