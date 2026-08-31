#!/usr/bin/env python3
import json
from pathlib import Path


EXP=Path(__file__).resolve().parent
RESULTS=EXP/"results"
S=json.loads((RESULTS/"summary.json").read_text())


def ci(block):
    return f"{block['mean']:.3f} [{block['lo']:.3f}, {block['hi']:.3f}]"


between=S["conditions"]["between_1.0"]
within=S["conditions"]["within_1.0"]
interaction=S["interactions"]["strength_1.0"]["1.0"]
lines=[
    "# Correctness-preserving a/an carrier assay",
    "",
    f"Model: `{S['model']['model']}`. Independent semantic families: "
    f"{S['selection']['counts']['between']} between article classes and "
    f"{S['selection']['counts']['within']} within `a`.",
    "",
    "All leave-one-family-out folds selected layer 18/25. The earlier frozen "
    "layer-14 transfer attempt is preserved in `frozen_layer14/` and fails local efficacy.",
    "",
    "| Native-strength result | Between article | Within article |",
    "| --- | ---: | ---: |",
    f"| Fixed-target-article lexical ΔΔ | {ci(between['target_branch_delta_delta'])} | {ci(within['target_branch_delta_delta'])} |",
    f"| Public TV, τ=1 | {ci(between['temperatures']['1.0']['public_tv'])} | {ci(within['temperatures']['1.0']['public_tv'])} |",
    f"| Private TV, τ=1 | {ci(between['temperatures']['1.0']['private_tv'])} | {ci(within['temperatures']['1.0']['private_tv'])} |",
    f"| Public target alignment, τ=1 | {ci(between['temperatures']['1.0']['public_target_minus_source'])} | {ci(within['temperatures']['1.0']['public_target_minus_source'])} |",
    f"| Private target alignment, τ=1 | {ci(between['temperatures']['1.0']['private_target_minus_source'])} | {ci(within['temperatures']['1.0']['private_target_minus_source'])} |",
    "",
    f"TV route interaction: {ci(interaction['tv_route_interaction'])}; exact semantic-family permutation "
    f"p={interaction['tv_exact_permutation']['two_sided_p']:.6f}.",
    f"Target-aligned route interaction: {ci(interaction['aligned_route_interaction'])}.",
    "",
    "Interpretation: when the article distinguishes two correct synonyms, the public route dominates; "
    "when both synonyms require `a`, intended lexical movement survives primarily in the private component. "
    "This is a constructed full-residual intervention result, not evidence of spontaneous or sparse carrier selection.",
    "",
]
(RESULTS/"report.md").write_text("\n".join(lines))
print(RESULTS/"report.md")
