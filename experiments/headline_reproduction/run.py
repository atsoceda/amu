#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"


def load(path: Path):
    return json.loads(path.read_text())


def mean(values):
    return sum(values) / len(values)


def main() -> None:
    six = load(ROOT / "experiments/six_cell_family_sweep/results/rows.json")
    s1 = [row for row in six if row["run_id"] == "S1_5x"]
    boundary = [row for row in load(ROOT / "experiments/prompt_aligned_article_boundary/results/rows.json") if row["bracketed"]]
    private_rows = [row for row in load(ROOT / "experiments/pre_article_public_private_factorial/results/rows.json") if row["split"] == "test" and row["layer"] == 12 and row["article"] == "a"]
    random_rows = [row for row in load(ROOT / "experiments/pre_article_public_private_factorial/results/random_rows.json") if row["layer"] == 12 and row["article"] == "a"]
    random_by_pair = {}
    for row in random_rows:
        random_by_pair.setdefault(row["pair_id"], []).append(float(row["delta_delta"]))
    paired = [float(row["delta_delta"]) - mean(random_by_pair[row["pair_id"]]) for row in private_rows]
    result = {
        "s1_six_cell": {
            "n": len(s1),
            "total_tv": mean([float(r["decomposition"]["total_tv"]) for r in s1]),
            "token_substitution_tv": mean([float(r["decomposition"]["mediator_tv"]) for r in s1]),
            "fixed_treated_leftover_tv": mean([float(r["decomposition"]["residual_tv"]) for r in s1]),
            "token_total_cosine": mean([float(r["decomposition"]["cosine_mediator_total"]) for r in s1]),
        },
        "prompt_aligned_boundary": {
            "n": len(boundary),
            "gain_width": mean([float(r["gain_width"]) for r in boundary]),
            "total_tv": mean([float(r["total_tv"]) for r in boundary]),
            "token_substitution_tv": mean([float(r["token_substitution_tv"]) for r in boundary]),
            "fixed_a_tv": mean([float(r["fixed_a_residual_tv"]) for r in boundary]),
            "fixed_an_tv": mean([float(r["fixed_an_residual_tv"]) for r in boundary]),
        },
        "pre_article_private_patch": {
            "n": len(private_rows),
            "target_delta_delta": mean([float(r["delta_delta"]) for r in private_rows]),
            "matched_random_delta_delta": mean([mean(random_by_pair[r["pair_id"]]) for r in private_rows]),
            "paired_target_minus_random": mean(paired),
        },
        "source_files": [
            "experiments/six_cell_family_sweep/results/rows.json",
            "experiments/prompt_aligned_article_boundary/results/rows.json",
            "experiments/pre_article_public_private_factorial/results/rows.json",
            "experiments/pre_article_public_private_factorial/results/random_rows.json"
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Independent headline-number reproduction", "", "```json", json.dumps(result, indent=2), "```", ""]
    (RESULTS / "report.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
