#!/usr/bin/env python3
"""Freeze low/medium/high modifier constructions before context generation."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path


EXP = Path(__file__).resolve().parent
FAMILIES = json.loads((EXP / "artifacts/02_tokenizer_filtered/retained.json").read_text())
ROWS = json.loads((EXP / "artifacts/03_branch_leverage/rows.json").read_text())
OUTPUT = EXP / "artifacts/04_context_candidates"


def choose_pairs(rows):
    ordered = sorted(rows, key=lambda r: r["target_probability_projection"])
    pairs = []
    for left, right in combinations(ordered, 2):
        delta = right["target_probability_projection"] - left["target_probability_projection"]
        pairs.append((delta, left, right))
    span = pairs[-1][0]
    targets = {"low": 0.1 * span, "medium": 0.5 * span, "high": span}
    chosen = {}
    used = set()
    for regime, target in targets.items():
        candidates = sorted(pairs, key=lambda x: (abs(x[0] - target), -min(x[1]["noun_pair_probability_mass"], x[2]["noun_pair_probability_mass"])))
        pick = next(x for x in candidates if (x[1]["modifier"], x[2]["modifier"]) not in used)
        used.add((pick[1]["modifier"], pick[2]["modifier"]))
        chosen[regime] = {
            "source_modifier": pick[1]["modifier"], "target_modifier": pick[2]["modifier"],
            "source_projection": pick[1]["target_probability_projection"],
            "target_projection": pick[2]["target_probability_projection"],
            "forced_modifier_leverage": pick[0],
            "source_pair_mass": pick[1]["noun_pair_probability_mass"],
            "target_pair_mass": pick[2]["noun_pair_probability_mass"],
        }
    return chosen, span, max(r["noun_pair_probability_mass"] for r in rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for row in ROWS: grouped.setdefault(row["candidate_id"], []).append(row)
    lookup = {f["candidate_id"]: f for f in FAMILIES}
    candidates = []
    for candidate_id, rows in grouped.items():
        if len(rows) < 6: continue
        constructions, span, max_mass = choose_pairs(rows)
        family = lookup[candidate_id]
        eligible = span >= 0.02 and max_mass >= 0.02
        candidates.append({**family, "forced_projection_span": span, "max_noun_pair_mass": max_mass,
                           "context_priority_score": span * max_mass,
                           "context_generation_eligible": eligible, "constructions": constructions})
    candidates.sort(key=lambda r: r["context_priority_score"], reverse=True)
    # Frozen before context outcomes: broad but bounded expensive generation set.
    selected = [r for r in candidates if r["context_generation_eligible"]][:120]
    for rank, row in enumerate(selected, 1): row["context_generation_rank"] = rank
    (OUTPUT / "all_ranked.json").write_text(json.dumps(candidates, indent=2) + "\n")
    (OUTPUT / "selected_120.json").write_text(json.dumps(selected, indent=2) + "\n")
    summary = {"selection_frozen_at": "2026-08-31T16:45:00+09:00",
               "criteria": "span >= .02, max noun-pair mass >= .02; take top 120 by span*max_mass",
               "families_scored": len(candidates), "families_eligible": sum(r["context_generation_eligible"] for r in candidates),
               "families_selected": len(selected), "constructions_per_family": 3,
               "modifier_pair_targets": {"low": "10% of family span", "medium": "50% of span", "high": "100% of span"}}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()

