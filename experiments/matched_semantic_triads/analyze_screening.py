#!/usr/bin/env python3
"""Diagnose why frozen matched-triad construction banks fail eligibility."""

import json
from collections import Counter, defaultdict
from pathlib import Path


EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"


def summarize(path: Path) -> dict:
    rows = json.loads(path.read_text())
    failures = Counter()
    family_any = defaultdict(bool)
    family_candidates = Counter()
    for row in rows:
        family = row.get("family", row["id"].split("__", 1)[0])
        family_candidates[family] += 1
        family_any[family] |= row["admissible"]
        if not all(row["single_token"].values()):
            failures["not_all_single_token"] += 1
        if not all(row["generated"][r]["article"] == row[f"{r}_article"] for r in ("source", "within", "cross")):
            failures["article_policy_mismatch"] += 1
        for name, value in row["margins"].items():
            if value <= 0:
                failures[f"nonpositive_{name}"] += 1
        if row["admissible"]:
            failures["admissible"] += 1
    return {
        "file": path.name,
        "candidates": len(rows),
        "families": len(family_candidates),
        "admissible_candidates": sum(r["admissible"] for r in rows),
        "admissible_families": sorted(f for f, valid in family_any.items() if valid),
        "failure_counts_nonexclusive": dict(sorted(failures.items())),
    }


def main() -> None:
    files = [RESULTS / "screen_rows.json", RESULTS / "screen_rows_batch2.json", RESULTS / "screen_rows_batch3.json"]
    report = {
        "interpretation": (
            "Behavioral construction audit only. Failure to form enough eligible triads is not a "
            "causal null. Counts are nonexclusive because one candidate can fail several rules."
        ),
        "banks": [summarize(p) for p in files],
    }
    (RESULTS / "screening_diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
