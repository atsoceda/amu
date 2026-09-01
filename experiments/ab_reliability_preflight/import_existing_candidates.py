#!/usr/bin/env python3
"""Import the pre-existing, independently generated synonym bank as unordered candidates."""

from __future__ import annotations

import json
from pathlib import Path

EXP = Path(__file__).resolve().parent
SOURCE = EXP.parent / "natural_modifier_routes/artifacts/01_parsed_candidates/all_candidates.json"
OUT = EXP / "artifacts/01_candidates/all_candidates.json"


def main() -> None:
    source_rows = json.loads(SOURCE.read_text())
    rows = []
    for row in source_rows:
        rows.append({
            "candidate_id": "existing_" + row["candidate_id"],
            "generation_batch": row["generation_batch"],
            "family_id": row["family_id"],
            "everyday_term": row["noun_0"],
            "formal_term": row["noun_1"],
            "shared_meaning": row["shared_definition"],
            "distinction_note": row["semantic_notes"],
            "source_experiment": "natural_modifier_routes",
            "orientation_before_screen": "unordered_noun_0_noun_1",
        })
    for path in sorted(OUT.parent.glob("batch_*.json")):
        rows.extend(json.loads(path.read_text()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"imported {len(rows)} unordered candidates from {SOURCE}")


if __name__ == "__main__":
    main()
