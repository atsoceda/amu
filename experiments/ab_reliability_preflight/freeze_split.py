#!/usr/bin/env python3
"""Freeze outcome-blind family splits after the lexical eligibility screen."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from experiments.lib.aan_protocol import write_json

EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text())
ART = EXP / "artifacts"


def main() -> None:
    eligible = json.loads((ART / "02_lexical_screen/eligible.json").read_text())
    split_cfg = CFG["split"]
    required = sum(split_cfg[key] for key in ("demonstration_families", "development_families", "confirmatory_families"))
    if len(eligible) < required:
        raise SystemExit(f"need {required} eligible families but only {len(eligible)} passed")
    rng = random.Random(split_cfg["seed"])
    ordered = sorted(eligible, key=lambda row: row["candidate_id"])
    rng.shuffle(ordered)
    nonoverlapping = []
    excluded_overlap = []
    used_terms: set[str] = set()
    for row in ordered:
        terms = {row["everyday_term"], row["formal_term"]}
        if terms & used_terms:
            excluded_overlap.append(row)
            continue
        nonoverlapping.append(row)
        used_terms.update(terms)
    if len(nonoverlapping) < required:
        raise SystemExit(f"need {required} term-disjoint families but only {len(nonoverlapping)} are available")
    ordered = nonoverlapping
    n_demo = split_cfg["demonstration_families"]
    n_dev = split_cfg["development_families"]
    frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "seed": split_cfg["seed"],
        "selection_basis": "Gemma lexical-register eligibility only; no A/B or route outcomes",
        "cross_split_term_overlap": False,
        "demonstration": ordered[:n_demo],
        "development": ordered[n_demo:n_demo + n_dev],
        "confirmatory": ordered[n_demo + n_dev:n_demo + n_dev + split_cfg["confirmatory_families"]],
        "eligible_reserve": ordered[required:],
        "excluded_shared_term": excluded_overlap,
    }
    write_json(ART / "03_frozen_split.json", frozen)
    print(json.dumps({key: len(value) for key, value in frozen.items() if isinstance(value, list)}, indent=2))


if __name__ == "__main__":
    main()
