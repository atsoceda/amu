#!/usr/bin/env python3
"""Generate the manuscript's transparent table of all frozen triad candidates."""
from __future__ import annotations

import json
from pathlib import Path

EXP = Path(__file__).resolve().parent
OUT = EXP / "results/candidate_table.md"


def reason(row, threshold):
    failures = []
    if not all(row["single_token"].values()):
        failures.append("tokenization")
    off = row["arms"]["within"]["off_support"]
    if not off["article_top"] or off["article_mass"] < threshold:
        failures.append("baseline mediator support")
    for arm in ("within", "cross"):
        values = row["arms"][arm]
        if values["local_target_minus_source_effect"] <= 0:
            failures.append(f"{arm} local efficacy")
        if not values["on_support"]["article_top"] or values["on_support"]["article_mass"] < threshold:
            failures.append(f"{arm} treated support")
    return "; ".join(failures) if failures else "--"


def main():
    cfg = json.loads((EXP / "config.json").read_text())
    rows = {row["triad_id"]: row for row in json.loads((EXP / "results/screen_rows.json").read_text())}
    lines = [
        "| Family | Source | Same-article target | Cross-article target | Retained | Route-blind failure reason |",
        "|---|---|---|---|:---:|---|",
    ]
    for triad in cfg["triads"]:
        row = rows[triad["id"]]
        lines.append(
            f"| {triad['id']} | "
            f"{triad['source_article']} {triad['source_word']} | "
            f"{triad['within_article']} {triad['within_word']} | "
            f"{triad['cross_article']} {triad['cross_word']} | "
            f"{'yes' if row['admissible'] else 'no'} | {reason(row, cfg['minimum_article_mass'])} |"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
