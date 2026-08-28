#!/usr/bin/env python3
"""Forensic analysis of S2 using frozen six-cell outputs."""
from __future__ import annotations

import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def interval(values: list[float], seed: int, resamples: int) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = random.Random(seed)
    n = len(values)
    boot = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    boot.sort()
    return {
        "n": n, "mean": sum(values) / n,
        "lo": boot[math.floor(0.025 * (len(boot) - 1))],
        "hi": boot[math.ceil(0.975 * (len(boot) - 1))],
        "method": "prompt-level nonparametric bootstrap", "resamples": resamples,
    }


def legal(article: str, word: str) -> bool:
    if article not in {"a", "an"} or not word:
        return False
    vowel = word[0].lower() in "aeiou"
    return (article == "an" and vowel) or (article == "a" and not vowel)


def lower_bound_mass(cell: dict[str, Any], vocabulary: set[str]) -> float:
    return sum(float(item["prob"]) for item in cell["top_tokens"] if item["token"].strip().lower() in vocabulary)


def fmt(block: dict[str, Any]) -> str:
    return f"{block['mean']:.3f} [{block['lo']:.3f}, {block['hi']:.3f}]"


def main() -> None:
    config = load(EXP_DIR / "config.json")
    rows = load((EXP_DIR / config["source_rows"]).resolve())
    source_summary = load((EXP_DIR / config["source_summary"]).resolve())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ids = [config["primary_run_id"], *config["comparison_run_ids"]]
    vocab = {str(r[k]).lower() for r in rows if r["run_id"] == config["primary_run_id"] for k in ("listed_word", "twin_word") if r.get(k)}
    resamples = int(config["bootstrap_resamples"])
    seed0 = int(config["bootstrap_seed"])
    analysis: dict[str, Any] = {}
    detail_rows = []
    for run_index, run_id in enumerate(run_ids):
        selected = [r for r in rows if r["run_id"] == run_id]
        twin_values = []
        for r in selected:
            free_on = r["free"]["on"]
            twin = r.get("pre_specified_twin_effects")
            twin_value = None if not twin else twin["an"]["delta_delta_target_minus_source"]
            if twin_value is not None:
                twin_values.append(float(twin_value))
            an_off = r["forced"]["an"]["off"]
            an_on = r["forced"]["an"]["on"]
            detail_rows.append({
                "run_id": run_id, "sentence": r["sentence"],
                "free_on_continuation": free_on["continuation"],
                "free_on_legal_article_noun": legal(free_on["article"], free_on["word"]),
                "forced_an_top1_off": an_off["top1"], "forced_an_top1_on": an_on["top1"],
                "forced_an_top1_changed": an_off["top1"] != an_on["top1"],
                "forced_an_tv": r["comparisons"]["an"]["tv_full_vocab"],
                "twin_delta_delta_an": twin_value,
                "occupation_mass_lower_bound_off": lower_bound_mass(an_off, vocab),
                "occupation_mass_lower_bound_on": lower_bound_mass(an_on, vocab),
            })
        run_details = [r for r in detail_rows if r["run_id"] == run_id]
        tvs = [float(r["forced_an_tv"]) for r in run_details]
        mass_deltas = [float(r["occupation_mass_lower_bound_on"] - r["occupation_mass_lower_bound_off"]) for r in run_details]
        analysis[run_id] = {
            "n_prompts": len(selected),
            "free_legal_article_noun_rate": sum(bool(r["free_on_legal_article_noun"]) for r in run_details) / max(len(run_details), 1),
            "free_other_prefix_rate": sum(r["free"]["on"]["article"] == "other" for r in selected) / max(len(selected), 1),
            "forced_an_top1_changed_rate": sum(bool(r["forced_an_top1_changed"]) for r in run_details) / max(len(run_details), 1),
            "forced_an_tv": interval(tvs, seed0 + run_index * 10, resamples),
            "occupation_mass_lower_bound_delta": interval(mass_deltas, seed0 + run_index * 10 + 1, resamples),
            "twin_delta_delta_an": interval(twin_values, seed0 + run_index * 10 + 2, resamples),
            "twin_positive_rate": sum(v > 0 for v in twin_values) / max(len(twin_values), 1),
            "twin_range": [min(twin_values), max(twin_values)] if twin_values else None,
        }
    s2 = analysis[config["primary_run_id"]]
    random_block = analysis["S1_random_5x"]
    conclusion = (
        "S2 combines free-generation disruption with a consistent subthreshold fixed-an target signal: "
        "its free generations frequently fail the legal article+noun parse, yet all seven pre-specified twins "
        "move in the target direction. Because the forced-prefix top noun almost never changes and only one "
        "frozen random set is available, S2 is evidence of target-aligned fixed-token influence but is not yet "
        "a clean persistence-dominant intervention."
    )
    summary = {
        "experiment": config["experiment_name"], "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_experiment": source_summary["experiment"], "analysis": analysis,
        "primary_conclusion": conclusion,
        "s2_exceeds_single_random_tv": s2["forced_an_tv"]["mean"] > random_block["forced_an_tv"]["mean"],
        "limitations": [
            "Occupation-vocabulary probability mass is a lower bound computed from stored top-10 tokens.",
            "This reanalysis cannot recover full-vocabulary entropy because full logit vectors were not stored.",
            "The comparison random arm contains one frozen four-feature set, not a multi-set empirical null distribution."
        ],
    }
    dump(RESULTS_DIR / "rows.json", detail_rows)
    dump(RESULTS_DIR / "summary.json", summary)
    lines = [
        "# S2 forensic analysis", "", conclusion, "",
        "| Handle | Legal free completion | Other-prefix rate | Forced-`an` top-1 changed | Forced-`an` TV | Twin ΔΔ `an` | Positive twins |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run_id in run_ids:
        block = analysis[run_id]
        lines.append(f"| `{run_id}` | {block['free_legal_article_noun_rate']:.2f} | {block['free_other_prefix_rate']:.2f} | {block['forced_an_top1_changed_rate']:.2f} | {fmt(block['forced_an_tv'])} | {fmt(block['twin_delta_delta_an'])} | {block['twin_positive_rate']:.2f} |")
    lines.extend(["", "## Limitations", "", *[f"- {x}" for x in summary["limitations"]], ""])
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
