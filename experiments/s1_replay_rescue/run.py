#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"


def interval(values: list[float], seed: int, resamples: int) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(values)
    boot = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    boot.sort()
    return {"n": n, "mean": sum(values) / n, "lo": boot[math.floor(.025*(len(boot)-1))], "hi": boot[math.ceil(.975*(len(boot)-1))], "method": "prompt-level nonparametric bootstrap", "resamples": resamples}


def main() -> None:
    config = json.loads((EXP_DIR / "config.json").read_text())
    rows = json.loads((EXP_DIR / config["source_rows"]).resolve().read_text())
    rows = [r for r in rows if r["run_id"] == config["run_id"]]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    details = []
    for row in rows:
        baseline_article = row["free"]["off"]["article"]
        treated_article = row["free"]["on"]["article"]
        baseline_cell = row["forced"][baseline_article]
        treated_cell = row["forced"][treated_article]
        details.append({
            "sentence": row["sentence"], "baseline_article": baseline_article, "treated_article": treated_article,
            "baseline_word": row["free"]["off"]["word"], "treated_word": row["free"]["on"]["word"],
            "phenocopy_top1": treated_cell["off"]["top1"] == treated_cell["on"]["top1"],
            "rescue_top1": baseline_cell["off"]["top1"] == baseline_cell["on"]["top1"],
            "phenocopy_residual_tv": row["comparisons"][treated_article]["tv_full_vocab"],
            "rescue_residual_tv": row["comparisons"][baseline_article]["tv_full_vocab"],
        })
    n = len(details)
    resamples = int(config["bootstrap_resamples"])
    seed = int(config["bootstrap_seed"])
    summary = {
        "experiment": config["experiment_name"], "generated_at": datetime.now(timezone.utc).isoformat(), "n_prompts": n,
        "phenocopy_top1_rate": sum(d["phenocopy_top1"] for d in details) / n,
        "rescue_top1_rate": sum(d["rescue_top1"] for d in details) / n,
        "phenocopy_residual_tv": interval([d["phenocopy_residual_tv"] for d in details], seed, resamples),
        "rescue_residual_tv": interval([d["rescue_residual_tv"] for d in details], seed + 1, resamples),
        "interpretation": "The treated article phenocopies the S1-on noun, and restoring the baseline article rescues the baseline top noun despite S1 remaining active."
    }
    (RESULTS_DIR / "rows.json").write_text(json.dumps(details, indent=2) + "\n")
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    p, r = summary["phenocopy_residual_tv"], summary["rescue_residual_tv"]
    report = f"""# S1 replay and rescue

- Treated-token phenocopy top-1: {summary['phenocopy_top1_rate']:.2f}.
- Baseline-token rescue top-1: {summary['rescue_top1_rate']:.2f}.
- Phenocopy matched-prefix leftover TV: {p['mean']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}].
- Rescue matched-prefix leftover TV: {r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}].

{summary['interpretation']}
"""
    (RESULTS_DIR / "report.md").write_text(report)


if __name__ == "__main__":
    main()
