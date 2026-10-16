"""Torch-free summary for the recurrent-carry pilot, recomputable from committed rows.

`pilot_summary` is the summary block that used to sit at the end of pilot.py's main();
pilot.py calls it, so new runs and `bin/resummarize` build the summary the same way.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "couplet_routes"))
from route_stats import boot  # noqa: E402

KEYS = ["persistence", "attention_only", "recurrent_only", "both_blocked", "retrieval", "relay"]


def pilot_summary(out_rows, model):
    """pilot.py -> pilot_summary.json."""
    return {"model": model, "n": len(out_rows), **{k: boot([x[k] for x in out_rows]) for k in KEYS},
            "recurrent_share": boot([x["recurrent_only"] / x["persistence"] for x in out_rows if abs(x["persistence"]) > 1])}


CHECKS = [
    {"rows": "pilot_rows.json", "summary": "pilot_summary.json", "fn": lambda rows, run: pilot_summary(rows, run.name)},
    # The 24-couplet Qwen3.5-27B pilot, kept under this name when the full run was queued (a2d7c65).
    {"rows": "pilot24_rows.json", "summary": "pilot24_summary.json", "fn": lambda rows, run: pilot_summary(rows, run.name)},
]
