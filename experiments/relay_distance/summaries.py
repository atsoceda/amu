"""Torch-free summaries for the relay-distance pilots, recomputable from committed rows.

`distance_summary` is the summary block that used to sit at the end of each pilot's
main(); the pilots call it, so new runs and `bin/resummarize` build summaries the same
way. Only the cell names differ between pilot.py, pilot_v2.py and pilot_v2x.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "couplet_routes"))
from route_stats import boot  # noqa: E402

PILOT_KEYS = ("persistence", "retrieval", "relay", "filler_relay")
PILOT_V2_KEYS = ("persistence", "retrieval", "relay", "relay_boundary", "relay_filler", "relay_line2")
PILOT_V2X_KEYS = PILOT_V2_KEYS + ("necessity_boundary", "necessity_filler", "necessity_line2")


def distance_summary(rows, model, dists, keys, skipped=None):
    """Summary for one pilot. `skipped` (pilot_v2 and v2x only) lists the couplets whose
    generated line 2 could not be cut, as logged during the run; pilot.py has none."""
    s = {"model": model, "by_distance": {}}
    for D in dists:
        rr = [x for x in rows if x["distance"] == D]
        s["by_distance"][D] = {"n": len(rr), "rhymes_orig": sum(x["line2_rhymes_orig"] for x in rr) / max(len(rr), 1),
                               **{k: boot([x[k] for x in rr]) for k in keys},
                               "relay_share": boot([x["relay"] / x["persistence"] for x in rr if abs(x["persistence"]) > 1])}
    if skipped is not None:
        s["skipped"] = skipped
    return s


def _distances(rows):
    """The run's --distances, which rows do not store: the distances that have rows, in increasing order."""
    return sorted({x["distance"] for x in rows})


def _check(rows_name, summary_name, keys, has_skipped):
    return {"rows": rows_name, "summary": summary_name,
            # skipped holds couplets whose line 2 could not be cut; they produce no row (logged by the run only).
            "ignore": ("skipped",) if has_skipped else (),
            "fn": lambda rows, run: distance_summary(rows, run.name, _distances(rows), keys, [] if has_skipped else None)}


CHECKS = [
    _check("pilot_rows.json", "pilot_summary.json", PILOT_KEYS, False),
    _check("pilot_v2_rows.json", "pilot_v2_summary.json", PILOT_V2_KEYS, True),
    _check("pilot_v2x_rows.json", "pilot_v2x_summary.json", PILOT_V2X_KEYS, True),
    _check("pilot_v2x_window_rows.json", "pilot_v2x_window_summary.json", PILOT_V2X_KEYS, True),
]
