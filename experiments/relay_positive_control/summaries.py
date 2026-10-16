"""Summary of induction.py, computed from its rows without torch or a model.

The function is the summary block that used to sit at the end of induction.py's
main(), moved unchanged (same keys, key order and floating-point order). induction.py
calls it, and bin/resummarize uses CHECKS to recompute every committed summary from
its committed rows. `model` is the runner's model argument, which is also the name
of the results folder, so CHECKS reads it from the run directory name.
"""
from __future__ import annotations

import sys
from pathlib import Path

_COUPLETS = str(Path(__file__).resolve().parents[1] / "couplet_routes")
if _COUPLETS not in sys.path:
    sys.path.append(_COUPLETS)  # appended, so it cannot shadow this folder's modules
from route_stats import boot  # noqa: E402


def induction_summary(model, rows):
    """induction.py: persistence, retrieval, relay and key relay of log p(B) at the target."""
    return {"model": model, "n": len(rows), "mean_p_clean": sum(r["p_clean"] for r in rows) / len(rows),
            **{k: boot([r[k] for r in rows]) for k in ("persistence", "retrieval", "relay", "key_relay")},
            "relay_share": boot([r["relay"] / r["persistence"] for r in rows if r["persistence"] < -1])}


CHECKS = [
    {"rows": "induction_rows.json", "summary": "induction_summary.json",
     "fn": lambda rows, d: induction_summary(d.name, rows)},
]
