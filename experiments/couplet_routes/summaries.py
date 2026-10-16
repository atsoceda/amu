"""Summaries of the couplet-route runners, computed from their rows without torch or a model.

Each function is the summary block that used to sit at the end of a runner's main(),
moved unchanged (same keys, key order and floating-point order). The runners call
these functions, and bin/resummarize uses CHECKS to recompute every committed summary
from its committed rows.

`model` is the runner's model argument, which is also the name of the results folder,
so CHECKS reads it from the run directory name. `elapsed_sec` is wall-clock time of
the model run: the runners pass it, CHECKS passes None and ignores the key because
rows do not record it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from route_stats import boot  # noqa: E402

STEP34_KEYS = ["d_total", "d_emission", "d_persistence", "d_persistence_p0", "d_retrieval", "d_relay",
               "tv_total", "tv_emission", "tv_persistence", "tv_persistence_p0"]
RELAY_POSITION_GROUPS = ("all", "late", "far", "tail", "early")
NECESSITY_GROUPS = ("all", "late", "tail", "early")
STEP6_KEYS = ["d_total", "d_emission", "d_persistence", "d_persistence_p0", "d_retrieval", "d_relay"]


def step2_summary(model, rows, elapsed_sec=None):
    """step2_state_edit.py: rhyme rates of line 2 with and without the anchor state edit."""
    n = len(rows)
    return {"model": model, "n": n,
            "matches_authors_original": sum(r["matches_authors"] for r in rows) / n,
            "off_rhymes_orig": sum(r["off_rhymes_orig"] for r in rows) / n,
            "on_rhymes_orig": sum(r["on_rhymes_orig"] for r in rows) / n,
            "on_rhymes_donor": sum(r["on_rhymes_donor"] for r in rows) / n,
            "off_rhymes_donor": sum(r["off_rhymes_donor"] for r in rows) / n,
            "on_changed_last_word": sum(r["on_changed_word"] for r in rows) / n,
            "elapsed_sec": elapsed_sec}


def step34_summary(model, rows, elapsed_sec=None):
    """step34_routes.py: emission, persistence and the retrieval/relay split."""
    keys = STEP34_KEYS
    return {"model": model, "n": len(rows), "elapsed_sec": elapsed_sec,
            "all": {k: boot([r[k] for r in rows]) for k in keys},
            "edit_succeeded": {k: boot([r[k] for r in rows if r["edit_succeeded"]]) for k in keys},
            "additivity_gap_p0": boot([r["d_persistence_p0"] - r["d_retrieval"] - r["d_relay"] for r in rows])}


def relay_positions_summary(model, rows, late, elapsed_sec=None):
    """relay_positions.py: relay sufficiency per group of in-between positions."""
    keys = ["d_persistence_p0"] + [f"d_relay_{g}" for g in RELAY_POSITION_GROUPS]
    return {"model": model, "n": len(rows), "late_positions": late, "elapsed_sec": elapsed_sec,
            "all": {k: boot([x[k] for x in rows]) for k in keys},
            "far_minus_late": boot([x["d_relay_far"] - x["d_relay_late"] for x in rows])}


def relay_necessity_summary(model, rows, sufficiency_rows=None):
    """relay_necessity.py. sufficiency_rows: relay_positions_rows.json of the same run, if present."""
    keys = ["d_persistence_p0"] + [f"necessity_{g}" for g in NECESSITY_GROUPS]
    s = {"model": model, "n": len(rows), "all": {k: boot([x[k] for x in rows]) for k in keys}}
    if sufficiency_rows is not None:
        srows = {x["idx"]: x for x in sufficiency_rows}
        both = [(x, srows[x["idx"]]) for x in rows if x["idx"] in srows]
        s["shapley"] = {g: boot([(x[f"necessity_{g}"] + y[f"d_relay_{g}"]) / 2 for x, y in both]) for g in NECESSITY_GROUPS}
        s["sufficiency"] = {g: boot([y[f"d_relay_{g}"] for _, y in both]) for g in NECESSITY_GROUPS}
    return s


def relay_boundary_summary(model, rows):
    """relay_boundary.py: per-token relay sufficiency and necessity in the prompt tail."""
    ref = rows[0]["tail_tokens"]
    same = [x for x in rows if x["tail_tokens"] == ref]
    return {"model": model, "n": len(rows), "n_same_tail": len(same), "tail_tokens": ref,
            "boundary_all": boot([x["boundary_all"] for x in rows]),
            "by_token": [{"offset": k + 1, "token": t, **boot([x["by_token"][k] for x in same])} for k, t in enumerate(ref)],
            "necessity_by_token": [{"offset": k + 1, "token": t, **boot([x["necessity_by_token"][k] for x in same])}
                                   for k, t in enumerate(ref)]}


def relay_boundary_summary_before_necessity(model, rows):
    """relay_boundary.py as it was before commit fdf745b (2026-09-26), which added per-token
    necessity. Its rows have no necessity_by_token, so its summary has none either; it
    wrote the committed Qwen3-1.7B, Qwen3-4B and Qwen3-8B boundary summaries."""
    ref = rows[0]["tail_tokens"]
    same = [x for x in rows if x["tail_tokens"] == ref]
    return {"model": model, "n": len(rows), "n_same_tail": len(same), "tail_tokens": ref,
            "boundary_all": boot([x["boundary_all"] for x in rows]),
            "by_token": [{"offset": k + 1, "token": t, **boot([x["by_token"][k] for x in same])} for k, t in enumerate(ref)]}


def anchor_specificity_summary(model, rows, elapsed_sec=None):
    """anchor_specificity.py: persistence, retrieval, relay and TV per placement of the donor state."""
    return {"model": model, "n": len(rows), "elapsed_sec": elapsed_sec,
            "by_position": {k: {m_: boot([r[f"{k}_{m_}"] for r in rows]) for m_ in ("persistence", "retrieval", "relay", "tv")}
                            for k in ("A", "M", "F", "C", "E")}}


def step6_summary(model, rows, elapsed_sec=None):
    """step6_routes.py: rhyme rates and route split under the authors' feature steering."""
    keys = STEP6_KEYS
    sub = [r for r in rows if "d_total" in r]
    return {"model": model, "n": len(rows), "n_routes": len(sub), "elapsed_sec": elapsed_sec,
            "on_rhymes_donor": sum(r["on_rhymes_donor"] for r in rows) / len(rows),
            "on_rhymes_orig": sum(r["on_rhymes_orig"] for r in rows) / len(rows),
            "authors_steered_rhymes_donor": sum(r["authors_steered_rhymes_donor"] for r in rows) / len(rows),
            "routes": {k: boot([r[k] for r in sub]) for k in keys},
            "routes_edit_succeeded": {k: boot([r[k] for r in sub if r["on_rhymes_donor"]]) for k in keys},
            "additivity_gap_p0": boot([r["d_persistence_p0"] - r["d_retrieval"] - r["d_relay"] for r in sub])}


def _necessity_check(rows, run_dir):
    # The runner adds the Shapley split when relay_positions_rows.json exists in the same folder.
    suff = run_dir / "relay_positions_rows.json"
    return relay_necessity_summary(run_dir.name, rows, json.loads(suff.read_text()) if suff.exists() else None)


def _boundary_check(rows, run_dir):
    # Rows written before commit fdf745b have no per-token necessity.
    if "necessity_by_token" not in rows[0]:
        return relay_boundary_summary_before_necessity(run_dir.name, rows)
    return relay_boundary_summary(run_dir.name, rows)


LATE = 3  # relay_positions.LATE, which the summary records as late_positions

# elapsed_sec: wall-clock time of the model run, not recorded in the rows.
CHECKS = [
    {"rows": "step2_rows.json", "summary": "step2_summary.json",
     "fn": lambda rows, d: step2_summary(d.name, rows), "ignore": ("elapsed_sec",)},
    {"rows": "step2_rows_same_rhyme.json", "summary": "step2_summary_same_rhyme.json",
     "fn": lambda rows, d: step2_summary(d.name, rows), "ignore": ("elapsed_sec",)},
    {"rows": "step34_rows.json", "summary": "step34_summary.json",
     "fn": lambda rows, d: step34_summary(d.name, rows), "ignore": ("elapsed_sec",)},
    {"rows": "step34_rows_same_rhyme.json", "summary": "step34_summary_same_rhyme.json",
     "fn": lambda rows, d: step34_summary(d.name, rows), "ignore": ("elapsed_sec",)},
    {"rows": "relay_positions_rows.json", "summary": "relay_positions_summary.json",
     "fn": lambda rows, d: relay_positions_summary(d.name, rows, LATE), "ignore": ("elapsed_sec",)},
    {"rows": "relay_positions_rows_same_rhyme.json", "summary": "relay_positions_summary_same_rhyme.json",
     "fn": lambda rows, d: relay_positions_summary(d.name, rows, LATE), "ignore": ("elapsed_sec",)},
    {"rows": "relay_necessity_rows.json", "summary": "relay_necessity_summary.json", "fn": _necessity_check},
    {"rows": "relay_boundary_rows.json", "summary": "relay_boundary_summary.json",
     "fn": _boundary_check},
    {"rows": "anchor_specificity_rows.json", "summary": "anchor_specificity_summary.json",
     "fn": lambda rows, d: anchor_specificity_summary(d.name, rows), "ignore": ("elapsed_sec",)},
    {"rows": "step6_rows.json", "summary": "step6_summary.json",
     "fn": lambda rows, d: step6_summary(d.name, rows), "ignore": ("elapsed_sec",)},
]
