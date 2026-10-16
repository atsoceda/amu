"""Torch-free summaries for the derived-value runners, recomputable from committed rows.

Each function is the summary block that used to sit at the end of the runner's
main(); the runners call these, so new runs and `bin/resummarize` build summaries
the same way. Rows read back from JSON have string keys where the runner used
integer positions ("by_position", "scan"); the functions convert them, which
leaves in-memory rows (integer keys) unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "couplet_routes"))
from route_stats import boot  # noqa: E402

KS = (1, 3, 5)  # chain lengths, as in chain_routes.KS


def _int_keys(d):
    return {int(k): v for k, v in d.items()}


def stage1_summary(rows):
    """stage01.py: the "stage1" block of stage01_summary.json."""
    bp = [_int_keys(r["by_position"]) for r in rows]
    toks0 = rows[0]["tokens"]
    offs = sorted({p - r["qmark"] for r, b in zip(rows, bp) for p in b})
    return {"qmark": boot([b[r["qmark"]] for r, b in zip(rows, bp)]), "operands": boot([r["operands"] for r in rows]),
            "after_question": boot([r["after_question"] for r in rows]),
            "by_offset_from_qmark": {o: {"token": toks0[rows[0]["qmark"] + o],
                                         **boot([b[r["qmark"] + o] for r, b in zip(rows, bp)])} for o in offs}}


def stage01_summary(rows, model, stage0, elapsed_sec):
    """stage01.py -> stage01_summary.json. stage0 (greedy accuracy per filler length) comes
    from generations that are not stored in the rows."""
    return {"model": model, "n": len(rows), "stage0": stage0, "stage1": stage1_summary(rows), "elapsed_sec": elapsed_sec}


def chain_K_summary(rows, K, block_only):
    """chain_routes.py: summary["by_K"][K] for one chain length."""
    if not block_only and "post_v0_block" not in rows[0]:
        return chain_K_summary_before_block(rows, K)
    if block_only:
        return {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
                "v0_edit": boot([r["v0_edit"] for r in rows]),
                "post_v0_block": boot([r["post_v0_block"] for r in rows])}
    tq = rows[0]["tokens"]
    scans = [_int_keys(r["scan"]) for r in rows]
    return {
        "n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "v0_edit": boot([r["v0_edit"] for r in rows]),
        "post_v0_block": boot([r["post_v0_block"] for r in rows]),
        "by_end": [{k: boot([r["by_end"][j][k] for r in rows]) for k in ("persistence", "retrieval", "relay", "chain_relay")}
                   for j in range(K + 1)],
        "scan": {p - rows[0]["v0_digits"][0]: {"token": tq[p], **boot([sc[p] for sc in scans])} for p in scans[0]},
    }


def chain_K_summary_before_block(rows, K):
    """chain_routes.py as first committed (7521122), before the post-v0 block edit was added
    (39f704f): no "post_v0_block" in rows or summary. Qwen3-8B's chain_summary.json and
    chain_K*_rows.json were written by this version; rows without "post_v0_block" can only
    come from it."""
    tq = rows[0]["tokens"]
    scans = [_int_keys(r["scan"]) for r in rows]
    return {
        "n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "v0_edit": boot([r["v0_edit"] for r in rows]),
        "by_end": [{k: boot([r["by_end"][j][k] for r in rows]) for k in ("persistence", "retrieval", "relay", "chain_relay")}
                   for j in range(K + 1)],
        "scan": {p - rows[0]["v0_digits"][0]: {"token": tq[p], **boot([sc[p] for sc in scans])} for p in scans[0]},
    }


def written_K_summary(rows):
    """written_chain.py: summary["by_K"][K] for one chain length."""
    u = [r for r in rows if r["usable"]]
    return {"n": len(rows), "n_usable": len(u),
            "on_writes_donor_prev": sum(r["written_prev_on"] == r["donor_prev"] for r in rows) / len(rows),
            **{k: boot([r[k] for r in u]) for k in ("total", "emission", "persistence_t0", "persistence_t1")}}


def _by_K(files, pattern, one, run, summary_name):
    """{K: summary} for the chain lengths the run covered. A summary covers the run's --ks
    only, while rows files from earlier runs stay in the folder (Qwen3-4B: written_K3_rows.json
    is from an earlier laptop run; written_summary.json is from a later --ks 5 run). --ks is
    not stored in the rows, so it is read from the committed summary's by_K keys; every
    by_K entry is still recomputed from its rows."""
    ks = sorted(int(k) for k in json.loads((run / summary_name).read_text())["by_K"])
    return {K: one(files[pattern.format(K=K)], K) for K in ks}


def _chain_check(block_only):
    tag = "_block" if block_only else ""
    pattern = "chain_K{K}_rows" + tag + ".json"
    return {"rows": [pattern.format(K=K) for K in KS], "summary": f"chain_summary{tag}.json",
            # elapsed_sec is wall-clock time of the run.
            "ignore": ("elapsed_sec",),
            "fn": lambda files, run: {"model": run.name,
                                      "by_K": _by_K(files, pattern, lambda rows, K: chain_K_summary(rows, K, block_only),
                                                    run, f"chain_summary{tag}.json")}}


def _written_check(recut):
    rs = "_recut" if recut else ""
    pattern = "written_K{K}_rows" + rs + ".json"
    return {"rows": [pattern.format(K=K) for K in (3, 5)], "summary": f"written_summary{rs}.json",
            "fn": lambda files, run: {"model": run.name, "by_K": _by_K(files, pattern, lambda rows, K: written_K_summary(rows),
                                                                       run, f"written_summary{rs}.json")}}


CHECKS = [
    # stage0 is greedy-answer accuracy from generations the rows do not store; elapsed_sec is wall-clock time.
    {"rows": "stage01_rows.json", "summary": "stage01_summary.json", "ignore": ("stage0", "elapsed_sec"),
     "fn": lambda rows, run: stage01_summary(rows, run.name, None, None)},
    _chain_check(False),
    _chain_check(True),
    _written_check(False),
    _written_check(True),
]
