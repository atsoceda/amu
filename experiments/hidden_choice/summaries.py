"""Torch-free summaries for the hidden-choice runners, recomputable from committed rows.

Each function is the summary block that used to sit at the end of the runner's
main(); the runners call these, so new runs and `bin/resummarize` build summaries
the same way. `model` is the run directory name (the runner's model argument).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "couplet_routes"))
from route_stats import boot  # noqa: E402

ROUTE_KEYS = ["text_swap", "list_edit", "post_edit", "post_retrieval", "post_relay", "sentence_edit", "post_and_sentence"]


def choice_summary(rows, model, lists_tried):
    """choice_routes.py -> choice_summary.json."""
    return {"model": model, "n": len(rows), "lists_tried": lists_tried, **{k: boot([r[k] for r in rows]) for k in ROUTE_KEYS}}


def free_summary(rows, model):
    """choice_free.py -> choice_free_summary.json."""
    ch = [r for r in rows if r["sentence_changed"]]
    return {"model": model, "n": len(rows), "n_sentence_changed": len(ch),
            "n_names_fruit": sum(r["names_fruit"] for r in rows),
            **{k: boot([r[k] for r in rows]) for k in ("total", "emission", "persistence_t1", "persistence_t0")},
            "emission_when_changed": boot([r["emission"] for r in ch])}


def outlist_summary(rows, model):
    """choice_outlist.py -> choice_outlist_summary.json."""
    return {"model": model, "n": len(rows),
            **{k: boot([r[k] for r in rows]) for k in ("text_swap", "list_edit", "post_edit", "post_edit_others", "text_swap_others")},
            "pick_specificity_post": boot([r["post_edit"] - r["post_edit_others"] for r in rows]),
            "donor_rank_off_mean": sum(r["donor_rank_off"] for r in rows) / len(rows),
            "donor_rank_post_mean": sum(r["donor_rank_post"] for r in rows) / len(rows)}


def replicate_summary(rows, model, domain, localize):
    """choice_replicate.py (pick replication, with or without --localize)."""
    s = {"model": model, "domain": domain, "n": len(rows),
         **{k_: boot([r[k_] for r in rows]) for k_ in ("text_swap_pick", "text_swap_specificity", "post_pick", "post_specificity")}}
    if localize and rows:
        ref = rows[0]["post_tokens"]
        same = [r for r in rows if r["post_tokens"] == ref]
        s["necessity_by_token"] = [{"token": t, **boot([r["necessity_by_token"][i] for r in same])} for i, t in enumerate(ref)]
    return s


def samepick_summary(rows, model):
    """choice_replicate.py --null-same-pick."""
    return {"model": model, "n": len(rows), "text_swap_rel": boot([r["text_swap_rel"] for r in rows]),
            "post_rel": boot([r["post_rel"] for r in rows])}


def replicate_tag(domain, localize, wording, samepick):
    """File-name tag used by choice_replicate.py."""
    tag = f"_{domain}" + ("_localize" if localize else "") + ("" if wording == "base" else f"_{wording}")
    return tag + "_samepick" if samepick else tag


def _replicate_checks():
    out = []
    for domain in ("animals", "fruits"):
        for localize in (False, True):
            for wording in ("base", "alt"):
                for samepick in (False, True):
                    tag = replicate_tag(domain, localize, wording, samepick)
                    if samepick:
                        fn = lambda rows, run, tag=tag: samepick_summary(rows, run.name)  # noqa: E731
                    else:
                        fn = lambda rows, run, d=domain, lo=localize: replicate_summary(rows, run.name, d, lo)  # noqa: E731
                    out.append({"rows": f"choice_replicate{tag}_rows.json", "summary": f"choice_replicate{tag}_summary.json", "fn": fn})
    return out


CHECKS = [
    # lists_tried counts screened lists, including those with no usable donor; rows hold only kept pairs.
    {"rows": "choice_rows.json", "summary": "choice_summary.json", "ignore": ("lists_tried",),
     "fn": lambda rows, run: choice_summary(rows, run.name, None)},
    {"rows": "choice_free_rows.json", "summary": "choice_free_summary.json", "fn": lambda rows, run: free_summary(rows, run.name)},
    {"rows": "choice_outlist_rows.json", "summary": "choice_outlist_summary.json", "fn": lambda rows, run: outlist_summary(rows, run.name)},
    *_replicate_checks(),
]
