#!/usr/bin/env python3
"""Check headline numbers in the working paper against the committed experiment summaries.

Run from the repository root (standard library only, no model weights):
    python manuscript/claims.py

Every number in the manuscript is typed by hand. Each claim below names the exact text that
states it, the committed summaries it comes from, and how it is computed; the check fails if
the text is no longer in the manuscript or the summaries no longer give the stated value.
When an edit changes a headline number, update its claim here in the same commit.

Kinds of check (values in percent):
  range  min and max over the runs round to the stated endpoints
  max    the maximum rounds to the stated value
  min    the minimum rounds to the stated value
  bound  the maximum does not exceed the stated value
A claim with an "open" note is a known discrepancy awaiting an author decision: it is reported
as OPEN and does not fail the check. Remove the note once the text or the claim is corrected.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
SECTIONS = ROOT / "manuscript" / "sections"


def load(rel):
    return json.loads((EXP / rel).read_text())


def mean(v):
    return v["mean"] if isinstance(v, dict) else v


def pct(a, b):
    return 100 * mean(a) / mean(b)


def runs(exp, *files):
    """Run directories of experiments/<exp>/results that hold every named file."""
    base = EXP / exp / "results"
    return sorted(d.name for d in base.iterdir() if d.is_dir() and all((d / f).exists() for f in files))


# ---- values, each a dict {run label: percent} -------------------------------------------

def couplet_relay():
    return {d: pct(load(f"couplet_routes/results/{d}/relay_positions_summary.json")["all"]["d_relay_early"],
                   load(f"couplet_routes/results/{d}/relay_positions_summary.json")["all"]["d_persistence_p0"])
            for d in runs("couplet_routes", "step34_summary.json", "relay_positions_summary.json")}


def couplet_retrieval(exclude=("gemma-3-27b-it-plain",)):
    out = {}
    for d in runs("couplet_routes", "step34_summary.json", "relay_positions_summary.json"):
        if d in exclude:
            continue
        s = load(f"couplet_routes/results/{d}/step34_summary.json")["all"]
        p = load(f"couplet_routes/results/{d}/relay_positions_summary.json")["all"]
        out[d] = pct(s["d_retrieval"], p["d_persistence_p0"])
    return out


def couplet_exception():
    d = "gemma-3-27b-it-plain"
    s = load(f"couplet_routes/results/{d}/step34_summary.json")["all"]
    p = load(f"couplet_routes/results/{d}/relay_positions_summary.json")["all"]
    n = load(f"couplet_routes/results/{d}/relay_necessity_summary.json")["all"]
    return {"retrieval": pct(s["d_retrieval"], p["d_persistence_p0"]),
            "copy_sufficiency": pct(p["d_relay_tail"], p["d_persistence_p0"]),
            "copy_necessity": pct(n["necessity_tail"], n["d_persistence_p0"])}


def couplet_emission():
    out = {}
    for d in runs("couplet_routes", "step34_summary.json", "relay_positions_summary.json"):
        s = load(f"couplet_routes/results/{d}/step34_summary.json")["all"]
        out[d] = pct(s["d_emission"], s["d_total"])
    return out


def sums_after_question():
    return {d: pct(load(f"derived_value_carry/results/{d}/stage01_summary.json")["stage1"]["after_question"],
                   load(f"derived_value_carry/results/{d}/stage01_summary.json")["stage1"]["operands"])
            for d in runs("derived_value_carry", "stage01_summary.json")}


def chains_after_source():
    out = {}
    for d in runs("derived_value_carry", "chain_summary_block.json"):
        for K, k in load(f"derived_value_carry/results/{d}/chain_summary_block.json")["by_K"].items():
            if "post_v0_block" in k:
                out[f"{d} K={K}"] = pct(k["post_v0_block"], k["v0_edit"])
    return out


def written_emission():
    out = {}
    for d in runs("derived_value_carry"):
        for f in ("written_summary_recut.json", "written_summary.json"):
            if (EXP / f"derived_value_carry/results/{d}/{f}").exists():
                for K, k in load(f"derived_value_carry/results/{d}/{f}")["by_K"].items():
                    if k.get("total") is not None:
                        out[f"{d} K={K}"] = pct(k["emission"], k["total"])
                break
    return out


def distance(key):
    out = {}
    for f in ("pilot_v2x_summary.json", "pilot_v2x_window_summary.json"):
        for D, v in load(f"relay_distance/results/gemma-3-27b-it/{f}")["by_distance"].items():
            if v.get("n") and int(D) >= 500:
                out[f"D={D}"] = pct(v[key], v["persistence"])
    return out


def recurrent_only():
    out = {}
    for d in runs("recurrent_carry"):
        for f in ("pilot_summary.json", "pilot24_summary.json"):
            if (EXP / f"recurrent_carry/results/{d}/{f}").exists():
                x = load(f"recurrent_carry/results/{d}/{f}")
                out[d] = pct(x["recurrent_only"], x["persistence"])
                break
    return out


def choice_through_sentence():
    return {d: pct(load(f"hidden_choice/results/{d}/choice_summary.json")["post_relay"],
                   load(f"hidden_choice/results/{d}/choice_summary.json")["text_swap"])
            for d in runs("hidden_choice", "choice_summary.json")}


def choice_stored(models):
    """Pick-specific stored share: post-list patch over the text swap, both as pick specificity.
    Fruits (out-of-list control) and animals (committed replication) for each model."""
    out = {}
    for d in models:
        o = load(f"hidden_choice/results/{d}/choice_outlist_summary.json")
        out[f"{d} fruits"] = pct(o["pick_specificity_post"], mean(o["text_swap"]) - mean(o["text_swap_others"]))
        a = load(f"hidden_choice/results/{d}/choice_replicate_animals_summary.json")
        out[f"{d} animals"] = pct(a["post_specificity"], a["text_swap_specificity"])
    return out


def induction_relay():
    return {d: pct(load(f"relay_positive_control/results/{d}/induction_summary.json")["relay"],
                   load(f"relay_positive_control/results/{d}/induction_summary.json")["persistence"])
            for d in runs("relay_positive_control", "induction_summary.json")}


GEMMA3 = ("gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it")
QWEN3 = ("Qwen3-8B", "Qwen3-14B", "Qwen3-32B")

# ---- claims: (id, section file, exact text, values, kind, stated, note) ------------------
# A negative share is reported as 0 in the text ("0 to 7%"): clamp=True rounds max(x, 0).

CLAIMS = [
    ("couplet-relay-21-runs", "01-introduction.qmd", "relay accounts for at most 1% in all 21 runs",
     couplet_relay, "bound", 1, {"count": 21}),
    ("couplet-rhyme-word", "01-introduction.qmd", "68 to 95% of persistence in all but one configuration",
     couplet_retrieval, "range", (68, 95), {}),
    ("couplet-exception", "01-introduction.qmd",
     r"19\% in Gemma~3 27B, plain, where the stored copy is 37\% by sufficiency and 74\% by necessity",
     couplet_exception, "each", {"retrieval": 19, "copy_sufficiency": 37, "copy_necessity": 74}, {}),
    ("couplet-emission", "01-introduction.qmd", r"13--25\% of the total effect",
     couplet_emission, "range", (13, 25), {}),
    ("chains-after-source", "01-introduction.qmd", "move the answer by at most 4.9% (variable",
     chains_after_source, "max", 4.9, {"decimals": 1}),
    ("sums-after-question", "01-introduction.qmd", "chains) and 6.4% (sums) of the source effect",
     sums_after_question, "max", 6.4, {"decimals": 1}),
    ("written-emission", "01-introduction.qmd", "emission accounts for at least 83% of the effect",
     written_emission, "range", (83, 101), {}),
    ("distance-generated-text", "01-introduction.qmd", "the generated text accounts for 9 to 13%",
     lambda: distance("relay_line2"), "range", (9, 13), {}),
    ("distance-rhyme-word", "01-introduction.qmd", r"rhyme word 40--64\%",
     lambda: distance("retrieval"), "range", (40, 64), {}),
    ("distance-copy-necessity", "01-introduction.qmd", r"stored copy necessary for 25--49\%",
     lambda: distance("necessity_boundary"), "range", (25, 49), {}),
    ("recurrent-only", "01-introduction.qmd", "the recurrent layers produces 2 to 4% of",
     recurrent_only, "range", (2, 4), {}),
    ("choice-through-sentence", "01-introduction.qmd", "at most 2% of its effect passes through",
     choice_through_sentence, "max", 2, {}),
    ("choice-stored-range", "01-introduction.qmd", r"pick-specific information stored after the list: 0--40\%",
     lambda: choice_stored(QWEN3 + GEMMA3), "range", (0, 40), {"clamp": True}),
    ("choice-stored-gemma3", "05-stress-tests.qmd", "(Gemma 3 15 to 40%",
     lambda: choice_stored(GEMMA3), "range", (15, 40), {}),
    ("choice-stored-gemma4", "05-stress-tests.qmd", "31B 81 to 91% of the text swap's pick specificity",
     lambda: choice_stored(("gemma-4-31B-it",)), "range", (81, 91), {}),
    ("choice-stored-qwen3-large", "10-appendix.qmd", "0 to 8% at 14B and 32B",
     lambda: choice_stored(("Qwen3-14B", "Qwen3-32B")), "range", (0, 8), {"clamp": True}),
    ("choice-stored-qwen3-8b", "10-appendix.qmd", "10 to 15% at 8B",
     lambda: choice_stored(("Qwen3-8B",)), "range", (10, 15), {}),
    # Appendix table tbl-hidden, one claim per row.
    ("tbl-hidden Qwen3-8B", "10-appendix.qmd", r"Qwen3-8B & 10\% & 15\%",
     lambda d="Qwen3-8B": {"fruits": choice_stored((d,))[f"{d} fruits"], "animals": choice_stored((d,))[f"{d} animals"]},
     "each", {"fruits": 10, "animals": 15}, {"clamp": True}),
    ("tbl-hidden Qwen3-14B", "10-appendix.qmd", r"Qwen3-14B & 0\% & 3\%",
     lambda d="Qwen3-14B": {"fruits": choice_stored((d,))[f"{d} fruits"], "animals": choice_stored((d,))[f"{d} animals"]},
     "each", {"fruits": 0, "animals": 3}, {"clamp": True}),
    ("tbl-hidden Qwen3-32B", "10-appendix.qmd", r"Qwen3-32B & 2\% & 8\%",
     lambda d="Qwen3-32B": {"fruits": choice_stored((d,))[f"{d} fruits"], "animals": choice_stored((d,))[f"{d} animals"]},
     "each", {"fruits": 2, "animals": 8}, {"clamp": True}),
    ("tbl-hidden Qwen3.5-27B", "10-appendix.qmd", r"Qwen3.5-27B & 3\% & 4\%",
     lambda d="Qwen3.5-27B": {"fruits": choice_stored((d,))[f"{d} fruits"], "animals": choice_stored((d,))[f"{d} animals"]},
     "each", {"fruits": 3, "animals": 4}, {"clamp": True}),
    ("tbl-hidden Gemma 3 4B", "10-appendix.qmd", r"Gemma 3 4B & 18\% & 22\%",
     lambda d="gemma-3-4b-it": {"fruits": choice_stored((d,))[f"{d} fruits"], "animals": choice_stored((d,))[f"{d} animals"]},
     "each", {"fruits": 18, "animals": 22}, {"clamp": True}),
    ("tbl-hidden Gemma 3 12B", "10-appendix.qmd", r"Gemma 3 12B & 15\% & 18\%",
     lambda d="gemma-3-12b-it": {"fruits": choice_stored((d,))[f"{d} fruits"], "animals": choice_stored((d,))[f"{d} animals"]},
     "each", {"fruits": 15, "animals": 18}, {"clamp": True}),
    ("tbl-hidden Gemma 3 27B", "10-appendix.qmd", r"Gemma 3 27B & 31\% & 40\%",
     lambda d="gemma-3-27b-it": {"fruits": choice_stored((d,))[f"{d} fruits"], "animals": choice_stored((d,))[f"{d} animals"]},
     "each", {"fruits": 31, "animals": 40}, {"clamp": True}),
    ("tbl-hidden Gemma 4 31B", "10-appendix.qmd", r"Gemma 4 31B & 81\% & 91\%",
     lambda d="gemma-4-31B-it": {"fruits": choice_stored((d,))[f"{d} fruits"], "animals": choice_stored((d,))[f"{d} animals"]},
     "each", {"fruits": 81, "animals": 91}, {"clamp": True}),
    ("induction-relay", "01-introduction.qmd", r"through the copied token's position (prompt): 53--137\%",
     induction_relay, "range", (53, 137), {"count": 9}),
]


def rnd(x, decimals, clamp):
    x = max(x, 0.0) if clamp else x
    return round(x, decimals) if decimals else int(round(x))


def check(claim, text_of):
    cid, section, text, fn, kind, stated, opt = claim
    problems = []
    if " ".join(text.split()) not in text_of(section):
        problems.append(f"text not found in {section}: {text!r}")
    try:
        vals = fn()
    except (FileNotFoundError, KeyError) as e:
        return problems + [f"cannot compute: {type(e).__name__}: {e}"], ""
    d, clamp = opt.get("decimals", 0), opt.get("clamp", False)
    v = list(vals.values())
    if "count" in opt and len(v) != opt["count"]:
        problems.append(f"{len(v)} runs, text implies {opt['count']}")
    if kind == "each":
        got = {k: rnd(vals[k], d, clamp) for k in stated}
        if got != stated:
            problems.append(f"computed {got}, stated {stated}")
        shown = ", ".join(f"{k} {vals[k]:.1f}" for k in stated)
    elif kind == "range":
        got = (rnd(min(v), d, clamp), rnd(max(v), d, clamp))
        if got != tuple(stated):
            problems.append(f"computed {got}, stated {tuple(stated)}")
        shown = f"{min(v):.2f} ({min(vals, key=vals.get)}) to {max(v):.2f} ({max(vals, key=vals.get)})"
    elif kind == "max":
        if rnd(max(v), d, clamp) != stated:
            problems.append(f"computed max {max(v):.2f}, stated {stated}")
        shown = f"max {max(v):.2f} ({max(vals, key=vals.get)})"
    elif kind == "bound":
        if max(v) > stated:
            problems.append(f"computed max {max(v):.2f} exceeds {stated}")
        shown = f"max {max(v):.2f} ({max(vals, key=vals.get)}), n={len(v)}"
    else:
        raise ValueError(kind)
    return problems, shown


def main():
    cache = {}

    def text_of(section):
        if section not in cache:
            cache[section] = " ".join((SECTIONS / section).read_text().split())
        return cache[section]

    failed = opened = 0
    for claim in CLAIMS:
        problems, shown = check(claim, text_of)
        note = claim[6].get("open")
        status = "ok  " if not problems else ("OPEN" if note else "FAIL")
        print(f"{status} {claim[0]:28s} {shown}")
        for p in problems + ([f"open: {note}"] if problems and note else []):
            print(f"     - {p}")
        failed += bool(problems) and not note
        opened += bool(problems) and bool(note)
    print(f"{len(CLAIMS) - failed - opened}/{len(CLAIMS)} claims verified, {opened} open, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
