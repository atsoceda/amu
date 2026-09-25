#!/usr/bin/env python3
"""Relay-tail analysis (existing data only): what distinguishes couplets with a
large relay share (> 30% of persistence with the original line-2 words)?

Compares high-relay couplets with the rest on: whether line 2 repeats line 1
(off and on), whether the line-2 prefix contains a donor- or original-rhyme word,
the number of positions between anchor and target, and the size of persistence.
Writes results/<model>/relay_tail.json.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
from step2_state_edit import rhymes  # noqa: E402

for m in ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]:
    d = EXP / "results" / m
    if not (d / "step34_rows.json").exists():
        continue
    r34 = {x["idx"]: x for x in json.loads((d / "step34_rows.json").read_text())}
    r2 = {x["idx"]: x for x in json.loads((d / "step2_rows.json").read_text())}
    feats = []
    for i, x in r34.items():
        if abs(x["d_persistence_p0"]) <= 1:
            continue
        g = r2[i]
        words = [w.strip(".,;:!?'\"") for w in x["prefix_off"].lower().split()]
        feats.append({"idx": i, "share": x["d_relay"] / x["d_persistence_p0"],
                      "line2_repeats_line1_off": "\n" in g["gen_off"].strip(),
                      "line2_repeats_line1_on": "\n" in g["gen_on"].strip(),
                      "donor_rhyme_in_prefix": any(w in rhymes(x["donor_word"]) for w in words),
                      "orig_rhyme_in_prefix": any(w in rhymes(x["orig_word"]) for w in words),
                      "n_between": x["n_between"], "persistence": x["d_persistence_p0"]})
    hi = [f for f in feats if f["share"] > 0.3]
    lo = [f for f in feats if f["share"] <= 0.3]
    out = {"model": m, "n_high": len(hi), "n_rest": len(lo), "high_idx": [f["idx"] for f in hi]}
    for k in ["line2_repeats_line1_off", "line2_repeats_line1_on", "donor_rhyme_in_prefix", "orig_rhyme_in_prefix"]:
        out[k] = {"high": sum(f[k] for f in hi), "rest": sum(f[k] for f in lo)}
    for k in ["n_between", "persistence"]:
        out[f"median_{k}"] = {"high": st.median(f[k] for f in hi) if hi else None, "rest": st.median(f[k] for f in lo)}
    (d / "relay_tail.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
