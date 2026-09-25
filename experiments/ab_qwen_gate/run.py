#!/usr/bin/env python3
"""A/B learnability gate on Qwen3 (Priority 2b). Behavioral only; no interventions.

Re-runs the frozen A/B few-shot preflight (experiments/ab_fewshot_preflight/config.json:
same demonstrations, label banks and held-out families) on Qwen3 models, to decide
whether an in-context code becomes a mediator before any route assay is built on it.

Gate (prespecified 2026-09-25, before any Qwen3 A/B result):
  G1 code follows context: in the 100%-reliable bank, mean over families of
     P(B | formal target context) - P(B | casual source context) >= 0.30.
  G2 forced code moves the term: 100%-bank mean forced-code leverage
     (formal-minus-common probability under forced B minus forced A, neutral
     context) >= 0.15.
  G3 leverage depends on reliability: 50%-bank leverage <= 1/3 of the 100%-bank
     leverage.
  G4 support: mean P(A)+P(B) >= 0.90 in every bank.
The paradigm proceeds to a route assay only for a model passing all four.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.ab_fewshot_preflight.run import code_prompt, demonstrations, lexical_stats, term_prompt  # noqa: E402

EXP = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "experiments/ab_fewshot_preflight/config.json").read_text())


def first_id(tok, word):
    return tok.encode(" " + word, add_special_tokens=False)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"Qwen/{a.model}")
    m = AutoModelForCausalLM.from_pretrained(f"Qwen/{a.model}", dtype=torch.bfloat16).to("mps").eval()
    ids = {c: first_id(tok, c) for c in ("A", "B")}

    @torch.no_grad()
    def logits(text):
        x = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
        return m(x).logits[0, -1].float().cpu()

    rows = []
    for bank in ("high", "medium", "low"):
        prefix = demonstrations(CFG, bank)
        for item in CFG["heldout"]:
            cid, fid = first_id(tok, item["common_term"]), first_id(tok, item["formal_term"])
            code, term = {}, {}
            for ctx in ("neutral", "source", "target"):
                p = torch.softmax(logits(code_prompt(prefix, item[f"{ctx}_context"])), -1)
                code[ctx] = {"p_A": float(p[ids["A"]]), "p_B": float(p[ids["B"]]), "ab_mass": float(p[ids["A"]] + p[ids["B"]])}
                term[ctx] = {c: lexical_stats(logits(term_prompt(prefix, item[f"{ctx}_context"], c)), cid, fid, tok) for c in ("A", "B")}
            rows.append({"bank": bank, "family": item["family"], "code": code, "term": term,
                         "code_follows_context": code["target"]["p_B"] - code["source"]["p_B"],
                         "forced_code_leverage": term["neutral"]["B"]["formal_minus_common_probability"]
                         - term["neutral"]["A"]["formal_minus_common_probability"],
                         "fixed_code_context_effect": sum(term["target"][c]["formal_minus_common_probability"]
                                                          - term["source"][c]["formal_minus_common_probability"] for c in "AB") / 2})
            print(bank, item["family"], round(rows[-1]["code_follows_context"], 3), round(rows[-1]["forced_code_leverage"], 3), flush=True)
    mean = lambda b, k: sum(r[k] for r in rows if r["bank"] == b) / sum(r["bank"] == b for r in rows)  # noqa: E731
    mass = {b: sum(r["code"][c]["ab_mass"] for r in rows if r["bank"] == b for c in r["code"]) / (3 * sum(r["bank"] == b for r in rows)) for b in ("high", "medium", "low")}
    s = {"model": a.model, "by_bank": {b: {"code_follows_context": mean(b, "code_follows_context"),
                                            "forced_code_leverage": mean(b, "forced_code_leverage"),
                                            "fixed_code_context_effect": mean(b, "fixed_code_context_effect"),
                                            "ab_mass": mass[b]} for b in ("high", "medium", "low")}}
    h, lo = s["by_bank"]["high"], s["by_bank"]["low"]
    s["gate"] = {"G1_code_follows_context": h["code_follows_context"] >= 0.30,
                 "G2_forced_code_moves_term": h["forced_code_leverage"] >= 0.15,
                 "G3_leverage_depends_on_reliability": lo["forced_code_leverage"] <= h["forced_code_leverage"] / 3,
                 "G4_support": all(v >= 0.90 for v in mass.values())}
    s["gate"]["pass"] = all(s["gate"].values())
    out = EXP / "results" / a.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
