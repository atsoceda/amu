#!/usr/bin/env python3
"""Validate corrected factorial construction and cached logits against full prompts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.ab_reliability_preflight.run_preflight import (
    build_variant,
    evaluate_variant,
    heldout_suffix,
    one_id,
)
from experiments.lib.aan_protocol import write_json

EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text())
ART = EXP / "artifacts"


def swap_demo_codes(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return match.group(1) + ("B" if match.group(2) == "A" else "A")
    return re.sub(r"(Code \(A/B\): )([AB])", replace, text)


@torch.inference_mode()
def full_prompt_metrics(model, tokenizer, prompt: str, kind: str, row: dict, code_ids: dict[str, int]) -> dict:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    logits = model(**encoded, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
    if kind == "code":
        probs = torch.softmax(logits, dim=-1)
        return {"p_A": float(probs[code_ids["A"]]), "p_B": float(probs[code_ids["B"]]),
                "ab_mass": float(probs[code_ids["A"]] + probs[code_ids["B"]]),
                "B_minus_A_logit": float(logits[code_ids["B"]] - logits[code_ids["A"]])}
    everyday_id, formal_id = row["everyday_token_id"], row["formal_token_id"]
    pair_probs = torch.softmax(logits[[everyday_id, formal_id]], dim=0)
    return {"formal_minus_everyday_logit": float(logits[formal_id] - logits[everyday_id]),
            "formal_pair_probability": float(pair_probs[1])}


def main() -> None:
    split = json.loads((ART / "03_frozen_split.json").read_text())
    demos, row = split["demonstration"], split["development"][0]
    rel_cfg = CFG["reliability"]
    reliability, bank_index, order_index = 0.5, 0, 0
    bank_seed = rel_cfg["seed"] + int(reliability * 1000) * 10000 + bank_index
    choice_seed = rel_cfg["seed"] + int(reliability * 1000) * 10000 + 1000 + bank_index
    order_seed = rel_cfg["seed"] + int(reliability * 1000) * 10000 + 2000 + order_index
    a = build_variant(demos, reliability, bank_index, order_index, "everyday_A", bank_seed, order_seed, choice_seed)
    b = build_variant(demos, reliability, bank_index, order_index, "everyday_B", bank_seed, order_seed, choice_seed)
    order_alt = build_variant(demos, reliability, bank_index, 1, "everyday_A", bank_seed,
                              rel_cfg["seed"] + int(reliability * 1000) * 10000 + 2001, choice_seed)
    construction = {
        "label_complement_exact": swap_demo_codes(a["prefix"]) == b["prefix"],
        "label_complement_assignment_equal": a["assignment"] == b["assignment"],
        "order_repeat_assignment_equal": a["assignment"] == order_alt["assignment"],
        "order_repeat_prefix_differs": a["prefix"] != order_alt["prefix"],
    }
    model_cfg = CFG["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["snapshot"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_cfg["snapshot"], dtype=getattr(torch, model_cfg["dtype"]),
                                                 local_files_only=True, low_cpu_mem_usage=True).eval()
    code_ids = {letter: one_id(tokenizer, letter) for letter in ("A", "B")}
    cached = evaluate_variant(model, tokenizer, a, [row], code_ids, batch_size=1, inference_mode="cached")[0]
    comparisons = []
    choice_reversed = False
    for register in ("everyday", "formal"):
        suffix = heldout_suffix(row, register, choice_reversed, None)
        direct = full_prompt_metrics(model, tokenizer, a["prefix"] + suffix, "code", row, code_ids)
        cached_metrics = cached["code"][register]
        comparisons.append({"kind": "code", "register": register, "forced_code": None,
                            "direct": direct, "cached": cached_metrics,
                            "max_abs_difference": max(abs(direct[k] - cached_metrics[k]) for k in direct)})
        for forced_code in ("A", "B"):
            suffix = heldout_suffix(row, register, choice_reversed, forced_code)
            direct = full_prompt_metrics(model, tokenizer, a["prefix"] + suffix, "term", row, code_ids)
            cached_metrics = cached["term"][register][forced_code]
            comparisons.append({"kind": "term", "register": register, "forced_code": forced_code,
                                "direct": direct, "cached": cached_metrics,
                                "max_abs_difference": max(abs(direct[k] - cached_metrics[k]) for k in direct)})
    result = {"construction": construction, "comparisons": comparisons,
              "overall_max_abs_difference": max(row["max_abs_difference"] for row in comparisons),
              "passed": all(construction.values()) and max(row["max_abs_difference"] for row in comparisons) <= 0.02}
    write_json(ART / "04_inference_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
