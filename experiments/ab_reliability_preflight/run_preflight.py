#!/usr/bin/env python3
"""Run the randomized, label-swapped behavioral A/B reliability preflight."""

from __future__ import annotations

import copy
import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.lib.aan_protocol import write_json

EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text())
ART = EXP / "artifacts"
RESULTS = EXP / "results"


def one_id(tokenizer, word: str) -> int:
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"expected one token for {word!r}, got {ids}")
    return ids[0]


def code_for(register: str, canonical: bool, role: str) -> str:
    everyday_code, formal_code = (("A", "B") if role == "everyday_A" else ("B", "A"))
    if not canonical:
        everyday_code, formal_code = formal_code, everyday_code
    return everyday_code if register == "everyday" else formal_code


def block(row: dict[str, Any], register: str, code: str, choice_reversed: bool) -> str:
    everyday, formal = row["everyday_term"], row["formal_term"]
    left, right = ((formal, everyday) if choice_reversed else (everyday, formal))
    register_text = "EVERYDAY" if register == "everyday" else "FORMAL"
    term = everyday if register == "everyday" else formal
    return (f"Register: {register_text}\nMeaning: {row['shared_meaning']}\n"
            f"Choices: {left} | {right}\nCode (A/B): {code}\nTerm: {term}")


def build_variant(demos: list[dict[str, Any]], reliability: float, bank_index: int,
                  order_index: int, role: str, bank_seed: int, order_seed: int,
                  choice_seed: int) -> dict[str, Any]:
    bank_rng = random.Random(bank_seed)
    order_rng = random.Random(order_seed)
    choice_rng = random.Random(choice_seed)
    n_canonical = round(reliability * len(demos))
    canonical_ids = set(bank_rng.sample(range(len(demos)), n_canonical))
    family_order = list(range(len(demos)))
    order_rng.shuffle(family_order)
    blocks = []
    choice_flags = {}
    balanced_flags = [False] * len(demos) + [True] * len(demos)
    choice_rng.shuffle(balanced_flags)
    for flag_index, key in enumerate((family_index, register)
                                     for family_index in range(len(demos))
                                     for register in ("everyday", "formal")):
        choice_flags[key] = balanced_flags[flag_index]
    assignment = {}
    for family_index in family_order:
        registers = ["everyday", "formal"]
        order_rng.shuffle(registers)
        canonical = family_index in canonical_ids
        assignment[demos[family_index]["family_id"]] = "canonical" if canonical else "reversed"
        for register in registers:
            blocks.append(block(demos[family_index], register, code_for(register, canonical, role),
                                choice_flags[(family_index, register)]))
    return {
        "variant_id": f"r{int(reliability*1000):04d}_b{bank_index}_o{order_index}_{role}",
        "reliability": reliability,
        "bank_index": bank_index,
        "order_index": order_index,
        "label_role": role,
        "canonical_families": n_canonical,
        "assignment": assignment,
        "bank_seed": bank_seed,
        "order_seed": order_seed,
        "choice_seed": choice_seed,
        "prefix": "Choose the term that best expresses the meaning in the requested register.\n\n" + "\n\n".join(blocks),
    }


def heldout_suffix(row: dict[str, Any], register: str, choice_reversed: bool, forced_code: str | None) -> str:
    everyday, formal = row["everyday_term"], row["formal_term"]
    left, right = ((formal, everyday) if choice_reversed else (everyday, formal))
    register_text = "EVERYDAY" if register == "everyday" else "FORMAL"
    text = (f"\n\nRegister: {register_text}\nMeaning: {row['shared_meaning']}\n"
            f"Choices: {left} | {right}\nCode (A/B):")
    if forced_code is not None:
        text += f" {forced_code}\nTerm:"
    return text


@torch.inference_mode()
def evaluate_variant(model, tokenizer, variant: dict[str, Any], heldout: list[dict[str, Any]],
                     code_ids: dict[str, int], batch_size: int = 4,
                     inference_mode: str = "direct") -> list[dict[str, Any]]:
    tasks = []
    for family_index, row in enumerate(heldout):
        # Alternate held-out choice order across repeated variants without coupling it to reliability or labels.
        choice_reversed = (family_index + variant["bank_index"] + variant["order_index"]) % 2 == 1
        for register in ("everyday", "formal"):
            tasks.append({"kind": "code", "row": row, "register": register, "forced_code": None,
                          "choice_reversed": choice_reversed,
                          "suffix": heldout_suffix(row, register, choice_reversed, None)})
            for forced_code in ("A", "B"):
                tasks.append({"kind": "term", "row": row, "register": register, "forced_code": forced_code,
                              "choice_reversed": choice_reversed,
                              "suffix": heldout_suffix(row, register, choice_reversed, forced_code)})
    if inference_mode == "direct":
        for start in range(0, len(tasks), batch_size):
            batch_tasks = tasks[start:start + batch_size]
            encoded = tokenizer([variant["prefix"] + task["suffix"] for task in batch_tasks],
                                return_tensors="pt", padding=True, add_special_tokens=True)
            out = model(**encoded, use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
            assign_metrics(batch_tasks, out, code_ids)
    elif inference_mode == "cached":
        prefix_tokens = tokenizer(variant["prefix"], return_tensors="pt", add_special_tokens=True)
        prefix_out = model(**prefix_tokens, use_cache=True, logits_to_keep=1)
        base_cache = prefix_out.past_key_values
        prefix_length = prefix_tokens.input_ids.shape[1]
        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for task in tasks:
            task["input_ids"] = tokenizer.encode(task["suffix"], add_special_tokens=False, return_tensors="pt")[0]
            by_length[len(task["input_ids"])].append(task)
        for suffix_length, length_tasks in sorted(by_length.items()):
            for start in range(0, len(length_tasks), batch_size):
                batch_tasks = length_tasks[start:start + batch_size]
                input_ids = torch.stack([task["input_ids"] for task in batch_tasks])
                cache = copy.deepcopy(base_cache)
                cache.batch_repeat_interleave(len(batch_tasks))
                attention_mask = torch.ones((len(batch_tasks), prefix_length + suffix_length), dtype=torch.long)
                cache_position = torch.arange(prefix_length, prefix_length + suffix_length)
                out = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=cache,
                            cache_position=cache_position, use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
                for task in batch_tasks:
                    task.pop("input_ids", None)
                assign_metrics(batch_tasks, out, code_ids)
        del base_cache
    else:
        raise ValueError(f"unknown inference mode: {inference_mode}")
    grouped = {}
    for task in tasks:
        key = task["row"]["family_id"]
        item = grouped.setdefault(key, {"family_id": key, "everyday_term": task["row"]["everyday_term"],
                                        "formal_term": task["row"]["formal_term"], "choice_reversed": task["choice_reversed"],
                                        "code": {}, "term": {"everyday": {}, "formal": {}}})
        if task["kind"] == "code":
            item["code"][task["register"]] = task["metrics"]
        else:
            item["term"][task["register"]][task["forced_code"]] = task["metrics"]
    return list(grouped.values())


def assign_metrics(tasks: list[dict[str, Any]], logits_rows: torch.Tensor,
                   code_ids: dict[str, int]) -> None:
    for task, logits in zip(tasks, logits_rows):
        if task["kind"] == "code":
            probs = torch.softmax(logits, dim=-1)
            task["metrics"] = {"p_A": float(probs[code_ids["A"]]), "p_B": float(probs[code_ids["B"]]),
                               "ab_mass": float(probs[code_ids["A"]] + probs[code_ids["B"]]),
                               "B_minus_A_logit": float(logits[code_ids["B"]] - logits[code_ids["A"]])}
        else:
            everyday_id, formal_id = task["row"]["everyday_token_id"], task["row"]["formal_token_id"]
            pair_probs = torch.softmax(logits[[everyday_id, formal_id]], dim=0)
            task["metrics"] = {"formal_minus_everyday_logit": float(logits[formal_id] - logits[everyday_id]),
                               "formal_pair_probability": float(pair_probs[1])}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-variants", type=int)
    parser.add_argument("--levels", default="0.5,1.0")
    parser.add_argument("--inference-mode", choices=("direct", "cached"), default="direct")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    split = json.loads((ART / "03_frozen_split.json").read_text())
    demos, heldout = split["demonstration"], split[CFG["gates"]["preflight_split"]]
    model_cfg = CFG["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["snapshot"], local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_cfg["snapshot"], dtype=getattr(torch, model_cfg["dtype"]),
                                                 local_files_only=True, low_cpu_mem_usage=True).eval()
    code_ids = {letter: one_id(tokenizer, letter) for letter in ("A", "B")}
    rel_cfg = CFG["reliability"]
    variants = []
    selected_levels = [float(value) for value in args.levels.split(",")]
    for reliability in selected_levels:
        for bank_index in range(rel_cfg["banks_per_level"]):
            bank_seed = rel_cfg["seed"] + int(reliability * 1000) * 10000 + bank_index
            choice_seed = rel_cfg["seed"] + int(reliability * 1000) * 10000 + 1000 + bank_index
            for order_index in range(rel_cfg["demonstration_orders_per_bank"]):
                order_seed = rel_cfg["seed"] + int(reliability * 1000) * 10000 + 2000 + order_index
                for role in rel_cfg["label_roles"]:
                    variants.append(build_variant(demos, reliability, bank_index, order_index, role,
                                                  bank_seed, order_seed, choice_seed))
    if args.max_variants is not None:
        variants = variants[:args.max_variants]
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows_path = RESULTS / "rows.json"
    all_rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    completed = {variant_id for variant_id in {row["variant_id"] for row in all_rows}
                 if sum(row["variant_id"] == variant_id for row in all_rows) == len(heldout)}
    for index, variant in enumerate(variants):
        if variant["variant_id"] in completed:
            print(f"variant {index + 1}/{len(variants)} {variant['variant_id']} cached", flush=True)
            continue
        family_rows = evaluate_variant(model, tokenizer, variant, heldout, code_ids,
                                       batch_size=args.batch_size, inference_mode=args.inference_mode)
        for row in family_rows:
            everyday_code, formal_code = (("A", "B") if variant["label_role"] == "everyday_A" else ("B", "A"))
            code_everyday, code_formal = row["code"]["everyday"], row["code"]["formal"]
            term = row["term"]
            role_policy_movement = ((code_formal[f"p_{formal_code}"] - code_everyday[f"p_{formal_code}"])
                                    - (code_formal[f"p_{everyday_code}"] - code_everyday[f"p_{everyday_code}"]))
            branch_probability = (term["everyday"][formal_code]["formal_pair_probability"]
                                  - term["everyday"][everyday_code]["formal_pair_probability"])
            fixed_effects = [term["formal"][code]["formal_pair_probability"]
                             - term["everyday"][code]["formal_pair_probability"] for code in ("A", "B")]
            all_rows.append({**{k: v for k, v in variant.items() if k != "prefix"}, **row,
                             "role_policy_movement": role_policy_movement,
                             "forced_code_branch_leverage": branch_probability,
                             "mean_fixed_code_register_effect": mean(fixed_effects)})
        write_json(rows_path, all_rows)
        print(f"variant {index + 1}/{len(variants)} {variant['variant_id']}", flush=True)
    summary_rows = []
    for reliability in selected_levels:
        group = [row for row in all_rows if row["reliability"] == reliability]
        if not group:
            continue
        summary_rows.append({
            "reliability": reliability,
            "families": len({row["family_id"] for row in group}),
            "repeated_rows": len(group),
            "ab_mass_mean": mean([cell["ab_mass"] for row in group for cell in row["code"].values()]),
            "ab_mass_min": min(cell["ab_mass"] for row in group for cell in row["code"].values()),
            "role_policy_movement_mean": mean([row["role_policy_movement"] for row in group]),
            "forced_code_branch_leverage_mean": mean([row["forced_code_branch_leverage"] for row in group]),
            "fixed_code_register_effect_mean": mean([row["mean_fixed_code_register_effect"] for row in group]),
        })
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "model": model_cfg["name"],
               "split": CFG["gates"]["preflight_split"], "confirmatory_assayed": False,
               "residual_interventions_run": False, "inference_mode": args.inference_mode,
               "batch_size": args.batch_size, "selected_levels": selected_levels,
               "variants": len(variants), "by_reliability": summary_rows}
    write_json(rows_path, all_rows)
    write_json(RESULTS / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
