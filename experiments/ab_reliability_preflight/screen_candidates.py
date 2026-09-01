#!/usr/bin/env python3
"""Model-specific lexical register eligibility screen, before any A/B assay."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.lib.aan_protocol import write_json

EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text())
ART = EXP / "artifacts"


def one_token_id(tokenizer, word: str) -> int | None:
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def prompt(register: str, meaning: str, left: str, right: str) -> str:
    return ("Choose the term that best expresses the meaning in the requested register.\n"
            f"Register: {register}\nMeaning: {meaning}\nChoices: {left} | {right}\nTerm:")


def orientation_metrics(cells: list[dict[str, Any]], sign: float) -> tuple[float, float]:
    """Return minimum and mean register contrast after averaging choice-order nuisance."""
    per_paraphrase = []
    # Paraphrases are paired in config order; compute within each requested register pair.
    for paraphrase_index, (everyday_label, formal_label) in enumerate(CFG["lexical_screen"]["register_paraphrases"]):
        everyday = [sign * cell["formal_minus_everyday_logit"] for cell in cells
                    if cell["paraphrase_index"] == paraphrase_index and cell["register_role"] == "everyday"]
        formal = [sign * cell["formal_minus_everyday_logit"] for cell in cells
                 if cell["paraphrase_index"] == paraphrase_index and cell["register_role"] == "formal"]
        if not everyday or not formal:
            continue
        everyday_mean, formal_mean = sum(everyday) / len(everyday), sum(formal) / len(formal)
        per_paraphrase.append(formal_mean - everyday_mean)
    if not per_paraphrase:
        return float("-inf"), float("-inf")
    return min(per_paraphrase), sum(per_paraphrase) / len(per_paraphrase)


@torch.inference_mode()
def evaluate_tasks(model, tokenizer, tasks: list[dict[str, Any]], batch_size: int = 8) -> None:
    for start in range(0, len(tasks), batch_size):
        batch_tasks = tasks[start:start + batch_size]
        encoded = tokenizer([task["prompt"] for task in batch_tasks], return_tensors="pt", padding=True,
                            add_special_tokens=True)
        logits = model(**encoded, use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
        for task, row_logits in zip(batch_tasks, logits):
            raw_margin = float(row_logits[task["formal_id"]] - row_logits[task["everyday_id"]])
            signed_margin = raw_margin if task["register_role"] == "formal" else -raw_margin
            pair_logits = row_logits[[task["everyday_id"], task["formal_id"]]]
            probs = torch.softmax(pair_logits, dim=0)
            expected_index = 1 if task["register_role"] == "formal" else 0
            task["cell"].update({
                "formal_minus_everyday_logit": raw_margin,
                "signed_expected_logit_margin": signed_margin,
                "expected_pair_conditional_probability": float(probs[expected_index]),
            })
        print(f"screened {min(start + batch_size, len(tasks))}/{len(tasks)} prompt cells", flush=True)


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    register_terms = re.compile(
        r"formal|informal|everyday|colloquial|technical|professional|official|clinical|administrative|"
        r"legal|literary|academic|scientific", re.IGNORECASE
    )
    seen_pairs: set[tuple[str, str]] = set()
    kept = []
    for row in rows:
        if not register_terms.search(row.get("distinction_note", "")):
            continue
        pair = tuple(sorted((row["everyday_term"], row["formal_term"])))
        family = row["family_id"]
        if not all((family, pair[0], pair[1], row["shared_meaning"])):
            continue
        if pair in seen_pairs or pair[0] == pair[1]:
            continue
        seen_pairs.add(pair)
        kept.append({**row, "family_id": row["candidate_id"]})
    return kept


def main() -> None:
    candidate_path = ART / "01_candidates/all_candidates.json"
    candidates = deduplicate(json.loads(candidate_path.read_text()))
    model_cfg = CFG["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["snapshot"], local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["snapshot"], dtype=getattr(torch, model_cfg["dtype"]),
        local_files_only=True, low_cpu_mem_usage=True,
    ).eval()
    screen = CFG["lexical_screen"]
    rows = []
    tasks = []
    for candidate in candidates:
        everyday_id = one_token_id(tokenizer, candidate["everyday_term"])
        formal_id = one_token_id(tokenizer, candidate["formal_term"])
        cells: list[dict[str, Any]] = []
        if everyday_id is not None and formal_id is not None:
            for paraphrase_index, (everyday_label, formal_label) in enumerate(screen["register_paraphrases"][:1]):
                for register_role, register_text in (("everyday", everyday_label), ("formal", formal_label)):
                    for choice_order, choices in (("everyday_first", (candidate["everyday_term"], candidate["formal_term"])),
                                                   ("formal_first", (candidate["formal_term"], candidate["everyday_term"]))):
                        prompt_text = prompt(register_text, candidate["shared_meaning"], *choices)
                        cell = {
                    "paraphrase_index": paraphrase_index,
                    "register_role": register_role,
                    "register_text": register_text,
                    "choice_order": choice_order,
                    "prompt": prompt_text,
                        }
                        cells.append(cell)
                        tasks.append({"prompt": prompt_text, "register_role": register_role,
                                      "everyday_id": everyday_id, "formal_id": formal_id, "cell": cell})
        rows.append({**candidate, "everyday_token_id": everyday_id, "formal_token_id": formal_id,
                     "single_token_pair": everyday_id is not None and formal_id is not None, "cells": cells})

    evaluate_tasks(model, tokenizer, tasks)
    second_stage_tasks = []
    for row in rows:
        cells = row["cells"]
        if not cells:
            continue
        as_is = orientation_metrics(cells, 1.0)
        reversed_metrics = orientation_metrics(cells, -1.0)
        best = max(as_is, reversed_metrics, key=lambda value: value[1])
        if best[0] <= 0.1:
            continue
        candidate = row
        everyday_id, formal_id = row["everyday_token_id"], row["formal_token_id"]
        for paraphrase_index, (everyday_label, formal_label) in enumerate(screen["register_paraphrases"][1:], start=1):
            for register_role, register_text in (("everyday", everyday_label), ("formal", formal_label)):
                for choice_order, choices in (("everyday_first", (candidate["everyday_term"], candidate["formal_term"])),
                                               ("formal_first", (candidate["formal_term"], candidate["everyday_term"]))):
                    prompt_text = prompt(register_text, candidate["shared_meaning"], *choices)
                    cell = {"paraphrase_index": paraphrase_index, "register_role": register_role, "register_text": register_text,
                            "choice_order": choice_order, "prompt": prompt_text}
                    cells.append(cell)
                    second_stage_tasks.append({"prompt": prompt_text, "register_role": register_role,
                                               "everyday_id": everyday_id, "formal_id": formal_id, "cell": cell})
    print(f"preliminary gate retained {len(second_stage_tasks) // 8} candidate pairs", flush=True)
    evaluate_tasks(model, tokenizer, second_stage_tasks)
    finalized = []
    for row in rows:
        candidate = {key: value for key, value in row.items() if key not in
                     ("everyday_token_id", "formal_token_id", "single_token_pair", "cells")}
        everyday_id = row["everyday_token_id"]
        formal_id = row["formal_token_id"]
        cells = row["cells"]
        as_is = orientation_metrics(cells, 1.0)
        reversed_metrics = orientation_metrics(cells, -1.0)
        orientation_reversed = reversed_metrics[1] > as_is[1]
        minimum_contrast, mean_contrast = reversed_metrics if orientation_reversed else as_is
        margins = [cell["signed_expected_logit_margin"] for cell in cells]
        if orientation_reversed:
            margins = [-value for value in margins]
            for cell in cells:
                cell["signed_expected_logit_margin"] *= -1
                cell["expected_pair_conditional_probability"] = 1.0 - cell["expected_pair_conditional_probability"]
            candidate = {
                **candidate,
                "everyday_term": candidate["formal_term"],
                "formal_term": candidate["everyday_term"],
            }
            everyday_id, formal_id = formal_id, everyday_id
        eligible = bool(cells) and minimum_contrast > screen["minimum_order_averaged_register_contrast_each_paraphrase"] \
            and mean_contrast >= screen["minimum_mean_register_contrast"]
        finalized.append({
            **candidate,
            "everyday_token_id": everyday_id,
            "formal_token_id": formal_id,
            "single_token_pair": everyday_id is not None and formal_id is not None,
            "orientation_selected_by_gemma_screen": "reversed" if orientation_reversed else "as_imported",
            "cells": cells,
            "minimum_signed_logit_margin": min(margins) if margins else None,
            "mean_signed_logit_margin": sum(margins) / len(margins) if margins else None,
            "minimum_order_averaged_register_contrast": minimum_contrast,
            "mean_register_contrast": mean_contrast,
            "eligible": eligible,
        })
    rows = finalized
    out_dir = ART / "02_lexical_screen"
    out_dir.mkdir(parents=True, exist_ok=True)
    eligible_rows = [row for row in rows if row["eligible"]]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_rows_after_deduplication": len(candidates),
        "single_token_pairs": sum(row["single_token_pair"] for row in rows),
        "eligible_pairs": len(eligible_rows),
        "thresholds": screen,
        "route_outcomes_used_for_selection": False,
    }
    write_json(out_dir / "rows.json", rows)
    write_json(out_dir / "eligible.json", eligible_rows)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
