#!/usr/bin/env python3
"""Behavioral A/B few-shot mediator preflight on Gemma 3 1B PT."""

from __future__ import annotations

import json
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.gemma_1b_residual_scale.run import first_id
from experiments.lib.aan_protocol import token_id_for_text, write_json


EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write a restart-safe checkpoint without exposing a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def demonstrations(cfg: dict[str, Any], bank: str) -> str:
    blocks = []
    labels = cfg["label_banks"][bank]
    for demo, (common_code, formal_code) in zip(cfg["demonstrations"], labels):
        blocks.extend([
            f"Context: {demo['common_context']}\nCode: {common_code}\nTerm: {demo['common_term']}",
            f"Context: {demo['formal_context']}\nCode: {formal_code}\nTerm: {demo['formal_term']}",
        ])
    return "\n\n".join(blocks)


def code_prompt(prefix: str, context: str) -> str:
    return f"{prefix}\n\nContext: {context}\nCode:"


def term_prompt(prefix: str, context: str, code: str) -> str:
    return f"{code_prompt(prefix, context)} {code}\nTerm:"


def lexical_stats(logits: torch.Tensor, common_id: int, formal_id: int, tokenizer) -> dict[str, Any]:
    probs = torch.softmax(logits, -1)
    top = int(logits.argmax())
    return {
        "common_probability": float(probs[common_id]),
        "formal_probability": float(probs[formal_id]),
        "formal_minus_common_probability": float(probs[formal_id] - probs[common_id]),
        "formal_minus_common_logit": float(logits[formal_id] - logits[common_id]),
        "pair_probability_mass": float(probs[formal_id] + probs[common_id]),
        "top_token": tokenizer.decode([top]).strip(),
        "top_is_pair_member": top in (common_id, formal_id),
    }


@torch.inference_mode()
def batch_logits(model, tokenizer, prompts: list[str], batch_size: int = 4) -> list[torch.Tensor]:
    outputs = []
    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[start:start + batch_size], return_tensors="pt", padding=True, add_special_tokens=True)
        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
        outputs.extend(logits.unbind(0))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=EXP / "results_corrected_direct",
        help="Checkpoint/output directory (historical results are never overwritten by default).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    cfg = json.loads((EXP / "config.json").read_text())
    results = args.results_dir.resolve()
    results.mkdir(parents=True, exist_ok=True)
    rows_path = results / "rows.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    completed = {(r["bank"], r["family"]) for r in rows}
    run_metadata = {
        "experiment": cfg["experiment_name"],
        "started_or_resumed_at": datetime.now(timezone.utc).isoformat(),
        "model": cfg["model"],
        "model_snapshot": cfg["model_snapshot"],
        "dtype": cfg["dtype"],
        "inference_mode": "direct_full_prompt",
        "use_cache": False,
        "batch_size": args.batch_size,
        "checkpoint_granularity": "one bank-family cell",
        "config": cfg,
    }
    atomic_write_json(results / "run_metadata.json", run_metadata)
    tok = AutoTokenizer.from_pretrained(cfg["model_snapshot"], local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_snapshot"], dtype=getattr(torch, cfg["dtype"]),
        local_files_only=True, low_cpu_mem_usage=True
    ).eval()
    code_ids = {code: token_id_for_text(tok, f" {code}") for code in ("A", "B")}
    for bank in ("high", "medium", "low"):
        prefix = demonstrations(cfg, bank)
        for item in cfg["heldout"]:
            if (bank, item["family"]) in completed:
                print(f"checkpoint already contains {bank} {item['family']}", flush=True)
                continue
            common_id = first_id(tok, item["common_term"])
            formal_id = first_id(tok, item["formal_term"])
            contexts = {name: item[f"{name}_context"] for name in ("neutral", "source", "target")}
            prompt_specs = []
            for context_name, context in contexts.items():
                cprompt = code_prompt(prefix, context)
                prompt_specs.append(("code", context_name, None, cprompt))
                for code in ("A", "B"):
                    prompt_specs.append(("term", context_name, code, term_prompt(prefix, context, code)))
            logits_list = batch_logits(model, tok, [x[3] for x in prompt_specs], batch_size=args.batch_size)
            code_results = {}
            term_results = {name: {} for name in contexts}
            for (kind, context_name, code, prompt_text), logits in zip(prompt_specs, logits_list):
                if kind == "code":
                    code_logits = logits
                    code_probs = torch.softmax(code_logits, -1)
                    top = int(code_logits.argmax())
                    code_results[context_name] = {
                        "prompt": prompt_text,
                        "p_A": float(code_probs[code_ids["A"]]),
                        "p_B": float(code_probs[code_ids["B"]]),
                        "ab_mass": float(code_probs[list(code_ids.values())].sum()),
                        "p_B_minus_A": float(code_probs[code_ids["B"]] - code_probs[code_ids["A"]]),
                        "top_token": tok.decode([top]).strip(),
                        "top_is_A_or_B": top in code_ids.values(),
                    }
                else:
                    term_results[context_name][str(code)] = {
                        "prompt": prompt_text,
                        **lexical_stats(logits, common_id, formal_id, tok),
                    }
            neutral_A = term_results["neutral"]["A"]["formal_minus_common_probability"]
            neutral_B = term_results["neutral"]["B"]["formal_minus_common_probability"]
            fixed_context_effects = {
                code: term_results["target"][code]["formal_minus_common_probability"]
                - term_results["source"][code]["formal_minus_common_probability"]
                for code in ("A", "B")
            }
            rows.append({
                "bank": bank,
                "family": item["family"],
                "common_term": item["common_term"],
                "formal_term": item["formal_term"],
                "code": code_results,
                "term": term_results,
                "neutral_forced_code_branch_leverage": neutral_B - neutral_A,
                "fixed_code_context_effects": fixed_context_effects,
                "mean_fixed_code_context_effect": sum(fixed_context_effects.values()) / 2,
            })
            atomic_write_json(rows_path, rows)
            atomic_write_json(results / "progress.json", {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "completed_cells": len(rows),
                "total_cells": 15,
                "last_completed": {"bank": bank, "family": item["family"]},
            })
            print(f"assayed and checkpointed {bank} {item['family']} ({len(rows)}/15)", flush=True)

    summary = {"experiment": cfg["experiment_name"], "generated_at": datetime.now(timezone.utc).isoformat(),
               "model": cfg["model"], "code_support_threshold": cfg["code_support_threshold"], "banks": {}}
    for bank in ("high", "medium", "low"):
        group = [r for r in rows if r["bank"] == bank]
        code_cells = [r["code"][context] for r in group for context in ("neutral", "source", "target")]
        values = lambda key: [float(r[key]) for r in group]
        summary["banks"][bank] = {
            "families": len(group),
            "code_cells": len(code_cells),
            "ab_mass_mean": sum(x["ab_mass"] for x in code_cells) / len(code_cells),
            "ab_mass_min": min(x["ab_mass"] for x in code_cells),
            "ab_mass_threshold_pass_rate": sum(x["ab_mass"] >= cfg["code_support_threshold"] for x in code_cells) / len(code_cells),
            "code_top1_valid_rate": sum(x["top_is_A_or_B"] for x in code_cells) / len(code_cells),
            "neutral_branch_leverage_mean": sum(values("neutral_forced_code_branch_leverage")) / len(group),
            "neutral_branch_leverage_by_family": {r["family"]: r["neutral_forced_code_branch_leverage"] for r in group},
            "mean_fixed_code_context_effect": sum(values("mean_fixed_code_context_effect")) / len(group),
            "fixed_code_context_effect_by_family": {r["family"]: r["mean_fixed_code_context_effect"] for r in group},
            "neutral_term_top1_pair_rate": sum(r["term"]["neutral"][code]["top_is_pair_member"] for r in group for code in ("A", "B")) / (2 * len(group)),
            "neutral_term_pair_mass_mean": sum(r["term"]["neutral"][code]["pair_probability_mass"] for r in group for code in ("A", "B")) / (2 * len(group)),
        }
    summary["preflight_contrasts"] = {
        "high_minus_low_branch_leverage": summary["banks"]["high"]["neutral_branch_leverage_mean"] - summary["banks"]["low"]["neutral_branch_leverage_mean"],
        "medium_minus_low_branch_leverage": summary["banks"]["medium"]["neutral_branch_leverage_mean"] - summary["banks"]["low"]["neutral_branch_leverage_mean"],
        "context_effect_survives_low_fixed_code": summary["banks"]["low"]["mean_fixed_code_context_effect"],
    }
    atomic_write_json(rows_path, rows)
    atomic_write_json(results / "summary.json", summary)
    atomic_write_json(results / "progress.json", {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_cells": len(rows),
        "total_cells": 15,
        "status": "complete",
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
