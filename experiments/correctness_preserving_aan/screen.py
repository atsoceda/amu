#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

import torch

from experiments.lib.aan_protocol import token_id_for_text, write_json
from experiments.gemma_1b_residual_scale.run import ResidualModel, first_id


EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def decode_two_steps(model, prompt: str) -> dict[str, Any]:
    article_logits = model.logits(prompt)
    article_id = int(torch.argmax(article_logits))
    article_piece = model.tokenizer.decode([article_id])
    noun_logits = model.logits(prompt + article_piece)
    noun_id = int(torch.argmax(noun_logits))
    return {
        "article": article_piece.strip(),
        "article_id": article_id,
        "noun": model.tokenizer.decode([noun_id]).strip().lower(),
        "noun_id": noun_id,
        "article_top5": [
            {"token": model.tokenizer.decode([int(i)]).strip(), "logit": float(article_logits[int(i)])}
            for i in torch.topk(article_logits, 5).indices
        ],
        "noun_top10": [
            {"token": model.tokenizer.decode([int(i)]).strip().lower(), "logit": float(noun_logits[int(i)])}
            for i in torch.topk(noun_logits, 10).indices
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gemma_270m", "gemma_1b"), default="gemma_270m")
    args = parser.parse_args()
    config = load(EXP_DIR / "config.json")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    spec = config["models"][args.model]
    model = ResidualModel(spec["model_snapshot"], getattr(torch, config["dtype"]))
    tokenizer = model.tokenizer
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    rows = []
    for pair in config["pairs"]:
        source_prompt = config["source_template"].format(
            definition=pair["definition"], initial=pair["source_word"][0].upper()
        )
        target_prompt = config["target_template"].format(
            definition=pair["definition"], initial=pair["target_word"][0].upper()
        )
        source = decode_two_steps(model, source_prompt)
        target = decode_two_steps(model, target_prompt)
        source_word_id = first_id(tokenizer, pair["source_word"])
        target_word_id = first_id(tokenizer, pair["target_word"])
        intended_source_article_id = a_id if pair["source_article"] == "a" else an_id
        intended_target_article_id = a_id if pair["target_article"] == "a" else an_id
        source_noun_logits = model.logits(source_prompt + tokenizer.decode([intended_source_article_id]))
        target_noun_logits = model.logits(target_prompt + tokenizer.decode([intended_target_article_id]))
        row = {
            **pair,
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "source_generation": source,
            "target_generation": target,
            "source_word_single_token": tokenizer.decode([source_word_id]).strip().lower() == pair["source_word"],
            "target_word_single_token": tokenizer.decode([target_word_id]).strip().lower() == pair["target_word"],
            "source_margin_under_intended_article": float(source_noun_logits[source_word_id] - source_noun_logits[target_word_id]),
            "target_margin_under_intended_article": float(target_noun_logits[target_word_id] - target_noun_logits[source_word_id]),
        }
        row["greedy_admissible"] = bool(
            row["source_word_single_token"]
            and row["target_word_single_token"]
            and source["article"] == pair["source_article"]
            and target["article"] == pair["target_article"]
            and source["noun"] == pair["source_word"]
            and target["noun"] == pair["target_word"]
        )
        row["admissible"] = bool(
            row["source_word_single_token"]
            and row["target_word_single_token"]
            and source["article"] == pair["source_article"]
            and target["article"] == pair["target_article"]
            and row["source_margin_under_intended_article"] > 0
            and row["target_margin_under_intended_article"] > 0
        )
        rows.append(row)
    summary = {
        "experiment": config["experiment_name"],
        "model": spec["model"],
        "screen": "article-support and signed lexical-contrast screen under noun-unnamed first-letter constraints",
        "counts": {
            regime: {
                "candidates": sum(r["regime"] == regime for r in rows),
                "admissible": sum(r["regime"] == regime and r["admissible"] for r in rows),
            }
            for regime in ("between", "within_a", "within_an")
        },
        "admissible_ids": [r["id"] for r in rows if r["admissible"]],
    }
    write_json(RESULTS_DIR / f"screen_rows_{args.model}.json", rows)
    write_json(RESULTS_DIR / f"screen_summary_{args.model}.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
