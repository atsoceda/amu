#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.lib.core import setup_file_logging, token_id_for_text

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rank_for_token(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item() + 1)


def build_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for example in config["examples"]:
        for context in config["in_context_examples"]:
            rows.append(
                {
                    "example_id": example["id"],
                    "expected_article": example["article"],
                    "planned_word": example["planned_word"],
                    "target_prompt": example["prompt"],
                    "in_context_id": context["id"],
                    "in_context_article": context["article"],
                    "prompt": f"{context['text']} {example['prompt']}",
                }
            )
    return rows


def score_rows(
    model,
    tokenizer,
    device: torch.device,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> None:
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [row["prompt"] for row in batch],
            return_tensors="pt",
            padding=True,
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits[:, -1, :].detach().float().cpu()
        probs = torch.softmax(logits, dim=-1)
        top_ids = logits.argmax(dim=-1)

        for index, row in enumerate(batch):
            row_logits = logits[index]
            row_probs = probs[index]
            top_id = int(top_ids[index])
            predicted_token = tokenizer.decode([top_id])
            predicted_article = predicted_token.strip().lower()
            if predicted_article not in {"a", "an"}:
                predicted_article = "other"
            row.update(
                {
                    "predicted_token": predicted_token,
                    "predicted_article": predicted_article,
                    "article_correct": predicted_article == row["expected_article"],
                    "prob_a": float(row_probs[a_id]),
                    "prob_an": float(row_probs[an_id]),
                    "logit_a": float(row_logits[a_id]),
                    "logit_an": float(row_logits[an_id]),
                    "an_minus_a_logit": float(row_logits[an_id] - row_logits[a_id]),
                    "rank_a": rank_for_token(row_logits, a_id),
                    "rank_an": rank_for_token(row_logits, an_id),
                    "top_token_id": top_id,
                }
            )


def recall(rows: list[dict[str, Any]], article: str) -> float:
    selected = [row for row in rows if row["expected_article"] == article]
    return sum(row["predicted_article"] == article for row in selected) / len(selected)


def grouped_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["expected_article"], row["in_context_article"])].append(row)
    output = []
    for (expected, context), group in sorted(groups.items()):
        output.append(
            {
                "expected_article": expected,
                "in_context_article": context,
                "n": len(group),
                "recall": sum(row["predicted_article"] == expected for row in group) / len(group),
                "mean_prob_a": sum(row["prob_a"] for row in group) / len(group),
                "mean_prob_an": sum(row["prob_an"] for row in group) / len(group),
                "mean_an_minus_a_logit": sum(row["an_minus_a_logit"] for row in group) / len(group),
            }
        )
    return output


def write_report(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    lines = [
        "# A/An Majority Baseline Report",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        "",
        "## Question",
        "",
        "Does Gemma 3 270M reproduce the paper's behavioral result that small Qwen-3 models default to the majority article `a` on prompts whose planned word requires `an`?",
        "",
        "This experiment measures only the immediate next-token article. It does not establish that Gemma internally planned the listed noun, and it does not test an intervention.",
        "",
        "## Demonstrations",
        "",
        "Each target prompt was evaluated twice, once after each completed example:",
        "",
    ]
    for context in config["in_context_examples"]:
        lines += [
            f"- `{context['article']}` demonstration: `{context['text']}`",
        ]
    lines += [
        "",
        "For example, the target `Someone who handles financial records is` was appended after each demonstration, and the next-token probabilities of `a` and `an` were measured.",
        "",
        "## Result",
        "",
        summary["interpretation"],
        "",
        f"- `an` recall: {summary['recall_an']:.1%} ({summary['correct_an']}/{summary['n_an']})",
        f"- `a` recall: {summary['recall_a']:.1%} ({summary['correct_a']}/{summary['n_a']})",
        f"- Other-token predictions: {summary['other_predictions']}/{summary['n_rows']}",
        "",
        "Each target was evaluated after both an `a` demonstration and an `an` demonstration. This exposes whether the one-shot example itself controls the answer.",
        "",
        "## Recall By Demonstration",
        "",
        "| Expected | Demonstration | N | Recall | Mean P(a) | Mean P(an) | Mean an-a Logit |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["grouped_metrics"]:
        lines.append(
            f"| `{item['expected_article']}` | `{item['in_context_article']}` | "
            f"{item['n']} | {item['recall']:.1%} | {item['mean_prob_a']:.4f} | "
            f"{item['mean_prob_an']:.4f} | {item['mean_an_minus_a_logit']:.3f} |"
        )

    lines += [
        "",
        "## Every Prediction",
        "",
        "| Target Prompt | Planned Word | Expected | Demonstration | Predicted | P(a) | P(an) | an-a Logit |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['target_prompt']}` | `{row['planned_word']}` | "
            f"`{row['expected_article']}` | `{row['in_context_article']}` | "
            f"`{row['predicted_article']}` | {row['prob_a']:.4f} | "
            f"{row['prob_an']:.4f} | {row['an_minus_a_logit']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation Boundary",
        "",
        "Low `an` recall would reproduce the paper's majority-class behavioral collapse. It would not by itself show that Gemma would generate an ungrammatical phrase such as `a accountant`, nor that a grammar circuit conceals latent planning. Those require continuation and causal tests.",
        "",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    config = load_config()
    setup_file_logging(RESULTS_DIR)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    device = choose_device()
    dtype = getattr(torch, config["dtype"])
    model_ref = config["model_snapshot"] if Path(config["model_snapshot"]).exists() else config["model"]
    logging.info("Loading %s on %s", model_ref, device)
    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        local_files_only=True,
        dtype=dtype,
    ).to(device)
    model.eval()

    rows = build_rows(config)
    started = time.time()
    score_rows(model, tokenizer, device, rows, int(config["batch_size"]))

    n_an = sum(row["expected_article"] == "an" for row in rows)
    n_a = sum(row["expected_article"] == "a" for row in rows)
    correct_an = sum(
        row["expected_article"] == "an" and row["predicted_article"] == "an" for row in rows
    )
    correct_a = sum(
        row["expected_article"] == "a" and row["predicted_article"] == "a" for row in rows
    )
    recall_an = recall(rows, "an")
    recall_a = recall(rows, "a")
    if recall_an <= 0.1 and recall_a >= 0.8:
        interpretation = (
            "Gemma shows the paper's majority-class collapse pattern: it nearly always predicts "
            "`a` when `an` is expected while retaining high `a` recall."
        )
    elif recall_an < recall_a - 0.2:
        interpretation = (
            "Gemma shows a meaningful minority-class disadvantage, but not a complete collapse "
            "to `a`."
        )
    else:
        interpretation = (
            "Gemma does not show the strong majority-class collapse reported for the smallest "
            "Qwen-3 models on this prompt set."
        )

    summary = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "model_ref": model_ref,
        "device": str(device),
        "runtime_seconds": time.time() - started,
        "n_unique_examples": len(config["examples"]),
        "n_rows": len(rows),
        "n_an": n_an,
        "n_a": n_a,
        "correct_an": correct_an,
        "correct_a": correct_a,
        "recall_an": recall_an,
        "recall_a": recall_a,
        "other_predictions": sum(row["predicted_article"] == "other" for row in rows),
        "grouped_metrics": grouped_metrics(rows),
        "interpretation": interpretation,
    }
    with (RESULTS_DIR / "examples.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary, rows, config)
    logging.info("an recall %.1f%%; a recall %.1f%%", recall_an * 100, recall_a * 100)


if __name__ == "__main__":
    main()
