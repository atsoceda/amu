#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.lib.core import setup_file_logging, token_id_for_text

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def first_word(text: str) -> str:
    match = re.search(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text)
    return match.group(0).lower() if match else ""


def word_after_article(text: str) -> str:
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text.lower())
    if words and words[0] in {"a", "an"}:
        return words[1] if len(words) > 1 else ""
    return ""


def generate_suffixes(
    model,
    tokenizer,
    device: torch.device,
    prefixes: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    for start in range(0, len(prefixes), batch_size):
        batch = prefixes[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        input_width = encoded.input_ids.shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        for sequence in generated:
            outputs.append(tokenizer.decode(sequence[input_width:], skip_special_tokens=True))
    return outputs


def next_article(model, tokenizer, device: torch.device, prompt: str) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = model(**encoded).logits[0, -1].detach().float().cpu()
    probs = torch.softmax(logits, dim=-1)
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    top_id = int(logits.argmax())
    top_text = tokenizer.decode([top_id])
    article = top_text.strip().lower()
    if article not in {"a", "an"}:
        article = "other"
    return {
        "predicted_article": article,
        "predicted_token": top_text,
        "prob_a": float(probs[a_id]),
        "prob_an": float(probs[an_id]),
    }


def continuation_logprob(
    model,
    tokenizer,
    device: torch.device,
    prefix: str,
    continuation: str,
) -> dict[str, Any]:
    prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
    continuation_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prefix_ids + continuation_ids], device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logprobs = []
    for offset, token_id in enumerate(continuation_ids):
        prediction_pos = len(prefix_ids) - 1 + offset
        token_logprobs.append(float(log_probs[prediction_pos, token_id].detach().cpu()))
    return {
        "token_ids": continuation_ids,
        "token_count": len(continuation_ids),
        "sum_logprob": sum(token_logprobs),
        "mean_logprob": sum(token_logprobs) / len(token_logprobs),
        "first_token_logprob": token_logprobs[0],
    }


def build_rows(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    vowel_examples = [example for example in baseline["examples"] if example["article"] == "an"]
    for example in vowel_examples:
        for context in baseline["in_context_examples"]:
            prompt = f"{context['text']} {example['prompt']}"
            rows.append(
                {
                    "example_id": example["id"],
                    "planned_word": example["planned_word"],
                    "target_prompt": example["prompt"],
                    "in_context_id": context["id"],
                    "in_context_article": context["article"],
                    "demonstration_text": context["text"],
                    "prompt": prompt,
                }
            )
    return rows


def classify(row: dict[str, Any]) -> str:
    if row["predicted_article"] != "a":
        return "article_not_a"
    if row["natural_word_after_article"] == row["planned_word"]:
        return "a_plus_planned_word_mismatch"
    if row["natural_word_after_article"].startswith(("a", "e", "i", "o", "u")):
        return "a_plus_other_vowel_word_mismatch"
    if row["natural_first_word"]:
        return "a_plus_consonant_word"
    return "a_plus_no_lexical_word"


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    mismatch_rows = [
        row
        for row in rows
        if row["classification"]
        in {"a_plus_planned_word_mismatch", "a_plus_other_vowel_word_mismatch"}
    ]
    lines = [
        "# A/An Continuation Check Report",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        "",
        "## Question",
        "",
        "When Gemma incorrectly selects `a` for a prompt whose listed target requires `an`, does it continue with that listed target anyway, as in `a accountant`, or choose a different continuation?",
        "",
        "## Demonstrations",
        "",
    ]
    for context in summary["demonstrations"]:
        lines.append(f"- `{context['article']}` demonstration: `{context['text']}`")
    lines += [
        "",
        "## Short Answer",
        "",
        summary["interpretation"],
        "",
        f"- Evaluations: {summary['n_rows']}",
        f"- Next-token `a` predictions: {summary['predicted_a_count']}",
        f"- Exact `a` + listed-word mismatches: {summary['exact_mismatch_count']}",
        f"- `a` + other vowel-initial word mismatches: {summary['other_vowel_mismatch_count']}",
        f"- `a` followed by a consonant-initial word: {summary['a_consonant_word_count']}",
        "",
        "An exact mismatch means the first generated lexical word after `a` equals the listed planned word. The broader mismatch count also includes a different vowel-initial answer, such as `a ophthalmologist`. This is a behavioral test only; the generated word has not yet been established as an internal plan.",
        "",
        "## Mismatch Cases",
        "",
        "These are the rows counted as preparation/content mismatches:",
        "",
        "| Target Prompt | Demonstration | Natural Continuation | Why It Counts |",
        "| --- | --- | --- | --- |",
    ]
    for row in mismatch_rows:
        natural = row["natural_suffix"].replace("\n", "\\n").strip()
        reason = (
            "The model selected `a`, then generated the listed vowel-initial word."
            if row["classification"] == "a_plus_planned_word_mismatch"
            else "The model selected `a`, then generated a different vowel-initial answer."
        )
        lines.append(
            f"| `{row['target_prompt']}` | `{row['in_context_article']}` | "
            f"`{natural}` | {reason} |"
        )
    lines += [
        "",
        f"The {len(mismatch_rows)} rows come from {len({row['example_id'] for row in mismatch_rows})} unique target prompt(s); the same target was evaluated under both demonstrations.",
        "",
        "## Every Continuation",
        "",
        "| Target Prompt | Listed Word | Demo | Predicted Article | Natural Continuation | After Forced `a` | After Forced `an` | Listed Word LogP After `a` | After `an` | Classification |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        natural = row["natural_suffix"].replace("\n", "\\n").strip()
        forced_a = row["forced_a_suffix"].replace("\n", "\\n").strip()
        forced_an = row["forced_an_suffix"].replace("\n", "\\n").strip()
        lines.append(
            f"| `{row['target_prompt']}` | `{row['planned_word']}` | "
            f"`{row['in_context_article']}` | `{row['predicted_article']}` | "
            f"`{natural}` | `{forced_a}` | `{forced_an}` | "
            f"{row['planned_after_a']['sum_logprob']:.3f} | "
            f"{row['planned_after_an']['sum_logprob']:.3f} | "
            f"`{row['classification']}` |"
        )
    lines += [
        "",
        "## Interpretation Boundary",
        "",
        "If Gemma selects `a` and then generates the listed vowel-initial word, that is the preparation/content mismatch needed to motivate a mechanistic planning test. If it selects a grammatically compatible alternative, the article may be shaping the later answer rather than failing to prepare for an existing plan.",
        "",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    config = load_json(CONFIG_PATH)
    baseline_path = (EXP_DIR / config["baseline_config"]).resolve()
    baseline = load_json(baseline_path)
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
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    rows = build_rows(baseline)
    started = time.time()
    for row in rows:
        row.update(next_article(model, tokenizer, device, row["prompt"]))

    natural_suffixes = generate_suffixes(
        model,
        tokenizer,
        device,
        [row["prompt"] for row in rows],
        int(config["batch_size"]),
        int(config["max_new_tokens_from_prompt"]),
    )
    forced_a_suffixes = generate_suffixes(
        model,
        tokenizer,
        device,
        [f"{row['prompt']} a" for row in rows],
        int(config["batch_size"]),
        int(config["max_new_tokens_after_article"]),
    )
    forced_an_suffixes = generate_suffixes(
        model,
        tokenizer,
        device,
        [f"{row['prompt']} an" for row in rows],
        int(config["batch_size"]),
        int(config["max_new_tokens_after_article"]),
    )

    for row, natural, forced_a, forced_an in zip(
        rows, natural_suffixes, forced_a_suffixes, forced_an_suffixes
    ):
        row["natural_suffix"] = natural
        row["natural_first_word"] = first_word(natural)
        row["natural_word_after_article"] = word_after_article(natural)
        row["forced_a_suffix"] = forced_a
        row["forced_a_first_word"] = first_word(forced_a)
        row["forced_an_suffix"] = forced_an
        row["forced_an_first_word"] = first_word(forced_an)
        planned_continuation = f" {row['planned_word']}"
        row["planned_after_a"] = continuation_logprob(
            model, tokenizer, device, f"{row['prompt']} a", planned_continuation
        )
        row["planned_after_an"] = continuation_logprob(
            model, tokenizer, device, f"{row['prompt']} an", planned_continuation
        )
        row["classification"] = classify(row)

    mismatch_count = sum(
        row["classification"] == "a_plus_planned_word_mismatch" for row in rows
    )
    other_vowel_mismatch_count = sum(
        row["classification"] == "a_plus_other_vowel_word_mismatch" for row in rows
    )
    a_consonant_count = sum(
        row["classification"] == "a_plus_consonant_word" for row in rows
    )
    total_mismatch_count = mismatch_count + other_vowel_mismatch_count
    if total_mismatch_count:
        interpretation = (
            f"Gemma produced {total_mismatch_count} preparation/content mismatch(es), where `a` "
            "was immediately followed by a vowel-initial answer. These cases justify a targeted "
            "mechanistic follow-up."
        )
    else:
        interpretation = (
            "Gemma produced no cases where `a` was followed by a vowel-initial answer. Its strong "
            "preference for `a` does not, by itself, provide the behavioral planning failure we sought."
        )

    summary = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "model_ref": model_ref,
        "device": str(device),
        "runtime_seconds": time.time() - started,
        "n_rows": len(rows),
        "n_unique_examples": len({row["example_id"] for row in rows}),
        "predicted_a_count": sum(row["predicted_article"] == "a" for row in rows),
        "predicted_an_count": sum(row["predicted_article"] == "an" for row in rows),
        "exact_mismatch_count": mismatch_count,
        "other_vowel_mismatch_count": other_vowel_mismatch_count,
        "total_mismatch_count": total_mismatch_count,
        "a_consonant_word_count": a_consonant_count,
        "class_counts": {
            name: sum(row["classification"] == name for row in rows)
            for name in sorted({row["classification"] for row in rows})
        },
        "demonstrations": baseline["in_context_examples"],
        "interpretation": interpretation,
    }
    with (RESULTS_DIR / "examples.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary, rows)
    logging.info(
        "preparation/content mismatches %d/%d; a plus consonant word %d/%d",
        total_mismatch_count,
        len(rows),
        a_consonant_count,
        len(rows),
    )


if __name__ == "__main__":
    main()
