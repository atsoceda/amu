#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.lib.core import setup_file_logging

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def ensure_source_data(config: dict[str, Any]) -> Path:
    source = config["source"]
    target = DATA_DIR / Path(source["path"]).name
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        logging.info("Downloading pinned paper dataset from %s", source["url"])
        urllib.request.urlretrieve(source["url"], target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        raise ValueError(f"Dataset checksum mismatch: expected {source['sha256']}, got {digest}")
    return target


def parse_verb_and_number(text: str) -> tuple[str, int | None]:
    tokens = re.findall(r"[A-Za-z]+|\d+", text.lower())
    verb = tokens[0] if tokens and tokens[0] in {"is", "are"} else "other"
    number = None
    for token in tokens[1:] if verb != "other" else tokens:
        if token.isdigit():
            number = int(token)
            break
        if token in NUMBER_WORDS:
            number = NUMBER_WORDS[token]
            break
    return verb, number


def generate_rows(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
) -> None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [row["prompt"] for row in batch],
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        input_width = encoded.input_ids.shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        for index, row in enumerate(batch):
            suffix = tokenizer.decode(generated[index, input_width:], skip_special_tokens=True)
            verb, number = parse_verb_and_number(suffix)
            row["natural_suffix"] = suffix
            row["predicted_verb"] = verb
            row["generated_number"] = number
            row["verb_correct"] = verb == row["gold_verb"]
            row["number_correct"] = number == row["gold_number"]
            row["are_one_mismatch"] = (
                row["gold_verb"] == "is"
                and verb == "are"
                and number == 1
            )


def build_rows(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    return [
        {
            "source_index": index,
            "prompt": row["prompt"].strip(),
            "animal": row["animal"].strip(),
            "original": int(row["original"]),
            "subtracted": int(row["subtracted"]),
            "gold_number": int(row["number"]),
            "gold_verb": row["answer"].strip(),
        }
        for index, row in enumerate(source_rows)
    ]


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    mismatches = [row for row in rows if row["are_one_mismatch"]]
    lines = [
        "# Is/Are Behavioral Screen",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        "",
        "## Question",
        "",
        "On the paper's released subtraction task, does Gemma 3 270M produce the specific preparation/content mismatch `are 1`?",
        "",
        "## Source",
        "",
        f"- Repository: `{summary['source']['repository']}`",
        f"- Commit: `{summary['source']['commit']}`",
        f"- Dataset: `{summary['source']['path']}`",
        f"- SHA-256: `{summary['source']['sha256']}`",
        "",
        "## Short Answer",
        "",
        summary["interpretation"],
        "",
        f"- Prompts screened: {summary['n_rows']}",
        f"- Singular-answer prompts: {summary['singular_rows']}",
        f"- Exact `are 1` mismatches: {summary['are_one_mismatch_count']}",
        f"- Unique animals with an `are 1` mismatch: {summary['mismatch_animals']}",
        f"- Overall number accuracy: {summary['number_accuracy']:.1%}",
        f"- Overall verb accuracy: {summary['verb_accuracy']:.1%}",
        "",
        "## Exact `are 1` Mismatches",
        "",
        "| Prompt | Natural Continuation |",
        "| --- | --- |",
    ]
    for row in mismatches:
        suffix = row["natural_suffix"].replace("\n", "\\n").strip()
        lines.append(f"| `{row['prompt']}` | `{suffix}` |")
    lines += [
        "",
        "## Outcome Counts",
        "",
        "| Gold Verb | Predicted Verb | Number Correct | Count |",
        "| --- | --- | --- | ---: |",
    ]
    for key, count in summary["outcome_counts"].items():
        gold, predicted, number_correct = key.split("|")
        lines.append(f"| `{gold}` | `{predicted}` | `{number_correct}` | {count} |")
    lines += [
        "",
        "Full prompt-level results are in `results/examples.jsonl`.",
        "",
        "## Interpretation Boundary",
        "",
        "An `are 1` continuation establishes a behavioral mismatch. It does not prove that a representation of `1` was active before the verb or causally supported the later number.",
        "",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    config = load_config()
    setup_file_logging(RESULTS_DIR)
    dataset_path = ensure_source_data(config)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    device = torch.device(config["device"])
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

    rows = build_rows(dataset_path)
    started = time.time()
    generate_rows(
        model,
        tokenizer,
        rows,
        int(config["batch_size"]),
        int(config["max_new_tokens"]),
    )
    mismatches = [row for row in rows if row["are_one_mismatch"]]
    outcome_counts = Counter(
        f"{row['gold_verb']}|{row['predicted_verb']}|{str(row['number_correct']).lower()}"
        for row in rows
    )
    if len(mismatches) >= 10:
        interpretation = (
            "Gemma produces enough `are 1` mismatches to support a broad mechanistic study of "
            "whether the future answer is represented before the incorrect verb."
        )
    elif mismatches:
        interpretation = (
            "Gemma produces some `are 1` mismatches, but fewer than the ten-example threshold "
            "for a broad mechanistic study."
        )
    else:
        interpretation = (
            "Gemma produced no `are 1` mismatches, so this task does not provide the required "
            "behavioral planning failures."
        )
    summary = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "model_ref": model_ref,
        "device": str(device),
        "runtime_seconds": time.time() - started,
        "source": config["source"],
        "n_rows": len(rows),
        "singular_rows": sum(row["gold_verb"] == "is" for row in rows),
        "are_one_mismatch_count": len(mismatches),
        "mismatch_animals": len({row["animal"] for row in mismatches}),
        "number_accuracy": sum(row["number_correct"] for row in rows) / len(rows),
        "verb_accuracy": sum(row["verb_correct"] for row in rows) / len(rows),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "interpretation": interpretation,
    }
    with (RESULTS_DIR / "examples.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary, rows)
    logging.info("found %d exact are-1 mismatches", len(mismatches))


if __name__ == "__main__":
    main()
