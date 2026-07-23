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

from experiments.lib.core import setup_file_logging, token_id_for_text

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results"


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


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text.lower())


def word_after_article(text: str) -> str:
    found = words(text)
    if found and found[0] in {"a", "an"} and len(found) > 1:
        return found[1]
    return ""


def generate_rows(
    model,
    tokenizer,
    device: torch.device,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
) -> None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [row["prompt"] for row in batch],
            return_tensors="pt",
            padding=True,
        ).to(device)
        input_width = encoded.input_ids.shape[1]
        with torch.inference_mode():
            prompt_logits = model(**encoded).logits[:, -1, :]
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        article_probs = torch.softmax(prompt_logits[:, [a_id, an_id]].float(), dim=-1).cpu()
        for index, row in enumerate(batch):
            suffix = tokenizer.decode(generated[index, input_width:], skip_special_tokens=True)
            generated_words = words(suffix)
            article = generated_words[0] if generated_words and generated_words[0] in {"a", "an"} else "other"
            answer_word = word_after_article(suffix)
            row.update(
                {
                    "natural_suffix": suffix,
                    "predicted_article": article,
                    "generated_answer_word": answer_word,
                    "prob_a_normalized": float(article_probs[index, 0]),
                    "prob_an_normalized": float(article_probs[index, 1]),
                }
            )
            if article == "a" and answer_word == row["listed_word"]:
                row["classification"] = "a_plus_listed_word_mismatch"
            elif article == "a" and answer_word.startswith(("a", "e", "i", "o", "u")):
                row["classification"] = "a_plus_other_vowel_word_candidate"
            elif article == "a" and answer_word:
                row["classification"] = "a_plus_consonant_word"
            elif article == "an":
                row["classification"] = "article_an"
            else:
                row["classification"] = "other"


def build_rows(dataset_path: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    with dataset_path.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    source_counts = Counter(row["article"].strip() for row in source_rows)
    rows = []
    for source_index, source in enumerate(source_rows):
        if source["article"].strip() != "an":
            continue
        for context in config["in_context_examples"]:
            target_prompt = source["sentence"].strip()
            rows.append(
                {
                    "source_index": source_index,
                    "target_prompt": target_prompt,
                    "listed_word": source["word"].strip().lower(),
                    "expected_article": "an",
                    "in_context_id": context["id"],
                    "in_context_article": context["article"],
                    "demonstration_text": context["text"],
                    "prompt": f"{context['text']} {target_prompt}",
                }
            )
    return rows, dict(source_counts)


def candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["classification"]
        in {"a_plus_listed_word_mismatch", "a_plus_other_vowel_word_candidate"}
    ]


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    candidates = candidate_rows(rows)
    lines = [
        "# Full A/An Dataset Screen",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        "",
        "## Question",
        "",
        "Across every `an`-target prompt released with *Latent Planning Emerges with Scale*, how often does Gemma 3 270M select `a` and then generate a vowel-initial answer?",
        "",
        "## Source",
        "",
        f"- Repository: `{summary['source']['repository']}`",
        f"- Commit: `{summary['source']['commit']}`",
        f"- Dataset: `{summary['source']['path']}`",
        f"- SHA-256: `{summary['source']['sha256']}`",
        f"- Released rows: {summary['source_total_rows']} (`a`: {summary['source_class_counts'].get('a', 0)}, `an`: {summary['source_class_counts'].get('an', 0)})",
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
        f"- Unique `an` targets screened: {summary['unique_targets']}",
        f"- Total evaluations: {summary['n_rows']}",
        f"- `an` article recall: {summary['article_an_recall']:.1%}",
        f"- Candidate mismatch rows: {summary['candidate_row_count']}",
        f"- Unique candidate targets: {summary['unique_candidate_targets']}",
        f"- Candidates reproduced under both demonstrations: {summary['reproduced_candidate_targets']}",
        "",
        "A candidate is an orthographic screen: `a` followed by a word beginning with a vowel letter. Candidate words must be manually checked for pronunciation before mechanistic follow-up.",
        "",
        "## Candidate Mismatches",
        "",
        "| Target Prompt | Listed Word | Demo | Natural Continuation | Classification |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in candidates:
        suffix = row["natural_suffix"].replace("\n", "\\n").strip()
        lines.append(
            f"| `{row['target_prompt']}` | `{row['listed_word']}` | "
            f"`{row['in_context_article']}` | `{suffix}` | `{row['classification']}` |"
        )
    lines += [
        "",
        "## Outcome Counts",
        "",
        "| Classification | Count |",
        "| --- | ---: |",
    ]
    for name, count in summary["class_counts"].items():
        lines.append(f"| `{name}` | {count} |")
    lines += [
        "",
        "Full prompt-level results are stored in `results/examples.jsonl`. The report intentionally foregrounds candidate mismatches rather than presenting a 210-row table.",
        "",
        "## Interpretation Boundary",
        "",
        "This screen identifies behavioral preparation/content mismatches. It does not establish that the generated answer was represented before the article or that a competing article circuit concealed latent planning.",
        "",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    config = load_config()
    setup_file_logging(RESULTS_DIR)
    dataset_path = ensure_source_data(config)
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

    rows, source_counts = build_rows(dataset_path, config)
    started = time.time()
    generate_rows(
        model,
        tokenizer,
        device,
        rows,
        int(config["batch_size"]),
        int(config["max_new_tokens"]),
    )
    candidates = candidate_rows(rows)
    candidate_target_ids = {row["source_index"] for row in candidates}
    candidate_demo_counts = Counter(row["source_index"] for row in candidates)
    class_counts = Counter(row["classification"] for row in rows)
    article_an_count = sum(row["predicted_article"] == "an" for row in rows)
    if len(candidate_target_ids) >= 10:
        interpretation = (
            "The released dataset contains enough distinct behavioral mismatch candidates to "
            "justify paraphrase validation before circuit tracing."
        )
    elif candidate_target_ids:
        interpretation = (
            "The screen found some behavioral mismatch candidates, but fewer than the ten distinct "
            "targets set as the threshold for a broad mechanistic study."
        )
    else:
        interpretation = (
            "The screen found no behavioral mismatch candidates; this task does not currently "
            "support the proposed mechanistic study."
        )
    summary = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "model_ref": model_ref,
        "device": str(device),
        "runtime_seconds": time.time() - started,
        "source": config["source"],
        "source_total_rows": sum(source_counts.values()),
        "source_class_counts": source_counts,
        "demonstrations": config["in_context_examples"],
        "unique_targets": len({row["source_index"] for row in rows}),
        "n_rows": len(rows),
        "article_an_recall": article_an_count / len(rows),
        "candidate_row_count": len(candidates),
        "unique_candidate_targets": len(candidate_target_ids),
        "reproduced_candidate_targets": sum(count == len(config["in_context_examples"]) for count in candidate_demo_counts.values()),
        "class_counts": dict(sorted(class_counts.items())),
        "interpretation": interpretation,
    }
    with (RESULTS_DIR / "examples.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary, rows)
    logging.info(
        "screened %d targets; found %d candidate rows across %d unique targets",
        summary["unique_targets"],
        len(candidates),
        len(candidate_target_ids),
    )


if __name__ == "__main__":
    main()
