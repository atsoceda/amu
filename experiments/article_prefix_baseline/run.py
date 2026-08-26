#!/usr/bin/env python3
"""Intervention-off article-prefix contrast: inserted `a` versus inserted `an`."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.lib.core import setup_file_logging, token_id_for_text
from experiments.lib.mediation_estimands import total_variation_from_logits


EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def percentile_interval(
    values: list[float],
    rng: random.Random,
    n_resamples: int,
) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_index = max(0, math.floor(0.025 * (len(means) - 1)))
    hi_index = min(len(means) - 1, math.ceil(0.975 * (len(means) - 1)))
    return {
        "n": n,
        "mean": sum(values) / n,
        "lo": means[lo_index],
        "hi": means[hi_index],
        "method": "prompt-level nonparametric bootstrap",
        "resamples": n_resamples,
    }


@torch.inference_mode()
def noun_logits(model, tokenizer, device: torch.device, prefix: str) -> torch.Tensor:
    encoded = tokenizer(prefix, return_tensors="pt").to(device)
    logits = model(**encoded).logits[0, -1].detach().float().cpu()
    return logits


def top1(tokenizer, logits: torch.Tensor) -> str:
    return tokenizer.decode([int(torch.argmax(logits).item())]).strip().lower()


def main() -> None:
    config = load_json(CONFIG_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR)
    started = time.time()

    dataset_path = (EXP_DIR / config["dataset_path"]).resolve()
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if digest != config["dataset_sha256"]:
        raise ValueError(f"Dataset checksum mismatch: {digest}")

    e1 = load_json((EXP_DIR / config["e1_config_path"]).resolve())
    heldout = [
        {
            "split": "heldout_e1",
            "sentence": example["sentence"],
            "listed_word": example.get("listed_word", ""),
            "gold_article": example.get("expected_article", ""),
        }
        for example in e1["test_examples"]
    ]
    with dataset_path.open(newline="") as handle:
        released = list(csv.DictReader(handle))
    an_targets = [
        {
            "split": "released_an_targets",
            "sentence": row["sentence"],
            "listed_word": row.get("word", row.get("listed_word", "")),
            "gold_article": "an",
        }
        for row in released
        if row.get("article", row.get("gold_article", "")).strip().lower() == "an"
    ]
    # Avoid duplicating held-out sentences inside the released an-target split.
    heldout_sentences = {item["sentence"] for item in heldout}
    an_targets = [
        item for item in an_targets if item["sentence"] not in heldout_sentences
    ]
    examples = heldout + an_targets

    model_ref = (
        config["model_snapshot"]
        if Path(config["model_snapshot"]).exists()
        else config["model"]
    )
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=getattr(torch, config["dtype"]),
    ).to(device)
    model.eval()

    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    rows = []
    for index, example in enumerate(examples, start=1):
        prompt = f"{config['demonstration']} {example['sentence']}"
        prefix_a = prompt + tokenizer.decode([a_id])
        prefix_an = prompt + tokenizer.decode([an_id])
        logits_a = noun_logits(model, tokenizer, device, prefix_a)
        logits_an = noun_logits(model, tokenizer, device, prefix_an)
        tv = total_variation_from_logits(logits_an, logits_a)
        row = {
            "index": index,
            **example,
            "top1_after_a": top1(tokenizer, logits_a),
            "top1_after_an": top1(tokenizer, logits_an),
            "top1_changed": top1(tokenizer, logits_a) != top1(tokenizer, logits_an),
            "tv_full_vocab": tv,
        }
        rows.append(row)
        if index % 20 == 0 or index == len(examples):
            print(f"{index}/{len(examples)} tv={tv:.3f}", flush=True)

    rng = random.Random(int(config["bootstrap_seed"]))
    resamples = int(config["bootstrap_resamples"])
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append(row)

    def summarize(split_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(split_rows),
            "top1_changed_rate": sum(row["top1_changed"] for row in split_rows)
            / max(len(split_rows), 1),
            "tv_full_vocab": percentile_interval(
                [float(row["tv_full_vocab"]) for row in split_rows],
                rng,
                resamples,
            ),
        }

    summary = {
        "experiment": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - started,
        "model": config["model"],
        "n": len(rows),
        "splits": {name: summarize(items) for name, items in by_split.items()},
        "all": summarize(rows),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Article-prefix baseline",
        "",
        f"Generated: {summary['generated_at']}",
        f"Runtime: {summary['elapsed_sec']:.1f}s",
        "",
        "Intervention off. Compare the noun-token distribution after inserted `a`",
        "versus inserted `an`.",
        "",
        "| Split | N | Top-1 noun changes | Mean TV [95% bootstrap] |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, block in summary["splits"].items():
        tv = block["tv_full_vocab"]
        lines.append(
            f"| {name} | {block['n']} | {block['top1_changed_rate']:.2f} | "
            f"{tv['mean']:.3f} [{tv['lo']:.3f}, {tv['hi']:.3f}] |"
        )
    tv = summary["all"]["tv_full_vocab"]
    lines.append(
        f"| all | {summary['all']['n']} | {summary['all']['top1_changed_rate']:.2f} | "
        f"{tv['mean']:.3f} [{tv['lo']:.3f}, {tv['hi']:.3f}] |"
    )
    (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Done in {summary['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
