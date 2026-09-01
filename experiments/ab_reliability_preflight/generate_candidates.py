#!/usr/bin/env python3
"""Generate and cache lexical-register candidates with Gemini."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text())
RAW = EXP / "artifacts/00_raw_generation"
PARSED = EXP / "artifacts/01_candidates"


def extract_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("response contains no JSON array")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("response is not an array")
    return value


def generation_prompt(batch: int) -> str:
    count = CFG["generation"]["families_per_batch"]
    return f"""Generate exactly {count} diverse English lexical-register pairs for a controlled language-model experiment.

Return only a JSON array. Every object must have exactly these fields:
- family_id: unique snake_case semantic-family name
- everyday_term: a common everyday single-word noun
- formal_term: a more formal, professional, official, or technical single-word noun with substantially the same denotation
- shared_meaning: a short definition that is equally true of both terms and contains neither term
- distinction_note: a short description of the register difference

Hard constraints:
1. Both terms must be singular nouns and close substitutes for the same referent in the supplied definition.
2. Prefer frequent words likely to be one tokenizer token when preceded by a space.
3. Exclude spelling variants, proper nouns, hyphenated or multiword expressions, abbreviations, offensive language, and pairs whose main distinction is semantic specificity rather than register.
4. Include diverse domains: people, occupations, objects, transport, places, institutions, events, documents, communication, medicine, law, technology, commerce, and ordinary life.
5. Do not include A/B labels, prompt templates, experimental outcomes, or duplicates.
6. Human intuitions are only candidate generation; Gemma will independently determine eligibility.

Batch index: {batch}. JSON only."""


def call(prompt: str, seed: int, key: str) -> tuple[dict[str, Any], str]:
    model = CFG["generation"]["model"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": CFG["generation"]["temperature"],
            "seed": seed,
            "responseMimeType": "application/json",
        },
    }
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = json.loads(response.read())
    text = raw["candidates"][0]["content"]["parts"][0]["text"]
    return raw, text


def normalize(batch: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for index, row in enumerate(rows):
        out.append({
            "candidate_id": f"b{batch:02d}_{index:02d}_{row.get('family_id', 'unnamed')}",
            "generation_batch": batch,
            "family_id": str(row.get("family_id", "")).strip().lower(),
            "everyday_term": str(row.get("everyday_term", "")).strip().lower(),
            "formal_term": str(row.get("formal_term", "")).strip().lower(),
            "shared_meaning": str(row.get("shared_meaning", "")).strip(),
            "distinction_note": str(row.get("distinction_note", "")).strip(),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=CFG["generation"]["batches"])
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()
    key = os.environ.get("GOOGLE_GEMINI_API_KEY")
    if not key:
        raise SystemExit("GOOGLE_GEMINI_API_KEY is not present; run from a login shell")
    RAW.mkdir(parents=True, exist_ok=True)
    PARSED.mkdir(parents=True, exist_ok=True)
    for batch in range(args.start, min(args.stop, CFG["generation"]["batches"])):
        parsed_path = PARSED / f"batch_{batch:02d}.json"
        if parsed_path.exists():
            print(f"batch {batch:02d}: cached", flush=True)
            continue
        prompt = generation_prompt(batch)
        seed = CFG["generation"]["seed_base"] + batch
        for attempt in range(args.max_retries):
            try:
                raw, text = call(prompt, seed, key)
                rows = normalize(batch, extract_array(text))
                envelope = {"provider": CFG["generation"]["provider"], "model": CFG["generation"]["model"],
                            "generated_at": datetime.now(timezone.utc).isoformat(), "batch": batch,
                            "seed": seed, "prompt": prompt, "response": raw}
                raw_tmp = RAW / f"batch_{batch:02d}.json.tmp"
                parsed_tmp = parsed_path.with_suffix(".json.tmp")
                raw_tmp.write_text(json.dumps(envelope, indent=2) + "\n")
                parsed_tmp.write_text(json.dumps(rows, indent=2) + "\n")
                raw_tmp.replace(RAW / f"batch_{batch:02d}.json")
                parsed_tmp.replace(parsed_path)
                print(f"batch {batch:02d}: {len(rows)} candidates", flush=True)
                break
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
                if attempt + 1 == args.max_retries:
                    raise
                delay = min(60, 2 ** attempt + random.random())
                print(f"batch {batch:02d}: retry after {type(exc).__name__}", flush=True)
                time.sleep(delay)
    paths = sorted(PARSED.glob("batch_*.json"))
    rows = [row for path in paths for row in json.loads(path.read_text())]
    (PARSED / "all_candidates.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"assembled {len(rows)} candidates from {len(paths)} batches")


if __name__ == "__main__":
    main()
