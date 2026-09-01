#!/usr/bin/env python3
"""Generate and cache a large noun-pair/modifier candidate bank with Gemini."""

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
PARSED = EXP / "artifacts/01_parsed_candidates"


def extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("response contains no JSON array")
    return json.loads(text[start : end + 1])


def prompt_for(batch: int, domain: str) -> str:
    n = CFG["generation"]["pairs_per_batch"]
    m = CFG["generation"]["modifiers_per_pair"]
    return f"""Generate exactly {n} diverse English near-synonym NOUN pairs in the domain: {domain}.

This is candidate generation for a mechanistic language-model experiment. Return only a JSON array. Each object must have exactly:
- family_id: unique snake_case identifier
- noun_0, noun_1: common singular count nouns with genuinely close denotation, preferably common enough to be single tokenizer tokens
- shared_definition: one short definition valid for both nouns
- neutral_stem: a natural sentence stem ending immediately before the fixed text \"the\"; it must invite an adjective then one of the nouns, without naming either noun, any modifier, an initial, or a synonym
- candidate_modifiers: exactly {m} common one-word attributive adjectives that can naturally modify BOTH nouns
- semantic_notes: one short note on any register or collocation difference

Hard constraints:
1. noun_0 and noun_1 must denote substantially the same kind of entity, not merely related entities.
2. Every modifier must be grammatically usable before both nouns.
3. Include modifiers expected to span low, moderate, and high differences in lexical collocation between the nouns; do not select only maximally discriminating modifiers.
4. Avoid proper nouns, compounds containing spaces/hyphens, plurals, offensive terms, and pairs differing only in spelling.
5. Do not generate contexts yet.
6. Do not repeat obvious examples excessively across the batch.

Batch index: {batch}. JSON only."""


def call_gemini(prompt: str, seed: int, key: str) -> tuple[dict[str, Any], str]:
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


def normalize(batch: int, domain: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = CFG["generation"]["modifiers_per_pair"]
    normalized = []
    for index, row in enumerate(rows):
        modifiers = row.get("candidate_modifiers", [])
        item = {
            "candidate_id": f"b{batch:02d}_{index:02d}_{row.get('family_id', 'unnamed')}",
            "generation_batch": batch,
            "domain": domain,
            "family_id": str(row.get("family_id", "")),
            "noun_0": str(row.get("noun_0", "")).strip().lower(),
            "noun_1": str(row.get("noun_1", "")).strip().lower(),
            "shared_definition": str(row.get("shared_definition", "")).strip(),
            "neutral_stem": str(row.get("neutral_stem", "")).strip(),
            "candidate_modifiers": [str(x).strip().lower() for x in modifiers],
            "semantic_notes": str(row.get("semantic_notes", "")).strip(),
            "generation_format_valid": len(modifiers) == expected,
        }
        normalized.append(item)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=CFG["generation"]["batches"])
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()
    key = os.environ.get("GOOGLE_GEMINI_API_KEY")
    if not key:
        raise SystemExit("GOOGLE_GEMINI_API_KEY is not present; run from a login shell")
    RAW.mkdir(parents=True, exist_ok=True); PARSED.mkdir(parents=True, exist_ok=True)
    domains = CFG["candidate_domains"]
    for batch in range(args.start, min(args.stop, CFG["generation"]["batches"])):
        out = PARSED / f"batch_{batch:02d}.json"
        if out.exists():
            print(f"batch {batch:02d}: cached", flush=True); continue
        domain = domains[batch % len(domains)]
        prompt = prompt_for(batch, domain)
        seed = CFG["generation"]["seed_base"] + batch
        for attempt in range(args.max_retries):
            try:
                raw, text = call_gemini(prompt, seed, key)
                rows = extract_json(text)
                normalized = normalize(batch, domain, rows)
                envelope = {
                    "provider": CFG["generation"]["provider"],
                    "model": CFG["generation"]["model"],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "batch": batch,
                    "domain": domain,
                    "seed": seed,
                    "prompt": prompt,
                    "response": raw,
                }
                tmp_raw = RAW / f"batch_{batch:02d}.json.tmp"
                tmp_out = out.with_suffix(".json.tmp")
                tmp_raw.write_text(json.dumps(envelope, indent=2) + "\n")
                tmp_out.write_text(json.dumps(normalized, indent=2) + "\n")
                tmp_raw.replace(RAW / f"batch_{batch:02d}.json")
                tmp_out.replace(out)
                print(f"batch {batch:02d}: {len(normalized)} pairs", flush=True)
                break
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
                if attempt + 1 == args.max_retries:
                    raise
                delay = min(60, 2 ** attempt + random.random())
                print(f"batch {batch:02d}: retry {attempt + 1} after {type(exc).__name__}", flush=True)
                time.sleep(delay)

    batches = sorted(PARSED.glob("batch_*.json"))
    all_rows = [row for path in batches for row in json.loads(path.read_text())]
    (PARSED / "all_candidates.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    print(f"assembled {len(all_rows)} candidates from {len(batches)} cached batches")


if __name__ == "__main__":
    main()
