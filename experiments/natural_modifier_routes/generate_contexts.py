#!/usr/bin/env python3
"""Generate cached dev/test context paraphrases for frozen modifier constructions."""

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

from generate_candidates import extract_json


EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text())
INPUT = EXP / "artifacts/04_context_candidates/selected_120.json"
RAW = EXP / "artifacts/05_raw_context_generation"
PARSED = EXP / "artifacts/06_generated_contexts"


def prompt_for(family):
    compact = [{"regime": name, **values} for name, values in family["constructions"].items()]
    return f"""Create natural context stimuli for one modifier-to-noun causal experiment.

Noun pair: {family['noun_0']} / {family['noun_1']}
Shared definition: {family['shared_definition']}
Fixed neutral stem: {family['neutral_stem']} the
Modifier constructions with measured low/medium/high noun discrimination:
{json.dumps(compact, indent=2)}

For EACH of the three constructions, write:
- source_context_dev: exactly 4 paraphrases
- target_context_dev: exactly 4 paraphrases
- source_context_test: exactly 4 untouched paraphrases
- target_context_test: exactly 4 untouched paraphrases
- semantic_audit_notes

Each context must be 1-2 natural sentences placed before the fixed stem. Source contexts should make noun_0 and the source modifier pragmatically preferable; target contexts should make noun_1 and the target modifier preferable. However, under EVERY source and target context, all four source/target modifier × noun combinations must remain grammatically and semantically permissible. Change register, emphasis, or collocation—not objective truth.

Never mention either noun, either modifier, their initials, or synonyms that trivially name them. Do not include the fixed stem. Do not label a context inside its text. The generic instruction is identical across conditions: "Complete the sentence with exactly two words: one adjective followed by one noun."

Return only a JSON array of exactly three objects with keys: regime, source_modifier, target_modifier, source_context_dev, target_context_dev, source_context_test, target_context_test, four_cross_combinations_valid, semantic_audit_notes. Set four_cross_combinations_valid true only if the non-negotiable cross-product criterion is met."""


def call(prompt, seed, key):
    model = CFG["generation"]["model"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": 0.7, "seed": seed, "responseMimeType": "application/json"}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as response: raw = json.loads(response.read())
    return raw, raw["candidates"][0]["content"]["parts"][0]["text"]


def valid(rows):
    needed = ("source_context_dev", "target_context_dev", "source_context_test", "target_context_test")
    return len(rows) == 3 and all(len(row.get(key, [])) == 4 for row in rows for key in needed)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--start", type=int, default=0); parser.add_argument("--stop", type=int, default=120); parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args(); key = os.environ.get("GOOGLE_GEMINI_API_KEY")
    if not key: raise SystemExit("GOOGLE_GEMINI_API_KEY is not present")
    families = json.loads(INPUT.read_text()); RAW.mkdir(parents=True, exist_ok=True); PARSED.mkdir(parents=True, exist_ok=True)
    for index in range(args.start, min(args.stop, len(families))):
        family = families[index]; out = PARSED / f"{index:03d}_{family['candidate_id']}.json"
        if out.exists(): print(f"context {index:03d}: cached", flush=True); continue
        prompt = prompt_for(family); seed = 20261100 + index
        for attempt in range(args.max_retries):
            try:
                raw, text = call(prompt, seed, key); rows = extract_json(text)
                if not valid(rows): raise ValueError("wrong context schema or counts")
                envelope = {"generated_at": datetime.now(timezone.utc).isoformat(), "provider": "Google Gemini API",
                            "model": CFG["generation"]["model"], "seed": seed, "candidate_id": family["candidate_id"],
                            "prompt": prompt, "response": raw}
                raw_tmp=(RAW/f"{index:03d}_{family['candidate_id']}.json.tmp"); out_tmp=out.with_suffix(".json.tmp")
                raw_tmp.write_text(json.dumps(envelope,indent=2)+"\n")
                out_tmp.write_text(json.dumps({"family":family,"constructions":rows},indent=2)+"\n")
                raw_tmp.replace(RAW/f"{index:03d}_{family['candidate_id']}.json"); out_tmp.replace(out)
                print(f"context {index:03d}: generated", flush=True); break
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
                if attempt+1 == args.max_retries: raise
                print(f"context {index:03d}: retry {attempt+1} after {type(exc).__name__}", flush=True); time.sleep(min(60,2**attempt+random.random()))
    paths=sorted(p for p in PARSED.glob("*.json") if p.name != "all_contexts.json"); combined=[json.loads(p.read_text()) for p in paths]
    (PARSED/"all_contexts.json").write_text(json.dumps(combined,indent=2)+"\n")
    print(f"assembled {len(combined)} context families")


if __name__ == "__main__": main()
