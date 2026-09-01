#!/usr/bin/env python3
"""Screen forced modifiers for signed noun-pair branch leverage in Gemma 3 1B."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text())
INPUT = EXP / "artifacts/02_tokenizer_filtered/retained.json"
OUTPUT = EXP / "artifacts/03_branch_leverage"
ROWS_PATH = OUTPUT / "rows.json"


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    families = json.loads(INPUT.read_text())
    tokenizer = AutoTokenizer.from_pretrained(CFG["model"]["snapshot"], local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        CFG["model"]["snapshot"], dtype=getattr(torch, CFG["model"]["dtype"]),
        local_files_only=True, low_cpu_mem_usage=True
    ).eval()
    existing = json.loads(ROWS_PATH.read_text()) if ROWS_PATH.exists() else []
    done = {(r["candidate_id"], r["modifier"]) for r in existing}
    jobs = []
    instruction = CFG["generic_instruction"]
    for family in families:
        for modifier in family["candidate_modifiers"]:
            key = (family["candidate_id"], modifier)
            if key in done: continue
            prompt = f"{instruction}\n{family['neutral_stem']} the {modifier}"
            jobs.append((family, modifier, prompt))
    if args.limit is not None: jobs = jobs[:args.limit]
    # Reuse the exact prefix once per family, then branch all one-token modifiers
    # in a single cached batch. This is numerically identical to full forwards.
    pending = {}
    for family, modifier, prompt in jobs: pending.setdefault(family["candidate_id"], {"family": family, "items": []})["items"].append((modifier, prompt))
    started = time.time(); rows = list(existing); completed_new = 0
    for family_index, record in enumerate(pending.values(), 1):
        family, items = record["family"], record["items"]
        prefix = f"{instruction}\n{family['neutral_stem']} the"
        prefix_inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=True)
        modifier_ids = torch.tensor([[tokenizer(" " + modifier, add_special_tokens=False).input_ids[0]] for modifier, _ in items])
        with torch.inference_mode():
            prefix_out = model(**prefix_inputs, use_cache=True, logits_to_keep=1)
            cache = prefix_out.past_key_values
            prefix_length = cache.get_seq_length()
            cache.batch_repeat_interleave(len(items))
            attention_mask = torch.ones((len(items), prefix_length + 1), dtype=torch.long)
            logits = model(input_ids=modifier_ids, attention_mask=attention_mask, past_key_values=cache,
                           use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu()
        probs = torch.softmax(logits, -1)
        for index, (modifier, prompt) in enumerate(items):
            y0 = int(family["noun_token_ids"]["noun_0"][0]); y1 = int(family["noun_token_ids"]["noun_1"][0])
            rows.append({
                "candidate_id": family["candidate_id"], "family_id": family["family_id"],
                "noun_0": family["noun_0"], "noun_1": family["noun_1"], "modifier": modifier,
                "prompt": prompt,
                "target_probability_projection": float(probs[index, y1] - probs[index, y0]),
                "target_logit_projection": float(logits[index, y1] - logits[index, y0]),
                "noun_pair_probability_mass": float(probs[index, y1] + probs[index, y0]),
                "top_token": tokenizer.decode([int(logits[index].argmax())]).strip(),
            })
        completed_new += len(items)
        if family_index % 10 == 0 or family_index == len(pending):
            atomic_json(ROWS_PATH, rows)
            print(f"scored {completed_new}/{len(jobs)} new; {len(rows)} total", flush=True)
    atomic_json(ROWS_PATH, rows)

    grouped = {}
    for row in rows: grouped.setdefault(row["candidate_id"], []).append(row)
    summaries = []
    family_lookup = {f["candidate_id"]: f for f in families}
    for candidate_id, group in grouped.items():
        ordered = sorted(group, key=lambda r: r["target_probability_projection"])
        base = family_lookup[candidate_id]
        summaries.append({
            "candidate_id": candidate_id, "family_id": base["family_id"], "domain": base["domain"],
            "noun_0": base["noun_0"], "noun_1": base["noun_1"],
            "modifier_count": len(group),
            "probability_projection_min": ordered[0]["target_probability_projection"],
            "probability_projection_max": ordered[-1]["target_probability_projection"],
            "probability_projection_span": ordered[-1]["target_probability_projection"] - ordered[0]["target_probability_projection"],
            "lowest_modifiers": [r["modifier"] for r in ordered[:3]],
            "highest_modifiers": [r["modifier"] for r in ordered[-3:]],
            "max_pair_mass": max(r["noun_pair_probability_mass"] for r in group),
        })
    summaries.sort(key=lambda r: r["probability_projection_span"], reverse=True)
    atomic_json(OUTPUT / "family_summary.json", summaries)
    atomic_json(OUTPUT / "run_manifest.json", {
        "completed_at": datetime.now(timezone.utc).isoformat(), "model": CFG["model"],
        "families": len(summaries), "modifier_branches": len(rows),
        "elapsed_sec_this_run": time.time() - started,
        "primary_metric": "P(noun_1)-P(noun_0) after forced modifier",
        "secondary_metric": "logit(noun_1)-logit(noun_0)",
    })
    print(f"complete: {len(rows)} branches across {len(summaries)} families")


if __name__ == "__main__":
    main()
