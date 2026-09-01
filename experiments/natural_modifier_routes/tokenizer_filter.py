#!/usr/bin/env python3
"""Apply immutable local tokenizer and lexical-format filters to candidates."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text())
INPUT = EXP / "artifacts/01_parsed_candidates/all_candidates.json"
OUTPUT = EXP / "artifacts/02_tokenizer_filtered"
WORD = re.compile(r"^[a-z]+$")


def one_token(tokenizer, word: str) -> tuple[bool, list[int]]:
    ids = tokenizer(" " + word, add_special_tokens=False).input_ids
    return len(ids) == 1 and tokenizer.decode(ids).strip().lower() == word, ids


def main() -> None:
    rows = json.loads(INPUT.read_text())
    tokenizer = AutoTokenizer.from_pretrained(CFG["model"]["snapshot"], local_files_only=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    decisions, retained = [], []
    seen_pairs = set()
    for row in rows:
        reasons = []
        y0, y1 = row["noun_0"], row["noun_1"]
        if not WORD.fullmatch(y0) or not WORD.fullmatch(y1): reasons.append("noun_format")
        if y0 == y1: reasons.append("identical_nouns")
        key = tuple(sorted((y0, y1)))
        if key in seen_pairs: reasons.append("duplicate_noun_pair")
        n0_ok, n0_ids = one_token(tokenizer, y0); n1_ok, n1_ids = one_token(tokenizer, y1)
        if not n0_ok or not n1_ok: reasons.append("noun_not_single_token")
        modifier_records = []
        for modifier in dict.fromkeys(row["candidate_modifiers"]):
            ok_format = bool(WORD.fullmatch(modifier))
            ok_token, ids = one_token(tokenizer, modifier) if ok_format else (False, [])
            modifier_records.append({"modifier": modifier, "token_ids": ids, "retained": ok_format and ok_token,
                                     "rejection_reason": None if ok_format and ok_token else "format_or_not_single_token"})
        good = [m for m in modifier_records if m["retained"]]
        if len(good) < 6: reasons.append("fewer_than_six_single_token_modifiers")
        accepted = not reasons
        decision = {**row, "noun_token_ids": {"noun_0": n0_ids, "noun_1": n1_ids},
                    "modifier_decisions": modifier_records, "retained": accepted,
                    "rejection_reasons": reasons}
        decisions.append(decision)
        if accepted:
            seen_pairs.add(key)
            retained.append({**row, "noun_token_ids": decision["noun_token_ids"],
                             "candidate_modifiers": [m["modifier"] for m in good],
                             "modifier_token_ids": {m["modifier"]: m["token_ids"] for m in good}})
    (OUTPUT / "decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")
    (OUTPUT / "retained.json").write_text(json.dumps(retained, indent=2) + "\n")
    summary = {"input_pairs": len(rows), "retained_pairs": len(retained),
               "rejection_counts_nonexclusive": dict(Counter(r for d in decisions for r in d["rejection_reasons"])),
               "retained_modifiers": sum(len(r["candidate_modifiers"]) for r in retained)}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
