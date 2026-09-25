#!/usr/bin/env python3
"""Build a steering subset for a model the authors did not run (Gemma 3).

Following the authors' evaluate_couplets.py: greedy line 2 for their first lines in
order (same prompt as our experiments, via models.prompt_ids), sanitized, rhyme
success = the last word of line 2 is a Datamuse rhyme of line 1's last word and
differs from it. The first 100 successes form the subset; rhyme groups are the
authors' (they depend only on the first lines and their order); donors are drawn
as in make_subset.py (seeded, different rhyme group). Writes
results/<model>/rhyme_screen.json and results/<model>/subset.csv.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
from make_subset import COUPLETS, SEED  # noqa: E402
from models import load, prompt_ids  # noqa: E402
from step2_state_edit import rhymes  # noqa: E402

LAST_WORD = re.compile(r"([A-Za-z']+)[^A-Za-z']*$")


def last_word(text: str) -> str:
    m = LAST_WORD.search(text.strip())
    return m.group(1).lower().strip("'") if m else ""


def sanitize(first_line: str, raw: str) -> str:
    """Port of the authors' sanitize_second_line (minus the Qwen think-tag handling)."""
    s = re.sub(r"^\s*(?:the\s+)?second\s+line\s*[:\-–]?\s*", "", raw, flags=re.I).strip().strip('"*')
    first = first_line.strip().lower()
    if s.lower().startswith(first):
        s = s[len(first_line.strip()):].lstrip(" \t\n,/:-—")
    parts = re.split(r"[\n/]+", s)
    if len(parts) > 1:
        s = next((p.strip(" \t-—") for p in parts if p.strip(" \t-—") and p.strip(" \t-—").lower() != first), s)
    return " ".join(s.split()).strip().strip('"*')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    tok, m, _ = load(a.model)
    src = pd.read_csv(COUPLETS / "Qwen3-14B.csv")  # first lines and the authors' rhyme groups
    rows = []
    for idx, r in src.iterrows():
        ids, _, _ = prompt_ids(tok, r["first_line"])
        with torch.no_grad():
            out = m.generate(ids.to("mps"), max_new_tokens=24, do_sample=False)
        raw = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        line2 = sanitize(r["first_line"], raw)
        l1, l2 = last_word(r["first_line"]), last_word(line2)
        ok = bool(l1 and l2 and l2 != l1 and l2 in rhymes(l1))
        rows.append({"idx": int(idx), "first_line": r["first_line"], "raw": raw, "line2": line2, "rhyme_success": ok})
        print(f"{idx:4d} {'OK ' if ok else '-- '} {line2}", flush=True)
        if sum(x["rhyme_success"] for x in rows) >= a.n:
            break
    out = EXP / "results" / a.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "rhyme_screen.json").write_text(json.dumps(rows, indent=1))
    ok = [x["idx"] for x in rows if x["rhyme_success"]]
    s = src.loc[ok, ["first_line", "first_last_word", "rhyme_group"]].copy()
    s["second_last_word"] = [last_word(x["line2"]) for x in rows if x["rhyme_success"]]
    s["original_generation"] = [x["line2"] for x in rows if x["rhyme_success"]]
    rng = random.Random(SEED)
    s["chosen_index"] = [rng.choice(c) if (c := [j for j in s.index if s.at[j, "rhyme_group"] != g]) else -999 for g in s["rhyme_group"]]
    s = s[s["chosen_index"] != -999]
    s["found_valid_row"] = True
    s.to_csv(out / "subset.csv")
    print(f"rhyme success {len(ok)}/{len(rows)} = {len(ok)/len(rows):.0%}; subset {len(s)}")


if __name__ == "__main__":
    main()
