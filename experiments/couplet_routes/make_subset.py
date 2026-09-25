#!/usr/bin/env python3
"""Build a steering subset for a model the authors did not steer (Qwen3-32B).

Hanna & Ameisen's rhyme_intervention_sample/<model>.csv is, for every model they
steered, exactly the first 100 couplets of couplets/results/couplets/<model>.csv
with rhyme_success, and each donor (chosen_index) is a random other couplet of
the subset with a different rhyme group (checked for 8B and 14B). Their donors
also required at least one rhyme feature, which needs transcoders; the state edit
does not, so that condition is dropped. original_generation is the authors'
clean_completion (validation only).

Writes results/<model>/subset.csv in the rhyme_intervention_sample format.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parent
COUPLETS = EXP.parents[1] / "external/model-planning-public/couplets/results/couplets"
SEED = 20260925


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-32B"
    c = pd.read_csv(COUPLETS / f"{model}.csv")
    s = c[c["rhyme_success"] == True].head(100).copy()  # noqa: E712
    rng = random.Random(SEED)
    s["chosen_index"] = [rng.choice([j for j in s.index if s.at[j, "rhyme_group"] != g]) for g in s["rhyme_group"]]
    s["found_valid_row"] = True
    s["original_generation"] = s["clean_completion"]
    out = EXP / "results" / model
    out.mkdir(parents=True, exist_ok=True)
    s[["first_line", "first_last_word", "second_last_word", "rhyme_group", "rhyme_success", "found_valid_row",
       "chosen_index", "original_generation"]].to_csv(out / "subset.csv")
    print(f"{len(s)} couplets, indexes {s.index.min()}-{s.index.max()}, {s['rhyme_group'].nunique()} rhyme groups")


if __name__ == "__main__":
    main()
