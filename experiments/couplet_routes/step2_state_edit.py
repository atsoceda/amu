#!/usr/bin/env python3
"""Couplet routes, step 2: does a construction-free state edit at the line-1 rhyme
anchor change the rhyme of the generated line 2?

Couplets and donor pairings are Hanna & Ameisen's released 100-couplet steering
subset (couplets/results/rhyme_intervention_sample/<model>.csv: each row's
chosen_index names a donor couplet with a different rhyme group). The prompt is
their chat format; the anchor is their rhyme-feature position (two tokens before
the end of the user turn: the last word of line 1).

State edit: at the anchor, every layer's output is replaced by the donor prompt's
states at its own anchor; text is unchanged. We generate line 2 greedily with and
without the edit and score the last word against Datamuse rhymes of the original
and the donor line-1 words (the authors' rhyme criterion). Validation: the
unedited generation should match the authors' released original_generation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import load, prompt_ids  # noqa: E402,F401  (prompt_ids re-exported for the other steps)

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
HA = ROOT / "external/model-planning-public/couplets/results/rhyme_intervention_sample"


def subset_csv(model: str) -> Path:
    """The authors' steering subset, or ours (make_subset.py) for models they did not steer."""
    p = HA / f"{model}.csv"
    return p if p.exists() else EXP / "results" / model / "subset.csv"


RHYME_CACHE = EXP / "results" / "datamuse_rhymes.json"
_rhyme_disk = json.loads(RHYME_CACHE.read_text()) if RHYME_CACHE.exists() else {}


@lru_cache(maxsize=None)
def rhymes(word: str) -> frozenset:
    """Datamuse rel_rhy set, cached on disk so later runs work offline."""
    w = word.lower()
    if w in _rhyme_disk:
        return frozenset(_rhyme_disk[w])
    q = urllib.parse.urlencode({"rel_rhy": w, "max": 1000})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"https://api.datamuse.com/words?{q}", timeout=20) as r:
                got = sorted(x["word"].lower() for x in json.load(r))
            _rhyme_disk[w] = got
            RHYME_CACHE.parent.mkdir(parents=True, exist_ok=True)
            RHYME_CACHE.write_text(json.dumps(_rhyme_disk))
            return frozenset(got)
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)
    return frozenset()


def last_word(text: str) -> str:
    words = re.findall(r"[A-Za-z']+", text.split("\n")[-1] if "\n" in text.strip() else text)
    return words[-1].lower().strip("'") if words else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", default="Qwen3-1.7B", nargs="?")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--control", choices=["same_rhyme"], help="null control: donor from the same rhyme group")
    a = ap.parse_args()
    tok, m, layers = load(a.model)
    df = pd.read_csv(subset_csv(a.model), index_col=0)
    df = df[df["found_valid_row"].astype(str).str.lower() == "true"]
    if a.limit:
        df = df.head(a.limit)

    @torch.no_grad()
    def anchor_states(first_line):
        ids, anchor, _ = prompt_ids(tok, first_line)
        out = m(ids.to("mps"), output_hidden_states=True)
        return [h[0, anchor].clone() for h in out.hidden_states[1:]]

    @torch.no_grad()
    def generate(first_line, patch=None):
        ids, anchor, _ = prompt_ids(tok, first_line)
        hooks = []
        if patch is not None:
            for li, layer in enumerate(layers):
                def fn(mod, inp, out, li=li):
                    h = out[0] if isinstance(out, tuple) else out
                    if h.shape[1] > anchor:  # prompt pass only; cached steps have length 1
                        h = h.clone()
                        h[0, anchor] = patch[li].to(h.dtype)
                        return (h, *out[1:]) if isinstance(out, tuple) else h
                    return out
                hooks.append(layer.register_forward_hook(fn))
        try:
            out = m.generate(ids.to("mps"), max_new_tokens=24, do_sample=False)
        finally:
            for h in hooks:
                h.remove()
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    all_rows = pd.read_csv(subset_csv(a.model), index_col=0)
    if a.control == "same_rhyme":
        import random
        rng = random.Random(20260925)
        same = {}
        for idx, r in df.iterrows():
            pool = [j for j, q in all_rows.iterrows() if j != idx and q["rhyme_group"] == r["rhyme_group"]]
            if pool:
                same[idx] = rng.choice(pool)
        df = df.loc[list(same)]
    rows = []
    t0 = time.time()
    for idx, r in df.iterrows():
        donor = all_rows.loc[same[idx] if a.control else int(r["chosen_index"])]
        orig_w, donor_w = r["first_last_word"], donor["first_last_word"]
        g0 = generate(r["first_line"])
        g1 = generate(r["first_line"], anchor_states(donor["first_line"]))
        w0, w1 = last_word(g0), last_word(g1)
        rec = {"idx": int(idx), "first_line": r["first_line"], "donor_first_line": donor["first_line"],
               "orig_word": orig_w, "donor_word": donor_w, "gen_off": g0, "gen_on": g1,
               "authors_original_generation": r["original_generation"],
               "matches_authors": g0.strip() == str(r["original_generation"]).strip(),
               "off_rhymes_orig": w0 in rhymes(orig_w), "off_rhymes_donor": w0 in rhymes(donor_w),
               "on_rhymes_orig": w1 in rhymes(orig_w), "on_rhymes_donor": w1 in rhymes(donor_w),
               "on_changed_word": w1 != w0}
        rows.append(rec)
        print(f"{len(rows):3d} [{time.time()-t0:5.0f}s] off='{g0}' | on='{g1}' | donor rhyme {rec['on_rhymes_donor']}", flush=True)
    out = EXP / "results" / a.model
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{a.control}" if a.control else ""
    (out / f"step2_rows{tag}.json").write_text(json.dumps(rows, indent=1))
    n = len(rows)
    s = {"model": a.model, "n": n,
         "matches_authors_original": sum(r["matches_authors"] for r in rows) / n,
         "off_rhymes_orig": sum(r["off_rhymes_orig"] for r in rows) / n,
         "on_rhymes_orig": sum(r["on_rhymes_orig"] for r in rows) / n,
         "on_rhymes_donor": sum(r["on_rhymes_donor"] for r in rows) / n,
         "off_rhymes_donor": sum(r["off_rhymes_donor"] for r in rows) / n,
         "on_changed_last_word": sum(r["on_changed_word"] for r in rows) / n,
         "elapsed_sec": time.time() - t0}
    (out / f"step2_summary{tag}.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
