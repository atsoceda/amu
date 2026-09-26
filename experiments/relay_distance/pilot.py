#!/usr/bin/env python3
"""Relay-at-a-distance pilot (design frozen 2026-09-26, README).

Couplets with D tokens of neutral filler prefilled in the assistant turn before line 2.
Cells per couplet and distance: persistence, direct retrieval, relay, filler relay.
Contiguous-slice state replacement (one write per layer), so long sequences stay fast.
Checkpointed per couplet and distance (couplet_routes/checkpoint.py).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch

EXP = Path(__file__).resolve().parent
CR = EXP.parents[0] / "couplet_routes"
sys.path.insert(0, str(CR))
from checkpoint import Checkpoint  # noqa: E402
from models import load  # noqa: E402
from step2_state_edit import prompt_ids, rhymes, subset_csv  # noqa: E402
from step34_routes import boot, strip_last_word  # noqa: E402

SEED = 20260925
SENTENCES = [
    "The morning was quiet and the streets were mostly empty.", "A light breeze moved through the trees near the park.",
    "People walked slowly past the shops on the main road.", "The river flowed under the old stone bridge.",
    "A few clouds drifted across the pale sky.", "Someone was reading a newspaper on a bench.",
    "The bakery opened its doors at the usual hour.", "Birds gathered on the roof of the train station.",
    "A delivery van stopped outside the library.", "The clock on the tower showed a quarter past nine.",
    "Children played a game in the small square.", "The market stalls were setting out their tables.",
    "A cyclist rode along the path beside the canal.", "The air smelled faintly of coffee and rain.",
    "Two neighbours talked about their gardens.", "A bus turned the corner and slowed down.",
    "The windows of the houses reflected the light.", "A dog waited patiently by a closed door.",
    "The school bell rang somewhere in the distance.", "Leaves collected along the edge of the pavement.",
    "A man carried a box of apples to his car.", "The fountain in the square was switched off.",
    "An old couple sat together near the entrance.", "The post office had a short queue.",
    "A painter set up an easel by the water.", "The café put out chairs for the afternoon.",
    "Traffic moved steadily along the wide avenue.", "A woman watered the plants on her balcony.",
    "The museum was preparing a new exhibition.", "Workers repaired a section of the road.",
]


def filler(tok, n, rng):
    if n == 0:
        return ""
    out, ids = [], []
    while len(ids) < n:
        out.append(rng.choice(SENTENCES))
        ids = tok(" ".join(out), add_special_tokens=False).input_ids
    return tok.decode(ids[:n]).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--distances", default="0,2000")
    a = ap.parse_args()
    tok, m, layers = load(a.model)
    df = pd.read_csv(subset_csv(a.model), index_col=0)
    df = df[df["found_valid_row"].astype(str).str.lower() == "true"].head(a.n)
    allrows = pd.read_csv(subset_csv(a.model), index_col=0)
    dists = [int(x) for x in a.distances.split(",")]
    ck = Checkpoint(EXP / "results" / a.model / "pilot_rows.json", key=lambda r: f"{r['idx']}-{r['distance']}")

    def rhyme_ids(word):
        return sorted({t[0] for w in rhymes(word) for t in [tok.encode(" " + w, add_special_tokens=False)] if len(t) == 1})

    @torch.no_grad()
    def run(ids, spans=(), want_states=False):
        """spans: list of (start, end, per-layer states of shape (end-start, d))."""
        hooks = []
        for li, layer in enumerate(layers):
            if not spans:
                break

            def fn(mod, inp, out, li=li):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                for s, e, st in spans:
                    h[0, s:e] = st[li].to(h.device, h.dtype)
                return (h, *out[1:]) if isinstance(out, tuple) else h
            hooks.append(layer.register_forward_hook(fn))
        try:
            out = m(ids.to("mps"), output_hidden_states=want_states, logits_to_keep=1)
        finally:
            for h in hooks:
                h.remove()
        p = torch.softmax(out.logits[0, -1].float(), -1).cpu()
        return p, ([h[0] for h in out.hidden_states[1:]] if want_states else None)

    @torch.no_grad()
    def gen_line(text):
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
        g = m.generate(ids, max_new_tokens=28, do_sample=False)
        return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0].strip()

    t0 = time.time()
    for idx, r in df.iterrows():
        donor = allrows.loc[int(r["chosen_index"])]
        dr, orr = rhyme_ids(donor["first_last_word"]), rhyme_ids(r["first_last_word"])
        if not dr or not orr:
            continue
        for D in dists:
            if ck.has(f"{int(idx)}-{D}"):
                continue
            rng = random.Random(SEED + int(idx) * 7919 + D)
            fill = filler(tok, D, rng)
            pre = (fill + "\n\n") if fill else ""
            _, anchor, text = prompt_ids(tok, r["first_line"])
            _, danchor, dtext = prompt_ids(tok, donor["first_line"])
            line2 = gen_line(text + pre)
            p0 = strip_last_word(line2)
            if not p0:
                continue
            full = text + pre + p0
            ids = tok(full, return_tensors="pt", add_special_tokens=False).input_ids
            n_text = len(tok(text, add_special_tokens=False).input_ids)
            n_pre = len(tok(text + pre, add_special_tokens=False).input_ids)
            target = ids.shape[1] - 1
            _, dsts = run(tok(dtext, return_tensors="pt", add_special_tokens=False).input_ids, want_states=True)
            donor_anchor = [s[danchor:danchor + 1] for s in dsts]
            del dsts
            R = lambda p: math.log(max(float(p[dr].sum()), 1e-12)) - math.log(max(float(p[orr].sum()), 1e-12))  # noqa: E731
            y_off, s_off = run(ids, want_states=True)
            edit = [(anchor, anchor + 1, donor_anchor)]
            y_on, s_on = run(ids, edit, want_states=True)
            a1 = anchor + 1
            clean_mid = [(a1, target, [s[a1:target] for s in s_off])]
            on_mid = [(a1, target, [s[a1:target] for s in s_on])]
            on_fill = [(n_text, n_pre, [s[n_text:n_pre] for s in s_on])] if n_pre > n_text else []
            base = R(y_off)
            rec = {"idx": int(idx), "distance": D, "n_tokens": int(ids.shape[1]), "line2": line2,
                   "line2_rhymes_orig": line2.split()[-1].strip(".,!?;:'\"").lower() in rhymes(r["first_last_word"]) if line2.split() else False,
                   "persistence": R(y_on) - base,
                   "retrieval": R(run(ids, edit + clean_mid)[0]) - base,
                   "relay": R(run(ids, on_mid)[0]) - base,
                   "filler_relay": (R(run(ids, on_fill)[0]) - base) if on_fill else 0.0}
            del s_off, s_on
            ck.add(rec)
            print(f"{len(ck.rows):3d} [{time.time()-t0:5.0f}s] D={D:5d} T={rec['n_tokens']:5d} pers {rec['persistence']:+.2f} "
                  f"ret {rec['retrieval']:+.2f} relay {rec['relay']:+.2f} filler {rec['filler_relay']:+.2f} | {line2[:50]!r}", flush=True)
    rows = ck.rows
    s = {"model": a.model, "by_distance": {}}
    for D in dists:
        rr = [x for x in rows if x["distance"] == D]
        s["by_distance"][D] = {"n": len(rr), "rhymes_orig": sum(x["line2_rhymes_orig"] for x in rr) / max(len(rr), 1),
                               **{k: boot([x[k] for x in rr]) for k in ("persistence", "retrieval", "relay", "filler_relay")},
                               "relay_share": boot([x["relay"] / x["persistence"] for x in rr if abs(x["persistence"]) > 1])}
    ck.finish(rows)
    out = EXP / "results" / a.model
    (out / "pilot_summary.json").write_text(json.dumps(s, indent=1))
    for D, v in s["by_distance"].items():
        print(D, {k: (round(x["mean"], 2), round(x["lo"], 2), round(x["hi"], 2)) if isinstance(x, dict) and x else x for k, x in v.items()})


if __name__ == "__main__":
    main()
