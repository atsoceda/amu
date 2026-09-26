#!/usr/bin/env python3
"""Hidden choice replication (animals) with pre-registered pick specificity, and
per-token localization of the stored pick (designs frozen 2026-09-26, README).

--domain animals: disjoint 6-item lists from 12 animals; post-list edit; outcome = log
p(donor pick) - log p(original pick) and the same for the donor's other items;
specificity = pick minus mean of others. --localize: fruits (4+4 disjoint), donor
post-list states everywhere, each post-list token reset to clean in turn; batched.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP.parents[0] / "couplet_routes"))
sys.path.insert(0, str(EXP.parents[0] / "derived_value_carry"))
from batching import run_edits  # noqa: E402
from chain_routes import F32Head  # noqa: E402
from choice_routes import FRUITS, SEED  # noqa: E402
from models import is_gemma, load  # noqa: E402
from step34_routes import boot  # noqa: E402

# Pool rule fixed in advance: the first 12 of these that are single tokens (with a leading space)
ANIMAL_POOL = ["horse", "tiger", "rabbit", "eagle", "whale", "snake", "mouse", "sheep", "zebra", "camel", "otter",
               "shark", "goat", "wolf", "bear", "duck"]


WORDINGS = {  # "base" is the original instruction; "alt" moves and changes the function words (added 2026-09-26)
    "base": "Do not write your choice yet. First write one sentence about the weather. Then write the {noun} you chose.",
    "alt": "Keep your choice private for now. Begin by describing today's weather in a single sentence. After that, reveal which {noun} you picked.",
}


def build(tok, lst, gemma, noun, wording="base"):
    q = f"Secretly choose one {noun} from this list: {', '.join(lst)}. " + WORDINGS[wording].format(noun=noun)
    msgs = [{"role": "user", "content": q if gemma else "/no_think " + q}]
    kw = {} if gemma else {"enable_thinking": False}
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    text = prompt + "The weather is calm and mild today.\n" + f"The {noun} I {'chose' if wording == 'base' else 'picked'} is"
    enc = tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    offs = enc.offset_mapping[0].tolist()
    ls, le = text.index("list: ") + 6, text.index(". ", text.index("list: "))
    lpos = [i for i, (s, e) in enumerate(offs) if s >= ls and e <= le and e > s]
    n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
    return enc.input_ids, lpos, list(range(max(lpos) + 1, n_prompt)), [tok.convert_ids_to_tokens(int(t)) for t in enc.input_ids[0]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", choices=["animals", "fruits"], default="animals")
    ap.add_argument("--localize", action="store_true")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--wording", choices=["base", "alt"], default="base")
    ap.add_argument("--null-same-pick", action="store_true", help="donor = same list reordered with the same pick (added 2026-09-26)")
    a = ap.parse_args()
    tok, m, layers = load(a.model)
    m.lm_head = F32Head(m.lm_head.weight)
    gemma = is_gemma(a.model)
    single = lambda w: len(tok.encode(" " + w, add_special_tokens=False)) == 1  # noqa: E731
    items, k, noun = ([w for w in ANIMAL_POOL if single(w)][:12], 6, "animal") if a.domain == "animals" else (FRUITS, 4, "fruit")
    assert len(items) == (12 if a.domain == "animals" else 8), items
    fid = {f: tok.encode(" " + f, add_special_tokens=False)[0] for f in items}
    print("items:", items, flush=True)
    rng = random.Random(SEED + (3 if a.domain == "animals" else 1))
    rows, t0 = [], time.time()
    import itertools
    for _ in range(a.pairs if not a.null_same_pick else 4 * a.pairs):
        if len(rows) >= a.pairs:
            break
        allf = rng.sample(items, 2 * k)
        orig, don = allf[:k], allf[k:]
        if a.null_same_pick:
            ids0, _, _, _ = build(tok, orig, gemma, noun, a.wording)
            (y00,), _ = run_edits(m, layers, ids0, [{}])
            pk = max(orig, key=lambda f: float(y00[fid[f]]))
            perms = [list(p) for p in itertools.permutations(orig) if list(p) != orig]
            rng.shuffle(perms)
            don = None
            for p in perms[:8]:
                idsp, _, _, _ = build(tok, p, gemma, noun, a.wording)
                (yp,), _ = run_edits(m, layers, idsp, [{}])
                if max(p, key=lambda f: float(yp[fid[f]])) == pk:
                    don = p
                    break
            if don is None:
                continue
        ids, lpos, post, toks = build(tok, orig, gemma, noun, a.wording)
        dids, dl, dp, _ = build(tok, don, gemma, noun, a.wording)
        if ids.shape != dids.shape or (lpos, post) != (dl, dp):
            continue
        (y0,), _ = run_edits(m, layers, ids, [{}])
        (yd,), dsts = run_edits(m, layers, dids, [{}], want_states=True)
        dst = dsts[0]
        po = max(orig, key=lambda f: float(y0[fid[f]]))
        pdn = max(don, key=lambda f: float(yd[fid[f]]))
        if a.null_same_pick:
            rel = lambda y: float(y[fid[po]]) - sum(float(y[fid[f]]) for f in orig if f != po) / (len(orig) - 1)  # noqa: E731
        others = [f for f in don if f != pdn]
        spec = lambda y: float(y[fid[pdn]] - y[fid[po]]) - sum(float(y[fid[f]] - y[fid[po]]) for f in others) / len(others)  # noqa: E731
        pick = lambda y: float(y[fid[pdn]] - y[fid[po]])  # noqa: E731
        Dpost = {p: [s[p] for s in dst] for p in post}
        reps = [Dpost]
        if a.localize:
            _, s0 = run_edits(m, layers, ids, [{}], want_states=True)
            s0 = s0[0]
            reps += [{p: v for p, v in Dpost.items() if p != q} | {q: [s[q] for s in s0]} for q in post]
        ys, _ = run_edits(m, layers, ids, reps)
        if a.null_same_pick:
            (ypost,), _ = run_edits(m, layers, ids, [{p: [s[p] for s in dst] for p in post}])
            rows.append({"list": orig, "donor_list": don, "pick": po, "donor_pick": pdn,
                         "text_swap_rel": rel(yd) - rel(y0), "post_rel": rel(ypost) - rel(y0)})
            print(f"{len(rows):3d} same-pick {po}: text {rows[-1]['text_swap_rel']:+.2f} post {rows[-1]['post_rel']:+.2f}", flush=True)
            continue
        rec = {"list": orig, "donor_list": don, "pick": po, "donor_pick": pdn,
               "text_swap_pick": pick(yd) - pick(y0), "text_swap_specificity": spec(yd) - spec(y0),
               "post_pick": pick(ys[0]) - pick(y0), "post_specificity": spec(ys[0]) - spec(y0)}
        if a.localize:
            rec["post_tokens"] = [toks[q] for q in post]
            rec["necessity_by_token"] = [rec["post_specificity"] - (spec(y) - spec(y0)) for y in ys[1:]]
        rows.append(rec)
        print(f"{len(rows):3d} [{time.time()-t0:5.0f}s] {po} vs {pdn}: text spec {rec['text_swap_specificity']:+.2f} "
              f"post spec {rec['post_specificity']:+.2f}", flush=True)
    tag = f"_{a.domain}" + ("_localize" if a.localize else "") + ("" if a.wording == "base" else f"_{a.wording}")
    if a.null_same_pick:
        tag += "_samepick"
        out = EXP / "results" / a.model
        out.mkdir(parents=True, exist_ok=True)
        (out / f"choice_replicate{tag}_rows.json").write_text(json.dumps(rows, indent=1))
        s_ = {"model": a.model, "n": len(rows), "text_swap_rel": boot([r["text_swap_rel"] for r in rows]),
              "post_rel": boot([r["post_rel"] for r in rows])}
        (out / f"choice_replicate{tag}_summary.json").write_text(json.dumps(s_, indent=1))
        print(s_)
        return
    out = EXP / "results" / a.model
    out.mkdir(parents=True, exist_ok=True)
    (out / f"choice_replicate{tag}_rows.json").write_text(json.dumps(rows, indent=1))
    s = {"model": a.model, "domain": a.domain, "n": len(rows),
         **{k_: boot([r[k_] for r in rows]) for k_ in ("text_swap_pick", "text_swap_specificity", "post_pick", "post_specificity")}}
    if a.localize and rows:
        ref = rows[0]["post_tokens"]
        same = [r for r in rows if r["post_tokens"] == ref]
        s["necessity_by_token"] = [{"token": t, **boot([r["necessity_by_token"][i] for r in same])} for i, t in enumerate(ref)]
    (out / f"choice_replicate{tag}_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k_: (round(v["mean"], 2), round(v["lo"], 2), round(v["hi"], 2)) if isinstance(v, dict) else v
                      for k_, v in s.items() if k_ != "necessity_by_token"}))
    for d in s.get("necessity_by_token", []):
        print(f"  {d['token']!r:>18} {d['mean']:+.2f} [{d['lo']:.2f}, {d['hi']:.2f}]")


if __name__ == "__main__":
    main()
