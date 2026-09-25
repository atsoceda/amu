#!/usr/bin/env python3
"""Hidden choice: route accounting for an unobservable pick (design frozen 2026-09-26, README).

Forced version: prompt + "The weather is calm and mild today.\\nThe fruit I chose is";
target = the next token. Pairs: the same 4 fruits in two orders with different greedy
picks. Edits (donor states, all layers): list positions, post-list block, sentence
block, post-list + sentence; plus the text swap (the donor order in the prompt
itself), the reference for full re-derivation. Route split for the post-list block: direct retrieval,
relay through the sentence block, and necessity. R = log p(donor pick) - log p(original
pick). Output projection in float32.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP.parents[0] / "couplet_routes"))
sys.path.insert(0, str(EXP.parents[0] / "derived_value_carry"))
from chain_routes import F32Head  # noqa: E402
from models import is_gemma, load  # noqa: E402
from step34_routes import boot  # noqa: E402

SEED = 20260925
FRUITS = ["apple", "banana", "cherry", "grape", "lemon", "mango", "peach", "plum"]
SENT = "The weather is calm and mild today.\n"
REVEAL = "The fruit I chose is"


def build(tok, lst, gemma):
    q = (f"Secretly choose one fruit from this list: {', '.join(lst)}. Do not write your choice yet. "
         "First write one sentence about the weather. Then write the fruit you chose.")
    msgs = [{"role": "user", "content": q if gemma else "/no_think " + q}]
    kw = {} if gemma else {"enable_thinking": False}
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    text = prompt + SENT + REVEAL
    enc = tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    offs = enc.offset_mapping[0].tolist()
    ls = text.index("list: ") + len("list: ")
    le = text.index(". Do not write")
    tokpos = lambda a, b: [i for i, (s, e) in enumerate(offs) if s >= a and e <= b and e > s]  # noqa: E731
    list_pos = tokpos(ls, le)
    n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
    target = enc.input_ids.shape[1] - 1
    post = [i for i in range(max(list_pos) + 1, n_prompt)]
    sent = [i for i in range(n_prompt, target)]
    return enc.input_ids, list_pos, post, sent, target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen3-8B")
    ap.add_argument("--pairs", type=int, default=100)
    a = ap.parse_args()
    tok, m, layers = load(a.model)
    m.lm_head = F32Head(m.lm_head.weight)
    gemma = is_gemma(a.model)
    fid = {f: tok.encode(" " + f, add_special_tokens=False)[0] for f in FRUITS}
    assert len(set(fid.values())) == len(FRUITS)

    @torch.no_grad()
    def run(ids, replace=None, want_states=False):
        hooks = []
        if replace:
            for li, layer in enumerate(layers):
                def fn(mod, inp, out, li=li):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h.clone()
                    for p, sts in replace.items():
                        h[0, p] = sts[li].to(h.dtype)
                    return (h, *out[1:]) if isinstance(out, tuple) else h
                hooks.append(layer.register_forward_hook(fn))
        try:
            out = m(ids.to("mps"), output_hidden_states=want_states)
        finally:
            for h in hooks:
                h.remove()
        return torch.log_softmax(out.logits[0, -1].float(), -1).cpu(), ([h[0] for h in out.hidden_states[1:]] if want_states else None)

    def pick(lst):
        y, _ = run(build(tok, lst, gemma)[0])
        return max(lst, key=lambda f: float(y[fid[f]]))

    rng = random.Random(SEED)
    rows, t0, tried = [], time.time(), 0
    while len(rows) < a.pairs and tried < 20 * a.pairs:
        tried += 1
        four = rng.sample(FRUITS, 4)
        perms = list(itertools.permutations(four))
        rng.shuffle(perms)
        picks = {p: pick(list(p)) for p in perms[:8]}
        orig = perms[0]
        donors = [p for p in perms[1:8] if picks[p] != picks[orig]]
        if not donors:
            continue
        don = donors[0]
        ids, lpos, post, sent, target = build(tok, list(orig), gemma)
        dids, dl, dp, ds, dt = build(tok, list(don), gemma)
        if ids.shape != dids.shape or (lpos, post, sent, target) != (dl, dp, ds, dt):
            continue
        po, pdn = picks[orig], picks[don]
        R = lambda y: float(y[fid[pdn]] - y[fid[po]])  # noqa: E731
        y0, s0 = run(ids, want_states=True)
        _, dst = run(dids, want_states=True)
        D = lambda ps: {p: [s[p] for s in dst] for p in ps}  # noqa: E731
        C = lambda ps: {p: [s[p] for s in s0] for p in ps}  # noqa: E731
        R0 = R(y0)
        y_post, s_post = run(ids, D(post), want_states=True)
        rec = {"list": list(orig), "donor_list": list(don), "pick": po, "donor_pick": pdn, "R_off": R0,
               "text_swap": R(run(dids)[0]) - R0,
               "list_edit": R(run(ids, D(lpos))[0]) - R0,
               "post_edit": R(y_post) - R0,
               "sentence_edit": R(run(ids, D(sent))[0]) - R0,
               "post_and_sentence": R(run(ids, D(post + sent))[0]) - R0,
               "post_retrieval": R(run(ids, D(post) | C(sent))[0]) - R0,
               "post_relay": R(run(ids, {q: [s[q] for s in s_post] for q in sent})[0]) - R0}
        rows.append(rec)
        print(f"{len(rows):3d} [{time.time()-t0:5.0f}s] {po}->{pdn} text {rec['text_swap']:+.2f} list {rec['list_edit']:+.2f} post {rec['post_edit']:+.2f} "
              f"(ret {rec['post_retrieval']:+.2f} relay {rec['post_relay']:+.2f}) sentence {rec['sentence_edit']:+.2f} "
              f"post+sent {rec['post_and_sentence']:+.2f}", flush=True)
    out = EXP / "results" / a.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "choice_rows.json").write_text(json.dumps(rows, indent=1))
    keys = ["text_swap", "list_edit", "post_edit", "post_retrieval", "post_relay", "sentence_edit", "post_and_sentence"]
    s = {"model": a.model, "n": len(rows), "lists_tried": tried, **{k: boot([r[k] for r in rows]) for k in keys}}
    (out / "choice_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: (round(v["mean"], 2), round(v["lo"], 2), round(v["hi"], 2)) if isinstance(v, dict) else v for k, v in s.items()}))


if __name__ == "__main__":
    main()
