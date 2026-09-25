#!/usr/bin/env python3
"""Relay positive control: route split on induction (design frozen 2026-09-26, README).

Random sequence x_0..x_{L-1} of distinct common word tokens, then x_0..x_k repeated;
target = last position (second x_k), correct next token B = x_{k+1}. Anchor = first
x_k (position k after any BOS). Donor = the same sequence with x_k replaced.
R = log p(B) at the target. Cells: persistence (donor state at the anchor, all
layers), retrieval (plus every in-between position clean), relay (anchor clean,
in-between positions with edited-run states), key relay (only B's first position).
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
sys.path.insert(0, str(EXP.parent / "couplet_routes"))
from models import load  # noqa: E402
from step34_routes import boot  # noqa: E402

SEED, L, K, N = 20260925, 24, 11, 100


def word_tokens(tok):
    """Common single-token lowercase words with a leading space (tokenizer-specific ids)."""
    words = ("time year people way day man thing woman life child world school state family student group country "
             "problem hand part place case week company system program question work government number night point "
             "home water room mother area money story fact month lot right study book eye job word business issue "
             "side kind head house service friend father power hour game line end member law car city community name "
             "president team minute idea kid body information back parent face others level office door health person "
             "art war history party result change morning reason research girl guy moment air teacher force education "
             "foot boy age policy music market sense nation plan college interest death experience effect class control "
             "care field development role effort rate heart drug show leader light voice wife police mind price report "
             "decision son view relationship town road arm difference value building action model season society tax "
             "director position player record paper space ground form event official matter center couple site project "
             "activity star table need court oil situation cost industry figure street image phone data picture practice "
             "piece land product doctor wall patient worker news test movie north love support technology step baby "
             "computer type attention film tree source organization hair window evidence population truth fire garden").split()
    ids = []
    for w in words:
        t = tok.encode(" " + w, add_special_tokens=False)
        if len(t) == 1 and t[0] not in ids:
            ids.append(t[0])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen3-1.7B")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    tok, m, layers = load(a.model)
    vocab = word_tokens(tok)
    rng = random.Random(SEED)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None and "gemma" in a.model.lower() else []

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
            out = m(torch.tensor([ids], device="mps"), output_hidden_states=want_states)
        finally:
            for h in hooks:
                h.remove()
        return torch.log_softmax(out.logits[0, -1].float(), -1).cpu(), ([h[0] for h in out.hidden_states[1:]] if want_states else None)

    rows, t0 = [], time.time()
    for i in range(a.limit or N):
        seq = rng.sample(vocab, L)
        y = rng.choice([v for v in vocab if v not in seq])
        ids = bos + seq + seq[: K + 1]
        dids = bos + seq[:K] + [y] + seq[K + 1:] + seq[: K + 1]
        anchor, key, target = len(bos) + K, len(bos) + K + 1, len(ids) - 1
        B = seq[K + 1]
        y_off, s_off = run(ids, want_states=True)
        _, dst = run(dids, want_states=True)
        edit = {anchor: [s[anchor] for s in dst]}
        y_on, s_on = run(ids, edit, want_states=True)
        mids = range(anchor + 1, target)
        R = lambda y_: float(y_[B])  # noqa: E731
        rec = {"i": i, "p_clean": float(y_off[B].exp()), "persistence": R(y_on) - R(y_off),
               "retrieval": R(run(ids, edit | {q: [s[q] for s in s_off] for q in mids})[0]) - R(y_off),
               "relay": R(run(ids, {q: [s[q] for s in s_on] for q in mids})[0]) - R(y_off),
               "key_relay": R(run(ids, {key: [s[key] for s in s_on]})[0]) - R(y_off)}
        rows.append(rec)
        print(f"{len(rows):3d} [{time.time()-t0:5.0f}s] p(B) {rec['p_clean']:.2f} pers {rec['persistence']:+.2f} "
              f"ret {rec['retrieval']:+.2f} relay {rec['relay']:+.2f} key {rec['key_relay']:+.2f}", flush=True)
    out = EXP / "results" / a.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "induction_rows.json").write_text(json.dumps(rows, indent=1))
    s = {"model": a.model, "n": len(rows), "mean_p_clean": sum(r["p_clean"] for r in rows) / len(rows),
         **{k: boot([r[k] for r in rows]) for k in ("persistence", "retrieval", "relay", "key_relay")},
         "relay_share": boot([r["relay"] / r["persistence"] for r in rows if r["persistence"] < -1])}
    (out / "induction_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: (round(v["mean"], 3) if isinstance(v, dict) else v) for k, v in s.items()}, indent=1))


if __name__ == "__main__":
    main()
