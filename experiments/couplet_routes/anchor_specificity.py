#!/usr/bin/env python3
"""Anchor specificity: is rhyme information retrieved from wherever it is placed,
or only from learned structural anchors?

Design frozen 2026-09-25 before any result (see README). Same couplets, donors,
prompt and outcome as step34_routes.py, original line-2 words fixed (prefix p0).
The donor's anchor states (all layers) are placed at one position q of the
original prompt:
  A  anchor: last word of line 1 (reference)
  M  middle word of line 1
  F  first word of line 1
  C  the token right after the anchor (line-1 final punctuation)
  E  end of the user turn (<|im_end|>; <end_of_turn> for Gemma 3)
For each q: persistence = R(on) - R(off) at the target (first token of line 2's
last word), split into retrieval_q (positions after q and before the target get
clean states, so only the target reads q) and relay_q (q clean, positions after
q get their edited-run states). R = log mass on donor rhymes - log mass on
original rhymes (single-token rhyme words).
Prediction: large persistence only at A (and possibly C/E); little at M and F.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
from models import end_of_user  # noqa: E402
from step2_state_edit import prompt_ids, rhymes  # noqa: E402

COLON = (":", "Ġ:", "▁:")
from step34_routes import boot, strip_last_word  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen3-1.7B")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    from models import load
    tok, m, layers = load(a.model)
    rows = json.loads((EXP / "results" / a.model / "step2_rows.json").read_text())[: a.limit or None]

    @torch.no_grad()
    def run(ids, replace=None, want_states=False):
        hooks = []
        if replace:
            for li, layer in enumerate(layers):
                def fn(mod, inp, out, li=li):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h.clone()
                    for pos, sts in replace.items():
                        h[0, pos] = sts[li].to(h.dtype)
                    return (h, *out[1:]) if isinstance(out, tuple) else h
                hooks.append(layer.register_forward_hook(fn))
        try:
            out = m(ids, output_hidden_states=want_states)
        finally:
            for h in hooks:
                h.remove()
        return torch.softmax(out.logits[0, -1].float(), -1).cpu(), ([h[0] for h in out.hidden_states[1:]] if want_states else None)

    def rhyme_ids(word):
        return sorted({t[0] for w in rhymes(word) for t in [tok.encode(" " + w, add_special_tokens=False)] if len(t) == 1})

    out_rows, t0 = [], time.time()
    for r in rows:
        p0 = strip_last_word(r["gen_off"])
        dr, orr = rhyme_ids(r["donor_word"]), rhyme_ids(r["orig_word"])
        if not p0 or not dr or not orr:
            continue
        _, anchor, text = prompt_ids(tok, r["first_line"])
        _, danchor, dtext = prompt_ids(tok, r["donor_first_line"])
        _, dsts = run(tok(dtext, return_tensors="pt", add_special_tokens=False).input_ids.to("mps"), want_states=True)
        donor = [s[danchor] for s in dsts]
        ids = tok(text + p0, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
        toks = tok.convert_ids_to_tokens(ids[0])
        start = max(i for i, t in enumerate(toks[:anchor]) if t in COLON) + 1 if any(t in COLON for t in toks[:anchor]) else anchor - 6
        positions = {"A": anchor, "M": (start + anchor) // 2, "F": start, "C": anchor + 1, "E": end_of_user(toks)}
        R = lambda p: math.log(max(float(p[dr].sum()), 1e-12)) - math.log(max(float(p[orr].sum()), 1e-12))  # noqa: E731
        y_off, s_off = run(ids, want_states=True)
        target = ids.shape[1] - 1
        rec = {"idx": r["idx"], "positions": positions, "tokens": {k: toks[v] for k, v in positions.items()}}
        for name, q in positions.items():
            y_on, s_on = run(ids, {q: donor}, want_states=True)
            after = range(q + 1, target)
            y_ret, _ = run(ids, {q: donor} | {p: [s[p] for s in s_off] for p in after})
            y_rel, _ = run(ids, {p: [s[p] for s in s_on] for p in after})
            rec[f"{name}_persistence"] = R(y_on) - R(y_off)
            rec[f"{name}_retrieval"] = R(y_ret) - R(y_off)
            rec[f"{name}_relay"] = R(y_rel) - R(y_off)
            rec[f"{name}_tv"] = 0.5 * float((y_on - y_off).abs().sum())
        out_rows.append(rec)
        print(f"{len(out_rows):3d} [{time.time()-t0:5.0f}s] " + " ".join(f"{k}:{rec[k+'_persistence']:+.1f}" for k in positions), flush=True)
    out = EXP / "results" / a.model
    (out / "anchor_specificity_rows.json").write_text(json.dumps(out_rows, indent=1))
    s = {"model": a.model, "n": len(out_rows), "elapsed_sec": time.time() - t0,
         "by_position": {k: {m_: boot([r[f"{k}_{m_}"] for r in out_rows]) for m_ in ("persistence", "retrieval", "relay", "tv")}
                         for k in ("A", "M", "F", "C", "E")}}
    (out / "anchor_specificity_summary.json").write_text(json.dumps(s, indent=1))
    for k, v in s["by_position"].items():
        print(k, {m_: (round(x["mean"], 2), round(x["lo"], 2), round(x["hi"], 2)) for m_, x in v.items()})


if __name__ == "__main__":
    main()
