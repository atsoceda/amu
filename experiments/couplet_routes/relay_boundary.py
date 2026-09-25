#!/usr/bin/env python3
"""Which boundary token carries the rhyme? Per-token relay in the prompt tail.

Question fixed 2026-09-26 before any result, after boundary relay (prompt tail
between line 1 and line 2) appeared from Qwen3-14B on (README, position split).
Same couplets, donors, state edit, outcome and original line-2 words as
relay_positions.py. For each boundary position q (every prompt position after the
anchor), only q gets its edited-run state (all layers); everything else is clean
and recomputed. relay_q = R(that run) - R(no edit). Also "boundary, all" (every
boundary position edited) for comparison with relay_positions.py `tail`.
Reported per boundary token (the tail has the same tokens in every couplet of a
format), with bootstrap CIs.
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
from models import load  # noqa: E402
from step2_state_edit import prompt_ids, rhymes  # noqa: E402
from step34_routes import boot, strip_last_word  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen3-1.7B")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    tok, m, layers = load(a.model)
    rows = json.loads((EXP / "results" / a.model / "step2_rows.json").read_text())[: a.limit or None]

    def rhyme_ids(word):
        return sorted({t[0] for w in rhymes(word) for t in [tok.encode(" " + w, add_special_tokens=False)] if len(t) == 1})

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

    out_rows, t0 = [], time.time()
    for r in rows:
        p0 = strip_last_word(r["gen_off"])
        dr, orr = rhyme_ids(r["donor_word"]), rhyme_ids(r["orig_word"])
        if not p0 or not dr or not orr:
            continue
        pids, anchor, text = prompt_ids(tok, r["first_line"])
        _, danchor, dtext = prompt_ids(tok, r["donor_first_line"])
        _, dsts = run(tok(dtext, return_tensors="pt", add_special_tokens=False).input_ids.to("mps"), want_states=True)
        donor = [s[danchor] for s in dsts]
        ids = tok(text + p0, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
        toks = tok.convert_ids_to_tokens(ids[0])
        R = lambda p: math.log(max(float(p[dr].sum()), 1e-12)) - math.log(max(float(p[orr].sum()), 1e-12))  # noqa: E731
        y_off, _ = run(ids)
        y_on, s_on = run(ids, {anchor: donor}, want_states=True)
        tail = list(range(anchor + 1, pids.shape[1]))
        rec = {"idx": r["idx"], "tail_tokens": [toks[q] for q in tail], "d_persistence_p0": R(y_on) - R(y_off), "by_token": []}
        for q in tail:
            rec["by_token"].append(R(run(ids, {q: [s[q] for s in s_on]})[0]) - R(y_off))
        rec["boundary_all"] = R(run(ids, {q: [s[q] for s in s_on] for q in tail})[0]) - R(y_off)
        out_rows.append(rec)
        print(f"{len(out_rows):3d} [{time.time()-t0:5.0f}s] all {rec['boundary_all']:+.2f} | "
              + " ".join(f"{t}:{v:+.2f}" for t, v in zip(rec["tail_tokens"], rec["by_token"])), flush=True)
    out = EXP / "results" / a.model
    (out / "relay_boundary_rows.json").write_text(json.dumps(out_rows, indent=1))
    ref = out_rows[0]["tail_tokens"]
    same = [x for x in out_rows if x["tail_tokens"] == ref]
    s = {"model": a.model, "n": len(out_rows), "n_same_tail": len(same), "tail_tokens": ref,
         "boundary_all": boot([x["boundary_all"] for x in out_rows]),
         "by_token": [{"offset": k + 1, "token": t, **boot([x["by_token"][k] for x in same])} for k, t in enumerate(ref)]}
    (out / "relay_boundary_summary.json").write_text(json.dumps(s, indent=1))
    print("boundary all", s["boundary_all"])
    for d in s["by_token"]:
        print(f"  +{d['offset']:<2d} {d['token']!r:>16} {d['mean']:+.2f} [{d['lo']:.2f}, {d['hi']:.2f}]")


if __name__ == "__main__":
    main()
