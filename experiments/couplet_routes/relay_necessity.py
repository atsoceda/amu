#!/usr/bin/env python3
"""Necessity split: how much persistence is lost when one downstream group is reset?

Analysis frozen 2026-09-26 before any result (README, "Necessity split"). Same
couplets, donors, state edit, outcome and original line-2 words as
relay_positions.py. With the donor state at the anchor, one group of in-between
positions gets its clean-run state (all layers); all other positions are
recomputed. necessity_g = persistence - (R(that run) - R(no edit)). Groups: late
(last 3 before the target), tail (prompt positions after the anchor: the line
boundary), early (line-2 positions before the late ones), all (so necessity_all =
persistence - retrieval). Sufficiency (relay_positions.py) and the Shapley average
(sufficiency + necessity) / 2 are reported alongside when available.
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

LATE = 3


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
        target, n_prompt = ids.shape[1] - 1, pids.shape[1]
        R = lambda p: math.log(max(float(p[dr].sum()), 1e-12)) - math.log(max(float(p[orr].sum()), 1e-12))  # noqa: E731
        y_off, s_off = run(ids, want_states=True)
        y_on, _ = run(ids, {anchor: donor})
        pers = R(y_on) - R(y_off)
        mids = list(range(anchor + 1, target))
        late = [q for q in mids if q >= target - LATE]
        groups = {"all": mids, "late": late, "tail": [q for q in mids if q < n_prompt],
                  "early": [q for q in mids if q >= n_prompt and q not in late]}
        rec = {"idx": r["idx"], "sizes": {g: len(v) for g, v in groups.items()}, "d_persistence_p0": pers}
        for g, qs in groups.items():
            rep = {anchor: donor} | {q: [s[q] for s in s_off] for q in qs}
            rec[f"necessity_{g}"] = pers - (R(run(ids, rep)[0]) - R(y_off)) if qs else 0.0
        out_rows.append(rec)
        print(f"{len(out_rows):3d} [{time.time()-t0:5.0f}s] pers {pers:+.2f} necessity all {rec['necessity_all']:+.2f} "
              f"late {rec['necessity_late']:+.2f} tail {rec['necessity_tail']:+.2f} early {rec['necessity_early']:+.2f}", flush=True)
    out = EXP / "results" / a.model
    (out / "relay_necessity_rows.json").write_text(json.dumps(out_rows, indent=1))
    keys = ["d_persistence_p0"] + [f"necessity_{g}" for g in ("all", "late", "tail", "early")]
    s = {"model": a.model, "n": len(out_rows), "all": {k: boot([x[k] for x in out_rows]) for k in keys}}
    suff = out / "relay_positions_rows.json"
    if suff.exists():
        srows = {x["idx"]: x for x in json.loads(suff.read_text())}
        both = [(x, srows[x["idx"]]) for x in out_rows if x["idx"] in srows]
        s["shapley"] = {g: boot([(x[f"necessity_{g}"] + y[f"d_relay_{g}"]) / 2 for x, y in both]) for g in ("all", "late", "tail", "early")}
        s["sufficiency"] = {g: boot([y[f"d_relay_{g}"] for _, y in both]) for g in ("all", "late", "tail", "early")}
    (out / "relay_necessity_summary.json").write_text(json.dumps(s, indent=1))
    print({k: (round(v["mean"], 2), round(v["lo"], 2), round(v["hi"], 2)) for k, v in s["all"].items()})
    if "shapley" in s:
        print("shapley", {g: round(v["mean"], 2) for g, v in s["shapley"].items()})


if __name__ == "__main__":
    main()
