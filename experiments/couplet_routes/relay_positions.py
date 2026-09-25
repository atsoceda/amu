#!/usr/bin/env python3
"""Position-resolved relay: where between the anchor and the rhyme word is relay carried?

Design frozen 2026-09-25 before any result (README, "Scale extension", criterion 2).
Same couplets, donors, prompt, state edit and outcome as step34_routes.py, original
line-2 words fixed (prefix p0). Relay (step 3/4) gives every position between the
anchor and the target its edited-run state while the anchor stays clean. Here only
one group of those positions gets its edited-run state; all others stay clean:
  late   the last 3 positions before the target (where the authors' circuit fetches
         rhyme features into line 2 and keeps them active)
  far    every other in-between position (prompt tail after the anchor and line 2
         up to 4 tokens before the target): long-range carrying
  tail   prompt positions after the anchor only (line-1 punctuation, end of turn,
         assistant header)
  early  line-2 positions more than 3 tokens before the target
Groups are not a partition of paths: positions outside the edited group are
recomputed and can read it (e.g. with only the tail edited, late line-2 positions
read the tail), so late and far overlap on paths through both.
--control same_rhyme repeats it with same-rhyme donors (null).
Outcome R = log mass on donor rhymes - log mass on original rhymes at the target.
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
    ap.add_argument("--control", choices=["same_rhyme"], help="null: donor from the same rhyme group (step2 --control rows)")
    a = ap.parse_args()
    tag = f"_{a.control}" if a.control else ""
    tok, m, layers = load(a.model)
    rows = json.loads((EXP / "results" / a.model / f"step2_rows{tag}.json").read_text())[: a.limit or None]

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
        y_off, _ = run(ids)
        y_on, s_on = run(ids, {anchor: donor}, want_states=True)
        mids = list(range(anchor + 1, target))
        late = [q for q in mids if q >= target - LATE]
        groups = {"all": mids, "late": late, "far": [q for q in mids if q not in late],
                  "tail": [q for q in mids if q < n_prompt], "early": [q for q in mids if q >= n_prompt and q not in late]}
        rec = {"idx": r["idx"], "n_between": len(mids), "n_line2_prefix": target - n_prompt + 1,
               "sizes": {g: len(v) for g, v in groups.items()}, "d_persistence_p0": R(y_on) - R(y_off)}
        for g, qs in groups.items():
            rec[f"d_relay_{g}"] = (R(run(ids, {q: [s[q] for s in s_on] for q in qs})[0]) - R(y_off)) if qs else 0.0
        out_rows.append(rec)
        print(f"{len(out_rows):3d} [{time.time()-t0:5.0f}s] pers {rec['d_persistence_p0']:+.2f} relay all {rec['d_relay_all']:+.2f} "
              f"late {rec['d_relay_late']:+.2f} far {rec['d_relay_far']:+.2f} (tail {rec['d_relay_tail']:+.2f} early {rec['d_relay_early']:+.2f})", flush=True)
    out = EXP / "results" / a.model
    (out / f"relay_positions_rows{tag}.json").write_text(json.dumps(out_rows, indent=1))
    keys = ["d_persistence_p0"] + [f"d_relay_{g}" for g in ("all", "late", "far", "tail", "early")]
    s = {"model": a.model, "n": len(out_rows), "late_positions": LATE, "elapsed_sec": time.time() - t0,
         "all": {k: boot([x[k] for x in out_rows]) for k in keys},
         "far_minus_late": boot([x["d_relay_far"] - x["d_relay_late"] for x in out_rows])}
    (out / f"relay_positions_summary{tag}.json").write_text(json.dumps(s, indent=1))
    print({k: (round(v["mean"], 2), round(v["lo"], 2), round(v["hi"], 2)) for k, v in s["all"].items() if v})


if __name__ == "__main__":
    main()
