#!/usr/bin/env python3
"""Couplet routes, steps 3-4: emission vs persistence, and the persistence path split.

Uses step-2 rows (same couplets, donors and anchor). For each couplet:
  p0 = line 2 generated without the edit, minus its last word
  p1 = line 2 generated with the edit, minus its last word
The target is the next-token distribution at the end of the prefix (the first
token of line 2's last word). Cells, full recomputation, no cache:
  Y(off,p0), Y(on,p0), Y(off,p1), Y(on,p1)
  emission          = Y(off,p1) - Y(off,p0)   (the edit's line-2 words, edit off)
  persistence       = Y(on,p1)  - Y(off,p1)   (edit on, words fixed)   [forward order]
  persistence_p0    = Y(on,p0)  - Y(off,p0)   (edit on, original words) [reverse order]
Path split of persistence_p0 (hidden states replaced at all layers):
  retrieval_only: edit on at the anchor; positions after the anchor and before the
                  target get their clean (edit-off) states -> only the target reads
                  the edited anchor directly.
  relay_only:     anchor clean; positions after the anchor and before the target get
                  their states from the edited run -> the target reads only what the
                  in-between positions carried.
Outcome: rhyme preference R = log mass(donor rhymes) - log mass(original rhymes)
over single-token rhyme words; also TV. Additivity: retrieval + relay vs
persistence_p0 (the gap is interaction).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
from step2_state_edit import prompt_ids, rhymes  # noqa: E402


def boot(xs, n=10000, seed=20260925):
    xs = [x for x in xs if x == x]
    if not xs:
        return None
    rng = random.Random(seed)
    bs = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return {"mean": sum(xs) / len(xs), "lo": bs[int(0.025 * n)], "hi": bs[int(0.975 * n) - 1], "n": len(xs)}


def strip_last_word(line: str) -> str | None:
    s = line.rstrip()
    s = s.rstrip(".,;:!?\"'")
    i = s.rstrip().rfind(" ")
    return s[:i] if i > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen3-1.7B")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--control", choices=["same_rhyme"])
    a = ap.parse_args()
    tag = f"_{a.control}" if a.control else ""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"Qwen/{a.model}")
    m = AutoModelForCausalLM.from_pretrained(f"Qwen/{a.model}", dtype=torch.bfloat16).to("mps").eval()
    layers = m.model.layers
    rows = json.loads((EXP / "results" / a.model / f"step2_rows{tag}.json").read_text())[: a.limit or None]

    def rhyme_ids(word):
        ids = set()
        for w in rhymes(word):
            t = tok.encode(" " + w, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
        return sorted(ids)

    @torch.no_grad()
    def run(ids, replace=None, want_states=False):
        """replace: {position: [per-layer state]} applied at every layer output."""
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
        p = torch.softmax(out.logits[0, -1].float(), -1).cpu()
        sts = [h[0] for h in out.hidden_states[1:]] if want_states else None
        return p, sts

    out_rows = []
    t0 = time.time()
    for r in rows:
        p0, p1 = strip_last_word(r["gen_off"]), strip_last_word(r["gen_on"])
        if not p0 or not p1:
            continue
        base, anchor, text = prompt_ids(tok, r["first_line"])
        _, danchor, dtext = prompt_ids(tok, r["donor_first_line"])
        dids = tok(dtext, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
        _, dsts = run(dids, want_states=True)
        donor_anchor = [s[danchor] for s in dsts]
        ids0 = tok(text + p0, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
        ids1 = tok(text + p1, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
        edit = {anchor: donor_anchor}
        y_off_p0, s_off_p0 = run(ids0, want_states=True)
        y_on_p0, s_on_p0 = run(ids0, edit, want_states=True)
        y_off_p1, _ = run(ids1)
        y_on_p1, _ = run(ids1, edit)
        mids = range(anchor + 1, ids0.shape[1] - 1)  # positions between anchor and target
        y_ret, _ = run(ids0, {anchor: donor_anchor} | {q: [s[q] for s in s_off_p0] for q in mids})
        y_rel, _ = run(ids0, {q: [s[q] for s in s_on_p0] for q in mids})
        dr, orr = rhyme_ids(r["donor_word"]), rhyme_ids(r["orig_word"])
        if not dr or not orr:
            continue

        def R(p):
            return math.log(max(float(p[dr].sum()), 1e-12)) - math.log(max(float(p[orr].sum()), 1e-12))

        def tv(p, q):
            return 0.5 * float((p - q).abs().sum())

        rec = {"idx": r["idx"], "orig_word": r["orig_word"], "donor_word": r["donor_word"],
               "prefix_off": p0, "prefix_on": p1, "n_between": len(mids),
               "R_off_p0": R(y_off_p0), "R_on_p0": R(y_on_p0), "R_off_p1": R(y_off_p1), "R_on_p1": R(y_on_p1),
               "R_ret": R(y_ret), "R_rel": R(y_rel),
               "tv_total": tv(y_on_p1, y_off_p0), "tv_emission": tv(y_off_p1, y_off_p0),
               "tv_persistence": tv(y_on_p1, y_off_p1), "tv_persistence_p0": tv(y_on_p0, y_off_p0),
               "edit_succeeded": bool(r["on_rhymes_donor"])}
        for k, (x, y) in {"total": ("R_on_p1", "R_off_p0"), "emission": ("R_off_p1", "R_off_p0"),
                          "persistence": ("R_on_p1", "R_off_p1"), "persistence_p0": ("R_on_p0", "R_off_p0"),
                          "retrieval": ("R_ret", "R_off_p0"), "relay": ("R_rel", "R_off_p0")}.items():
            rec[f"d_{k}"] = rec[x] - rec[y]
        out_rows.append(rec)
        print(f"{len(out_rows):3d} [{time.time()-t0:5.0f}s] total {rec['d_total']:+.2f} emis {rec['d_emission']:+.2f} "
              f"pers {rec['d_persistence']:+.2f} | p0-pers {rec['d_persistence_p0']:+.2f} = ret {rec['d_retrieval']:+.2f} + rel {rec['d_relay']:+.2f}", flush=True)
    out = EXP / "results" / a.model
    (out / f"step34_rows{tag}.json").write_text(json.dumps(out_rows, indent=1))
    keys = ["d_total", "d_emission", "d_persistence", "d_persistence_p0", "d_retrieval", "d_relay",
            "tv_total", "tv_emission", "tv_persistence", "tv_persistence_p0"]
    s = {"model": a.model, "n": len(out_rows), "elapsed_sec": time.time() - t0,
         "all": {k: boot([r[k] for r in out_rows]) for k in keys},
         "edit_succeeded": {k: boot([r[k] for r in out_rows if r["edit_succeeded"]]) for k in keys},
         "additivity_gap_p0": boot([r["d_persistence_p0"] - r["d_retrieval"] - r["d_relay"] for r in out_rows])}
    (out / f"step34_summary{tag}.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: (round(v["mean"], 3), round(v["lo"], 3), round(v["hi"], 3)) if v else None for k, v in s["all"].items()}, indent=1))
    print("edit-succeeded subset n =", s["edit_succeeded"]["d_total"]["n"] if s["edit_succeeded"]["d_total"] else 0)


if __name__ == "__main__":
    main()
