#!/usr/bin/env python3
"""A/B route assay on Qwen3-4B: does mediator leverage set the public share?

Design frozen 2026-09-25 before any result (see README.md). Uses the frozen A/B
preflight config, banks 100% ("high") and 75% ("medium") only: the 50% bank
failed the gate's support check, a disclosed deviation approved by the user.
Each held-out family is run with the original labels and with A/B swapped.

Upstream change: casual (source) -> formal (target) context, as
  text:  the context text itself is changed (private part = reading visible text)
  state: casual text kept; at the pre-code position the hidden states of every
         layer are replaced by those at the formal prompt's pre-code position
         (private part = carried-forward state)
Six cells: change off/on x {free, do(A), do(B)}; outcome y = P(formal term) -
P(common term) at the first term token, and the full next-token distribution.
With q = renormalized P(B) over {A, B} (tau = 1):
  public  = (q1 - q0) * (y0(B) - y0(A))          (exact; the leverage identity)
  private = q1*(y1(B) - y0(B)) + (1-q1)*(y1(A) - y0(A))
Primary prediction: public(100%) > public(75%) for both changes, paired over the
10 family x labeling units (exact sign-flip test), with private roughly equal.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.ab_fewshot_preflight.run import code_prompt, term_prompt  # noqa: E402

EXP = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "experiments/ab_fewshot_preflight/config.json").read_text())
MODEL = "Qwen3-4B"
BANKS = ("high", "medium")


def demonstrations(bank: str, swap: bool) -> str:
    blocks = []
    for demo, (c_code, f_code) in zip(CFG["demonstrations"], CFG["label_banks"][bank]):
        if swap:
            c_code, f_code = {"A": "B", "B": "A"}[c_code], {"A": "B", "B": "A"}[f_code]
        blocks += [f"Context: {demo['common_context']}\nCode: {c_code}\nTerm: {demo['common_term']}",
                   f"Context: {demo['formal_context']}\nCode: {f_code}\nTerm: {demo['formal_term']}"]
    return "\n\n".join(blocks)


def sign_flip(xs):
    n = len(xs)
    obs = sum(xs) / n
    tot = ge = 0
    for signs in itertools.product((1, -1), repeat=n):
        tot += 1
        ge += sum(s * x for s, x in zip(signs, xs)) / n >= obs - 1e-15
    return {"mean": obs, "one_sided_p": ge / tot, "n": n}


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"Qwen/{MODEL}")
    m = AutoModelForCausalLM.from_pretrained(f"Qwen/{MODEL}", dtype=torch.bfloat16).to("mps").eval()
    layers = m.model.layers
    ids = {c: tok.encode(" " + c, add_special_tokens=False)[0] for c in "AB"}

    def enc(text):
        return tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")

    @torch.no_grad()
    def states_last(text):
        out = m(enc(text), output_hidden_states=True)
        return [h[0, -1].clone() for h in out.hidden_states[1:]]  # layer outputs

    @torch.no_grad()
    def dist(text, patch=None):
        hooks = []
        if patch is not None:
            pos, sts = patch
            for li, layer in enumerate(layers):
                def fn(mod, inp, out, li=li):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h.clone()
                    h[0, pos] = sts[li].to(h.dtype)
                    return (h, *out[1:]) if isinstance(out, tuple) else h
                hooks.append(layer.register_forward_hook(fn))
        try:
            return torch.softmax(m(enc(text)).logits[0, -1].float(), -1).cpu()
        finally:
            for h in hooks:
                h.remove()

    rows = []
    for bank in BANKS:
        for swap in (False, True):
            prefix = demonstrations(bank, swap)
            formal_code = "A" if swap else "B"  # the code the demonstrations tie to formal register (high bank)
            for item in CFG["heldout"]:
                cid = tok.encode(" " + item["common_term"], add_special_tokens=False)[0]
                fid = tok.encode(" " + item["formal_term"], add_special_tokens=False)[0]
                src_code = code_prompt(prefix, item["source_context"])
                tgt_code = code_prompt(prefix, item["target_context"])
                pos = enc(src_code).shape[1] - 1
                tgt_states = states_last(tgt_code)

                def cells(context, patch):
                    cp = code_prompt(prefix, context)
                    d = dist(cp, patch)
                    pa, pb = float(d[ids["A"]]), float(d[ids["B"]])
                    ys = {}
                    for c in "AB":
                        t = dist(term_prompt(prefix, context, c), patch)
                        ys[c] = {"y": float(t[fid] - t[cid]), "dist": t}
                    return {"q_formal_code": (pb if formal_code == "B" else pa) / (pa + pb), "ab_mass": pa + pb, "ys": ys}

                off = cells(item["source_context"], None)
                on = {"text": cells(item["target_context"], None),
                      "state": cells(item["source_context"], (pos, tgt_states))}
                fc, cc = formal_code, ("A" if formal_code == "B" else "B")
                lev = off["ys"][fc]["y"] - off["ys"][cc]["y"]
                row = {"bank": bank, "swap": swap, "family": item["family"], "leverage": lev,
                       "off_q_formal_code": off["q_formal_code"], "off_ab_mass": off["ab_mass"]}
                for kind, c1 in on.items():
                    q0, q1 = off["q_formal_code"], c1["q_formal_code"]
                    public = (q1 - q0) * lev
                    private = q1 * (c1["ys"][fc]["y"] - off["ys"][fc]["y"]) + (1 - q1) * (c1["ys"][cc]["y"] - off["ys"][cc]["y"])
                    mix0 = q0 * off["ys"][fc]["dist"] + (1 - q0) * off["ys"][cc]["dist"]
                    mixp = q1 * off["ys"][fc]["dist"] + (1 - q1) * off["ys"][cc]["dist"]
                    mix1 = q1 * c1["ys"][fc]["dist"] + (1 - q1) * c1["ys"][cc]["dist"]
                    row[kind] = {"q_formal_code": q1, "ab_mass": c1["ab_mass"], "public": public, "private": private,
                                 "total": public + private,
                                 "tv_public": 0.5 * float((mixp - mix0).abs().sum()),
                                 "tv_private": 0.5 * float((mix1 - mixp).abs().sum())}
                rows.append(row)
                print(bank, swap, item["family"], f"lev {lev:+.3f}",
                      {k: (round(row[k]["public"], 3), round(row[k]["private"], 3)) for k in on}, flush=True)
    out = EXP / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    summary = {"model": MODEL, "banks": list(BANKS), "units_per_bank": sum(r["bank"] == "high" for r in rows)}
    for kind in ("text", "state"):
        by = {b: [r for r in rows if r["bank"] == b] for b in BANKS}
        key = lambda r: (r["swap"], r["family"])  # noqa: E731
        hi = {key(r): r for r in by["high"]}
        md = {key(r): r for r in by["medium"]}
        units = sorted(hi)
        summary[kind] = {
            b: {f: sum(r[kind][f] for r in by[b]) / len(by[b]) for f in ("public", "private", "total", "tv_public", "tv_private")}
            | {"leverage": sum(r["leverage"] for r in by[b]) / len(by[b])} for b in BANKS}
        summary[kind]["public_high_minus_medium"] = sign_flip([hi[u][kind]["public"] - md[u][kind]["public"] for u in units])
        summary[kind]["private_high_minus_medium"] = sign_flip([hi[u][kind]["private"] - md[u][kind]["private"] for u in units])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
