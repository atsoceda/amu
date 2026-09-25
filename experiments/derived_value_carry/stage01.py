#!/usr/bin/env python3
"""Derived-value carry, stages 0 and 1 (gates before any route measurement).

Format (fixed at stage 0, see README): user turn "/no_think What is {a} + {b}?"
(Qwen3 chat template, thinking off), identical in every condition; the assistant
turn is prefilled with "{filler} The answer is " (filler omitted at length 0), so
the next token is the answer's first digit (Qwen3 tokenizes digits singly, and
the space before a number is its own token, which ends the prefill).

Items: two-digit operands with a two-digit sum, 13 per tens digit of the sum
(2-9), each with a donor whose sum has a different tens digit (seed 20260925).
Operands are always two tokens, so original and donor prompts align position by
position.

Stage 0: greedy answer accuracy with no edit, per filler length (gate: >= 80%).
Stage 1: at filler length 0, the donor's state (all layers) placed at one
position of the original prompt; outcome R = log p(donor first digit) - log
p(original first digit) at the target, minus R without the edit. Scanned over
every position from the first operand to the token before the target, plus the
four operand tokens together, plus every position from the question mark to the
token before the target together (does the post-question block carry the sum
collectively?). Gate: the pre-registered computation position (the
question mark) moves R clearly above zero.
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

SEED = 20260925
PASSAGE = ("The weather is calm and the sky is a pale shade of blue over the quiet town where people walk "
           "slowly along the river and talk about ordinary things like gardens and books and the small cafe "
           "near the old bridge while birds rest on the rooftops and the afternoon light fades softly")
LENGTHS = (0, 8, 16, 32)


def make_items(n_per_digit=13):
    rng = random.Random(SEED)

    def draw(t):
        while True:
            a = rng.randint(10, 89)
            b = 10 * t + rng.randint(0, 9) - a
            if 10 <= b <= 89:
                return a, b
    items = []
    for t in range(2, 10):
        for _ in range(n_per_digit):
            a, b = draw(t)
            dt = rng.choice([d for d in range(2, 10) if d != t])
            da, db = draw(dt)
            items.append({"a": a, "b": b, "sum": a + b, "da": da, "db": db, "dsum": da + db})
    return items


def filler(tok, n):
    """Longest word prefix of PASSAGE with at most n tokens (as it appears after the template)."""
    if n == 0:
        return ""
    words, best = PASSAGE.split(), ""
    for k in range(1, len(words) + 1):
        f = " ".join(words[:k]) + "."
        if len(tok(f, add_special_tokens=False).input_ids) > n:
            break
        best = f
    return best


def build(tok, a, b, fill):
    msgs = [{"role": "user", "content": f"/no_think What is {a} + {b}?"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    text += (fill + " " if fill else "") + "The answer is "
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    toks = tok.convert_ids_to_tokens(ids[0])
    qmark = max(i for i, t in enumerate(toks) if "?" in t)
    digits = [i for i, t in enumerate(toks[:qmark]) if t.isdigit()]
    return ids, qmark, digits, toks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen3-1.7B")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    tok, m, layers = load(a.model)
    items = make_items()[: a.limit or None]
    digit_id = {d: tok.convert_tokens_to_ids(str(d)) for d in range(10)}
    fills = {n: filler(tok, n) for n in LENGTHS}

    @torch.no_grad()
    def run(ids, replace=None, want_states=False):
        hooks = []
        if replace:
            for li, layer in enumerate(layers):
                def fn(mod, inp, out, li=li):
                    h = out[0] if isinstance(out, tuple) else out
                    if h.shape[1] == 1:  # cached generation step
                        return out
                    h = h.clone()
                    for pos, sts in replace.items():
                        h[0, pos] = sts[li].to(h.dtype)
                    return (h, *out[1:]) if isinstance(out, tuple) else h
                hooks.append(layer.register_forward_hook(fn))
        try:
            out = m(ids.to("mps"), output_hidden_states=want_states)
        finally:
            for h in hooks:
                h.remove()
        return torch.log_softmax(out.logits[0, -1].float(), -1).cpu(), ([h[0] for h in out.hidden_states[1:]] if want_states else None)

    @torch.no_grad()
    def answer(ids):
        out = m.generate(ids.to("mps"), max_new_tokens=4, do_sample=False)
        txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        num = "".join(c for c in txt.strip().split(" ")[0] if c.isdigit())
        return int(num) if num else None

    t0 = time.time()
    stage0 = {}
    for n, f in fills.items():
        ok = [answer(build(tok, it["a"], it["b"], f)[0]) == it["sum"] for it in items]
        stage0[n] = {"filler": f, "n_tokens": len(tok(f, add_special_tokens=False).input_ids) if f else 0,
                     "accuracy": sum(ok) / len(ok)}
        print(f"stage 0 filler {n:2d} ({stage0[n]['n_tokens']} tokens): accuracy {stage0[n]['accuracy']:.0%}", flush=True)

    rows = []
    for it in items:
        ids, q, digs, toks = build(tok, it["a"], it["b"], "")
        dids, dq, ddigs, _ = build(tok, it["da"], it["db"], "")
        assert ids.shape == dids.shape and q == dq and digs == ddigs
        o, dn = digit_id[it["sum"] // 10], digit_id[it["dsum"] // 10]
        y0, _ = run(ids)
        _, dsts = run(dids, want_states=True)
        R0 = float(y0[dn] - y0[o])
        rec = {**it, "R_off": R0, "tokens": toks, "qmark": q, "operand_positions": digs, "by_position": {}}
        for p in range(digs[0], ids.shape[1] - 1):
            y, _ = run(ids, {p: [s[p] for s in dsts]})
            rec["by_position"][p] = float(y[dn] - y[o]) - R0
        y, _ = run(ids, {p: [s[p] for s in dsts] for p in digs})
        rec["operands"] = float(y[dn] - y[o]) - R0
        after = range(q, ids.shape[1] - 1)
        y, _ = run(ids, {p: [s[p] for s in dsts] for p in after})
        rec["after_question"] = float(y[dn] - y[o]) - R0
        rows.append(rec)
        print(f"{len(rows):3d} [{time.time()-t0:5.0f}s] {it['a']}+{it['b']} donor {it['da']}+{it['db']}: "
              f"'?' {rec['by_position'][q]:+.2f} after-? block {rec['after_question']:+.2f} operands {rec['operands']:+.2f}", flush=True)
    toks0 = rows[0]["tokens"]
    offs = sorted({p - r["qmark"] for r in rows for p in r["by_position"]})
    stage1 = {"qmark": boot([r["by_position"][r["qmark"]] for r in rows]), "operands": boot([r["operands"] for r in rows]),
              "after_question": boot([r["after_question"] for r in rows]),
              "by_offset_from_qmark": {o: {"token": toks0[rows[0]["qmark"] + o],
                                           **boot([r["by_position"][r["qmark"] + o] for r in rows])} for o in offs}}
    out = EXP / "results" / a.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage01_rows.json").write_text(json.dumps(rows, indent=1))
    (out / "stage01_summary.json").write_text(json.dumps({"model": a.model, "n": len(rows), "stage0": stage0, "stage1": stage1,
                                                          "elapsed_sec": time.time() - t0}, indent=1))
    print("stage 1 '?':", stage1["qmark"], "after-? block:", stage1["after_question"], "operands:", stage1["operands"])
    for o, v in stage1["by_offset_from_qmark"].items():
        print(f"  {o:+3d} {v['token']!r:>14} {v['mean']:+.2f} [{v['lo']:.2f}, {v['hi']:.2f}]")


if __name__ == "__main__":
    main()
