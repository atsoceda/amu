#!/usr/bin/env python3
"""Derived value, stage 2: variable chains as a relay positive control.

Design frozen 2026-09-26 before any result (README, "Stage 2 redesign"). Prompt
`a = v0 / b = a + d1 / ... / What is <last>?` in the Qwen3 chat template (thinking
off), assistant prefill "The answer is "; target = first digit of the final value.
Donor: same increments, different two-digit v0 with a different final tens digit,
so prompts align token by token. Chain lengths K = 1, 3, 5.

Per item and K: stage 0 greedy accuracy; efficacy scan (donor state, all layers,
at each single position from v0 to the token before the target, plus the v0 digits
together); and for each statement end s_j (the newline ending statement j):
persistence, direct retrieval (positions after s_j clean), relay (s_j clean,
positions after s_j with edited-run states) and chain relay (only later statement
ends with edited-run states). Added 2026-09-26 (before the 14B/32B runs): the
post-v0 block, donor states at every position after v0's digits up to the token
before the target, which detects a running value carried redundantly downstream
(single-position edits cannot). R = log p(donor first digit) - log p(original first
digit) at the target, minus R without the edit.
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
from models import is_gemma, load  # noqa: E402
from step34_routes import boot  # noqa: E402

SEED = 20260925
NAMES = "abcdefgh"
KS = (1, 3, 5)


class F32Head(torch.nn.Module):
    """Output projection in float32 (a separate copy: small Qwen3 models tie it to the
    input embeddings). bf16 logits resolve differences only to about 0.125 nats, too
    coarse for small route effects."""

    def __init__(self, weight):
        super().__init__()
        self.register_buffer("w", weight.detach().float().clone())

    def forward(self, x):
        return x.float() @ self.w.T


def make_items(K, n_per_digit=12):
    """Balanced over the final tens digits reachable with v0 in 10-89 and finals <= 99."""
    rng = random.Random(SEED + K)
    buckets = {t: [] for t in range(1, 10)}
    for _ in range(200000):
        inc = [rng.randint(1, 9) for _ in range(K)]
        v0 = rng.randint(10, 89)
        final = v0 + sum(inc)
        t = final // 10
        if final > 99 or len(buckets[t]) >= n_per_digit:
            continue
        dv0s = [v for v in range(10, 90) if v + sum(inc) <= 99 and (v + sum(inc)) // 10 != t]
        dv = rng.choice(dv0s)
        buckets[t].append({"K": K, "v0": v0, "inc": inc, "final": final, "dv0": dv, "dfinal": dv + sum(inc)})
        if all(len(b) >= n_per_digit for b in buckets.values()):
            break
    return [it for t in sorted(buckets) for it in buckets[t]]


def prompt(tok, v0, inc, gemma):
    lines = [f"a = {v0}"] + [f"{NAMES[i + 1]} = {NAMES[i]} + {d}" for i, d in enumerate(inc)]
    q = "\n".join(lines) + f"\nWhat is {NAMES[len(inc)]}?"
    msgs = [{"role": "user", "content": q if gemma else "/no_think " + q}]
    kw = {} if gemma else {"enable_thinking": False}
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw) + "The answer is "
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    toks = tok.convert_ids_to_tokens(ids[0])
    body_start = text.index("a = ")
    offs = tok(text, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
    # statement ends: the newline token after each statement (the last statement ends at the newline before "What")
    ends, pos = [], body_start
    for ln in lines:
        nl = pos + len(ln)
        assert text[nl] == "\n"
        ends.append(next(i for i, (s, e) in enumerate(offs) if s <= nl < e))
        pos = nl + 1
    v0_start = next(i for i, (s, e) in enumerate(offs) if s <= body_start + 4 < e)
    v0_digits = [i for i in range(v0_start, ends[0]) if toks[i].strip("Ġ▁").isdigit()]
    return ids, toks, ends, v0_digits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen3-1.7B")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ks", default=",".join(map(str, KS)))
    ap.add_argument("--block-only", action="store_true", help="only the post-v0 block edit (added 2026-09-26)")
    a = ap.parse_args()
    tok, m, layers = load(a.model)
    m.lm_head = F32Head(m.lm_head.weight)
    gemma = is_gemma(a.model)
    digit = {d: tok.convert_tokens_to_ids(str(d)) for d in range(10)}

    @torch.no_grad()
    def run(ids, replace=None, want_states=False):
        hooks = []
        if replace:
            for li, layer in enumerate(layers):
                def fn(mod, inp, out, li=li):
                    h = out[0] if isinstance(out, tuple) else out
                    if h.shape[1] == 1:
                        return out
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

    @torch.no_grad()
    def answer(ids):
        out = m.generate(ids.to("mps"), max_new_tokens=4, do_sample=False)
        txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        num = "".join(c for c in txt.split(" ")[0] if c.isdigit())
        return int(num) if num else None

    out_dir = EXP / "results" / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, t0 = {"model": a.model, "by_K": {}}, time.time()
    for K in map(int, a.ks.split(",")):
        items = make_items(K)[: a.limit or None]
        rows = []
        for it in items:
            ids, toks, ends, v0d = prompt(tok, it["v0"], it["inc"], gemma)
            dids, _, dends, dv0d = prompt(tok, it["dv0"], it["inc"], gemma)
            assert ids.shape == dids.shape and ends == dends and v0d == dv0d
            target = ids.shape[1] - 1
            o, dn = digit[it["final"] // 10], digit[it["dfinal"] // 10]
            y0, s0 = run(ids, want_states=True)
            _, dst = run(dids, want_states=True)
            R0 = float(y0[dn] - y0[o])
            dR = lambda y: float(y[dn] - y[o]) - R0  # noqa: E731
            block = range(ends[0], target)  # everything after v0's digits, up to the token before the target
            rec = {**it, "correct": answer(ids) == it["final"], "tokens": toks, "ends": ends, "v0_digits": v0d,
                   "scan": {}, "v0_edit": dR(run(ids, {p: [s[p] for s in dst] for p in v0d})[0]), "by_end": [],
                   "post_v0_block": dR(run(ids, {p: [s[p] for s in dst] for p in block})[0])}
            if a.block_only:
                rows.append(rec)
                print(f"K={K} {len(rows):3d} [{time.time()-t0:5.0f}s] ok={rec['correct']} v0-edit {rec['v0_edit']:+.1f} "
                      f"post-v0 block {rec['post_v0_block']:+.2f}", flush=True)
                continue
            for p in range(v0d[0], target):
                rec["scan"][p] = dR(run(ids, {p: [s[p] for s in dst]})[0])
            for j, sj in enumerate(ends):
                y_on, s_on = run(ids, {sj: [s[sj] for s in dst]}, want_states=True)
                after = range(sj + 1, target)
                later = [e for e in ends if e > sj]
                d = {"j": j, "pos": sj, "persistence": dR(y_on),
                     "retrieval": dR(run(ids, {sj: [s[sj] for s in dst]} | {q: [s[q] for s in s0] for q in after})[0]),
                     "relay": dR(run(ids, {q: [s[q] for s in s_on] for q in after})[0]),
                     "chain_relay": dR(run(ids, {q: [s[q] for s in s_on] for q in later})[0]) if later else 0.0}
                rec["by_end"].append(d)
            rows.append(rec)
            print(f"K={K} {len(rows):3d} [{time.time()-t0:5.0f}s] ok={rec['correct']} v0-edit {rec['v0_edit']:+.1f} | "
                  + " ".join(f"s{d['j']}: pers {d['persistence']:+.2f} ret {d['retrieval']:+.2f} rel {d['relay']:+.2f} chain {d['chain_relay']:+.2f}"
                             for d in rec["by_end"]), flush=True)
        tag = "_block" if a.block_only else ""
        (out_dir / f"chain_K{K}_rows{tag}.json").write_text(json.dumps(rows, indent=1))
        if a.block_only:
            summary["by_K"][K] = {"n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
                                  "v0_edit": boot([r["v0_edit"] for r in rows]),
                                  "post_v0_block": boot([r["post_v0_block"] for r in rows])}
            print(f"K={K}: v0 edit {summary['by_K'][K]['v0_edit']}; post-v0 block {summary['by_K'][K]['post_v0_block']}")
            continue
        tq = rows[0]["tokens"]
        summary["by_K"][K] = {
            "n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "v0_edit": boot([r["v0_edit"] for r in rows]),
            "post_v0_block": boot([r["post_v0_block"] for r in rows]),
            "by_end": [{k: boot([r["by_end"][j][k] for r in rows]) for k in ("persistence", "retrieval", "relay", "chain_relay")}
                       for j in range(K + 1)],
            "scan": {p - rows[0]["v0_digits"][0]: {"token": tq[p], **boot([r["scan"][p] for r in rows])} for p in rows[0]["scan"]},
        }
        s = summary["by_K"][K]
        print(f"K={K}: accuracy {s['accuracy']:.0%}; v0 edit {s['v0_edit']['mean']:+.1f}")
        for j, d in enumerate(s["by_end"]):
            print(f"  s{j}: " + "  ".join(f"{k} {v['mean']:+.2f} [{v['lo']:.2f}, {v['hi']:.2f}]" for k, v in d.items()))
    summary["elapsed_sec"] = time.time() - t0
    (out_dir / f"chain_summary{'_block' if a.block_only else ''}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
