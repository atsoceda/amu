#!/usr/bin/env python3
"""Derived value, stage 3: written chains, emission vs persistence (design frozen
2026-09-26, README "Stage 3").

The model writes the chain (greedy, thinking off). The written text is cut after the
line that computes the next-to-last variable and followed by "\\nThe answer is ", so
the answer needs one more addition from the last *written* value. Donor state (all
layers) at the v0 digits. text0 / text1 = the chain written without / with the edit.
Cells R(off|on, text0|text1); R = log p(donor first digit) - log p(original first
digit) at the answer position. Output projection in float32 (see chain_routes.py).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP.parent / "couplet_routes"))
from chain_routes import NAMES, F32Head, make_items  # noqa: E402
from batching import generate_batch, run_edits  # noqa: E402
from models import is_gemma, load  # noqa: E402
from step34_routes import boot  # noqa: E402

INSTR = "Compute each variable in order, one per line, then state the value of {last}."


def prompt_text(tok, v0, inc, gemma):
    lines = [f"a = {v0}"] + [f"{NAMES[i + 1]} = {NAMES[i]} + {d}" for i, d in enumerate(inc)]
    q = "\n".join(lines) + "\n" + INSTR.format(last=NAMES[len(inc)])
    msgs = [{"role": "user", "content": q if gemma else "/no_think " + q}]
    kw = {} if gemma else {"enable_thinking": False}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw), lines


def cut_chain(text, K):
    """Written text up to the end of the line computing the next-to-last variable, and its value."""
    name = NAMES[K - 1]
    pos = 0
    for line in text.split("\n"):
        end = pos + len(line)
        if re.search(rf"\b{name}\s*=", line):
            nums = re.findall(r"=\s*\$?\s*(\d+)\s*\$?\s*$", line.strip().rstrip(".").rstrip())
            if nums:
                return text[:end], int(nums[-1])
        pos = end + 1
    return None, None


def batched_K(K, a, tok, m, layers, gemma, digit, states):
    """Same items, generations and cells as the one-at-a-time path, with generation batched."""
    prep = []
    for it in make_items(K)[: a.limit or None]:
        text, _ = prompt_text(tok, it["v0"], it["inc"], gemma)
        dtext, _ = prompt_text(tok, it["dv0"], it["inc"], gemma)
        enc = tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        dids = tok(dtext, return_tensors="pt", add_special_tokens=False).input_ids
        if enc.input_ids.shape != dids.shape:
            continue
        offs = enc.offset_mapping[0].tolist()
        s0 = text.index("a = ") + 4
        v0pos = [i for i, (s, e) in enumerate(offs) if s0 <= s < s0 + 2 and e <= s0 + 2]
        dst = states(dids)
        prep.append((it, text, {p: [x[p] for x in dst] for p in v0pos}))
    texts = [t for _, t, _ in prep]
    g0s = generate_batch(m, tok, layers, texts, 240, chunk=a.batch)
    g1s = generate_batch(m, tok, layers, texts, 240, patches=[pt for _, _, pt in prep], chunk=a.batch)
    rows = []
    for (it, text, patch), g0, g1 in zip(prep, g0s, g1s):
        c0, val0 = cut_chain(g0, K)
        c1, val1 = cut_chain(g1, K)
        true_prev = it["v0"] + sum(it["inc"][:-1])
        rec = {**it, "gen_off": g0, "gen_on": g1, "written_prev_off": val0, "written_prev_on": val1,
               "true_prev": true_prev, "donor_prev": it["dv0"] + sum(it["inc"][:-1]), "usable": bool(c0 and c1 and val0 == true_prev)}
        if rec["usable"]:
            o, dn = digit[it["final"] // 10], digit[it["dfinal"] // 10]
            R = lambda y: float(y[dn] - y[o])  # noqa: E731
            t0ids = tok(text + c0 + "\nThe answer is ", return_tensors="pt", add_special_tokens=False).input_ids
            t1ids = tok(text + c1 + "\nThe answer is ", return_tensors="pt", add_special_tokens=False).input_ids
            (y00, y01), _ = run_edits(m, layers, t0ids, [{}, patch])
            (y10, y11), _ = run_edits(m, layers, t1ids, [{}, patch])
            v = {"off_t0": R(y00), "on_t1": R(y11), "off_t1": R(y10), "on_t0": R(y01)}
            rec.update(v)
            rec.update({"total": v["on_t1"] - v["off_t0"], "emission": v["off_t1"] - v["off_t0"],
                        "persistence_t1": v["on_t1"] - v["off_t1"], "persistence_t0": v["on_t0"] - v["off_t0"],
                        "text_changed": c0 != c1})
        rows.append(rec)
    print(f"K={K}: {len(rows)} items done (batched)", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen3-4B")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ks", default="3,5")
    ap.add_argument("--batch", type=int, default=16, help="items per generation batch (1 = original one-at-a-time path)")
    a = ap.parse_args()
    tok, m, layers = load(a.model)
    m.lm_head = F32Head(m.lm_head.weight)
    gemma = is_gemma(a.model)
    digit = {d: tok.convert_tokens_to_ids(str(d)) for d in range(10)}

    def hooks_for(patch, positions, prompt_only):
        hs = []
        for li, layer in enumerate(layers):
            def fn(mod, inp, out, li=li):
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[1] == 1 or (prompt_only and h.shape[1] <= max(positions)):
                    return out
                h = h.clone()
                for p in positions:
                    h[0, p] = patch[p][li].to(h.dtype)
                return (h, *out[1:]) if isinstance(out, tuple) else h
            hs.append(layer.register_forward_hook(fn))
        return hs

    @torch.no_grad()
    def states(ids):
        return [h[0] for h in m(ids.to("mps"), output_hidden_states=True).hidden_states[1:]]

    @torch.no_grad()
    def generate(ids, patch=None, positions=()):
        hs = hooks_for(patch, positions, True) if patch else []
        try:
            out = m.generate(ids.to("mps"), max_new_tokens=240, do_sample=False)
        finally:
            for h in hs:
                h.remove()
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def logp(ids, patch=None, positions=()):
        hs = hooks_for(patch, positions, False) if patch else []
        try:
            out = m(ids.to("mps"))
        finally:
            for h in hs:
                h.remove()
        return torch.log_softmax(out.logits[0, -1].float(), -1).cpu()

    out_dir = EXP / "results" / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, t0 = {"model": a.model, "by_K": {}}, time.time()
    for K in map(int, a.ks.split(",")):
        rows = []
        if a.batch > 1:
            rows = batched_K(K, a, tok, m, layers, gemma, digit, states)
        for it in ([] if a.batch > 1 else make_items(K)[: a.limit or None]):
            text, lines = prompt_text(tok, it["v0"], it["inc"], gemma)
            dtext, _ = prompt_text(tok, it["dv0"], it["inc"], gemma)
            ids = tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
            dids = tok(dtext, return_tensors="pt", add_special_tokens=False).input_ids
            offs = ids.offset_mapping[0].tolist()
            ids = ids.input_ids
            if ids.shape != dids.shape:
                continue
            s0 = text.index("a = ") + 4
            v0pos = [i for i, (s, e) in enumerate(offs) if s0 <= s < s0 + 2 and e <= s0 + 2]
            dst = states(dids)
            patch = {p: [s[p] for s in dst] for p in v0pos}
            g0 = generate(ids)
            g1 = generate(ids, patch, v0pos)
            c0, val0 = cut_chain(g0, K)
            c1, val1 = cut_chain(g1, K)
            true_prev = it["v0"] + sum(it["inc"][:-1])
            rec = {**it, "gen_off": g0, "gen_on": g1, "written_prev_off": val0, "written_prev_on": val1,
                   "true_prev": true_prev, "donor_prev": it["dv0"] + sum(it["inc"][:-1]), "usable": bool(c0 and c1 and val0 == true_prev)}
            if rec["usable"]:
                o, dn = digit[it["final"] // 10], digit[it["dfinal"] // 10]
                t0ids = tok(text + c0 + "\nThe answer is ", return_tensors="pt", add_special_tokens=False).input_ids
                t1ids = tok(text + c1 + "\nThe answer is ", return_tensors="pt", add_special_tokens=False).input_ids
                R = lambda y: float(y[dn] - y[o])  # noqa: E731
                v = {"off_t0": R(logp(t0ids)), "on_t1": R(logp(t1ids, patch, v0pos)),
                     "off_t1": R(logp(t1ids)), "on_t0": R(logp(t0ids, patch, v0pos))}
                rec.update(v)
                rec.update({"total": v["on_t1"] - v["off_t0"], "emission": v["off_t1"] - v["off_t0"],
                            "persistence_t1": v["on_t1"] - v["off_t1"], "persistence_t0": v["on_t0"] - v["off_t0"],
                            "text_changed": c0 != c1})
            rows.append(rec)
            print(f"K={K} {len(rows):3d} [{time.time()-t0:5.0f}s] prev written off {val0} on {val1} (true {true_prev}, donor "
                  f"{rec['donor_prev']}) " + (f"total {rec['total']:+.2f} emis {rec['emission']:+.2f} pers(t0) {rec['persistence_t0']:+.2f} "
                  f"pers(t1) {rec['persistence_t1']:+.2f}" if rec["usable"] else "unusable"), flush=True)
        (out_dir / f"written_K{K}_rows.json").write_text(json.dumps(rows, indent=1))
        u = [r for r in rows if r["usable"]]
        summary["by_K"][K] = {"n": len(rows), "n_usable": len(u),
                              "on_writes_donor_prev": sum(r["written_prev_on"] == r["donor_prev"] for r in rows) / len(rows),
                              **{k: boot([r[k] for r in u]) for k in ("total", "emission", "persistence_t0", "persistence_t1")}}
        s = summary["by_K"][K]
        print(f"K={K}: usable {len(u)}/{len(rows)}; edited run writes donor values {s['on_writes_donor_prev']:.0%}; "
              + " ".join(f"{k} {s[k]['mean']:+.2f} [{s[k]['lo']:.2f}, {s[k]['hi']:.2f}]" for k in ("total", "emission", "persistence_t0", "persistence_t1") if s[k]))
    (out_dir / "written_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
