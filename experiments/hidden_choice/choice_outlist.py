#!/usr/bin/env python3
"""Hidden choice control: out-of-list donor (design frozen 2026-09-26, README "Control").

Original and donor lists are disjoint 4-fruit lists (all 8 fruits between them); the
donor's pick is absent from the original prompt. Donor states (all layers) at the
post-list block and at the list positions; text swap for reference. R_out = log
p(donor pick) - log p(original pick) at the reveal, relative to no edit. Prompts align
token by token because every fruit is a single token and the lists have equal length.
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
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP.parents[0] / "couplet_routes"))
sys.path.insert(0, str(EXP.parents[0] / "derived_value_carry"))
from chain_routes import F32Head  # noqa: E402
from choice_routes import FRUITS, SEED, build  # noqa: E402
from models import is_gemma, load  # noqa: E402
from step34_routes import boot  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen3-8B")
    ap.add_argument("--pairs", type=int, default=100)
    a = ap.parse_args()
    tok, m, layers = load(a.model)
    m.lm_head = F32Head(m.lm_head.weight)
    gemma = is_gemma(a.model)
    fid = {f: tok.encode(" " + f, add_special_tokens=False)[0] for f in FRUITS}

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
            out = m(ids.to("mps"), output_hidden_states=want_states)
        finally:
            for h in hooks:
                h.remove()
        return torch.log_softmax(out.logits[0, -1].float(), -1).cpu(), ([h[0] for h in out.hidden_states[1:]] if want_states else None)

    rng = random.Random(SEED + 1)
    rows, t0 = [], time.time()
    for _ in range(a.pairs):
        allf = rng.sample(FRUITS, 8)
        orig, don = allf[:4], allf[4:]
        ids, lpos, post, sent, target = build(tok, orig, gemma)
        dids, dl, dp, ds, dt = build(tok, don, gemma)
        if ids.shape != dids.shape or (lpos, post, sent, target) != (dl, dp, ds, dt):
            continue
        y0, _ = run(ids)
        yd, dst = run(dids, want_states=True)
        po = max(orig, key=lambda f: float(y0[fid[f]]))
        pdn = max(don, key=lambda f: float(yd[fid[f]]))
        R = lambda y: float(y[fid[pdn]] - y[fid[po]])  # noqa: E731
        rank = lambda y: 1 + sum(float(y[fid[f]]) > float(y[fid[pdn]]) for f in FRUITS)  # noqa: E731
        D = lambda ps: {p: [s[p] for s in dst] for p in ps}  # noqa: E731
        R0 = R(y0)
        y_post, _ = run(ids, D(post))
        rec = {"list": orig, "donor_list": don, "pick": po, "donor_pick": pdn, "R_off": R0,
               "text_swap": R(yd) - R0, "list_edit": R(run(ids, D(lpos))[0]) - R0, "post_edit": R(y_post) - R0,
               "donor_rank_off": rank(y0), "donor_rank_post": rank(y_post)}
        rows.append(rec)
        print(f"{len(rows):3d} [{time.time()-t0:5.0f}s] {po} vs out-of-list {pdn}: text {rec['text_swap']:+.2f} "
              f"list {rec['list_edit']:+.2f} post {rec['post_edit']:+.2f} rank {rec['donor_rank_off']}->{rec['donor_rank_post']}", flush=True)
    out = EXP / "results" / a.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "choice_outlist_rows.json").write_text(json.dumps(rows, indent=1))
    s = {"model": a.model, "n": len(rows), **{k: boot([r[k] for r in rows]) for k in ("text_swap", "list_edit", "post_edit")},
         "donor_rank_off_mean": sum(r["donor_rank_off"] for r in rows) / len(rows),
         "donor_rank_post_mean": sum(r["donor_rank_post"] for r in rows) / len(rows)}
    (out / "choice_outlist_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: (round(v["mean"], 2), round(v["lo"], 2), round(v["hi"], 2)) if isinstance(v, dict) else v for k, v in s.items()}))


if __name__ == "__main__":
    main()
