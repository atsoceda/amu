#!/usr/bin/env python3
"""Couplet routes, step 6b: route split under Hanna & Ameisen's rhyme-feature steering.

The edit is the authors' intervention (rhyme_intervention_sample.py): at the
line-1 anchor, the couplet's own rhyme features are set to -3x their activation
and the donor's rhyme features to 7x the donor's activation. It is applied as
sum_f (target_f - a_f) * W_dec[f] added to each layer's MLP output at the anchor
(per-layer transcoders, live error terms). Donors are the authors' chosen_index.

1. Generate line 2 with and without steering; score rhymes (validation against
   the authors' released intervention_generation rhyme rate).
2. Same cells and path split as step34_routes.py, with steering as the edit.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
from step2_state_edit import HA, last_word, prompt_ids, rhymes  # noqa: E402
from step34_routes import boot, strip_last_word  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen3-4B")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    fdir = EXP / "results" / a.model / "features"
    sel = json.loads((fdir / "selection.json").read_text())
    dec = torch.load(fdir / "decoders.pt")
    tok = AutoTokenizer.from_pretrained(f"Qwen/{a.model}")
    m = AutoModelForCausalLM.from_pretrained(f"Qwen/{a.model}", dtype=torch.bfloat16).to("mps").eval()
    layers = m.model.layers
    df = pd.read_csv(HA / f"{a.model}.csv", index_col=0)
    df = df[df["found_valid_row"].astype(str).str.lower() == "true"]
    if a.limit:
        df = df.head(a.limit)

    def steer_vectors(i, j):
        own = {(n["layer"], n["feature"]): n["act"] for n in sel[str(i)]}
        vec = {}
        for (l, f), act in own.items():
            vec[l] = vec.get(l, 0) + (-3 * act - act) * dec[(l, f)].float()
        for n in sel[str(j)]:
            l, f = n["layer"], n["feature"]
            vec[l] = vec.get(l, 0) + (7 * n["act"] - own.get((l, f), 0.0)) * dec[(l, f)].float()
        return {l: v.to("mps") for l, v in vec.items()}

    def hooks_for(steer, anchor, replace=None, prompt_only=False):
        hs = []
        if steer:
            for l, v in steer.items():
                def fn(mod, inp, out, v=v):
                    if prompt_only and out.shape[1] <= anchor:
                        return out
                    out = out.clone()
                    out[0, anchor] += v.to(out.dtype)
                    return out
                hs.append(layers[l].mlp.register_forward_hook(fn))
        if replace:
            for li, layer in enumerate(layers):
                def fr(mod, inp, out, li=li):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h.clone()
                    for pos, sts in replace.items():
                        h[0, pos] = sts[li].to(h.dtype)
                    return (h, *out[1:]) if isinstance(out, tuple) else h
                hs.append(layer.register_forward_hook(fr))
        return hs

    @torch.no_grad()
    def run(ids, anchor, steer=None, replace=None, want_states=False):
        hs = hooks_for(steer, anchor, replace)
        try:
            out = m(ids, output_hidden_states=want_states)
        finally:
            for h in hs:
                h.remove()
        return torch.softmax(out.logits[0, -1].float(), -1).cpu(), ([h[0] for h in out.hidden_states[1:]] if want_states else None)

    @torch.no_grad()
    def generate(ids, anchor, steer=None):
        hs = hooks_for(steer, anchor, prompt_only=True)
        try:
            out = m.generate(ids, max_new_tokens=24, do_sample=False)
        finally:
            for h in hs:
                h.remove()
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def rhyme_ids(word):
        return sorted({t[0] for w in rhymes(word) for t in [tok.encode(" " + w, add_special_tokens=False)] if len(t) == 1})

    all_rows = pd.read_csv(HA / f"{a.model}.csv", index_col=0)
    rows, t0 = [], time.time()
    for i, r in df.iterrows():
        j = int(r["chosen_index"])
        if not sel.get(str(i)) or not sel.get(str(j)):
            continue
        donor = all_rows.loc[j]
        steer = steer_vectors(i, j)
        base, anchor, text = prompt_ids(tok, r["first_line"])
        base = base.to("mps")
        g0, g1 = generate(base, anchor), generate(base, anchor, steer)
        authors = str(r["intervention_generation"])
        ow, dw = r["first_last_word"], donor["first_last_word"]
        rec = {"idx": int(i), "orig_word": ow, "donor_word": dw, "gen_off": g0, "gen_on": g1, "authors_steered": authors,
               "n_own_features": len(sel[str(i)]), "n_donor_features": len(sel[str(j)]),
               "on_rhymes_donor": last_word(g1) in rhymes(dw), "on_rhymes_orig": last_word(g1) in rhymes(ow),
               "off_rhymes_orig": last_word(g0) in rhymes(ow),
               "authors_steered_rhymes_donor": last_word(authors) in rhymes(dw)}
        p0, p1 = strip_last_word(g0), strip_last_word(g1)
        dr, orr = rhyme_ids(dw), rhyme_ids(ow)
        if p0 and p1 and dr and orr:
            ids0 = tok(text + p0, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
            ids1 = tok(text + p1, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
            y_off_p0, s_off = run(ids0, anchor, want_states=True)
            y_on_p0, s_on = run(ids0, anchor, steer, want_states=True)
            y_off_p1, _ = run(ids1, anchor)
            y_on_p1, _ = run(ids1, anchor, steer)
            mids = range(anchor + 1, ids0.shape[1] - 1)
            y_ret, _ = run(ids0, anchor, steer, {q: [s[q] for s in s_off] for q in mids})
            y_rel, _ = run(ids0, anchor, None, {q: [s[q] for s in s_on] for q in mids})
            R = lambda p: math.log(max(float(p[dr].sum()), 1e-12)) - math.log(max(float(p[orr].sum()), 1e-12))  # noqa: E731
            v = {"off_p0": R(y_off_p0), "on_p0": R(y_on_p0), "off_p1": R(y_off_p1), "on_p1": R(y_on_p1), "ret": R(y_ret), "rel": R(y_rel)}
            rec.update({"d_total": v["on_p1"] - v["off_p0"], "d_emission": v["off_p1"] - v["off_p0"],
                        "d_persistence": v["on_p1"] - v["off_p1"], "d_persistence_p0": v["on_p0"] - v["off_p0"],
                        "d_retrieval": v["ret"] - v["off_p0"], "d_relay": v["rel"] - v["off_p0"], "n_between": len(mids)})
        rows.append(rec)
        print(f"{len(rows):3d} [{time.time()-t0:5.0f}s] donor-rhyme {rec['on_rhymes_donor']} (authors {rec['authors_steered_rhymes_donor']}) "
              + (f"total {rec['d_total']:+.2f} emis {rec['d_emission']:+.2f} ret {rec['d_retrieval']:+.2f} rel {rec['d_relay']:+.2f}" if "d_total" in rec else ""), flush=True)
    out = EXP / "results" / a.model
    (out / "step6_rows.json").write_text(json.dumps(rows, indent=1))
    keys = ["d_total", "d_emission", "d_persistence", "d_persistence_p0", "d_retrieval", "d_relay"]
    sub = [r for r in rows if "d_total" in r]
    s = {"model": a.model, "n": len(rows), "n_routes": len(sub), "elapsed_sec": time.time() - t0,
         "on_rhymes_donor": sum(r["on_rhymes_donor"] for r in rows) / len(rows),
         "on_rhymes_orig": sum(r["on_rhymes_orig"] for r in rows) / len(rows),
         "authors_steered_rhymes_donor": sum(r["authors_steered_rhymes_donor"] for r in rows) / len(rows),
         "routes": {k: boot([r[k] for r in sub]) for k in keys},
         "routes_edit_succeeded": {k: boot([r[k] for r in sub if r["on_rhymes_donor"]]) for k in keys},
         "additivity_gap_p0": boot([r["d_persistence_p0"] - r["d_retrieval"] - r["d_relay"] for r in sub])}
    (out / "step6_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: s[k] for k in ("n", "n_routes", "on_rhymes_donor", "on_rhymes_orig", "authors_steered_rhymes_donor")}, indent=1))
    print({k: (round(v["mean"], 2), round(v["lo"], 2), round(v["hi"], 2)) for k, v in s["routes"].items() if v})


if __name__ == "__main__":
    main()
