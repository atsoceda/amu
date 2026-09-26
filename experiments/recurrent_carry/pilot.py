#!/usr/bin/env python3
"""Recurrent-carry pilot on a hybrid model (design frozen 2026-09-26, README).

Reads couplet_routes/results/<model>/step2_rows.json. Cells: persistence, attention-only
(recurrent mixers get the clean anchor input), recurrent-only (attention mixers get the
clean anchor input), both blocked, direct retrieval and relay. Mixer inputs are replaced
with forward pre-hooks on each layer's token mixer (linear_attn or self_attn).
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
CR = EXP.parents[0] / "couplet_routes"
sys.path.insert(0, str(CR))
from models import load  # noqa: E402
from step2_state_edit import prompt_ids, rhymes  # noqa: E402
from step34_routes import boot, strip_last_word  # noqa: E402


def mixers(layers):
    out = []
    for layer in layers:
        if hasattr(layer, "linear_attn"):
            out.append(("recurrent", layer.linear_attn))
        elif hasattr(layer, "self_attn"):
            out.append(("attention", layer.self_attn))
        else:
            raise AttributeError(f"no token mixer in {type(layer).__name__}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen3.5-4B")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    tok, m, layers = load(a.model)
    mix = mixers(layers)
    print("mixers:", [k for k, _ in mix], flush=True)
    rows = json.loads((CR / "results" / a.model / "step2_rows.json").read_text())[: a.limit or None]

    def rhyme_ids(word):
        return sorted({t[0] for w in rhymes(word) for t in [tok.encode(" " + w, add_special_tokens=False)] if len(t) == 1})

    @torch.no_grad()
    def run(ids, replace=None, mixer_clean=None, capture=False, want_states=False):
        """replace: {pos: per-layer outputs}; mixer_clean: (kind, pos, per-layer clean mixer inputs)."""
        hooks, cap = [], [None] * len(layers)
        if replace:
            for li, layer in enumerate(layers):
                def fo(mod, inp, out, li=li):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h.clone()
                    for p, sts in replace.items():
                        h[0, p] = sts[li].to(h.dtype)
                    return (h, *out[1:]) if isinstance(out, tuple) else h
                hooks.append(layer.register_forward_hook(fo))
        for li, (kind, mod) in enumerate(mix):
            if capture:
                def fc(mod_, args, kwargs, li=li):
                    x = kwargs.get("hidden_states", args[0] if args else None)
                    cap[li] = x[0].detach().clone()
                hooks.append(mod.register_forward_pre_hook(fc, with_kwargs=True))
            if mixer_clean and kind in mixer_clean[0]:
                _, pos, clean = mixer_clean

                def fm(mod_, args, kwargs, li=li, pos=pos, clean=clean):
                    if "hidden_states" in kwargs:
                        x = kwargs["hidden_states"].clone(); x[0, pos] = clean[li][pos].to(x.dtype); kwargs["hidden_states"] = x
                    else:
                        x = args[0].clone(); x[0, pos] = clean[li][pos].to(x.dtype); args = (x, *args[1:])
                    return args, kwargs
                hooks.append(mod.register_forward_pre_hook(fm, with_kwargs=True))
        try:
            out = m(ids, output_hidden_states=want_states, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        p = torch.softmax(out.logits[0, -1].float(), -1).cpu()
        return p, ([h[0] for h in out.hidden_states[1:]] if want_states else None), (cap if capture else None)

    out_rows, t0 = [], time.time()
    for r in rows:
        p0 = strip_last_word(r["gen_off"])
        dr, orr = rhyme_ids(r["donor_word"]), rhyme_ids(r["orig_word"])
        if not p0 or not dr or not orr:
            continue
        _, anchor, text = prompt_ids(tok, r["first_line"])
        _, danchor, dtext = prompt_ids(tok, r["donor_first_line"])
        dids = tok(dtext, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
        _, dsts, _ = run(dids, want_states=True)
        donor = [s[danchor] for s in dsts]
        ids = tok(text + p0, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
        R = lambda p: math.log(max(float(p[dr].sum()), 1e-12)) - math.log(max(float(p[orr].sum()), 1e-12))  # noqa: E731
        y_off, s_off, clean_in = run(ids, want_states=True, capture=True)
        edit = {anchor: donor}
        y_on, s_on, _ = run(ids, edit, want_states=True)
        mids = range(anchor + 1, ids.shape[1] - 1)
        base = R(y_off)
        rec = {"idx": r["idx"], "persistence": R(y_on) - base,
               "attention_only": R(run(ids, edit, ("recurrent", anchor, clean_in))[0]) - base,
               "recurrent_only": R(run(ids, edit, ("attention", anchor, clean_in))[0]) - base,
               "both_blocked": R(run(ids, edit, (("recurrent", "attention"), anchor, clean_in))[0]) - base,
               "retrieval": R(run(ids, edit | {q: [s[q] for s in s_off] for q in mids})[0]) - base,
               "relay": R(run(ids, {q: [s[q] for s in s_on] for q in mids})[0]) - base}
        out_rows.append(rec)
        print(f"{len(out_rows):3d} [{time.time()-t0:5.0f}s] pers {rec['persistence']:+.2f} attn-only {rec['attention_only']:+.2f} "
              f"rec-only {rec['recurrent_only']:+.2f} both {rec['both_blocked']:+.2f} | ret {rec['retrieval']:+.2f} relay {rec['relay']:+.2f}", flush=True)
    out = EXP / "results" / a.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot_rows.json").write_text(json.dumps(out_rows, indent=1))
    keys = ["persistence", "attention_only", "recurrent_only", "both_blocked", "retrieval", "relay"]
    s = {"model": a.model, "n": len(out_rows), **{k: boot([x[k] for x in out_rows]) for k in keys},
         "recurrent_share": boot([x["recurrent_only"] / x["persistence"] for x in out_rows if abs(x["persistence"]) > 1])}
    (out / "pilot_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: (round(v["mean"], 2), round(v["lo"], 2), round(v["hi"], 2)) if isinstance(v, dict) else v for k, v in s.items()}))


if __name__ == "__main__":
    main()
