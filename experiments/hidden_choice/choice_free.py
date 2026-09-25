#!/usr/bin/env python3
"""Hidden choice, free version (secondary arm of the frozen design, README).

Same pairs as choice_routes.py (same seed and screening). The model writes its own
weather sentence (greedy, cut at the first sentence end); the reveal phrase "\\nThe
fruit I chose is" is appended and R = log p(donor pick) - log p(original pick) is
read at the reveal. Intervention = the text swap (donor list order in the prompt).
Six-cell accounting with the sentence as mediator: text0 / text1 = the sentence
written under the original / donor prompt.
  emission    = R(orig prompt, text1) - R(orig prompt, text0)   (does the written
                sentence alone carry the pick? a covert channel if so)
  persistence = R(donor prompt, text1) - R(orig prompt, text1)
Also reports how often the sentence differs and whether it names a fruit.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP.parents[0] / "couplet_routes"))
sys.path.insert(0, str(EXP.parents[0] / "derived_value_carry"))
from chain_routes import F32Head  # noqa: E402
from choice_routes import FRUITS, REVEAL, SEED  # noqa: E402
from models import is_gemma, load  # noqa: E402
from step34_routes import boot  # noqa: E402


def prompt_of(tok, lst, gemma):
    q = (f"Secretly choose one fruit from this list: {', '.join(lst)}. Do not write your choice yet. "
         "First write one sentence about the weather. Then write the fruit you chose.")
    msgs = [{"role": "user", "content": q if gemma else "/no_think " + q}]
    kw = {} if gemma else {"enable_thinking": False}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen3-8B")
    ap.add_argument("--pairs", type=int, default=100)
    a = ap.parse_args()
    tok, m, _ = load(a.model)
    m.lm_head = F32Head(m.lm_head.weight)
    gemma = is_gemma(a.model)
    fid = {f: tok.encode(" " + f, add_special_tokens=False)[0] for f in FRUITS}

    @torch.no_grad()
    def logp(text):
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
        return torch.log_softmax(m(ids).logits[0, -1].float(), -1).cpu()

    @torch.no_grad()
    def sentence(prompt):
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
        out = m.generate(ids, max_new_tokens=40, do_sample=False)
        txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        mm = re.match(r"(.+?[.!?])(\s|$)", txt.replace("\n", " "))
        return (mm.group(1) if mm else txt.split("\n")[0]).strip()

    def pick(lst):
        y = logp(prompt_of(tok, lst, gemma) + "The weather is calm and mild today.\n" + REVEAL)
        return max(lst, key=lambda f: float(y[fid[f]]))

    rng = random.Random(SEED)
    rows, t0, tried = [], time.time(), 0
    while len(rows) < a.pairs and tried < 20 * a.pairs:
        tried += 1
        four = rng.sample(FRUITS, 4)
        perms = list(itertools.permutations(four))
        rng.shuffle(perms)
        picks = {p: pick(list(p)) for p in perms[:8]}
        orig = perms[0]
        donors = [p for p in perms[1:8] if picks[p] != picks[orig]]
        if not donors:
            continue
        don = donors[0]
        po, pdn = picks[orig], picks[don]
        P0, P1 = prompt_of(tok, list(orig), gemma), prompt_of(tok, list(don), gemma)
        t_0, t_1 = sentence(P0), sentence(P1)
        R = lambda y: float(y[fid[pdn]] - y[fid[po]])  # noqa: E731
        c = {k: R(logp(p + t + "\n" + REVEAL)) for k, (p, t) in
             {"o0": (P0, t_0), "d1": (P1, t_1), "o1": (P0, t_1), "d0": (P1, t_0)}.items()}
        rec = {"list": list(orig), "donor_list": list(don), "pick": po, "donor_pick": pdn, "text0": t_0, "text1": t_1,
               "sentence_changed": t_0 != t_1, "names_fruit": any(f in (t_0 + " " + t_1).lower() for f in FRUITS),
               "total": c["d1"] - c["o0"], "emission": c["o1"] - c["o0"],
               "persistence_t1": c["d1"] - c["o1"], "persistence_t0": c["d0"] - c["o0"]}
        rows.append(rec)
        print(f"{len(rows):3d} [{time.time()-t0:5.0f}s] {po}->{pdn} changed={rec['sentence_changed']} total {rec['total']:+.2f} "
              f"emission {rec['emission']:+.2f} pers {rec['persistence_t1']:+.2f} | {t_0[:50]!r} / {t_1[:50]!r}", flush=True)
    out = EXP / "results" / a.model
    out.mkdir(parents=True, exist_ok=True)
    (out / "choice_free_rows.json").write_text(json.dumps(rows, indent=1))
    ch = [r for r in rows if r["sentence_changed"]]
    s = {"model": a.model, "n": len(rows), "n_sentence_changed": len(ch),
         "n_names_fruit": sum(r["names_fruit"] for r in rows),
         **{k: boot([r[k] for r in rows]) for k in ("total", "emission", "persistence_t1", "persistence_t0")},
         "emission_when_changed": boot([r["emission"] for r in ch])}
    (out / "choice_free_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: (round(v["mean"], 2), round(v["lo"], 2), round(v["hi"], 2)) if isinstance(v, dict) and v else v for k, v in s.items()}))


if __name__ == "__main__":
    main()
