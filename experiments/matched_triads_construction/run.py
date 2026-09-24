#!/usr/bin/env python3
"""Workstreams C and D on the frozen 14 matched triads (Gemma 3 1B, layer 17).

C: Is cross-class public recruitment a property of the model or of how the
   steering vector is built? Compare three constructions of the same donor
   direction delta = h(target donor) - h(source donor):
     named            delta (the paper's construction)
     article_removed  delta minus its component along g_hat
     article_only     the component along g_hat alone (diagnostic)
   g is the gradient of logit(an) - logit(a) at the neutral prompt's final
   position with respect to the layer-17 residual at that position (the local
   article-readout direction). We also record delta.g, the first-order article
   push of each construction, and compare it across arms.

D: For every cell, record the reverse-order decomposition (private at the
   baseline article policy; public under the treated branch) and target-vs-source
   log-odds at each endpoint, so route shares can be read on a log-odds scale.

Triads, layer, strengths and temperatures are those frozen in
experiments/matched_semantic_triads_repaired (14 admissible triads).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gemma_1b_residual_scale.run import ResidualModel, first_id, stats  # noqa: E402
from experiments.lib.aan_protocol import token_id_for_text  # noqa: E402
from experiments.matched_semantic_triads_repaired.run import (  # noqa: E402
    atomic_json, bootstrap, capture, exact_sign_flip, vector)

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"
FROZEN = ROOT / "experiments/matched_semantic_triads_repaired"


def article_gradient(rm: ResidualModel, text: str, position: int, layer: int, ids: dict[str, int]) -> torch.Tensor:
    """d(logit_an - logit_a)/d(residual at layer, position), evaluated at the unpatched prompt."""
    holder = {}

    def hook(_m, _i, output):
        hidden = output[0] if isinstance(output, tuple) else output
        leaf = hidden[:, position, :].detach().clone().requires_grad_(True)
        holder["leaf"] = leaf
        changed = hidden.clone()
        changed[:, position, :] = leaf
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    h = rm.layers[layer].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            logits = rm.model(**rm.inputs(text), use_cache=False).logits[0, -1]
            (logits[ids["an"]] - logits[ids["a"]]).backward()
    finally:
        h.remove()
    return holder["leaf"].grad[0].detach().float().cpu()


def logodds(p: torch.Tensor, tgt: int, src: int) -> float:
    return math.log(max(float(p[tgt]), 1e-30)) - math.log(max(float(p[src]), 1e-30))


def main() -> None:
    cfg = json.loads((FROZEN / "config.json").read_text())
    screen = json.loads((FROZEN / "results/screen_summary.json").read_text())
    triads = [t for t in cfg["triads"] if t["id"] in set(screen["admissible_ids"])]
    layer = cfg["fixed_layer"]
    rm = ResidualModel(cfg["model_snapshot"], getattr(torch, cfg["dtype"]))
    tok = rm.tokenizer
    ids = {a: token_id_for_text(tok, f" {a}") for a in ("a", "an")}
    art = [ids["a"], ids["an"]]
    rows_path = RESULTS / "rows.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {(r["triad_id"], r["arm"], r["construction"], r["strength"]) for r in rows}
    for triad in triads:
        neutral = cfg["neutral_template"].format(definition=triad["definition"])
        pos, states = capture(rm, neutral)
        g = article_gradient(rm, neutral, pos, layer, ids)
        g_hat = g / g.norm()
        lex = {role: first_id(tok, triad[f"{role}_word"]) for role in ("source", "within", "cross")}
        donor = {}
        for role in ("source", "within", "cross"):
            _, st = capture(rm, cfg["donor_template"].format(word=triad[f"{role}_word"], definition=triad["definition"]))
            donor[role] = st[layer]
        off_art = rm.logits(neutral)
        off_br = {a: rm.logits(neutral + tok.decode([i])) for a, i in ids.items()}
        for role in ("within", "cross"):
            delta = donor[role] - donor["source"]
            comp = float(delta @ g_hat) * g_hat
            constructions = {"named": delta, "article_removed": delta - comp, "article_only": comp}
            for cname, d in constructions.items():
                for s in cfg["strengths"]:
                    if (triad["id"], role, cname, s) in done:
                        continue
                    patch = (layer, pos, states[layer] + s * d)
                    on_art = rm.logits(neutral, patch)
                    on_br = {a: rm.logits(neutral + tok.decode([i]), patch) for a, i in ids.items()}
                    tgt, src = lex[role], lex["source"]
                    cells = {}
                    for tau in cfg["temperatures"]:
                        q0 = torch.softmax(off_art[art] / tau, -1)
                        q1 = torch.softmax(on_art[art] / tau, -1)
                        y0 = {a: torch.softmax(v, -1) for a, v in off_br.items()}
                        y1 = {a: torch.softmax(v, -1) for a, v in on_br.items()}
                        off = q0[0] * y0["a"] + q0[1] * y0["an"]
                        pub = q1[0] * y0["a"] + q1[1] * y0["an"]       # forward order midpoint
                        trt = q1[0] * y1["a"] + q1[1] * y1["an"]
                        rev = q0[0] * y1["a"] + q0[1] * y1["an"]       # reverse order midpoint
                        cells[str(tau)] = {
                            "public": vector(pub - off, src, tgt), "private": vector(trt - pub, src, tgt),
                            "total": vector(trt - off, src, tgt),
                            "private_reverse": vector(rev - off, src, tgt), "public_reverse": vector(trt - rev, src, tgt),
                            "logodds": {"off": logodds(off, tgt, src), "public_mid": logodds(pub, tgt, src),
                                        "reverse_mid": logodds(rev, tgt, src), "treated": logodds(trt, tgt, src)},
                            "q_target_article_off": float(q0[art.index(ids[triad[f"{role}_article"]])]),
                            "q_target_article_on": float(q1[art.index(ids[triad[f"{role}_article"]])]),
                            "on_article_mass": float(torch.softmax(on_art / tau, -1)[art].sum()),
                        }
                    ta = triad[f"{role}_article"]
                    rows.append({
                        "triad_id": triad["id"], "arm": role, "construction": cname, "strength": s,
                        "target_article": ta, "layer": layer,
                        "delta_norm": float(d.norm()), "article_push_first_order": float(s * (d @ g)),
                        "article_logit_change": float((on_art[ids["an"]] - on_art[ids["a"]]) - (off_art[ids["an"]] - off_art[ids["a"]])),
                        "fixed_target_article_effect": stats(on_br[ta], lex["source"], tgt)["target_minus_source"]
                        - stats(off_br[ta], lex["source"], tgt)["target_minus_source"],
                        "stochastic": cells,
                    })
                    atomic_json(rows_path, rows)
        print(f"done {triad['id']}", flush=True)
    summarize(cfg, triads, rows)


def summarize(cfg, triads, rows) -> None:
    s0, t0 = cfg["primary_strength"], str(cfg["primary_temperature"])
    out = {"primary": {"layer": cfg["fixed_layer"], "strength": s0, "temperature": float(t0)}, "constructions": {}}
    for cname in ("named", "article_removed", "article_only"):
        per = []
        for t in triads:
            r = {row["arm"]: row for row in rows if row["triad_id"] == t["id"] and row["construction"] == cname and row["strength"] == s0}
            if set(r) != {"within", "cross"}:
                continue
            R = {arm: r[arm]["stochastic"][t0]["public"]["target_minus_source"] - r[arm]["stochastic"][t0]["private"]["target_minus_source"] for arm in r}
            Rrev = {arm: r[arm]["stochastic"][t0]["public_reverse"]["target_minus_source"] - r[arm]["stochastic"][t0]["private_reverse"]["target_minus_source"] for arm in r}
            per.append({"triad_id": t["id"], "R_within": R["within"], "R_cross": R["cross"], "interaction": R["cross"] - R["within"],
                        "interaction_reverse_order": Rrev["cross"] - Rrev["within"],
                        "article_push_within": r["within"]["article_push_first_order"], "article_push_cross": r["cross"]["article_push_first_order"],
                        "efficacy_within": r["within"]["fixed_target_article_effect"], "efficacy_cross": r["cross"]["fixed_target_article_effect"]})
        if not per:
            continue
        inter = [p["interaction"] for p in per]
        out["constructions"][cname] = {
            "n": len(per), "rows": per,
            "interaction": bootstrap(inter, cfg["bootstrap_seed"], cfg["bootstrap_resamples"]),
            "interaction_sign_flip": exact_sign_flip(inter),
            "positive_interactions": sum(x > 0 for x in inter),
            "interaction_reverse_order": bootstrap([p["interaction_reverse_order"] for p in per], cfg["bootstrap_seed"], cfg["bootstrap_resamples"]),
            "article_push_cross_minus_within": bootstrap([p["article_push_cross"] - p["article_push_within"] for p in per], cfg["bootstrap_seed"], cfg["bootstrap_resamples"]),
            "efficacy_cross": bootstrap([p["efficacy_cross"] for p in per], cfg["bootstrap_seed"], cfg["bootstrap_resamples"]),
            "efficacy_within": bootstrap([p["efficacy_within"] for p in per], cfg["bootstrap_seed"], cfg["bootstrap_resamples"]),
        }
    atomic_json(RESULTS / "summary.json", out)
    print(json.dumps({k: {kk: v[kk] for kk in ("n", "interaction", "positive_interactions", "article_push_cross_minus_within")}
                      for k, v in out["constructions"].items()}, indent=1))


if __name__ == "__main__":
    main()
