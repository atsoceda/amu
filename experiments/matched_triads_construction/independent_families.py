#!/usr/bin/env python3
"""Workstream C on the independent-family assay (paper Fig. 4C): same three constructions.

Uses the frozen pairs, templates, strengths and temperatures of
experiments/neutral_synonym_repair and its selected layer (17 in every fold).
The between-minus-within interaction of R = public - private target-aligned
effects is tested with the original exact family-label permutation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gemma_1b_residual_scale.run import ResidualModel, first_id, stats  # noqa: E402
from experiments.lib.aan_protocol import token_id_for_text  # noqa: E402
from experiments.matched_semantic_triads_repaired.run import atomic_json  # noqa: E402
from experiments.matched_triads_construction.run import article_gradient  # noqa: E402
from experiments.neutral_synonym_repair.run import difference_interval, exact_permutation, prompt, vec  # noqa: E402

EXP = Path(__file__).resolve().parent
SRC = ROOT / "experiments/neutral_synonym_repair"


def main() -> None:
    cfg = json.loads((SRC / "config.json").read_text())
    pairs = json.loads((SRC / cfg["selected_pairs_path"]).resolve().read_text())
    selected = json.loads((SRC / "results/summary.json").read_text())["selected_layers"]
    rm = ResidualModel(cfg["model_snapshot"], getattr(torch, cfg["dtype"]))
    tok = rm.tokenizer
    ids = {a: token_id_for_text(tok, f" {a}") for a in ("a", "an")}
    art = [ids["a"], ids["an"]]
    rows = []
    for pair in pairs:
        layer = int(selected[pair["id"]])
        np_, sp, tp = (prompt(cfg, pair, k) for k in ("neutral", "source", "target"))
        pos = len(tok(np_, add_special_tokens=True).input_ids) - 1
        spos = len(tok(sp, add_special_tokens=True).input_ids) - 1
        tpos = len(tok(tp, add_special_tokens=True).input_ids) - 1
        ns = rm.states(np_, pos)[layer]
        delta = rm.states(tp, tpos)[layer] - rm.states(sp, spos)[layer]
        g = article_gradient(rm, np_, pos, layer, ids)
        g_hat = g / g.norm()
        comp = float(delta @ g_hat) * g_hat
        sid, tid = first_id(tok, pair["source_word"]), first_id(tok, pair["target_word"])
        off_art = rm.logits(np_)
        off_br = {a: rm.logits(np_ + tok.decode([i])) for a, i in ids.items()}
        ta = pair["target_article"]
        for cname, d in {"named": delta, "article_removed": delta - comp, "article_only": comp}.items():
            for s in cfg["strengths"]:
                patch = (layer, pos, ns + float(s) * d)
                on_art = rm.logits(np_, patch)
                on_br = {a: rm.logits(np_ + tok.decode([i]), patch) for a, i in ids.items()}
                cells = {}
                for tau0 in cfg["temperatures"]:
                    tau = float(tau0)
                    q0 = torch.softmax(off_art[art] / tau, -1)
                    q1 = torch.softmax(on_art[art] / tau, -1)
                    y0 = {a: torch.softmax(v, -1) for a, v in off_br.items()}
                    y1 = {a: torch.softmax(v, -1) for a, v in on_br.items()}
                    off = q0[0] * y0["a"] + q0[1] * y0["an"]
                    pub = q1[0] * y0["a"] + q1[1] * y0["an"]
                    on = q1[0] * y1["a"] + q1[1] * y1["an"]
                    k = 0 if ta == "a" else 1
                    cells[str(tau0)] = {"public": vec(pub - off, sid, tid), "private": vec(on - pub, sid, tid),
                                        "delta_q_target_article": float(q1[k] - q0[k])}
                rows.append({"pair_id": pair["id"], "regime": pair["analysis_regime"], "construction": cname,
                             "strength": s, "layer": layer, "delta_norm": float(d.norm()),
                             "fixed_target_article_effect": stats(on_br[ta], sid, tid)["target_minus_source"]
                             - stats(off_br[ta], sid, tid)["target_minus_source"],
                             "stochastic": cells})
        atomic_json(EXP / "results/independent_family_rows.json", rows)
        print(f"done {pair['id']}", flush=True)

    seed, nboot = int(cfg["bootstrap_seed"]), int(cfg["bootstrap_resamples"])
    out = {}
    for cname in ("named", "article_removed", "article_only"):
        out[cname] = {}
        for s in cfg["strengths"]:
            for tau in cfg["temperatures"]:
                def R(r):
                    c = r["stochastic"][str(tau)]
                    return c["public"]["target_minus_source"] - c["private"]["target_minus_source"]
                b = [R(r) for r in rows if r["construction"] == cname and r["strength"] == s and r["regime"] == "between"]
                w = [R(r) for r in rows if r["construction"] == cname and r["strength"] == s and r["regime"] == "within"]
                eff = [r["fixed_target_article_effect"] for r in rows if r["construction"] == cname and r["strength"] == s]
                out[cname][f"s{s}_tau{tau}"] = {"interaction": difference_interval(b, w, seed + 500, nboot),
                                                "exact_permutation": exact_permutation(b, w),
                                                "between_positive": sum(x > 0 for x in b), "n_between": len(b),
                                                "within_negative": sum(x < 0 for x in w), "n_within": len(w),
                                                "mean_local_efficacy": sum(eff) / len(eff)}
    atomic_json(EXP / "results/independent_family_summary.json", out)
    for cname, v in out.items():
        x = v["s1.0_tau1.0"] if "s1.0_tau1.0" in v else v[next(iter(v))]
        print(cname, json.dumps({k: x[k] for k in ("interaction", "between_positive", "within_negative", "mean_local_efficacy")}))


if __name__ == "__main__":
    main()
