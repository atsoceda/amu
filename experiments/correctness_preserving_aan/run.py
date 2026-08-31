#!/usr/bin/env python3
"""Correctness-preserving a/an public/private carrier assay in Gemma 3 1B PT."""
from __future__ import annotations

import itertools
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.gemma_1b_residual_scale.run import ResidualModel, cosine, first_id, stats
from experiments.lib.aan_protocol import token_id_for_text, write_json


EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def prompt(cfg: dict[str, Any], pair: dict[str, Any], which: str) -> str:
    word = pair[f"{which}_word"]
    return cfg[f"{which}_template"].format(definition=pair["definition"], initial=word[0].upper())


def interval(values: list[float], seed: int, resamples: int = 10000) -> dict[str, Any]:
    rng = random.Random(seed); n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return {"n": n, "mean": sum(values) / n,
            "lo": draws[math.floor(.025 * (resamples - 1))],
            "hi": draws[math.ceil(.975 * (resamples - 1))],
            "method": "semantic-family-level nonparametric bootstrap", "resamples": resamples}


def difference_interval(left: list[float], right: list[float], seed: int, resamples: int) -> dict[str, Any]:
    rng = random.Random(seed); nl, nr = len(left), len(right)
    draws = sorted(
        sum(left[rng.randrange(nl)] for _ in range(nl)) / nl
        - sum(right[rng.randrange(nr)] for _ in range(nr)) / nr
        for _ in range(resamples)
    )
    return {"n_left": nl, "n_right": nr, "mean": sum(left)/nl-sum(right)/nr,
            "lo": draws[math.floor(.025*(resamples-1))], "hi": draws[math.ceil(.975*(resamples-1))],
            "method": "independent semantic-family bootstrap", "resamples": resamples}


def exact_permutation(left: list[float], right: list[float]) -> dict[str, Any]:
    values = left + right; n_left = len(left)
    observed = sum(left)/len(left)-sum(right)/len(right)
    diffs = []
    for indices in itertools.combinations(range(len(values)), n_left):
        chosen = set(indices)
        a = [v for i, v in enumerate(values) if i in chosen]
        b = [v for i, v in enumerate(values) if i not in chosen]
        diffs.append(sum(a)/len(a)-sum(b)/len(b))
    return {"observed": observed, "two_sided_p": sum(abs(x) >= abs(observed)-1e-12 for x in diffs)/len(diffs),
            "assignments": len(diffs), "unit": "semantic family"}


def select_pairs(screen_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    family_alias = {
        "direction": "organization_management", "leadership": "organization_management", "management": "organization_management",
        "events": "activity_organization", "planning": "activity_organization",
        "illustration": "visual_art", "painting": "visual_art",
    }
    eligible = [r for r in screen_rows if r["admissible"]]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in eligible:
        regime = "between" if row["regime"] == "between" else "within"
        raw_family = row["id"].split("_", 1)[0]
        family = family_alias.get(raw_family, raw_family)
        grouped.setdefault((regime, family), []).append(row)
    selected, audit = [], []
    for (regime, family), rows in sorted(grouped.items()):
        ranked = sorted(rows, key=lambda r: (min(r["source_margin_under_intended_article"], r["target_margin_under_intended_article"]), r["id"]), reverse=True)
        winner = dict(ranked[0]); winner["analysis_regime"] = regime; winner["meaning_family"] = family
        selected.append(winner)
        audit.append({"regime": regime, "meaning_family": family, "selected_id": winner["id"],
                      "criterion": "largest minimum source/target signed screen margin",
                      "eligible_ids": [r["id"] for r in ranked]})
    return selected, audit


def vec_metrics(effect: torch.Tensor, source_id: int, target_id: int) -> dict[str, float]:
    desired = torch.zeros_like(effect); desired[target_id] = 1; desired[source_id] = -1
    return {"tv": float(.5*effect.abs().sum()), "target_minus_source": float(effect[target_id]-effect[source_id]),
            "desired_cosine": cosine(effect, desired)}


def main() -> None:
    cfg = load(EXP/"config.json"); RESULTS.mkdir(parents=True, exist_ok=True)
    screen = load(RESULTS/"screen_rows_gemma_1b.json")
    pairs, selection_audit = select_pairs(screen)
    if min(sum(p["analysis_regime"] == r for p in pairs) for r in ("between", "within")) < 5:
        raise RuntimeError("Fewer than five independent semantic families in a regime")
    started = time.time(); spec = cfg["models"]["gemma_1b"]
    rm = ResidualModel(spec["model_snapshot"], getattr(torch, cfg["dtype"])); tok = rm.tokenizer
    frozen_layer = int(cfg["frozen_layer"]); article_ids = {a: token_id_for_text(tok, f" {a}") for a in ("a", "an")}
    cache: dict[str, dict[str, Any]] = {}
    for index, pair in enumerate(pairs, 1):
        sp, tp = prompt(cfg, pair, "source"), prompt(cfg, pair, "target")
        spos = len(tok(sp, add_special_tokens=True).input_ids)-1
        tpos = len(tok(tp, add_special_tokens=True).input_ids)-1
        ss, ts = rm.states(sp, spos), rm.states(tp, tpos)
        cache[pair["id"]] = {"sp": sp, "tp": tp, "spos": spos, "ss": ss, "ts": ts,
                              "delta": [t-s for s,t in zip(ss,ts)], "sid": first_id(tok, pair["source_word"]), "tid": first_id(tok, pair["target_word"])}
        print(f"captured {index}/{len(pairs)}", flush=True)

    # Every pair is evaluated at a layer chosen without its semantic family.
    layer_rows = []
    for pair_index, pair in enumerate(pairs, 1):
        c=cache[pair["id"]]; article=pair["target_article"]; prefix=c["sp"]+tok.decode([article_ids[article]])
        baseline=rm.logits(prefix); base=stats(baseline,c["sid"],c["tid"])
        for candidate_layer in range(rm.n_layers):
            replacement=c["ss"][candidate_layer]+c["delta"][candidate_layer]
            treated=rm.logits(prefix,(candidate_layer,c["spos"],replacement)); effect=stats(treated,c["sid"],c["tid"])
            layer_rows.append({"pair_id":pair["id"],"meaning_family":pair["meaning_family"],"regime":pair["analysis_regime"],
                               "layer":candidate_layer,"target_branch_delta_delta":effect["target_minus_source"]-base["target_minus_source"]})
        print(f"layer sweep {pair_index}/{len(pairs)}",flush=True)
    selected_layers={}; selection_scores={}
    for heldout in pairs:
        scores={}
        for candidate_layer in range(rm.n_layers):
            regime_means=[]
            for regime in ("between","within"):
                values=[r["target_branch_delta_delta"] for r in layer_rows if r["pair_id"]!=heldout["id"] and r["regime"]==regime and r["layer"]==candidate_layer]
                regime_means.append(sum(values)/len(values))
            scores[candidate_layer]=sum(regime_means)/len(regime_means)
        selected_layers[heldout["id"]]=max(scores,key=scores.get); selection_scores[heldout["id"]]=scores

    # Wrong-target controls reuse a direction from another semantic family in the same regime.
    controls: dict[str, dict[str, str | None]] = {}
    for pair in pairs:
        peers = [p for p in pairs if p["analysis_regime"] == pair["analysis_regime"] and p["id"] != pair["id"]]
        same_letter = [p for p in peers if p["target_word"][0] == pair["target_word"][0] and p["target_article"] == pair["target_article"]]
        controls[pair["id"]] = {"wrong": peers[0]["id"], "letter_matched": same_letter[0]["id"] if same_letter else None}

    rows = []
    for pair_index, pair in enumerate(pairs, 1):
        c = cache[pair["id"]]; layer=selected_layers[pair["id"]]; off_article = rm.logits(c["sp"])
        off_branches = {a: rm.logits(c["sp"]+tok.decode([article_ids[a]])) for a in article_ids}
        off_id = int(off_article.argmax()); off_noun = rm.logits(c["sp"]+tok.decode([off_id]))
        source_article, target_article = pair["source_article"], pair["target_article"]
        base_target_branch = stats(off_branches[target_article], c["sid"], c["tid"])
        base_source_branch = stats(off_branches[source_article], c["sid"], c["tid"])
        for strength in cfg["strengths"]:
            replacement = c["ss"][layer] + float(strength)*c["delta"][layer]
            patch = (layer, c["spos"], replacement)
            on_article = rm.logits(c["sp"], patch)
            on_branches = {a: rm.logits(c["sp"]+tok.decode([article_ids[a]]), patch) for a in article_ids}
            on_id = int(on_article.argmax()); on_noun = rm.logits(c["sp"]+tok.decode([on_id]), patch)
            replay = rm.logits(c["sp"]+tok.decode([on_id]))
            greedy_total = torch.softmax(on_noun,-1)-torch.softmax(off_noun,-1)
            greedy_public = torch.softmax(replay,-1)-torch.softmax(off_noun,-1)
            greedy_private = torch.softmax(on_noun,-1)-torch.softmax(replay,-1)
            treated_target_branch = stats(on_branches[target_article], c["sid"], c["tid"])
            treated_source_branch = stats(on_branches[source_article], c["sid"], c["tid"])
            control_effects = {}
            control_ids = controls[pair["id"]]
            for name, other_id in control_ids.items():
                if other_id is None:
                    control_effects[name] = None; continue
                rep = c["ss"][layer] + float(strength)*cache[str(other_id)]["delta"][layer]
                logits = rm.logits(c["sp"]+tok.decode([article_ids[target_article]]), (layer,c["spos"],rep))
                control_effects[name] = stats(logits,c["sid"],c["tid"])["target_minus_source"]-base_target_branch["target_minus_source"]
            reverse_rep = c["ss"][layer]-float(strength)*c["delta"][layer]
            reverse_logits = rm.logits(c["sp"]+tok.decode([article_ids[target_article]]),(layer,c["spos"],reverse_rep))
            control_effects["sign_reversed"] = stats(reverse_logits,c["sid"],c["tid"])["target_minus_source"]-base_target_branch["target_minus_source"]
            stochastic = {}
            for tau_value in cfg["temperatures"]:
                tau = float(tau_value)
                q0 = torch.softmax(off_article[[article_ids["a"],article_ids["an"]]]/tau,-1)
                q1 = torch.softmax(on_article[[article_ids["a"],article_ids["an"]]]/tau,-1)
                y0 = {a: torch.softmax(off_branches[a],-1) for a in article_ids}
                y1 = {a: torch.softmax(on_branches[a],-1) for a in article_ids}
                off_mix = q0[0]*y0["a"]+q0[1]*y0["an"]
                public_mix = q1[0]*y0["a"]+q1[1]*y0["an"]
                on_mix = q1[0]*y1["a"]+q1[1]*y1["an"]
                total, public, private = on_mix-off_mix, public_mix-off_mix, on_mix-public_mix
                target_index = 0 if target_article == "a" else 1
                stochastic[str(tau_value)] = {
                    "delta_q_target_article": float(q1[target_index]-q0[target_index]),
                    "off_article_mass": float(torch.softmax(off_article/tau,-1)[list(article_ids.values())].sum()),
                    "on_article_mass": float(torch.softmax(on_article/tau,-1)[list(article_ids.values())].sum()),
                    "branch_leverage_tv": float(.5*(y0["an"]-y0["a"]).abs().sum()),
                    "total": vec_metrics(total,c["sid"],c["tid"]),
                    "public": vec_metrics(public,c["sid"],c["tid"]),
                    "private": vec_metrics(private,c["sid"],c["tid"]),
                    "reconstruction_l1": float((total-public-private).abs().sum()),
                }
            rows.append({"pair_id":pair["id"],"meaning_family":pair["meaning_family"],"regime":pair["analysis_regime"],
                         "source_word":pair["source_word"],"target_word":pair["target_word"],"source_article":source_article,"target_article":target_article,
                         "strength":strength,"layer":layer,"greedy_article_off":tok.decode([off_id]).strip(),"greedy_article_on":tok.decode([on_id]).strip(),
                         "article_changed":off_id!=on_id,"target_branch_delta_delta":treated_target_branch["target_minus_source"]-base_target_branch["target_minus_source"],
                         "source_branch_delta_delta":treated_source_branch["target_minus_source"]-base_source_branch["target_minus_source"],
                         "article_margin_movement":float((on_article[article_ids[target_article]]-on_article[article_ids[source_article]])-(off_article[article_ids[target_article]]-off_article[article_ids[source_article]])) if source_article!=target_article else 0.0,
                         "controls":control_effects,"greedy":{"total":vec_metrics(greedy_total,c["sid"],c["tid"]),"public":vec_metrics(greedy_public,c["sid"],c["tid"]),"private":vec_metrics(greedy_private,c["sid"],c["tid"])},
                         "stochastic":stochastic})
        print(f"assayed {pair_index}/{len(pairs)}", flush=True)

    seed, resamples = int(cfg["bootstrap_seed"]), int(cfg["bootstrap_resamples"])
    summary: dict[str, Any] = {"experiment":cfg["experiment_name"],"generated_at":datetime.now(timezone.utc).isoformat(),"elapsed_sec":time.time()-started,
        "model":spec,"source":cfg["source"],"layer_selection":cfg["primary_layer_selection"],
        "frozen_layer_baseline":{"layer":frozen_layer,"provenance":cfg["frozen_layer_provenance"],"results_path":"results/frozen_layer14"},
        "selected_layers":selected_layers,"selection_scores":selection_scores,
        "selection":{"counts":{"between":sum(p["analysis_regime"]=="between" for p in pairs),"within":sum(p["analysis_regime"]=="within" for p in pairs)},"audit":selection_audit},"conditions":{},"interactions":{}}
    scalar_metrics = ("target_branch_delta_delta","source_branch_delta_delta","article_margin_movement")
    for strength in cfg["strengths"]:
        for regime in ("between","within"):
            group=[r for r in rows if r["strength"]==strength and r["regime"]==regime]; key=f"{regime}_{strength}"
            summary["conditions"][key]={m:interval([float(r[m]) for r in group],seed+len(summary["conditions"])*100+i,resamples) for i,m in enumerate(scalar_metrics)}
            summary["conditions"][key]["article_change_rate"]=sum(r["article_changed"] for r in group)/len(group)
            summary["conditions"][key]["controls"]={name:interval([float(r["controls"][name]) for r in group if r["controls"][name] is not None],seed+700+i,resamples) for i,name in enumerate(("wrong","letter_matched","sign_reversed")) if any(r["controls"][name] is not None for r in group)}
            summary["conditions"][key]["temperatures"]={}
            for tau in cfg["temperatures"]:
                block={}
                for route in ("total","public","private"):
                    for metric in ("tv","target_minus_source","desired_cosine"):
                        block[f"{route}_{metric}"]=interval([r["stochastic"][str(tau)][route][metric] for r in group],seed+900+len(block),resamples)
                block["delta_q_target_article"]=interval([r["stochastic"][str(tau)]["delta_q_target_article"] for r in group],seed+990,resamples)
                summary["conditions"][key]["temperatures"][str(tau)]=block
        interaction_key=f"strength_{strength}"; summary["interactions"][interaction_key]={}
        for tau in cfg["temperatures"]:
            between=[r for r in rows if r["strength"]==strength and r["regime"]=="between"]
            within=[r for r in rows if r["strength"]==strength and r["regime"]=="within"]
            b_route=[r["stochastic"][str(tau)]["public"]["tv"]-r["stochastic"][str(tau)]["private"]["tv"] for r in between]
            w_route=[r["stochastic"][str(tau)]["public"]["tv"]-r["stochastic"][str(tau)]["private"]["tv"] for r in within]
            b_align=[r["stochastic"][str(tau)]["public"]["target_minus_source"]-r["stochastic"][str(tau)]["private"]["target_minus_source"] for r in between]
            w_align=[r["stochastic"][str(tau)]["public"]["target_minus_source"]-r["stochastic"][str(tau)]["private"]["target_minus_source"] for r in within]
            summary["interactions"][interaction_key][str(tau)]={"tv_route_interaction":difference_interval(b_route,w_route,seed+1100,resamples),
                "aligned_route_interaction":difference_interval(b_align,w_align,seed+1200,resamples),"tv_exact_permutation":exact_permutation(b_route,w_route)}
    write_json(RESULTS/"selected_pairs.json",pairs); write_json(RESULTS/"layer_rows.json",layer_rows); write_json(RESULTS/"rows.json",rows); write_json(RESULTS/"summary.json",summary)
    print(json.dumps({"counts":summary["selection"]["counts"],"elapsed_sec":summary["elapsed_sec"],"native_tau1":summary["interactions"]["strength_1.0"]["1.0"]},indent=2))


if __name__ == "__main__":
    main()
