#!/usr/bin/env python3
"""Neutral evaluated-prompt repair of the correctness-preserving synonym assay."""
from __future__ import annotations

import itertools
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.gemma_1b_residual_scale.run import ResidualModel, cosine, first_id, stats
from experiments.lib.aan_protocol import token_id_for_text

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def interval(values: list[float], seed: int, resamples: int) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = random.Random(seed); n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(resamples))
    return {"n": n, "mean": sum(values)/n,
            "lo": draws[math.floor(.025*(resamples-1))], "hi": draws[math.ceil(.975*(resamples-1))],
            "method": "semantic-family nonparametric bootstrap", "resamples": resamples}


def difference_interval(left: list[float], right: list[float], seed: int, resamples: int) -> dict[str, Any]:
    rng = random.Random(seed); nl, nr = len(left), len(right)
    draws = sorted(sum(left[rng.randrange(nl)] for _ in range(nl))/nl-
                   sum(right[rng.randrange(nr)] for _ in range(nr))/nr for _ in range(resamples))
    return {"n_left": nl, "n_right": nr, "mean": sum(left)/nl-sum(right)/nr,
            "lo": draws[math.floor(.025*(resamples-1))], "hi": draws[math.ceil(.975*(resamples-1))],
            "method": "independent semantic-family bootstrap", "resamples": resamples}


def exact_permutation(left: list[float], right: list[float]) -> dict[str, Any]:
    values=left+right; observed=sum(left)/len(left)-sum(right)/len(right); diffs=[]
    for idx in itertools.combinations(range(len(values)),len(left)):
        chosen=set(idx); a=[v for i,v in enumerate(values) if i in chosen]; b=[v for i,v in enumerate(values) if i not in chosen]
        diffs.append(sum(a)/len(a)-sum(b)/len(b))
    return {"observed":observed,"two_sided_p":sum(abs(x)>=abs(observed)-1e-12 for x in diffs)/len(diffs),"assignments":len(diffs)}


def vec(effect: torch.Tensor, sid: int, tid: int) -> dict[str, float]:
    desired=torch.zeros_like(effect); desired[tid]=1; desired[sid]=-1
    return {"tv":float(.5*effect.abs().sum()),"target_minus_source":float(effect[tid]-effect[sid]),"desired_cosine":cosine(effect,desired)}


def prompt(cfg: dict[str, Any], pair: dict[str, Any], kind: str) -> str:
    if kind == "neutral": return cfg["neutral_template"].format(definition=pair["definition"])
    return cfg["donor_template"].format(word=pair[f"{kind}_word"],definition=pair["definition"])


def main() -> None:
    cfg=load(EXP/"config.json"); RESULTS.mkdir(parents=True,exist_ok=True)
    pairs=load((EXP/cfg["selected_pairs_path"]).resolve())
    atomic_json(RESULTS/"run_metadata.json",{"started_at":datetime.now(timezone.utc).isoformat(),"config":cfg,
        "pair_ids":[p["id"] for p in pairs],"inference":"direct full prompt; use_cache=False",
        "intervention":"h_neutral + strength * (h_target_donor - h_source_donor)"})
    started=time.time(); rm=ResidualModel(cfg["model_snapshot"],getattr(torch,cfg["dtype"])); tok=rm.tokenizer
    articles={a:token_id_for_text(tok,f" {a}") for a in ("a","an")}; layers=list(range(rm.n_layers)); cache={}
    screen_rows=[]
    for i,pair in enumerate(pairs,1):
        np=prompt(cfg,pair,"neutral"); sp=prompt(cfg,pair,"source"); tp=prompt(cfg,pair,"target")
        npos=len(tok(np,add_special_tokens=True).input_ids)-1; spos=len(tok(sp,add_special_tokens=True).input_ids)-1; tpos=len(tok(tp,add_special_tokens=True).input_ids)-1
        sid,tid=first_id(tok,pair["source_word"]),first_id(tok,pair["target_word"])
        ns,ss,ts=rm.states(np,npos),rm.states(sp,spos),rm.states(tp,tpos)
        cache[pair["id"]]={"np":np,"npos":npos,"ns":ns,"ss":ss,"ts":ts,"sid":sid,"tid":tid}
        neutral_article=rm.logits(np); neutral_branches={a:rm.logits(np+tok.decode([aid])) for a,aid in articles.items()}
        source_logits=rm.logits(sp+tok.decode([articles[pair["source_article"]]])); target_logits=rm.logits(tp+tok.decode([articles[pair["target_article"]]]))
        nstat=stats(neutral_branches[pair["source_article"]],sid,tid); sstat=stats(source_logits,sid,tid); tstat=stats(target_logits,sid,tid)
        screen_rows.append({"pair_id":pair["id"],"meaning_family":pair["meaning_family"],"regime":pair["analysis_regime"],
            "neutral_prompt":np,"source_donor_prompt":sp,"target_donor_prompt":tp,
            "neutral_source_article":tok.decode([int(neutral_article.argmax())]).strip(),
            "neutral_source_branch":nstat,"source_donor":sstat,"target_donor":tstat,
            "donor_separation":tstat["target_minus_source"]-sstat["target_minus_source"]})
        atomic_json(RESULTS/"screen_rows.json",screen_rows)
        print(f"captured and screened {i}/{len(pairs)}",flush=True)

    layer_rows=[]
    for i,pair in enumerate(pairs,1):
        c=cache[pair["id"]]; article=pair["target_article"]; prefix=c["np"]+tok.decode([articles[article]])
        base=stats(rm.logits(prefix),c["sid"],c["tid"])
        for layer in layers:
            delta=c["ts"][layer]-c["ss"][layer]; replacement=c["ns"][layer]+delta
            treated=stats(rm.logits(prefix,(layer,c["npos"],replacement)),c["sid"],c["tid"])
            layer_rows.append({"pair_id":pair["id"],"meaning_family":pair["meaning_family"],"regime":pair["analysis_regime"],"layer":layer,
                "target_branch_delta_delta":treated["target_minus_source"]-base["target_minus_source"]})
        atomic_json(RESULTS/"layer_rows.json",layer_rows)
        print(f"layer sweep {i}/{len(pairs)}",flush=True)

    selected_layers={}; selection_scores={}
    for heldout in pairs:
        scores={}
        for layer in layers:
            regime_means=[]
            for regime in ("between","within"):
                values=[r["target_branch_delta_delta"] for r in layer_rows if r["meaning_family"]!=heldout["meaning_family"] and r["regime"]==regime and r["layer"]==layer]
                regime_means.append(sum(values)/len(values))
            scores[layer]=sum(regime_means)/2
        selected_layers[heldout["id"]]=max(scores,key=scores.get); selection_scores[heldout["id"]]=scores

    rows=load(RESULTS/"rows.json") if (RESULTS/"rows.json").exists() else []
    completed={r["pair_id"] for r in rows if sum(x["pair_id"]==r["pair_id"] for x in rows)==len(cfg["strengths"])}
    for i,pair in enumerate(pairs,1):
        if pair["id"] in completed:
            print(f"assay checkpoint already contains {pair['id']} ({i}/{len(pairs)})",flush=True)
            continue
        c=cache[pair["id"]]; layer=selected_layers[pair["id"]]; delta=c["ts"][layer]-c["ss"][layer]
        off_article=rm.logits(c["np"]); off_branches={a:rm.logits(c["np"]+tok.decode([aid])) for a,aid in articles.items()}
        source_article,target_article=pair["source_article"],pair["target_article"]
        wrong=next(q for q in pairs if q["analysis_regime"]==pair["analysis_regime"] and q["meaning_family"]!=pair["meaning_family"])
        wc=cache[wrong["id"]]; wrong_delta=wc["ts"][layer]-wc["ss"][layer]
        for strength in cfg["strengths"]:
            replacement=c["ns"][layer]+float(strength)*delta; patch=(layer,c["npos"],replacement)
            on_article=rm.logits(c["np"],patch); on_branches={a:rm.logits(c["np"]+tok.decode([aid]),patch) for a,aid in articles.items()}
            base=stats(off_branches[target_article],c["sid"],c["tid"]); treated=stats(on_branches[target_article],c["sid"],c["tid"])
            controls={}
            for name,rep in {"wrong":c["ns"][layer]+float(strength)*wrong_delta,"sign_reversed":c["ns"][layer]-float(strength)*delta}.items():
                cs=stats(rm.logits(c["np"]+tok.decode([articles[target_article]]),(layer,c["npos"],rep)),c["sid"],c["tid"])
                controls[name]=cs["target_minus_source"]-base["target_minus_source"]
            stochastic={}
            for tau0 in cfg["temperatures"]:
                tau=float(tau0); q0=torch.softmax(off_article[[articles["a"],articles["an"]]]/tau,-1); q1=torch.softmax(on_article[[articles["a"],articles["an"]]]/tau,-1)
                y0={a:torch.softmax(x,-1) for a,x in off_branches.items()}; y1={a:torch.softmax(x,-1) for a,x in on_branches.items()}
                off=q0[0]*y0["a"]+q0[1]*y0["an"]; pub=q1[0]*y0["a"]+q1[1]*y0["an"]; on=q1[0]*y1["a"]+q1[1]*y1["an"]
                stochastic[str(tau0)]={"delta_q_target_article":float(q1[0 if target_article=="a" else 1]-q0[0 if target_article=="a" else 1]),
                    "off_article_mass":float(torch.softmax(off_article/tau,-1)[list(articles.values())].sum()),"on_article_mass":float(torch.softmax(on_article/tau,-1)[list(articles.values())].sum()),
                    "total":vec(on-off,c["sid"],c["tid"]),"public":vec(pub-off,c["sid"],c["tid"]),"private":vec(on-pub,c["sid"],c["tid"]),
                    "reconstruction_l1":float((on-off-(pub-off)-(on-pub)).abs().sum())}
            rows.append({"pair_id":pair["id"],"meaning_family":pair["meaning_family"],"regime":pair["analysis_regime"],"source_word":pair["source_word"],"target_word":pair["target_word"],
                "source_article":source_article,"target_article":target_article,"strength":strength,"layer":layer,
                "greedy_article_off":tok.decode([int(off_article.argmax())]).strip(),"greedy_article_on":tok.decode([int(on_article.argmax())]).strip(),
                "target_branch_delta_delta":treated["target_minus_source"]-base["target_minus_source"],"target_rank_before":base["target_rank"],"target_rank_after":treated["target_rank"],
                "local_efficacy_pass":treated["target_minus_source"]-base["target_minus_source"]>0,
                "controls":controls,"stochastic":stochastic})
        atomic_json(RESULTS/"rows.json",rows); atomic_json(RESULTS/"progress.json",{"completed_families":len({r['pair_id'] for r in rows}),"total_families":len(pairs),"last_pair":pair["id"]})
        print(f"assayed and checkpointed {i}/{len(pairs)}",flush=True)

    seed=int(cfg["bootstrap_seed"]); nboot=int(cfg["bootstrap_resamples"]); summary={"experiment":cfg["experiment_name"],"generated_at":datetime.now(timezone.utc).isoformat(),
        "elapsed_sec":time.time()-started,"model":cfg["model"],"intervention":"h_neutral + strength * (h_target_donor - h_source_donor)",
        "selected_layers":selected_layers,"selection_scores":selection_scores,"conditions":{},"interactions":{}}
    for strength in cfg["strengths"]:
        for regime in ("between","within"):
            group=[r for r in rows if r["strength"]==strength and r["regime"]==regime]; key=f"{regime}_{strength}"; block={"n":len(group),"local_efficacy_pass_n":sum(r["local_efficacy_pass"] for r in group)}
            block["target_branch_delta_delta"]=interval([r["target_branch_delta_delta"] for r in group],seed+len(summary["conditions"]),nboot)
            block["temperatures"]={}
            for tau in cfg["temperatures"]:
                tb={}
                for route in ("total","public","private"):
                    for metric in ("tv","target_minus_source","desired_cosine"):
                        tb[f"{route}_{metric}"]=interval([r["stochastic"][str(tau)][route][metric] for r in group],seed+100+len(tb),nboot)
                block["temperatures"][str(tau)]=tb
            summary["conditions"][key]=block
        summary["interactions"][f"strength_{strength}"]={}
        for tau in cfg["temperatures"]:
            b=[r for r in rows if r["strength"]==strength and r["regime"]=="between"]; w=[r for r in rows if r["strength"]==strength and r["regime"]=="within"]
            br=[r["stochastic"][str(tau)]["public"]["target_minus_source"]-r["stochastic"][str(tau)]["private"]["target_minus_source"] for r in b]
            wr=[r["stochastic"][str(tau)]["public"]["target_minus_source"]-r["stochastic"][str(tau)]["private"]["target_minus_source"] for r in w]
            summary["interactions"][f"strength_{strength}"][str(tau)]={"aligned_route_interaction":difference_interval(br,wr,seed+500,nboot),"exact_permutation":exact_permutation(br,wr)}
    atomic_json(RESULTS/"summary.json",summary); atomic_json(RESULTS/"progress.json",{"completed_families":len(pairs),"total_families":len(pairs),"status":"complete"})
    print(json.dumps({"elapsed_sec":summary["elapsed_sec"],"selected_layers":selected_layers,"native_tau1":summary["interactions"]["strength_1.0"]["1.0"]},indent=2))


if __name__ == "__main__": main()
