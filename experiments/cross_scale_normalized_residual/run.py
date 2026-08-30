#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.gemma_1b_residual_scale.run import ResidualModel, first_id, mixture, stats, tv
from experiments.lib.aan_protocol import token_id_for_text, write_json

EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1];RESULTS=EXP/"results"
NATURAL=json.loads((ROOT/"experiments/natural_residual_carrier_regimes/config.json").read_text())
PAIRS=[pair for pair in NATURAL["pairs"] if pair["split"]=="test"]
MODELS={
    "gemma_270m":{"snapshot":"/Users/anthony/.cache/huggingface/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1","layer":12},
    "gemma_1b":{"snapshot":"/Users/anthony/.cache/huggingface/hub/models--google--gemma-3-1b-pt/snapshots/fcf18a2a879aab110ca39f8bffbccd5d49d8eb29","layer":14},
}
STRENGTHS=(0.5,1.0,1.5);TEMPERATURES=(0.1,1.0);DEMO=NATURAL["demonstration"]


def interval(values,seed,resamples=10000):
    rng=random.Random(seed);n=len(values);draws=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(resamples))
    return {"n":n,"mean":sum(values)/n,"lo":draws[math.floor(.025*(resamples-1))],"hi":draws[math.ceil(.975*(resamples-1))]}


def run_model(name,config):
    rm=ResidualModel(config["snapshot"],torch.bfloat16);tok=rm.tokenizer;layer=config["layer"]
    article_ids={article:token_id_for_text(tok,f" {article}") for article in ("a","an")};cache={};source_states=[]
    for pair in PAIRS:
        source_prompt=f"{DEMO} {pair['source_sentence']}";target_prompt=f"{DEMO} {pair['target_sentence']}"
        source_pos=len(tok(source_prompt,add_special_tokens=True).input_ids)-1;target_pos=len(tok(target_prompt,add_special_tokens=True).input_ids)-1
        source_state=rm.states(source_prompt,source_pos)[layer];target_state=rm.states(target_prompt,target_pos)[layer]
        source_states.append(source_state);cache[pair["id"]]={"source_prompt":source_prompt,"position":source_pos,"source_state":source_state,
            "target_state":target_state,"source_id":first_id(tok,pair["source_word"]),"target_id":first_id(tok,pair["target_word"])}
    matrix=torch.stack(source_states);center=matrix.mean(0);natural_rms=float(torch.sqrt(((matrix-center)**2).sum(-1).mean()))
    rows=[]
    for pair_index,pair in enumerate(PAIRS,1):
        item=cache[pair["id"]];prompt=item["source_prompt"];position=item["position"];delta=item["target_state"]-item["source_state"]
        off_article=rm.logits(prompt);off_id=int(off_article.argmax());off_branches={article:rm.logits(prompt+tok.decode([article_ids[article]])) for article in article_ids}
        native=pair["source_article"];baseline=stats(off_branches[native],item["source_id"],item["target_id"]);gap=float(baseline["target_minus_source"])
        for strength in STRENGTHS:
            replacement=item["source_state"]+strength*delta;patch=(layer,position,replacement)
            on_article=rm.logits(prompt,patch);on_branches={article:rm.logits(prompt+tok.decode([article_ids[article]]),patch) for article in article_ids}
            treated=stats(on_branches[native],item["source_id"],item["target_id"]);delta_delta=treated["target_minus_source"]-gap
            stochastic={}
            for tau in TEMPERATURES:
                off_mix,off_mass=mixture(off_article,off_branches,article_ids,tau);on_mix,on_mass=mixture(on_article,on_branches,article_ids,tau)
                public_mix,_=mixture(on_article,off_branches,article_ids,tau)
                stochastic[str(tau)]={"total_tv":float(.5*(on_mix-off_mix).abs().sum()),"public_tv":float(.5*(public_mix-off_mix).abs().sum()),
                                      "private_tv":float(.5*(on_mix-public_mix).abs().sum()),"off_article_mass":off_mass,"on_article_mass":on_mass}
            patch_norm=float((strength*delta).norm());gap_closed=delta_delta/(-gap) if gap<0 else None
            rows.append({"model":name,"pair_id":pair["id"],"regime":pair["regime"],"strength":strength,"layer":layer,
                         "relative_depth":layer/(rm.n_layers-1),"baseline_target_minus_source":gap,"delta_delta":delta_delta,
                         "fraction_gap_closed":gap_closed,"target_logit_change":treated["target_logit"]-baseline["target_logit"],
                         "target_log_odds_change":treated["target_logit"]-baseline["target_logit"],
                         "target_odds_multiplier":math.exp(max(-20,min(20,treated["target_logit"]-baseline["target_logit"]))),
                         "target_probability_change":treated["target_prob"]-baseline["target_prob"],
                         "target_rank_change":baseline["target_rank"]-treated["target_rank"],"target_rank_before":baseline["target_rank"],"target_rank_after":treated["target_rank"],
                         "patch_norm":patch_norm,"patch_norm_over_natural_rms":patch_norm/natural_rms,"effect_per_patch_norm":delta_delta/patch_norm if patch_norm else None,
                         "article_changed":int(on_article.argmax())!=off_id,"stochastic":stochastic})
        print(f"{name} normalized pair {pair_index}/{len(PAIRS)}",flush=True)
    del rm;gc.collect();return rows,{"selected_layer":layer,"natural_residual_rms":natural_rms}


def main():
    RESULTS.mkdir(parents=True,exist_ok=True);started=time.time();rows=[];metadata={}
    for name,config in MODELS.items():
        model_rows,model_meta=run_model(name,config);rows.extend(model_rows);metadata[name]=model_meta
    summary={"experiment":"cross_scale_normalized_natural_residual","generated_at":datetime.now(timezone.utc).isoformat(),"elapsed_sec":time.time()-started,"metadata":metadata,"conditions":{}}
    metrics=("baseline_target_minus_source","delta_delta","fraction_gap_closed","target_odds_multiplier","target_probability_change","target_rank_change","patch_norm_over_natural_rms","effect_per_patch_norm")
    seed=20260830
    for model in MODELS:
        for regime in ("between","within"):
            for strength in STRENGTHS:
                group=[row for row in rows if row["model"]==model and row["regime"]==regime and row["strength"]==strength];key=f"{model}__{regime}__{strength}"
                summary["conditions"][key]={metric:interval([row[metric] for row in group if row[metric] is not None],seed+len(summary["conditions"])*20+i) for i,metric in enumerate(metrics)}
                summary["conditions"][key]["article_change_rate"]=sum(row["article_changed"] for row in group)/len(group)
                summary["conditions"][key]["stochastic"]={str(tau):{metric:interval([row["stochastic"][str(tau)][metric] for row in group],seed+500+i) for i,metric in enumerate(("public_tv","private_tv","total_tv"))} for tau in TEMPERATURES}
    write_json(RESULTS/"rows.json",rows);write_json(RESULTS/"summary.json",summary);print(json.dumps({"elapsed_sec":summary["elapsed_sec"],"metadata":metadata},indent=2))


if __name__=="__main__":main()
