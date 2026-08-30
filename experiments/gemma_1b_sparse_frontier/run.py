#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

import torch
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from experiments.lib.aan_protocol import write_json
from experiments.lib.core import load_replacement_model,token_id_for_text
from experiments.six_cell_family_sweep.run import activations_at_position,next_logits

EXP=Path(__file__).resolve().parent;RESULTS=EXP/"results"
CONFIG=json.loads((ROOT/"experiments/gemma_1b_effect_matched/config.json").read_text())
SOURCE=json.loads((ROOT/"experiments/gemma_1b_sparse_scale/config.json").read_text())
RANKED=json.loads((ROOT/"experiments/gemma_1b_effect_matched/results/margin_ranked_features.json").read_text())
FEATURES=[row for row in RANKED if row["prompt_count"]>=3]
K_VALUES=(1,2,4,8,12,16,20,24)
GAIN_VALUES=(1.25,1.5,2.0,3.0,4.0,5.0,7.5,10.0,15.0,20.0)
MAX_MEAN_KL=0.1;MIN_MEAN_ARTICLE_MASS=0.5


def decoder_norms(features):
    root=Path(CONFIG["transcoder_weight_snapshot"])/"clt/width_262k_l0_medium_affine";by_layer={};out={}
    for feature in features:by_layer.setdefault(int(feature["layer"]),[]).append(int(feature["feature_idx"]))
    for layer,ids in by_layer.items():
        with safe_open(root/f"params_layer_{layer}.safetensors",framework="pt",device="cpu") as sf:
            decoder=sf.get_slice("w_dec")[ids,:,:].float().flatten(1)
            for row_index,feature_idx in enumerate(ids):out[(layer,feature_idx)]=float(decoder[row_index].norm())
    return out


def kl(on_logits,off_logits):
    log_on=torch.log_softmax(on_logits,-1);log_off=torch.log_softmax(off_logits,-1);on=torch.softmax(on_logits,-1)
    return float((on*(log_on-log_off)).sum())


def main():
    RESULTS.mkdir(parents=True,exist_ok=True);started=time.time();norms=decoder_norms(FEATURES);model=load_replacement_model(CONFIG);tok=model.tokenizer
    a_id,an_id=token_id_for_text(tok," a"),token_id_for_text(tok," an");article_ids={a_id,an_id};rows=[]
    for prompt_index,sentence in enumerate(SOURCE["selection_sentences"],1):
        prompt=f"{CONFIG['demonstration']} {sentence}";position=len(tok(prompt,add_special_tokens=True).input_ids)-1
        activations=activations_at_position(model,prompt,position);off=next_logits(model,prompt,[]);off_margin=float(off[an_id]-off[a_id])
        for k in K_VALUES:
            selected=FEATURES[:k]
            for gain in GAIN_VALUES:
                interventions=[];norm_sq=0.0;active=0
                for feature in selected:
                    layer,feature_idx=int(feature["layer"]),int(feature["feature_idx"]);activation=float(activations[layer,feature_idx].float().cpu())
                    if activation>0:active+=1
                    interventions.append({"layer":layer,"pos":position,"feature_idx":feature_idx,"value":activation*gain})
                    norm_sq+=((gain-1)*activation*norms[(layer,feature_idx)])**2
                on=next_logits(model,prompt,interventions);probs=torch.softmax(on,-1);top=int(on.argmax())
                rows.append({"prompt_index":prompt_index,"sentence":sentence,"k":k,"gain":gain,"baseline_margin":off_margin,
                             "margin_movement":float((on[an_id]-on[a_id])-off_margin),"mediator_valid":top in article_ids,
                             "top_token":tok.decode([top]),"article_mass":float(probs[a_id]+probs[an_id]),"kl_on_from_off":kl(on,off),
                             "residual_norm_proxy":math.sqrt(norm_sq),"active_features":active})
        print(f"frontier development prompt {prompt_index}/{len(SOURCE['selection_sentences'])}",flush=True)
    settings=[]
    for k in K_VALUES:
        for gain in GAIN_VALUES:
            group=[row for row in rows if row["k"]==k and row["gain"]==gain]
            settings.append({"k":k,"gain":gain,"mean_margin_movement":sum(row["margin_movement"] for row in group)/len(group),
                             "mediator_valid_rate":sum(row["mediator_valid"] for row in group)/len(group),
                             "mean_article_mass":sum(row["article_mass"] for row in group)/len(group),
                             "mean_kl":sum(row["kl_on_from_off"] for row in group)/len(group),
                             "max_kl":max(row["kl_on_from_off"] for row in group),"mean_residual_norm_proxy":sum(row["residual_norm_proxy"] for row in group)/len(group),
                             "min_active_features":min(row["active_features"] for row in group)})
    eligible=[setting for setting in settings if setting["mediator_valid_rate"]==1 and setting["mean_kl"]<=MAX_MEAN_KL and setting["mean_article_mass"]>=MIN_MEAN_ARTICLE_MASS]
    chosen=max(eligible,key=lambda setting:(setting["mean_margin_movement"],-setting["mean_residual_norm_proxy"],-setting["k"])) if eligible else None
    # Pareto points maximize efficacy while minimizing structural cost (invalidity, KL, norm).
    pareto=[]
    for setting in settings:
        cost=(1-setting["mediator_valid_rate"],setting["mean_kl"],setting["mean_residual_norm_proxy"])
        dominated=False
        for other in settings:
            other_cost=(1-other["mediator_valid_rate"],other["mean_kl"],other["mean_residual_norm_proxy"])
            if other["mean_margin_movement"]>=setting["mean_margin_movement"] and all(a<=b for a,b in zip(other_cost,cost)) and (other["mean_margin_movement"]>setting["mean_margin_movement"] or any(a<b for a,b in zip(other_cost,cost))):dominated=True;break
        if not dominated:pareto.append(setting)
    summary={"experiment":"gemma_1b_sparse_mediator_valid_frontier","generated_at":datetime.now(timezone.utc).isoformat(),"elapsed_sec":time.time()-started,
             "selection":"Corrected positive an-minus-a and future attribution ranking; development prompts only.","constraints":{"mediator_valid_rate":1.0,"max_mean_kl":MAX_MEAN_KL,"min_mean_article_mass":MIN_MEAN_ARTICLE_MASS},
             "settings":settings,"eligible_count":len(eligible),"chosen":chosen,"pareto":pareto}
    write_json(RESULTS/"development_rows.json",rows);write_json(RESULTS/"summary.json",summary);print(json.dumps({"eligible_count":len(eligible),"chosen":chosen,"pareto_count":len(pareto)},indent=2))


if __name__=="__main__":main()
