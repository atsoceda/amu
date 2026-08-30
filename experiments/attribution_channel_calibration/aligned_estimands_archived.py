#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.lib.aan_protocol import token_id_for_text, write_json

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
RESULTS = EXP / "results"
CONFIG = json.loads((EXP / "config.json").read_text())
SOURCE = json.loads((EXP / CONFIG["e1_config_path"]).resolve().read_text())
MANIFEST = json.loads((ROOT / "experiments/archived_feature_vectors/gemma_scope_2_270m_pt_affine_discussed.manifest.json").read_text())
VECTORS = ROOT / "experiments/archived_feature_vectors/gemma_scope_2_270m_pt_affine_discussed.safetensors"


def first_id(tokenizer, word):
    ids = tokenizer(f" {word}", add_special_tokens=False).input_ids
    if not ids: raise ValueError(word)
    return int(ids[0])


def rankdata(values):
    order=sorted(range(len(values)),key=lambda i:values[i]); ranks=[0.0]*len(values); cursor=0
    while cursor<len(order):
        end=cursor+1
        while end<len(order) and values[order[end]]==values[order[cursor]]: end+=1
        rank=(cursor+end-1)/2+1
        for index in order[cursor:end]: ranks[index]=rank
        cursor=end
    return np.asarray(ranks)


def rho(left,right):
    x,y=rankdata(left),rankdata(right);x-=x.mean();y-=y.mean();den=float(np.linalg.norm(x)*np.linalg.norm(y))
    return None if den==0 else float(np.dot(x,y)/den)


def bootstrap(left,right,seed,resamples=10000):
    rng=random.Random(seed);n=len(left);draws=[]
    for _ in range(resamples):
        indices=[rng.randrange(n) for _ in range(n)];value=rho([left[i] for i in indices],[right[i] for i in indices])
        if value is not None:draws.append(value)
    draws.sort();return {"rho":rho(left,right),"lo":draws[int(.025*(len(draws)-1))],"hi":draws[int(.975*(len(draws)-1))],"n":n,"resamples":resamples}


class NativeFeatureModel:
    def __init__(self):
        path=CONFIG["model_snapshot"]
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True).eval()
        self.layers=self.model.model.layers

    @torch.inference_mode()
    def logits(self,prompt,deltas=None,position=None):
        hooks=[]
        if deltas:
            for layer,delta in deltas.items():
                def add_delta(_module,_inputs,output,delta=delta):
                    changed=output.clone();changed[:,position,:]+=delta.to(changed.device,changed.dtype);return changed
                hooks.append(self.layers[layer].post_feedforward_layernorm.register_forward_hook(add_delta))
        try:return self.model(**self.tokenizer(prompt,return_tensors="pt",add_special_tokens=True),use_cache=False).logits[0,-1].detach().float().cpu()
        finally:
            for hook in hooks:hook.remove()

    @torch.inference_mode()
    def feature_inputs(self,prompt,position):
        captured={};hooks=[]
        for layer,module in enumerate(self.layers):
            def save(_module,_inputs,output,layer=layer):captured[layer]=output[0,position].detach().float().cpu()
            hooks.append(module.pre_feedforward_layernorm.register_forward_hook(save))
        try:self.model(**self.tokenizer(prompt,return_tensors="pt",add_special_tokens=True),use_cache=False)
        finally:
            for hook in hooks:hook.remove()
        return captured


def main():
    started=time.time();model=NativeFeatureModel();tokenizer=model.tokenizer
    a_id,an_id=token_id_for_text(tokenizer," a"),token_id_for_text(tokenizer," an")
    selection=json.loads((RESULTS/"selection.json").read_text())["features"]
    archived={(row["layer"],row["feature_idx"]):row for row in MANIFEST["features"] if any(role.startswith("calibration_") for role in row["roles"])}
    tensors={}
    with safe_open(VECTORS,framework="pt",device="cpu") as sf:
        for feature in selection:
            key=(feature["layer"],feature["feature_idx"]);meta=archived[key]
            tensors[key]={"encoder":sf.get_tensor(meta["encoder_key"]).float(),"decoder":sf.get_tensor(meta["decoder_key"]).float(),
                          "b_enc":float(sf.get_tensor(meta["b_enc_key"])),"threshold":float(sf.get_tensor(meta["threshold_key"]))}
    rows=[];gain=float(CONFIG["amplify_factor"])
    for prompt_index,example in enumerate(SOURCE["test_examples"],1):
        prompt=f"{CONFIG['demonstration']} {example['sentence']}";position=len(tokenizer(prompt,add_special_tokens=True).input_ids)-1
        inputs=model.feature_inputs(prompt,position);off_article=model.logits(prompt)
        baseline_margin=float(off_article[an_id]-off_article[a_id])
        off_branches={"a":model.logits(prompt+tokenizer.decode([a_id])),"an":model.logits(prompt+tokenizer.decode([an_id]))}
        target_id=first_id(tokenizer,example["listed_word"]);expected=example["expected_article"];source_id=int(off_branches[expected].argmax())
        branch_probs={key:torch.softmax(value,-1) for key,value in off_branches.items()};branch_leverage=float(.5*(branch_probs["an"]-branch_probs["a"]).abs().sum())
        for feature_index,feature in enumerate(selection):
            layer,feature_idx=int(feature["layer"]),int(feature["feature_idx"]);weights=tensors[(layer,feature_idx)]
            preactivation=float(torch.dot(inputs[layer],weights["encoder"])+weights["b_enc"])
            activation=preactivation if preactivation>weights["threshold"] else 0.0;delta_activation=(gain-1)*activation
            deltas={output_layer:delta_activation*weights["decoder"][output_layer-layer] for output_layer in range(layer,len(model.layers))}
            on_article=model.logits(prompt,deltas,position);article_effect=float((on_article[an_id]-on_article[a_id])-baseline_margin)
            expected_id=a_id if expected=="a" else an_id
            on_expected=model.logits(prompt+tokenizer.decode([expected_id]),deltas,position)
            target_change=float(on_expected[target_id]-off_branches[expected][target_id]);source_change=float(on_expected[source_id]-off_branches[expected][source_id])
            rows.append({"feature_index":feature_index,"prompt_index":prompt_index,"layer":layer,"feature_idx":feature_idx,
                         "article_attribution":feature["article_attribution"],"future_attribution":feature["future_attribution"],
                         "activation_archived_reconstruction":activation,"baseline_margin":baseline_margin,"article_margin_effect":article_effect,
                         "target_logit_change_fixed_expected":target_change,"source_logit_change_fixed_expected":source_change,
                         "target_minus_source_change_fixed_expected":target_change-source_change,"branch_leverage_tv":branch_leverage})
        print(f"completed archived aligned prompt {prompt_index}/{len(SOURCE['test_examples'])}",flush=True)
    feature_rows=[]
    old=json.loads((RESULTS/"feature_rows.json").read_text())
    for index,feature in enumerate(selection):
        group=[row for row in rows if row["feature_index"]==index];mean=lambda key:sum(float(row[key]) for row in group)/len(group)
        prior=next(row for row in old if row["layer"]==feature["layer"] and row["feature_idx"]==feature["feature_idx"])
        feature_rows.append({**feature,"article_margin_effect_direct_patch":mean("article_margin_effect"),
                             "article_margin_effect_replacement_model":prior["article_margin_effect"],
                             "target_logit_change_fixed_expected":mean("target_logit_change_fixed_expected"),
                             "target_minus_source_change_fixed_expected":mean("target_minus_source_change_fixed_expected"),
                             "fixed_target_effect_magnitude":sum(abs(row["target_logit_change_fixed_expected"]) for row in group)/len(group)})
    validation={"rho_direct_patch_vs_replacement_article_effect":rho([row["article_margin_effect_direct_patch"] for row in feature_rows],
                                                                      [row["article_margin_effect_replacement_model"] for row in feature_rows]),
                "mean_absolute_article_effect_error":sum(abs(row["article_margin_effect_direct_patch"]-row["article_margin_effect_replacement_model"]) for row in feature_rows)/len(feature_rows)}
    analyses={"signed_future_vs_signed_fixed_target":bootstrap([row["future_attribution"] for row in feature_rows],[row["target_logit_change_fixed_expected"] for row in feature_rows],20260830),
              "signed_future_vs_signed_fixed_target_minus_source":bootstrap([row["future_attribution"] for row in feature_rows],[row["target_minus_source_change_fixed_expected"] for row in feature_rows],20260831),
              "absolute_future_vs_fixed_target_magnitude":bootstrap([abs(row["future_attribution"]) for row in feature_rows],[row["fixed_target_effect_magnitude"] for row in feature_rows],20260832)}
    summary={"experiment":"gemma_270m_aligned_attribution_estimands_from_archived_vectors","generated_at":datetime.now(timezone.utc).isoformat(),
             "elapsed_sec":time.time()-started,"n_features":len(feature_rows),"n_prompts":len(SOURCE["test_examples"]),"gain":gain,
             "intervention":"Archived encoder/decoder vectors applied directly to native post-feedforward outputs.","validation":validation,"analyses":analyses}
    write_json(RESULTS/"aligned_rows.json",rows);write_json(RESULTS/"aligned_feature_rows.json",feature_rows);write_json(RESULTS/"aligned_summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
