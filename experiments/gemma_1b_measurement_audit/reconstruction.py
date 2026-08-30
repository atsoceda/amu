#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from experiments.lib.aan_protocol import slugify, token_id_for_text, write_json
from experiments.lib.core import load_replacement_model
from experiments.six_cell_family_sweep.run import next_logits

EXP=Path(__file__).resolve().parent;RESULTS=EXP/"results"
SOURCE=json.loads((ROOT/"experiments/gemma_1b_sparse_scale/config.json").read_text())
WEIGHTS=Path(SOURCE["transcoder_weight_snapshot"])/"clt/width_262k_l0_medium_affine"
GRAPH_DIR=ROOT/"experiments/gemma_1b_sparse_scale/results/graphs"
LAYERS=(0,14,15,18,25)
SENTENCES=(SOURCE["selection_sentences"][0],SOURCE["selection_sentences"][1],SOURCE["selection_sentences"][3],SOURCE["selection_sentences"][6])


def cosine(left,right):
    denominator=float(left.norm()*right.norm());return float(torch.dot(left,right)/denominator) if denominator else 0.0


def nmse(prediction,target):
    return float(((prediction-target)**2).mean()/(target.float().var(unbiased=False)+1e-12))


class NativeCapture:
    def __init__(self):
        path=SOURCE["model_snapshot"];self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True).eval();self.layers=self.model.model.layers

    @torch.inference_mode()
    def run(self,prompt,position,capture=False):
        inputs={};outputs={};hooks=[]
        if capture:
            for layer,module in enumerate(self.layers):
                def save_input(_module,_args,value,layer=layer):inputs[layer]=value[0,position].detach().float().cpu()
                def save_output(_module,_args,value,layer=layer):outputs[layer]=value[0,position].detach().float().cpu()
                hooks.extend([module.pre_feedforward_layernorm.register_forward_hook(save_input),module.post_feedforward_layernorm.register_forward_hook(save_output)])
        try:logits=self.model(**self.tokenizer(prompt,return_tensors="pt",add_special_tokens=True),use_cache=False).logits[0,-1].detach().float().cpu()
        finally:
            for hook in hooks:hook.remove()
        return logits,inputs,outputs


def main():
    RESULTS.mkdir(parents=True,exist_ok=True);started=time.time();native=NativeCapture();tok=native.tokenizer
    records=[];states={};native_logits={}
    for sentence in SENTENCES:
        meta=json.loads((GRAPH_DIR/f"{slugify(sentence)}__meta.json").read_text());article=meta["article_prompt"];future=meta["future_prompt"]
        position=len(tok(article,add_special_tokens=True).input_ids)-1
        article_logits,inputs,outputs=native.run(article,position,True);future_logits,_,_=native.run(future,position,False)
        states[sentence]={"inputs":inputs,"outputs":outputs};native_logits[sentence]={"article":article_logits,"future":future_logits,"meta":meta}
    del native;gc.collect()

    sparse={(sentence,layer):torch.zeros(1152) for sentence in SENTENCES for layer in LAYERS}
    skip={}
    for input_layer in range(26):
        path=WEIGHTS/f"params_layer_{input_layer}.safetensors"
        with safe_open(path,framework="pt",device="cpu") as sf:
            w_enc=sf.get_tensor("w_enc").float();b_enc=sf.get_tensor("b_enc").float();threshold=sf.get_tensor("threshold").float()
            if input_layer in LAYERS:
                affine=sf.get_tensor("affine_skip_connection").float();b_dec=sf.get_tensor("b_dec").float()
                for sentence in SENTENCES:
                    skip[(sentence,input_layer)]=states[sentence]["inputs"][input_layer]@affine
                    sparse[(sentence,input_layer)]+=b_dec
            target_layers=[layer for layer in LAYERS if layer>=input_layer]
            if not target_layers:continue
            activations={}
            for sentence in SENTENCES:
                pre=states[sentence]["inputs"][input_layer]@w_enc+b_enc;activations[sentence]=pre*(pre>threshold)
            for target_layer in target_layers:
                decoder=sf.get_slice("w_dec")[:,target_layer-input_layer,:].float()
                for sentence in SENTENCES:sparse[(sentence,target_layer)]+=activations[sentence]@decoder
        print(f"reconstructed source layer {input_layer+1}/26",flush=True)

    for sentence in SENTENCES:
        for layer in LAYERS:
            target=states[sentence]["outputs"][layer];s=sparse[(sentence,layer)];k=skip[(sentence,layer)];recon=s+k
            records.append({"sentence":sentence,"layer":layer,"reconstruction_cosine":cosine(recon,target),"normalized_mse":nmse(recon,target),
                            "sparse_only_cosine":cosine(s,target),"skip_only_cosine":cosine(k,target),
                            "sparse_norm_fraction":float(s.norm()/(recon.norm()+1e-12)),"skip_norm_fraction":float(k.norm()/(recon.norm()+1e-12)),
                            "sparse_skip_cosine":cosine(s,k)})
    del states,sparse,skip;gc.collect()

    replacement=load_replacement_model(SOURCE);a_id=token_id_for_text(replacement.tokenizer," a");an_id=token_id_for_text(replacement.tokenizer," an")
    logit_rows=[]
    for sentence in SENTENCES:
        meta=native_logits[sentence]["meta"];native_article=native_logits[sentence]["article"];native_future=native_logits[sentence]["future"]
        replacement_article=next_logits(replacement,meta["article_prompt"],[]);replacement_future=next_logits(replacement,meta["future_prompt"],[])
        content_id=token_id_for_text(replacement.tokenizer,meta["content_token_text"])
        logit_rows.append({"sentence":sentence,"native_article_margin":float(native_article[an_id]-native_article[a_id]),
                           "replacement_article_margin":float(replacement_article[an_id]-replacement_article[a_id]),
                           "article_margin_error":float((replacement_article[an_id]-replacement_article[a_id])-(native_article[an_id]-native_article[a_id])),
                           "native_future_target_logit":float(native_future[content_id]),"replacement_future_target_logit":float(replacement_future[content_id]),
                           "future_target_logit_error":float(replacement_future[content_id]-native_future[content_id]),
                           "article_logit_cosine":cosine(replacement_article,native_article),"future_logit_cosine":cosine(replacement_future,native_future)})
    summary={"experiment":"gemma_1b_clt_reconstruction_audit","generated_at":datetime.now(timezone.utc).isoformat(),"elapsed_sec":time.time()-started,
             "sentences":list(SENTENCES),"layers":list(LAYERS),"position":"pre-article",
             "mean_reconstruction_cosine":sum(row["reconstruction_cosine"] for row in records)/len(records),
             "mean_normalized_mse":sum(row["normalized_mse"] for row in records)/len(records),
             "mean_skip_norm_fraction":sum(row["skip_norm_fraction"] for row in records)/len(records),
             "mean_sparse_norm_fraction":sum(row["sparse_norm_fraction"] for row in records)/len(records),
             "mean_abs_article_margin_error":sum(abs(row["article_margin_error"]) for row in logit_rows)/len(logit_rows),
             "mean_abs_future_target_logit_error":sum(abs(row["future_target_logit_error"]) for row in logit_rows)/len(logit_rows)}
    write_json(RESULTS/"reconstruction_rows.json",records);write_json(RESULTS/"logit_reconstruction_rows.json",logit_rows);write_json(RESULTS/"reconstruction_summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
