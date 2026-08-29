#!/usr/bin/env python3
"""Matched no-transcoder 270M replay of the 1B natural-patch assay."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from experiments.gemma_1b_residual_scale.run import ResidualModel, first_id, mixture, stats, tv
from experiments.lib.aan_protocol import token_id_for_text

EXP=Path(__file__).resolve().parent; ROOT=EXP.parents[1]; OUT=EXP/"results/rows_270m_matched.json"
cfg=json.loads((EXP/"config.json").read_text()); natural=json.loads((EXP/cfg["natural_config_path"]).resolve().read_text())
snapshot="/Users/anthony/.cache/huggingface/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
rm=ResidualModel(snapshot,torch.bfloat16); tok=rm.tokenizer; article_ids={a:token_id_for_text(tok,f" {a}") for a in ("a","an")}
demo=cfg["demonstration"]; layer=12; rows=[]; tests=[p for p in natural["pairs"] if p["split"]=="test"]
for i,pair in enumerate(tests,1):
    sp=f"{demo} {pair['source_sentence']}"; tp=f"{demo} {pair['target_sentence']}"; pos=len(tok(sp,add_special_tokens=True).input_ids)-1
    tpos=len(tok(tp,add_special_tokens=True).input_ids)-1; source=rm.states(sp,pos)[layer]; target=rm.states(tp,tpos)[layer]
    article_off=rm.logits(sp); article_on=rm.logits(sp,(layer,pos,target)); off_id=int(article_off.argmax()); on_id=int(article_on.argmax())
    branches_off={a:rm.logits(sp+tok.decode([article_ids[a]])) for a in article_ids}
    branches_on={a:rm.logits(sp+tok.decode([article_ids[a]]),(layer,pos,target)) for a in article_ids}
    sid=first_id(tok,pair["source_word"]); tid=first_id(tok,pair["target_word"]); native=pair["source_article"]
    bs=stats(branches_off[native],sid,tid); ps=stats(branches_on[native],sid,tid)
    stochastic={}
    for tau in cfg["temperatures"]:
        off,om=mixture(article_off,branches_off,article_ids,float(tau)); on,nm=mixture(article_on,branches_on,article_ids,float(tau)); public,_=mixture(article_on,branches_off,article_ids,float(tau))
        stochastic[str(tau)]={"total_tv":float(.5*(on-off).abs().sum()),"public_tv":float(.5*(public-off).abs().sum()),
            "private_tv":float(.5*(on-public).abs().sum()),"off_article_mass":om,"on_article_mass":nm}
    rows.append({"pair_id":pair["id"],"regime":pair["regime"],"layer":layer,"relative_depth":layer/17,
        "article_changed":off_id!=on_id,"target_delta_delta":ps["target_minus_source"]-bs["target_minus_source"],
        "private_tv_native":tv(branches_on[native],branches_off[native]),"target_top1_after":ps["target_top1"],"stochastic":stochastic})
    print(f"270M matched {i}/{len(tests)}",flush=True)
OUT.write_text(json.dumps(rows,indent=2)+"\n")
