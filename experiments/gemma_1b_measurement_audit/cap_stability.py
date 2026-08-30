#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from circuit_tracer import attribute
from circuit_tracer.graph import Graph

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from experiments.lib.aan_protocol import slugify,token_id_for_text,write_json
from experiments.lib.core import feature_effect_map,load_replacement_model

EXP=Path(__file__).resolve().parent;RESULTS=EXP/"results";GRAPH_DIR=RESULTS/"cap_graphs"
CONFIG=json.loads((ROOT/"experiments/gemma_1b_sparse_scale/config.json").read_text())
SOURCE_GRAPH_DIR=ROOT/"experiments/gemma_1b_sparse_scale/results/graphs"
SENTENCE=CONFIG["selection_sentences"][3]
CAPS=(1200,2400)


def prompts(tokenizer):
    slug=slugify(SENTENCE);meta=json.loads((SOURCE_GRAPH_DIR/f"{slug}__meta.json").read_text())
    return meta,{"article":(meta["article_prompt"],[token_id_for_text(tokenizer," a"),token_id_for_text(tokenizer," an")]),
                 "future":(meta["future_prompt"],[token_id_for_text(tokenizer,meta["content_token_text"])])}


def run():
    GRAPH_DIR.mkdir(parents=True,exist_ok=True);model=load_replacement_model(CONFIG);_,specs=prompts(model.tokenizer);slug=slugify(SENTENCE)
    for cap in (2400,):
        for kind,(prompt,target_ids) in specs.items():
            path=GRAPH_DIR/f"{slug}__{kind}__cap{cap}.pt"
            if path.exists():print(f"reusing {path}",flush=True);continue
            print(f"attributing {kind} cap={cap}",flush=True)
            graph=attribute(prompt=prompt,model=model,attribution_targets=torch.tensor(target_ids),batch_size=int(CONFIG["batch_size"]),
                            max_feature_nodes=cap,verbose=True,offload=CONFIG.get("offload"));graph.to_pt(str(path));print(f"wrote {path}",flush=True)


def load_graph(kind,cap):
    slug=slugify(SENTENCE)
    path=(SOURCE_GRAPH_DIR/f"{slug}__{kind}.pt") if cap==1200 else (GRAPH_DIR/f"{slug}__{kind}__cap{cap}.pt")
    return Graph.from_pt(str(path))


def analyze():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(CONFIG["model_snapshot"],local_files_only=True);meta,_=prompts(tokenizer)
    a_id,an_id=token_id_for_text(tokenizer," a"),token_id_for_text(tokenizer," an");future_id=token_id_for_text(tokenizer,meta["content_token_text"])
    position=len(tokenizer(meta["article_prompt"],add_special_tokens=True).input_ids)-1;blocks={}
    for cap in CAPS:
        article,future=load_graph("article",cap),load_graph("future",cap);a=feature_effect_map(article,a_id);an=feature_effect_map(article,an_id);f=feature_effect_map(future,future_id)
        candidates=[]
        for key in set(a)&set(an)&set(f):
            layer,pos,feature_idx=key
            if pos!=position:continue
            margin=an[key]["direct_effect"]-a[key]["direct_effect"];future_effect=f[key]["direct_effect"]
            if margin>0 and future_effect>0:candidates.append({"layer":layer,"feature_idx":feature_idx,"margin_attribution":margin,"future_attribution":future_effect,
                                                               "score":min(margin,future_effect),"activation":an[key]["activation"]})
        candidates.sort(key=lambda row:row["score"],reverse=True)
        margin_mass=sum(abs(an[key]["direct_effect"]-a[key]["direct_effect"]) for key in set(a)&set(an) if key[1]==position)
        future_mass=sum(abs(row["direct_effect"]) for key,row in f.items() if key[1]==position)
        blocks[str(cap)]={"article_selected":len(article.selected_features),"article_active":len(article.active_features),"future_selected":len(future.selected_features),
                          "future_active":len(future.active_features),"prearticle_abs_margin_mass":margin_mass,"prearticle_abs_future_mass":future_mass,
                          "dual_candidate_count":len(candidates),"top20":candidates[:20]}
    reference=blocks["2400"];reference_keys={(row["layer"],row["feature_idx"]) for row in reference["top20"]}
    for cap in CAPS:
        block=blocks[str(cap)];keys={(row["layer"],row["feature_idx"]) for row in block["top20"]}
        block["top20_overlap_with_full_graph"]=len(keys&reference_keys)/20
        block["margin_mass_fraction_of_full_graph"]=block["prearticle_abs_margin_mass"]/reference["prearticle_abs_margin_mass"]
        block["future_mass_fraction_of_full_graph"]=block["prearticle_abs_future_mass"]/reference["prearticle_abs_future_mass"]
    out={"experiment":"gemma_1b_graph_cap_stability","sentence":SENTENCE,"prearticle_position":position,"caps":blocks}
    write_json(RESULTS/"cap_stability.json",out);print(json.dumps(out,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("run","analyze"));args=parser.parse_args();run() if args.mode=="run" else analyze()
