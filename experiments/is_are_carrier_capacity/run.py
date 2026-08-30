#!/usr/bin/env python3
"""Natural-state-derived is/are mediator-capacity experiment on the pinned paper dataset."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gemma_1b_residual_scale.run import ResidualModel, cosine, interval, stats, tv
from experiments.lib.aan_protocol import token_id_for_text, write_json

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"


def load(path): return json.loads(Path(path).read_text())


def mixture(verb_logits, branches, ids, tau):
    weights = torch.softmax(verb_logits[[ids["is"], ids["are"]]] / tau, -1)
    mass = float(torch.softmax(verb_logits / tau, -1)[[ids["is"], ids["are"]]].sum())
    return weights[0]*torch.softmax(branches["is"],-1)+weights[1]*torch.softmax(branches["are"],-1), mass, float(weights[1])


def select_prompt(rows, animal, number, original):
    found = [r for r in rows if r["animal"] == animal and int(r["number"]) == number and int(r["original"]) == original]
    if len(found) != 1: raise ValueError((animal,number,original,len(found)))
    return found[0]["prompt"].strip()


def make_pairs(rows, dev_animals):
    animals = list(dict.fromkeys(r["animal"] for r in rows)); pairs=[]
    for animal in animals:
        split = "dev" if animal in dev_animals else "test"
        for regime, source_n, target_n, original in (("between",1,3,4),("within",3,5,6)):
            pairs.append({"id":f"{regime}_{animal}","animal":animal,"split":split,"regime":regime,
                          "source_number":source_n,"target_number":target_n,
                          "source_prompt":select_prompt(rows,animal,source_n,original),
                          "target_prompt":select_prompt(rows,animal,target_n,original)})
    return pairs


def run_model(cfg, spec, pairs):
    started=time.time(); rm=ResidualModel(spec["snapshot"],getattr(torch,cfg["dtype"])); tok=rm.tokenizer
    ids={name:token_id_for_text(tok,f" {name}") for name in ("is","are")}
    # Gemma emits a shared whitespace token after is/are before the digit.
    # Clamp that formatting token so the outcome distribution is over number tokens.
    num_ids={n:token_id_for_text(tok,str(n)) for n in (1,3,5)}
    cache={}
    for pair in pairs:
        sp,tp=pair["source_prompt"],pair["target_prompt"]
        spos=len(tok(sp,add_special_tokens=True).input_ids)-1; tpos=len(tok(tp,add_special_tokens=True).input_ids)-1
        cache[pair["id"]]={"sp":sp,"tp":tp,"spos":spos,"ss":rm.states(sp,spos),"ts":rm.states(tp,tpos),
                           "sid":num_ids[pair["source_number"]],"tid":num_ids[pair["target_number"]]}
    layer_rows=[]
    for pair in pairs:
        c=cache[pair["id"]]; verb="is" if pair["source_number"]==1 else "are"; prefix=c["sp"]+tok.decode([ids[verb]])+" "
        base=rm.logits(prefix); bs=stats(base,c["sid"],c["tid"])
        for layer in range(rm.n_layers):
            patched=rm.logits(prefix,(layer,c["spos"],c["ts"][layer])); ps=stats(patched,c["sid"],c["tid"])
            layer_rows.append({"model":spec["id"],"pair_id":pair["id"],"split":pair["split"],"regime":pair["regime"],
                               "layer":layer,"relative_depth":layer/(rm.n_layers-1),"delta_delta":ps["target_minus_source"]-bs["target_minus_source"],
                               "fixed_tv":tv(patched,base)})
    dev={layer:sum(r["delta_delta"] for r in layer_rows if r["split"]=="dev" and r["layer"]==layer)/4 for layer in range(rm.n_layers)}
    selected=max(dev,key=dev.get); rows=[]; test=[p for p in pairs if p["split"]=="test"]
    for pair_index,pair in enumerate(test):
        c=cache[pair["id"]]; delta=c["ts"][selected]-c["ss"][selected]
        off_article=rm.logits(c["sp"]); off_id=int(off_article.argmax()); off_branches={v:rm.logits(c["sp"]+tok.decode([ids[v]])+" ") for v in ids}
        native="is" if pair["source_number"]==1 else "are"; bs=stats(off_branches[native],c["sid"],c["tid"])
        wrong=cache[test[(pair_index+1)%len(test)] ["id"]]["ts"][selected]
        for strength in cfg["strengths"]:
            replacement=c["ss"][selected]+float(strength)*delta
            on_article=rm.logits(c["sp"],(selected,c["spos"],replacement)); on_id=int(on_article.argmax())
            on_branches={v:rm.logits(c["sp"]+tok.decode([ids[v]])+" ",(selected,c["spos"],replacement)) for v in ids}
            ps=stats(on_branches[native],c["sid"],c["tid"])
            wrong_stats=stats(rm.logits(c["sp"]+tok.decode([ids[native]])+" ",(selected,c["spos"],wrong)),c["sid"],c["tid"])
            stochastic={}
            for tau in cfg["temperatures"]:
                off_mix,off_mass,q0=mixture(off_article,off_branches,ids,float(tau)); on_mix,on_mass,q1=mixture(on_article,on_branches,ids,float(tau))
                public_mix,_,_=mixture(on_article,off_branches,ids,float(tau)); total=on_mix-off_mix; public=public_mix-off_mix; private=on_mix-public_mix
                stochastic[str(tau)]={"delta_q_are":q1-q0,"total_tv":float(.5*total.abs().sum()),"public_tv":float(.5*public.abs().sum()),
                    "private_tv":float(.5*private.abs().sum()),"public_total_cosine":cosine(public,total),"private_total_cosine":cosine(private,total),
                    "off_verb_mass":off_mass,"on_verb_mass":on_mass}
            rows.append({"model":spec["id"],"pair_id":pair["id"],"animal":pair["animal"],"regime":pair["regime"],"strength":strength,
                         "selected_layer":selected,"relative_depth":selected/(rm.n_layers-1),"greedy_verb_off":tok.decode([off_id]).strip(),
                         "greedy_verb_on":tok.decode([on_id]).strip(),"verb_changed":off_id!=on_id,
                         "target_delta_delta_fixed_native":ps["target_minus_source"]-bs["target_minus_source"],
                         "wrong_target_delta_delta":wrong_stats["target_minus_source"]-bs["target_minus_source"],"stochastic":stochastic})
        print(f"{spec['id']} selected-layer {pair_index+1}/{len(test)}",flush=True)
    del rm
    return {"model":spec["id"],"model_name":spec["model"],"n_layers":len(dev),"selected_layer":selected,
            "selected_relative_depth":selected/(len(dev)-1),"dev_layer_means":dev,"elapsed_sec":time.time()-started},layer_rows,rows


def main():
    cfg=load(EXP/"config.json"); dataset=(EXP/cfg["dataset_path"]).resolve()
    if hashlib.sha256(dataset.read_bytes()).hexdigest()!=cfg["source"]["sha256"]: raise ValueError("dataset checksum mismatch")
    with dataset.open(newline="") as handle: source=list(csv.DictReader(handle))
    pairs=make_pairs(source,set(cfg["development_animals"])); model_summaries=[]; layer_rows=[]; rows=[]
    for spec in cfg["models"]:
        summary,lrows,rrows=run_model(cfg,spec,pairs); model_summaries.append(summary); layer_rows.extend(lrows); rows.extend(rrows)
    out={"experiment":cfg["experiment_name"],"generated_at":datetime.now(timezone.utc).isoformat(),"source":cfg["source"],
         "pair_design":{"development_animals":cfg["development_animals"],"test_animals":sorted({p['animal'] for p in pairs if p['split']=='test'}),
                        "between":"1 to 3 (is to are)","within":"3 to 5 (are to are)"},"models":model_summaries,"conditions":{}}
    seed=cfg["bootstrap_seed"]
    for model in [s["id"] for s in cfg["models"]]:
        for regime in ("between","within"):
            for strength in cfg["strengths"]:
                group=[r for r in rows if r["model"]==model and r["regime"]==regime and r["strength"]==strength]; key=f"{model}_{regime}_{strength}"
                out["conditions"][key]={"n":len(group),"target_delta_delta_fixed_native":interval([r["target_delta_delta_fixed_native"] for r in group],seed,10000),
                    "wrong_target_delta_delta":interval([r["wrong_target_delta_delta"] for r in group],seed+1,10000),
                    "verb_switch_rate":sum(r["verb_changed"] for r in group)/len(group),"stochastic":{}}
                for tau in cfg["temperatures"]:
                    out["conditions"][key]["stochastic"][str(tau)]={metric:interval([r["stochastic"][str(tau)][metric] for r in group],seed+10,10000)
                        for metric in ("delta_q_are","public_tv","private_tv","total_tv","public_total_cosine","private_total_cosine","off_verb_mass","on_verb_mass")}
    RESULTS.mkdir(parents=True,exist_ok=True); write_json(RESULTS/"pairs.json",pairs);write_json(RESULTS/"layer_rows.json",layer_rows);write_json(RESULTS/"rows.json",rows);write_json(RESULTS/"summary.json",out)
    print(json.dumps({"models":model_summaries,"condition_keys":list(out["conditions"])},indent=2))


if __name__=="__main__": main()
