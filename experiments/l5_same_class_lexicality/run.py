#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, random, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch

from experiments.lib.aan_protocol import token_id_for_text, write_json
from experiments.lib.core import load_replacement_model, setup_file_logging
from experiments.lib.mediation_estimands import total_variation_from_logits
from experiments.six_cell_family_sweep.run import activations_at_position, build_interventions, next_logits

EXP=Path(__file__).resolve().parent; RESULTS=EXP/"results"
def load(p): return json.loads(Path(p).read_text())
def first_id(tok,word): return int(tok(" "+word,add_special_tokens=False).input_ids[0])
def ci(xs,seed,nboot):
    rng=random.Random(seed); n=len(xs); b=sorted(sum(xs[rng.randrange(n)] for _ in range(n))/n for _ in range(nboot))
    return {"n":n,"mean":sum(xs)/n,"lo":b[math.floor(.025*(nboot-1))],"hi":b[math.ceil(.975*(nboot-1))],"resamples":nboot,"method":"prompt-level nonparametric bootstrap"}

def occupation_classes(tok,path):
    classes={"consonant":set(),"vowel":set()}; vowels=set("aeiou")
    with Path(path).open(newline="") as f:
        for r in csv.DictReader(f):
            word=r["word"].strip().lower()
            if word: classes["vowel" if word[0] in vowels else "consonant"].add(first_id(tok,word))
    return {k:sorted(v) for k,v in classes.items()}

def main():
    cfg=load(EXP/"config.json"); RESULTS.mkdir(parents=True,exist_ok=True); setup_file_logging(RESULTS); started=time.time()
    model=load_replacement_model(cfg); tok=model.tokenizer; target=cfg["target_feature"]; controls=load((EXP/cfg["control_path"]).resolve())
    features=[{"id":target["id"],"layer":target["layer"],"feature_idx":target["feature_idx"],"kind":"target"}]+[
        {"id":f"L5/F{x['feature_idx']}","layer":x["layer"],"feature_idx":x["feature_idx"],"kind":"matched_feature_control"} for x in controls]
    article_ids={a:token_id_for_text(tok," "+a) for a in ("a","an")}; classes=occupation_classes(tok,(EXP/cfg["dataset_path"]).resolve())
    group_pairs={g:[p for p in cfg["pairs"] if p["group"]==g] for g in ("within_consonant","within_vowel","cross_class_reference")}
    group_target_ids={g:sorted({first_id(tok,p["target"]) for p in pairs}) for g,pairs in group_pairs.items()}
    wrong={}
    for group,pairs in group_pairs.items():
        for i,p in enumerate(pairs): wrong[p["id"]]=pairs[(i+1)%len(pairs)]["target"]
    rows=[]
    for pi,p in enumerate(cfg["pairs"],1):
        prompt=f"{cfg['demonstration']} {p['sentence']}"; pos=len(tok(prompt,add_special_tokens=True).input_ids)-1
        acts=activations_at_position(model,prompt,pos); prefix=prompt+tok.decode([article_ids[p["article"]]]); base=next_logits(model,prefix,[]); bp=torch.softmax(base,-1)
        sid=first_id(tok,p["source"]); tid=first_id(tok,p["target"]); wid=first_id(tok,wrong[p["id"]])
        cls="vowel" if p["article"]=="an" else "consonant"; other="consonant" if cls=="vowel" else "vowel"
        for gain in cfg["gains"]:
            for feat in features:
                ints,arows=build_interventions(acts,pos,[feat],float(gain)); logits=next_logits(model,prefix,ints); pp=torch.softmax(logits,-1)
                target_change=float(logits[tid]-base[tid]); source_change=float(logits[sid]-base[sid]); wrong_change=float(logits[wid]-base[wid])
                delta_logits=logits-base; other_targets=[x for x in group_target_ids[p["group"]] if x!=tid]
                other_target_mean=float(delta_logits[other_targets].mean())
                same_class_mean=float(delta_logits[classes[cls]].mean()); opposite_class_mean=float(delta_logits[classes[other]].mean())
                same_changes=delta_logits[classes[cls]]
                rows.append({"pair_id":p["id"],"group":p["group"],"sentence":p["sentence"],"source":p["source"],"target":p["target"],"wrong_target":wrong[p["id"]],
                    "article":p["article"],"gain":gain,"feature_id":feat["id"],"kind":feat["kind"],"activation":arows[0]["activation"],
                    "fixed_tv":total_variation_from_logits(logits,base),"target_logit_change":target_change,"source_logit_change":source_change,"wrong_target_logit_change":wrong_change,
                    "target_delta_delta":target_change-source_change,"wrong_delta_delta":wrong_change-source_change,
                    "lexical_selectivity":target_change-wrong_change,"all_alternatives_selectivity":target_change-other_target_mean,
                    "same_class_mean_logit_change":same_class_mean,"opposite_class_mean_logit_change":opposite_class_mean,
                    "class_mean_logit_contrast":same_class_mean-opposite_class_mean,
                    "target_change_percentile_within_class":float((same_changes<target_change).float().mean()),"target_prob_change":float(pp[tid]-bp[tid]),
                    "same_class_mass_change":float(pp[classes[cls]].sum()-bp[classes[cls]].sum()),"opposite_class_mass_change":float(pp[classes[other]].sum()-bp[classes[other]].sum())})
        print(f"pair {pi}/{len(cfg['pairs'])}",flush=True)
    nboot=cfg["bootstrap_resamples"]; seed=cfg["bootstrap_seed"]; metrics=("fixed_tv","target_logit_change","source_logit_change","wrong_target_logit_change","target_delta_delta","wrong_delta_delta","lexical_selectivity","all_alternatives_selectivity","same_class_mean_logit_change","opposite_class_mean_logit_change","class_mean_logit_contrast","target_change_percentile_within_class","target_prob_change","same_class_mass_change","opposite_class_mass_change")
    summary={"experiment":cfg["experiment_name"],"generated_at":datetime.now(timezone.utc).isoformat(),"elapsed_sec":time.time()-started,
        "target_feature":target,"control_features":controls,"wrong_target_rule":"cyclic target from another prompt in the same experimental group","conditions":{}}
    idx=0
    for group in group_pairs:
        for gain in cfg["gains"]:
            for feat in features:
                selected=[r for r in rows if r["group"]==group and r["gain"]==gain and r["feature_id"]==feat["id"]]; key=f"{group}__{gain}x__{feat['id']}"
                summary["conditions"][key]={m:ci([float(r[m]) for r in selected],seed+idx*20+j,nboot) for j,m in enumerate(metrics)}
                summary["conditions"][key].update({"n":len(selected),"active_rate":sum(r["activation"]>0 for r in selected)/len(selected),"kind":feat["kind"]}); idx+=1
    # Target feature versus the mean of matched-feature controls, paired by prompt.
    comparisons={}
    for group in group_pairs:
        for gain in cfg["gains"]:
            targets=[r for r in rows if r["group"]==group and r["gain"]==gain and r["kind"]=="target"]
            blocks={}
            for metric in ("target_delta_delta","lexical_selectivity","all_alternatives_selectivity","class_mean_logit_contrast","same_class_mass_change","fixed_tv"):
                diffs=[]
                for tr in targets:
                    cs=[r[metric] for r in rows if r["pair_id"]==tr["pair_id"] and r["gain"]==gain and r["kind"]=="matched_feature_control"]
                    diffs.append(tr[metric]-sum(cs)/len(cs))
                blocks[metric+"_target_minus_controls"]=ci(diffs,seed+900+len(comparisons)*10+len(blocks),nboot)
            comparisons[f"{group}__{gain}x"]=blocks
    summary["comparisons"]=comparisons
    write_json(RESULTS/"rows.json",rows); write_json(RESULTS/"summary.json",summary)
    def f(b): return f"{b['mean']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"
    lines=["# L5/F383 same-class lexicality test","","Target feature results:","","| Group | Gain | Target ΔΔ | Target-vs-all alternatives | Class mean logit contrast | Same-class mass Δ | Fixed TV |","| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group in group_pairs:
        for gain in cfg["gains"]:
            b=summary["conditions"][f"{group}__{gain}x__L5/F383"]
            lines.append(f"| {group} | {gain:g}x | {f(b['target_delta_delta'])} | {f(b['all_alternatives_selectivity'])} | {f(b['class_mean_logit_contrast'])} | {f(b['same_class_mass_change'])} | {f(b['fixed_tv'])} |")
    lines.extend(["","Interpretation: L5/F383 reproduces the original cross-class target contrast, but neither same-class arm shows reliable intended target control. The intended targets move less than the other pre-specified same-class alternatives, while the mean logit shift favors the noun class licensed by the fixed article under both `a` and `an`. This supports a prefix-conditioned article-class compatibility direction, not a target-specific private lexical plan.",""])
    (RESULTS/"report.md").write_text("\n".join(lines)+"\n")

if __name__=="__main__": main()
