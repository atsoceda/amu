#!/usr/bin/env python3
from __future__ import annotations
import json, math, random
from pathlib import Path

EXP=Path(__file__).resolve().parent; ROOT=EXP.parents[1]

def load(p): return json.loads(Path(p).read_text())
def ci(xs,seed=20260901,nboot=10000):
    rng=random.Random(seed); n=len(xs); b=sorted(sum(xs[rng.randrange(n)] for _ in range(n))/n for _ in range(nboot))
    return {"n":n,"mean":sum(xs)/n,"lo":b[math.floor(.025*(nboot-1))],"hi":b[math.ceil(.975*(nboot-1))]}

rows=load(EXP/"results/rows.json"); layers=load(EXP/"results/layer_rows.json")
rows270=load(EXP/"results/rows_270m_matched.json")
one=[r for r in rows if r["strength"]==1.0]
stochastic={}
for regime in ("between","within"):
    group=[r for r in one if r["regime"]==regime]
    stochastic[regime]={}
    for tau in ("0.1","0.25","0.5","1.0"):
        stochastic[regime][tau]={k:ci([r["stochastic"][tau][k] for r in group],20260901+int(float(tau)*100)+i)
            for i,k in enumerate(("total_tv","public_tv","private_tv","off_article_mass","on_article_mass"))}

stochastic270={}
for regime in ("between","within"):
    group=[r for r in rows270 if r["regime"]==regime]; stochastic270[regime]={}
    for tau in ("0.1","0.25","0.5","1.0"):
        stochastic270[regime][tau]={k:ci([r["stochastic"][tau][k] for r in group],20261901+int(float(tau)*100)+i)
            for i,k in enumerate(("total_tv","public_tv","private_tv","off_article_mass","on_article_mass"))}

test_profile={}
for layer in sorted({r["layer"] for r in layers}):
    group=[r for r in layers if r["split"]=="test" and r["layer"]==layer]
    test_profile[str(layer)]={"relative_depth":group[0]["relative_depth"],
        "target_delta_delta":ci([r["delta_delta"] for r in group],20261000+layer),
        "fixed_tv":ci([r["fixed_tv"] for r in group],20262000+layer),
        "target_top1_rate":sum(r["target_top1_after"] for r in group)/len(group)}

old=load(ROOT/"experiments/natural_residual_carrier_regimes/results/summary.json")
new=load(EXP/"results/summary.json")
comparison={}
for regime in ("between","within"):
    n=new["conditions"][f"{regime}_1.0"]
    o=old["regimes"][regime]
    comparison[regime]={
        "gemma_270m":{"selected_layer":old["selected_layer"],"relative_depth":old["selected_layer"]/17,
            "target_delta_delta":o["target_delta_delta"],"private_tv":o["residual_tv"],
            "article_change_rate":o["article_change_rate"],"target_top1_rate":0.0},
        "gemma_1b":{"selected_layer":new["selected_layer"],"relative_depth":new["selected_relative_depth"],
            "target_delta_delta":n["target_delta_delta"],"private_tv":n["private_tv"],
            "article_change_rate":n["article_change_rate"],"target_top1_rate":n["target_top1_rate"]}}

paired_scale={}
for regime in ("between","within"):
    nrows=[r for r in one if r["regime"]==regime]; orows=[r for r in rows270 if r["regime"]==regime]
    old_by={r["pair_id"]:r for r in orows}; paired_scale[regime]={}
    paired_scale[regime]["target_delta_delta_1b_minus_270m"]=ci([r["target_delta_delta"]-old_by[r["pair_id"]]["target_delta_delta"] for r in nrows],20263001)
    paired_scale[regime]["private_tv_1b_minus_270m"]=ci([r["private_tv"]-old_by[r["pair_id"]]["private_tv_native"] for r in nrows],20263002)
    paired_scale[regime]["stochastic_tau1"]={k:ci([r["stochastic"]["1.0"][k]-old_by[r["pair_id"]]["stochastic"]["1.0"][k] for r in nrows],20263010+i)
        for i,k in enumerate(("total_tv","public_tv","private_tv"))}

out={"stochastic_natural_strength":{"gemma_270m":stochastic270,"gemma_1b":stochastic},"test_layer_profile":test_profile,"cross_scale_natural_strength":comparison,"paired_scale_differences":paired_scale,
     "scope":"Two-point cross-scale contrast of the same natural full-residual intervention; not a scaling law."}
(EXP/"results/derived_analysis.json").write_text(json.dumps(out,indent=2)+"\n")
lines=["# Gemma 270M--1B residual comparison","","Natural-strength target patch:","",
       "| Regime | Model | Layer (relative) | Target ΔΔ | Private TV | Article change | Target top-1 |","| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
for regime in ("between","within"):
    for model,b in comparison[regime].items():
        lines.append(f"| {regime} | {model} | {b['selected_layer']} ({b['relative_depth']:.3f}) | {b['target_delta_delta']['mean']:.3f} | {b['private_tv']['mean']:.3f} | {b['article_change_rate']:.2f} | {b['target_top1_rate']:.2f} |")
lines.extend(["","At natural strength neither model changes the article. The 1B target contrast is modestly larger, especially within class, but distributional private TV is not uniformly larger. This is stronger private target efficacy, not evidence of a public/private carrier reallocation.",""])
(EXP/"results/cross_scale_report.md").write_text("\n".join(lines))
