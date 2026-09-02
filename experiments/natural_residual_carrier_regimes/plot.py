#!/usr/bin/env python3
"""Main Figure 3: target-specific private influence under a fixed article."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP=Path(__file__).resolve().parent;ROOT=EXP.parents[1]
OUT=ROOT/"manuscript/figures/fig_natural_carrier_regimes.png"
BLUE,ORANGE="#1769aa","#d96c21"

def interval(values,seed):
    values=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    draws=rng.choice(values,size=(10_000,values.size),replace=True).mean(axis=1)
    return float(values.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975))

def main():
    rows270=json.loads((EXP/"results/rows.json").read_text())
    rows1b=json.loads((ROOT/"experiments/gemma_1b_residual_scale/results/rows.json").read_text())
    rows1b=[r for r in rows1b if r["strength"]==1.0]
    data={"270M":rows270,"1B":rows1b};colors={"270M":BLUE,"1B":ORANGE}
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.55),gridspec_kw={"width_ratios":[1.35,1]})

    ax=axes[0];conditions=[("target_delta_delta","Target direction"),("wrong_target_delta_delta","Wrong target"),("sign_reversed_delta_delta","Sign reversed")]
    x=np.arange(3);offsets={"270M":-.15,"1B":.15};rng=np.random.default_rng(20260902)
    for model,rows in data.items():
        for j,(key,_) in enumerate(conditions):
            vals=np.asarray([r[key] for r in rows],dtype=float);jitter=rng.uniform(-.055,.055,size=vals.size)
            ax.scatter(np.full(vals.size,x[j]+offsets[model])+jitter,vals,s=15,alpha=.38,color=colors[model])
            mean,lo,hi=interval(vals,100+j+(0 if model=="270M" else 10))
            ax.errorbar(x[j]+offsets[model],mean,yerr=[[mean-lo],[hi-mean]],fmt="o",ms=7,capsize=4,color=colors[model],label=model if j==0 else None,zorder=4)
    ax.axhline(0,color="#263238",lw=.9);ax.set_xticks(x,[label for _,label in conditions]);ax.set_ylabel("Fixed-article target-minus-source change (logits)")
    ax.set_title("A  Target-specific private control replicates",loc="left",weight="bold");ax.legend(frameon=False,title="Gemma 3")

    ax=axes[1];x=np.arange(2);width=.32
    for i,model in enumerate(("270M","1B")):
        rows=data[model];t=[r["target_logit_change"] for r in rows];s=[r["source_logit_change"] for r in rows]
        for j,(vals,label,color) in enumerate(((t,"Target enhancement",BLUE),(s,"Source change","#7b3fb2"))):
            mean,lo,hi=interval(vals,300+i*10+j);pos=x[i]+(j-.5)*width
            ax.bar(pos,mean,width,color=color,label=label if i==0 else None)
            ax.errorbar(pos,mean,yerr=[[mean-lo],[hi-mean]],fmt="none",ecolor="black",capsize=3,lw=1)
    ax.axhline(0,color="#263238",lw=.9);ax.set_xticks(x,["270M","1B"]);ax.set_ylabel("Absolute noun-logit change")
    ax.set_title("B  Mainly target enhancement",loc="left",weight="bold");ax.legend(frameon=False,fontsize=8)
    for ax in axes:ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.16);ax.tick_params(labelsize=8)
    fig.tight_layout();fig.savefig(OUT,dpi=240,bbox_inches="tight");print(OUT)

if __name__=="__main__":main()
