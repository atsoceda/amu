#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
S=json.loads((Path(__file__).parent/"results/summary.json").read_text())
OUT=ROOT/"manuscript/figures/fig_is_are_carrier_capacity.png"

fig,axes=plt.subplots(1,2,figsize=(9.2,3.55)); colors={"gemma_270m":"#2878B5","gemma_1b":"#D95319"}
labels={"gemma_270m":"270M","gemma_1b":"1B"}; regimes=("between","within"); models=("gemma_270m","gemma_1b")
ax=axes[0];x=np.arange(2);width=.34
for offset,model in zip((-.17,.17),models):
 vals=[];lo=[];hi=[]
 for regime in regimes:
  d=S["conditions"][f"{model}_{regime}_1.0"]["target_delta_delta_fixed_native"];vals.append(d["mean"]);lo.append(d["mean"]-d["lo"]);hi.append(d["hi"]-d["mean"])
 ax.bar(x+offset,vals,width,color=colors[model],label=labels[model],yerr=[lo,hi],capsize=4)
ax.axhline(0,color="black",lw=.8);ax.set_xticks(x,["1→3\n(is→are)","3→5\n(are→are)"]);ax.set(ylabel="Fixed-verb target $\\Delta\\Delta$",title="A  Target-aligned local efficacy");ax.legend(frameon=False)
ax=axes[1];x=np.arange(4);width=.34
conditions=[(m,r) for m in models for r in regimes]
pub=[];priv=[];names=[]
for model,regime in conditions:
 d=S["conditions"][f"{model}_{regime}_1.0"]["stochastic"]["1.0"];pub.append(d["public_tv"]["mean"]);priv.append(d["private_tv"]["mean"]);names.append(f"{labels[model]}\n{regime}")
ax.bar(x-width/2,pub,width,label="Public",color="#3b82f6");ax.bar(x+width/2,priv,width,label="Matched-policy private",color="#8b5cf6")
ax.set_xticks(x,names);ax.set(ylabel="TV at $\\tau=1$",title="B  Distribution movement without target control");ax.legend(frameon=False,fontsize=8)
for ax in axes:ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.18)
fig.tight_layout();fig.savefig(OUT,dpi=220,bbox_inches="tight");print(OUT)
