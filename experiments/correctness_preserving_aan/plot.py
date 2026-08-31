#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXP=Path(__file__).resolve().parent
S=json.loads((EXP/"results/summary.json").read_text())
OUT=EXP.parents[1]/"manuscript/figures/fig_correctness_preserving_carriers.png"
COLORS={"between":"#2878B5","within":"#E07A2D","public":"#2878B5","private":"#8E5CC2"}


def err(block):
    return [[block["mean"]-block["lo"]],[block["hi"]-block["mean"]]]


fig,axs=plt.subplots(2,2,figsize=(9.2,6.3))

# A: local efficacy and controls.
ax=axs[0,0]; x=np.arange(3); width=.34
for j,regime in enumerate(("between","within")):
    b=S["conditions"][f"{regime}_1.0"]
    blocks=[b["target_branch_delta_delta"],b["controls"]["wrong"],b["controls"]["sign_reversed"]]
    means=[z["mean"] for z in blocks]
    yerr=np.array([[z["mean"]-z["lo"] for z in blocks],[z["hi"]-z["mean"] for z in blocks]])
    ax.bar(x+(j-.5)*width,means,width,yerr=yerr,capsize=3,color=COLORS[regime],label=regime.replace("between","Between article").replace("within","Within article"))
ax.axhline(0,color="black",lw=.7);ax.set_xticks(x,["Target patch","Wrong target","Sign reversed"]);ax.set_ylabel("Fixed-target-article lexical ΔΔ");ax.set_title("A  Held-out local efficacy");ax.legend(frameon=False,fontsize=8)

# B: route TV at tau=1.
ax=axs[0,1]; x=np.arange(2); width=.34
for j,route in enumerate(("public","private")):
    blocks=[S["conditions"][f"{regime}_1.0"]["temperatures"]["1.0"][f"{route}_tv"] for regime in ("between","within")]
    means=[z["mean"] for z in blocks];yerr=np.array([[z["mean"]-z["lo"] for z in blocks],[z["hi"]-z["mean"] for z in blocks]])
    ax.bar(x+(j-.5)*width,means,width,yerr=yerr,capsize=3,color=COLORS[route],label=route.title())
ax.set_xticks(x,["Between article","Within article"]);ax.set_ylabel("TV at τ=1");ax.set_title("B  Causal route magnitude");ax.legend(frameon=False,fontsize=8)

# C: target-aligned probability-vector components.
ax=axs[1,0];x=np.arange(2)
for j,route in enumerate(("public","private")):
    blocks=[S["conditions"][f"{regime}_1.0"]["temperatures"]["1.0"][f"{route}_target_minus_source"] for regime in ("between","within")]
    means=[z["mean"] for z in blocks];yerr=np.array([[z["mean"]-z["lo"] for z in blocks],[z["hi"]-z["mean"] for z in blocks]])
    ax.bar(x+(j-.5)*width,means,width,yerr=yerr,capsize=3,color=COLORS[route],label=route.title())
ax.axhline(0,color="black",lw=.7);ax.set_xticks(x,["Between article","Within article"]);ax.set_ylabel("Target-minus-source probability effect");ax.set_title("C  Intended lexical direction")

# D: public-minus-private interaction across policy temperatures.
ax=axs[1,1];taus=np.array([.1,.25,.5,1.0])
for strength,color in zip((.5,1.0,1.5),("#76A5D1","#2878B5","#163E63")):
    blocks=[S["interactions"][f"strength_{strength}"][str(t)]["tv_route_interaction"] for t in taus]
    means=np.array([z["mean"] for z in blocks]);lo=np.array([z["lo"] for z in blocks]);hi=np.array([z["hi"] for z in blocks])
    ax.plot(taus,means,marker="o",color=color,label=f"{strength:g}×")
    ax.fill_between(taus,lo,hi,color=color,alpha=.14)
ax.axhline(0,color="black",lw=.7);ax.set_xscale("log");ax.set_xticks(taus,["0.1","0.25","0.5","1"]);ax.set_xlabel("Article-policy temperature τ");ax.set_ylabel("Between − within route contrast");ax.set_title("D  Double-dissociation interaction");ax.legend(title="Patch",frameon=False,fontsize=8,title_fontsize=8)

for ax in axs.flat:
    ax.spines[["top","right"]].set_visible(False);ax.tick_params(labelsize=8);ax.title.set_fontsize(10)
fig.tight_layout();OUT.parent.mkdir(parents=True,exist_ok=True);fig.savefig(OUT,dpi=220,bbox_inches="tight");print(OUT)
