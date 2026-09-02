#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

EXP=Path(__file__).resolve().parent;S=json.loads((EXP/"results/summary.json").read_text());A=json.loads((EXP/"results/analysis.json").read_text())
OUT=EXP.parents[1]/"manuscript/figures/fig_neutral_synonym_repair.png";C={"between":"#2878B5","within":"#E07A2D","public":"#2878B5","private":"#8E5CC2"}
fig,axs=plt.subplots(1,3,figsize=(9.2,3.0));width=.34
ax=axs[0];x=np.arange(2)
for j,route in enumerate(("public","private")):
    blocks=[S["conditions"][f"{r}_1.0"]["temperatures"]["1.0"][f"{route}_target_minus_source"] for r in ("between","within")]
    m=[b["mean"] for b in blocks];e=np.array([[b["mean"]-b["lo"] for b in blocks],[b["hi"]-b["mean"] for b in blocks]])
    ax.bar(x+(j-.5)*width,m,width,yerr=e,capsize=3,color=C[route],label=route.title())
ax.axhline(0,color="black",lw=.7);ax.set_xticks(x,["Between","Within"]);ax.set_ylabel("Target-minus-source effect");ax.set_title("A  Neutral-prompt routes");ax.legend(frameon=False,fontsize=8)
ax=axs[1];x=np.arange(2);blocks=[A["effects"][r]["target_aligned"]["interval"] for r in ("between","within")];m=[b["mean"] for b in blocks];e=np.array([[b["mean"]-b["lo"] for b in blocks],[b["hi"]-b["mean"] for b in blocks]])
ax.bar(x,m,yerr=e,capsize=3,color=[C["between"],C["within"]]);ax.axhline(0,color="black",lw=.7);ax.set_xticks(x,["Public − private\nBetween","Private − public\nWithin"]);ax.set_ylabel("Predicted signed simple effect");ax.set_title("B  Double dissociation")
ax=axs[2];taus=np.array([.1,.25,.5,1.0])
for strength,color in zip((.5,1.0,1.5),("#76A5D1","#2878B5","#163E63")):
    b=[S["interactions"][f"strength_{strength}"][str(t)]["aligned_route_interaction"] for t in taus];m=np.array([z["mean"] for z in b]);lo=np.array([z["lo"] for z in b]);hi=np.array([z["hi"] for z in b]);ax.plot(taus,m,marker="o",color=color,label=f"{strength:g}×");ax.fill_between(taus,lo,hi,color=color,alpha=.14)
ax.axhline(0,color="black",lw=.7);ax.set_xscale("log");ax.set_xticks(taus,[".1",".25",".5","1"]);ax.set_xlabel("Article-policy temperature");ax.set_ylabel("Between − within route contrast");ax.set_title("C  Robust interaction");ax.legend(frameon=False,fontsize=8)
for ax in axs:ax.spines[["top","right"]].set_visible(False);ax.tick_params(labelsize=8);ax.title.set_fontsize(10)
fig.tight_layout();OUT.parent.mkdir(parents=True,exist_ok=True);fig.savefig(OUT,dpi=220,bbox_inches="tight");print(OUT)
