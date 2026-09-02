#!/usr/bin/env python3
"""Figure 6/4: mediator-relative route double dissociation."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

EXP=Path(__file__).resolve().parent
S=json.loads((EXP/"results/summary.json").read_text())
ROWS=json.loads((EXP/"results/rows.json").read_text())
OUT=EXP.parents[1]/"manuscript/figures/fig_neutral_synonym_repair.png"
PUBLIC,PRIVATE="#1769aa","#7b3fb2"

def native_rows(regime):return [r for r in ROWS if r["regime"]==regime and r["strength"]==1.0]

def main():
    fig=plt.figure(figsize=(12.0,4.0));gs=fig.add_gridspec(1,3,width_ratios=(1.02,1.52,1.08),wspace=.42)
    ax=fig.add_subplot(gs[0]);ax.axis("off");ax.set_title("A  What can the article express?",loc="left",weight="bold",fontsize=11)
    examples=[(.74,"BETWEEN classes","teacher  →  educator","a  →  an",PUBLIC,"public-visible"),
              (.28,"WITHIN one class","doctor  →  physician","a  →  a",PRIVATE,"publicly collapsed")]
    for y,heading,words,articles,color,note in examples:
        ax.text(.02,y+.14,heading,color=color,weight="bold",fontsize=9.2);ax.text(.02,y,words,fontsize=9.5,weight="bold")
        ax.text(.02,y-.13,articles,fontsize=13,color=color,weight="bold");ax.text(.02,y-.23,note,fontsize=8.2,color="#455a64")
    ax=fig.add_subplot(gs[1]);values=[];labels=[];colors=[]
    for regime in ("between","within"):
        for r in native_rows(regime):
            st=r["stochastic"]["1.0"];values.append(st["public"]["target_minus_source"]-st["private"]["target_minus_source"])
            labels.append(f"{r['source_word']} → {r['target_word']}");colors.append(PUBLIC if regime=="between" else PRIVATE)
    y=np.arange(len(values))[::-1];ax.axvline(0,color="#263238",lw=.9);ax.scatter(values,y,c=colors,s=30,zorder=3)
    ax.set_yticks(y,labels,fontsize=6.8);ax.set_xlabel("Route contrast  $R=P-H$",fontsize=8.5)
    ax.set_title("B  Every family has the predicted sign",loc="left",weight="bold",fontsize=11)
    ax.text(.76,.02,"public-dominant  →",transform=ax.transAxes,color=PUBLIC,fontsize=7.5,ha="center")
    ax.text(.20,.02,"←  private-dominant",transform=ax.transAxes,color=PRIVATE,fontsize=7.5,ha="center")
    ax=fig.add_subplot(gs[2]);x=np.arange(2);width=.34
    for j,route in enumerate(("public","private")):
        blocks=[S["conditions"][f"{r}_1.0"]["temperatures"]["1.0"][f"{route}_target_minus_source"] for r in ("between","within")]
        means=[b["mean"] for b in blocks];errors=np.array([[b["mean"]-b["lo"] for b in blocks],[b["hi"]-b["mean"] for b in blocks]])
        ax.bar(x+(j-.5)*width,means,width,yerr=errors,capsize=3,color=PUBLIC if route=="public" else PRIVATE,label=route.title())
    ax.axhline(0,color="#263238",lw=.8);ax.set_xticks(x,["Between\n(a → an)","Within\n(a → a)"],fontsize=8)
    ax.set_ylabel("Target-aligned probability effect",fontsize=8.5);ax.set_title("C  Signed double dissociation",loc="left",weight="bold",fontsize=11)
    ax.legend(frameon=False,fontsize=8);ax.text(.5,-.22,"interaction = .264  [.137, .399]",transform=ax.transAxes,ha="center",fontsize=8,weight="bold")
    for ax in fig.axes[1:]:ax.spines[["top","right"]].set_visible(False);ax.grid(axis="x" if ax is fig.axes[1] else "y",alpha=.16);ax.tick_params(labelsize=7.5)
    fig.savefig(OUT,dpi=240,bbox_inches="tight");print(OUT)

if __name__=="__main__":main()
