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
ROBUST_OUT=EXP.parents[1]/"manuscript/figures/fig_neutral_synonym_robustness.png"
TRIAD_ANALYSIS=json.loads((EXP.parent/"matched_semantic_triads_repaired/results/analysis.json").read_text())
PUBLIC,PRIVATE="#1769aa","#7b3fb2"

def native_rows(regime):return [r for r in ROWS if r["regime"]==regime and r["strength"]==1.0]

def main():
    fig=plt.figure(figsize=(12.0,4.0));gs=fig.add_gridspec(1,3,width_ratios=(.95,1.45,1.25),wspace=.45)
    ax=fig.add_subplot(gs[0]);ax.axis("off");ax.set_title("A  What can the article express?",loc="left",weight="bold",fontsize=11)
    examples=[(.74,"CROSS class","teacher  →  educator","a  →  an",PUBLIC,"article distinguishes"),
              (.28,"WITHIN class","teacher  →  tutor","a  →  a",PRIVATE,"article collapses")]
    for y,heading,words,articles,color,note in examples:
        ax.text(.02,y+.14,heading,color=color,weight="bold",fontsize=9.2);ax.text(.02,y,words,fontsize=9.5,weight="bold")
        ax.text(.02,y-.13,articles,fontsize=13,color=color,weight="bold");ax.text(.02,y-.23,note,fontsize=8.2,color="#455a64")
    ax=fig.add_subplot(gs[2]);values=[];labels=[];colors=[]
    for regime in ("between","within"):
        for r in native_rows(regime):
            st=r["stochastic"]["1.0"];values.append(st["public"]["target_minus_source"]-st["private"]["target_minus_source"])
            labels.append(f"{r['source_word']} → {r['target_word']}");colors.append(PUBLIC if regime=="between" else PRIVATE)
    y=np.arange(len(values))[::-1];ax.axvline(0,color="#111111",lw=1.5);ax.scatter(values,y,c=colors,s=30,zorder=3)
    ax.set_yticks(y);ax.set_yticklabels([]);ax.set_xlabel("Route contrast  $R=P-H$",fontsize=8.5)
    ax.set_title("C  Independent-family convergence",loc="left",weight="bold",fontsize=11)
    ax.text(.76,.02,"PUBLIC-DOMINANT  →",transform=ax.transAxes,color=PUBLIC,fontsize=7.3,ha="center",weight="bold")
    ax.text(.20,.02,"←  PRIVATE-DOMINANT",transform=ax.transAxes,color=PRIVATE,fontsize=7.3,ha="center",weight="bold")
    ax.text(.02,.98,"Between  $n=6$",transform=ax.transAxes,color=PUBLIC,fontsize=7.5,va="top",weight="bold")
    ax.text(.02,.53,"Within  $n=8$",transform=ax.transAxes,color=PRIVATE,fontsize=7.5,va="top",weight="bold")
    ax=fig.add_subplot(gs[1]);primary=TRIAD_ANALYSIS["settings"]["1.0"]["1.0"]
    for row in primary["rows"]:
        ax.plot([0,1],[row["within"],row["cross"]],color="#78909c",alpha=.55,lw=1)
        ax.scatter([0],[row["within"]],color=PRIVATE,s=15,alpha=.75,zorder=3)
        ax.scatter([1],[row["cross"]],color=PUBLIC,s=15,alpha=.75,zorder=3)
    for x0,key,color in ((0,"within_route",PRIVATE),(1,"cross_route",PUBLIC)):
        block=primary[key]
        ax.errorbar(x0,block["mean"],yerr=[[block["mean"]-block["lo"]],[block["hi"]-block["mean"]]],
                    fmt="D",ms=6,capsize=4,color=color,mec="white",mew=.5,zorder=5)
    ax.axhline(0,color="#263238",lw=1.2);ax.set_xlim(-.22,1.22);ax.set_xticks([0,1],["Within class\n(same article)","Cross class\n(other article)"],fontsize=8)
    ax.set_ylabel("Route contrast  $R=P-H$",fontsize=8.5);ax.set_title("B  Matched within-family test ($n=14$)",loc="left",weight="bold",fontsize=11)
    effect=primary["paired_interaction"]
    ax.text(.5,-.22,f"mean $\\Delta R$ = {effect['mean']:.3f}  [{effect['lo']:.3f}, {effect['hi']:.3f}]",transform=ax.transAxes,ha="center",fontsize=8,weight="bold")
    ax.text(.04,.96,"13/14 shift toward public\n8/14 cross zero",transform=ax.transAxes,va="top",fontsize=8,weight="bold",color="#37474f")
    for ax in fig.axes[1:]:ax.spines[["top","right"]].set_visible(False);ax.grid(axis="x" if ax is fig.axes[1] else "y",alpha=.16);ax.tick_params(labelsize=7.5)
    fig.savefig(OUT,dpi=240,bbox_inches="tight");plt.close(fig);print(OUT)

    strengths=(.5,1.0,1.5);taus=(.1,.25,.5,1.0)
    old_heat=np.asarray([[S["interactions"][f"strength_{strength}"][str(tau)]["aligned_route_interaction"]["mean"] for tau in taus] for strength in strengths])
    paired_heat=np.asarray([[TRIAD_ANALYSIS["settings"][str(strength)][str(tau)]["paired_interaction"]["mean"] for tau in taus] for strength in strengths])
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.0),sharey=True)
    vmax=max(old_heat.max(),paired_heat.max())
    for ax,heat,title in zip(axes,(old_heat,paired_heat),("A  Independent-family interaction","B  Matched-triad paired shift")):
        image=ax.imshow(heat,aspect="auto",cmap="PuBu",vmin=0,vmax=vmax)
        for i in range(len(strengths)):
            for j in range(len(taus)):ax.text(j,i,f"{heat[i,j]:.3f}",ha="center",va="center",color="white" if heat[i,j]>.12 else "#263238",fontsize=9,weight="bold")
        ax.set_xticks(range(len(taus)),[str(t) for t in taus]);ax.set_yticks(range(len(strengths)),[str(s) for s in strengths]);ax.set_xlabel("Article-policy temperature");ax.set_title(title,loc="left",weight="bold",fontsize=10)
    axes[0].set_ylabel("Patch strength")
    fig.colorbar(image,ax=axes,label="Route interaction / paired shift",fraction=.035,pad=.025)
    fig.savefig(ROBUST_OUT,dpi=240,bbox_inches="tight");plt.close(fig);print(ROBUST_OUT)

if __name__=="__main__":main()
