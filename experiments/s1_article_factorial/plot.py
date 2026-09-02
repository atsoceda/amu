#!/usr/bin/env python3
"""Figure 2: S1 replay/rescue and decoder-boundary amplification."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SIX=json.loads((ROOT/"experiments/s1_article_factorial/results/summary.json").read_text())
BOUND=json.loads((ROOT/"experiments/boundary_precision_policy/results/summary.json").read_text())
OUT=ROOT/"manuscript/figures/fig_mediation_decomposition.png"

def main():
    public,private,total="#1769aa","#7b3fb2","#263238"
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.45),gridspec_kw={"width_ratios":[1.2,.85,1.05]})
    ax=axes[0];ax.axis("off");ax.set_title("A  Replay and rescue",loc="left",weight="bold",fontsize=11)
    rows=[("Baseline","a","pilot","#eceff1"),("S1 on, free","an","aviator","#dceaf7"),
          ("Replay treated article, S1 off","an","aviator","#d9efdf"),("Restore baseline article, S1 on","a","pilot","#fff0b8")]
    y=.82
    for label,article,noun,color in rows:
        ax.text(.01,y,label,fontsize=8.2,va="center")
        ax.text(.76,y,article,fontsize=11,weight="bold",color=public,ha="right",va="center",
                bbox=dict(boxstyle="round,pad=.18",facecolor=color,edgecolor="none"))
        ax.text(.79,y,noun,fontsize=11,weight="bold",va="center");y-=.205
    ax.text(.01,.01,"Phenocopy 20/20   •   Rescue 20/20",fontsize=8.5,weight="bold")
    ax=axes[1]
    block=SIX["analysis"]["recomputed"]["decomposition"]
    vals=[block["total_tv_full_vocab"]["mean"],block["article_only_tv_full_vocab"]["mean"],block["residual_tv_full_vocab"]["mean"]]
    bars=ax.bar(np.arange(3),vals,color=[total,public,private],width=.68)
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.025,f"{v:.3f}",ha="center",fontsize=9,weight="bold")
    ax.set_xticks(np.arange(3),["Total","Public\nreplay","Private\nleftover"],fontsize=8);ax.set_ylim(0,1.02)
    ax.set_ylabel("Noun-distribution TV",fontsize=8.5);ax.set_title("B  Public replay nearly matches total",loc="left",weight="bold",fontsize=11)
    ax=axes[2];b=BOUND["native_boundary_reference"]
    vals=[b["total_tv"]["mean"],b["token_substitution_tv"]["mean"],b["fixed_a_residual_tv"]["mean"],b["fixed_an_residual_tv"]["mean"]]
    bars=ax.bar(np.arange(4),vals,color=[total,public,private,"#a477c8"],width=.68)
    for bar,value in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,value+.025,f"{value:.3f}",ha="center",fontsize=8.5,weight="bold")
    ax.set_xticks(np.arange(4),["Free","Token\nsubstitution","Fixed a","Fixed an"],fontsize=7.6);ax.set_ylim(0,1.02)
    ax.set_title("C  The decoder boundary supplies gain",loc="left",weight="bold",fontsize=11)
    ax.text(.5,-.24,"gain interval ≤ .0039",transform=ax.transAxes,ha="center",fontsize=8,color="#455a64")
    for ax in axes[1:]:ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.18);ax.tick_params(axis="y",labelsize=8)
    fig.tight_layout(w_pad=1.2);fig.savefig(OUT,dpi=240,bbox_inches="tight");print(OUT)

if __name__=="__main__":main()
