#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

EXP=Path(__file__).resolve().parent; ROOT=EXP.parents[1]
s=json.loads((EXP/"results/summary.json").read_text()); groups=("within_consonant","within_vowel","cross_class_reference")
labels=("Consonant→consonant","Vowel→vowel","Cross-class reference"); colors=("#4c78a8","#f58518","#54a24b")
fig,axes=plt.subplots(1,2,figsize=(8.4,3.15)); x=np.arange(3)
for gi,(group,label,color) in enumerate(zip(groups,labels,colors)):
    b=s["conditions"][f"{group}__5.0x__L5/F383"]
    vals=[b["target_delta_delta"]["mean"],b["all_alternatives_selectivity"]["mean"]]
    err=np.array([[vals[0]-b["target_delta_delta"]["lo"],vals[1]-b["all_alternatives_selectivity"]["lo"]],
                  [b["target_delta_delta"]["hi"]-vals[0],b["all_alternatives_selectivity"]["hi"]-vals[1]]])
    offset=(gi-1)*.14; axes[0].errorbar([0+offset,1+offset],vals,yerr=err,fmt="o",capsize=3,color=color,label=label)
    c=b["class_mean_logit_contrast"]; axes[1].bar(gi,c["mean"],color=color,yerr=[[c["mean"]-c["lo"]],[c["hi"]-c["mean"]]],capsize=3)
axes[0].axhline(0,color="0.45",lw=.8); axes[0].set_xticks([0,1],["Intended vs source","Intended vs alternatives"],fontsize=8)
axes[0].set_xlim(-.45,1.45); axes[0].set_ylabel("Logit contrast change"); axes[0].set_title("A Lexical specificity"); axes[0].legend(frameon=False,fontsize=7,loc="lower left")
axes[1].axhline(0,color="0.45",lw=.8); axes[1].set_xticks(x,labels,rotation=15,ha="right",fontsize=8); axes[1].set_ylabel("Licensed − unlicensed class\nmean logit change"); axes[1].set_title("B Article-class compatibility")
fig.tight_layout(); out=ROOT/"manuscript/figures/fig_l5_same_class_lexicality.png"; fig.savefig(out,dpi=220,bbox_inches="tight"); print(out)
