#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt

EXP=Path(__file__).resolve().parent; ROOT=EXP.parents[1]
summary=json.loads((EXP/"results/summary.json").read_text())
derived=json.loads((EXP/"results/derived_analysis.json").read_text())
layers=[int(k) for k in summary["dev_layer_means"]]; vals=[summary["dev_layer_means"][str(k)] for k in layers]
fig,axes=plt.subplots(1,2,figsize=(8.5,3.1))
axes[0].plot([x/(summary["n_layers"]-1) for x in layers],vals,marker="o",ms=3)
axes[0].axhline(0,color="0.5",lw=.8); axes[0].set(xlabel="Relative depth",ylabel="Dev target-minus-source ΔΔ",title="A 1B layer sweep")
keys=[f"{r}_{s}" for r in ("between","within") for s in (.5,1.0,1.5)]; x=range(len(keys))
axes[1].bar([i-.18 for i in x],[summary["conditions"][k]["public_tv"]["mean"] for k in keys],.36,label="Public")
axes[1].bar([i+.18 for i in x],[summary["conditions"][k]["private_tv"]["mean"] for k in keys],.36,label="Private")
axes[1].set_xticks(list(x),[k.replace("_","\n",1) for k in keys],fontsize=7); axes[1].set(ylabel="TV",title="B Causal channel map"); axes[1].legend(frameon=False)
fig.tight_layout(); out=ROOT/"manuscript/figures/fig_gemma_1b_residual_scale.png"; fig.savefig(out,dpi=220,bbox_inches="tight"); print(out)
