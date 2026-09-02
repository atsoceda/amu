#!/usr/bin/env python3
"""Figure 1: feedback circuit, identification assay, and validity ladder."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/figures/fig_hypotheses_protocol.png"

def box(ax, x, y, w, h, text, face, size=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.012",
        facecolor=face, edgecolor="#263238", linewidth=1.2))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=size, weight=weight)

def arrow(ax, start, end, color, rad=0, lw=2):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
        color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))

def main():
    public, private = "#1769aa", "#7b3fb2"
    fig = plt.figure(figsize=(11.5, 6.3), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=(1.6, .9), hspace=.28)
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.set_title("A  Autoregressive generation closes a causal loop", loc="left", weight="bold", fontsize=13)
    box(ax,.03,.38,.19,.22,"Intervention\n$ I_t $","#dceaf7",11,"bold")
    box(ax,.39,.63,.22,.20,"Generated mediator\n$B_{t+1}$","#fff0b8",11,"bold")
    box(ax,.39,.16,.22,.20,"Persistent context\n$H_{t+1}$","#eadcf6",11,"bold")
    box(ax,.78,.38,.19,.22,"Later prediction\n$Y_{t+2}$","#d9efdf",11,"bold")
    arrow(ax,(.22,.54),(.39,.70),public); arrow(ax,(.61,.70),(.78,.55),public)
    arrow(ax,(.22,.44),(.39,.27),private); arrow(ax,(.61,.27),(.78,.43),private)
    ax.text(.50,.91,"PUBLIC PROJECTION",color=public,ha="center",weight="bold",fontsize=10)
    ax.text(.50,.06,"PRIVATE PERSISTENCE",color=private,ha="center",weight="bold",fontsize=10)
    ax.text(.50,.49,"token identity is fed back\nas a high-leverage input",ha="center",va="center",fontsize=8.5,color="#455a64")
    inset=ax.inset_axes([.025,.015,.30,.28]); inset.axis("off")
    inset.set_title("Six-cell identification",loc="left",fontsize=9.5,weight="bold",pad=2)
    table=inset.table(cellText=[["free","do(a)","do(an)"],["free","do(a)","do(an)"]],
        rowLabels=["I off","I on"],cellLoc="center",loc="center",bbox=[0,0,1,.72])
    table.auto_set_font_size(False); table.set_fontsize(7.5)
    for cell in table.get_celld().values(): cell.set_edgecolor("#78909c"); cell.set_linewidth(.7)
    ax=fig.add_subplot(gs[1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.set_title("B  Four distinct claims require four distinct checks",loc="left",weight="bold",fontsize=13)
    items=[("1","Candidate","signal identified","#dceaf7"),("2","Local efficacy","intended quantity moves","#d9efdf"),
           ("3","Mediator support","assay covers induced states","#fff0b8"),("4","Route","public / private / hybrid","#eadcf6")]
    xs=[.025,.275,.525,.775]
    for i,(num,title,detail,color) in enumerate(items):
        box(ax,xs[i],.27,.20,.45,f"{num}   {title}\n{detail}",color,9.2,"bold")
        if i<3: arrow(ax,(xs[i]+.205,.495),(xs[i+1]-.005,.495),"#546e7a",lw=1.4)
    ax.text(.5,.08,"candidate  ≠  causal  ≠  support-valid  ≠  transmission-identified",
        ha="center",fontsize=10,weight="bold",color="#37474f")
    fig.savefig(OUT,dpi=240,bbox_inches="tight"); print(OUT)

if __name__=="__main__": main()
