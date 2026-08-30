#!/usr/bin/env python3
"""Figure 1: causal-validity ladder, routes, and six-cell identification assay."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript/figures/fig_hypotheses_protocol.png"


def box(ax, xy, wh, text, color, fontsize=9, lw=1.2):
    patch = FancyBboxPatch(xy, *wh, boxstyle="round,pad=.015", facecolor=color,
                           edgecolor="#263238", linewidth=lw)
    ax.add_patch(patch)
    ax.text(xy[0]+wh[0]/2, xy[1]+wh[1]/2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, color="#37474f", lw=1.5):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw))


def main():
    fig = plt.figure(figsize=(12, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=(.9, 1.15), hspace=.25, wspace=.2)
    ax = fig.add_subplot(grid[0, :]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("A  Causal-validity ladder", loc="left", fontweight="bold", fontsize=13)
    labels = [
        ("1  Candidate signal identified", "Decoder, attribution graph,\nor sparse feature", "#dbe9f6"),
        ("2  Locally causal", "Intervention moves the\nselected local quantity", "#d9ead3"),
        ("3  Mediator-support valid", "Assay covers every mediator\nstate induced by intervention", "#fff0c2"),
        ("4  Transmission identified", "Public relay, private persistence,\nhybrid, or negligible", "#eadcf8"),
    ]
    xs = [.02, .27, .52, .77]
    for index, ((title, detail, color), x) in enumerate(zip(labels, xs)):
        box(ax, (x,.36), (.21,.34), f"{title}\n\n{detail}", color, fontsize=8.4)
        if index < 3: arrow(ax, (x+.215,.53), (xs[index+1]-.005,.53))
    examples = ["1B: selected features", "top-4: too weak", "top-24: exits to ‘called’", "frontier: stochastic assay"]
    for x, text in zip(xs, examples): ax.text(x+.105,.22,text,ha="center",va="center",fontsize=8,color="#455a64",style="italic")
    ax.text(.5,.07,"Passing a later rung does not retroactively establish an earlier one; each requires a distinct measurement.",ha="center",fontsize=9)

    ax = fig.add_subplot(grid[1,0]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.set_title("B  Public and private routes", loc="left", fontweight="bold", fontsize=13)
    box(ax,(.04,.38),(.23,.22),"Intervention $I_t$\nat pre-mediator $P$","#dbe9f6",8.5)
    box(ax,(.40,.66),(.25,.20),"Generated mediator $B_{t+1}$\n(public token)","#ffe79a",8.5)
    box(ax,(.40,.14),(.25,.20),"Context from $P$\n(private state)","#eadcf8",9)
    box(ax,(.75,.38),(.21,.22),"Later outcome\n$Y_{t+2}$","#bdeecf",9)
    arrow(ax,(.27,.53),(.40,.76),"#2563eb"); arrow(ax,(.65,.76),(.75,.55),"#2563eb")
    arrow(ax,(.27,.45),(.40,.24),"#7c3aed"); arrow(ax,(.65,.24),(.75,.43),"#7c3aed")
    ax.text(.52,.91,"public: clamp or replay $B$",ha="center",color="#2563eb",fontsize=9)
    ax.text(.52,.06,"private: compare with $B$ fixed",ha="center",color="#7c3aed",fontsize=9)

    ax = fig.add_subplot(grid[1,1]); ax.axis("off")
    ax.set_title("C  Six-cell transmission assay", loc="left", fontweight="bold", fontsize=13)
    table = ax.table(cellText=[[r"$Y(0,B_0)$",r"$Y(0,a)$",r"$Y(0,an)$"],
                              [r"$Y(1,B_1)$",r"$Y(1,a)$",r"$Y(1,an)$"]],
                     rowLabels=[r"$I=0$",r"$I=1$"],
                     colLabels=["Free mediator",r"$do(a)$",r"$do(an)$"],
                     cellLoc="center",loc="center",bbox=[.02,.36,.96,.48])
    table.auto_set_font_size(False); table.set_fontsize(10)
    for (row,col), cell in table.get_celld().items():
        cell.set_edgecolor("#37474f"); cell.set_linewidth(.8)
        if row == 0 or col == -1: cell.set_facecolor("#eceff1"); cell.set_text_props(weight="bold")
    ax.text(.5,.22,r"$E_T=Y(1,B_1)-Y(0,B_0)=E_{public}+E_{private}$",ha="center",fontsize=10)
    ax.text(.5,.11,"Free generation is a policy; fixed-mediator cells identify what survives token control.",ha="center",fontsize=8.5)
    fig.savefig(OUT,dpi=220,bbox_inches="tight",facecolor="white"); print(OUT)


if __name__ == "__main__": main()
