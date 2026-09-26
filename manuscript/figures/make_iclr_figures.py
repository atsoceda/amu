#!/usr/bin/env python3
"""Figures for the ICLR 2027 manuscript, generated from the committed experiment results.

Run from the repository root:
    /Users/anthony/miniconda3/bin/python manuscript/figures/make_iclr_figures.py
Every number is read from experiments/*/results. Runs that are still in progress are skipped
silently, so the figures can be regenerated as results arrive. Figures are drawn at their
printed size (ICLR text width 5.5 in), so font sizes are the sizes on the page (at least 7 pt).
One colour per path throughout (Okabe-Ito): direct retrieval blue, the stored copy at the
line's final token green, late lookup via the last three positions sky blue, relay along
generated text (and any indirect path not split further, and recurrent memory) reddish purple, emission
vermillion, interaction light grey; totals and references are grey. One marker per model family
(filled: Qwen3 / Qwen3.5, open: Gemma 3).

Intervals are 95% bootstrap intervals from the summary JSON. For a share num/den, the interval
is the numerator's interval divided by the mean denominator (the denominator is held fixed);
for a sum of paths, the half-widths are combined in quadrature.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript" / "figures"
EXP = ROOT / "experiments"
CR, HC, PC = EXP / "couplet_routes/results", EXP / "hidden_choice/results", EXP / "relay_positive_control/results"
RC, RD, DV = EXP / "recurrent_carry/results", EXP / "relay_distance/results", EXP / "derived_value_carry/results"
SC = EXP / "qwen3_planning_six_cell/results"
W = 5.5  # ICLR text width, inches

COL = {"retrieval": "#0072B2", "storage": "#009E73", "late": "#56B4E9", "relay": "#CC79A7",
       "emission": "#D55E00", "grey": "#8C8C8C", "interaction": "#DADADA", "ink": "#222222",
       "muted": "#555555"}
MARK = {"Qwen3": "o", "Gemma 3": "s", "Qwen3.5": "^", "Gemma 4": "D"}
FILLED = {"Qwen3": True, "Qwen3.5": True, "Gemma 3": False, "Gemma 4": False}
plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.55, "xtick.major.width": 0.55, "ytick.major.width": 0.55, "xtick.major.size": 2.2,
    "ytick.major.size": 2.2, "xtick.major.pad": 1.8, "ytick.major.pad": 1.8, "axes.labelpad": 2.0,
    "savefig.dpi": 400, "legend.frameon": False, "axes.titlelocation": "left", "axes.titleweight": "bold",
    "axes.titlepad": 4.0, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2})
NAN = float("nan")

QWEN = [("Qwen3-1.7B", 1.7), ("Qwen3-4B", 4), ("Qwen3-8B", 8), ("Qwen3-14B", 14), ("Qwen3-32B", 32)]
GEMMA = [("gemma-3-1b-it", 1), ("gemma-3-4b-it", 4), ("gemma-3-12b-it", 12), ("gemma-3-27b-it", 27)]


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def mean(v):
    return v["mean"] if isinstance(v, dict) else v


def ratio(a, b):
    a, b = mean(a), mean(b)
    return float("nan") if a is None or not b else a / b


def share(num, den):
    """100 * num / den with the numerator's 95% interval scaled by the mean denominator."""
    d = mean(den)
    if num is None or d is None or not d:
        return (NAN, NAN, NAN)
    m = 100 * mean(num) / d
    if isinstance(num, dict) and "lo" in num:
        lo, hi = sorted((100 * num["lo"] / d, 100 * num["hi"] / d))
        return (m, lo, hi)
    return (m, NAN, NAN)


def share_sum(nums, den):
    """100 * sum(nums) / den; interval half-widths of the terms combined in quadrature."""
    d = mean(den)
    if not d or any(n is None for n in nums):
        return (NAN, NAN, NAN)
    m = sum(mean(n) for n in nums)
    if all(isinstance(n, dict) and "lo" in n for n in nums):
        dl = math.sqrt(sum((n["mean"] - n["lo"]) ** 2 for n in nums))
        dh = math.sqrt(sum((n["hi"] - n["mean"]) ** 2 for n in nums))
        lo, hi = sorted((100 * (m - dl) / d, 100 * (m + dh) / d))
        return (100 * m / d, lo, hi)
    return (100 * m / d, NAN, NAN)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


def couplet_shares(run):
    """Shares of persistence (original line-2 words) by path, as in couplet_routes/scale_summary.py."""
    s34, pos, nec = (load(CR / run / f) for f in ("step34_summary.json", "relay_positions_summary.json",
                                                  "relay_necessity_summary.json"))
    if not s34 or not pos:
        return None
    a, pers = pos["all"], mean(pos["all"]["d_persistence_p0"])
    out = {"persistence": pers, "retrieval": mean(s34["all"]["d_retrieval"]) / pers,
           "late": mean(a["d_relay_late"]) / pers, "boundary": mean(a["d_relay_tail"]) / pers,
           "early": mean(a["d_relay_early"]) / pers, "relay": mean(a["d_relay_all"]) / pers,
           "gap": mean(s34["additivity_gap_p0"]) / pers}
    if nec:
        n, npers = nec["all"], mean(nec["all"]["d_persistence_p0"])
        out.update(nec_boundary=mean(n["necessity_tail"]) / npers, nec_late=mean(n["necessity_late"]) / npers,
                   nec_early=mean(n["necessity_early"]) / npers)
    return out


def couplet_route_ci(run):
    """(looked up, relay) as % of persistence with intervals, for Figure 1B."""
    s34, pos = load(CR / run / "step34_summary.json"), load(CR / run / "relay_positions_summary.json")
    if not s34 or not pos:
        return None
    a, den = pos["all"], pos["all"]["d_persistence_p0"]
    return (share_sum([s34["all"]["d_retrieval"], a["d_relay_tail"], a["d_relay_late"]], den),
            share(a["d_relay_early"], den))


def distance_data(model="gemma-3-27b-it"):
    byd = {}
    for f in ("pilot_v2x_summary.json", "pilot_v2x_window_summary.json"):
        x = load(RD / model / f)
        if x:
            byd.update({int(k): v for k, v in x["by_distance"].items() if v.get("n")})
    return byd


def recurrent_data():
    out = []
    for d, lab in [("Qwen3.5-4B", "4B"), ("Qwen3.5-9B", "9B"), ("Qwen3.5-27B", "27B"),
                   ("Qwen3.5-35B-A3B", "35B-A3B")]:
        x = load(RC / d / "pilot_summary.json") or load(RC / d / "pilot24_summary.json")
        if x:
            out.append((lab, x))
    return out


def dot(ax, x, v, color, marker="o", filled=True, s=14, lw=0.7, z=4, horizontal=False):
    """A mark with its interval. v = (mean, lo, hi) in plotted units."""
    m, lo, hi = v
    if not np.isfinite(m):
        return
    if np.isfinite(lo) and np.isfinite(hi):
        if horizontal:
            ax.plot([lo, hi], [x, x], color=color, lw=lw, zorder=z - 1, solid_capstyle="butt")
        else:
            ax.plot([x, x], [lo, hi], color=color, lw=lw, zorder=z - 1, solid_capstyle="butt")
    xy = (m, x) if horizontal else (x, m)
    ax.scatter(*xy, marker=marker, s=s, facecolor=color if filled else "white", edgecolor=color,
               lw=0.8, zorder=z)


# --------------------------------------------------------------------------- Figure 1

def bez(ax, x0, y0, x1, y1, h, share_, col, ls="-", arrow=True):
    """Quadratic arc from (x0, y0) to (x1, y1) peaking near height h; width from the path share."""
    t = np.linspace(0, 1, 80)
    xm, c = (x0 + x1) / 2, 2 * h - (y0 + y1) / 2
    xs = (1 - t) ** 2 * x0 + 2 * t * (1 - t) * xm + t ** 2 * x1
    ys = (1 - t) ** 2 * y0 + 2 * t * (1 - t) * c + t ** 2 * y1
    lw = 0.7 + 6.0 * max(share_, 0.0)
    ax.plot(xs, ys, color=col, lw=lw, ls=ls, solid_capstyle="butt", alpha=0.95)
    if arrow:
        ax.plot([x1], [y1 + 0.05], marker="v", ms=2.6 + 0.3 * lw, color=col, mec="none")


def fig1_schematic(axA):
    s = couplet_shares("Qwen3-32B")
    axA.set_xlim(-0.1, 9.35)
    axA.set_ylim(-0.75, 4.45)
    axA.axis("off")
    toks = [(0.0, 0.7, "lonely", "ctx"), (0.8, 0.7, "night", "src"), (1.6, 0.36, ",", "bd"),
            (2.06, 0.74, "⟨turn⟩", "tpl"), (3.0, 0.55, "And", "gen"), (3.65, 0.65, "stars", "gen"),
            (4.4, 0.55, "will", "gen"), (5.05, 0.65, "guide", "late"), (5.8, 0.5, "me", "late"),
            (6.4, 0.8, "through", "late"), (7.3, 0.55, "the", "tgt")]
    fc = {"ctx": "#F5F5F5", "src": "#DCEAF5", "bd": "#D9F0E7", "tpl": "#F3F3F3", "gen": "#F8EEF4",
          "late": "#E6F4FB", "tgt": "#FFFFFF"}
    ec = {"ctx": "#C8C8C8", "src": COL["retrieval"], "bd": COL["storage"], "tpl": "#C8C8C8", "gen": COL["relay"],
          "late": COL["late"], "tgt": COL["ink"]}
    cx, box = {}, {}
    for x, w, t, k in toks:
        axA.add_patch(FancyBboxPatch((x, 0.0), w, 0.62, boxstyle="round,pad=0.015,rounding_size=0.1", fc=fc[k],
                                     ec=ec[k], lw=1.0 if k in ("src", "tgt") else 0.7))
        axA.text(x + w / 2, 0.31, t, ha="center", va="center", fontsize=7, color=COL["ink"],
                 weight="bold" if k in ("src", "tgt") else "normal")
        cx[t], box[t] = x + w / 2, (x, w)
    # Donor badge on the rhyme word.
    axA.text(cx["night"], -0.02, "donor", ha="center", va="center", fontsize=7, color="white", weight="bold",
             bbox=dict(boxstyle="round,pad=0.12,rounding_size=0.25", fc=COL["retrieval"], ec="none"))
    axA.text(7.95, 0.31, "→ rhyme?", ha="left", va="center", fontsize=7, color="#666666", style="italic")
    axA.plot([0.0, 2.8], [-0.3, -0.3], color="#AAAAAA", lw=0.6)
    axA.plot([3.0, 7.85], [-0.3, -0.3], color="#AAAAAA", lw=0.6)
    axA.text(1.4, -0.4, "end of line 1 (prompt)", ha="center", va="top", fontsize=7, color="#666666")
    axA.text(5.42, -0.4, "line 2 so far (generated, held fixed)", ha="center", va="top", fontsize=7,
             color="#666666")
    y0 = 0.68
    # Source retrieval and the stored copy at the line's final token.
    bez(axA, cx["night"], y0, cx["the"], y0, 2.2, s["retrieval"], COL["retrieval"])
    bez(axA, cx["night"] + 0.12, y0, cx[","], y0, 0.98, s["boundary"], COL["storage"])
    bez(axA, cx[","], y0, cx["the"] - 0.08, y0, 1.72, s["boundary"], COL["storage"])
    # Late lookup: into a bracket over the last three line-2 positions, then a short arc to the target.
    bx0, bx1 = box["guide"][0] + 0.02, box["through"][0] + box["through"][1] - 0.02
    by = 1.02
    axA.plot([bx0, bx0, bx1, bx1], [by - 0.12, by, by, by - 0.12], color=COL["late"], lw=0.9)
    bm = (bx0 + bx1) / 2
    bez(axA, cx["night"] - 0.12, y0, bm - 0.2, by + 0.02, 1.42, s["late"], COL["late"])
    bez(axA, bx1 - 0.1, by + 0.02, cx["the"] + 0.1, y0, 1.22, s["late"], COL["late"], arrow=True)
    # Relay: a chain of small arcs from the rhyme word through line 2 to the target.
    chain = ["night", "And", "stars", "will", "guide", "me", "through", "the"]
    for a, b in zip(chain[:-1], chain[1:]):
        h = 0.86 if a == "night" else 0.8
        bez(axA, cx[a] + (0.18 if a == "night" else 0.0), y0, cx[b], y0, h, max(s["early"], 0.0),
            COL["relay"], ls=(0, (1.6, 1.0)), arrow=(b == "the"))
    rows = [(COL["retrieval"], "-", "direct retrieval from the rhyme word", f"{100 * s['retrieval']:.0f}%"),
            (COL["storage"], "-", "stored copy at the line's final token",
             f"{100 * s['boundary']:.0f}% (nec. {100 * s['nec_boundary']:.0f}%)"),
            (COL["late"], "-", "late lookup via the last three positions", f"{100 * s['late']:.0f}%"),
            (COL["relay"], (0, (1.6, 1.0)), "relay along line 2", f"{100 * s['early']:.0f}%")]
    for i, (c, ls, lab, num) in enumerate(rows):
        yy = 4.0 - 0.42 * i
        axA.plot([3.75, 4.2], [yy, yy], color=c, lw=2.0, ls=ls)
        axA.text(4.32, yy, lab, va="center", fontsize=7, color=COL["ink"])
        axA.text(9.35, yy, num, va="center", ha="right", fontsize=7,
                 color=c if c != COL["late"] else "#2B86BD", weight="bold")
    axA.text(-0.1, 4.45, "A   Where a rhyme plan travels", fontsize=8, weight="bold", va="top")
    axA.text(-0.1, 3.95, "Qwen3-32B, chat prompt", fontsize=7, color=COL["muted"], va="top")


def fig1_points():
    """Groups of (label, family, looked up, relay or unsplit indirect) for Figure 1B, in % of each group's denominator."""
    ctrl = []
    for d, fam, lab in [("Qwen3-1.7B", "Qwen3", "1.7B"), ("Qwen3-4B", "Qwen3", "4B"), ("Qwen3-8B", "Qwen3", "8B"),
                        ("Qwen3-14B", "Qwen3", "14B"), ("Qwen3-32B", "Qwen3", "32B"),
                        ("Qwen3.5-9B", "Qwen3.5", "9B"), ("Qwen3.5-27B", "Qwen3.5", "27B"), ("Qwen3.5-35B-A3B", "Qwen3.5", "35B"),
                        ("gemma-3-12b-it", "Gemma 3", "12B"), ("gemma-3-27b-it", "Gemma 3", "27B")]:
        x = load(PC / d / "induction_summary.json")
        if x:
            ctrl.append((lab, fam, share(x["retrieval"], x["persistence"]), share(x["relay"], x["persistence"])))
    coup = []
    for runs, fam, suffix, sub in ((QWEN, "Qwen3", "", "chat"), (QWEN, "Qwen3", "-plain", "plain"),
                                   (GEMMA, "Gemma 3", "", "chat"), (GEMMA, "Gemma 3", "-plain", "plain")):
        grp = []
        for d, size in runs:
            c = couplet_route_ci(d + suffix)
            if c:
                grp.append((f"{size:g}B", fam, c[0], c[1]))
        if grp:
            coup.append((sub, fam, grp))
    hid = []
    for d, fam, lab in [("Qwen3-4B", "Qwen3", "4B"), ("Qwen3-8B", "Qwen3", "8B"), ("Qwen3-14B", "Qwen3", "14B"),
                        ("Qwen3-32B", "Qwen3", "32B"), ("Qwen3.5-9B", "Qwen3.5", "9B"), ("Qwen3.5-27B", "Qwen3.5", "27B"),
                        ("Qwen3.5-35B-A3B", "Qwen3.5", "35B"), ("gemma-3-12b-it", "Gemma 3", "12B"),
                        ("gemma-3-27b-it", "Gemma 3", "27B")]:
        x = load(HC / d / "choice_summary.json")
        if x and x["post_edit"]["lo"] > 0.1:  # shares of a stored part indistinguishable from zero are noise
            hid.append((lab, fam, share(x["post_retrieval"], x["post_edit"]), share(x["post_relay"], x["post_edit"])))
    rec = [(lab, "Qwen3.5", share(x["attention_only"], x["persistence"]), share(x["recurrent_only"], x["persistence"]))
           for lab, x in recurrent_data()]
    byd = distance_data()
    dist = [(f"{D:,}", "Gemma 3", share_sum([byd[D]["retrieval"], byd[D]["relay_boundary"]], byd[D]["persistence"]),
             share(byd[D]["relay_line2"], byd[D]["persistence"])) for D in sorted(byd)]
    return ctrl, coup, hid, rec, dist


def fig1():
    H = 5.15
    fig = plt.figure(figsize=(W, H))
    fy = lambda inch: inch / H  # noqa: E731
    fx = lambda inch: inch / W  # noqa: E731
    axA = fig.add_axes([0.0, fy(3.36), 1.0, fy(1.79)])
    fig1_schematic(axA)
    ctrl, coup, hid, rec, dist = fig1_points()

    # B. Paired marks per setting: looked up (blue) and relay or unsplit indirect (purple).
    ybot, ytop = 0.92, 2.64     # inches: plot area of panel B
    brk = 0.07                  # gap of the broken axis
    top_frac = 0.40             # share of the right-hand plot height given to the upper segment
    hb = (ytop - ybot - brk) * (1 - top_frac)
    ht = (ytop - ybot - brk) * top_frac
    axC = fig.add_axes([fx(0.42), fy(ybot), fx(0.98), fy(ytop - ybot)])
    xr0, xr1 = 1.92, 5.3
    axT = fig.add_axes([fx(xr0), fy(ybot + hb + brk), fx(xr1 - xr0), fy(ht)])
    axL = fig.add_axes([fx(xr0), fy(ybot), fx(xr1 - xr0), fy(hb)])
    fig.text(0.0, fy(3.3), "B   The indirect path carries induction; elsewhere relay, or an unsplit indirect path, is small", fontsize=8,
             weight="bold", va="top")
    den_y = fy(0.5)

    def pair(axes, x, fam, look, rel, s=13):
        for ax in axes:
            dot(ax, x, look, COL["retrieval"], MARK[fam], FILLED[fam], s=s)
            dot(ax, x, rel, COL["relay"], MARK[fam], FILLED[fam], s=s)

    # Control group: continuous 0-150%.
    for i, (lab, fam, look, rel) in enumerate(ctrl):
        pair([axC], i, fam, look, rel)
    axC.axhline(80, color=COL["relay"], lw=0.7, ls=(0, (3, 2)), zorder=1)
    axC.text(-0.55, 77, "pass threshold\nset in advance (80%)", fontsize=7, color="#8E4F7A", ha="left", va="top",
             linespacing=0.95)
    axC.set_ylim(-5, 150)
    axC.set_yticks([0, 50, 100, 150])
    axC.set_xlim(-0.7, len(ctrl) - 0.3)
    axC.set_xticks(range(len(ctrl)))
    axC.set_xticklabels([c[0] for c in ctrl], rotation=90)
    axC.tick_params(axis="x", length=0)
    axC.set_ylabel("% of persistence")
    axC.axhline(0, color="#CCCCCC", lw=0.5, zorder=0)

    # Right-hand groups on a broken axis: upper 40-125%, lower -5-30%.
    xpos, labels, spans, sub_spans, stress = [], [], [], [], []
    x = 0.0
    cstep = 0.85
    for sub, fam_, grp in coup:
        x0 = x
        for lab, fam, look, rel in grp:
            pair([axT, axL], x, fam, look, rel)
            xpos.append(x)
            labels.append("")
            x += cstep
        rng = f"{grp[0][0][:-1]}–{grp[-1][0]}"
        sub_spans.append((x0, x - cstep, f"{sub}\n{rng}", fam_))
        x += 0.9
    if coup:
        spans.append(("rhyme plans (couplets)", sub_spans[0][0], sub_spans[-1][1], "share of persistence", "task"))
    x += 0.5
    if hid:
        x0 = x
        for lab, fam, look, rel in hid:
            pair([axT, axL], x, fam, look, rel)
            xpos.append(x)
            labels.append(lab)
            x += 1.3
        spans.append(("hidden\nchoice", x0, x - 1.3, "share of the\nstored part", "task"))
        x += 0.6
    x += 1.4
    for grp, step, name in ((rec, 1.7, "recurrent\nhybrid"), (dist, 1.9, "beyond the\nwindow")):
        if not grp:
            continue
        x0 = x
        for lab, fam, look, rel in grp:
            pair([axT, axL], x, fam, look, rel)
            xpos.append(x)
            labels.append(lab)
            x += step
        spans.append((name, x0, x - step, None, "stress"))
        stress.append((x0 - 0.6, x - step + 0.6))
        x += 1.1
    xmax = x - 1.1 + 0.6
    for ax in (axT, axL):
        ax.set_xlim(-0.6, xmax)
    # Frozen decision lines for the stress tests.
    for sx0, sx1 in stress:
        for yv in (10, 25):
            axL.plot([sx0, sx1], [yv, yv], color="#8E4F7A", lw=0.7, ls=(0, (3, 2)) if yv == 10 else "-", zorder=1)
    if stress:
        axL.text(xmax + 0.1, 10, "stop", fontsize=7, color="#8E4F7A", va="center", ha="left", clip_on=False)
        axL.text(xmax + 0.1, 25, "go", fontsize=7, color="#8E4F7A", va="center", ha="left", clip_on=False)
    axT.set_ylim(40, 125)
    axT.set_yticks([50, 75, 100])
    axL.set_ylim(-5, 30)
    axL.set_yticks([0, 10, 20, 30])
    axT.spines["bottom"].set_visible(False)
    axT.tick_params(axis="x", bottom=False, labelbottom=False)
    axL.set_xticks(xpos)
    axL.set_xticklabels(labels, rotation=90)
    axL.tick_params(axis="x", length=0)
    axL.axhline(0, color="#CCCCCC", lw=0.5, zorder=0)
    for ax, yy in ((axT, 0.0), (axL, 1.0)):  # break marks on the left spine
        ax.plot([-0.012, 0.012], [yy - 0.03, yy + 0.03], transform=ax.transAxes, color="k", lw=0.6,
                clip_on=False)
    axL.set_ylabel("% of the denominator\nnamed under each group", y=(hb + brk + ht) / 2 / hb)
    fams = {}
    for x0, x1, sub, fam_ in sub_spans:  # couplet sub-groups, in the empty upper part of the lower segment
        w1, w2 = sub.split("\n")
        axL.text((x0 + x1) / 2, 21.6, w1, fontsize=7, ha="center", va="baseline", color=COL["muted"])
        axL.text((x0 + x1) / 2, 18.4, w2, fontsize=7, ha="center", va="baseline", color=COL["muted"])
        fams.setdefault(fam_, []).extend([x0, x1])
    for fam_, xs_ in fams.items():
        axL.text((min(xs_) + max(xs_)) / 2, 28.5, fam_, fontsize=7, ha="center", va="top", color=COL["ink"])

    def xin(xd):  # data x of the right-hand axes -> figure fraction
        return fx(xr0 + (xd + 0.6) / (xmax + 0.6) * (xr1 - xr0))
    trT = axT.get_xaxis_transform()
    for name, x0, x1, den, kind in spans:
        axT.text((x0 + x1) / 2, 1.03, name, transform=trT, fontsize=7, ha="center", va="bottom", linespacing=0.95)
        if den:
            fig.text(xin((x0 + x1) / 2), den_y, den, fontsize=7, ha="center", va="top", color=COL["muted"],
                     linespacing=0.95)
    st = [sp for sp in spans if sp[4] == "stress"]
    if st:
        fig.text(xin((st[0][1] + st[-1][2]) / 2), den_y, "share of persistence\n(at each distance)", fontsize=7,
                 ha="center", va="top", color=COL["muted"], linespacing=0.95)
    trC = axC.get_xaxis_transform()
    axC.text((len(ctrl) - 1) / 2, 1.02, "induction", transform=trC, fontsize=7, ha="center", va="bottom")
    fig.text(fx(0.42 + 0.49), den_y, "share of\npersistence", fontsize=7, ha="center", va="top",
             color=COL["muted"], linespacing=0.95)
    sup_y = fy(ytop + 0.33)
    fig.text(fx(0.42 + 0.49), sup_y, "positive control", fontsize=7, ha="center", va="bottom", weight="bold")
    fig.add_artist(Line2D([fx(0.47), fx(1.35)], [sup_y - fy(0.02)] * 2, color="#999999", lw=0.6))
    for kind, lab in (("task", "tasks"), ("stress", "stress tests")):
        grp = [sp for sp in spans if sp[4] == kind]
        if grp:
            a_, b_ = xin(grp[0][1] - 0.4), xin(grp[-1][2] + 0.4)
            fig.text((a_ + b_) / 2, sup_y, lab, fontsize=7, ha="center", va="bottom", weight="bold")
            fig.add_artist(Line2D([a_, b_], [sup_y - fy(0.02)] * 2, color="#999999", lw=0.6))
    hs = [Line2D([], [], ls="none", marker="o", ms=4, mfc=COL["retrieval"], mec=COL["retrieval"]),
          Line2D([], [], ls="none", marker="o", ms=4, mfc=COL["relay"], mec=COL["relay"]),
          Line2D([], [], ls="none", marker="o", ms=4, mfc="#444444", mec="#444444"),
          Line2D([], [], ls="none", marker="^", ms=4, mfc="#444444", mec="#444444"),
          Line2D([], [], ls="none", marker="s", ms=4, mfc="white", mec="#444444")]
    fig.legend(hs, ["looked up (source, stored copy or late lookup)", "relay, or an unsplit indirect path", "Qwen3",
                    "Qwen3.5", "Gemma 3"], loc="lower center", bbox_to_anchor=(0.5, -0.005), ncol=5,
               handletextpad=0.2, columnspacing=0.9)
    save(fig, "iclr_fig1_overview.png")


# --------------------------------------------------------------------------- Figure 2

def fig2():
    # Instruction-tuned checkpoints only; the base-checkpoint comparison is in Table 3 and the appendix.
    panels = [("Qwen3, chat", [(d, s) for d, s in QWEN]),
              ("Qwen3, plain", [(d + "-plain", s) for d, s in QWEN]),
              ("Gemma 3, chat", [(d, s) for d, s in GEMMA]),
              ("Gemma 3, plain", [(d + "-plain", s) for d, s in GEMMA])]
    fig = plt.figure(figsize=(W, 4.0))
    # Panel widths proportional to the number of models, so every bar has the same width.
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0], width_ratios=[len(r) for _, r in panels],
                          hspace=0.66, wspace=0.12, left=0.09, right=0.99, top=0.83, bottom=0.1)
    parts = [("retrieval", "direct retrieval", COL["retrieval"]),
             ("boundary", "stored copy at the line's final token", COL["storage"]),
             ("late", "late lookup (last 3 positions)", COL["late"]),
             ("early", "relay along line 2", COL["relay"])]
    lx = np.log10
    xt = [1, 2, 4, 8, 16, 32]
    series = []
    for ci, (title, runs) in enumerate(panels):
        ax = fig.add_subplot(gs[0, ci])
        avail = [r for r in runs if couplet_shares(r[0])]
        suff, nec = [], []
        # One evenly spaced bar per model, labelled with its actual size.
        for x, (run, size) in enumerate(avail):
            c = couplet_shares(run)
            bottom = 0.0
            for k, _, colr in parts:
                v = 100 * max(c[k], 0.0)
                ax.bar(x, v, 0.62, bottom=bottom, color=colr, lw=0)
                bottom += v
            ax.bar(x, max(100 - bottom, 0.0), 0.62, bottom=bottom, color=COL["interaction"], lw=0)
            suff.append((size, 100 * c["boundary"]))
            if "nec_boundary" in c:
                nec.append((size, 100 * c["nec_boundary"]))
        series.append((title, suff, nec))
        ax.set_xlim(-0.6, len(avail) - 0.4)
        ax.set_xticks(range(len(avail)))
        ax.set_xticklabels([f"{r[1]:g}" for r in avail])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=7, weight="normal", pad=3)
        ax.set_xlabel("parameters (B)")
        if ci == 0:
            ax.set_ylabel("share of persistence (%)")
        else:
            ax.set_yticklabels([])
    fig.text(0.0, 0.995, "A   Direct retrieval dominates, except Gemma 3 27B (plain), where the stored copy takes over; relay ≤ 1%", fontsize=8,
             weight="bold", va="top")
    hs = [Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts]
    hs.append(Rectangle((0, 0), 1, 1, color=COL["interaction"]))
    fig.legend(hs, [l for _, l, _ in parts] + ["interaction"], loc="upper left", ncol=3,
               bbox_to_anchor=(0.04, 0.955), handlelength=1.0, columnspacing=0.9, labelspacing=0.2,
               borderaxespad=0.0)

    # B. The stored copy at the line's final token, all four settings on one axis.
    bx = fig.add_subplot(gs[1, :])
    x_band = (lx(12) + lx(14)) / 2
    bx.axvspan(x_band, lx(60), color="#E6F4EE", lw=0, zorder=0)
    bx.text(x_band + 0.03, 57, "copy present\n(≥ 14B)", fontsize=7, color="#1B6E53", va="top", ha="left")
    style = {"Qwen3, chat": ("o", "-"), "Qwen3, plain": ("o", (0, (3, 1.6))),
             "Gemma 3, chat": ("s", "-"), "Gemma 3, plain": ("s", (0, (3, 1.6)))}
    ends = []
    for title, suff, nec in series:
        mk, ls = style[title]
        if suff:
            xs, ys = [lx(p[0]) for p in suff], [p[1] for p in suff]
            bx.plot(xs, ys, ls=ls, color=COL["storage"], lw=1.4, zorder=3)
            bx.scatter(xs, ys, marker=mk, s=16, color=COL["storage"], zorder=4)
            ends.append([xs[-1], ys[-1], title.replace(",", "")])
        if nec:
            bx.scatter([lx(p[0]) for p in nec], [p[1] for p in nec], marker=mk, s=16, facecolor="white",
                       edgecolor=COL["storage"], lw=0.8, zorder=4)
            for p_ in nec:
                if p_[1] > 45:
                    bx.annotate(f"necessity {p_[1]:.0f}%", xy=(lx(p_[0]), p_[1]),
                                xytext=(lx(p_[0]) + 0.12, p_[1]), fontsize=7, color="#1B6E53", ha="left",
                                va="center", arrowprops=dict(arrowstyle="-", color="#9BCDB9", lw=0.5))
    # Direct labels at the line ends, spread so they do not overlap.
    ends.sort(key=lambda e: e[1])
    ly = [e[1] for e in ends]
    for i in range(1, len(ly)):
        ly[i] = max(ly[i], ly[i - 1] + 9.0)
    for (x, y, lab), yy in zip(ends, ly):
        bx.annotate(lab, xy=(x, y), xytext=(lx(34) + 0.04, yy), fontsize=7, color="#1B6E53", va="center",
                    ha="left", arrowprops=dict(arrowstyle="-", color="#9BCDB9", lw=0.5))
    bx.axhline(0, color="#BBBBBB", lw=0.5, zorder=1)
    bx.set_xlim(lx(0.85), lx(60))
    bx.set_xticks([lx(v) for v in xt])
    bx.set_xticklabels([str(v) for v in xt])
    bx.set_ylim(-5, 82)
    bx.set_yticks([0, 20, 40, 60, 80])
    bx.set_xlabel("parameters (B)")
    bx.set_ylabel("stored copy\n(% of persistence)")
    hs = [Line2D([], [], ls="none", marker="o", ms=4, mfc=COL["storage"], mec=COL["storage"]),
          Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec=COL["storage"])]
    bx.legend(hs, ["sufficiency (patched in)", "necessity (patched out)"], loc="upper left",
              handletextpad=0.2, labelspacing=0.2, borderaxespad=0.1)
    fig.text(0.0, bx.get_position().y1 + 0.035, "B   A stored copy at the line's final token appears from about 12 to 14B parameters",
             fontsize=8, weight="bold", va="bottom")
    save(fig, "iclr_fig2_couplet_routes.png")


# --------------------------------------------------------------------------- Figure 3

PRETTY = {"<|im_end|>": "⟨im_end⟩", "<|im_start|>": "⟨im_start⟩", "<think>": "⟨think⟩", "</think>": "⟨/think⟩",
          "<end_of_turn>": "⟨end_turn⟩", "<start_of_turn>": "⟨start_turn⟩", "Ċ": "↵", "ĊĊ": "↵↵", "\n": "↵",
          ",\n": ",↵", ",Ċ": ",↵"}
FS3 = 7.0          # token font size (pt)
CMAP3 = LinearSegmentedColormap.from_list("share", ["#FFFFFF", "#9ED9C3", COL["storage"], "#00563F"])
_TW = {}


def pretty(t):
    return PRETTY.get(t, t.replace("▁", "").replace("Ġ", "").replace("Ċ", "↵").replace("\n", "↵"))


def joined(tokens):
    """Readable text of a run of tokens (spaces restored from the tokenizer's space markers)."""
    out = ""
    for t in tokens:
        if t in PRETTY:
            out += PRETTY[t]
        else:
            out += t.replace("▁", " ").replace("Ġ", " ").replace("Ċ", "↵").replace("\n", "↵")
    return " ".join(out.split())


def text_width(s, fs=FS3):
    """Rendered width of a string in points."""
    key = (s, fs)
    if key not in _TW:
        f = plt.figure(figsize=(1, 1), dpi=72)
        t = f.text(0, 0, s, fontsize=fs)
        _TW[key] = t.get_window_extent(f.canvas.get_renderer()).width
        plt.close(f)
    return _TW[key]


def storage_cells(toks, vals, collapse):
    """Cells [(text, share or None)]; share = token's part of the row's total positive necessity."""
    pos = sum(max(v, 0.0) for v in vals) or 1e-9
    sh = [max(v, 0.0) / pos for v in vals]
    cells = []
    if not collapse:
        for t, s in zip(toks, sh):
            cells.append(([t], s))
        return cells
    run = []
    for t, s in zip(toks, sh):
        if s < 0.03:
            run.append(t)
            continue
        if run:
            cells.append((run, None))
            run = []
        cells.append(([t], s))
    if run:
        cells.append((run, None))
    return cells


def cell_label(tokens, s):
    if s is None:
        return joined(tokens)
    t = pretty(tokens[0])
    return f"{t}  {100 * s:.0f}%" if s >= 0.05 else t


def layout_cells(cells, width_pt):
    """Place cells left to right, wrapping lines; long grey runs are split at token boundaries."""
    out, x, line = [], 0.0, 0
    pad, gap, minw = 5.0, 1.2, 12.0

    def place(label, s):
        nonlocal x, line
        w = max(text_width(label) + pad, minw)
        if x + w > width_pt and x > 0:
            line, x = line + 1, 0.0
        out.append((line, x, w, label, s))
        x += w + gap

    for tokens, s in cells:
        if s is not None or text_width(joined(tokens)) + pad <= width_pt - x:
            place(cell_label(tokens, s), s)
            continue
        chunk = []
        for t in tokens:
            trial = joined(chunk + [t])
            if chunk and text_width(trial) + pad > width_pt - x:
                place(joined(chunk), None)
                if x > 0 and text_width(joined([t])) + pad > width_pt - x:
                    line, x = line + 1, 0.0
                chunk = [t]
            else:
                chunk.append(t)
        if chunk:
            place(joined(chunk), None)
    return out


def storage_rows(main):
    rowsA = ([("Qwen3-32B", "Qwen3-32B, chat prompt"), ("gemma-3-27b-it", "Gemma 3 27B, chat prompt"),
              ("gemma-3-12b-it-plain", "Gemma 3 12B, plain prompt"), ("gemma-3-27b-it-plain", "Gemma 3 27B, plain prompt")]
             if main else
             [(d + suf, f"{lab}, {'plain' if suf else 'chat'} prompt")
              for d, lab in [("Qwen3-1.7B", "Qwen3-1.7B"), ("Qwen3-4B", "Qwen3-4B"), ("Qwen3-8B", "Qwen3-8B"),
                             ("Qwen3-14B", "Qwen3-14B"), ("Qwen3-32B", "Qwen3-32B"),
                             ("gemma-3-1b-it", "Gemma 3 1B"), ("gemma-3-4b-it", "Gemma 3 4B"),
                             ("gemma-3-12b-it", "Gemma 3 12B"), ("gemma-3-27b-it", "Gemma 3 27B")]
              for suf in ("", "-plain")])
    rows = []
    for run, lab in rowsA:
        s = load(CR / run / "relay_boundary_summary.json")
        if s and s.get("necessity_by_token"):
            rows.append(("A", lab, [x["token"] for x in s["necessity_by_token"]],
                         [x["mean"] for x in s["necessity_by_token"]]))
    fr, fa, an = ("choice_replicate_fruits_localize_summary.json", "choice_replicate_fruits_localize_alt_summary.json",
                  "choice_replicate_animals_localize_summary.json")
    rowsB = [("gemma-3-12b-it", fr, "Gemma 3 12B, fruits"), ("gemma-3-12b-it", fa, "Gemma 3 12B, fruits, reworded"),
             ("gemma-3-27b-it", fr, "Gemma 3 27B, fruits"), ("gemma-3-27b-it", fa, "Gemma 3 27B, fruits, reworded")]
    if not main:
        rowsB += [("gemma-3-12b-it", an, "Gemma 3 12B, animals"), ("gemma-3-27b-it", an, "Gemma 3 27B, animals"),
                  ("gemma-3-4b-it", fr, "Gemma 3 4B, fruits"), ("gemma-3-4b-it", fa, "Gemma 3 4B, fruits, reworded"),
                  ("gemma-3-4b-it", an, "Gemma 3 4B, animals")]
    for run, name, lab in rowsB:
        s = load(HC / run / name)
        if s and s.get("necessity_by_token"):
            rows.append(("B", lab, [x["token"] for x in s["necessity_by_token"]],
                         [x["mean"] for x in s["necessity_by_token"]]))
    return rows


def storage_figure(main, name):
    rows = storage_rows(main)
    width_pt = W * 72 - 4
    line_h, lab_h, head_h, gap = 13.5, 11.0, 13.0, 4.0
    plan, y = [], 22.0  # room for the colour bar
    for key in ("A", "B"):
        sel = [r for r in rows if r[0] == key]
        if not sel:
            continue
        plan.append(("head", key, y))
        y += head_h
        for _, lab, t, v in sel:
            lay = layout_cells(storage_cells(t, v, collapse=main), width_pt)
            plan.append(("row", (lab, lay), y))
            y += lab_h + line_h * (max(l[0] for l in lay) + 1) + gap
        y += 5.0
    H = y + 1
    fig = plt.figure(figsize=(W, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W * 72)
    ax.set_ylim(H, 0)
    ax.axis("off")
    heads = {"A": "A   Rhyme plan: where the plan is stored between line 1 and line 2",
             "B": "B   Hidden choice: where the pick is stored between the list and the reveal"}
    for kind, obj, y0 in plan:
        if kind == "head":
            ax.text(1, y0 + 1, heads[obj], fontsize=8, weight="bold", va="top")
            continue
        lab, lay = obj
        ax.text(1, y0 + 1, lab, fontsize=7, va="top", color="#444444", style="italic")
        for line, x, w, label, s in lay:
            yt = y0 + lab_h + line * line_h
            if s is None:
                fc, tc, wt = "#EFEFEF", "#707070", "normal"
            else:
                fc, tc = CMAP3(min(s, 1.0)), ("white" if s > 0.45 else COL["ink"])
                wt = "bold" if s >= 0.05 else "normal"
            ax.add_patch(Rectangle((x + 1, yt), w, line_h - 2.5, fc=fc, ec="#D0D0D0", lw=0.3))
            ax.text(x + 1 + w / 2, yt + (line_h - 2.5) / 2, label, ha="center", va="center", fontsize=FS3,
                    color=tc, weight=wt)
    # Shared colour scale.
    cb_w, cb_h, cb_x = 90.0, 5.0, W * 72 - 96.0
    for i in range(60):
        ax.add_patch(Rectangle((cb_x + i * cb_w / 60, 4.0), cb_w / 60 + 0.1, cb_h, fc=CMAP3(i / 59), ec="none"))
    ax.add_patch(Rectangle((cb_x, 4.0), cb_w, cb_h, fc="none", ec="#AAAAAA", lw=0.3))
    for v in (0, 50, 100):
        ax.text(cb_x + cb_w * v / 100, 4.0 + cb_h + 1.0, f"{v}%", fontsize=7, ha="center", va="top", color="#444444")
    ax.text(cb_x - 4, 4.0 + cb_h / 2, "share of the row's total positive necessity", fontsize=7, ha="right",
            va="center", color="#444444")
    if main:
        ax.text(1, 4.0 + cb_h / 2, "grey: consecutive tokens under 3%, joined", fontsize=7, ha="left",
                va="center", color="#707070")
    save(fig, name)


def pos_share(vals):
    tot = sum(max(v, 0.0) for v in vals) or 1e-9
    return [max(v, 0.0) / tot for v in vals], tot


def line_end_index(toks):
    """Index of line 1's final token: the comma before a chat template, or the line break of a plain prompt."""
    if len(toks) > 1 and pretty(toks[1]) == "↵":
        return 1
    return 0


def colourbar_pts(ax, x, y, w=80.0, h=5.0, label="share of the row's total positive necessity"):
    for i in range(60):
        ax.add_patch(Rectangle((x + i * w / 60, y), w / 60 + 0.1, h, fc=CMAP3(i / 59), ec="none"))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="#AAAAAA", lw=0.3))
    for v in (0, 50, 100):
        ax.text(x + w * v / 100, y + h + 1.0, f"{v}%", fontsize=7, ha="center", va="top", color="#444444")
    ax.text(x - 4, y + h / 2, label, fontsize=7, ha="right", va="center", color="#444444")


def fig3():
    storage_figure(False, "iclr_figA_storage_all.png")
    Hpt = 318.0
    Wpt = W * 72
    fig = plt.figure(figsize=(W, Hpt / 72))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, Wpt)
    bg.set_ylim(Hpt, 0)
    bg.axis("off")
    bg.set_zorder(0)

    def axes_pts(x0, y0, w, h):  # rectangle in points from the top-left -> figure axes
        return fig.add_axes([x0 / Wpt, 1 - (y0 + h) / Hpt, w / Wpt, h / Hpt])

    colourbar_pts(bg, Wpt - 86, 17.0, label="cell shading: share of the total")
    bg.text(0, 1, "A   Rhyme plan: the copy sits almost entirely on line 1's final token", fontsize=8, weight="bold",
            va="top")
    # One annotated example: Qwen3-32B, chat prompt.
    ex = load(CR / "Qwen3-32B" / "relay_boundary_summary.json")
    y = 16.0
    if ex:
        toks = [x["token"] for x in ex["necessity_by_token"]]
        sh, _ = pos_share([x["mean"] for x in ex["necessity_by_token"]])
        k = line_end_index(toks)
        bg.text(0, y, "Qwen3-32B, chat prompt. Example couplet:", fontsize=7, style="italic", color="#444444", va="top")
        y += 11
        bg.text(8, y, "line 1: \u201cIn shadows deep where silence weeps,\u201d", fontsize=7, color="#444444", va="top")
        y += 10
        bg.text(8, y, "line 2: \u201cThe moon whispers through the dark it keeps.\u201d", fontsize=7, color="#444444", va="top")
        y += 12
        bg.text(0, y, "Tokens between the rhyme word and line 2 (shares averaged over 100 couplets):", fontsize=7,
                style="italic", color="#444444", va="top")
        y += 16
        cells = [("weeps", "src", None)] + [(pretty(t), "end" if i == k else "tpl", sh[i])
                                            for i, t in enumerate(toks)] + [("The moon \u2026 dark", "gap", None), ("it", "tgt", None)]
        fs, ch, gap = 7.5, 17.0, 2.0
        widths = [text_width(c[0] + (f"  {100 * c[2]:.0f}%" if c[1] == "end" else ""), fs) + 8 for c in cells]
        scale = min(1.0, (Wpt - 4 - gap * len(cells)) / sum(widths))
        x = 1.0
        for (lab, kind, v), w in zip(cells, widths):
            w *= scale
            if kind == "gap":
                bg.text(x + w / 2, y + ch / 2, lab, fontsize=fs, ha="center", va="center", color="#777777")
                x += w + gap
                continue
            fc = {"src": "#DCEAF5", "tgt": "white"}.get(kind, CMAP3(min(v or 0.0, 1.0)))
            ec = {"src": COL["retrieval"], "tgt": COL["ink"], "end": COL["storage"]}.get(kind, "#C8C8C8")
            bg.add_patch(FancyBboxPatch((x, y), w, ch, boxstyle="round,pad=0,rounding_size=2.5", fc=fc, ec=ec,
                                        lw=1.0 if kind in ("src", "tgt", "end") else 0.5))
            txt = f"{lab}  {100 * v:.0f}%" if kind == "end" else lab
            bg.text(x + w / 2, y + ch / 2, txt, fontsize=fs, ha="center", va="center",
                    color="white" if kind == "end" and v > 0.45 else COL["ink"],
                    weight="bold" if kind in ("src", "tgt", "end") else "normal")
            if kind in ("src", "tgt"):
                bg.text(x + w / 2, y + ch + 2, "source" if kind == "src" else "target",
                        fontsize=7, ha="center", va="top",
                        color=COL["retrieval"] if kind == "src" else COL["ink"])
            if kind == "end":
                bg.text(x + 1, y + ch + 2, "line 1's final token", fontsize=7, ha="left", va="top",
                        color="#1B6E53")
            x += w + gap
        y += ch + 26
    # Dot strip: share on the final token vs the largest other boundary token.
    rowsA = []
    for run, lab in [("Qwen3-14B", "Qwen3-14B, chat"), ("Qwen3-32B", "Qwen3-32B, chat"),
                     ("gemma-3-27b-it", "Gemma 3 27B, chat"), ("gemma-3-12b-it-plain", "Gemma 3 12B, plain"),
                     ("gemma-3-27b-it-plain", "Gemma 3 27B, plain")]:
        sm = load(CR / run / "relay_boundary_summary.json")
        if not (sm and sm.get("necessity_by_token")):
            continue
        nb = sm["necessity_by_token"]
        toks = [x["token"] for x in nb]
        sh, tot = pos_share([x["mean"] for x in nb])
        k = line_end_index(toks)
        others = [i for i in range(len(nb)) if i != k]
        j = max(others, key=lambda i: sh[i]) if others else None
        fin = (100 * sh[k], 100 * max(nb[k]["lo"], 0) / tot, 100 * nb[k]["hi"] / tot)
        oth = (100 * sh[j], 100 * max(nb[j]["lo"], 0) / tot, 100 * max(nb[j]["hi"], 0) / tot) if j is not None \
            else (NAN, NAN, NAN)
        rowsA.append((f"{lab} ({pretty(toks[k])})", fin, oth, pretty(toks[j]) if j is not None else ""))
    row_h = 11.0
    if rowsA:
        hA = row_h * len(rowsA) + 4
        ax = axes_pts(112, y, Wpt - 112 - 6, hA)
        for i, (lab, fin, oth, olab) in enumerate(rowsA):
            dot(ax, i, fin, COL["storage"], "o", True, s=18, horizontal=True)
            dot(ax, i, oth, COL["grey"], "o", False, s=18, horizontal=True)
        ax.set_yticks(range(len(rowsA)))
        ax.set_yticklabels([r[0] for r in rowsA])
        ax.set_ylim(len(rowsA) - 0.5, -0.5)
        ax.set_xlim(-2, 102)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#EEEEEE", lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("share of the boundary tokens' total positive necessity (%)")
        hs = [Line2D([], [], ls="none", marker="o", ms=4, mfc=COL["storage"], mec=COL["storage"]),
              Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec=COL["grey"])]
        ax.legend(hs, ["line 1's final token", "largest other boundary token"], loc="lower right",
                  bbox_to_anchor=(1.0, 1.0), ncol=2, handletextpad=0.2, borderaxespad=0.1)
        y += hA + 26
    # B. The hidden-choice instruction once, with its storage tokens highlighted.
    y += 4
    bg.text(0, y, "B   Hidden choice: the stored signal sits mostly on periods and one function word or possessive token",
            fontsize=8, weight="bold", va="top")
    y += 14
    fr, fa, an = ("choice_replicate_fruits_localize_summary.json", "choice_replicate_fruits_localize_alt_summary.json",
                  "choice_replicate_animals_localize_summary.json")
    shown = None
    for run, lab in (("gemma-3-27b-it", "Gemma 3 27B"), ("gemma-3-12b-it", "Gemma 3 12B")):
        sm = load(HC / run / fr)
        if sm and sm.get("necessity_by_token"):
            shown = (lab, sm)
            break
    if shown:
        lab, sm = shown
        nb = sm["necessity_by_token"]
        sh, _ = pos_share([x["mean"] for x in nb])
        bg.text(0, y, f"{lab}, fruits: the instruction after the list (highlight: share of the pick-specific "
                "storage)", fontsize=7, style="italic", color="#444444", va="top")
        y += 12
        fs, lh = 8.0, 21.0
        x = 1.0
        words = [("⟨list of fruits⟩ ", None, False)] + [(pretty(t) if t in PRETTY else t.replace("▁", "").replace("\n", "↵"),
                                                         sh[i], t.startswith("▁")) for i, t in enumerate(nb and [q["token"] for q in nb])]
        for wtxt, v, space in words:
            wd = text_width(wtxt, fs)
            if space:
                x += text_width(" ", fs) + 1.0
            if x + wd > Wpt - 4:
                x, y = 1.0, y + lh
            hi_ = v is not None and v >= 0.03
            if hi_:
                bg.add_patch(Rectangle((x - 1.2, y + 7.5), wd + 2.4, 11.5, fc=CMAP3(min(v, 1.0)), ec="none"))
                bg.text(x + wd / 2, y + 7.0, f"{100 * v:.0f}%", fontsize=7, ha="center", va="bottom",
                        color="#1B6E53", weight="bold")
            bg.text(x, y + 13.25, wtxt, fontsize=fs, ha="left", va="center",
                    color=("white" if hi_ and v > 0.45 else COL["ink"]) if v is not None else "#888888",
                    weight="bold" if hi_ else "normal", style="normal" if v is not None else "italic")
            x += wd + 0.8
        y += lh + 16
    # Composition per model and wording: periods, the word before "weather", everything else.
    rowsB = []
    for run, lab in (("gemma-3-4b-it", "Gemma 3 4B"), ("gemma-3-12b-it", "Gemma 3 12B"),
                     ("gemma-3-27b-it", "Gemma 3 27B")):
        for f, wl in ((fr, "fruits"), (fa, "fruits, reworded"), (an, "animals")):
            sm = load(HC / run / f)
            if not (sm and sm.get("necessity_by_token")):
                continue
            toks = [x["token"] for x in sm["necessity_by_token"]]
            sh, _ = pos_share([x["mean"] for x in sm["necessity_by_token"]])
            per = sum(v for t, v in zip(toks, sh) if t.strip() == ".")
            # function word: the largest non-period, non-template token (with a trailing "s" of a possessive)
            cand = [(v, i) for i, (t, v) in enumerate(zip(toks, sh))
                    if t.strip() != "." and not t.startswith("<") and t.strip() not in ("", "model")]
            fw, word = 0.0, ""
            if cand:
                v, i = max(cand)
                fw, word = v, toks[i].replace("\u2581", "").strip()
                if word in ("'", "\u2019") and i + 1 < len(toks) and toks[i + 1].replace("\u2581", "") == "s":
                    fw += sh[i + 1]
                    word = "'s"
            rowsB.append((f"{lab}, {wl}", 100 * per, 100 * fw, 100 * max(1 - per - fw, 0.0), word))
    if rowsB:
        hB = row_h * len(rowsB) + 4
        ax = axes_pts(112, y, Wpt - 112 - 6, hB)
        cols = [("#00563F", "segment-closing periods"), (COL["storage"], "one function word or possessive token (named)"),
                ("#E3E3E3", "all other tokens")]
        for i, vals in enumerate(rowsB):
            left = 0.0
            for k, ((c, lab), v) in enumerate(zip(cols, vals[1:4])):
                ax.barh(i, v, 0.72, left=left, color=c, lw=0, label=lab if i == 0 else None)
                if v >= 9:
                    txt = f"\u201c{vals[4]}\u201d {v:.0f}%" if k == 1 else f"{v:.0f}%"
                    ax.text(left + v / 2, i, txt, fontsize=7, ha="center", va="center",
                            color="white" if c != "#E3E3E3" else "#444444", weight="bold")
                left += v
        ax.set_yticks(range(len(rowsB)))
        ax.set_yticklabels([r[0] for r in rowsB])
        ax.set_ylim(len(rowsB) - 0.5, -0.5)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("share of the pick-specific storage (% of the row's total positive necessity)")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=1.0, columnspacing=0.9,
                  borderaxespad=0.1)
        y += hB + 26
    fig.set_size_inches(W, (y + 2) / 72)
    Hn = y + 2
    for a_ in fig.axes[1:]:
        pos = a_.get_position()
        y0 = (1 - pos.y1) * Hpt
        a_.set_position([pos.x0, 1 - (y0 + pos.height * Hpt) / Hn, pos.width, pos.height * Hpt / Hn])
    bg.set_position([0, 0, 1, 1])
    bg.set_ylim(Hn, 0)
    save(fig, "iclr_fig3_storage_sites.png")


# --------------------------------------------------------------------------- Figure 4

def fig4():
    fig = plt.figure(figsize=(W, 4.95))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.08], hspace=0.78)
    g1 = gs[0].subgridspec(1, 4, width_ratios=[1.05, 1.05, 0.5, 0.8], wspace=0.62)
    g2 = gs[1].subgridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.08)

    # A. Recurrent hybrid.
    ax = fig.add_subplot(g1[0])
    rows = recurrent_data()
    for i, (lab, x) in enumerate(rows):
        a = share(x["attention_only"], x["persistence"])
        r = share(x["recurrent_only"], x["persistence"])
        b = share(x.get("both_blocked"), x["persistence"])
        ax.bar(i - 0.2, a[0], 0.34, color=COL["retrieval"], label="via attention layers" if i == 0 else None)
        ax.bar(i + 0.16, r[0], 0.34, color=COL["relay"], label="via recurrent layers" if i == 0 else None)
        for xx, v in ((i - 0.2, a), (i + 0.16, r)):
            if np.isfinite(v[1]):
                ax.plot([xx, xx], [v[1], v[2]], color=COL["ink"], lw=0.7)
        ax.text(i + 0.27, 11.5, f"{r[0]:.0f}%", ha="center", va="bottom", fontsize=7, color="#8E4F7A")
        if np.isfinite(b[0]):
            ax.scatter(i + 0.44, b[0], marker="x", s=10, color=COL["grey"], lw=0.8, zorder=4,
                       label="both blocked" if i == 0 else None)
    for yv, lab, ls in ((10, "stop", (0, (3, 2))), (25, "go", "-")):
        ax.axhline(yv, color="#8E4F7A", lw=0.6, ls=ls, zorder=0)
        ax.text(len(rows) + 0.12, yv, lab, fontsize=7, color="#8E4F7A", ha="right", va="center",
                bbox=dict(fc="white", ec="none", pad=0.3))
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_xlim(-0.55, len(rows) + 0.15)
    ax.set_ylim(0, 172)
    ax.spines["left"].set_bounds(0, 100)
    ax.set_yticks([0, 10, 25, 50, 75, 100])
    ax.set_ylabel("% of persistence")
    ax.set_xlabel("Qwen3.5 (75% of layers recurrent)")
    ax.legend(loc="upper left", handlelength=1.0, borderaxespad=0.0, labelspacing=0.2, bbox_to_anchor=(0, 1.04), fontsize=6.5, title="donor entry", title_fontsize=6.5)
    ax.set_title("A   Recurrent hybrid", x=-0.3)

    # B. Beyond Gemma 3's local attention window: measured distances only.
    byd = distance_data()
    ds = sorted(byd)
    ax = fig.add_subplot(g1[1])
    ax.axhspan(-6, 10, color="#EFEFEF", lw=0, zorder=0)
    ax.axhspan(10, 25, color="#F8ECF3", lw=0, zorder=0)
    for yv in (10, 25):
        ax.axhline(yv, color="#8E4F7A", lw=0.5, ls=(0, (3, 2)) if yv == 10 else "-", zorder=0)
    for i, D in enumerate(ds):
        v = byd[D]
        for off, k, c in ((-0.22, "retrieval", COL["retrieval"]), (0.0, "necessity_boundary", COL["storage"]),
                          (0.22, "relay_line2", COL["relay"])):
            dot(ax, i + off, share(v[k], v["persistence"]), c, "s", False, s=16)
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels([f"{D:,} ({byd[D]['n']})" for D in ds], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(ds) - 0.4)
    ax.set_ylim(-6, 132)
    if 1000 in ds and 1500 in ds:
        xw = (ds.index(1000) + ds.index(1500)) / 2
        ax.axvline(xw, color="#555555", lw=0.8, ls=(0, (1, 1.5)), zorder=1)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines["left"].set_bounds(-6, 100)
    xr = len(ds) - 0.42
    ax.text(xr, 4, "stop", fontsize=7, color="#8E4F7A", ha="right", va="center")
    ax.text(xr, 17.5, "extend", fontsize=7, color="#8E4F7A", ha="right", va="center")
    ax.text(xr, 28, "go", fontsize=7, color="#8E4F7A", ha="right", va="bottom")
    ax.set_xlabel("filler tokens before line 2 (couplets)")
    ax.set_ylabel("% of persistence")
    hs = [Line2D([], [], ls="none", marker="s", ms=3.5, mfc="white", mec=c) for c in
          (COL["retrieval"], COL["storage"], COL["relay"])]
    ax.legend(hs, ["direct retrieval", "stored copy (nec.)", "late lookup + relay\n(cue and line 2)"], loc="upper right",
              handletextpad=0.1, borderaxespad=0.0, labelspacing=0.2, bbox_to_anchor=(1.02, 1.0), fontsize=6.5)
    ax.set_title("B   Beyond the attention window (Gemma 3 27B)", x=-0.3)
    # B, side: absolute persistence at each distance.
    ax = fig.add_subplot(g1[2])
    for i, D in enumerate(ds):
        p = byd[D]["persistence"]
        dot(ax, i, (p["mean"], p["lo"], p["hi"]), COL["grey"], "s", True, s=14)
    ax.set_xticks(range(len(ds)))
    ax.set_xticklabels([f"{D:,}" for D in ds], rotation=90 if len(ds) > 2 else 0)
    ax.set_xlim(-0.6, len(ds) - 0.4)
    ax.set_ylim(0, 50)
    ax.axhline(0, color="#CCCCCC", lw=0.5)
    ax.set_ylabel("persistence (log-odds)")
    ax.set_xlabel("filler tokens")
    # B, side: how often line 2 still rhymes with the original word.
    ax = fig.add_subplot(g1[3])
    rh = []
    for d, f, lab, fam in [("gemma-3-27b-it", "pilot_v2x_summary.json", "Gemma 3 27B", "Gemma 3"),
                           ("gemma-3-12b-it", "pilot_v2_summary.json", "Gemma 3 12B", "Gemma 3"),
                           ("Qwen3-14B", "pilot_v2_summary.json", "Qwen3-14B", "Qwen3"),
                           ("Qwen3.5-9B", "pilot_v2_summary.json", "Qwen3.5-9B", "Qwen3.5")]:
        x = load(RD / d / f)
        bd = (x or {}).get("by_distance", {})
        if "0" in bd and "2000" in bd:
            rh.append((lab, fam, 100 * bd["0"]["rhymes_orig"], 100 * bd["2000"]["rhymes_orig"]))
    # One marker shape here: the models are named on the axis, so the legend can match every point.
    for i, (lab, fam, r0, r2) in enumerate(rh):
        ax.plot([i, i], [r0, r2], color="#BBBBBB", lw=0.8, zorder=1)
        ax.scatter(i, r0, marker="o", s=14, facecolor="white", edgecolor=COL["grey"], lw=0.8, zorder=3)
        ax.scatter(i, r2, marker="o", s=14, facecolor=COL["ink"], edgecolor=COL["ink"], lw=0.8, zorder=3)
    ax.set_xticks(range(len(rh)))
    ax.set_xticklabels([r[0] for r in rh], rotation=90)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.6, len(rh) - 0.4)
    ax.set_ylim(-44, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines["left"].set_bounds(0, 100)
    ax.set_ylabel("line 2 rhymes (% of couplets)")
    hs = [Line2D([], [], ls="none", marker="o", ms=3.5, mfc="white", mec=COL["grey"]),
          Line2D([], [], ls="none", marker="o", ms=3.5, mfc=COL["ink"], mec=COL["ink"])]
    ax.legend(hs, ["no filler", "2,000 filler"], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=1,
              handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2, frameon=True, fancybox=False,
              edgecolor="#BBBBBB", framealpha=1.0, fontsize=6.5)

    # C. Hidden choice: composition of the text-swap reference.
    models = [("Qwen3-4B", "Qwen3 4B"), ("Qwen3-8B", "Qwen3 8B"), ("Qwen3-14B", "Qwen3 14B"),
              ("Qwen3-32B", "Qwen3 32B"), ("Qwen3.5-9B", "Qwen3.5 9B"), ("Qwen3.5-27B", "Qwen3.5 27B"), ("gemma-3-12b-it", "Gemma 3 12B"),
              ("gemma-3-27b-it", "Gemma 3 27B")]
    models = [(d, lab) for d, lab in models if load(HC / d / "choice_summary.json")]
    axc = fig.add_subplot(g2[0])
    axz = fig.add_subplot(g2[1], sharey=axc)
    first = True
    for i, (d, lab) in enumerate(models):
        c = load(HC / d / "choice_summary.json")
        st = share(c["post_edit"], c["text_swap"])
        rl = share(c["post_relay"], c["text_swap"])
        stored, relayed = max(st[0], 0.0), min(max(rl[0], 0.0), max(st[0], 0.0))
        axc.barh(i, relayed, 0.62, color=COL["relay"], lw=0, label="via the sentence" if first else None)
        axc.barh(i, stored - relayed, 0.62, left=relayed, color=COL["storage"], lw=0,
                 label="stored after the list" if first else None)
        axc.barh(i, 100 - stored, 0.62, left=stored, color=COL["retrieval"], lw=0, alpha=0.85,
                 label="not reproduced by the patch" if first else None)
        if np.isfinite(st[1]):
            axc.plot([st[1], st[2]], [i, i], color=COL["ink"], lw=0.7)
        axc.text(max(st[2], stored) + 1.5, i, f"{stored:.0f}%", fontsize=7, va="center", ha="left",
                 color="white", weight="bold")
        first = False
        fr = load(HC / d / "choice_free_summary.json")
        dot(axz, i + 0.22, share(c["post_relay"], c["text_swap"]), COL["relay"], "o", True, s=12, horizontal=True)
        dot(axz, i, share(c["sentence_edit"], c["text_swap"]), COL["relay"], "o", False, s=12, horizontal=True)
        if fr:
            dot(axz, i - 0.22, share(fr["emission"], fr["total"]), COL["emission"], "D", True, s=10,
                horizontal=True)
    axc.set_yticks(range(len(models)))
    axc.set_yticklabels([m[1] for m in models])
    axc.set_ylim(len(models) - 0.45, -0.55)
    axc.set_xlim(0, 100)
    axc.set_xticks([0, 25, 50, 75, 100])
    axc.set_xlabel("% of the text-swap reference")
    axc.legend(loc="lower center", bbox_to_anchor=(0.44, 1.0), ncol=3, handlelength=0.9, columnspacing=0.6,
               handletextpad=0.4, borderaxespad=0.1)
    axc.set_title("C   Hidden choice: stored after the list or not reproduced by\n     the post-list patch; the sentence adds almost nothing", x=-0.24, pad=16)
    axz.axvline(0, color="#999999", lw=0.6, zorder=0)
    axz.tick_params(axis="y", labelleft=False)
    axz.set_xlim(-8, 12)
    axz.set_xticks([-5, 0, 5, 10])
    axz.set_xlabel("% of the text-swap reference\n(written: % of the free-choice total)")
    hs = [Line2D([], [], ls="none", marker="o", ms=3.5, mfc=COL["relay"], mec=COL["relay"]),
          Line2D([], [], ls="none", marker="o", ms=3.5, mfc="white", mec=COL["relay"]),
          Line2D([], [], ls="none", marker="D", ms=3.2, mfc=COL["emission"], mec=COL["emission"])]
    axz.legend(hs, ["via the sentence (indirect)", "sentence positions", "written (free version)"],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, handletextpad=0.1, borderaxespad=0.1,
               labelspacing=0.15)
    save(fig, "iclr_fig4_stress_tests.png")


# --------------------------------------------------------------------------- Figure 5

def written(d, K):
    s = load(DV / d / "written_summary_recut.json") or load(DV / d / "written_summary.json")
    k = (s or {}).get("by_K", {}).get(K)
    if k and k.get("total") is not None:
        return mean(k["emission"]) / mean(k["total"]), mean(k["persistence_t1"]) / mean(k["total"]), k["n_usable"]
    rows = load(DV / d / f"written_K{K}_rows.json")
    u = [r for r in (rows or []) if r.get("usable")]
    if u:
        tot = sum(r["total"] for r in u) / len(u)
        return (sum(r["emission"] for r in u) / len(u) / tot, sum(r["persistence_t1"] for r in u) / len(u) / tot,
                len(u))
    return None


def fig5():
    fig, ax = plt.subplots(figsize=(W, 2.3))
    lx = np.log10
    unw = []
    for d, sz in QWEN:
        for f in ("chain_summary_block.json", "chain_summary.json"):
            k = (load(DV / d / f) or {}).get("by_K", {}).get("3", {})
            if "post_v0_block" in k:
                unw.append((sz, share(k["post_v0_block"], k["v0_edit"])))
                break
    wr = {}
    for K in ("3", "5"):
        for d, sz in QWEN:
            s = load(DV / d / "written_summary_recut.json") or load(DV / d / "written_summary.json")
            k = (s or {}).get("by_K", {}).get(K)
            if not (k and k.get("total") is not None):
                rows = [r for r in (load(DV / d / f"written_K{K}_rows.json") or []) if r.get("usable")]
                if rows:
                    import random
                    rng = random.Random(20260925)
                    def bs(key):
                        xs = [r[key] for r in rows]
                        m = sum(xs) / len(xs)
                        b = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(2000))
                        return {"mean": m, "lo": b[50], "hi": b[1949], "n": len(xs)}
                    k = {"emission": bs("emission"), "total": bs("total")}
            if k and k.get("total") is not None:
                wr.setdefault(K, []).append((sz, share(k["emission"], k["total"])))
    ax.axhline(0, color="#BBBBBB", lw=0.5, zorder=0)
    if unw:
        ax.plot([lx(p[0]) for p in unw], [p[1][0] for p in unw], "-", color=COL["relay"], lw=0.9, zorder=2)
        for sz, v in unw:
            dot(ax, lx(sz), v, COL["relay"], "o", True, s=16)
        ax.text(lx(unw[-1][0]) + 0.05, unw[-1][1][0] + 6, "unwritten: downstream patch effect \u2248 0",
                fontsize=7, color="#8E4F7A", ha="right", va="bottom", weight="bold")
    for K, mk, off in (("3", "o", -0.012), ("5", "s", 0.012)):
        pts = wr.get(K, [])
        if not pts:
            continue
        ax.plot([lx(p[0]) + off for p in pts], [p[1][0] for p in pts], "-", color=COL["emission"], lw=0.9,
                zorder=2, alpha=1.0 if K == "3" else 0.6)
        for sz, v in pts:
            dot(ax, lx(sz) + off, v, COL["emission"], mk, K == "3", s=16)
    if wr:
        allw = [p for K in wr for p in wr[K]]
        ax.text(lx(min(p[0] for p in allw)) - 0.02, 70, "written: the answer follows the text",
                fontsize=7, color=COL["emission"], ha="left", va="top", weight="bold")
    hs = [Line2D([], [], ls="none", marker="o", ms=3.5, mfc=COL["emission"], mec=COL["emission"]),
          Line2D([], [], ls="none", marker="s", ms=3.5, mfc="white", mec=COL["emission"]),
          Line2D([], [], ls="none", marker="o", ms=3.5, mfc=COL["relay"], mec=COL["relay"])]
    ax.legend(hs, ["written chain, K = 3", "written chain, K = 5", "unwritten chain, K = 3"], loc="center right",
              handletextpad=0.1, labelspacing=0.25)
    xt = [1.7, 4, 8, 14, 32]
    ax.set_xticks([lx(v) for v in xt])
    ax.set_xticklabels([f"{v:g}" for v in xt])
    ax.set_xlim(lx(1.4), lx(40))
    ax.set_ylim(-8, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Qwen3 parameters (B)")
    ax.set_ylabel("effect through the measured\ndownstream path (%)")
    ax.set_title("Unwritten: near-zero downstream patch effect; written: the answer follows the text")
    save(fig, "iclr_fig5_computed_and_written.png")


def figA_computed():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.0))
    fig.subplots_adjust(wspace=0.38)
    sizes = [("Qwen3-1.7B", "1.7B"), ("Qwen3-4B", "4B"), ("Qwen3-8B", "8B"), ("Qwen3-14B", "14B")]
    shades = ["#C8C8C8", "#9C9C9C", "#686868", "#2E2E2E"]
    ax = axes[0]
    for (d, lab), c in zip(sizes, shades):
        x = load(DV / d / "chain_summary.json")
        if x:
            ks = sorted(int(k) for k in x["by_K"])
            ax.plot(ks, [100 * x["by_K"][str(k)]["accuracy"] for k in ks], "-o", ms=3, lw=1, color=c, label=lab)
    ax.set_xticks([1, 3, 5])
    ax.set_xlabel("updates in the chain (K)")
    ax.set_ylabel("accuracy (% of chains)")
    ax.set_ylim(-3, 105)
    ax.legend(loc="lower left", handlelength=1.2, title="Qwen3", title_fontsize=7)
    ax.set_title("A   Unwritten chains fail with length")
    ax = axes[1]
    lev = load(SC / "leverage_analysis.json")
    if lev:
        c = lev["Qwen3-4B"]["conditions"]
        tl = ["low", "mid", "high"]
        for cond, a in [("multiplied_5x", 1.0), ("zeroed", 0.45)]:
            if cond not in c:
                continue
            t = c[cond]["by_leverage_tertile"]
            sign = 1 if cond == "multiplied_5x" else -1
            ax.plot(range(3), [sign * mean(t[k]["public"]) for k in tl], "-o", ms=3, lw=1, color=COL["emission"],
                    alpha=a, label="through the article" if cond == "multiplied_5x" else None)
            ax.plot(range(3), [sign * mean(t[k]["private"]) for k in tl], "--o", ms=3, lw=0.9, mfc="white",
                    color=COL["grey"], alpha=a, label="bypassing it" if cond == "multiplied_5x" else None)
        ax.set_ylim(-0.01, 0.42)
        ax.text(2.05, 0.335, "amplified", fontsize=7, color=COL["emission"], ha="right", va="bottom")
        ax.text(2.05, 0.19, "zeroed", fontsize=7, color=COL["emission"], alpha=0.6, ha="right", va="top")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["low", "mid", "high"])
        ax.set_xlabel("leverage of the article (tertile)")
        ax.set_ylabel("|Δ log p(planned noun)|")
        ax.legend(loc="upper left", handlelength=1.4)
    ax.set_title("B   a/an planning features (Qwen3-4B)")
    save(fig, "iclr_figA_computed_extra.png")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, figA_computed):
        f()
