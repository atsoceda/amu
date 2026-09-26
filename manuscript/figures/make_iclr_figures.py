#!/usr/bin/env python3
"""Figures for the ICLR 2027 manuscript, generated from the committed experiment results.

Run from the repository root:
    /Users/anthony/miniconda3/bin/python manuscript/figures/make_iclr_figures.py
Every number is read from experiments/*/results. Runs that are still pending are skipped,
so the figures can be regenerated as results arrive. One colour per path throughout
(Okabe-Ito): direct retrieval blue, storage at the line's final token green, late lookup
sky blue, relay along generated text reddish purple, emission vermillion, recurrent memory
orange; one marker per model family.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "manuscript" / "figures"
EXP = ROOT / "experiments"
CR, HC, PC = EXP / "couplet_routes/results", EXP / "hidden_choice/results", EXP / "relay_positive_control/results"
RC, RD, DV = EXP / "recurrent_carry/results", EXP / "relay_distance/results", EXP / "derived_value_carry/results"
SC = EXP / "qwen3_planning_six_cell/results"

COL = {"retrieval": "#0072B2", "storage": "#009E73", "late": "#56B4E9", "relay": "#CC79A7",
       "emission": "#D55E00", "recurrent": "#E69F00", "grey": "#9A9A9A", "ink": "#222222"}
MARK = {"Qwen3": "o", "Gemma 3": "s", "Qwen3.5": "^", "Gemma 4": "D"}
plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 7.8, "axes.labelsize": 7.3, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "savefig.dpi": 300, "legend.frameon": False, "axes.titlelocation": "left",
    "axes.titleweight": "bold"})

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


# --------------------------------------------------------------------------- Figure 1

def fig1():
    fig = plt.figure(figsize=(7.0, 2.9))
    axA = fig.add_axes([0.0, 0.0, 0.43, 1.0])
    axB = fig.add_axes([0.52, 0.2, 0.48, 0.7])

    # A. Paths of the rhyme plan in one couplet, line widths from Qwen3-32B (chat prompt).
    s = couplet_shares("Qwen3-32B")
    axA.set_xlim(-0.1, 9.3)
    axA.set_ylim(-1.25, 4.35)
    axA.axis("off")
    toks = [(0.0, 0.7, "lonely", "ctx"), (0.8, 0.7, "night", "src"), (1.6, 0.36, ",", "bd"),
            (2.06, 0.74, "⟨turn⟩", "tpl"), (3.0, 0.55, "And", "gen"), (3.65, 0.65, "stars", "gen"),
            (4.4, 0.55, "will", "gen"), (5.05, 0.65, "guide", "late"), (5.8, 0.5, "me", "late"),
            (6.4, 0.8, "through", "late"), (7.3, 0.55, "the", "tgt")]
    fc = {"ctx": "#F5F5F5", "src": "#DCEAF5", "bd": "#D9F0E7", "tpl": "#F3F3F3", "gen": "#F8EEF4",
          "late": "#E6F4FB", "tgt": "#FFFFFF"}
    ec = {"ctx": "#C8C8C8", "src": COL["retrieval"], "bd": COL["storage"], "tpl": "#C8C8C8", "gen": COL["relay"],
          "late": COL["late"], "tgt": COL["ink"]}
    cx = {}
    for x, w, t, k in toks:
        axA.add_patch(FancyBboxPatch((x, 0.0), w, 0.62, boxstyle="round,pad=0.015,rounding_size=0.1", fc=fc[k],
                                     ec=ec[k], lw=1.0 if k in ("src", "tgt") else 0.7))
        axA.text(x + w / 2, 0.31, t, ha="center", va="center", fontsize=5.7, color=COL["ink"],
                 weight="bold" if k in ("src", "tgt") else "normal")
        cx[t] = x + w / 2
    axA.text(7.95, 0.31, "→ rhyme?", ha="left", va="center", fontsize=6.2, color="#666666", style="italic")
    axA.plot([0.0, 2.8], [-0.2, -0.2], color="#AAAAAA", lw=0.6)
    axA.plot([3.0, 7.85], [-0.2, -0.2], color="#AAAAAA", lw=0.6)
    axA.text(1.4, -0.42, "end of line 1 (prompt)", ha="center", va="top", fontsize=5.9, color="#666666")
    axA.text(5.42, -0.42, "line 2 so far (generated, held fixed)", ha="center", va="top", fontsize=5.9,
             color="#666666")

    def bez(x0, x1, h, share, col, ls="-", y0=0.68):
        import numpy as np
        t = np.linspace(0, 1, 60)
        xm = (x0 + x1) / 2
        xs = (1 - t) ** 2 * x0 + 2 * t * (1 - t) * xm + t ** 2 * x1
        ys = (1 - t) ** 2 * y0 + 2 * t * (1 - t) * h + t ** 2 * y0
        lw = 0.7 + 7.0 * max(share, 0.0)
        axA.plot(xs, ys, color=col, lw=lw, ls=ls, solid_capstyle="butt", alpha=0.95)
        axA.plot([x1], [y0 + 0.04], marker="v", ms=2.6 + 0.35 * lw, color=col, mec="none")

    bez(cx["night"], cx["the"], 3.3, s["retrieval"], COL["retrieval"])
    bez(cx["night"] + 0.12, cx[","], 1.1, s["boundary"], COL["storage"])
    bez(cx[","], cx["the"] - 0.08, 2.5, s["boundary"], COL["storage"])
    bez(cx["night"] - 0.12, cx["me"], 1.8, s["late"], COL["late"])
    bez(cx["me"], cx["the"] + 0.08, 1.05, s["late"], COL["late"])
    for a, b in (("And", "stars"), ("stars", "will"), ("will", "guide"), ("guide", "me")):
        bez(cx[a], cx[b], 0.95, max(s["early"], 0.0), COL["relay"], ls=(0, (1.6, 1.0)))
    rows = [(COL["retrieval"], "-", "direct retrieval from the rhyme word", f"{100 * s['retrieval']:.0f}%"),
            (COL["storage"], "-", "stored at the line's final token (the comma)",
             f"{100 * s['boundary']:.0f}%  (necessity {100 * s['nec_boundary']:.0f}%)"),
            (COL["late"], "-", "late lookup via the last three positions", f"{100 * s['late']:.0f}%"),
            (COL["relay"], (0, (1.6, 1.0)), "relay along the rest of line 2", f"{100 * s['early']:.0f}%")]
    for i, (c, ls, lab, num) in enumerate(rows):
        yy = 4.0 - 0.36 * i
        axA.plot([0.0, 0.55], [yy, yy], color=c, lw=2.0, ls=ls)
        axA.text(0.7, yy, lab, va="center", fontsize=6.2, color=COL["ink"])
        axA.text(6.05, yy, num, va="center", fontsize=6.2, color=c if c != COL["late"] else "#2B86BD",
                 weight="bold")
    axA.text(-0.05, -1.0, "Qwen3-32B, chat prompt. The donor's rhyme state is written at the rhyme word (all layers);\n"
             "line widths show each path's share of the effect at the position that predicts the rhyme.",
             fontsize=5.6, color="#555555", va="center")
    axA.set_title("A   Where a rhyme plan travels", fontsize=7.8, x=0.0, y=0.99)

    # B. Relay along intermediate positions, every setting, next to the induction positive control.
    groups = [("induction\n(positive\ncontrol)", []), ("rhyme plan\n(couplets)", []), ("computed\nvalue", []),
              ("hidden\nchoice", []), ("recurrent\nmemory", []), ("beyond the\nattention\nwindow", [])]
    ctrl_ret = []
    for d, fam in [("Qwen3-1.7B", "Qwen3"), ("Qwen3-4B", "Qwen3"), ("Qwen3-8B", "Qwen3"), ("Qwen3-14B", "Qwen3"),
                   ("Qwen3-32B", "Qwen3"), ("gemma-3-12b-it", "Gemma 3"), ("gemma-3-27b-it", "Gemma 3")]:
        x = load(PC / d / "induction_summary.json")
        if x:
            groups[0][1].append((fam, ratio(x["relay"], x["persistence"])))
            ctrl_ret.append((fam, ratio(x["retrieval"], x["persistence"])))
    for runs, fam in ((QWEN, "Qwen3"), (GEMMA, "Gemma 3")):
        for d, _ in runs:
            for r in (d, d + "-plain"):
                c = couplet_shares(r)
                if c:
                    groups[1][1].append((fam, c["late"] + c["early"]))
    for d in ("Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B"):
        x = load(DV / d / "chain_summary_block.json") or load(DV / d / "chain_summary.json")
        if x:
            for K in ("1", "3", "5"):
                k = x["by_K"].get(K, {})
                if "post_v0_block" in k:
                    groups[2][1].append(("Qwen3", ratio(k["post_v0_block"], k["v0_edit"])))
    for d, fam in [("Qwen3-4B", "Qwen3"), ("Qwen3-8B", "Qwen3"), ("Qwen3-14B", "Qwen3"), ("Qwen3-32B", "Qwen3"),
                   ("gemma-3-12b-it", "Gemma 3"), ("gemma-3-27b-it", "Gemma 3"), ("Qwen3.5-27B", "Qwen3.5")]:
        x = load(HC / d / "choice_summary.json")
        if x:
            groups[3][1].append((fam, ratio(x["post_relay"], x["text_swap"])))
    for d in ("Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-27B", "Qwen3.5-35B-A3B"):
        x = load(RC / d / "pilot_summary.json") or load(RC / d / "pilot24_summary.json")
        if x:
            groups[4][1].append(("Qwen3.5", ratio(x["recurrent_only"], x["persistence"])))
    for f in ("pilot_v2x_summary.json", "pilot_v2x_window_summary.json"):
        x = load(RD / "gemma-3-27b-it" / f)
        if x:
            for D, v in x["by_distance"].items():
                if int(D) >= 1500 and v.get("n"):
                    groups[5][1].append(("Gemma 3", ratio(v["relay_line2"], v["persistence"])))
    axB.axvspan(-0.5, 0.5, color="#F1F1F1", zorder=0, lw=0)
    for gi, (_, pts) in enumerate(groups):
        n = len(pts)
        for j, (fam, v) in enumerate(pts):
            dx = (j - (n - 1) / 2) * min(0.08, 0.7 / max(n, 1))
            axB.scatter(gi + dx, 100 * min(v, 1.45), marker=MARK[fam], s=13, color=COL["relay"], zorder=3,
                        edgecolor="white", linewidth=0.3)
    for j, (fam, v) in enumerate(ctrl_ret):
        dx = (j - (len(ctrl_ret) - 1) / 2) * 0.08
        axB.scatter(dx, 100 * v, marker=MARK[fam], s=13, facecolor="white", edgecolor=COL["retrieval"], lw=0.8,
                    zorder=3)
    axB.axhline(0, color="#BBBBBB", lw=0.5, zorder=1)
    axB.set_xticks(range(len(groups)))
    axB.set_xticklabels([g for g, _ in groups], fontsize=6.1)
    axB.set_xlim(-0.5, len(groups) - 0.5)
    axB.set_ylim(-8, 150)
    axB.set_yticks([0, 25, 50, 75, 100, 125])
    axB.set_ylabel("share of the effect carried along\nintermediate positions (%)")
    axB.text(0.0, 149, "shares > 100%:\ninteraction", fontsize=5.3, ha="center", va="top", color="#777777")
    hs = [plt.Line2D([], [], marker=MARK[f], ls="", color="#555555", ms=3.6, label=f)
          for f in ("Qwen3", "Gemma 3", "Qwen3.5")]
    hs += [plt.Line2D([], [], marker="o", ls="", color=COL["relay"], ms=3.6, label="relay"),
           plt.Line2D([], [], marker="o", ls="", mfc="white", mec=COL["retrieval"], ms=3.6,
                      label="direct retrieval (control)")]
    axB.legend(handles=hs, loc="upper right", ncol=2, fontsize=5.8, handletextpad=0.2, columnspacing=0.8)
    axB.set_title("B   Relay is detected where it is the mechanism, and is small elsewhere", fontsize=7.8)
    save(fig, "iclr_fig1_overview.png")


# --------------------------------------------------------------------------- Figure 2

def fig2():
    panels = [("Qwen3, chat prompt", [(d, s, False) for d, s in QWEN]),
              ("Qwen3, plain prompt", [(d + "-plain", s, False) for d, s in QWEN]
               + [("Qwen3-8B-Base-plain", 8, True), ("Qwen3-14B-Base-plain", 14, True)]),
              ("Gemma 3, chat prompt", [(d, s, False) for d, s in GEMMA]),
              ("Gemma 3, plain prompt", [(d + "-plain", s, False) for d, s in GEMMA]
               + [("gemma-3-12b-pt-plain", 12, True), ("gemma-3-27b-pt-plain", 27, True)])]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.55), gridspec_kw={"height_ratios": [1.35, 1]})
    parts = [("retrieval", "direct retrieval", COL["retrieval"]),
             ("boundary", "stored at the line's final token", COL["storage"]),
             ("late", "late lookup (last 3 positions)", COL["late"]),
             ("early", "relay along earlier line 2", COL["relay"])]
    for ci, (title, runs) in enumerate(panels):
        ax, bx = axes[0, ci], axes[1, ci]
        labels, xs, suff, nec, base_pts = [], [], [], [], []
        for i, (run, size, is_base) in enumerate([r for r in runs if couplet_shares(r[0])]):
            c = couplet_shares(run)
            bottom = 0.0
            for k, _, colr in parts:
                v = 100 * max(c[k], 0.0)
                ax.bar(i, v, 0.72, bottom=bottom, color=colr, lw=0)
                bottom += v
            ax.bar(i, max(100 - bottom, 0.0), 0.72, bottom=bottom, color="white", edgecolor="#BBBBBB",
                   hatch="/////", lw=0.3)
            labels.append(f"{size:g}B" + ("\nbase" if is_base else ""))
            xs.append(i)
            if is_base:
                base_pts.append((size, 100 * c["boundary"]))
            else:
                suff.append((size, 100 * c["boundary"]))
                if "nec_boundary" in c:
                    nec.append((size, 100 * c["nec_boundary"]))
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=7.2)
        if ci == 0:
            ax.set_ylabel("share of persistence (%)")
        else:
            ax.set_yticklabels([])
        if suff:
            bx.plot(*zip(*suff), "-o", color=COL["storage"], ms=3.2, lw=1.1, label="sufficiency")
        if nec:
            bx.plot(*zip(*nec), "--o", color=COL["storage"], mfc="white", ms=3.2, lw=0.9, label="necessity")
        if base_pts:
            bx.scatter(*zip(*base_pts), marker="*", s=36, color=COL["ink"], zorder=4, label="base checkpoint")
        bx.set_xscale("log")
        bx.set_xticks([1, 2, 4, 8, 16, 32])
        bx.set_xticklabels(["1", "2", "4", "8", "16", "32"])
        bx.minorticks_off()
        bx.set_xlim(0.8, 40)
        bx.set_ylim(-6, 40)
        bx.axhline(0, color="#BBBBBB", lw=0.5)
        bx.set_xlabel("parameters (B)")
        if ci == 0:
            bx.set_ylabel("line's final token\n(% of persistence)")
        else:
            bx.set_yticklabels([])
        if ci == 1:
            bx.legend(loc="upper left", fontsize=5.8, handlelength=1.6)
    hs = [Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts]
    hs.append(Rectangle((0, 0), 1, 1, fc="white", ec="#BBBBBB", hatch="/////", lw=0.3))
    fig.legend(hs, [l for _, l, _ in parts] + ["interaction (non-additive remainder)"], loc="upper center",
               ncol=5, fontsize=6.1, bbox_to_anchor=(0.5, 1.045), handlelength=1.1, columnspacing=1.0)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, "iclr_fig2_couplet_routes.png")


# --------------------------------------------------------------------------- Figure 3

PRETTY = {"<|im_end|>": "⟨im_end⟩", "<|im_start|>": "⟨im_start⟩", "<think>": "⟨think⟩", "</think>": "⟨/think⟩",
          "<end_of_turn>": "⟨end_turn⟩", "<start_of_turn>": "⟨start_turn⟩", "Ċ": "↵", "ĊĊ": "↵↵", "\n": "↵",
          ",\n": ",↵", ",Ċ": ",↵"}


def pretty(t):
    return PRETTY.get(t, t.replace("▁", "").replace("Ġ", "").replace("Ċ", "↵").replace("\n", "↵"))


def strip(ax, toks, vals, label, vmax):
    n = len(toks)
    ax.set_xlim(-0.05, n + 0.05)
    ax.set_ylim(-0.05, 1.45)
    ax.axis("off")
    for i, (t, v) in enumerate(zip(toks, vals)):
        a = max(0.0, min(1.0, v / vmax))
        ax.add_patch(Rectangle((i + 0.04, 0.05), 0.92, 0.72, fc=COL["storage"], alpha=0.05 + 0.9 * a,
                               ec="#D0D0D0", lw=0.3))
        lab = pretty(t)
        fs = 5.6 if len(lab) <= 6 else 4.4
        ax.text(i + 0.5, 0.41, lab, ha="center", va="center", fontsize=fs,
                color="white" if a > 0.55 else COL["ink"])
        if v >= 0.1 * vmax:
            ax.text(i + 0.5, 0.98, f"{v:+.2f}", ha="center", va="center", fontsize=5.3, color="#1B6E53")
    ax.text(-0.05, 1.33, label, ha="left", va="center", fontsize=6.4, color=COL["ink"])


def fig3():
    rows = []
    for run, lab in [("Qwen3-14B", "Qwen3-14B, chat prompt"), ("Qwen3-32B", "Qwen3-32B, chat prompt"),
                     ("gemma-3-27b-it", "Gemma 3 27B, chat prompt"),
                     ("gemma-3-12b-it-plain", "Gemma 3 12B, plain prompt")]:
        s = load(CR / run / "relay_boundary_summary.json")
        if s and s.get("necessity_by_token"):
            rows.append((lab, [x["token"] for x in s["necessity_by_token"]], [x["mean"] for x in s["necessity_by_token"]]))
    hid = []
    for run, dom, lab in [("gemma-3-27b-it", "fruits", "Gemma 3 27B, fruits"), ("gemma-3-12b-it", "fruits",
                                                                                   "Gemma 3 12B, fruits"),
                          ("gemma-3-27b-it", "animals", "Gemma 3 27B, animals"),
                          ("gemma-3-27b-it", "fruits_alt", "Gemma 3 27B, fruits, reworded instruction")]:
        s = load(HC / run / f"choice_replicate_{dom}_localize_summary.json")
        if s:
            hid.append((lab, [x["token"] for x in s["necessity_by_token"]], [x["mean"] for x in s["necessity_by_token"]]))
    n = len(rows) + len(hid)
    H = 0.42 * n + 0.55
    fig = plt.figure(figsize=(7.0, H))
    rh = 0.42 / H
    y = 1.0 - 0.2 / H
    fig.text(0.0, y, "A   Rhyme plan: necessity of each token between line 1 and line 2", fontsize=7.6,
             weight="bold", va="top")
    y -= 0.12 / H
    for lab, t, v in rows:
        y -= rh
        ax = fig.add_axes([0.0, y, 0.035 * len(t), rh * 0.95])
        strip(ax, t, v, lab, max(max(v), 1e-6))
    y -= 0.2 / H
    fig.text(0.0, y, "B   Hidden choice: necessity of each token between the list and the reveal "
             "(pick-specific part)", fontsize=7.6, weight="bold", va="top")
    y -= 0.12 / H
    for lab, t, v in hid:
        y -= rh
        ax = fig.add_axes([0.0, y, min(1.0, 0.035 * len(t)), rh * 0.95])
        strip(ax, t, v, lab, max(max(v), 1e-6))
    save(fig, "iclr_fig3_storage_sites.png")


# --------------------------------------------------------------------------- Figure 4

def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [0.9, 1.25, 1.3]})
    # A. Recurrent hybrid.
    ax = axes[0]
    rows = []
    for d, lab in [("Qwen3.5-4B", "4B"), ("Qwen3.5-9B", "9B"), ("Qwen3.5-27B", "27B"),
                   ("Qwen3.5-35B-A3B", "35B\nMoE")]:
        x = load(RC / d / "pilot_summary.json") or load(RC / d / "pilot24_summary.json")
        if x:
            rows.append((lab, 100 * ratio(x["attention_only"], x["persistence"]),
                         100 * ratio(x["recurrent_only"], x["persistence"])))
    for i, (lab, a, r) in enumerate(rows):
        ax.bar(i - 0.19, a, 0.36, color=COL["retrieval"], label="through attention layers" if i == 0 else None)
        ax.bar(i + 0.19, r, 0.36, color=COL["recurrent"], label="through recurrent memory" if i == 0 else None)
        ax.text(i + 0.19, r + 2, f"{r:.0f}%", ha="center", fontsize=5.8, color="#9A6A00")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("share of persistence (%)")
    ax.set_xlabel("Qwen3.5 (3 of 4 layers recurrent)")
    ax.legend(loc="upper left", fontsize=5.8, handlelength=1.0)
    ax.set_title("A   Recurrent hybrid")
    # B. Distance beyond Gemma 3's local attention window.
    ax = axes[1]
    byd = {}
    for f in ("pilot_v2x_summary.json", "pilot_v2x_window_summary.json"):
        x = load(RD / "gemma-3-27b-it" / f)
        if x:
            byd.update({int(k): v for k, v in x["by_distance"].items() if v.get("n")})
    ds = sorted(byd)
    if ds:
        def ser(k):
            return [100 * ratio(byd[d][k], byd[d]["persistence"]) for d in ds]
        ax.axvspan(1024, max(ds) + 150, color="#F3F3F3", lw=0, zorder=0)
        ax.text(1070, 29, "line 1 beyond the 1,024-token\nwindow of 5 of 6 layers", fontsize=5.5,
                color="#666666", va="top")
        ax.plot(ds, ser("retrieval"), "-o", color=COL["retrieval"], ms=3.3, lw=1.1, label="direct retrieval")
        ax.plot(ds, ser("necessity_boundary"), "-o", color=COL["storage"], ms=3.3, lw=1.1,
                label="line's final token (necessity)")
        ax.plot(ds, ser("relay_line2"), "-o", color=COL["relay"], ms=3.3, lw=1.1, label="relay along generated text")
        ax.set_xlim(-120, max(ds) + 150)
        ax.set_xticks(ds)
    ax.set_ylim(-3, 100)
    ax.set_xlabel("filler tokens before line 2 (Gemma 3 27B)")
    ax.set_ylabel("share of persistence (%)")
    ax.legend(loc="upper left", fontsize=5.8, bbox_to_anchor=(0.0, 1.02), handlelength=1.4)
    ax.set_title("B   Beyond the attention window")
    # C. Hidden choice: stored, but neither relayed nor written.
    ax = axes[2]
    models = [("Qwen3-8B", "8B"), ("Qwen3-14B", "14B"), ("Qwen3-32B", "32B"), ("Qwen3.5-27B", "27B"),
              ("gemma-3-4b-it", "4B"), ("gemma-3-12b-it", "12B"), ("gemma-3-27b-it", "27B")]
    for x0, x1, fam in ((0, 2, "Qwen3"), (3, 3, "Qwen3.5"), (4, 6, "Gemma 3")):
        ax.text((x0 + x1) / 2, -21, fam, ha="center", va="top", fontsize=5.9, color="#444444")
        ax.plot([x0 - 0.3, x1 + 0.3], [-19.5, -19.5], color="#999999", lw=0.5, clip_on=False)
    first = {"a": True, "f": True, "r": True, "e": True}
    for i, (d, lab) in enumerate(models):
        a = load(HC / d / "choice_replicate_animals_summary.json")
        f = load(HC / d / "choice_outlist_summary.json")
        c = load(HC / d / "choice_summary.json")
        fr = load(HC / d / "choice_free_summary.json")
        if a:
            ax.bar(i - 0.2, 100 * ratio(a["post_specificity"], a["text_swap_specificity"]), 0.38, color=COL["storage"],
                   label="stored pick, animals (pre-registered)" if first["a"] else None)
            first["a"] = False
        if f and "pick_specificity_post" in f:
            sw = mean(f["text_swap"]) - mean(f["text_swap_others"])
            ax.bar(i + 0.2, 100 * ratio(f["pick_specificity_post"], sw), 0.38, color=COL["storage"], alpha=0.45,
                   label="stored pick, fruits" if first["f"] else None)
            first["f"] = False
        if c:
            ax.scatter(i - 0.12, 100 * ratio(c["post_relay"], c["text_swap"]) - 7, marker="v", s=11,
                       color=COL["relay"], zorder=4, label="relayed through the sentence" if first["r"] else None)
            first["r"] = False
        if fr:
            ax.scatter(i + 0.14, 100 * ratio(fr["emission"], fr["total"]) - 7, marker="x", s=11,
                       color=COL["emission"], lw=0.9, zorder=4,
                       label="written into the sentence" if first["e"] else None)
            first["e"] = False
    ax.axhline(-7, color="#CCCCCC", lw=0.4)
    ax.text(len(models) - 0.45, -6.2, "≈ 0", fontsize=5.6, color="#777777", va="bottom", ha="right")
    ax.axhline(0, color="#BBBBBB", lw=0.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m[1] for m in models], fontsize=5.8)
    ax.set_ylim(-12, 55)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_ylabel("% of the text-swap reference")
    ax.legend(loc="upper left", fontsize=5.5, handlelength=1.0)
    ax.set_title("C   A hidden choice is stored, not relayed")
    fig.tight_layout(w_pad=0.8)
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
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95), gridspec_kw={"width_ratios": [1, 1.05, 1, 1.05]})
    sizes = [("Qwen3-1.7B", "1.7B"), ("Qwen3-4B", "4B"), ("Qwen3-8B", "8B"), ("Qwen3-14B", "14B")]
    shades = ["#C8C8C8", "#9C9C9C", "#686868", "#2E2E2E"]
    ax = axes[0]
    for (d, lab), c in zip(sizes, shades):
        x = load(DV / d / "chain_summary.json")
        if x:
            ks = sorted(int(k) for k in x["by_K"])
            ax.plot(ks, [100 * x["by_K"][str(k)]["accuracy"] for k in ks], "-o", ms=3, lw=1, color=c, label=lab)
    ax.set_xticks([1, 3, 5])
    ax.set_xlabel("updates in the chain")
    ax.set_ylabel("accuracy, unwritten (%)")
    ax.set_ylim(-3, 105)
    ax.legend(fontsize=5.6, loc="lower left", handlelength=1.2)
    ax.set_title("A   Unwritten chains")
    ax = axes[1]
    labs, src, blk = [], [], []
    for d, lab in sizes:
        x = load(DV / d / "chain_summary_block.json") or load(DV / d / "chain_summary.json")
        k = (x or {}).get("by_K", {}).get("3", {})
        if "post_v0_block" in k:
            labs.append(lab)
            src.append(mean(k["v0_edit"]))
            blk.append(mean(k["post_v0_block"]))
    xs = range(len(labs))
    ax.bar([i - 0.19 for i in xs], src, 0.36, color=COL["retrieval"], label="source digits")
    ax.bar([i + 0.19 for i in xs], blk, 0.36, color=COL["relay"], label="every later position")
    for i, b in enumerate(blk):
        ax.text(i + 0.19, max(b, 0) + 0.8, f"{b:+.1f}", ha="center", fontsize=5.4, color="#8E4F7A")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labs)
    ax.set_ylabel("effect of donor states (log-odds)")
    ax.set_ylim(0, 44)
    ax.legend(fontsize=5.6, loc="upper center", ncol=2, columnspacing=0.6, handlelength=1.0)
    ax.set_title("B   Source vs later positions")
    ax = axes[2]
    for K, mk in (("3", "o"), ("5", "s")):
        pts = [(sz, written(d, K)) for d, sz in [("Qwen3-4B", 4), ("Qwen3-8B", 8), ("Qwen3-14B", 14), ("Qwen3-32B", 32)]]
        pts = [(sz, w) for sz, w in pts if w]
        if pts:
            ax.plot([p[0] for p in pts], [100 * p[1][0] for p in pts], "-", marker=mk, ms=3, lw=1,
                    color=COL["emission"], label=f"emission, K={K}")
            ax.plot([p[0] for p in pts], [100 * p[1][1] for p in pts], "--", marker=mk, ms=3, lw=0.9, mfc="white",
                    color=COL["grey"], label=f"bypass, K={K}")
    ax.set_xscale("log")
    ax.set_xticks([4, 8, 16, 32])
    ax.set_xticklabels(["4", "8", "16", "32"])
    ax.minorticks_off()
    ax.set_ylim(-3, 105)
    ax.set_xlabel("parameters (B)")
    ax.set_ylabel("share of the effect (%)")
    ax.legend(fontsize=5.2, loc="center right", handlelength=1.4)
    ax.set_title("C   Written chains")
    ax = axes[3]
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
        ax.text(2.0, 0.335, "amplified", fontsize=5.4, color=COL["emission"], ha="right", va="bottom")
        ax.text(2.0, 0.19, "zeroed", fontsize=5.4, color=COL["emission"], alpha=0.6, ha="right", va="top")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["low", "mid", "high"])
        ax.set_xlabel("leverage of the article (tertile)")
        ax.set_ylabel("|Δ log p(planned noun)|")
        ax.legend(fontsize=5.6, loc="upper left", handlelength=1.4)
    ax.set_title("D   a/an planning (Qwen3-4B)")
    fig.tight_layout(w_pad=0.6)
    save(fig, "iclr_fig5_computed_and_written.png")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5):
        f()
