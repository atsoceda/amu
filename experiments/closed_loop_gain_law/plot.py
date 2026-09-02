#!/usr/bin/env python3
"""Main gain synthesis plus transparent supplementary diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/closed_loop_gain_law/results"
MAIN_OUT = ROOT / "manuscript/figures/fig_attribution_channel_calibration.png"
APP_OUT = ROOT / "manuscript/figures/fig_gain_diagnostics_appendix.png"
BLUE, ORANGE, PURPLE = "#1769aa", "#d96c21", "#7b3fb2"


def load(path):
    return json.loads(path.read_text())


def main_subset(predictions):
    return [r for r in predictions if r["model"] == "gemma_270m" and r["temperature"] == .1 and r["scheme"] == "feature_prompt"]


def flow_box(ax, x, text, color):
    ax.add_patch(FancyBboxPatch((x, .37), .21, .25, boxstyle="round,pad=.015",
        facecolor=color, edgecolor="#37474f", linewidth=1.0))
    ax.text(x+.105, .495, text, ha="center", va="center", fontsize=9, weight="bold")


def plot_main(predictions, diagnostics):
    subset = main_subset(predictions)
    obs = np.asarray([r["observed"] for r in subset])
    pred = np.asarray([r["predictions"]["full_gain_model"] for r in subset])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), gridspec_kw={"width_ratios": [1.05, 1.45]})

    ax = axes[0]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("A  Closed-loop gain composition", loc="left", weight="bold", fontsize=11)
    flow_box(ax, .02, "Calibrated\nlocal pressure", "#dceaf7")
    flow_box(ax, .395, "Decoder\nsusceptibility", "#fff0b8")
    flow_box(ax, .77, "Continuation\nleverage", "#eadcf6")
    for x in (.24, .615):
        ax.add_patch(FancyArrowPatch((x, .495), (x+.14, .495), arrowstyle="-|>", mutation_scale=12, color="#546e7a", lw=1.5))
        ax.text(x+.07, .56, "×", ha="center", fontsize=15, weight="bold")
    ax.text(.5, .22, "How much does the intervention move article policy?", ha="center", fontsize=8.5, color=BLUE)
    ax.text(.5, .11, "How different are the noun continuations it selects?", ha="center", fontsize=8.5, color=PURPLE)

    ax = axes[1]
    ax.scatter(pred, obs, s=13, alpha=.25, color=BLUE, edgecolors="none", label="Held-out cells")
    limit = max(float(obs.max()), float(pred.max()))
    ax.plot([0, limit], [0, limit], color="#263238", lw=1, ls="--", label="Perfect prediction")
    bins = diagnostics["binned_calibration"]
    bx = np.asarray([b["mean_predicted"] for b in bins]); by = np.asarray([b["mean_observed"] for b in bins])
    xerr = np.asarray([[b["mean_predicted"]-b["predicted_lo"] for b in bins], [b["predicted_hi"]-b["mean_predicted"] for b in bins]])
    yerr = np.asarray([[b["mean_observed"]-b["observed_lo"] for b in bins], [b["observed_hi"]-b["mean_observed"] for b in bins]])
    ax.errorbar(bx, by, xerr=xerr, yerr=yerr, fmt="o", ms=6, capsize=2.5, color=ORANGE, label="Five equal-count bins")
    d = diagnostics["full_gain_model"]; tail = diagnostics["tail_sensitivity"]["exclude_top_5pct"]
    ax.text(.03, .96, f"overall $R^2$={d['r2']:.3f}   $\\rho$={d['spearman_rho']:.3f}\nMAE={d['mae']:.4f}   median AE={d['median_absolute_error']:.5f}",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=2))
    ax.text(.97, .05, f"excluding top 5%: $R^2$={tail['r2']:.2f}", transform=ax.transAxes,
            ha="right", fontsize=8, color="#8b1a1a")
    ax.set(xlabel="Predicted public noun TV", ylabel="Observed public noun TV",
           title="B  Unseen 270M feature × prompt cells ($N=640$)")
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(MAIN_OUT, dpi=240, bbox_inches="tight"); plt.close(fig)


def plot_appendix(validation, predictions, diagnostics, aligned):
    colors = {"270M": BLUE, "1B": ORANGE}
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.8))
    ax = axes[0, 0]
    local = {"270M": (.530,.200,.772), "1B": (.432,.014,.729)}
    for i, model in enumerate(("270M", "1B")):
        value, lo, hi = local[model]; ax.errorbar(i, value, yerr=[[value-lo],[hi-value]], fmt="o", ms=7, capsize=4, color=colors[model])
    ax.axhline(0,color="black",lw=.8); ax.set_xticks([0,1],["270M","1B"]); ax.set(ylabel="Spearman $\\rho$",title="A  Attribution → local margin",ylim=(-.2,.9))

    ax = axes[0, 1]
    metrics=[("signed_future_vs_signed_fixed_target","Target logit"),("signed_future_vs_signed_fixed_target_minus_source","Target-source")]
    for j,(key,label) in enumerate(metrics):
        for offset,model in ((-.1,"270M"),(.1,"1B")):
            d=aligned[model][key]; ax.errorbar(j+offset,d["rho"],yerr=[[d["rho"]-d["lo"]],[d["hi"]-d["rho"]]],fmt="o",capsize=4,color=colors[model],label=model if j==0 else None)
    ax.axhline(0,color="black",lw=.8);ax.set_xticks(range(2),[x[1] for x in metrics]);ax.set(title="B  Fixed-token attribution",ylim=(-.65,.75));ax.legend(frameon=False,fontsize=8)

    ax=axes[0,2]; names=[("constant","Constant"),("attribution_only","Attribution"),("susceptibility_only","Susceptibility"),("full_gain_model","Full")]
    x=np.arange(len(names));width=.36
    for offset,(key,label) in zip((-.18,.18),(("gemma_270m","270M"),("gemma_1b","1B"))):
        vals=[validation[key]["0.1"]["feature_prompt"][name]["r2"] for name,_ in names];ax.bar(x+offset,vals,width,color=colors[label],label=label)
    ax.axhline(0,color="black",lw=.8);ax.set_xticks(x,[label for _,label in names],rotation=18);ax.set(ylabel="Held-out $R^2$",title="C  Predictive baselines",ylim=(-.7,1));ax.legend(frameon=False,fontsize=8)

    ax=axes[1,0]; bins=diagnostics["binned_calibration"]
    bx=np.asarray([b["mean_predicted"] for b in bins]);by=np.asarray([b["mean_observed"] for b in bins])
    yerr=np.asarray([[b["mean_observed"]-b["observed_lo"] for b in bins],[b["observed_hi"]-b["mean_observed"] for b in bins]])
    eps=1e-7;ax.errorbar(bx+eps,by+eps,yerr=yerr,fmt="o",capsize=3,color=ORANGE)
    lim=max(bx.max(),by.max())*1.2;ax.plot([eps,lim],[eps,lim],ls="--",lw=.8,color="black");ax.set_xscale("log");ax.set_yscale("log");ax.set(xlabel="Mean predicted TV",ylabel="Mean observed TV",title="D  Five-bin calibration")

    ax=axes[1,1]; keys=["exclude_top_0pct","exclude_top_5pct","exclude_top_10pct"]; vals=[diagnostics["tail_sensitivity"][k]["r2"] for k in keys]
    bars=ax.bar(range(3),vals,color=[BLUE,"#b85c3a","#8b1a1a"]);ax.axhline(0,color="black",lw=.8)
    ax.set_yscale("symlog",linthresh=1);ax.set_ylim(-50,2)
    ax.set_xticks(range(3),["All","Drop top 5%","Drop top 10%"],rotation=15);ax.set(ylabel="$R^2$ (symlog)",title="E  High-effect-tail sensitivity")
    label_positions=[1.05,-1.0,-18.0]
    for bar,value,label_y in zip(bars,vals,label_positions):
        ax.text(bar.get_x()+bar.get_width()/2, label_y, f"{value:.2f}", ha="center", va="bottom", fontsize=7)

    ax=axes[1,2]; temps=(.1,.25,.5,1.0)
    for model,label in (("gemma_270m","270M"),("gemma_1b","1B")):
        vals=[validation[model][str(t)]["feature_prompt"]["full_gain_model"]["r2"] for t in temps];ax.plot(temps,vals,marker="o",color=colors[label],label=label)
    ax.axhline(0,color="black",lw=.8);ax.set_xscale("log");ax.set_xticks(temps,[".1",".25",".5","1"]);ax.set(xlabel="Article temperature",ylabel="Held-out $R^2$",title="F  Temperature and scale");ax.legend(frameon=False,fontsize=8)
    for ax in axes.flat: ax.spines[["top","right"]].set_visible(False);ax.grid(alpha=.15);ax.tick_params(labelsize=8);ax.title.set_fontsize(10)
    fig.tight_layout();fig.savefig(APP_OUT,dpi=240,bbox_inches="tight");plt.close(fig)


def main():
    validation=load(RESULTS/"validation_summary.json")["models"]
    predictions=load(RESULTS/"validation_predictions.json")
    diagnostics=load(RESULTS/"diagnostics.json")
    aligned={"270M":load(ROOT/"experiments/attribution_channel_calibration/results/aligned_summary.json")["analyses"],
             "1B":load(ROOT/"experiments/gemma_1b_attribution_channel_calibration/results/aligned_summary.json")["analyses"]}
    plot_main(predictions,diagnostics);plot_appendix(validation,predictions,diagnostics,aligned)
    print(MAIN_OUT);print(APP_OUT)


if __name__=="__main__":main()
