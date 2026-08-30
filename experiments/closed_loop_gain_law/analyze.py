#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
RESULTS = EXP / "results"
TEMPERATURES = (0.1, 0.25, 0.5, 1.0)


def load(path):
    return json.loads(path.read_text())


def q(margin, tau):
    value=max(-80.0,min(80.0,margin/tau));return 1/(1+math.exp(-value))


def r2(observed,predicted):
    y,p=np.asarray(observed),np.asarray(predicted);den=float(((y-y.mean())**2).sum())
    return None if den==0 else 1-float(((y-p)**2).sum())/den


def mae(observed,predicted):
    return float(np.mean(np.abs(np.asarray(observed)-np.asarray(predicted))))


def evaluate(rows,tau):
    feature_ids=sorted({row["feature_index"] for row in rows})
    observed=[];predictions={key:[] for key in ("attribution_only","attribution_susceptibility","attribution_full",
                                                 "measured_margin_only","measured_susceptibility","measured_full")}
    for heldout in feature_ids:
        train=[row for row in rows if row["feature_index"]!=heldout];test=[row for row in rows if row["feature_index"]==heldout]
        def fit(keys,outcome):
            design=np.asarray([[1.0]+[transform(row[key]) for key,transform in keys] for row in train]);target=np.asarray([outcome(row) for row in train])
            return np.linalg.lstsq(design,target,rcond=None)[0]
        beta_attr_public=fit([("article_attribution",abs)],lambda row:abs(q(row["baseline_margin"]+row["article_margin_effect"],tau)-q(row["baseline_margin"],tau))*row["branch_leverage_tv"])
        beta_attr_margin=fit([("article_attribution",float)],lambda row:row["article_margin_effect"])
        beta_margin_public=fit([("article_margin_effect",abs)],lambda row:abs(q(row["baseline_margin"]+row["article_margin_effect"],tau)-q(row["baseline_margin"],tau))*row["branch_leverage_tv"])
        mean_leverage=sum(row["branch_leverage_tv"] for row in train)/len(train)
        for row in test:
            actual_dq=q(row["baseline_margin"]+row["article_margin_effect"],tau)-q(row["baseline_margin"],tau)
            actual=abs(actual_dq)*row["branch_leverage_tv"]
            predicted_dm=float(beta_attr_margin[0]+beta_attr_margin[1]*row["article_attribution"])
            predicted_dq=q(row["baseline_margin"]+predicted_dm,tau)-q(row["baseline_margin"],tau)
            observed.append(actual)
            predictions["attribution_only"].append(max(0.0,float(beta_attr_public[0]+beta_attr_public[1]*abs(row["article_attribution"]))))
            predictions["attribution_susceptibility"].append(abs(predicted_dq)*mean_leverage)
            predictions["attribution_full"].append(abs(predicted_dq)*row["branch_leverage_tv"])
            predictions["measured_margin_only"].append(max(0.0,float(beta_margin_public[0]+beta_margin_public[1]*abs(row["article_margin_effect"]))))
            predictions["measured_susceptibility"].append(abs(actual_dq)*mean_leverage)
            predictions["measured_full"].append(actual)
    return {key:{"r2":r2(observed,value),"mae":mae(observed,value)} for key,value in predictions.items()}|{"n":len(observed),"mean_observed_public_tv":float(np.mean(observed))}


def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    models={
        "gemma_270m":load(ROOT/"experiments/attribution_channel_calibration/results/aligned_rows.json"),
        "gemma_1b":load(ROOT/"experiments/gemma_1b_attribution_channel_calibration/results/aligned_rows.json"),
    }
    out={"identity":"TV_public = |q1-q0| * TV(Y_an,Y_a)","cross_validation":"leave one feature out","models":{}}
    for model,rows in models.items():out["models"][model]={str(tau):evaluate(rows,tau) for tau in TEMPERATURES}
    (RESULTS/"summary.json").write_text(json.dumps(out,indent=2)+"\n")
    lines=["# Closed-loop public gain law","","Leave-one-feature-out prediction of public TV.","",
           "| Model | tau | Attribution only R2 | + susceptibility R2 | + leverage R2 | Measured margin only R2 | + susceptibility R2 | Full identity R2 |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model,temperatures in out["models"].items():
        for tau,block in temperatures.items():
            lines.append(f"| {model} | {tau} | {block['attribution_only']['r2']:.3f} | {block['attribution_susceptibility']['r2']:.3f} | {block['attribution_full']['r2']:.3f} | {block['measured_margin_only']['r2']:.3f} | {block['measured_susceptibility']['r2']:.3f} | {block['measured_full']['r2']:.3f} |")
    (RESULTS/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))


if __name__=="__main__":main()
