#!/usr/bin/env python3
from __future__ import annotations
import json, math, random
from pathlib import Path
import numpy as np
from safetensors import safe_open

EXP=Path(__file__).resolve().parent; ROOT=EXP.parents[1]; OUT=EXP/"results/robustness.json"
rows=json.loads((EXP/"results/rows.json").read_text())
manifest=json.loads((ROOT/"experiments/archived_feature_vectors/gemma_scope_2_270m_pt_affine_discussed.manifest.json").read_text())
vecpath=ROOT/"experiments/archived_feature_vectors/gemma_scope_2_270m_pt_affine_discussed.safetensors"
norms={}
with safe_open(vecpath,framework="pt",device="cpu") as sf:
    for f in manifest["features"]:
        if any(str(r).startswith("calibration_") for r in f["roles"]):
            norms[(f["layer"],f["feature_idx"])]=float(sf.get_tensor(f["decoder_key"]).float().norm())

features=[]
for key in sorted({(r["layer"],r["feature_idx"]) for r in rows}):
    g=[r for r in rows if (r["layer"],r["feature_idx"])==key]
    mean=lambda k:sum(float(x[k]) for x in g)/len(g)
    features.append({"layer":key[0],"feature_idx":key[1],"article_attribution":mean("article_attribution"),
        "future_attribution":mean("future_attribution"),"activation":mean("activation"),"decoder_norm":norms[key],
        "article_margin_effect":mean("article_margin_effect"),"article_changed_rate":mean("article_changed"),
        "total_tv":mean("total_tv"),"mediator_tv":mean("mediator_tv"),"residual_tv_treated":mean("residual_tv_treated"),
        "fixed_mean_tv":.5*(mean("fixed_a_tv")+mean("fixed_an_tv"))})

def rankdata(values):
    order=sorted(range(len(values)),key=lambda i:values[i]); ranks=[0.0]*len(values); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]: j+=1
        r=(i+j-1)/2+1
        for q in order[i:j]: ranks[q]=r
        i=j
    return np.asarray(ranks,dtype=float)
def rho(xs,ys):
    x,y=rankdata(xs),rankdata(ys); x=x-x.mean(); y=y-y.mean(); den=float(np.linalg.norm(x)*np.linalg.norm(y))
    return None if den==0 else float(np.dot(x,y)/den)
def bootstrap(x,y,seed=20260829,n=10000):
    rng=random.Random(seed); m=len(x); vals=[]
    for _ in range(n):
        ix=[rng.randrange(m) for _ in range(m)]; v=rho([x[i] for i in ix],[y[i] for i in ix])
        if v is not None: vals.append(v)
    vals.sort(); return {"rho":rho(x,y),"lo":vals[int(.025*(len(vals)-1))],"hi":vals[int(.975*(len(vals)-1))],"resamples":n}
def residualize_rank(y,controls):
    yr=rankdata(y); X=np.column_stack([np.ones(len(y))]+[rankdata(c) for c in controls]); return yr-X@np.linalg.lstsq(X,yr,rcond=None)[0]
def auc(scores,labels):
    pos=[s for s,l in zip(scores,labels) if l]; neg=[s for s,l in zip(scores,labels) if not l]
    if not pos or not neg:return None
    return sum((p>n)+.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))

out={"n_features":len(features),"features":features,"analyses":{}}
for predictor in ("article_attribution","future_attribution"):
    x=[f[predictor] for f in features]; out["analyses"][predictor]={}
    for outcome in ("article_margin_effect","total_tv","mediator_tv","residual_tv_treated","fixed_mean_tv"):
        y=[f[outcome] for f in features]; base=bootstrap(x,y,seed=20260829+len(outcome))
        loo=[rho(x[:i]+x[i+1:],y[:i]+y[i+1:]) for i in range(len(x))]
        j=max(range(len(y)),key=lambda i:abs(y[i])); controls=[[f[k] for f in features] for k in ("layer","activation","decoder_norm")]
        px=residualize_rank(x,controls); py=residualize_rank(y,controls)
        threshold=float(np.median(np.abs([f["article_margin_effect"] for f in features])))
        keep=[i for i,f in enumerate(features) if abs(f["article_margin_effect"])>=threshold]
        out["analyses"][predictor][outcome]={**base,"loo_min":min(v for v in loo if v is not None),"loo_max":max(v for v in loo if v is not None),
            "largest_outcome_excluded_rho":rho(x[:j]+x[j+1:],y[:j]+y[j+1:]),"excluded_feature":{"layer":features[j]["layer"],"feature_idx":features[j]["feature_idx"]},
            "partial_spearman_layer_activation_decoder_norm":rho(list(px),list(py)),
            "conditional_on_above_median_abs_margin_rho":rho([x[i] for i in keep],[y[i] for i in keep]),"conditional_n":len(keep)}
    out["analyses"][predictor]["article_boundary_crossing_auc"]=auc(x,[f["article_changed_rate"]>0 for f in features])
OUT.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({k:{o:v if o=='article_boundary_crossing_auc' else {q:v[q] for q in ('rho','lo','hi','loo_min','loo_max','largest_outcome_excluded_rho','partial_spearman_layer_activation_decoder_norm','conditional_on_above_median_abs_margin_rho')} for o,v in b.items()} for k,b in out['analyses'].items()},indent=2))
