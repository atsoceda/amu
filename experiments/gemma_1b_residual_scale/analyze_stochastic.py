#!/usr/bin/env python3
import json, math, random
from pathlib import Path

EXP=Path(__file__).resolve().parent; R=EXP/"results"
models={"gemma_1b":json.loads((R/"rows.json").read_text()),"gemma_270m":json.loads((R/"rows_270m_matched.json").read_text())}
def interval(v,seed=20260829,n=10000):
 rng=random.Random(seed); m=len(v); b=sorted(sum(v[rng.randrange(m)] for _ in range(m))/m for _ in range(n))
 return {"n":m,"mean":sum(v)/m,"lo":b[math.floor(.025*(n-1))],"hi":b[math.ceil(.975*(n-1))]}
out={}
for model,rows in models.items():
 out[model]={}
 for regime in ("between","within"):
  group=[r for r in rows if r["regime"]==regime and (model=="gemma_270m" or r.get("strength")==1.0)]
  out[model][regime]={}
  for tau in ("0.1","0.25","0.5","1.0"):
   out[model][regime][tau]={k:interval([r["stochastic"][tau][k] for r in group],seed=20260829+i) for i,k in enumerate(("total_tv","public_tv","private_tv","off_article_mass","on_article_mass"))}
(R/"stochastic_cross_scale.json").write_text(json.dumps(out,indent=2)+"\n")
for model in out:
 for regime in out[model]:
  print(model,regime,{t:{k:round(v["mean"],4) for k,v in out[model][regime][t].items()} for t in out[model][regime]})
