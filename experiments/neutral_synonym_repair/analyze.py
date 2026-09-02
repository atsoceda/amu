#!/usr/bin/env python3
"""Primary simple effects and audit summary for the neutral synonym repair."""
from __future__ import annotations
import itertools, json, math, random
from pathlib import Path
from typing import Any

EXP=Path(__file__).resolve().parent; RESULTS=EXP/"results"

def atomic(path:Path,value:Any):
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2)+"\n");tmp.replace(path)

def interval(values,seed,resamples=10000):
    rng=random.Random(seed);n=len(values);draws=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(resamples))
    return {"n":n,"mean":sum(values)/n,"lo":draws[math.floor(.025*(resamples-1))],"hi":draws[math.ceil(.975*(resamples-1))],"method":"semantic-family bootstrap","resamples":resamples}

def sign_flip(values):
    observed=sum(values)/len(values);null=[sum(s*v for s,v in zip(signs,values))/len(values) for signs in itertools.product((-1,1),repeat=len(values))]
    return {"observed":observed,"one_sided_positive_p":sum(x>=observed-1e-15 for x in null)/len(null),
        "two_sided_p":sum(abs(x)>=abs(observed)-1e-15 for x in null)/len(null),"assignments":len(null),"method":"exact within-family sign-flip"}

def main():
    rows=json.loads((RESULTS/"rows.json").read_text());screen=json.loads((RESULTS/"screen_rows.json").read_text())
    out={"primary_setting":{"strength":1.0,"temperature":1.0},"effects":{},"audits":{}}
    for regime in ("between","within"):
        g=[r for r in rows if r["regime"]==regime and r["strength"]==1.0];tau="1.0"
        if regime=="between":
            tv=[r["stochastic"][tau]["public"]["tv"]-r["stochastic"][tau]["private"]["tv"] for r in g]
            aligned=[r["stochastic"][tau]["public"]["target_minus_source"]-r["stochastic"][tau]["private"]["target_minus_source"] for r in g]
            direction="public minus private"
        else:
            tv=[r["stochastic"][tau]["private"]["tv"]-r["stochastic"][tau]["public"]["tv"] for r in g]
            aligned=[r["stochastic"][tau]["private"]["target_minus_source"]-r["stochastic"][tau]["public"]["target_minus_source"] for r in g]
            direction="private minus public"
        out["effects"][regime]={"direction":direction,"tv":{"interval":interval(tv,20260910),"randomization":sign_flip(tv),"values":tv},
            "target_aligned":{"interval":interval(aligned,20260911),"randomization":sign_flip(aligned),"values":aligned}}
        out["audits"][regime]={"local_efficacy":interval([r["target_branch_delta_delta"] for r in g],20260912),
            "local_efficacy_pass_n":sum(r["local_efficacy_pass"] for r in g),"n":len(g),
            "wrong_target":interval([r["controls"]["wrong"] for r in g],20260913),"sign_reversed":interval([r["controls"]["sign_reversed"] for r in g],20260914),
            "donor_separation":interval([r["donor_separation"] for r in screen if r["regime"]==regime],20260915)}
    out["terminology_decision"]="double dissociation" if all(out["effects"][r]["target_aligned"]["interval"]["lo"]>0 and out["effects"][r]["target_aligned"]["randomization"]["one_sided_positive_p"]<=.05 for r in ("between","within")) else "route interaction"
    atomic(RESULTS/"analysis.json",out)
    lines=["# Neutral-test synonym repair","",f"Primary setting: strength 1.0, conditional article policy temperature 1.0.","", "| Regime | N | Local efficacy | Predicted signed simple effect | Bootstrap 95% CI | Exact one-sided p |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for regime in ("between","within"):
        a=out["audits"][regime]["local_efficacy"];e=out["effects"][regime]["target_aligned"];iv=e["interval"];p=e["randomization"]["one_sided_positive_p"]
        lines.append(f"| {regime} | {a['n']} | {a['mean']:.3f} | {iv['mean']:.3f} | [{iv['lo']:.3f}, {iv['hi']:.3f}] | {p:.5f} |")
    lines += ["",f"Terminology decision: **{out['terminology_decision']}**.","", "The evaluated prompt contains neither synonym nor a first-letter instruction. Donors explicitly name the desired synonym; only their residual-state difference is added to the neutral evaluated state.",""]
    (RESULTS/"report.md").write_text("\n".join(lines));print(json.dumps(out,indent=2))

if __name__=="__main__":main()
