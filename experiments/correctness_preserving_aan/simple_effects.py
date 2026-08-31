#!/usr/bin/env python3
"""Compute the pre-specified simple effects underlying the route interaction."""
from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path

from experiments.lib.aan_protocol import write_json


EXP=Path(__file__).resolve().parent
RESULTS=EXP/"results"


def interval(values, seed, resamples=10000):
    rng=random.Random(seed);n=len(values)
    draws=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(resamples))
    return {"n":n,"mean":sum(values)/n,"lo":draws[math.floor(.025*(resamples-1))],"hi":draws[math.ceil(.975*(resamples-1))],
            "method":"semantic-family bootstrap","resamples":resamples}


def sign_flip(values):
    observed=sum(values)/len(values);null=[]
    for signs in itertools.product((-1,1),repeat=len(values)):
        null.append(sum(s*v for s,v in zip(signs,values))/len(values))
    return {"observed":observed,"one_sided_positive_p":sum(x>=observed-1e-15 for x in null)/len(null),
            "two_sided_p":sum(abs(x)>=abs(observed)-1e-15 for x in null)/len(null),"assignments":len(null),
            "method":"exact within-family sign-flip randomization"}


def main():
    rows=json.loads((RESULTS/"rows.json").read_text());out={"primary_setting":{"strength":1.0,"temperature":1.0},"effects":{}}
    for regime in ("between","within"):
        group=[r for r in rows if r["regime"]==regime and r["strength"]==1.0]
        tau="1.0"
        if regime=="between":
            tv=[r["stochastic"][tau]["public"]["tv"]-r["stochastic"][tau]["private"]["tv"] for r in group]
            aligned=[r["stochastic"][tau]["public"]["target_minus_source"]-r["stochastic"][tau]["private"]["target_minus_source"] for r in group]
            definitions={"tv":"public TV minus private TV","target_aligned":"public minus private signed target projection"}
        else:
            tv=[r["stochastic"][tau]["private"]["tv"]-r["stochastic"][tau]["public"]["tv"] for r in group]
            aligned=[r["stochastic"][tau]["private"]["target_minus_source"]-r["stochastic"][tau]["public"]["target_minus_source"] for r in group]
            definitions={"tv":"private TV minus public TV","target_aligned":"private minus public signed target projection"}
        out["effects"][regime]={"definitions":definitions,"tv":{"interval":interval(tv,20260902),"randomization":sign_flip(tv),"values":tv},
                                         "target_aligned":{"interval":interval(aligned,20260903),"randomization":sign_flip(aligned),"values":aligned}}
    within=out["effects"]["within"]["target_aligned"]
    out["terminology_decision"]="double dissociation" if within["interval"]["lo"]>0 and within["randomization"]["one_sided_positive_p"]<=.05 else "capacity-dependent route interaction"
    write_json(RESULTS/"simple_effects.json",out);print(json.dumps(out,indent=2))


if __name__=="__main__":main()
