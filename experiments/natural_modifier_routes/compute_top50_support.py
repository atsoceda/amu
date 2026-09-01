#!/usr/bin/env python3
"""Compute expanded valid-mediator mass and policy movement diagnostics."""

import json,math,random
from pathlib import Path

EXP=Path(__file__).resolve().parent
REQUESTS=json.loads((EXP/"artifacts/11_top50_support_requests/requests.json").read_text())
POLICY=json.loads((EXP/"artifacts/10_dev_policy_probe/rows.json").read_text())
AUDIT_DIR=EXP/"artifacts/13_top50_support_audit";OUTPUT=EXP/"artifacts/14_top50_support_results";OUTPUT.mkdir(parents=True,exist_ok=True)
audits={}
for path in AUDIT_DIR.glob("*.json"):
    a=json.loads(path.read_text());audits[(a["candidate_id"],a["regime"])]=a
groups={}
for row in POLICY:groups.setdefault((row["candidate_id"],row["regime"]),[]).append(row)
rows=[]
for r in REQUESTS:
    key=(r["candidate_id"],r["regime"])
    if key not in audits:continue
    audit=audits[key];valid=set(map(int,audit["valid_token_ids"]));prompt_rows=groups[key];per=[]
    for p in prompt_rows:
        top={int(x["token_id"]):float(x["probability"]) for x in p["top50"]}
        valid_mass=sum(prob for tid,prob in top.items() if tid in valid);top50_mass=sum(top.values())
        per.append({"state":p["state"],"paraphrase_index":p["paraphrase_index"],"valid_top50_mass":valid_mass,"top50_total_mass":top50_mass,"omitted_mass_upper":1-valid_mass})
    src=[x for x in per if x["state"]=="source"];tgt=[x for x in per if x["state"]=="target"]
    mean=lambda xs,k:sum(x[k] for x in xs)/len(xs)
    # TV on the audited finite support plus an explicit other/omitted bin.
    def vector(p):
        probs={int(x["token_id"]):float(x["probability"]) for x in p["top50"] if int(x["token_id"]) in valid};probs[-1]=1-sum(probs.values());return probs
    movements=[]
    for s,t in zip(sorted([p for p in prompt_rows if p["state"]=="source"],key=lambda x:x["paraphrase_index"]),sorted([p for p in prompt_rows if p["state"]=="target"],key=lambda x:x["paraphrase_index"])):
        sv,tv=vector(s),vector(t);movements.append(.5*sum(abs(tv.get(k,0)-sv.get(k,0)) for k in set(sv)|set(tv)))
    rows.append({"candidate_id":r["candidate_id"],"regime":r["regime"],"noun_0":r["noun_0"],"noun_1":r["noun_1"],"candidate_tokens":len(r["candidate_tokens"]),"valid_modifier_tokens":len(valid),
                 "source_valid_mass_mean":mean(src,"valid_top50_mass"),"target_valid_mass_mean":mean(tgt,"valid_top50_mass"),"minimum_prompt_valid_mass":min(x["valid_top50_mass"] for x in per),
                 "source_top50_total_mass_mean":mean(src,"top50_total_mass"),"target_top50_total_mass_mean":mean(tgt,"top50_total_mass"),
                 "paired_policy_tv_with_other_mean":sum(movements)/len(movements),"prompt_rows":per})
(OUTPUT/"construction_rows.json").write_text(json.dumps(rows,indent=2)+"\n")
thresholds={str(x):sum(min(r["source_valid_mass_mean"],r["target_valid_mass_mean"])>=x for r in rows) for x in (.5,.8,.9,.95)}
minimum_thresholds={str(x):sum(r["minimum_prompt_valid_mass"]>=x for r in rows) for x in (.5,.8,.9,.95)}
missing=[{"candidate_id":r["candidate_id"],"regime":r["regime"]} for r in REQUESTS if (r["candidate_id"],r["regime"]) not in audits]
summary={"planned_constructions":len(REQUESTS),"audited_constructions":len(rows),"missing_constructions":len(missing),"missing":missing,
         "coverage_by_both_state_mean":thresholds,"coverage_by_every_dev_prompt":minimum_thresholds,
         "coverage_bounds_over_all_planned":{threshold:{"lower":count,"upper":count+len(missing)} for threshold,count in thresholds.items()},
         "policy_tv_with_other":{"median":sorted(r["paired_policy_tv_with_other_mean"] for r in rows)[len(rows)//2],"above_0.05":sum(r["paired_policy_tv_with_other_mean"]>=.05 for r in rows),"above_0.1":sum(r["paired_policy_tv_with_other_mean"]>=.1 for r in rows)},
         "interpretation_guard":"The other bin makes TV descriptive of total captured-vs-omitted movement; route decomposition is valid only for constructions meeting the declared support threshold. No test outcomes used."}
(OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
