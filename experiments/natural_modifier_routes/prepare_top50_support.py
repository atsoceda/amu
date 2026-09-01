#!/usr/bin/env python3
"""Build immutable semantic-audit requests from actual dev top-50 policies."""

import json,re
from pathlib import Path

EXP=Path(__file__).resolve().parent
PASSED=json.loads((EXP/"artifacts/09_semantic_pass/passed.json").read_text())
ROWS=json.loads((EXP/"artifacts/10_dev_policy_probe/rows.json").read_text())
OUTPUT=EXP/"artifacts/11_top50_support_requests";OUTPUT.mkdir(parents=True,exist_ok=True)
WORD=re.compile(r"^[A-Za-z]+$")
groups={}
for row in ROWS:groups.setdefault((row["candidate_id"],row["regime"]),[]).append(row)
requests=[]
for item in PASSED:
    cid=item["candidate_id"];regime=item["construction"]["regime"];policy=groups[(cid,regime)]
    tokens={}
    for row in policy:
        for token in row["top50"]:
            word=token["token"].strip()
            if WORD.fullmatch(word):tokens[int(token["token_id"])]=word
    requests.append({"candidate_id":cid,"regime":regime,"noun_0":item["family"]["noun_0"],"noun_1":item["family"]["noun_1"],
                     "shared_definition":item["family"]["shared_definition"],"neutral_stem":item["family"]["neutral_stem"],
                     "source_context_dev":item["construction"]["source_context_dev"],"target_context_dev":item["construction"]["target_context_dev"],
                     "candidate_tokens":[{"token_id":tid,"surface":tokens[tid]} for tid in sorted(tokens)]})
(OUTPUT/"requests.json").write_text(json.dumps(requests,indent=2)+"\n")
summary={"constructions":len(requests),"candidate_token_instances":sum(len(r["candidate_tokens"]) for r in requests),
         "mean_candidates_per_construction":sum(len(r["candidate_tokens"]) for r in requests)/len(requests),
         "criterion":"alphabetic actual top-50 tokens; semantic audit must confirm one-word attributive adjective valid before both nouns under all eight dev contexts"}
(OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))

