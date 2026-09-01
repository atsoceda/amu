#!/usr/bin/env python3
"""Join generated stimuli to independent audits and freeze semantic passes."""

import json
from pathlib import Path

EXP=Path(__file__).resolve().parent
contexts=json.loads((EXP/"artifacts/06_generated_contexts/all_contexts.json").read_text())
audits=json.loads((EXP/"artifacts/08_semantic_validity/all_audits.json").read_text())
OUTPUT=EXP/"artifacts/09_semantic_pass";OUTPUT.mkdir(parents=True,exist_ok=True)
audit_map={(a["candidate_id"],c["regime"]):c for a in audits for c in a["constructions"]}
passed=[];rejected=[]
for item in contexts:
    cid=item["family"]["candidate_id"]
    for construction in item["constructions"]:
        audit=audit_map[(cid,construction["regime"])]
        row={"candidate_id":cid,"family":item["family"],"construction":construction,"independent_audit":audit}
        (passed if audit["pass"] else rejected).append(row)
(OUTPUT/"passed.json").write_text(json.dumps(passed,indent=2)+"\n")
(OUTPUT/"rejected.json").write_text(json.dumps(rejected,indent=2)+"\n")
(OUTPUT/"summary.json").write_text(json.dumps({"passed":len(passed),"rejected":len(rejected),"families_with_pass":len(set(r["candidate_id"] for r in passed))},indent=2)+"\n")
print((OUTPUT/"summary.json").read_text())

