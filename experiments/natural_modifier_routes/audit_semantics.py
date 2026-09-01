#!/usr/bin/env python3
"""Independent cached semantic audit of generated modifier/noun cross-products."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text())
INPUT = EXP / "artifacts/06_generated_contexts/all_contexts.json"
RAW = EXP / "artifacts/07_raw_semantic_audit"
OUTPUT = EXP / "artifacts/08_semantic_validity"


def extract_object(text):
    text=text.strip()
    if text.startswith("```"): text=text.split("\n",1)[1].rsplit("```",1)[0]
    return json.loads(text[text.find("{"):text.rfind("}")+1])


def prompt_for(item):
    f=item["family"]
    return f"""Act as a strict independent semantic auditor for a language-model experiment.

Nouns: {f['noun_0']} / {f['noun_1']}
Shared definition: {f['shared_definition']}
Fixed stem: {f['neutral_stem']} the
Generated constructions:
{json.dumps(item['constructions'], indent=2)}

For every construction and every source/target development/test context, audit all four combinations of source/target modifier × noun. A construction passes only if every combination is grammatical and semantically permissible under every context, the contexts alter pragmatic preference rather than objective truth, no context mentions or trivially names a candidate, and the stem plus completion is natural.

Return JSON only with keys candidate_id and constructions. constructions must be an array of three objects with: regime, pass, failed_context_indices (list), failure_reasons (list), and concise_notes. Be conservative. Do not repair stimuli."""


def call(prompt, seed, key):
    model=CFG["generation"]["model"]; url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload={"contents":[{"role":"user","parts":[{"text":prompt}]}],"generationConfig":{"temperature":0.0,"seed":seed,"responseMimeType":"application/json"}}
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=240) as response: raw=json.loads(response.read())
    return raw,raw["candidates"][0]["content"]["parts"][0]["text"]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--start",type=int,default=0);parser.add_argument("--stop",type=int,default=120);parser.add_argument("--max-retries",type=int,default=5);args=parser.parse_args()
    key=os.environ.get("GOOGLE_GEMINI_API_KEY");
    if not key: raise SystemExit("GOOGLE_GEMINI_API_KEY is not present")
    items=json.loads(INPUT.read_text());RAW.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
    for index in range(args.start,min(args.stop,len(items))):
        item=items[index];cid=item["family"]["candidate_id"];out=OUTPUT/f"{index:03d}_{cid}.json"
        if out.exists(): print(f"audit {index:03d}: cached",flush=True);continue
        prompt=prompt_for(item);seed=20261300+index
        for attempt in range(args.max_retries):
            try:
                raw,text=call(prompt,seed,key);audit=extract_object(text);audit["candidate_id"]=cid
                if len(audit.get("constructions",[]))!=3: raise ValueError("invalid audit schema")
                envelope={"audited_at":datetime.now(timezone.utc).isoformat(),"provider":"Google Gemini API","model":CFG["generation"]["model"],"seed":seed,"candidate_id":cid,"prompt":prompt,"response":raw}
                rt=RAW/f"{index:03d}_{cid}.json.tmp";ot=out.with_suffix(".json.tmp");rt.write_text(json.dumps(envelope,indent=2)+"\n");ot.write_text(json.dumps(audit,indent=2)+"\n");rt.replace(RAW/f"{index:03d}_{cid}.json");ot.replace(out)
                print(f"audit {index:03d}: complete",flush=True);break
            except (urllib.error.URLError,TimeoutError,ValueError,KeyError,json.JSONDecodeError) as exc:
                if attempt+1==args.max_retries: raise
                print(f"audit {index:03d}: retry {attempt+1} after {type(exc).__name__}",flush=True);time.sleep(min(60,2**attempt+random.random()))
    paths=sorted(p for p in OUTPUT.glob("*.json") if p.name not in {"all_audits.json","summary.json"});audits=[]
    for path in paths:
        audit=json.loads(path.read_text())
        # Early cached responses did not echo the ID; the immutable filename does.
        audit["candidate_id"]=path.stem.split("_",1)[1]
        audits.append(audit)
    (OUTPUT/"all_audits.json").write_text(json.dumps(audits,indent=2)+"\n")
    passed=sum(c.get("pass",False) for a in audits for c in a["constructions"]);total=sum(len(a["constructions"]) for a in audits)
    (OUTPUT/"summary.json").write_text(json.dumps({"families_audited":len(audits),"constructions_audited":total,"constructions_passed":passed},indent=2)+"\n")
    print(f"audited {len(audits)} families; {passed}/{total} constructions passed")


if __name__=="__main__":main()
