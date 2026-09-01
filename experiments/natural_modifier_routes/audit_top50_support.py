#!/usr/bin/env python3
"""Gemini-audit actual top-50 tokens for valid finite modifier support."""

from __future__ import annotations
import argparse,json,os,random,time,urllib.error,urllib.request
from datetime import datetime,timezone
from pathlib import Path

EXP=Path(__file__).resolve().parent;CFG=json.loads((EXP/"config.json").read_text())
INPUT=EXP/"artifacts/11_top50_support_requests/requests.json";RAW=EXP/"artifacts/12_raw_top50_audit";OUTPUT=EXP/"artifacts/13_top50_support_audit"

def extract(text):
    text=text.strip()
    if text.startswith("```"):text=text.split("\n",1)[1].rsplit("```",1)[0]
    return json.loads(text[text.find("{"):text.rfind("}")+1])

def prompt_for(r):
    return f"""Strictly classify candidate next-token words for a finite-mediator causal experiment.

Nouns: {r['noun_0']} / {r['noun_1']}
Shared definition: {r['shared_definition']}
Completion form: [context] + \"{r['neutral_stem']} the\" + CANDIDATE + NOUN
Source development contexts: {json.dumps(r['source_context_dev'])}
Target development contexts: {json.dumps(r['target_context_dev'])}
Candidate token IDs and surfaces: {json.dumps(r['candidate_tokens'])}

A token is VALID only if its surface is a genuine one-word attributive adjective or adjectival modifier in this exact slot, naturally modifies BOTH nouns, and leaves BOTH noun completions semantically permissible under ALL eight contexts. Reject determiners, adverbs, punctuation, nouns used unnaturally as modifiers, sentence fragments, inflections that do not fit, and any word making a cross-cell false or contradictory. Be conservative and do not add tokens.

Return JSON only: {{"valid_token_ids":[integers],"rejected":[{{"token_id":integer,"reason":short_string}}]}}. Classify every supplied token exactly once."""

def call(prompt,seed,key):
    model=CFG["generation"]["model"];url=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload={"contents":[{"role":"user","parts":[{"text":prompt}]}],"generationConfig":{"temperature":0.0,"seed":seed,"responseMimeType":"application/json"}}
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=240) as response:raw=json.loads(response.read())
    return raw,raw["candidates"][0]["content"]["parts"][0]["text"]

def main():
    p=argparse.ArgumentParser();p.add_argument("--start",type=int,default=0);p.add_argument("--stop",type=int,default=100000);p.add_argument("--max-retries",type=int,default=5);a=p.parse_args()
    key=os.environ.get("GOOGLE_GEMINI_API_KEY");
    if not key:raise SystemExit("GOOGLE_GEMINI_API_KEY is not present")
    requests=json.loads(INPUT.read_text());RAW.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
    for i in range(a.start,min(a.stop,len(requests))):
        r=requests[i];stem=f"{i:03d}_{r['candidate_id']}__{r['regime']}";out=OUTPUT/f"{stem}.json"
        if out.exists():print(f"support {i:03d}: cached",flush=True);continue
        prompt=prompt_for(r);seed=20261500+i;expected={x["token_id"] for x in r["candidate_tokens"]}
        for attempt in range(a.max_retries):
            try:
                raw,text=call(prompt,seed,key);audit=extract(text);valid={int(x) for x in audit.get("valid_token_ids",[])};rejected={int(x["token_id"]) for x in audit.get("rejected",[])}
                if valid|rejected!=expected or valid&rejected:raise ValueError("audit did not classify every token exactly once")
                audit.update({"candidate_id":r["candidate_id"],"regime":r["regime"],"candidate_token_count":len(expected)})
                envelope={"audited_at":datetime.now(timezone.utc).isoformat(),"provider":"Google Gemini API","model":CFG["generation"]["model"],"seed":seed,"prompt":prompt,"response":raw}
                rt=RAW/f"{stem}.json.tmp";ot=out.with_suffix(".json.tmp");rt.write_text(json.dumps(envelope,indent=2)+"\n");ot.write_text(json.dumps(audit,indent=2)+"\n");rt.replace(RAW/f"{stem}.json");ot.replace(out);print(f"support {i:03d}: complete",flush=True);break
            except (urllib.error.URLError,TimeoutError,ValueError,KeyError,json.JSONDecodeError) as exc:
                if attempt+1==a.max_retries:raise
                print(f"support {i:03d}: retry {attempt+1} after {type(exc).__name__}",flush=True);time.sleep(min(60,2**attempt+random.random()))

if __name__=="__main__":main()

