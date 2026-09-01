#!/usr/bin/env python3
"""Probe Gemma modifier policies on development paraphrases only."""

from __future__ import annotations

import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

EXP=Path(__file__).resolve().parent;CFG=json.loads((EXP/"config.json").read_text())
INPUT=EXP/"artifacts/09_semantic_pass/passed.json";OUTPUT=EXP/"artifacts/10_dev_policy_probe";ROWS=OUTPUT/"rows.json"

def atomic(path,value):
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2)+"\n");tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument("--batch-size",type=int,default=12);p.add_argument("--limit",type=int);args=p.parse_args()
    items=json.loads(INPUT.read_text());OUTPUT.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(CFG["model"]["snapshot"],local_files_only=True);tok.padding_side="left";tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(CFG["model"]["snapshot"],dtype=getattr(torch,CFG["model"]["dtype"]),local_files_only=True,low_cpu_mem_usage=True).eval()
    existing=json.loads(ROWS.read_text()) if ROWS.exists() else [];done={(r["candidate_id"],r["regime"],r["state"],r["paraphrase_index"]) for r in existing}
    jobs=[]
    for item in items:
        f,c=item["family"],item["construction"]
        for state in ("source","target"):
            for index,context in enumerate(c[f"{state}_context_dev"]):
                key=(item["candidate_id"],c["regime"],state,index)
                if key in done:continue
                prompt=f"{CFG['generic_instruction']}\n{context}\n{f['neutral_stem']} the"
                jobs.append((item,state,index,context,prompt))
    if args.limit is not None:jobs=jobs[:args.limit]
    rows=list(existing);started=time.time()
    for start in range(0,len(jobs),args.batch_size):
        batch=jobs[start:start+args.batch_size];inputs=tok([x[4] for x in batch],return_tensors="pt",padding=True,add_special_tokens=True)
        with torch.inference_mode():logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1].float().cpu();probs=torch.softmax(logits,-1)
        values,indices=torch.topk(probs,k=50,dim=-1)
        for i,(item,state,index,context,prompt) in enumerate(batch):
            f,c=item["family"],item["construction"];mods=f["candidate_modifiers"];mod_ids=[int(f["modifier_token_ids"][m][0]) for m in mods]
            rows.append({"candidate_id":item["candidate_id"],"regime":c["regime"],"state":state,"paraphrase_index":index,"context":context,"prompt":prompt,
                         "source_modifier":c["source_modifier"],"target_modifier":c["target_modifier"],
                         "candidate_modifier_mass":float(probs[i,mod_ids].sum()),
                         "source_modifier_probability":float(probs[i,int(f["modifier_token_ids"][c["source_modifier"]][0])]),
                         "target_modifier_probability":float(probs[i,int(f["modifier_token_ids"][c["target_modifier"]][0])]),
                         "top50":[{"token_id":int(tid),"token":tok.decode([int(tid)]).strip(),"probability":float(value)} for tid,value in zip(indices[i],values[i])]})
        if (start//args.batch_size+1)%10==0 or start+len(batch)==len(jobs):atomic(ROWS,rows);print(f"policy {start+len(batch)}/{len(jobs)} new; {len(rows)} total",flush=True)
    atomic(ROWS,rows)
    groups={}
    for r in rows:groups.setdefault((r["candidate_id"],r["regime"]),[]).append(r)
    summary=[]
    for (cid,regime),group in groups.items():
        src=[r for r in group if r["state"]=="source"];tgt=[r for r in group if r["state"]=="target"]
        if len(src)!=4 or len(tgt)!=4:continue
        mean=lambda rs,k:sum(r[k] for r in rs)/len(rs)
        summary.append({"candidate_id":cid,"regime":regime,"n_source":4,"n_target":4,
                        "source_candidate_mass":mean(src,"candidate_modifier_mass"),"target_candidate_mass":mean(tgt,"candidate_modifier_mass"),
                        "source_intended_probability":mean(src,"source_modifier_probability"),"source_target_probability":mean(src,"target_modifier_probability"),
                        "target_source_probability":mean(tgt,"source_modifier_probability"),"target_intended_probability":mean(tgt,"target_modifier_probability"),
                        "signed_target_policy_movement":mean(tgt,"target_modifier_probability")-mean(src,"target_modifier_probability")-(mean(tgt,"source_modifier_probability")-mean(src,"source_modifier_probability"))})
    atomic(OUTPUT/"construction_summary.json",summary);atomic(OUTPUT/"run_manifest.json",{"dev_only":True,"constructions_complete":len(summary),"policy_rows":len(rows),"elapsed_sec_this_run":time.time()-started,"top_tokens_retained_per_prompt":50})
    print(f"complete: {len(summary)} constructions")

if __name__=="__main__":main()

