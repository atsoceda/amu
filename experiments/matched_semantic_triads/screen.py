#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
from pathlib import Path

import torch

from experiments.gemma_1b_residual_scale.run import ResidualModel, first_id
from experiments.lib.aan_protocol import token_id_for_text, write_json


EXP=Path(__file__).resolve().parent; RESULTS=EXP/"results"


def generation(rm,prompt):
    logits=rm.logits(prompt);article_id=int(logits.argmax());piece=rm.tokenizer.decode([article_id]);noun=rm.logits(prompt+piece);noun_id=int(noun.argmax())
    return {"article":piece.strip(),"noun":rm.tokenizer.decode([noun_id]).strip().lower(),"article_id":article_id,"noun_id":noun_id}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="config.json");parser.add_argument("--output-suffix",default="")
    args=parser.parse_args();cfg=json.loads((EXP/args.config).read_text());RESULTS.mkdir(parents=True,exist_ok=True)
    rm=ResidualModel(cfg["model_snapshot"],getattr(torch,cfg["dtype"]));tok=rm.tokenizer
    article_ids={a:token_id_for_text(tok,f" {a}") for a in ("a","an")};rows=[]
    for index,triad in enumerate(cfg["triads"],1):
        prompts={role:cfg["prompt_template"].format(initial=triad[f"{role}_word"][0].upper(),definition=triad["definition"]) for role in ("source","within","cross")}
        generated={role:generation(rm,p) for role,p in prompts.items()}
        ids={role:first_id(tok,triad[f"{role}_word"]) for role in ("source","within","cross")}
        single={role:tok.decode([ids[role]]).strip().lower()==triad[f"{role}_word"] for role in ids}
        branch_logits={}
        for role in prompts:
            article=triad[f"{role}_article"]
            branch_logits[role]=rm.logits(prompts[role]+tok.decode([article_ids[article]]))
        margins={
            "source_over_within":float(branch_logits["source"][ids["source"]]-branch_logits["source"][ids["within"]]),
            "source_over_cross":float(branch_logits["source"][ids["source"]]-branch_logits["source"][ids["cross"]]),
            "within_over_source":float(branch_logits["within"][ids["within"]]-branch_logits["within"][ids["source"]]),
            "cross_over_source":float(branch_logits["cross"][ids["cross"]]-branch_logits["cross"][ids["source"]]),
        }
        admissible=all(single.values()) and all(generated[r]["article"]==triad[f"{r}_article"] for r in generated) and all(v>0 for v in margins.values())
        rows.append({**triad,"prompts":prompts,"generated":generated,"single_token":single,"margins":margins,"admissible":admissible})
        print(f"screened {index}/{len(cfg['triads'])}",flush=True)
    summary={"experiment":cfg["experiment_name"],"model":cfg["model"],"frozen_before_screening":cfg["frozen_before_screening"],
             "screening_rule":cfg["screening_rule"],"candidates":len(rows),"admissible":sum(r["admissible"] for r in rows),
             "admissible_ids":[r["id"] for r in rows if r["admissible"]]}
    suffix=f"_{args.output_suffix}" if args.output_suffix else ""
    write_json(RESULTS/f"screen_rows{suffix}.json",rows);write_json(RESULTS/f"screen_summary{suffix}.json",summary);print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
