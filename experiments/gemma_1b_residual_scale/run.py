#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.lib.aan_protocol import token_id_for_text, write_json


EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def interval(values: list[float], seed: int, resamples: int) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = random.Random(seed); n = len(values)
    boot = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    boot.sort()
    return {"n": n, "mean": sum(values)/n,
            "lo": boot[math.floor(.025*(len(boot)-1))],
            "hi": boot[math.ceil(.975*(len(boot)-1))],
            "method": "pair/prompt-level nonparametric bootstrap", "resamples": resamples}


class ResidualModel:
    def __init__(self, path: str, dtype: torch.dtype):
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
        ).eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)

    def inputs(self, prompt: str) -> dict[str, torch.Tensor]:
        return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    @torch.inference_mode()
    def logits(self, prompt: str, patch: tuple[int, int, torch.Tensor] | None = None) -> torch.Tensor:
        hook = None
        if patch is not None:
            layer, position, replacement = patch
            def replace(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                changed = hidden.clone()
                changed[:, position, :] = replacement.to(changed.device, changed.dtype)
                return (changed, *output[1:]) if isinstance(output, tuple) else changed
            hook = self.layers[layer].register_forward_hook(replace)
        try:
            return self.model(**self.inputs(prompt), use_cache=False).logits[0, -1].detach().float().cpu()
        finally:
            if hook is not None: hook.remove()

    @torch.inference_mode()
    def states(self, prompt: str, position: int) -> list[torch.Tensor]:
        out = self.model(**self.inputs(prompt), use_cache=False, output_hidden_states=True)
        # hidden_states[0] is the embedding output; index l+1 is layer-l output.
        return [out.hidden_states[layer+1][0, position].detach().float().cpu() for layer in range(self.n_layers)]


def first_id(tokenizer, word: str) -> int:
    ids = tokenizer(f" {word}", add_special_tokens=False).input_ids
    if not ids: raise ValueError(word)
    return int(ids[0])


def tv(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(.5 * (torch.softmax(left, -1)-torch.softmax(right, -1)).abs().sum())


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = left.norm()*right.norm()
    return float(torch.dot(left,right)/denom) if float(denom) else 0.0


def stats(logits: torch.Tensor, source_id: int, target_id: int) -> dict[str, Any]:
    probs=torch.softmax(logits,-1); top=int(logits.argmax())
    return {"source_logit":float(logits[source_id]), "target_logit":float(logits[target_id]),
            "target_minus_source":float(logits[target_id]-logits[source_id]),
            "source_prob":float(probs[source_id]), "target_prob":float(probs[target_id]),
            "target_rank":int((logits>logits[target_id]).sum()+1), "target_top1":top==target_id}


def mixture(article_logits: torch.Tensor, branches: dict[str, torch.Tensor], ids: dict[str,int], tau: float) -> tuple[torch.Tensor,float]:
    mass = float(torch.softmax(article_logits/tau,-1)[[ids["a"],ids["an"]]].sum())
    weights = torch.softmax(article_logits[[ids["a"],ids["an"]]]/tau,-1)
    return weights[0]*torch.softmax(branches["a"],-1)+weights[1]*torch.softmax(branches["an"],-1), mass


def main() -> None:
    cfg=load(EXP_DIR/"config.json"); natural=load((EXP_DIR/cfg["natural_config_path"]).resolve())
    e1=load((EXP_DIR/cfg["e1_config_path"]).resolve()); RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    started=time.time(); dtype=getattr(torch,cfg["dtype"]); rm=ResidualModel(cfg["model_snapshot"],dtype)
    tok=rm.tokenizer; ids={a:token_id_for_text(tok,f" {a}") for a in ("a","an")}; demo=cfg["demonstration"]
    layers=list(range(rm.n_layers)); pairs=natural["pairs"]

    # Baseline article behavior and intervention-off article-prefix leverage.
    baseline_rows=[]
    for ex in e1["test_examples"]:
        prompt=f"{demo} {ex['sentence']}"; article_logits=rm.logits(prompt); article_id=int(article_logits.argmax())
        branches={a:rm.logits(prompt+tok.decode([ids[a]])) for a in ids}
        noun_id=int(rm.logits(prompt+tok.decode([article_id])).argmax())
        baseline_rows.append({"scope":"heldout20","sentence":ex["sentence"],"word":ex["listed_word"],
            "article":tok.decode([article_id]).strip(),"article_correct":article_id==ids[ex["expected_article"]],
            "noun":tok.decode([noun_id]).strip(),"noun_target_first_token":noun_id==first_id(tok,ex["listed_word"]),
            "prefix_tv":tv(branches["a"],branches["an"])})
        print(f"baseline heldout {len(baseline_rows)}/{len(e1['test_examples'])}", flush=True)
    dataset=(EXP_DIR/cfg["dataset_path"]).resolve()
    heldout_sentences={ex["sentence"] for ex in e1["test_examples"]}
    with dataset.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("article")!="an" or row["sentence"] in heldout_sentences: continue
            prompt=f"{demo} {row['sentence']}"; article_logits=rm.logits(prompt); article_id=int(article_logits.argmax())
            branches={a:rm.logits(prompt+tok.decode([ids[a]])) for a in ids}
            noun_id=int(rm.logits(prompt+tok.decode([article_id])).argmax())
            baseline_rows.append({"scope":"released_an","sentence":row["sentence"],"word":row["word"],
                "article":tok.decode([article_id]).strip(),"article_correct":article_id==ids["an"],
                "noun":tok.decode([noun_id]).strip(),"noun_target_first_token":noun_id==first_id(tok,row["word"]),
                "prefix_tv":tv(branches["a"],branches["an"])})
            if sum(r["scope"]=="released_an" for r in baseline_rows) % 10 == 0:
                print(f"baseline released-an {sum(r['scope']=='released_an' for r in baseline_rows)}", flush=True)

    cache={}
    for pair in pairs:
        sp=f"{demo} {pair['source_sentence']}"; tp=f"{demo} {pair['target_sentence']}"
        spos=len(tok(sp,add_special_tokens=True).input_ids)-1; tpos=len(tok(tp,add_special_tokens=True).input_ids)-1
        cache[pair["id"]]={"sp":sp,"tp":tp,"spos":spos,"tpos":tpos,"ss":rm.states(sp,spos),"ts":rm.states(tp,tpos),
                           "sid":first_id(tok,pair["source_word"]),"tid":first_id(tok,pair["target_word"])}
        print(f"captured states {len(cache)}/{len(pairs)}", flush=True)

    # All-depth, native-article target patches at natural strength.
    layer_rows=[]
    for pair in pairs:
        c=cache[pair["id"]]; article=pair["source_article"]; prefix=c["sp"]+tok.decode([ids[article]])
        base=rm.logits(prefix); bs=stats(base,c["sid"],c["tid"])
        for layer in layers:
            patched=rm.logits(prefix,(layer,c["spos"],c["ts"][layer])); ps=stats(patched,c["sid"],c["tid"])
            layer_rows.append({"pair_id":pair["id"],"split":pair["split"],"regime":pair["regime"],"layer":layer,
                "relative_depth":layer/(rm.n_layers-1),"delta_delta":ps["target_minus_source"]-bs["target_minus_source"],
                "target_logit_change":ps["target_logit"]-bs["target_logit"],"source_logit_change":ps["source_logit"]-bs["source_logit"],
                "fixed_tv":tv(patched,base),"target_rank_before":bs["target_rank"],"target_rank_after":ps["target_rank"],
                "target_top1_after":ps["target_top1"]})
        print(f"layer sweep {len({r['pair_id'] for r in layer_rows})}/{len(pairs)}", flush=True)
    dev_means={layer:sum(r["delta_delta"] for r in layer_rows if r["split"]=="dev" and r["layer"]==layer)/8 for layer in layers}
    selected=max(layers,key=dev_means.get)

    # Selected-layer strength map, controls, free/fixed split, and stochastic mixtures.
    rows=[]; test=[p for p in pairs if p["split"]=="test"]
    for pair_index,pair in enumerate(test):
        c=cache[pair["id"]]; delta=c["ts"][selected]-c["ss"][selected]
        wrong=cache[test[(pair_index+1)%len(test)]["id"]]["ts"][selected]
        off_article=rm.logits(c["sp"]); off_id=int(off_article.argmax())
        off_branches={a:rm.logits(c["sp"]+tok.decode([ids[a]])) for a in ids}
        off_noun=rm.logits(c["sp"]+tok.decode([off_id]))
        for strength in cfg["strengths"]:
            replacement=c["ss"][selected]+float(strength)*delta
            on_article=rm.logits(c["sp"],(selected,c["spos"],replacement)); on_id=int(on_article.argmax())
            on_branches={a:rm.logits(c["sp"]+tok.decode([ids[a]]),(selected,c["spos"],replacement)) for a in ids}
            on_noun=rm.logits(c["sp"]+tok.decode([on_id]),(selected,c["spos"],replacement))
            replay=rm.logits(c["sp"]+tok.decode([on_id])); total=torch.softmax(on_noun,-1)-torch.softmax(off_noun,-1)
            public=torch.softmax(replay,-1)-torch.softmax(off_noun,-1); private=torch.softmax(on_noun,-1)-torch.softmax(replay,-1)
            native=pair["source_article"]; bs=stats(off_branches[native],c["sid"],c["tid"]); ps=stats(on_branches[native],c["sid"],c["tid"])
            controls={}
            for name,rep in {"wrong_target":wrong,"sign_reversed":c["ss"][selected]-float(strength)*delta}.items():
                cs=stats(rm.logits(c["sp"]+tok.decode([ids[native]]),(selected,c["spos"],rep)),c["sid"],c["tid"])
                controls[name]=cs["target_minus_source"]-bs["target_minus_source"]
            random_values=[]
            for seed in cfg["random_seeds"]:
                g=torch.Generator().manual_seed(int(seed)); rd=torch.randn(delta.shape,generator=g); rd=rd/rd.norm()*delta.norm()*float(strength)
                rs=stats(rm.logits(c["sp"]+tok.decode([ids[native]]),(selected,c["spos"],c["ss"][selected]+rd)),c["sid"],c["tid"])
                random_values.append(rs["target_minus_source"]-bs["target_minus_source"])
            stochastic={}
            for tau in cfg["temperatures"]:
                off_mix,off_mass=mixture(off_article,off_branches,ids,float(tau)); on_mix,on_mass=mixture(on_article,on_branches,ids,float(tau))
                # Public counterfactual: treated policy weights with untreated branches.
                public_mix,_=mixture(on_article,off_branches,ids,float(tau))
                stochastic[str(tau)]={"total_tv":float(.5*(on_mix-off_mix).abs().sum()),
                    "public_tv":float(.5*(public_mix-off_mix).abs().sum()),
                    "private_tv":float(.5*(on_mix-public_mix).abs().sum()),"off_article_mass":off_mass,"on_article_mass":on_mass}
            rows.append({"pair_id":pair["id"],"regime":pair["regime"],"strength":strength,"layer":selected,
                "article_changed":off_id!=on_id,"total_tv":float(.5*total.abs().sum()),"public_tv":float(.5*public.abs().sum()),
                "private_tv":float(.5*private.abs().sum()),"public_total_cosine":cosine(public,total),"private_total_cosine":cosine(private,total),
                "target_delta_delta":ps["target_minus_source"]-bs["target_minus_source"],
                "target_logit_change":ps["target_logit"]-bs["target_logit"],"source_logit_change":ps["source_logit"]-bs["source_logit"],
                "target_rank_before":bs["target_rank"],"target_rank_after":ps["target_rank"],"target_top1_after":ps["target_top1"],
                "wrong_target_delta_delta":controls["wrong_target"],"sign_reversed_delta_delta":controls["sign_reversed"],
                "random_delta_delta":sum(random_values)/len(random_values),"stochastic":stochastic})
        print(f"selected-layer map {pair_index+1}/{len(test)}", flush=True)

    seed=cfg["bootstrap_seed"]; resamples=cfg["bootstrap_resamples"]
    summary={"experiment":cfg["experiment_name"],"generated_at":datetime.now(timezone.utc).isoformat(),"elapsed_sec":time.time()-started,
        "model":cfg["model"],"model_snapshot":cfg["model_snapshot"],"n_layers":rm.n_layers,"selected_layer":selected,
        "selected_relative_depth":selected/(rm.n_layers-1),"dev_layer_means":{str(k):v for k,v in dev_means.items()},"baseline":{},"conditions":{}}
    for scope in ("heldout20","released_an"):
        group=[r for r in baseline_rows if r["scope"]==scope]
        summary["baseline"][scope]={"n":len(group),"article_accuracy":sum(r["article_correct"] for r in group)/len(group),
            "noun_target_first_token_rate":sum(r["noun_target_first_token"] for r in group)/len(group),
            "prefix_tv":interval([r["prefix_tv"] for r in group],seed+len(group),resamples)}
    metrics=("total_tv","public_tv","private_tv","target_delta_delta","target_logit_change","source_logit_change","wrong_target_delta_delta","sign_reversed_delta_delta","random_delta_delta")
    for regime in ("between","within"):
        for strength in cfg["strengths"]:
            group=[r for r in rows if r["regime"]==regime and r["strength"]==strength]; key=f"{regime}_{strength}"
            summary["conditions"][key]={m:interval([float(r[m]) for r in group],seed+100+len(summary["conditions"])*20+i,resamples) for i,m in enumerate(metrics)}
            summary["conditions"][key].update({"n":len(group),"article_change_rate":sum(r["article_changed"] for r in group)/len(group),
                "target_top1_rate":sum(r["target_top1_after"] for r in group)/len(group)})
    write_json(RESULTS_DIR/"baseline_rows.json",baseline_rows); write_json(RESULTS_DIR/"layer_rows.json",layer_rows)
    write_json(RESULTS_DIR/"rows.json",rows); write_json(RESULTS_DIR/"summary.json",summary)
    lines=["# Gemma 3 1B residual-scale screen","",f"Selected layer: {selected}/{rm.n_layers-1} (relative depth {selected/(rm.n_layers-1):.3f})","",
           "| Regime / strength | Article change | Total TV | Public TV | Private TV | Target ΔΔ | Target top-1 |","| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key,b in summary["conditions"].items(): lines.append(f"| {key} | {b['article_change_rate']:.2f} | {b['total_tv']['mean']:.3f} | {b['public_tv']['mean']:.3f} | {b['private_tv']['mean']:.3f} | {b['target_delta_delta']['mean']:.3f} | {b['target_top1_rate']:.2f} |")
    (RESULTS_DIR/"report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__": main()
