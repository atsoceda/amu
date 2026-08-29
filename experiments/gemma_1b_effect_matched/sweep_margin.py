#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from circuit_tracer.graph import Graph

from experiments.lib.aan_protocol import first_content_token_text, slugify, write_json
from experiments.lib.core import feature_effect_map, load_replacement_model, setup_file_logging, token_id_for_text
from experiments.six_cell_family_sweep.run import activations_at_position, build_interventions, next_logits

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"


def load(path: Path):
    return json.loads(path.read_text())


def derive_ranked(config, source_config, tokenizer):
    graph_dir = (EXP / config["graph_dir"]).resolve()
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    stats = defaultdict(lambda: {"n": 0, "margin": 0.0, "future": 0.0, "activation": 0.0, "prompts": []})
    for sentence in source_config["selection_sentences"]:
        slug = slugify(sentence)
        meta = load(graph_dir / f"{slug}__meta.json")
        ag = Graph.from_pt(str(graph_dir / f"{slug}__article.pt"))
        fg = Graph.from_pt(str(graph_dir / f"{slug}__future.pt"))
        content = meta.get("content_token_text") or first_content_token_text(tokenizer, meta["listed_word"])
        content_id = token_id_for_text(tokenizer, content)
        pos = len(tokenizer(meta["article_prompt"], add_special_tokens=True).input_ids) - 1
        ae, ane, fe = feature_effect_map(ag, a_id), feature_effect_map(ag, an_id), feature_effect_map(fg, content_id)
        for key in set(ae) & set(ane) & set(fe):
            layer, p, fid = key
            if p != pos:
                continue
            margin = ane[key]["direct_effect"] - ae[key]["direct_effect"]
            future = fe[key]["direct_effect"]
            if margin <= 0 or future <= 0:
                continue
            s = stats[(layer, fid)]
            s["n"] += 1; s["margin"] += margin; s["future"] += future
            s["activation"] += ane[key]["activation"]; s["prompts"].append(sentence)
    ranked = []
    for (layer, fid), s in stats.items():
        n = s["n"]
        ranked.append({"layer": layer, "feature_idx": fid, "prompt_count": n,
                       "mean_margin_attribution": s["margin"]/n,
                       "mean_future_attribution": s["future"]/n,
                       "mean_activation": s["activation"]/n,
                       "mean_score": min(s["margin"]/n, s["future"]/n),
                       "prompts": s["prompts"], "label": f"`L{layer}/F{fid}`"})
    ranked.sort(key=lambda x: (x["prompt_count"], x["mean_score"], x["mean_margin_attribution"]), reverse=True)
    return ranked


def main():
    RESULTS.mkdir(parents=True, exist_ok=True); setup_file_logging(RESULTS)
    started = time.time(); config = load(EXP / "config.json")
    source = load((EXP / config["selection_config_path"]).resolve())
    model = load_replacement_model(config); tok = model.tokenizer
    ranked = derive_ranked(config, source, tok)
    write_json(RESULTS / "margin_ranked_features.json", ranked)
    features = [x for x in ranked if x["prompt_count"] >= 3]
    k_values = [k for k in config["k_values"] if k <= len(features)]
    if len(features) >= 24 and 24 not in k_values:
        k_values.append(24)
    k_values = sorted(k_values)
    a_id, an_id = token_id_for_text(tok, " a"), token_id_for_text(tok, " an")
    rows = []
    for idx, sentence in enumerate(source["selection_sentences"], 1):
        prompt = f"{config['demonstration']} {sentence}"
        pos = len(tok(prompt, add_special_tokens=True).input_ids)-1
        acts = activations_at_position(model, prompt, pos); base = next_logits(model, prompt, [])
        for k in k_values:
            for gain in config["gain_values"]:
                interventions, _ = build_interventions(acts, pos, features[:int(k)], float(gain))
                out = next_logits(model, prompt, interventions)
                rows.append({"index": idx, "sentence": sentence, "k": int(k), "gain": float(gain),
                             "baseline_margin": float(base[an_id]-base[a_id]),
                             "treated_margin": float(out[an_id]-out[a_id]),
                             "margin_movement": float((out[an_id]-out[a_id])-(base[an_id]-base[a_id])),
                             "crossed_to_an": bool(base[an_id] <= base[a_id] and out[an_id] > out[a_id])})
    sweep=[]
    for k in k_values:
        for gain in config["gain_values"]:
            g=[r for r in rows if r["k"]==k and r["gain"]==gain]
            sweep.append({"k":k,"gain":gain,"mean_margin_movement":sum(r["margin_movement"] for r in g)/len(g),
                          "crossed_to_an_rate":sum(r["crossed_to_an"] for r in g)/len(g)})
    eligible=[x for x in sweep if x["mean_margin_movement"]>=config["minimum_acceptable_margin_movement"]]
    chosen=min(eligible,key=lambda x:(x["k"],x["gain"])) if eligible else min(sweep,key=lambda x:abs(x["mean_margin_movement"]-config["target_mean_margin_movement"]))
    summary={"generated_at":datetime.now(timezone.utc).isoformat(),"elapsed_sec":time.time()-started,
             "selection_rule":"positive an-minus-a margin attribution and positive future attribution; score=min(means)",
             "candidate_count":len(ranked),"eligible_prompt_count_ge3":sum(x["prompt_count"]>=3 for x in ranked),
             "k_values": k_values,
             "target_270m_mean_margin_movement":config["target_mean_margin_movement"],"sweep":sweep,"chosen":chosen}
    write_json(RESULTS/"margin_sweep_rows.json",rows); write_json(RESULTS/"margin_sweep_summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
