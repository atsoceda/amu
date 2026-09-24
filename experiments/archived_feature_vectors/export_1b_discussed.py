#!/usr/bin/env python3
"""Archive the 1B CLT vectors used in the paper, plus a frozen noun-only candidate pool.

Run before deleting the local Gemma Scope 2 1B cache. The noun-only pool is ranked
by future-noun direct attribution at the pre-article position alone (no article
term), from the saved 1B selection graphs, so a later noun-only screen can steer
these features without re-downloading the full CLT.
"""
from __future__ import annotations
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parents[2]; EXP=ROOT/"experiments"; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
SNAPSHOT="b738dc06961818c011fb2e44a316352ca0f4e873"
WEIGHTS=Path.home()/f".cache/huggingface/hub/models--google--gemma-scope-2-1b-pt/snapshots/{SNAPSHOT}/clt/width_262k_l0_medium_affine"
GRAPHS=EXP/"gemma_1b_sparse_scale/results/graphs"
NOUN_POOL_SIZE=256; NOUN_POOL_MIN_PROMPTS=3

def load(p): return json.loads(Path(p).read_text())
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

features={}
def add(layer,feat,role): features.setdefault((int(layer),int(feat)),set()).add(role)

for set_name,block in load(EXP/"gemma_1b_sparse_scale/results/selection.json")["sets"].items():
    for x in block["selected_features"]: add(x["layer"],x["feature_idx"],set_name)
for x in load(EXP/"gemma_1b_attribution_channel_calibration/results/selection.json")["features"]:
    add(x["layer"],x["feature_idx"],"calibration_"+x["stratum"])
for x in load(EXP/"gemma_1b_effect_matched/results/margin_ranked_features.json"):
    add(x["layer"],x["feature_idx"],"effect_matched_margin_ranked_pool")
for run in load(EXP/"gemma_1b_effect_matched/results/summary.json")["runs"]:
    for x in run["features"]: add(x["layer"],x["feature_idx"],"effect_matched_run_"+run["id"])
add(18,5015,"sparse_frontier_chosen")

# Noun-only pool: future-noun direct effect at the pre-article position, no article criterion.
from circuit_tracer.graph import Graph
from experiments.lib.aan_protocol import slugify, load_tokenizer, first_content_token_text
from experiments.lib.core import feature_effect_map, token_id_for_text
config=load(EXP/"gemma_1b_sparse_scale/config.json")
tokenizer=load_tokenizer(config)
stats={}
for sentence in config["selection_sentences"]:
    pid=slugify(sentence); meta=load(GRAPHS/f"{pid}__meta.json")
    content_text=meta.get("content_token_text") or first_content_token_text(tokenizer,meta["listed_word"])
    effects=feature_effect_map(Graph.from_pt(str(GRAPHS/f"{pid}__future.pt")),token_id_for_text(tokenizer,content_text))
    pre=len(tokenizer(meta["article_prompt"],add_special_tokens=True).input_ids)-1
    for (layer,pos,feat),e in effects.items():
        if pos!=pre: continue
        s=stats.setdefault((layer,feat),{"n":0,"future":0.0,"prompts":[]})
        s["n"]+=1; s["future"]+=float(e["direct_effect"]); s["prompts"].append(sentence)
pool=sorted(((k,v["future"]/v["n"],v["n"]) for k,v in stats.items() if v["n"]>=NOUN_POOL_MIN_PROMPTS),key=lambda t:-t[1])[:NOUN_POOL_SIZE]
noun_pool=[{"rank":i+1,"layer":k[0],"feature_idx":k[1],"mean_future_direct_effect":m,"prompt_count":n} for i,(k,m,n) in enumerate(pool)]
for x in noun_pool: add(x["layer"],x["feature_idx"],"noun_only_pool")

tensors={}; layer_tensors={}; records=[]; by_layer={}
for layer,feat in sorted(features): by_layer.setdefault(layer,[]).append(feat)
for layer,feats in sorted(by_layer.items()):
    with safe_open(str(WEIGHTS/f"params_layer_{layer}.safetensors"),framework="pt",device="cpu") as f:
        w_enc=f.get_tensor("w_enc"); w_dec=f.get_tensor("w_dec"); b_enc=f.get_tensor("b_enc"); thr=f.get_tensor("threshold")
        layer_tensors[f"L{layer:02d}__affine_skip_connection"]=f.get_tensor("affine_skip_connection").contiguous()
        layer_tensors[f"L{layer:02d}__b_dec"]=f.get_tensor("b_dec").contiguous()
    for feat in feats:
        stem=f"L{layer:02d}_F{feat:05d}"
        tensors[stem+"__encoder"]=w_enc[:,feat].contiguous()
        tensors[stem+"__decoder"]=w_dec[feat,layer:,:].contiguous()
        tensors[stem+"__b_enc"]=b_enc[feat].reshape(1).contiguous()
        tensors[stem+"__threshold"]=thr[feat].reshape(1).contiguous()
        records.append({"layer":layer,"feature_idx":feat,"roles":sorted(features[(layer,feat)]),
            "encoder_key":stem+"__encoder","decoder_key":stem+"__decoder","b_enc_key":stem+"__b_enc","threshold_key":stem+"__threshold",
            "decoder_shape":list(tensors[stem+"__decoder"].shape),"decoder_note":"output layers layer..25"})
    del w_enc,w_dec

vector_path=OUT/"gemma_scope_2_1b_pt_affine_discussed.safetensors"
save_file(tensors,str(vector_path),metadata={"source_snapshot":SNAPSHOT,"variant":"clt/width_262k_l0_medium_affine"})
layer_path=OUT/"gemma_scope_2_1b_pt_affine_layer_skips.local.safetensors"  # large; gitignored
save_file(layer_tensors,str(layer_path),metadata={"source_snapshot":SNAPSHOT,"variant":"clt/width_262k_l0_medium_affine"})
graph_records=[{"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(GRAPHS.glob("*")) if p.is_file()]
source_hashes={p.name:sha256(p) for p in sorted(WEIGHTS.glob("*")) if p.is_file()}
manifest={"generated_at":datetime.now(timezone.utc).isoformat(),"model":"google/gemma-3-1b-pt",
    "model_snapshot":"fcf18a2a879aab110ca39f8bffbccd5d49d8eb29","scope_repo":"google/gemma-scope-2-1b-pt",
    "scope_snapshot":SNAPSHOT,"variant":"clt/width_262k_l0_medium_affine","vector_file":vector_path.name,
    "vector_sha256":sha256(vector_path),"n_unique_features":len(records),"features":records,
    "noun_only_pool_rule":{"score":"mean future-noun direct effect at the pre-article position; no article term",
        "min_prompt_count":NOUN_POOL_MIN_PROMPTS,"size":NOUN_POOL_SIZE,"graphs":str(GRAPHS.relative_to(ROOT))},
    "noun_only_pool":noun_pool,"per_layer_tensor_file":layer_path.name,"per_layer_tensor_sha256":sha256(layer_path),
    "per_layer_tensors":"affine_skip_connection and b_dec for every archived layer (local only, gitignored)",
    "source_file_sha256":source_hashes,"selection_graphs_local_manifest":graph_records,
    "note":"Exact slices of the 1B CLT made before deleting the local cache. Not a complete CLT; covers every 1B feature used in the experiments plus the frozen noun-only pool."}
(OUT/"gemma_scope_2_1b_pt_affine_discussed.manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"vectors":str(vector_path),"bytes":vector_path.stat().st_size,"features":len(records),"noun_pool":len(noun_pool)},indent=2))
