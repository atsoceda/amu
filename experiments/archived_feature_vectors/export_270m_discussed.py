#!/usr/bin/env python3
"""Archive the small subset of 270M CLT vectors discussed in the paper."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import torch

ROOT=Path(__file__).resolve().parents[2]; EXP=ROOT/"experiments"; OUT=Path(__file__).resolve().parent
WEIGHTS=Path("/Users/anthony/.cache/huggingface/hub/models--google--gemma-scope-2-270m-pt/snapshots/b218cd5d69dc2fa71cff448b68d625e6c9702d49/clt/width_262k_l0_medium_affine")

def load(p): return json.loads(Path(p).read_text())
def sha256(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

features={}
selection=load(EXP/"selection_criterion_ablation/results/selection.json")
for set_name,block in selection["sets"].items():
    for x in block["selected_features"]:
        features.setdefault((int(x["layer"]),int(x["feature_idx"])),set()).add(set_name)
cal=load(EXP/"attribution_channel_calibration/results/selection.json")
for x in cal["features"]: features.setdefault((int(x["layer"]),int(x["feature_idx"])),set()).add("calibration_"+x["stratum"])
features.setdefault((5,383),set()).add("S2_principal_L5_F383")
for x in load(EXP/"l5_fixed_token_matched_null/results/selected_controls.json"):
    features.setdefault((int(x["layer"]),int(x["feature_idx"])),set()).add("L5_matched_control")

tensors={}; records=[]
for layer,feat in sorted(features):
    path=WEIGHTS/f"params_layer_{layer}.safetensors"; stem=f"L{layer:02d}_F{feat:05d}"
    with safe_open(path,framework="pt",device="cpu") as f:
        encoder=f.get_tensor("w_enc")[:,feat].contiguous()
        decoder=f.get_tensor("w_dec")[feat,layer:,:].contiguous()
        b_enc=f.get_tensor("b_enc")[feat].reshape(1)
        threshold=f.get_tensor("threshold")[feat].reshape(1)
    tensors[stem+"__encoder"]=encoder; tensors[stem+"__decoder"]=decoder
    tensors[stem+"__b_enc"]=b_enc; tensors[stem+"__threshold"]=threshold
    records.append({"layer":layer,"feature_idx":feat,"roles":sorted(features[(layer,feat)]),
        "encoder_key":stem+"__encoder","decoder_key":stem+"__decoder","b_enc_key":stem+"__b_enc","threshold_key":stem+"__threshold",
        "encoder_shape":list(encoder.shape),"decoder_shape":list(decoder.shape)})

vector_path=OUT/"gemma_scope_2_270m_pt_affine_discussed.safetensors"
save_file(tensors,str(vector_path),metadata={"source_snapshot":"b218cd5d69dc2fa71cff448b68d625e6c9702d49","variant":"clt/width_262k_l0_medium_affine"})
graphs=EXP/"selection_criterion_ablation/results/graphs"
graph_records=[{"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(graphs.glob("*")) if p.is_file()]
manifest={"generated_at":datetime.now(timezone.utc).isoformat(),"model":"google/gemma-3-270m",
    "model_snapshot":"9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1","scope_repo":"google/gemma-scope-2-270m-pt",
    "scope_snapshot":"b218cd5d69dc2fa71cff448b68d625e6c9702d49","variant":"clt/width_262k_l0_medium_affine",
    "vector_file":vector_path.name,"vector_sha256":sha256(vector_path),"n_unique_features":len(records),"features":records,
    "selection_graphs_local_manifest":graph_records,
    "note":"Graph .pt files remain local and are gitignored; this manifest fixes their paths, sizes, and hashes before CLT cache deletion."}
(OUT/"gemma_scope_2_270m_pt_affine_discussed.manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"vectors":str(vector_path),"bytes":vector_path.stat().st_size,"features":len(records),"graphs":len(graph_records)},indent=2))
