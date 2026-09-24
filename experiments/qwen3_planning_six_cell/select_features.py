#!/usr/bin/env python3
"""Reproduce Hanna & Ameisen's a/an planning-feature selection for Qwen3 without storing transcoders.

Their rule (a_an/planning_node_intervention.py, commit 993cff5): a planning node is
any transcoder feature active at the final (pre-article) position whose feature
card represents the planned word -- the word appears near the peak of more than
five top-activating examples, or in the feature's top/bottom logits. There is no
article criterion. Stages (each checkpointed under results/<model>/):

  capture   MLP inputs (post_attention_layernorm output) at the last position
  encode    stream each layer's encoder, record active features per prompt
  classify  fetch feature cards of active features, apply the word-feature test
  decoders  fetch decoder rows of selected features
  validate  compare per-prompt selected counts with the authors' released counts
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP))
from hf_stream import SafetensorsRemote, fetch_feature_records, load_feature_index  # noqa: E402

CONFIG = json.loads((EXP / "config.json").read_text())


def paths(model: str) -> dict[str, Path]:
    out = EXP / "results" / model
    out.mkdir(parents=True, exist_ok=True)
    return {"out": out, "mlp_in": out / "mlp_in_last.pt", "active": out / "active",
            "cards": out / "selected_cards.json", "selection": out / "selection.json",
            "decoders": out / "decoders.pt", "validation": out / "validation.json",
            "headers": out / "headers"}


def prompts(model: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / CONFIG["hanna_ameisen_repo"] / "a_an/results/interventions" / f"{model}.csv")


# ---- the authors' word-feature test, copied without behavioural change ----
def term_in_logits(term, top, bottom, use_bottom=True, substring_ok=True, k=10):
    logits = top[:k] + bottom[:k] if use_bottom else top[:k]
    logits = [re.sub(r'^[^\w]+|[^\w]+$', '', logit.strip()).lower() for logit in logits]
    term = term.strip().lower()
    if substring_ok:
        len1 = 0
        for logit in logits:
            if logit == '' or logit == 'a' or logit == 'an':
                continue
            if term.startswith(logit):
                if len(logit) == 1:
                    len1 += 1
                    if len1 >= 2:
                        return True
                else:
                    return True
        return False
    return any(term in logit for logit in logits)


def card_summary(feature_data):
    top_quantile = feature_data['examples_quantiles'][0]
    assert top_quantile['quantile_name'] == 'Top'
    tokens, top_indices = [], []
    for example in top_quantile['examples']:
        tokens.append(example['tokens'])
        acts = example['tokens_acts_list']
        top_indices.append(max(range(len(acts)), key=lambda i: acts[i]))
    return {'top_logits': feature_data['top_logits'], 'bottom_logits': feature_data['bottom_logits'],
            'tokens': tokens, 'top_indices': top_indices}


def is_word_feature(info, word):
    if info is None:
        return False
    word_counts = 0
    for tokens, top_index in zip(info['tokens'], info['top_indices']):
        top_segment = ''.join(tokens[top_index - 10: top_index + 10])
        if word in top_segment:
            word_counts += 1
    in_logits = term_in_logits(word, info['top_logits'], info['bottom_logits'])
    return (word_counts > 5) or in_logits
# ---------------------------------------------------------------------------


def stage_capture(model: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = paths(model)
    if p["mlp_in"].exists():
        return
    tok = AutoTokenizer.from_pretrained(f"Qwen/{model}")
    m = AutoModelForCausalLM.from_pretrained(f"Qwen/{model}", dtype=torch.bfloat16).to("mps").eval()
    layers = m.model.layers
    store: dict[int, torch.Tensor] = {}
    hooks = [layers[i].post_attention_layernorm.register_forward_hook(
        lambda mod, inp, out, i=i: store.__setitem__(i, out[0, -1].detach().to("cpu")))
        for i in range(len(layers))]
    rows = []
    for text in prompts(model)["prompt_before_article"]:
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
        with torch.no_grad():
            m(ids)
        rows.append(torch.stack([store[i] for i in range(len(layers))]))
    for h in hooks:
        h.remove()
    torch.save(torch.stack(rows), p["mlp_in"])  # [prompts, layers, d_model], bf16


def stage_encode(model: str) -> None:
    p = paths(model)
    repo = CONFIG["transcoders"][model]
    x = torch.load(p["mlp_in"])  # [N, L, d]
    p["active"].mkdir(exist_ok=True)
    for layer in range(x.shape[1]):
        out = p["active"] / f"layer_{layer:02d}.pt"
        if out.exists():
            continue
        t0 = time.time()
        st = SafetensorsRemote(repo, f"layer_{layer}.safetensors", p["headers"])
        w_enc, b_enc = st.tensor("W_enc"), st.tensor("b_enc")  # [F, d], [F]
        acts = torch.relu(x[:, layer].float() @ w_enc.float().T + b_enc.float())  # [N, F]
        nz = acts.nonzero()
        torch.save({"prompt": nz[:, 0], "feature": nz[:, 1], "act": acts[nz[:, 0], nz[:, 1]]}, out)
        print(f"{model} layer {layer}: {len(nz)} active, {time.time() - t0:.0f}s", flush=True)
        del w_enc, b_enc, acts


def load_active(model: str):
    p = paths(model)
    per_prompt = defaultdict(list)  # prompt -> [(layer, feature, act)]
    for f in sorted(p["active"].glob("layer_*.pt")):
        layer = int(f.stem.split("_")[1])
        d = torch.load(f)
        for pr, fe, a in zip(d["prompt"].tolist(), d["feature"].tolist(), d["act"].tolist()):
            per_prompt[pr].append((layer, fe, a))
    return per_prompt


def stage_classify(model: str) -> None:
    p = paths(model)
    if p["selection"].exists():
        return
    repo = CONFIG["transcoders"][model]
    index = load_feature_index(ROOT / CONFIG["feature_index"][model])
    per_prompt = load_active(model)
    by_layer = defaultdict(set)
    for items in per_prompt.values():
        for layer, fe, _ in items:
            by_layer[layer].add(fe)
    cache_dir = p["out"] / "cards"
    cache_dir.mkdir(exist_ok=True)
    summaries: dict[tuple[int, int], dict | None] = {}
    for layer in sorted(by_layer):
        cf = cache_dir / f"layer_{layer:02d}.json"
        if cf.exists():
            got = {int(k): v for k, v in json.loads(cf.read_text()).items()}
        else:
            t0 = time.time()
            raw = fetch_feature_records(repo, index, layer, sorted(by_layer[layer]))
            got = {f: (card_summary(r) if r is not None else None) for f, r in raw.items()}
            cf.write_text(json.dumps({str(k): v for k, v in got.items()}))
            print(f"{model} cards layer {layer}: {len(got)} features, {time.time() - t0:.0f}s", flush=True)
        for f, v in got.items():
            summaries[(layer, f)] = v
    df = prompts(model)
    selection = []
    for i, row in df.iterrows():
        nodes = [(l, f, a) for l, f, a in per_prompt.get(i, []) if is_word_feature(summaries[(l, f)], row["planned"])]
        selection.append({"prompt_index": int(i), "planned": row["planned"], "article": row["article"],
                          "n_active": len(per_prompt.get(i, [])),
                          "nodes": [{"layer": l, "feature": f, "activation": a} for l, f, a in nodes]})
    p["selection"].write_text(json.dumps(selection, indent=1))


def stage_decoders(model: str) -> None:
    p = paths(model)
    if p["decoders"].exists():
        return
    repo = CONFIG["transcoders"][model]
    selection = json.loads(p["selection"].read_text())
    by_layer = defaultdict(set)
    for s in selection:
        for n in s["nodes"]:
            by_layer[n["layer"]].add(n["feature"])
    out = {}
    for layer, feats in sorted(by_layer.items()):
        st = SafetensorsRemote(repo, f"layer_{layer}.safetensors", p["headers"])
        for f, row in st.rows("W_dec", sorted(feats)).items():
            out[(layer, f)] = row
    torch.save(out, p["decoders"])


def stage_validate(model: str) -> None:
    p = paths(model)
    selection = json.loads(p["selection"].read_text())
    df = prompts(model)
    ours = [len(s["nodes"]) for s in selection]
    theirs = df["selected_nodes_count"].fillna(0).astype(int).tolist()
    exact = sum(a == b for a, b in zip(ours, theirs))
    res = {"model": model, "n_prompts": len(ours), "exact_count_match": exact,
           "ours_with_nodes": sum(o > 0 for o in ours), "theirs_with_nodes": sum(t > 0 for t in theirs),
           "ours_total_nodes": sum(ours), "theirs_total_nodes": sum(theirs),
           "mismatches": [{"prompt_index": i, "ours": a, "theirs": b}
                          for i, (a, b) in enumerate(zip(ours, theirs)) if a != b][:50]}
    p["validation"].write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "mismatches"}, indent=1))


STAGES = {"capture": stage_capture, "encode": stage_encode, "classify": stage_classify,
          "decoders": stage_decoders, "validate": stage_validate}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(CONFIG["transcoders"]))
    ap.add_argument("stages", nargs="*", default=list(STAGES))
    a = ap.parse_args()
    for s in a.stages:
        print(f"== {a.model}: {s}", flush=True)
        STAGES[s](a.model)
