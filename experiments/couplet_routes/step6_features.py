#!/usr/bin/env python3
"""Couplet routes, step 6a: Hanna & Ameisen's rhyme features at the line-1 anchor.

Reproduces the rule in couplets/rhyme_intervention_sample.py (_is_rhyme_feature):
a transcoder feature active at the anchor whose top-activating tokens are short
(all <= 4 characters), not dominated by one token (most common token <= 5 of the
top examples), and share either a vowel first character (>= 7) or a last
character (>= 7). Candidates: all features active at the anchor (the authors used
graph.active_features at that position). Transcoders are streamed with
skills/hf-range-streaming (never stored whole).

Stages (checkpointed under results/<model>/features/): capture, encode, classify,
decoders. Output: features/selection.json with, per couplet index, the rhyme
features and their activations at the anchor.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT / "skills/hf-range-streaming/scripts"))
from hf_stream import SafetensorsRemote, fetch_feature_records, load_feature_index  # noqa: E402
from step2_state_edit import HA, prompt_ids  # noqa: E402

REPO = {"Qwen3-1.7B": "mwhanna/qwen3-1.7b-transcoders-lowl0", "Qwen3-4B": "mwhanna/qwen3-4b-transcoders",
        "Qwen3-8B": "mwhanna/qwen3-8b-transcoders", "Qwen3-14B": "mwhanna/qwen3-14b-transcoders-lowl0"}
INDEX = {m: ROOT / "external/qwen3_transcoder_meta" / r.split("/")[1] / "features/index.json.gz" for m, r in REPO.items()}


def ensure_index(model: str) -> Path:
    """Download the transcoder feature index (about 20 MB) if it is not present."""
    path = INDEX[model]
    if not path.exists():
        from hf_stream import fetch_range, url
        import urllib.request
        u = url(REPO[model], "features/index.json.gz")
        with urllib.request.urlopen(urllib.request.Request(u, method="HEAD"), timeout=60) as r:
            size = int(r.headers["Content-Length"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch_range(u, 0, size - 1))
        print(f"downloaded {path} ({size/1e6:.1f} MB)", flush=True)
    return path


def is_rhyme_feature(card: dict | None) -> bool:
    """The authors' _is_rhyme_feature, copied (exclude=None, as in their call)."""
    if card is None:
        return False
    first_chars, last_chars, token_counts = Counter(), Counter(), Counter()
    for ex in card["examples_quantiles"][0]["examples"]:
        acts = ex["tokens_acts_list"]
        token = ex["tokens"][max(range(len(acts)), key=lambda i: acts[i])]
        token = re.sub(r'^[^\w]+|[^\w]+$', '', token.strip()).lower()
        if len(token) > 4:
            return False
        if not token:
            continue
        first_chars[token[0]] += 1
        last_chars[token[-1]] += 1
        token_counts[token] += 1
    if not first_chars:
        return False
    fc, fn = first_chars.most_common(1)[0]
    _, ln = last_chars.most_common(1)[0]
    _, tn = token_counts.most_common(1)[0]
    return (tn <= 5) and ((fc in 'aeiou' and fn >= 7) or ln >= 7)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="Qwen3-4B")
    ap.add_argument("stages", nargs="*", default=["capture", "encode", "classify", "decoders"])
    a = ap.parse_args()
    out = EXP / "results" / a.model / "features"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(HA / f"{a.model}.csv", index_col=0)
    idxs = sorted(df.index.tolist())

    if "capture" in a.stages and not (out / "anchor_mlp_in.pt").exists():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(f"Qwen/{a.model}")
        m = AutoModelForCausalLM.from_pretrained(f"Qwen/{a.model}", dtype=torch.bfloat16).to("mps").eval()
        store = {}
        pos = {"a": None}
        hooks = [m.model.layers[i].post_attention_layernorm.register_forward_hook(
            lambda mod, inp, o, i=i: store.__setitem__(i, o[0, pos["a"]].detach().cpu()))
            for i in range(len(m.model.layers))]
        xs = []
        for i in idxs:
            ids, anchor, _ = prompt_ids(tok, df.loc[i, "first_line"])
            pos["a"] = anchor
            with torch.no_grad():
                m(ids.to("mps"))
            xs.append(torch.stack([store[l] for l in range(len(m.model.layers))]))
        for h in hooks:
            h.remove()
        torch.save({"idx": idxs, "x": torch.stack(xs)}, out / "anchor_mlp_in.pt")
        print("captured", len(idxs), flush=True)

    if "encode" in a.stages:
        cap = torch.load(out / "anchor_mlp_in.pt")
        x = cap["x"]
        (out / "active").mkdir(exist_ok=True)
        for layer in range(x.shape[1]):
            f = out / "active" / f"layer_{layer:02d}.pt"
            if f.exists():
                continue
            t0 = time.time()
            st = SafetensorsRemote(REPO[a.model], f"layer_{layer}.safetensors", out / "headers")
            w, b = st.tensor("W_enc"), st.tensor("b_enc")
            acts = torch.relu(x[:, layer].float() @ w.float().T + b.float())
            nz = acts.nonzero()
            torch.save({"row": nz[:, 0], "feature": nz[:, 1], "act": acts[nz[:, 0], nz[:, 1]]}, f)
            print(f"encode layer {layer}: {len(nz)} active, {time.time() - t0:.0f}s", flush=True)
            del w, b, acts

    if "classify" in a.stages and not (out / "selection.json").exists():
        cap = torch.load(out / "anchor_mlp_in.pt")
        index = load_feature_index(ensure_index(a.model))
        per = defaultdict(list)
        by_layer = defaultdict(set)
        for f in sorted((out / "active").glob("layer_*.pt")):
            layer = int(f.stem.split("_")[1])
            d = torch.load(f)
            for r, fe, ac in zip(d["row"].tolist(), d["feature"].tolist(), d["act"].tolist()):
                per[cap["idx"][r]].append((layer, fe, ac))
                by_layer[layer].add(fe)
        rhyme = {}
        for layer, feats in sorted(by_layer.items()):
            cards = fetch_feature_records(REPO[a.model], index, layer, sorted(feats))
            rhyme.update({(layer, fe): is_rhyme_feature(c) for fe, c in cards.items()})
            print(f"classify layer {layer}: {sum(rhyme[(layer, fe)] for fe in feats)} rhyme / {len(feats)}", flush=True)
        sel = {str(i): [{"layer": l, "feature": fe, "act": ac} for l, fe, ac in per[i] if rhyme.get((l, fe))] for i in idxs}
        (out / "selection.json").write_text(json.dumps(sel, indent=1))
        counts = [len(v) for v in sel.values()]
        print(f"rhyme features per couplet: mean {sum(counts)/len(counts):.1f}; authors' feature_count mean "
              f"{df['feature_count'].mean():.1f}; exact matches {sum(int(df.loc[i,'feature_count'])==len(sel[str(i)]) for i in idxs)}/{len(idxs)}", flush=True)

    if "decoders" in a.stages and not (out / "decoders.pt").exists():
        sel = json.loads((out / "selection.json").read_text())
        by_layer = defaultdict(set)
        for v in sel.values():
            for n in v:
                by_layer[n["layer"]].add(n["feature"])
        dec = {}
        for layer, feats in sorted(by_layer.items()):
            st = SafetensorsRemote(REPO[a.model], f"layer_{layer}.safetensors", out / "headers")
            dec.update({(layer, fe): r for fe, r in st.rows("W_dec", sorted(feats)).items()})
        torch.save(dec, out / "decoders.pt")
        print("decoders", len(dec), flush=True)


if __name__ == "__main__":
    main()
