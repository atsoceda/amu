#!/usr/bin/env python3
"""Six-cell public/private route assay on Hanna & Ameisen's Qwen3 a/an planning features.

For every prompt with at least one selected planning node, and for each condition
(zeroed, 5x multiplied, and an equal-size random control of active features at
the same position), we evaluate intervention off/on x {free, do(a), do(an)} and
record the next-token distribution at the first noun position. The intervention
edits the selected features' activations at the pre-article position and stays
in effect while the noun is predicted (full recomputation, no cache).

Estimands (Y(i,b) = noun distribution; B_i = greedy article, q_i = a/an policy):
  total   = Y(1,B1) - Y(0,B0)
  public  = Y(0,B1) - Y(0,B0)            (treated-article replay)
  private = Y(1,B1) - Y(0,B1)            (fixed treated article)
  private_baseline_article = Y(1,B0) - Y(0,B0)   (reverse order)
and the tau=1 policy-weighted versions over {a, an}. Also recorded: planned-word
first-token probability and log-odds in every cell, top-1 nouns, and a/an support.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
from select_features import MEDIATORS, SLOT, TASK, paths, prompts, load_active  # noqa: E402

SEED = 20260924


def tv(p, q):
    return 0.5 * (p - q).abs().sum().item()


def mixture(q_an, y_a, y_an):
    return (1 - q_an) * y_a + q_an * y_an


class Steer:
    """Adds sum_f (target_f - a_f) * W_dec[f] to layer MLP outputs at one position."""

    def __init__(self, model, deltas_by_layer, pos):
        self.handles = [model.model.layers[l].mlp.register_forward_hook(self._hook(vec, pos))
                        for l, vec in deltas_by_layer.items()]

    @staticmethod
    def _hook(vec, pos):
        def fn(mod, inp, out):
            out = out.clone()
            out[0, pos] += vec.to(out.dtype)
            return out
        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


def run(model_name: str, limit: int | None, out: str | None = None) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = paths(model_name)
    out_path = Path(out) if out else p["out"] / "six_cell_rows.jsonl"
    done = set()
    if out_path.exists():
        done = {(r["prompt_index"], r["condition"]) for r in map(json.loads, out_path.read_text().splitlines())}
    tok = AutoTokenizer.from_pretrained(f"Qwen/{model_name}")
    m = AutoModelForCausalLM.from_pretrained(f"Qwen/{model_name}", dtype=torch.bfloat16).to("mps").eval()
    # Slot "a"/"an" = the task's two mediator tokens (for el_la: el / la).
    a_id, an_id = (tok.encode(t, add_special_tokens=False)[0] for t in MEDIATORS)
    decoders = torch.load(p["decoders"])
    selection = json.loads(p["selection"].read_text())
    active = load_active(model_name)
    df = prompts(model_name)
    rng = random.Random(SEED)
    todo = [s for s in selection if s["nodes"]]
    if os.environ.get("AMU_SUBSET") == "first150":
        # The authors' el/la protocol intervened on the first 150 items of their table.
        keep = set(df.index[df["source_index"] < 150])
        todo = [s for s in todo if s["prompt_index"] in keep]
    todo = todo[: limit or None]

    def cells(steer):
        """Full recomputation of the prompt and both forced-article sequences.
        (A KV-cache shortcut was tried and rejected: in bf16 it shifted noun TV by
        up to 0.016 and planned log-prob by up to 0.12 on 0.6B, the same size as
        the effects measured.)"""
        res = {}
        for k, ids in seqs.items():
            h = Steer(m, steer, steer_pos) if steer else None
            try:
                with torch.no_grad():
                    res[k] = torch.softmax(m(ids).logits[0, -1].float(), -1).cpu()
            finally:
                if h:
                    h.remove()
        return res

    need_random_dec = {}
    with out_path.open("a") as fh:
        for s in todo:
            i = s["prompt_index"]
            row = df.iloc[i]
            base = tok(row["prompt_before_article"], return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
            steer_pos = base.shape[1] - 1
            planned_id = tok.encode(" " + row["planned"], add_special_tokens=False)[0]
            seqs = {"none": base}
            for art, tid in (("a", a_id), ("an", an_id)):
                seqs[art] = torch.cat([base, torch.tensor([[tid]], device="mps")], 1)
            nodes = [(n["layer"], n["feature"], n["activation"]) for n in s["nodes"]]
            pool = [x for x in active[i] if (x[0], x[1]) not in {(l, f) for l, f, _ in nodes}]
            rand_nodes = rng.sample(pool, min(len(nodes), len(pool)))
            conditions = {"zeroed": [(l, f, a, 0.0) for l, f, a in nodes],
                          "multiplied_5x": [(l, f, a, 5 * a) for l, f, a in nodes],
                          "random_zeroed": [(l, f, a, 0.0) for l, f, a in rand_nodes],
                          "random_multiplied_5x": [(l, f, a, 5 * a) for l, f, a in rand_nodes]}
            off = cells(None)
            for cond, spec in conditions.items():
                if (i, cond) in done or not spec:
                    continue
                missing = [(l, f) for l, f, _, _ in spec if (l, f) not in decoders]
                if missing:
                    from hf_stream import SafetensorsRemote
                    from select_features import CONFIG
                    for l in sorted({l for l, _ in missing}):
                        st = SafetensorsRemote(CONFIG["transcoders"][model_name], f"layer_{l}.safetensors", p["headers"])
                        decoders.update({(l, f): r for f, r in st.rows("W_dec", [f for ll, f in missing if ll == l]).items()})
                    need_random_dec.update({k: decoders[k] for k in missing})
                steer = {}
                for l, f, a, target in spec:
                    steer[l] = steer.get(l, 0) + (target - a) * decoders[(l, f)].float()
                steer = {l: v.to("mps") for l, v in steer.items()}
                on = cells(steer)
                rec = {"prompt_index": i, "condition": cond, "planned": row["planned"], "article": SLOT[row["article"]],
                       "task": TASK, "article_surface": row["article"],
                       "n_nodes": len(spec)}
                for tag, d in (("off", off), ("on", on)):
                    art = d["none"]
                    rec[f"{tag}_p_a"], rec[f"{tag}_p_an"] = art[a_id].item(), art[an_id].item()
                    rec[f"{tag}_greedy_is_article"] = int(art.argmax()) in (a_id, an_id)
                    rec[f"{tag}_greedy_article"] = "an" if art[an_id] > art[a_id] else "a"
                    for b in ("a", "an"):
                        y = d[b]
                        rec[f"{tag}_{b}_planned_p"] = y[planned_id].item()
                        rec[f"{tag}_{b}_top1"] = tok.decode([int(y.argmax())])
                b0, b1 = rec["off_greedy_article"], rec["on_greedy_article"]
                y = {(0, b): off[b] for b in ("a", "an")} | {(1, b): on[b] for b in ("a", "an")}
                rec["tv_total"] = tv(y[1, b1], y[0, b0])
                rec["tv_public"] = tv(y[0, b1], y[0, b0])
                rec["tv_private"] = tv(y[1, b1], y[0, b1])
                rec["tv_private_baseline_article"] = tv(y[1, b0], y[0, b0])
                for tag, d, key in (("off", off, 0), ("on", on, 1)):
                    pa, pan = d["none"][a_id].item(), d["none"][an_id].item()
                    rec[f"{tag}_q_an"] = pan / (pa + pan)
                q0, q1 = rec["off_q_an"], rec["on_q_an"]
                m0 = mixture(q0, y[0, "a"], y[0, "an"])
                m1 = mixture(q1, y[1, "a"], y[1, "an"])
                m01 = mixture(q1, y[0, "a"], y[0, "an"])
                rec["tau1_tv_total"], rec["tau1_tv_public"], rec["tau1_tv_private"] = tv(m1, m0), tv(m01, m0), tv(m1, m01)
                pl = lambda v: math.log(max(v[planned_id].item(), 1e-12))  # noqa: E731
                rec["tau1_planned_logp_total"] = pl(m1) - pl(m0)
                rec["tau1_planned_logp_public"] = pl(m01) - pl(m0)
                rec["tau1_planned_logp_private"] = pl(m1) - pl(m01)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            print(f"prompt {i} done", flush=True)
    if need_random_dec:
        torch.save(decoders, p["decoders"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    run(a.model, a.limit, a.out)
