#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from circuit_tracer.graph import Graph
from safetensors import safe_open

from experiments.lib.aan_protocol import slugify, write_json
from experiments.lib.core import load_replacement_model, setup_file_logging, token_id_for_text
from experiments.lib.mediation_estimands import total_variation_from_logits
from experiments.six_cell_family_sweep.run import activations_at_position, build_interventions, next_logits

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"


def load(path: Path):
    return json.loads(path.read_text())


def decoder_norms(config: dict, features: list[dict]) -> dict[tuple[int, int], float]:
    root = Path(config["transcoder_weight_snapshot"]) / "clt/width_262k_l0_medium_affine"
    by_layer: dict[int, list[int]] = {}
    for f in features:
        by_layer.setdefault(int(f["layer"]), []).append(int(f["feature_idx"]))
    out = {}
    for layer, ids in by_layer.items():
        with safe_open(root / f"params_layer_{layer}.safetensors", framework="pt", device="cpu") as sf:
            dec = sf.get_tensor("w_dec")
            for idx in ids:
                out[(layer, idx)] = float(torch.linalg.vector_norm(dec[idx].float()))
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS)
    started = time.time()
    config = load(EXP / "config.json")
    selection = load((EXP / config["selection_path"]).resolve())
    source_config = load((EXP / config["selection_config_path"]).resolve())
    ranked = selection["sets"]["S1_dual_effect"]["ranked_features"]
    max_k = max(config["k_values"])
    features = ranked[:max_k]
    norms = decoder_norms(config, features)
    model = load_replacement_model(config)
    tok = model.tokenizer
    a_id = token_id_for_text(tok, " a")
    an_id = token_id_for_text(tok, " an")
    rows = []
    activation_rows = []
    baseline_rows = []
    for split, examples in (("development", [{"sentence": s} for s in source_config["selection_sentences"]]),
                            ("held_out", source_config["test_examples"])):
        for idx, ex in enumerate(examples, 1):
            prompt = f"{config['demonstration']} {ex['sentence']}"
            pos = len(tok(prompt, add_special_tokens=True).input_ids) - 1
            acts = activations_at_position(model, prompt, pos)
            base = next_logits(model, prompt, [])
            forced = {art: next_logits(model, prompt + f" {art}", []) for art in ("a", "an")}
            article_logits = torch.stack([base[a_id], base[an_id]])
            temp = {}
            for t in config["temperatures"]:
                p = torch.softmax(article_logits / float(t), dim=0)
                temp[str(t)] = {"p_a_conditional": float(p[0]), "p_an_conditional": float(p[1])}
            baseline_rows.append({"split": split, "index": idx, "sentence": ex["sentence"],
                                  "margin_an_minus_a": float(base[an_id]-base[a_id]),
                                  "top_article": "an" if base[an_id] > base[a_id] else "a",
                                  "forced_a_an_noun_tv": total_variation_from_logits(forced["an"], forced["a"]),
                                  "article_policy": temp})
            for rank, f in enumerate(features, 1):
                layer, fid = int(f["layer"]), int(f["feature_idx"])
                activation_rows.append({"split": split, "index": idx, "sentence": ex["sentence"],
                                        "rank": rank, "layer": layer, "feature_idx": fid,
                                        "activation": float(acts[layer, fid].float().cpu()),
                                        "selection_mean_activation": float(f["mean_activation"]),
                                        "selection_score": float(f["mean_score"]),
                                        "decoder_norm": norms[(layer, fid)]})
            if split == "development":
                for k in config["k_values"]:
                    for gain in config["gain_values"]:
                        interventions, _ = build_interventions(acts, pos, features[:int(k)], float(gain))
                        out = next_logits(model, prompt, interventions)
                        rows.append({"index": idx, "sentence": ex["sentence"], "k": int(k), "gain": float(gain),
                                     "baseline_margin": float(base[an_id]-base[a_id]),
                                     "treated_margin": float(out[an_id]-out[a_id]),
                                     "margin_movement": float((out[an_id]-out[a_id])-(base[an_id]-base[a_id])),
                                     "crossed_to_an": bool(base[an_id] <= base[a_id] and out[an_id] > out[a_id])})
    sweep = []
    for k in config["k_values"]:
        for gain in config["gain_values"]:
            group = [r for r in rows if r["k"] == k and r["gain"] == gain]
            sweep.append({"k": k, "gain": gain,
                          "mean_margin_movement": sum(r["margin_movement"] for r in group)/len(group),
                          "crossed_to_an_rate": sum(r["crossed_to_an"] for r in group)/len(group)})
    eligible = [x for x in sweep if x["mean_margin_movement"] >= config["minimum_acceptable_margin_movement"]]
    if eligible:
        chosen = min(eligible, key=lambda x: (x["k"], x["gain"]))
    else:
        chosen = min(sweep, key=lambda x: abs(x["mean_margin_movement"]-config["target_mean_margin_movement"]))
    graph_rows = []
    graph_dir = (EXP / config["graph_dir"]).resolve()
    for p in sorted(graph_dir.glob("*.pt")):
        g = Graph.from_pt(str(p))
        graph_rows.append({"graph": p.name, "active_features": int(len(g.active_features)),
                           "selected_features": int(len(g.selected_features)),
                           "hit_cap": int(len(g.selected_features)) >= 1200,
                           "selected_fraction_of_active": len(g.selected_features)/len(g.active_features)})
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "elapsed_sec": time.time()-started,
               "target_270m_mean_margin_movement": config["target_mean_margin_movement"],
               "sweep": sweep, "chosen": chosen,
               "graph_coverage": {"n": len(graph_rows), "hit_cap_count": sum(x["hit_cap"] for x in graph_rows),
                                  "mean_active_features": sum(x["active_features"] for x in graph_rows)/len(graph_rows),
                                  "mean_selected_fraction": sum(x["selected_fraction_of_active"] for x in graph_rows)/len(graph_rows)}}
    write_json(RESULTS / "sweep_rows.json", rows)
    write_json(RESULTS / "activation_rows.json", activation_rows)
    write_json(RESULTS / "baseline_rows.json", baseline_rows)
    write_json(RESULTS / "graph_rows.json", graph_rows)
    write_json(RESULTS / "sweep_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
