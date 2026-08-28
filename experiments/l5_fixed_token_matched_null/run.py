#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.lib.aan_protocol import token_id_for_text, write_json
from experiments.lib.core import load_replacement_model, setup_file_logging
from experiments.lib.mediation_estimands import total_variation_from_logits
from experiments.six_cell_family_sweep.run import activations_at_position, build_interventions, next_logits, word_token_ids


EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def interval(values: list[float], seed: int, resamples: int) -> dict[str, Any]:
    rng = random.Random(seed); n = len(values)
    boot = [sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(resamples)]; boot.sort()
    return {"n": n, "mean": sum(values)/n, "lo": boot[math.floor(.025*(len(boot)-1))], "hi": boot[math.ceil(.975*(len(boot)-1))], "method": "prompt-level nonparametric bootstrap", "resamples": resamples}


def main() -> None:
    config = load(EXP_DIR / "config.json")
    e1 = load((EXP_DIR / config["e1_config_path"]).resolve())
    selection = load((EXP_DIR / config["e1_selection_path"]).resolve())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True); setup_file_logging(RESULTS_DIR)
    started = time.time(); model = load_replacement_model(config); tokenizer = model.tokenizer
    target = config["target_feature"]; layer = int(target["layer"])
    forbidden = {(int(f["layer"]), int(f["feature_idx"])) for block in selection["sets"].values() for f in block["selected_features"]}
    sums = None
    for sentence in e1["selection_sentences"]:
        prompt = f"{config['demonstration']} {sentence}"
        pos = len(tokenizer(prompt, add_special_tokens=True).input_ids)-1
        acts = activations_at_position(model, prompt, pos)[layer].detach().float().cpu()
        sums = acts.clone() if sums is None else sums + acts
    means = sums / len(e1["selection_sentences"])
    candidates = []
    for feature_idx in (means > 0).nonzero(as_tuple=False).view(-1).tolist():
        if (layer, int(feature_idx)) in forbidden: continue
        mean = float(means[int(feature_idx)])
        candidates.append({"layer": layer, "feature_idx": int(feature_idx), "mean_activation": mean, "activation_distance": abs(mean-float(target["mean_activation"]))})
    candidates.sort(key=lambda x: (x["activation_distance"], x["feature_idx"]))
    pool = candidates[:max(200, int(config["control_count"])*10)]
    rng = random.Random(int(config["control_seed"]))
    requested = int(config["control_count"])
    controls = pool if len(pool) <= requested else rng.sample(pool, requested)
    controls.sort(key=lambda x: x["feature_idx"])
    features = [{"id": "target_L5_F383", **target, "kind": "target"}] + [{"id": f"control_L5_F{x['feature_idx']}", **x, "kind": "control"} for x in controls]
    an_id = token_id_for_text(tokenizer, " an"); rows = []
    for index, example in enumerate(e1["test_examples"], start=1):
        prompt = f"{config['demonstration']} {example['sentence']}"; pos = len(tokenizer(prompt, add_special_tokens=True).input_ids)-1
        acts = activations_at_position(model, prompt, pos); prefix = prompt + tokenizer.decode([an_id]); baseline = next_logits(model, prefix, [])
        listed = word_token_ids(tokenizer, example.get("listed_word", "")); twin = word_token_ids(tokenizer, example.get("twin_word", ""))
        for feature in features:
            ints, activation_rows = build_interventions(acts, pos, [feature], float(config["amplify_factor"])); logits = next_logits(model, prefix, ints)
            twin_delta = None
            if listed and twin:
                twin_delta = float((logits[twin[0]]-logits[listed[0]]) - (baseline[twin[0]]-baseline[listed[0]]))
            rows.append({"index": index, "sentence": example["sentence"], "feature_id": feature["id"], "kind": feature["kind"], "feature": {"layer": layer, "feature_idx": int(feature["feature_idx"])}, "activation": activation_rows[0]["activation"], "fixed_an_tv": total_variation_from_logits(logits, baseline), "twin_delta_delta_an": twin_delta})
    by_feature = {}
    resamples = int(config["bootstrap_resamples"]); seed = int(config["bootstrap_seed"])
    for i, feature in enumerate(features):
        selected = [r for r in rows if r["feature_id"] == feature["id"]]; twins = [float(r["twin_delta_delta_an"]) for r in selected if r["twin_delta_delta_an"] is not None]
        by_feature[feature["id"]] = {"kind": feature["kind"], "feature": {"layer": layer, "feature_idx": int(feature["feature_idx"])}, "selection_mean_activation": float(feature["mean_activation"]), "tv": interval([float(r["fixed_an_tv"]) for r in selected], seed+i*10, resamples), "twin_delta_delta": interval(twins, seed+i*10+1, resamples), "twin_positive_rate": sum(x>0 for x in twins)/max(len(twins),1)}
    target_block = by_feature["target_L5_F383"]; control_blocks = [b for b in by_feature.values() if b["kind"] == "control"]
    target_tv = target_block["tv"]["mean"]; target_twin = target_block["twin_delta_delta"]["mean"]
    summary = {"experiment": config["experiment_name"], "generated_at": datetime.now(timezone.utc).isoformat(), "elapsed_sec": time.time()-started, "n_prompts": len(e1["test_examples"]), "requested_controls": requested, "n_controls": len(control_blocks), "matching": "same layer; nearest mean activation on eight selection prompts; frozen before held-out evaluation", "features": by_feature, "target": target_block, "control_tv_means": interval([b["tv"]["mean"] for b in control_blocks], seed+900, resamples), "control_twin_means": interval([b["twin_delta_delta"]["mean"] for b in control_blocks], seed+901, resamples), "empirical_p_tv": (1+sum(b["tv"]["mean"] >= target_tv for b in control_blocks))/(1+len(control_blocks)), "empirical_p_twin": (1+sum(b["twin_delta_delta"]["mean"] >= target_twin for b in control_blocks))/(1+len(control_blocks))}
    write_json(RESULTS_DIR/"rows.json", rows); write_json(RESULTS_DIR/"summary.json", summary); write_json(RESULTS_DIR/"selected_controls.json", controls)
    def fmt(b): return f"{b['mean']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"
    (RESULTS_DIR/"report.md").write_text(f"""# L5/F383 matched null

- Target fixed-`an` TV: {fmt(target_block['tv'])}.
- Across-feature matched-control mean TV: {fmt(summary['control_tv_means'])}; empirical p={summary['empirical_p_tv']:.3f}.
- Target twin ΔΔ: {fmt(target_block['twin_delta_delta'])}.
- Across-feature matched-control mean twin ΔΔ: {fmt(summary['control_twin_means'])}; empirical p={summary['empirical_p_twin']:.3f}.
- Target positive-twin rate: {target_block['twin_positive_rate']:.2f}.
""")


if __name__ == "__main__": main()
