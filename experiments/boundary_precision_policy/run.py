#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.lib.aan_protocol import token_id_for_text, write_json
from experiments.lib.core import load_replacement_model, setup_file_logging
from experiments.lib.precision_logits import dtype_audit, feature_intervention_precision
from experiments.six_cell_family_sweep.run import activations_at_position, build_interventions, next_logits

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def vector(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().float().cpu()
    while value.ndim > 1:
        value = value[-1]
    return value


def tv_vector(delta: torch.Tensor) -> float:
    return float(0.5 * delta.abs().sum())


def tv_prob(left: torch.Tensor, right: torch.Tensor) -> float:
    return tv_vector(torch.softmax(left, -1) - torch.softmax(right, -1))


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = left.norm() * right.norm()
    return float(torch.dot(left, right) / denom) if float(denom) else 0.0


def interval(values: list[float], seed: int, resamples: int) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(values)
    boot = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    boot.sort()
    return {"n": n, "mean": sum(values) / n, "lo": boot[math.floor(.025 * (len(boot)-1))], "hi": boot[math.ceil(.975 * (len(boot)-1))], "method": "prompt-level nonparametric bootstrap", "resamples": resamples}


def main() -> None:
    config = load(EXP_DIR / "config.json")
    e1 = load((EXP_DIR / config["e1_config_path"]).resolve())
    selection = load((EXP_DIR / config["e1_selection_path"]).resolve())
    old_rows = {int(r["index"]): r for r in load((EXP_DIR / config["boundary_rows_path"]).resolve()) if r["bracketed"]}
    old_summary = load((EXP_DIR / config["boundary_summary_path"]).resolve())
    features = selection["sets"]["S1_dual_effect"]["selected_features"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    native_cache: dict[tuple[str, float], torch.Tensor] = {}
    audit: dict[str, str] | None = None
    dense_rows: list[dict[str, Any]] = []
    precision_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []

    for index, example in enumerate(e1["test_examples"], start=1):
        if index not in old_rows:
            continue
        prior = old_rows[index]
        center = (float(prior["gain_low"]) + float(prior["gain_high"])) / 2
        prompt = f"{config['demonstration']} {example['sentence']}"
        position = len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1
        activations = activations_at_position(model, prompt, position)

        def intervention_dicts(gain: float):
            return build_interventions(activations, position, features, gain)[0]

        def intervention_tuples(gain: float):
            return [(x["layer"], x["pos"], x["feature_idx"], x["value"]) for x in intervention_dicts(gain)]

        def native(prefix: str, gain: float) -> torch.Tensor:
            key = (prefix, round(float(gain), 12))
            if key not in native_cache:
                native_cache[key] = next_logits(model, prefix, intervention_dicts(gain))
            return native_cache[key]

        policy_low = center - float(config["policy_half_width"])
        policy_high = center + float(config["policy_half_width"])
        article_logits = {"low": native(prompt, policy_low), "high": native(prompt, policy_high)}
        noun_logits = {side: {"a": native(prompt + tokenizer.decode([a_id]), gain), "an": native(prompt + tokenizer.decode([an_id]), gain)} for side, gain in (("low", policy_low), ("high", policy_high))}
        noun_probs = {side: {article: torch.softmax(logits, -1) for article, logits in values.items()} for side, values in noun_logits.items()}
        for temperature in config["temperatures"]:
            weights: dict[str, dict[str, float]] = {}
            for side in ("low", "high"):
                probs = torch.softmax(article_logits[side] / float(temperature), -1)
                mass = float(probs[a_id] + probs[an_id])
                weights[side] = {"a": float(probs[a_id] / mass), "an": float(probs[an_id] / mass), "a_an_mass": mass}
            mix_low = sum(weights["low"][b] * noun_probs["low"][b] for b in ("a", "an"))
            mix_high = sum(weights["high"][b] * noun_probs["high"][b] for b in ("a", "an"))
            policy = sum((weights["high"][b] - weights["low"][b]) * noun_probs["low"][b] for b in ("a", "an"))
            fixed = sum(weights["high"][b] * (noun_probs["high"][b] - noun_probs["low"][b]) for b in ("a", "an"))
            total = mix_high - mix_low
            policy_rows.append({"index": index, "temperature": float(temperature), "gain_low": policy_low, "gain_high": policy_high, "pi_an_low": weights["low"]["an"], "pi_an_high": weights["high"]["an"], "a_an_mass_low": weights["low"]["a_an_mass"], "a_an_mass_high": weights["high"]["a_an_mass"], "total_tv": tv_vector(total), "policy_tv": tv_vector(policy), "fixed_tv": tv_vector(fixed), "policy_total_cosine": cosine(policy, total), "fixed_total_cosine": cosine(fixed, total), "reconstruction_l1": float((total-policy-fixed).abs().sum())})

        if index not in set(config["dense_prompt_indices"]):
            continue
        count = int(config["dense_points"])
        half = float(config["dense_half_width"])
        for grid_index in range(count):
            offset = -half + 2 * half * grid_index / (count - 1)
            gain = center + offset
            payload = feature_intervention_precision(model, prompt, intervention_tuples(gain))
            if audit is None:
                audit = dtype_audit(model, payload)
                audit["recorded_feature_cache"] = str(activations.dtype)
            native_logits = vector(payload["native_logits"])
            fp32_logits = vector(payload["float32_logits"])
            dense_rows.append({"index": index, "grid_index": grid_index, "gain": gain, "offset": offset, "native_margin": float(native_logits[an_id]-native_logits[a_id]), "fp32_margin": float(fp32_logits[an_id]-fp32_logits[a_id])})

        endpoints: dict[str, dict[str, torch.Tensor]] = {"low": {}, "high": {}}
        endpoint_margins: dict[str, dict[str, float]] = {}
        for side, gain in (("low", float(prior["gain_low"])), ("high", float(prior["gain_high"]))):
            article_payload = feature_intervention_precision(model, prompt, intervention_tuples(gain))
            article_native = vector(article_payload["native_logits"])
            article_fp32 = vector(article_payload["float32_logits"])
            selected_id = an_id if article_fp32[an_id] > article_fp32[a_id] else a_id
            endpoint_margins[side] = {"native": float(article_native[an_id]-article_native[a_id]), "fp32": float(article_fp32[an_id]-article_fp32[a_id])}
            for article, token_id in (("a", a_id), ("an", an_id)):
                payload = feature_intervention_precision(model, prompt + tokenizer.decode([token_id]), intervention_tuples(gain))
                endpoints[side][article] = vector(payload["float32_logits"])
            endpoints[side]["free"] = endpoints[side]["an" if selected_id == an_id else "a"]
        precision_rows.append({"index": index, "sentence": example["sentence"], "gain_low": prior["gain_low"], "gain_high": prior["gain_high"], "native_margin_low": endpoint_margins["low"]["native"], "native_margin_high": endpoint_margins["high"]["native"], "fp32_margin_low": endpoint_margins["low"]["fp32"], "fp32_margin_high": endpoint_margins["high"]["fp32"], "free_tv_fp32": tv_prob(endpoints["high"]["free"], endpoints["low"]["free"]), "fixed_a_tv_fp32": tv_prob(endpoints["high"]["a"], endpoints["low"]["a"]), "fixed_an_tv_fp32": tv_prob(endpoints["high"]["an"], endpoints["low"]["an"])})

    seed = int(config["bootstrap_seed"])
    resamples = int(config["bootstrap_resamples"])
    summary: dict[str, Any] = {"experiment": config["experiment_name"], "generated_at": datetime.now(timezone.utc).isoformat(), "elapsed_sec": time.time()-started, "dtype_audit": audit, "native_boundary_reference": old_summary, "precision_audit": {"n_prompts": len(precision_rows), "free_tv_fp32": interval([r["free_tv_fp32"] for r in precision_rows], seed, resamples), "fixed_a_tv_fp32": interval([r["fixed_a_tv_fp32"] for r in precision_rows], seed+1, resamples), "fixed_an_tv_fp32": interval([r["fixed_an_tv_fp32"] for r in precision_rows], seed+2, resamples), "native_margin_low": interval([r["native_margin_low"] for r in precision_rows], seed+3, resamples), "native_margin_high": interval([r["native_margin_high"] for r in precision_rows], seed+4, resamples), "fp32_margin_low": interval([r["fp32_margin_low"] for r in precision_rows], seed+5, resamples), "fp32_margin_high": interval([r["fp32_margin_high"] for r in precision_rows], seed+6, resamples)}, "dense_grid": {"n_prompts": len(set(r["index"] for r in dense_rows)), "native_unique_margins": len(set(r["native_margin"] for r in dense_rows)), "fp32_unique_margins_1e6": len(set(round(r["fp32_margin"], 6) for r in dense_rows))}, "policy": {}, "scope": "Precision audit is pre-specified on the first three bracketed prompts; stochastic results condition the article policy on a/an and report retained mass."}
    for temp_index, temperature in enumerate(config["temperatures"]):
        group = [r for r in policy_rows if r["temperature"] == float(temperature)]
        summary["policy"][str(temperature)] = {key: interval([float(r[key]) for r in group], seed+100+10*temp_index+offset, resamples) for offset, key in enumerate(("total_tv", "policy_tv", "fixed_tv", "policy_total_cosine", "fixed_total_cosine", "a_an_mass_low", "a_an_mass_high", "reconstruction_l1"))}
    write_json(RESULTS_DIR / "precision_rows.json", precision_rows)
    write_json(RESULTS_DIR / "dense_grid.json", dense_rows)
    write_json(RESULTS_DIR / "policy_rows.json", policy_rows)
    write_json(RESULTS_DIR / "summary.json", summary)
    lines = ["# Boundary precision and stochastic-policy audit", "", f"Precision-audited prompts: {len(precision_rows)}; stochastic-policy prompts: {len(old_rows)}.", f"Native dense-grid unique margins: {summary['dense_grid']['native_unique_margins']}; float32-head unique margins: {summary['dense_grid']['fp32_unique_margins_1e6']}.", "", "| Temperature | Total TV | Policy TV | Fixed-token TV | Policy/total cosine | a/an mass low/high |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for temperature in config["temperatures"]:
        block = summary["policy"][str(temperature)]
        lines.append(f"| {temperature:g} | {block['total_tv']['mean']:.3f} | {block['policy_tv']['mean']:.3f} | {block['fixed_tv']['mean']:.3f} | {block['policy_total_cosine']['mean']:.3f} | {block['a_an_mass_low']['mean']:.3f}/{block['a_an_mass_high']['mean']:.3f} |")
    (RESULTS_DIR / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
