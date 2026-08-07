#!/usr/bin/env python3
"""Corrected causal edge tests: fixed-b with content clamps ON at c-step.

N0: protocol smoke (content-off vs content-on under fixed native article)
N1: within-class C→c under fixed b (primary)
N2: factorial C→B at article step (interpreted only if N1 finds a dial)
N3: selective B→b via S2
N4: latent plan vs executed-token readouts
N5: skipped (no pure A set / needs validated C dial)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.lib.aan_protocol import (
    article_and_word,
    build_amplify_interventions,
    build_zero_interventions,
    choose_control_features,
    first_content_token_text,
    vowel_initial,
    write_json,
)
from experiments.lib.core import (
    dict_intervention_result,
    generate_with_interventions,
    load_replacement_model,
    logits_for_prompt,
    setup_file_logging,
    token_id_for_text,
)

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def load_set_features(config: dict[str, Any], set_id: str) -> list[dict[str, Any]]:
    selection = json.loads((EXP_DIR / config["e1_selection_path"]).resolve().read_text())
    feats = selection["sets"][set_id]["selected_features"]
    if not feats:
        raise RuntimeError(f"Empty feature set {set_id}")
    return [
        {
            "layer": int(f["layer"]),
            "feature_idx": int(f["feature_idx"]),
            "mean_activation": float(f.get("mean_activation", 0.0)),
            "label": f.get("label", f"`L{f['layer']}/F{f['feature_idx']}`"),
        }
        for f in feats
    ]


def pre_article_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1


def word_token_id(tokenizer, word: str) -> int:
    return token_id_for_text(tokenizer, first_content_token_text(tokenizer, word))


def mean_set_activation(
    model,
    prompt: str,
    position: int,
    features: list[dict[str, Any]],
    interventions: list[dict[str, Any]] | None = None,
) -> float:
    if not features:
        return 0.0
    tuples = [
        (
            int(item["layer"]),
            int(item["pos"]),
            int(item["feature_idx"]),
            float(item["value"]),
        )
        for item in (interventions or [])
    ]
    _, activations = model.feature_intervention(
        prompt,
        interventions=tuples,
        freeze_attention=bool(tuples),
        sparse=True,
        return_activations=True,
    )
    if activations is None:
        return 0.0
    vals = [
        float(
            activations[int(f["layer"]), position, int(f["feature_idx"])]
            .detach()
            .float()
            .cpu()
        )
        for f in features
    ]
    return float(sum(vals) / len(vals))


def generate_force_native_then_c(
    model,
    prompt: str,
    *,
    native_article_id: int,
    c_step_interventions: list[dict[str, Any]],
    max_new_tokens: int,
) -> dict[str, Any]:
    """Paste native article, then generate noun with c-step clamps (may be empty)."""
    article_text = model.tokenizer.decode([native_article_id])
    current = prompt + article_text
    generated_ids = [native_article_id]
    tuples = [
        (
            int(item["layer"]),
            int(item["pos"]),
            int(item["feature_idx"]),
            float(item["value"]),
        )
        for item in c_step_interventions
    ]
    for _ in range(max(0, max_new_tokens - 1)):
        logits, _ = model.feature_intervention(
            current,
            interventions=tuples,
            freeze_attention=True,
            sparse=True,
            return_activations=False,
        )
        token_id = int(torch.argmax(logits[0, -1]).item())
        generated_ids.append(token_id)
        token_text = model.tokenizer.decode([token_id])
        current += token_text
        if token_text.strip() in {".", "!", "?"}:
            break
    return {
        "continuation": model.tokenizer.decode(generated_ids),
        "forced_article_id": native_article_id,
        "forced_article_text": article_text,
        "c_step_intervention_count": len(c_step_interventions),
    }


def select_contrast_features(
    model,
    prompt: str,
    position: int,
    *,
    target_word: str,
    config: dict[str, Any],
    forbidden: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Hint-prompt activation contrast toward target noun (graph-free fallback)."""
    hint_prompt = f"Think of a {target_word}. {prompt}"
    hint_pos = pre_article_pos(model.tokenizer, hint_prompt)
    layers = [int(x) for x in config.get("contrast_layers", [11, 12, 13, 14])]
    top_k = int(config.get("contrast_top_k", 4))

    _, base_acts = model.feature_intervention(
        prompt,
        interventions=[],
        freeze_attention=False,
        sparse=False,
        return_activations=True,
    )
    _, hint_acts = model.feature_intervention(
        hint_prompt,
        interventions=[],
        freeze_attention=False,
        sparse=False,
        return_activations=True,
    )
    candidates: list[dict[str, Any]] = []
    for layer in layers:
        base_vec = base_acts[layer, position].detach().float().cpu()
        hint_vec = hint_acts[layer, hint_pos].detach().float().cpu()
        delta = hint_vec - base_vec
        # Prefer features that rise under the target hint and are active.
        nonzero = ((hint_vec > 0) | (base_vec > 0)).nonzero(as_tuple=False).view(-1).tolist()
        for feature_idx in nonzero:
            key = (layer, int(feature_idx))
            if key in forbidden:
                continue
            score = float(delta[int(feature_idx)])
            if score <= 0:
                continue
            candidates.append(
                {
                    "layer": layer,
                    "feature_idx": int(feature_idx),
                    "mean_activation": float(base_vec[int(feature_idx)]),
                    "contrast_score": score,
                    "label": f"`L{layer}/F{feature_idx}`",
                }
            )
    candidates.sort(key=lambda x: (-x["contrast_score"], x["layer"], x["feature_idx"]))
    return candidates[:top_k]


def noun_logits(
    model,
    prompt_with_article: str,
    interventions: list[dict[str, Any]],
    token_ids: list[int],
) -> dict[int, float]:
    baseline = logits_for_prompt(
        model, prompt_with_article, token_ids, top_k=5, return_activations=False
    )
    if not interventions:
        return {
            tid: float(baseline["targets"][str(tid)]["logit"]) for tid in token_ids
        }
    intervened = dict_intervention_result(
        model,
        prompt_with_article,
        interventions,
        token_ids,
        baseline,
        filter_to_prompt_length=True,
    )
    return {tid: float(intervened["targets"][str(tid)]["logit"]) for tid in token_ids}


def legal_for_article(article: str, word: str) -> bool:
    if not word or article not in {"a", "an"}:
        return False
    return (article == "an") == vowel_initial(word)


def summarize_n1(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1

    def rate(key: str) -> float:
        return sum(1 for r in rows if r.get(key)) / n

    def mean(key: str) -> float:
        return sum(float(r.get(key, 0.0)) for r in rows) / n

    return {
        "n": len(rows),
        "content_changed_on_rate": rate("content_changed_on"),
        "within_class_on_rate": rate("within_class_on"),
        "c_to_c_signal_rate": rate("c_to_c_signal"),
        "matched_same_class_on_rate": rate("matched_same_class_on"),
        "content_changed_off_rate": rate("content_changed_off"),
        "c_to_c_off_rate": rate("c_to_c_off"),
        "mean_delta_same_minus_source_on": mean("delta_same_minus_source_on"),
        "mean_delta_same_minus_source_off": mean("delta_same_minus_source_off"),
        "mean_delta_an_minus_a_bstep": mean("delta_an_minus_a_bstep"),
        "protocol_differs_on_vs_off_rate": rate("protocol_differs_on_vs_off"),
    }


def evaluate_family_condition(
    model,
    tokenizer,
    family: dict[str, Any],
    config: dict[str, Any],
    *,
    condition_name: str,
    stage: str,
    features: list[dict[str, Any]] | None,
    op: str | None,
    c_step_features: list[dict[str, Any]] | None,
    readout_sets: dict[str, list[dict[str, Any]]],
    amplify_factor: float,
) -> dict[str, Any]:
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    article_ids = [a_id, an_id]

    prompt = f"{config['demonstration']} {family['sentence']}"
    position = pre_article_pos(tokenizer, prompt)
    source_word = family["source_word"]
    same_word = family["same_class_word"]
    cross_word = family["cross_class_word"]

    source_id = word_token_id(tokenizer, source_word)
    same_id = word_token_id(tokenizer, same_word)
    # aviator / multi-token: use first content token for logit scoring
    try:
        cross_id = word_token_id(tokenizer, cross_word)
    except ValueError:
        cross_id = tokenizer(
            first_content_token_text(tokenizer, cross_word),
            add_special_tokens=False,
        ).input_ids[0]
        cross_id = int(cross_id)

    baseline_continuation = generate_with_interventions(
        model, prompt, [], max_new_tokens=int(config["max_new_tokens"])
    )
    baseline_article, baseline_word = article_and_word(baseline_continuation)
    native_article = baseline_article if baseline_article in {"a", "an"} else family[
        "native_article"
    ]
    native_article_id = a_id if native_article == "a" else an_id

    b_interventions: list[dict[str, Any]] = []
    if features and op == "amplify":
        b_interventions, _ = build_amplify_interventions(
            model, prompt, position, features, amplify_factor
        )
    elif features and op == "zero":
        b_interventions = build_zero_interventions(features, position)

    # c-step: content features only, values from planning-time prompt
    c_interventions: list[dict[str, Any]] = []
    if c_step_features and op == "amplify":
        c_interventions, _ = build_amplify_interventions(
            model, prompt, position, c_step_features, amplify_factor
        )
    elif c_step_features and op == "zero":
        c_interventions = build_zero_interventions(c_step_features, position)

    # N2/N3: article logits at b-step
    base_article = logits_for_prompt(
        model, prompt, article_ids, top_k=5, return_activations=False
    )
    if b_interventions:
        int_article = dict_intervention_result(
            model, prompt, b_interventions, article_ids, base_article
        )
        delta_a = float(int_article["targets"][str(a_id)]["delta_logit"])
        delta_an = float(int_article["targets"][str(an_id)]["delta_logit"])
    else:
        delta_a = 0.0
        delta_an = 0.0

    prompt_plus_b = prompt + tokenizer.decode([native_article_id])
    noun_ids = [source_id, same_id, cross_id]
    logits_off = noun_logits(model, prompt_plus_b, [], noun_ids)
    logits_on = noun_logits(model, prompt_plus_b, c_interventions, noun_ids)

    force_off = generate_force_native_then_c(
        model,
        prompt,
        native_article_id=native_article_id,
        c_step_interventions=[],
        max_new_tokens=int(config["max_new_tokens"]),
    )
    force_on = generate_force_native_then_c(
        model,
        prompt,
        native_article_id=native_article_id,
        c_step_interventions=c_interventions,
        max_new_tokens=int(config["max_new_tokens"]),
    )
    free_continuation = generate_with_interventions(
        model,
        prompt,
        b_interventions,
        max_new_tokens=int(config["max_new_tokens"]),
    )

    off_article, off_word = article_and_word(force_off["continuation"])
    on_article, on_word = article_and_word(force_on["continuation"])
    free_article, free_word = article_and_word(free_continuation)

    content_changed_off = bool(baseline_word) and bool(off_word) and off_word != baseline_word
    content_changed_on = bool(baseline_word) and bool(on_word) and on_word != baseline_word
    within_class_off = legal_for_article(native_article, off_word)
    within_class_on = legal_for_article(native_article, on_word)
    c_to_c_off = content_changed_off and within_class_off
    c_to_c_on = content_changed_on and within_class_on
    matched_same_on = on_word == same_word.lower()
    matched_same_off = off_word == same_word.lower()
    protocol_differs = (on_word != off_word) or (
        abs(logits_on[same_id] - logits_off[same_id]) > 1e-4
    )

    act_s3_b = mean_set_activation(
        model, prompt, position, readout_sets["S3_content_only"], b_interventions or None
    )
    act_s2_b = mean_set_activation(
        model, prompt, position, readout_sets["S2_article_only"], b_interventions or None
    )
    act_s3_c = mean_set_activation(
        model,
        prompt_plus_b,
        position,
        readout_sets["S3_content_only"],
        c_interventions or None,
    )

    row = {
        "stage": stage,
        "condition": condition_name,
        "op": op or "none",
        "amplify_factor": amplify_factor,
        "family_id": family["id"],
        "sentence": family["sentence"],
        "native_article_expected": family["native_article"],
        "native_article_used": native_article,
        "source_word": source_word,
        "same_class_word": same_word,
        "cross_class_word": cross_word,
        "position": position,
        "baseline_continuation": baseline_continuation,
        "baseline_article": baseline_article,
        "baseline_word": baseline_word,
        "delta_a_bstep": delta_a,
        "delta_an_bstep": delta_an,
        "delta_an_minus_a_bstep": delta_an - delta_a,
        "force_off_continuation": force_off["continuation"],
        "force_on_continuation": force_on["continuation"],
        "free_continuation": free_continuation,
        "force_off_word": off_word,
        "force_on_word": on_word,
        "free_word": free_word,
        "force_off_article": off_article,
        "force_on_article": on_article,
        "free_article": free_article,
        "content_changed_off": content_changed_off,
        "content_changed_on": content_changed_on,
        "within_class_off": within_class_off,
        "within_class_on": within_class_on,
        "c_to_c_off": c_to_c_off,
        "c_to_c_signal": c_to_c_on,
        "matched_same_class_off": matched_same_off,
        "matched_same_class_on": matched_same_on,
        "protocol_differs_on_vs_off": protocol_differs,
        "logit_source_off": logits_off[source_id],
        "logit_same_off": logits_off[same_id],
        "logit_cross_off": logits_off[cross_id],
        "logit_source_on": logits_on[source_id],
        "logit_same_on": logits_on[same_id],
        "logit_cross_on": logits_on[cross_id],
        "delta_same_minus_source_off": logits_off[same_id] - logits_off[source_id],
        "delta_same_minus_source_on": logits_on[same_id] - logits_on[source_id],
        "delta_cross_minus_source_on": logits_on[cross_id] - logits_on[source_id],
        "act_S2_b": act_s2_b,
        "act_S3_b": act_s3_b,
        "act_S3_c": act_s3_c,
        "b_intervention_count": len(b_interventions),
        "c_intervention_count": len(c_interventions),
        "class_shifted_free": (
            bool(baseline_word)
            and bool(free_word)
            and vowel_initial(baseline_word) != vowel_initial(free_word)
        ),
    }
    logging.info(
        "%s %s %s on=%r off=%r free=%r c→c=%s Δsame-src=%.3f proto_diff=%s",
        stage,
        condition_name,
        family["id"],
        force_on["continuation"].strip(),
        force_off["continuation"].strip(),
        free_continuation.strip(),
        c_to_c_on,
        row["delta_same_minus_source_on"],
        protocol_differs,
    )
    return row


def interpret(summary: dict[str, Any]) -> str:
    n0 = summary.get("n0", {})
    n1 = summary.get("n1", {}).get("by_condition", {})
    parts = []

    if n0:
        parts.append(
            f"N0: content-on vs off differs on {n0.get('protocol_differs_on_vs_off_rate', 0):.0%} "
            f"of smoke rows (must be >0 for a valid C→c assay)."
        )

    best = None
    best_rate = -1.0
    for name, block in n1.items():
        rate = float(block.get("c_to_c_signal_rate", 0.0))
        if rate > best_rate:
            best_rate = rate
            best = (name, block)
    if best:
        name, block = best
        parts.append(
            f"N1 best `{name}`: c→c signal={block['c_to_c_signal_rate']:.2f}, "
            f"content_changed_on={block['content_changed_on_rate']:.2f}, "
            f"matched_same_class={block['matched_same_class_on_rate']:.2f}, "
            f"mean Δ(same−source) logit={block['mean_delta_same_minus_source_on']:.3f}."
        )
        off_rate = float(block.get("c_to_c_off_rate", 0.0))
        if block["c_to_c_signal_rate"] <= off_rate + 0.05 and block[
            "content_changed_on_rate"
        ] <= 0.1:
            parts.append(
                "N1 verdict: no independent within-class C→c dial with these handles "
                "(S3/contrast/controls). Supports packaged trajectories for this sparse set."
            )
            dial = False
        elif block["c_to_c_signal_rate"] >= 0.25:
            parts.append(
                "N1 verdict: within-class noun moves under fixed b with content clamps ON — "
                "possible modular C→c handle; N2 interpretable."
            )
            dial = True
        else:
            parts.append(
                "N1 verdict: weak/mixed C→c signal; treat any N2 article effects cautiously."
            )
            dial = False
    else:
        dial = False
        parts.append("N1: no condition rows.")

    n2 = summary.get("n2", {}).get("by_condition", {})
    if n2:
        for name, block in n2.items():
            parts.append(
                f"N2 `{name}`: mean Δ(an−a)={block.get('mean_delta_an_minus_a_bstep', 0):.3f}."
            )
        if not dial:
            parts.append("N2 not used as H1 evidence (no validated N1 dial).")

    n3 = summary.get("n3", {}).get("by_condition", {})
    if n3:
        for name, block in n3.items():
            parts.append(
                f"N3 `{name}`: mean Δ(an−a)={block.get('mean_delta_an_minus_a_bstep', 0):.3f}, "
                f"content_changed_on={block.get('content_changed_on_rate', 0):.2f}."
            )

    n4 = summary.get("n4", {})
    if n4:
        parts.append(
            f"N4: mean S3@b={n4.get('mean_act_S3_b', 0):.1f}, "
            f"S3@c(fixed b)={n4.get('mean_act_S3_c', 0):.1f}; "
            f"same-class logit gap on={n4.get('mean_delta_same_minus_source_on', 0):.3f}."
        )

    parts.append("N5 skipped (no pure A feature set; gated on N1 dial).")
    return " ".join(parts)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Causal edge independence",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        f"Runtime seconds: {summary['runtime_seconds']:.1f}",
        "",
        "## Protocol",
        "",
        "1. Measure article logits at pre-article position P (optional b-step).",
        "2. Paste native baseline article b.",
        "3. Keep content-feature clamps active at the same P while predicting noun c.",
        "4. Compare content-on vs content-off under identical fixed b.",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "## N1 condition table",
        "",
        "| Condition | c→c on | contentΔ on | match same | Δ(same−src) | proto differs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in summary.get("n1", {}).get("by_condition", {}).items():
        lines.append(
            f"| {name} | {block['c_to_c_signal_rate']:.2f} | "
            f"{block['content_changed_on_rate']:.2f} | "
            f"{block['matched_same_class_on_rate']:.2f} | "
            f"{block['mean_delta_same_minus_source_on']:.3f} | "
            f"{block['protocol_differs_on_vs_off_rate']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def run_condition_block(
    model,
    tokenizer,
    families: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    stage: str,
    condition_name: str,
    features: list[dict[str, Any]] | None,
    op: str | None,
    c_step_features: list[dict[str, Any]] | None,
    readout_sets: dict[str, list[dict[str, Any]]],
    amplify_factor: float,
) -> list[dict[str, Any]]:
    rows = []
    for family in families:
        # Per-family contrast features when requested
        feats = features
        c_feats = c_step_features
        if condition_name.startswith("contrast_") and op == "amplify":
            forbidden = {
                (int(f["layer"]), int(f["feature_idx"]))
                for feats_list in readout_sets.values()
                for f in feats_list
            }
            prompt = f"{config['demonstration']} {family['sentence']}"
            position = pre_article_pos(tokenizer, prompt)
            feats = select_contrast_features(
                model,
                prompt,
                position,
                target_word=family["same_class_word"],
                config=config,
                forbidden=forbidden,
            )
            c_feats = feats
            if not feats:
                logging.warning("No contrast features for %s", family["id"])
        rows.append(
            evaluate_family_condition(
                model,
                tokenizer,
                family,
                config,
                condition_name=condition_name,
                stage=stage,
                features=feats,
                op=op,
                c_step_features=c_feats,
                readout_sets=readout_sets,
                amplify_factor=amplify_factor,
            )
        )
    return rows


def main() -> None:
    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    logging.info("Loading model for causal_edge_independence")
    model = load_replacement_model(config)
    tokenizer = model.tokenizer

    readout_sets = {
        "S1_dual_effect": load_set_features(config, "S1_dual_effect"),
        "S2_article_only": load_set_features(config, "S2_article_only"),
        "S3_content_only": load_set_features(config, "S3_content_only"),
    }
    families = list(config["families"])
    factors = [float(x) for x in config.get("amplify_factors", [5.0])]
    primary_factor = factors[0]

    all_rows: list[dict[str, Any]] = []

    # --- N0 smoke on first 2 families ---
    logging.info("=== N0 protocol smoke ===")
    n0_rows = run_condition_block(
        model,
        tokenizer,
        families[:2],
        config,
        stage="N0",
        condition_name="S3_content_only_amplify",
        features=readout_sets["S3_content_only"],
        op="amplify",
        c_step_features=readout_sets["S3_content_only"],
        readout_sets=readout_sets,
        amplify_factor=primary_factor,
    )
    all_rows.extend(n0_rows)
    n0_summary = summarize_n1(n0_rows)

    # --- N1 primary within-class C→c ---
    logging.info("=== N1 within-class fixed-b C→c ===")
    n1_conditions: dict[str, Any] = {}
    n1_specs = [
        ("baseline", None, None, None),
        ("S3_content_only_amplify", readout_sets["S3_content_only"], "amplify", readout_sets["S3_content_only"]),
        ("S3_content_only_zero", readout_sets["S3_content_only"], "zero", readout_sets["S3_content_only"]),
        ("S2_article_only_amplify", readout_sets["S2_article_only"], "amplify", None),  # c-step empty
        ("control_amplify", None, "amplify", None),  # filled below
    ]

    # activation-matched controls from first family
    demo_prompt = f"{config['demonstration']} {families[0]['sentence']}"
    demo_pos = pre_article_pos(tokenizer, demo_prompt)
    control_feats = choose_control_features(
        model, demo_prompt, demo_pos, readout_sets["S3_content_only"], config
    )

    for name, feats, op, c_feats in n1_specs:
        if name == "control_amplify":
            feats = control_feats
            c_feats = control_feats
        for factor in factors if op == "amplify" else [primary_factor]:
            cond = name if op != "amplify" else f"{name}_x{factor:g}"
            if name == "baseline":
                cond = "baseline"
            rows = run_condition_block(
                model,
                tokenizer,
                families,
                config,
                stage="N1",
                condition_name=cond,
                features=feats,
                op=op,
                c_step_features=c_feats,
                readout_sets=readout_sets,
                amplify_factor=factor,
            )
            all_rows.extend(rows)
            n1_conditions[cond] = {
                "summary": summarize_n1(rows),
                "features": [
                    {"layer": f["layer"], "feature_idx": f["feature_idx"]}
                    for f in (feats or [])
                ],
            }
            if name == "baseline":
                break

    # contrast fallback at primary factor
    contrast_rows = run_condition_block(
        model,
        tokenizer,
        families,
        config,
        stage="N1",
        condition_name=f"contrast_amplify_x{primary_factor:g}",
        features=[],
        op="amplify",
        c_step_features=[],
        readout_sets=readout_sets,
        amplify_factor=primary_factor,
    )
    all_rows.extend(contrast_rows)
    n1_conditions[f"contrast_amplify_x{primary_factor:g}"] = {
        "summary": summarize_n1(contrast_rows),
        "features": "per_family_hint_contrast",
    }

    # Gate for N2
    best_c2c = max(
        (block["summary"]["c_to_c_signal_rate"] for block in n1_conditions.values()),
        default=0.0,
    )
    dial_found = best_c2c >= 0.25

    # --- N2 factorial C→B (always run; interpret only if dial) ---
    logging.info("=== N2 C→B at b-step (S3 / contrast) ===")
    n2_conditions: dict[str, Any] = {}
    for name in ("S3_content_only_amplify", "contrast_amplify"):
        feats = readout_sets["S3_content_only"] if name.startswith("S3") else []
        c_feats = feats
        rows = run_condition_block(
            model,
            tokenizer,
            families,
            config,
            stage="N2",
            condition_name=f"{name}_x{primary_factor:g}",
            features=feats,
            op="amplify",
            c_step_features=c_feats,
            readout_sets=readout_sets,
            amplify_factor=primary_factor,
        )
        all_rows.extend(rows)
        n2_conditions[rows[0]["condition"]] = {"summary": summarize_n1(rows)}

    # --- N3 selective B→b ---
    logging.info("=== N3 selective B→b (S2) ===")
    n3_conditions: dict[str, Any] = {}
    for op in ("amplify", "zero"):
        cond = f"S2_article_only_{op}" + (f"_x{primary_factor:g}" if op == "amplify" else "")
        rows = run_condition_block(
            model,
            tokenizer,
            families,
            config,
            stage="N3",
            condition_name=cond,
            features=readout_sets["S2_article_only"],
            op=op,
            c_step_features=None,
            readout_sets=readout_sets,
            amplify_factor=primary_factor,
        )
        all_rows.extend(rows)
        n3_conditions[cond] = {"summary": summarize_n1(rows)}

    # --- N4 latent readouts from N1 S3 amplify rows ---
    n1_s3 = [
        r
        for r in all_rows
        if r["stage"] == "N1" and r["condition"].startswith("S3_content_only_amplify")
    ]
    n4 = {
        "n": len(n1_s3),
        "mean_act_S3_b": (
            sum(r["act_S3_b"] for r in n1_s3) / len(n1_s3) if n1_s3 else 0.0
        ),
        "mean_act_S3_c": (
            sum(r["act_S3_c"] for r in n1_s3) / len(n1_s3) if n1_s3 else 0.0
        ),
        "mean_act_S2_b": (
            sum(r["act_S2_b"] for r in n1_s3) / len(n1_s3) if n1_s3 else 0.0
        ),
        "mean_delta_same_minus_source_on": (
            sum(r["delta_same_minus_source_on"] for r in n1_s3) / len(n1_s3)
            if n1_s3
            else 0.0
        ),
    }

    summary = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "amplify_factors": factors,
        "runtime_seconds": time.time() - started,
        "protocol": "force_native_b__content_clamps_on_at_c__full_recompute",
        "dial_found": dial_found,
        "n0": n0_summary,
        "n1": {
            "by_condition": {k: v["summary"] for k, v in n1_conditions.items()},
            "feature_sets": {k: v.get("features") for k, v in n1_conditions.items()},
        },
        "n2": {
            "by_condition": {k: v["summary"] for k, v in n2_conditions.items()},
            "interpreted_as_h1": dial_found,
        },
        "n3": {"by_condition": {k: v["summary"] for k, v in n3_conditions.items()}},
        "n4": n4,
        "n5": {"status": "skipped", "reason": "no_pure_A_set_and_gated_on_n1_dial"},
        "control_features": [
            {"layer": f["layer"], "feature_idx": f["feature_idx"]} for f in control_feats
        ],
    }
    summary["interpretation"] = interpret(summary)
    write_json(RESULTS_DIR / "summary.json", summary)
    write_json(RESULTS_DIR / "rows.json", all_rows)
    write_report(summary, RESULTS_DIR / "report.md")
    logging.info("Done. %s", summary["interpretation"])
    print(summary["interpretation"])


if __name__ == "__main__":
    main()
