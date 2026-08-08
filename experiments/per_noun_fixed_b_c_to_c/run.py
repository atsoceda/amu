#!/usr/bin/env python3
"""Experiment 2: per-noun attribution features under Stage XVI fixed-b protocol.

Select Latent-Planning-style features for same-class target nouns, then test
independent C→c with native article pasted and content clamps ON at c-step.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
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
    slugify,
    vowel_initial,
    write_json,
)
from experiments.lib.core import (
    dict_intervention_result,
    feature_effect_map,
    generate_with_interventions,
    load_replacement_model,
    logits_for_prompt,
    run_graph,
    setup_file_logging,
    token_id_for_text,
)

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-only",
        nargs=3,
        metavar=("FAMILY_ID", "GRAPH_NAME", "WORD"),
        help="Isolated attribution: family_id article|future word_or_-",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip graph build; evaluate from existing graphs/selection",
    )
    return parser.parse_args()


def pre_article_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1


def word_token_id(tokenizer, word: str) -> int:
    return token_id_for_text(tokenizer, first_content_token_text(tokenizer, word))


def legal_for_article(article: str, word: str) -> bool:
    if not word or article not in {"a", "an"}:
        return False
    return (article == "an") == vowel_initial(word)


def generate_force_native_then_c(
    model,
    prompt: str,
    *,
    native_article_id: int,
    c_step_interventions: list[dict[str, Any]],
    max_new_tokens: int,
) -> dict[str, Any]:
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
        "forced_article_text": article_text,
    }


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


def family_by_id(config: dict[str, Any], family_id: str) -> dict[str, Any]:
    for family in config["families"]:
        if family["id"] == family_id:
            return family
    raise KeyError(family_id)


def ensure_graphs(config: dict[str, Any]) -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    for family in config["families"]:
        fid = family["id"]
        jobs = [
            (fid, "article", "-"),
            (fid, "future", family["source_word"]),
            (fid, "future", family["same_class_word"]),
        ]
        for family_id, graph_name, word in jobs:
            if graph_name == "article":
                path = GRAPHS_DIR / f"{family_id}__article.pt"
            else:
                path = GRAPHS_DIR / f"{family_id}__future_{slugify(word)}.pt"
            if path.exists():
                logging.info("Reusing %s", path)
                continue
            logging.info(
                "Starting isolated graph %s/%s/%s", family_id, graph_name, word
            )
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--graph-only",
                    family_id,
                    graph_name,
                    word,
                ],
                check=True,
            )


def run_graph_phase(family_id: str, graph_name: str, word: str) -> None:
    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR)
    family = family_by_id(config, family_id)
    demo = config["demonstration"]
    article_prompt = f"{demo} {family['sentence']}"
    native = family["native_article"]
    future_prompt = f"{article_prompt} {native}"
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    if graph_name == "article":
        target_ids = [
            token_id_for_text(tokenizer, " a"),
            token_id_for_text(tokenizer, " an"),
        ]
        prompt = article_prompt
        prompt_id = f"{family_id}__article"
    else:
        content_text = first_content_token_text(tokenizer, word)
        target_ids = [token_id_for_text(tokenizer, content_text)]
        prompt = future_prompt
        prompt_id = f"{family_id}__future_{slugify(word)}"
    run_graph(model, prompt, prompt_id, target_ids, config, GRAPHS_DIR)
    meta_path = GRAPHS_DIR / f"{family_id}__meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(
        {
            "family_id": family_id,
            "sentence": family["sentence"],
            "native_article": native,
            "source_word": family["source_word"],
            "same_class_word": family["same_class_word"],
            "article_prompt": article_prompt,
            "future_prompt": future_prompt,
            "pre_article_pos": pre_article_pos(tokenizer, article_prompt),
        }
    )
    if graph_name == "future":
        futures = meta.get("future_targets", {})
        futures[word] = {
            "content_token_text": first_content_token_text(tokenizer, word),
            "graph": f"{family_id}__future_{slugify(word)}.pt",
        }
        meta["future_targets"] = futures
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    logging.info("Wrote graph %s", prompt_id)


def select_features_for_family(
    config: dict[str, Any],
    family: dict[str, Any],
    tokenizer,
) -> dict[str, list[dict[str, Any]]]:
    from circuit_tracer.graph import Graph

    fid = family["id"]
    meta = json.loads((GRAPHS_DIR / f"{fid}__meta.json").read_text())
    position = int(meta["pre_article_pos"])
    max_article = float(config["max_abs_article_direct_effect"])
    top_k = int(config["top_features_per_method"])

    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    source_word = family["source_word"]
    target_word = family["same_class_word"]
    source_id = word_token_id(tokenizer, source_word)
    target_id = word_token_id(tokenizer, target_word)

    article_graph = Graph.from_pt(str(GRAPHS_DIR / f"{fid}__article.pt"))
    source_graph = Graph.from_pt(
        str(GRAPHS_DIR / f"{fid}__future_{slugify(source_word)}.pt")
    )
    target_graph = Graph.from_pt(
        str(GRAPHS_DIR / f"{fid}__future_{slugify(target_word)}.pt")
    )

    an_effects = feature_effect_map(article_graph, an_id)
    a_effects = feature_effect_map(article_graph, a_id)
    source_effects = feature_effect_map(source_graph, source_id)
    target_effects = feature_effect_map(target_graph, target_id)

    def article_ok(key: tuple[int, int, int]) -> bool:
        # Missing from article graph ⇒ treated as ~0 article DE (LP-style).
        an_de = abs(float(an_effects.get(key, {}).get("direct_effect", 0.0)))
        a_de = abs(float(a_effects.get(key, {}).get("direct_effect", 0.0)))
        return an_de <= max_article and a_de <= max_article

    lp_candidates: list[dict[str, Any]] = []
    contrast_candidates: list[dict[str, Any]] = []
    for key, te in target_effects.items():
        layer, pos, feature_idx = key
        if pos != position:
            continue
        future_de = float(te["direct_effect"])
        if future_de <= 0:
            continue
        source_de = float(source_effects.get(key, {}).get("direct_effect", 0.0))
        an_de = float(an_effects.get(key, {}).get("direct_effect", 0.0))
        a_de = float(a_effects.get(key, {}).get("direct_effect", 0.0))
        activation = float(te["activation"])
        row = {
            "layer": layer,
            "feature_idx": feature_idx,
            "mean_activation": activation,
            "direct_effect_target": future_de,
            "direct_effect_source": source_de,
            "direct_effect_an": an_de,
            "direct_effect_a": a_de,
            "label": f"`L{layer}/F{feature_idx}`",
        }
        if article_ok(key):
            lp_candidates.append({**row, "score": future_de})
            contrast_candidates.append(
                {**row, "score": future_de - source_de}
            )

    # Fallback: relax article bound if too few LP candidates
    if len(lp_candidates) < top_k:
        logging.warning(
            "%s: only %d LP candidates with article bound; relaxing",
            fid,
            len(lp_candidates),
        )
        for key, te in target_effects.items():
            layer, pos, feature_idx = key
            if pos != position:
                continue
            future_de = float(te["direct_effect"])
            if future_de <= 0:
                continue
            source_de = float(source_effects.get(key, {}).get("direct_effect", 0.0))
            an_de = float(an_effects.get(key, {}).get("direct_effect", 0.0))
            a_de = float(a_effects.get(key, {}).get("direct_effect", 0.0))
            lp_candidates.append(
                {
                    "layer": layer,
                    "feature_idx": feature_idx,
                    "mean_activation": float(te["activation"]),
                    "direct_effect_target": future_de,
                    "direct_effect_source": source_de,
                    "direct_effect_an": an_de,
                    "direct_effect_a": a_de,
                    "score": future_de,
                    "label": f"`L{layer}/F{feature_idx}`",
                    "article_bound_relaxed": True,
                }
            )

    def top(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cands = sorted(
            cands,
            key=lambda r: (-float(r["score"]), r["layer"], r["feature_idx"]),
        )
        # Deduplicate by (layer, feature_idx)
        seen: set[tuple[int, int]] = set()
        out = []
        for row in cands:
            key = (int(row["layer"]), int(row["feature_idx"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
            if len(out) >= top_k:
                break
        return out

    return {
        "lp_target": top(lp_candidates),
        "contrast": top(contrast_candidates),
    }


def evaluate_row(
    model,
    tokenizer,
    family: dict[str, Any],
    config: dict[str, Any],
    *,
    condition_name: str,
    features: list[dict[str, Any]],
    op: str,
    amplify_factor: float,
) -> dict[str, Any]:
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    prompt = f"{config['demonstration']} {family['sentence']}"
    position = pre_article_pos(tokenizer, prompt)
    source_word = family["source_word"]
    same_word = family["same_class_word"]
    cross_word = family["cross_class_word"]
    source_id = word_token_id(tokenizer, source_word)
    same_id = word_token_id(tokenizer, same_word)
    try:
        cross_id = word_token_id(tokenizer, cross_word)
    except ValueError:
        cross_id = int(
            tokenizer(
                first_content_token_text(tokenizer, cross_word),
                add_special_tokens=False,
            ).input_ids[0]
        )

    baseline_continuation = generate_with_interventions(
        model, prompt, [], max_new_tokens=int(config["max_new_tokens"])
    )
    baseline_article, baseline_word = article_and_word(baseline_continuation)
    native_article = (
        baseline_article if baseline_article in {"a", "an"} else family["native_article"]
    )
    native_article_id = a_id if native_article == "a" else an_id

    interventions: list[dict[str, Any]] = []
    if features and op == "amplify":
        interventions, _ = build_amplify_interventions(
            model, prompt, position, features, amplify_factor
        )
    elif features and op == "zero":
        interventions = build_zero_interventions(features, position)

    base_article = logits_for_prompt(
        model, prompt, [a_id, an_id], top_k=5, return_activations=False
    )
    if interventions:
        int_article = dict_intervention_result(
            model, prompt, interventions, [a_id, an_id], base_article
        )
        delta_a = float(int_article["targets"][str(a_id)]["delta_logit"])
        delta_an = float(int_article["targets"][str(an_id)]["delta_logit"])
    else:
        delta_a = 0.0
        delta_an = 0.0

    prompt_plus_b = prompt + tokenizer.decode([native_article_id])
    noun_ids = [source_id, same_id, cross_id]
    logits_off = noun_logits(model, prompt_plus_b, [], noun_ids)
    logits_on = noun_logits(model, prompt_plus_b, interventions, noun_ids)

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
        c_step_interventions=interventions,
        max_new_tokens=int(config["max_new_tokens"]),
    )
    free_continuation = generate_with_interventions(
        model,
        prompt,
        interventions,
        max_new_tokens=int(config["max_new_tokens"]),
    )

    _, off_word = article_and_word(force_off["continuation"])
    _, on_word = article_and_word(force_on["continuation"])
    free_article, free_word = article_and_word(free_continuation)

    content_changed_on = bool(baseline_word) and bool(on_word) and on_word != baseline_word
    within_class_on = legal_for_article(native_article, on_word)
    c_to_c = content_changed_on and within_class_on
    matched_same = on_word == same_word.lower()

    row = {
        "condition": condition_name,
        "op": op,
        "amplify_factor": amplify_factor,
        "family_id": family["id"],
        "sentence": family["sentence"],
        "source_word": source_word,
        "same_class_word": same_word,
        "native_article_used": native_article,
        "n_features": len(features),
        "features": [
            {"layer": f["layer"], "feature_idx": f["feature_idx"]} for f in features
        ],
        "baseline_continuation": baseline_continuation,
        "baseline_word": baseline_word,
        "force_off_continuation": force_off["continuation"],
        "force_on_continuation": force_on["continuation"],
        "free_continuation": free_continuation,
        "force_off_word": off_word,
        "force_on_word": on_word,
        "free_word": free_word,
        "free_article": free_article,
        "content_changed_on": content_changed_on,
        "within_class_on": within_class_on,
        "c_to_c_signal": c_to_c,
        "matched_same_class_on": matched_same,
        "delta_an_minus_a_bstep": delta_an - delta_a,
        "delta_same_minus_source_off": logits_off[same_id] - logits_off[source_id],
        "delta_same_minus_source_on": logits_on[same_id] - logits_on[source_id],
        "delta_same_minus_source_delta": (logits_on[same_id] - logits_on[source_id])
        - (logits_off[same_id] - logits_off[source_id]),
        "logit_source_on": logits_on[source_id],
        "logit_same_on": logits_on[same_id],
        "protocol_differs_on_vs_off": on_word != off_word
        or abs(logits_on[same_id] - logits_off[same_id]) > 1e-4,
    }
    logging.info(
        "%s %s on=%r off=%r free=%r c→c=%s Δsame-src=%.3f matched=%s",
        condition_name,
        family["id"],
        force_on["continuation"].strip(),
        force_off["continuation"].strip(),
        free_continuation.strip(),
        c_to_c,
        row["delta_same_minus_source_on"],
        matched_same,
    )
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1

    def rate(key: str) -> float:
        return sum(1 for r in rows if r.get(key)) / n

    def mean(key: str) -> float:
        return sum(float(r.get(key, 0.0)) for r in rows) / n

    return {
        "n": len(rows),
        "c_to_c_signal_rate": rate("c_to_c_signal"),
        "content_changed_on_rate": rate("content_changed_on"),
        "matched_same_class_on_rate": rate("matched_same_class_on"),
        "mean_delta_same_minus_source_on": mean("delta_same_minus_source_on"),
        "mean_delta_same_minus_source_delta": mean("delta_same_minus_source_delta"),
        "mean_delta_an_minus_a_bstep": mean("delta_an_minus_a_bstep"),
        "protocol_differs_on_vs_off_rate": rate("protocol_differs_on_vs_off"),
    }


def interpret(summary: dict[str, Any]) -> str:
    by = summary.get("by_condition", {})
    parts = []
    best_name = None
    best_rate = -1.0
    for name, block in by.items():
        rate = float(block.get("c_to_c_signal_rate", 0.0))
        if rate > best_rate:
            best_rate = rate
            best_name = name
        parts.append(
            f"{name}: c→c={block['c_to_c_signal_rate']:.2f}, "
            f"match_same={block['matched_same_class_on_rate']:.2f}, "
            f"Δ(same−src)={block['mean_delta_same_minus_source_on']:.3f}, "
            f"ΔΔ(same−src)={block['mean_delta_same_minus_source_delta']:.3f}."
        )
    if best_rate >= 0.34:
        parts.append(
            f"Upset: `{best_name}` shows within-class C→c under fixed b — "
            "per-noun LP-style features can be a content dial; S1/S3 were the wrong object."
        )
        dial = True
    elif best_rate > 0:
        parts.append(
            f"Weak/mixed signal on `{best_name}`; inspect rows before claiming a dial."
        )
        dial = False
    else:
        parts.append(
            "Null: per-noun LP-style and contrast features do not move within-class "
            "nouns under fixed b. Fairer negative than recurring E1 S3 alone."
        )
        dial = False
    summary["dial_found"] = dial
    return " ".join(parts)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Per-noun fixed-b C→c (Experiment 2)",
        "",
        f"Generated: {summary['generated_at']}",
        f"Runtime seconds: {summary['runtime_seconds']:.1f}",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "## Condition table",
        "",
        "| Condition | c→c | contentΔ | match same | Δ(same−src) | ΔΔ(same−src) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in summary.get("by_condition", {}).items():
        lines.append(
            f"| {name} | {block['c_to_c_signal_rate']:.2f} | "
            f"{block['content_changed_on_rate']:.2f} | "
            f"{block['matched_same_class_on_rate']:.2f} | "
            f"{block['mean_delta_same_minus_source_on']:.3f} | "
            f"{block['mean_delta_same_minus_source_delta']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def run_eval(config: dict[str, Any]) -> None:
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    logging.info("Loading model for per-noun evaluation")
    model = load_replacement_model(config)
    tokenizer = model.tokenizer

    selection: dict[str, Any] = {"families": {}}
    all_rows: list[dict[str, Any]] = []
    by_condition: dict[str, Any] = {}

    # Build selection
    for family in config["families"]:
        feats = select_features_for_family(config, family, tokenizer)
        selection["families"][family["id"]] = {
            "source_word": family["source_word"],
            "same_class_word": family["same_class_word"],
            "methods": {
                name: [
                    {
                        "layer": f["layer"],
                        "feature_idx": f["feature_idx"],
                        "mean_activation": f.get("mean_activation", 0.0),
                        "score": f.get("score"),
                        "direct_effect_target": f.get("direct_effect_target"),
                        "direct_effect_source": f.get("direct_effect_source"),
                        "direct_effect_an": f.get("direct_effect_an"),
                        "direct_effect_a": f.get("direct_effect_a"),
                        "article_bound_relaxed": f.get("article_bound_relaxed", False),
                    }
                    for f in feat_list
                ]
                for name, feat_list in feats.items()
            },
        }
        logging.info(
            "Selected %s: lp=%d contrast=%d",
            family["id"],
            len(feats["lp_target"]),
            len(feats["contrast"]),
        )

    write_json(RESULTS_DIR / "selection.json", selection)

    # Controls from first family / lp features as forbidden
    first = config["families"][0]
    demo_prompt = f"{config['demonstration']} {first['sentence']}"
    demo_pos = pre_article_pos(tokenizer, demo_prompt)
    seed_feats = selection["families"][first["id"]]["methods"]["lp_target"] or [
        {"layer": 11, "feature_idx": 0, "mean_activation": 0.0}
    ]
    control_feats = choose_control_features(
        model,
        demo_prompt,
        demo_pos,
        [
            {
                "layer": f["layer"],
                "feature_idx": f["feature_idx"],
                "mean_activation": float(f.get("mean_activation", 0.0)),
            }
            for f in seed_feats
        ],
        config,
    )

    factors = [float(x) for x in config["amplify_factors"]]

    # Baseline
    base_rows = []
    for family in config["families"]:
        base_rows.append(
            evaluate_row(
                model,
                tokenizer,
                family,
                config,
                condition_name="baseline",
                features=[],
                op="none",
                amplify_factor=1.0,
            )
        )
    all_rows.extend(base_rows)
    by_condition["baseline"] = summarize(base_rows)

    # Per-method conditions (features are family-specific)
    for method in ("lp_target", "contrast"):
        for op in ("amplify", "zero"):
            for factor in factors if op == "amplify" else [factors[0]]:
                cond = (
                    f"{method}_{op}_x{factor:g}"
                    if op == "amplify"
                    else f"{method}_{op}"
                )
                rows = []
                for family in config["families"]:
                    feats = selection["families"][family["id"]]["methods"][method]
                    feat_dicts = [
                        {
                            "layer": f["layer"],
                            "feature_idx": f["feature_idx"],
                            "mean_activation": float(f.get("mean_activation", 0.0)),
                        }
                        for f in feats
                    ]
                    # Fill mean_activation from live baseline for amplify
                    rows.append(
                        evaluate_row(
                            model,
                            tokenizer,
                            family,
                            config,
                            condition_name=cond,
                            features=feat_dicts,
                            op=op,
                            amplify_factor=factor,
                        )
                    )
                all_rows.extend(rows)
                by_condition[cond] = summarize(rows)

    # Controls
    for factor in factors:
        cond = f"control_amplify_x{factor:g}"
        rows = [
            evaluate_row(
                model,
                tokenizer,
                family,
                config,
                condition_name=cond,
                features=control_feats,
                op="amplify",
                amplify_factor=factor,
            )
            for family in config["families"]
        ]
        all_rows.extend(rows)
        by_condition[cond] = summarize(rows)

    summary = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "runtime_seconds": time.time() - started,
        "protocol": "per_noun_features__force_native_b__content_on_at_c",
        "families": [f["id"] for f in config["families"]],
        "by_condition": by_condition,
        "control_features": [
            {"layer": f["layer"], "feature_idx": f["feature_idx"]} for f in control_feats
        ],
        "selection_path": str(RESULTS_DIR / "selection.json"),
    }
    summary["interpretation"] = interpret(summary)
    write_json(RESULTS_DIR / "summary.json", summary)
    write_json(RESULTS_DIR / "rows.json", all_rows)
    write_report(summary, RESULTS_DIR / "report.md")
    logging.info("Done. %s", summary["interpretation"])
    print(summary["interpretation"])


def main() -> None:
    args = parse_args()
    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.graph_only:
        run_graph_phase(args.graph_only[0], args.graph_only[1], args.graph_only[2])
        return
    setup_file_logging(RESULTS_DIR)
    if not args.eval_only:
        logging.info("=== Building per-noun attribution graphs ===")
        ensure_graphs(config)
    logging.info("=== Evaluating Stage XVI with per-noun features ===")
    run_eval(config)


if __name__ == "__main__":
    main()
