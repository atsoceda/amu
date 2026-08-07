#!/usr/bin/env python3
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

from experiments.lib.aan_protocol import (
    choose_control_features,
    evaluate_amplify_condition,
    first_content_token_text,
    load_tokenizer,
    slugify,
    summarize_condition,
    verify_dataset,
    write_json,
)
from experiments.lib.core import (
    feature_effect_map,
    load_replacement_model,
    run_graph,
    setup_file_logging,
    token_id_for_text,
)

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
SELECTION_PATH = RESULTS_DIR / "selection.json"

SET_SPECS = {
    "S1_dual_effect": {
        "label": "S1 Dual-effect",
        "rule": "+attr(an) and +attr(future)",
    },
    "S2_article_only": {
        "label": "S2 Article-only",
        "rule": "+attr(an), |attr(future)| near zero",
    },
    "S3_content_only": {
        "label": "S3 Content-only",
        "rule": "+attr(future), |attr(an)| near zero",
    },
    "S4_competing_a": {
        "label": "S4 Competing / a-favoring",
        "rule": "+attr(a-an), |attr(future)| near zero",
    },
}


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-only",
        nargs=2,
        metavar=("PROMPT_ID", "GRAPH_NAME"),
        help="Internal isolated attribution phase: prompt_id article|future",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Build/reuse graphs and write selection.json without evaluation",
    )
    return parser.parse_args()


def ensure_selection_graphs(
    config: dict[str, Any], dataset: dict[str, dict[str, str]]
) -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    demo = config["demonstration"]
    for sentence in config["selection_sentences"]:
        prompt_id = slugify(sentence)
        article_path = GRAPHS_DIR / f"{prompt_id}__article.pt"
        future_path = GRAPHS_DIR / f"{prompt_id}__future.pt"
        for graph_name, path in (("article", article_path), ("future", future_path)):
            if path.exists():
                logging.info("Reusing existing graph %s", path)
                continue
            logging.info("Starting isolated %s attribution for %s", graph_name, sentence)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--graph-only",
                    prompt_id,
                    graph_name,
                ],
                check=True,
            )
        meta_path = GRAPHS_DIR / f"{prompt_id}__meta.json"
        if not meta_path.exists():
            row = dataset[sentence]
            meta_path.write_text(
                json.dumps(
                    {
                        "sentence": sentence,
                        "listed_word": row["word"].lower(),
                        "expected_article": row["article"],
                        "article_prompt": f"{demo} {sentence}",
                        "future_prompt": f"{demo} {sentence} a",
                    },
                    indent=2,
                )
                + "\n"
            )


def run_graph_phase(prompt_id: str, graph_name: str) -> None:
    config = load_config()
    setup_file_logging(RESULTS_DIR)
    dataset = verify_dataset(EXP_DIR, config)
    sentence = next(
        item for item in config["selection_sentences"] if slugify(item) == prompt_id
    )
    row = dataset[sentence]
    demo = config["demonstration"]
    article_prompt = f"{demo} {sentence}"
    future_prompt = f"{demo} {sentence} a"
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    if graph_name == "article":
        target_ids = [
            token_id_for_text(tokenizer, " a"),
            token_id_for_text(tokenizer, " an"),
        ]
        prompt = article_prompt
    else:
        content_text = first_content_token_text(tokenizer, row["word"])
        target_ids = [token_id_for_text(tokenizer, content_text)]
        prompt = future_prompt
        (GRAPHS_DIR / f"{prompt_id}__meta.json").write_text(
            json.dumps(
                {
                    "sentence": sentence,
                    "listed_word": row["word"].lower(),
                    "expected_article": row["article"],
                    "article_prompt": article_prompt,
                    "future_prompt": future_prompt,
                    "content_token_text": content_text,
                },
                indent=2,
            )
            + "\n"
        )
    run_graph(
        model,
        prompt,
        f"{prompt_id}__{graph_name}",
        target_ids,
        config,
        GRAPHS_DIR,
    )


def _empty_stats(layer: int, feature_idx: int) -> dict[str, Any]:
    return {
        "layer": layer,
        "feature_idx": feature_idx,
        "prompt_count": 0,
        "prompts": [],
        "mean_direct_effect_an": 0.0,
        "mean_direct_effect_a": 0.0,
        "mean_direct_effect_a_minus_an": 0.0,
        "mean_direct_effect_future": 0.0,
        "mean_activation": 0.0,
        "score_sum": 0.0,
    }


def _accumulate(
    feature_stats: dict[tuple[int, int], dict[str, Any]],
    layer: int,
    feature_idx: int,
    sentence: str,
    an_effect: float,
    a_effect: float,
    future_effect: float,
    activation: float,
    score: float,
) -> None:
    feature_key = (layer, feature_idx)
    stats = feature_stats.setdefault(feature_key, _empty_stats(layer, feature_idx))
    stats["prompt_count"] += 1
    stats["prompts"].append(sentence)
    stats["mean_direct_effect_an"] += an_effect
    stats["mean_direct_effect_a"] += a_effect
    stats["mean_direct_effect_a_minus_an"] += a_effect - an_effect
    stats["mean_direct_effect_future"] += future_effect
    stats["mean_activation"] += activation
    stats["score_sum"] += score


def _finalize_ranked(
    feature_stats: dict[tuple[int, int], dict[str, Any]],
    score_key: str = "mean_score",
) -> list[dict[str, Any]]:
    ranked = []
    for stats in feature_stats.values():
        count = stats["prompt_count"]
        ranked.append(
            {
                "layer": stats["layer"],
                "feature_idx": stats["feature_idx"],
                "prompt_count": count,
                "prompts": stats["prompts"],
                "mean_direct_effect_an": stats["mean_direct_effect_an"] / count,
                "mean_direct_effect_a": stats["mean_direct_effect_a"] / count,
                "mean_direct_effect_a_minus_an": (
                    stats["mean_direct_effect_a_minus_an"] / count
                ),
                "mean_direct_effect_future": stats["mean_direct_effect_future"] / count,
                "mean_activation": stats["mean_activation"] / count,
                "mean_score": stats["score_sum"] / count,
                "label": f"`L{stats['layer']}/F{stats['feature_idx']}`",
            }
        )
    ranked.sort(
        key=lambda item: (
            item["prompt_count"],
            item[score_key],
            abs(item["mean_direct_effect_future"]),
        ),
        reverse=True,
    )
    return ranked


def _select_top(
    ranked: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], bool]:
    k = int(config["top_features_per_set"])
    min_count = int(config["min_selection_prompt_count"])
    selected = [item for item in ranked if item["prompt_count"] >= min_count][:k]
    fallback_used = False
    if len(selected) < k:
        fallback_used = True
        selected = ranked[:k]
    return selected, fallback_used


def select_feature_sets(config: dict[str, Any]) -> dict[str, Any]:
    from circuit_tracer.graph import Graph

    tokenizer = load_tokenizer(config)
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    max_future = float(config["max_abs_future_direct_effect"])
    max_article = float(config["max_abs_article_direct_effect"])

    stats_by_set: dict[str, dict[tuple[int, int], dict[str, Any]]] = {
        key: {} for key in SET_SPECS
    }
    prompt_records = []

    for sentence in config["selection_sentences"]:
        prompt_id = slugify(sentence)
        meta = json.loads((GRAPHS_DIR / f"{prompt_id}__meta.json").read_text())
        article_graph = Graph.from_pt(str(GRAPHS_DIR / f"{prompt_id}__article.pt"))
        future_graph = Graph.from_pt(str(GRAPHS_DIR / f"{prompt_id}__future.pt"))
        content_text = meta.get("content_token_text") or first_content_token_text(
            tokenizer, meta["listed_word"]
        )
        content_id = token_id_for_text(tokenizer, content_text)
        pre_article_pos = (
            len(tokenizer(meta["article_prompt"], add_special_tokens=True).input_ids)
            - 1
        )
        an_effects = feature_effect_map(article_graph, an_id)
        a_effects = feature_effect_map(article_graph, a_id)
        future_effects = feature_effect_map(future_graph, content_id)
        keys = set(an_effects) & set(future_effects) & set(a_effects)
        counts = {key: 0 for key in SET_SPECS}

        for key in keys:
            layer, pos, feature_idx = key
            if pos != pre_article_pos:
                continue
            an_de = an_effects[key]["direct_effect"]
            a_de = a_effects[key]["direct_effect"]
            future_de = future_effects[key]["direct_effect"]
            activation = an_effects[key]["activation"]

            if an_de > 0 and future_de > 0:
                _accumulate(
                    stats_by_set["S1_dual_effect"],
                    layer,
                    feature_idx,
                    sentence,
                    an_de,
                    a_de,
                    future_de,
                    activation,
                    min(an_de, future_de),
                )
                counts["S1_dual_effect"] += 1

            if an_de > 0 and abs(future_de) <= max_future:
                _accumulate(
                    stats_by_set["S2_article_only"],
                    layer,
                    feature_idx,
                    sentence,
                    an_de,
                    a_de,
                    future_de,
                    activation,
                    an_de,
                )
                counts["S2_article_only"] += 1

            if future_de > 0 and abs(an_de) <= max_article:
                _accumulate(
                    stats_by_set["S3_content_only"],
                    layer,
                    feature_idx,
                    sentence,
                    an_de,
                    a_de,
                    future_de,
                    activation,
                    future_de,
                )
                counts["S3_content_only"] += 1

            if (a_de - an_de) > 0 and abs(future_de) <= max_future:
                _accumulate(
                    stats_by_set["S4_competing_a"],
                    layer,
                    feature_idx,
                    sentence,
                    an_de,
                    a_de,
                    future_de,
                    activation,
                    a_de - an_de,
                )
                counts["S4_competing_a"] += 1

        prompt_records.append(
            {
                "sentence": sentence,
                "listed_word": meta["listed_word"],
                "content_token_text": content_text,
                "pre_article_pos": pre_article_pos,
                "candidate_counts": counts,
            }
        )

    sets_out: dict[str, Any] = {}
    for set_id, spec in SET_SPECS.items():
        ranked = _finalize_ranked(stats_by_set[set_id])
        selected, fallback_used = _select_top(ranked, config)
        sets_out[set_id] = {
            "set_id": set_id,
            "label": spec["label"],
            "rule": spec["rule"],
            "fallback_used": fallback_used,
            "ranked_features": ranked[:50],
            "selected_features": selected,
        }

    selection = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_prompt_count": len(config["selection_sentences"]),
        "min_selection_prompt_count": int(config["min_selection_prompt_count"]),
        "top_features_per_set": int(config["top_features_per_set"]),
        "max_abs_future_direct_effect": float(config["max_abs_future_direct_effect"]),
        "max_abs_article_direct_effect": float(config["max_abs_article_direct_effect"]),
        "prompt_records": prompt_records,
        "sets": sets_out,
    }
    write_json(SELECTION_PATH, selection)
    return selection


def classify_set(
    summary: dict[str, Any], control_summary: dict[str, Any]
) -> tuple[str, str]:
    signal = (
        summary["article_moved_toward_an_rate"] >= 0.25
        or summary["article_moved_toward_a_rate"] >= 0.25
        or abs(summary["mean_delta_an_minus_a"]) >= 0.25
        or summary["generated_article_changed_rate"] >= 0.2
        or summary["class_shifted_rate"] >= 0.2
    )
    control_signal = (
        control_summary["article_moved_toward_an_rate"] >= 0.25
        or abs(control_summary["mean_delta_an_minus_a"]) >= 0.25
        or control_summary["generated_article_changed_rate"] >= 0.2
        or control_summary["class_shifted_rate"] >= 0.2
    )
    beats_control = abs(summary["mean_delta_an_minus_a"]) > abs(
        control_summary["mean_delta_an_minus_a"]
    ) + 0.125 or (
        summary["trajectory_like_rate"]
        > control_summary["trajectory_like_rate"] + 1e-9
        or summary["wrapper_like_rate"]
        > control_summary["wrapper_like_rate"] + 1e-9
        or summary["class_shifted_rate"]
        > control_summary["class_shifted_rate"] + 0.1
    )
    if not signal or (control_signal and not beats_control):
        return (
            "nonspecific_or_null",
            "Did not beat activation-matched controls enough to classify.",
        )
    if (
        summary["wrapper_like_rate"] >= 0.15
        and summary["wrapper_like_rate"] > summary["trajectory_like_rate"]
        and summary["content_preserved_rate"]
        >= summary["content_word_changed_rate"]
    ):
        return (
            "wrapper_like",
            "Article movement with content preservation above class switching.",
        )
    if summary["trajectory_like_rate"] >= 0.15 or (
        summary["class_shifted_rate"] >= 0.2
        and summary["content_preserved_rate"] < 0.5
    ):
        return (
            "trajectory_like",
            "Control-beating effect mainly changes content / vowel-consonant class.",
        )
    return (
        "mixed_or_article_shift",
        "Beats controls with a mixed or article-only pattern.",
    )


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Selection-Criterion Ablation (E1)",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        "",
        "## Question",
        "",
        "Does *how* we pick sparse features determine whether gain-of-function looks like content-preserving wrappers or compiled trajectory packages?",
        "",
        "## Design",
        "",
        f"- Amplify factor: {summary['amplify_factor']}×",
        f"- Selection prompts: {summary['selection_prompt_count']}",
        f"- Held-out test prompts: {summary['test_prompt_count']}",
        f"- Features per set: {summary['top_features_per_set']}",
        f"- Near-zero future |attr|: {summary['max_abs_future_direct_effect']}",
        f"- Near-zero article |attr(an)|: {summary['max_abs_article_direct_effect']}",
        "",
        "## Aggregate Comparison",
        "",
        "| Set | Decision | Mean Δ(an−a) | Wrapper-like | Trajectory-like | Content preserved | Class shifted | vs control Δ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for set_id, block in summary["set_results"].items():
        s = block["summary"]
        c = block["control_summary"]
        lines.append(
            f"| {block['label']} | `{block['decision']}` | "
            f"{s['mean_delta_an_minus_a']:.3f} | {s['wrapper_like_rate']:.3f} | "
            f"{s['trajectory_like_rate']:.3f} | {s['content_preserved_rate']:.3f} | "
            f"{s['class_shifted_rate']:.3f} | "
            f"{s['mean_delta_an_minus_a'] - c['mean_delta_an_minus_a']:.3f} |"
        )
    lines += [
        "",
        "## Short Answer",
        "",
        summary["interpretation"],
        "",
        "## Selected Features",
        "",
    ]
    for set_id, block in summary["set_results"].items():
        lines.append(f"### {block['label']}")
        lines.append("")
        lines.append(f"Rule: {block['rule']}")
        lines.append(f"Fallback used: {block['fallback_used']}")
        lines.append("")
        feats = block["selected_features"]
        if not feats:
            lines.append("No features selected.")
        else:
            lines.append(
                "| Feature | Prompt count | Mean score | Mean attr `an` | Mean attr future |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for item in feats:
                lines.append(
                    f"| {item['label']} | {item['prompt_count']} | "
                    f"{item['mean_score']:.3f} | {item['mean_direct_effect_an']:.3f} | "
                    f"{item['mean_direct_effect_future']:.3f} |"
                )
        lines.append("")
        lines.append(
            f"Control features: "
            + (", ".join(x["label"] for x in block["control_features"]) or "None")
        )
        lines.append("")

    lines += [
        "## Per-set article-change examples",
        "",
    ]
    for set_id, block in summary["set_results"].items():
        changed = [
            item
            for item in block["examples"]
            if item["generated_article_changed"] or item["class_shifted"]
        ]
        lines.append(f"### {block['label']}")
        lines.append("")
        if not changed:
            lines.append("No article or class changes under amplification.")
            lines.append("")
            continue
        lines.append(
            "| Prompt | Baseline | Intervention | Content preserved? | Class shifted? | Δ(an−a) |"
        )
        lines.append("| --- | --- | --- | --- | --- | ---: |")
        for item in changed[:12]:
            lines.append(
                f"| `{item['target_prompt']}` | `{item['baseline_continuation']}` | "
                f"`{item['intervention_continuation']}` | {item['content_preserved']} | "
                f"{item['class_shifted']} | {item['delta_an_minus_a']:.3f} |"
            )
        lines.append("")

    lines += [
        "## Interpretation Boundary",
        "",
        "Wrapper-like ≥ 0.25 on any set supports Outcome A pathway for later dual-lock tests. "
        "Trajectory-like dominance on dual-effect (S1) with no wrapper-like set supports Outcome B. "
        "E2 dose-sweeps S1 and any set with wrapper-like ≥ 0.25 or strong article effect.",
        "",
        "## Source Artifacts",
        "",
        "- `results/selection.json`",
        "- `results/summary.json`",
        "- `results/graphs/`",
        "",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def interpret_overall(set_results: dict[str, Any]) -> str:
    decisions = {sid: block["decision"] for sid, block in set_results.items()}
    wrapper_sets = [sid for sid, d in decisions.items() if d == "wrapper_like"]
    traj_sets = [sid for sid, d in decisions.items() if d == "trajectory_like"]
    if wrapper_sets:
        return (
            f"At least one selection rule looks wrapper-like ({', '.join(wrapper_sets)}). "
            "E2/E3 should dose-sweep and dual-lock those sets; Outcome A remains live."
        )
    if "S1_dual_effect" in traj_sets and not wrapper_sets:
        return (
            "S1 dual-effect is trajectory-like and no rule yielded a clean wrapper-like "
            "held-out pattern. Favors Outcome B so far; E3 dual lock is still required."
        )
    if traj_sets:
        return (
            f"Trajectory-like sets: {', '.join(traj_sets)}. No wrapper-like set. "
            "Proceed to E2/E3 with S1 and any strong article movers."
        )
    return (
        "No set cleanly beat controls into wrapper-like or trajectory-like territory. "
        "Inspect absolute Δ and illicit mismatch rates before redesigning selection thresholds."
    )


def main() -> None:
    args = parse_args()
    if args.graph_only:
        run_graph_phase(args.graph_only[0], args.graph_only[1])
        return

    config = load_config()
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    dataset = verify_dataset(EXP_DIR, config)
    ensure_selection_graphs(config, dataset)
    selection = select_feature_sets(config)
    if args.selection_only:
        logging.info("selection-only complete: %s", SELECTION_PATH)
        return

    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    first_prompt = (
        f"{config['demonstration']} {config['test_examples'][0]['sentence']}"
    )
    first_pos = len(tokenizer(first_prompt, add_special_tokens=True).input_ids) - 1

    set_results: dict[str, Any] = {}
    all_selected_for_control_forbid: list[dict[str, Any]] = []
    for set_id, set_block in selection["sets"].items():
        all_selected_for_control_forbid.extend(set_block["selected_features"])

    for set_id, set_block in selection["sets"].items():
        selected = set_block["selected_features"]
        if not selected:
            logging.warning("No features for %s; skipping evaluation", set_id)
            set_results[set_id] = {
                "set_id": set_id,
                "label": set_block["label"],
                "rule": set_block["rule"],
                "fallback_used": set_block["fallback_used"],
                "selected_features": [],
                "control_features": [],
                "summary": summarize_condition([]),
                "control_summary": summarize_condition([]),
                "decision": "empty_set",
                "decision_note": "No features selected under this rule.",
                "examples": [],
                "control_examples": [],
            }
            continue
        control_features = choose_control_features(
            model,
            first_prompt,
            first_pos,
            selected,
            config,
        )
        rows = evaluate_amplify_condition(
            model,
            tokenizer,
            config["test_examples"],
            selected,
            config,
            f"{set_id}_amplify",
        )
        control_rows = evaluate_amplify_condition(
            model,
            tokenizer,
            config["test_examples"],
            control_features,
            config,
            f"{set_id}_control",
        )
        summary = summarize_condition(rows)
        control_summary = summarize_condition(control_rows)
        decision, note = classify_set(summary, control_summary)
        set_results[set_id] = {
            "set_id": set_id,
            "label": set_block["label"],
            "rule": set_block["rule"],
            "fallback_used": set_block["fallback_used"],
            "selected_features": selected,
            "control_features": control_features,
            "summary": summary,
            "control_summary": control_summary,
            "decision": decision,
            "decision_note": note,
            "examples": rows,
            "control_examples": control_rows,
        }
        logging.info("%s decision: %s", set_id, decision)

    interpretation = interpret_overall(set_results)
    e2_candidates = []
    for set_id, block in set_results.items():
        s = block["summary"]
        strong_article = (
            abs(s["mean_delta_an_minus_a"]) >= 0.25
            or s["generated_article_changed_rate"] >= 0.2
            or s["article_moved_toward_an_rate"] >= 0.25
            or s["article_moved_toward_a_rate"] >= 0.25
        )
        if set_id == "S1_dual_effect" or block["decision"] == "wrapper_like" or (
            s["wrapper_like_rate"] >= 0.25 or strong_article
        ):
            e2_candidates.append(set_id)

    payload = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "runtime_seconds": time.time() - started,
        "demonstration": config["demonstration"],
        "amplify_factor": float(config["amplify_factor"]),
        "selection_prompt_count": len(config["selection_sentences"]),
        "test_prompt_count": len(config["test_examples"]),
        "top_features_per_set": int(config["top_features_per_set"]),
        "max_abs_future_direct_effect": float(config["max_abs_future_direct_effect"]),
        "max_abs_article_direct_effect": float(config["max_abs_article_direct_effect"]),
        "e2_dose_sweep_sets": sorted(set(e2_candidates)),
        "interpretation": interpretation,
        "set_results": set_results,
    }
    write_json(RESULTS_DIR / "summary.json", payload)
    write_report(payload)
    logging.info("E1 complete. %s", interpretation)


if __name__ == "__main__":
    main()
