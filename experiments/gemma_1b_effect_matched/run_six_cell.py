#!/usr/bin/env python3
"""Six-cell generated-token design across feature families and S1 doses.

For each prompt the intervention-off cells are computed once. Each handle then
adds intervention-on free generation and intervention-on forced `a` / `an`.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.lib.aan_protocol import article_and_word, token_id_for_text, write_json
from experiments.lib.core import load_replacement_model, setup_file_logging
from experiments.lib.mediation_estimands import (
    effect_vector_decomposition,
    js_divergence_from_logits,
    total_variation_from_logits,
)


EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_config_path(config: dict[str, Any], key: str) -> Path:
    return (EXP_DIR / str(config[key])).resolve()


def load_handle_features(
    config: dict[str, Any],
    run: dict[str, Any],
) -> list[dict[str, Any]]:
    set_id = str(run["feature_set"])
    if run.get("ranked_features_path"):
        ranked = load_json((EXP_DIR / str(run["ranked_features_path"])).resolve())
        selected = ranked[: int(run["feature_count"])]
        source = "development_margin_ranked_features"
    elif run.get("use_control_features"):
        summary = load_json(resolve_config_path(config, "e1_summary_path"))
        selected = summary["set_results"][set_id]["control_features"]
        source = "e1_control_features"
    else:
        selection = load_json(resolve_config_path(config, "e1_selection_path"))
        selected = selection["sets"][set_id]["selected_features"]
        source = "e1_selected_features"
    if not selected:
        raise RuntimeError(f"No features for {run['id']}")
    features = []
    for item in selected:
        features.append(
            {
                "layer": int(item["layer"]),
                "feature_idx": int(item["feature_idx"]),
                "mean_activation": float(item.get("mean_activation", 0.0)),
                "label": str(item.get("label", "")),
                "source": source,
            }
        )
    return features


def intervention_tuples(
    interventions: list[dict[str, Any]],
) -> list[tuple[int, int, int, float]]:
    return [
        (
            int(item["layer"]),
            int(item["pos"]),
            int(item["feature_idx"]),
            float(item["value"]),
        )
        for item in interventions
    ]


@torch.no_grad()
def next_logits(
    model,
    prompt: str,
    interventions: list[dict[str, Any]],
) -> torch.Tensor:
    logits, _ = model.feature_intervention(
        prompt,
        interventions=intervention_tuples(interventions),
        freeze_attention=False,
        sparse=True,
        return_activations=False,
    )
    return logits[0, -1].detach().float().cpu()


def top_tokens(tokenizer, logits: torch.Tensor, k: int) -> list[dict[str, Any]]:
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=min(k, logits.numel()))
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "logit": float(logits[int(token_id)]),
            "prob": float(prob),
        }
        for rank, (prob, token_id) in enumerate(zip(top_probs, top_ids), start=1)
    ]


def greedy_generate(
    model,
    prompt: str,
    interventions: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    top_k: int,
) -> dict[str, Any]:
    current = prompt
    ids: list[int] = []
    steps: list[dict[str, Any]] = []
    step_logits: list[torch.Tensor] = []
    for _ in range(max_new_tokens):
        logits = next_logits(model, current, interventions)
        token_id = int(torch.argmax(logits).item())
        steps.append(
            {
                "token_id": token_id,
                "token": model.tokenizer.decode([token_id]),
                "top_tokens": top_tokens(model.tokenizer, logits, top_k),
            }
        )
        step_logits.append(logits)
        ids.append(token_id)
        piece = model.tokenizer.decode([token_id])
        current += piece
        if piece.strip() in {".", "!", "?"}:
            break
    continuation = model.tokenizer.decode(ids)
    article, word = article_and_word(continuation)
    return {
        "continuation": continuation,
        "article": article,
        "word": word,
        "ids": ids,
        "steps": steps,
        "_step_logits": step_logits,
    }


def word_token_ids(tokenizer, word: str) -> list[int]:
    if not word:
        return []
    return [int(x) for x in tokenizer(f" {word}", add_special_tokens=False).input_ids]


def sequence_logprob(
    model,
    prefix: str,
    word: str,
    interventions: list[dict[str, Any]],
    *,
    first_logits: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    ids = word_token_ids(model.tokenizer, word)
    if not ids:
        return None
    current = prefix
    total = 0.0
    logits = first_logits
    for token_id in ids:
        if logits is None:
            logits = next_logits(model, current, interventions)
        log_probs = torch.log_softmax(logits, dim=-1)
        total += float(log_probs[token_id])
        current += model.tokenizer.decode([token_id])
        logits = None
    return {
        "word": word,
        "n_tokens": len(ids),
        "logprob_sum": total,
        "logprob_mean": total / len(ids),
    }


def activations_at_position(model, prompt: str, position: int):
    _, activations = model.feature_intervention(
        prompt,
        interventions=[],
        freeze_attention=False,
        sparse=False,
        return_activations=True,
    )
    if activations is None:
        raise RuntimeError("Missing sparse activations")
    return activations[:, position]


def build_interventions(
    activations,
    position: int,
    features: list[dict[str, Any]],
    amplify_factor: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    interventions = []
    rows = []
    for feature in features:
        layer = int(feature["layer"])
        feature_idx = int(feature["feature_idx"])
        activation = float(activations[layer, feature_idx].detach().float().cpu())
        value = float(activation * amplify_factor)
        interventions.append(
            {
                "layer": layer,
                "pos": position,
                "feature_idx": feature_idx,
                "value": value,
            }
        )
        rows.append(
            {
                "layer": layer,
                "feature_idx": feature_idx,
                "activation": activation,
                "value": value,
            }
        )
    return interventions, rows


def compact_generation(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "continuation": payload["continuation"],
        "article": payload["article"],
        "word": payload["word"],
        "top1": payload["steps"][0]["token"] if payload["steps"] else "",
    }


def top1_word(tokenizer, logits: torch.Tensor) -> str:
    return tokenizer.decode([int(torch.argmax(logits).item())]).strip().lower()


def distribution_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    return {
        "js_full_vocab": js_divergence_from_logits(left, right),
        "tv_full_vocab": total_variation_from_logits(left, right),
    }


def percentile_interval(
    values: list[float],
    rng: random.Random,
    n_resamples: int,
) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "lo": None, "hi": None}
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_index = max(0, math.floor(0.025 * (len(means) - 1)))
    hi_index = min(len(means) - 1, math.ceil(0.975 * (len(means) - 1)))
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if n % 2
        else 0.5 * (ordered[midpoint - 1] + ordered[midpoint])
    )
    return {
        "n": n,
        "mean": sum(values) / n,
        "median": median,
        "lo": means[lo_index],
        "hi": means[hi_index],
        "method": "prompt-level nonparametric bootstrap",
        "resamples": n_resamples,
    }


def summarize_handle(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    rng = random.Random(int(config["bootstrap_seed"]))
    resamples = int(config["bootstrap_resamples"])

    def interval(key_path: tuple[str, ...]) -> dict[str, Any]:
        values = []
        for row in rows:
            cursor: Any = row
            missing = False
            for key in key_path:
                if cursor is None or key not in cursor:
                    missing = True
                    break
                cursor = cursor[key]
            if missing or cursor is None:
                continue
            values.append(float(cursor))
        return percentile_interval(values, rng, resamples)

    article_pairs = [row for row in rows if row["decomposition"] is not None]
    switch_pairs = [
        row
        for row in article_pairs
        if row["decomposition"]["baseline_article"]
        != row["decomposition"]["selected_article"]
    ]
    return {
        "n_prompts": len(rows),
        "rates": {
            "free_on_article_an": sum(row["free"]["on"]["article"] == "an" for row in rows)
            / max(len(rows), 1),
            "free_on_article_a": sum(row["free"]["on"]["article"] == "a" for row in rows)
            / max(len(rows), 1),
            "free_on_article_other": sum(
                row["free"]["on"]["article"] == "other" for row in rows
            )
            / max(len(rows), 1),
            "free_noun_changed": sum(
                row["free"]["on"]["word"] != row["free"]["off"]["word"] for row in rows
            )
            / max(len(rows), 1),
            "treated_token_top1_reproduction": sum(
                bool(row["reproduction"] and row["reproduction"]["top1_match"])
                for row in rows
            )
            / max(
                sum(row["reproduction"] is not None for row in rows),
                1,
            ),
        },
        "matched_prefix": {
            "a": interval(("comparisons", "a", "tv_full_vocab")),
            "an": interval(("comparisons", "an", "tv_full_vocab")),
        },
        "decomposition_all_articled": {
            "n": len(article_pairs),
            "total_tv": interval(("decomposition", "total_tv")),
            "mediator_tv": interval(("decomposition", "mediator_tv")),
            "residual_tv": interval(("decomposition", "residual_tv")),
            "cosine_mediator_total": interval(
                ("decomposition", "cosine_mediator_total")
            ),
            "cosine_residual_total": interval(
                ("decomposition", "cosine_residual_total")
            ),
            "reconstruction_l1": interval(("decomposition", "reconstruction_l1")),
        },
        "descriptive_article_switch_subset": {
            "n": len(switch_pairs),
            "total_tv": percentile_interval(
                [float(row["decomposition"]["total_tv"]) for row in switch_pairs],
                rng,
                resamples,
            ),
            "mediator_tv": percentile_interval(
                [float(row["decomposition"]["mediator_tv"]) for row in switch_pairs],
                rng,
                resamples,
            ),
            "residual_tv": percentile_interval(
                [float(row["decomposition"]["residual_tv"]) for row in switch_pairs],
                rng,
                resamples,
            ),
            "cosine_mediator_total": percentile_interval(
                [
                    float(row["decomposition"]["cosine_mediator_total"])
                    for row in switch_pairs
                ],
                rng,
                resamples,
            ),
        },
        "generic_article_prefix_tv": interval(("generic_article_prefix", "tv_full_vocab")),
        "pre_specified_twins": {
            "a": interval(
                ("pre_specified_twin_effects", "a", "delta_delta_target_minus_source")
            ),
            "an": interval(
                ("pre_specified_twin_effects", "an", "delta_delta_target_minus_source")
            ),
        },
    }


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Six-cell family sweep",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        f"Runtime: {summary['elapsed_sec']:.1f}s",
        "",
        "Each handle uses intervention off/on crossed with free generation,",
        "inserted `a`, and inserted `an`. Intervention-off cells are shared.",
        "",
        "| Handle | Free `an` | Noun changed | Treated-token top-1 match | Total TV | Token-substitution TV | Matched-prefix residual TV | Residual TV under `a` | Residual TV under `an` | Twin ΔΔ `a` | Twin ΔΔ `an` |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in summary["runs"]:
        block = summary["analysis"][run["id"]]
        dec = block["decomposition_all_articled"]
        lines.append(
            "| `{id}` | {an:.2f} | {changed:.2f} | {match:.2f} | {total} | {med} | {res} | {cde_a} | {cde_an} | {twin_a} | {twin_an} |".format(
                id=run["id"],
                an=block["rates"]["free_on_article_an"],
                changed=block["rates"]["free_noun_changed"],
                match=block["rates"]["treated_token_top1_reproduction"],
                total=_fmt_interval(dec["total_tv"]),
                med=_fmt_interval(dec["mediator_tv"]),
                res=_fmt_interval(dec["residual_tv"]),
                cde_a=_fmt_interval(block["matched_prefix"]["a"]),
                cde_an=_fmt_interval(block["matched_prefix"]["an"]),
                twin_a=_fmt_interval(block["pre_specified_twins"]["a"]),
                twin_an=_fmt_interval(block["pre_specified_twins"]["an"]),
            )
        )
    generic = next(iter(summary["analysis"].values()))["generic_article_prefix_tv"]
    lines.extend(
        [
            "",
            "## Shared intervention-off article contrast",
            "",
            "Total variation between inserted `a` and inserted `an` with the",
            "intervention off, on the same 20 held-out prompts:",
            f"{_fmt_interval(generic)}.",
            "",
            "## Notes",
            "",
            "- Total / token-substitution / residual TV come from the additive",
            "  probability-vector split and are reported only when both free articles",
            "  are `a` or `an`.",
            "- Matched-prefix residual TV under `a` and `an` uses all 20 prompts.",
            "- Twin ΔΔ is target-minus-source first-token logits on the seven",
            "  pre-specified pairs.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def _fmt_interval(block: dict[str, Any]) -> str:
    if not block or block.get("mean") is None:
        return "—"
    return f"{block['mean']:.3f} [{block['lo']:.3f}, {block['hi']:.3f}]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(CONFIG_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR)
    started = time.time()

    e1_config = load_json(resolve_config_path(config, "e1_config_path"))
    examples = list(e1_config["test_examples"])
    if args.max_prompts is not None:
        examples = examples[: args.max_prompts]
    runs = list(config["runs"])
    if args.run_ids:
        wanted = set(args.run_ids)
        runs = [run for run in runs if run["id"] in wanted]
        if not runs:
            raise RuntimeError(f"No runs matched {sorted(wanted)}")

    handle_features = {run["id"]: load_handle_features(config, run) for run in runs}

    logging.info("Loading model for %s", config["experiment_name"])
    model = load_replacement_model(config)
    tokenizer = model.tokenizer

    vocabulary_words = sorted(
        {
            word
            for example in examples
            for word in (example.get("listed_word", ""), example.get("twin_word", ""))
            if word
        }
    )

    all_rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        prompt = f"{config['demonstration']} {example['sentence']}"
        position = len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1
        activations = activations_at_position(model, prompt, position)

        free_off = greedy_generate(
            model,
            prompt,
            [],
            max_new_tokens=int(config["max_new_tokens"]),
            top_k=int(config["top_k"]),
        )
        free_on_by_run: dict[str, dict[str, Any]] = {}
        interventions_by_run: dict[str, list[dict[str, Any]]] = {}
        activation_rows_by_run: dict[str, list[dict[str, Any]]] = {}
        for run in runs:
            interventions, activation_rows = build_interventions(
                activations,
                position,
                handle_features[run["id"]],
                float(run["amplify_factor"]),
            )
            interventions_by_run[run["id"]] = interventions
            activation_rows_by_run[run["id"]] = activation_rows
            free_on_by_run[run["id"]] = greedy_generate(
                model,
                prompt,
                interventions,
                max_new_tokens=int(config["max_new_tokens"]),
                top_k=int(config["top_k"]),
            )

        words = list(
            dict.fromkeys(
                [
                    str(example.get("listed_word", "")),
                    str(example.get("twin_word", "")),
                    str(free_off["word"]),
                    *[str(free_on_by_run[run["id"]]["word"]) for run in runs],
                ]
            )
        )
        words = [word for word in words if word]

        forced_off_logits: dict[str, torch.Tensor] = {}
        forced_off_cells: dict[str, dict[str, Any]] = {}
        for article in ("a", "an"):
            article_id = token_id_for_text(tokenizer, f" {article}")
            prefix = prompt + tokenizer.decode([article_id])
            logits = next_logits(model, prefix, [])
            forced_off_logits[article] = logits
            forced_off_cells[article] = {
                "article": article,
                "top1": top1_word(tokenizer, logits),
                "top_tokens": top_tokens(tokenizer, logits, int(config["top_k"])),
                "word_sequences": {},
            }
            for word in words:
                result = sequence_logprob(
                    model, prefix, word, [], first_logits=logits
                )
                if result is not None:
                    forced_off_cells[article]["word_sequences"][word] = result

        generic_article = distribution_metrics(
            forced_off_logits["an"],
            forced_off_logits["a"],
        )

        listed_word = str(example.get("listed_word", ""))
        twin_word = str(example.get("twin_word", ""))
        listed_ids = word_token_ids(tokenizer, listed_word)
        twin_ids = word_token_ids(tokenizer, twin_word)

        for run in runs:
            run_id = run["id"]
            free_on = free_on_by_run[run_id]
            interventions = interventions_by_run[run_id]
            forced_on_logits: dict[str, torch.Tensor] = {}
            forced_on_cells: dict[str, dict[str, Any]] = {}
            for article in ("a", "an"):
                article_id = token_id_for_text(tokenizer, f" {article}")
                prefix = prompt + tokenizer.decode([article_id])
                logits = next_logits(model, prefix, interventions)
                forced_on_logits[article] = logits
                forced_on_cells[article] = {
                    "article": article,
                    "top1": top1_word(tokenizer, logits),
                    "top_tokens": top_tokens(tokenizer, logits, int(config["top_k"])),
                    "word_sequences": {},
                }
                for word in words:
                    result = sequence_logprob(
                        model,
                        prefix,
                        word,
                        interventions,
                        first_logits=logits,
                    )
                    if result is not None:
                        forced_on_cells[article]["word_sequences"][word] = result

            comparisons = {
                article: distribution_metrics(
                    forced_on_logits[article],
                    forced_off_logits[article],
                )
                for article in ("a", "an")
            }

            selected_article = free_on["article"]
            baseline_article = free_off["article"]
            reproduction = None
            decomposition = None
            if selected_article in {"a", "an"}:
                selected_off = forced_off_logits[selected_article]
                selected_on = forced_on_logits[selected_article]
                free_noun_logits = (
                    free_on["_step_logits"][1]
                    if len(free_on["_step_logits"]) > 1
                    else None
                )
                reproduction = {
                    "selected_article": selected_article,
                    "free_on_word": free_on["word"],
                    "treated_token_replay_word": top1_word(tokenizer, selected_off),
                    "matched_prefix_on_word": top1_word(tokenizer, selected_on),
                    "top1_match": int(torch.argmax(selected_off).item())
                    == int(torch.argmax(selected_on).item()),
                    "free_on_vs_forced_on_max_abs_logit": (
                        float(torch.max(torch.abs(free_noun_logits - selected_on)))
                        if free_noun_logits is not None
                        else None
                    ),
                }
                if baseline_article in {"a", "an"}:
                    vectors = effect_vector_decomposition(
                        baseline_off=forced_off_logits[baseline_article],
                        treated_article_off=selected_off,
                        treated_article_on=selected_on,
                    )
                    source_ids = word_token_ids(tokenizer, free_off["word"])
                    target_ids = word_token_ids(tokenizer, free_on["word"])
                    signed = {}
                    if source_ids and target_ids:
                        source_id = source_ids[0]
                        target_id = target_ids[0]

                        def contrast(logits: torch.Tensor) -> float:
                            return float(logits[target_id] - logits[source_id])

                        signed = {
                            "posthoc_source_word": free_off["word"],
                            "posthoc_target_word": free_on["word"],
                            "total_delta_delta": contrast(selected_on)
                            - contrast(forced_off_logits[baseline_article]),
                            "mediator_delta_delta": contrast(selected_off)
                            - contrast(forced_off_logits[baseline_article]),
                            "residual_delta_delta": contrast(selected_on)
                            - contrast(selected_off),
                        }
                    decomposition = {
                        "baseline_article": baseline_article,
                        "selected_article": selected_article,
                        **vectors,
                        **signed,
                    }

            twin_effects = None
            if listed_ids and twin_ids:
                listed_id = listed_ids[0]
                twin_id = twin_ids[0]
                twin_effects = {}
                for article in ("a", "an"):
                    off_logits = forced_off_logits[article]
                    on_logits = forced_on_logits[article]
                    first_off = float(off_logits[twin_id] - off_logits[listed_id])
                    first_on = float(on_logits[twin_id] - on_logits[listed_id])
                    off_seq = forced_off_cells[article]["word_sequences"]
                    on_seq = forced_on_cells[article]["word_sequences"]
                    seq_off = (
                        off_seq[twin_word]["logprob_sum"]
                        - off_seq[listed_word]["logprob_sum"]
                        if listed_word in off_seq and twin_word in off_seq
                        else None
                    )
                    seq_on = (
                        on_seq[twin_word]["logprob_sum"]
                        - on_seq[listed_word]["logprob_sum"]
                        if listed_word in on_seq and twin_word in on_seq
                        else None
                    )
                    twin_effects[article] = {
                        "target": twin_word,
                        "source": listed_word,
                        "target_minus_source_off": first_off,
                        "target_minus_source_on": first_on,
                        "delta_delta_target_minus_source": first_on - first_off,
                        "delta_sequence_log_odds": (
                            None if seq_off is None or seq_on is None else seq_on - seq_off
                        ),
                    }

            all_rows.append(
                {
                    "index": index,
                    "run_id": run_id,
                    "feature_set": run["feature_set"],
                    "amplify_factor": run["amplify_factor"],
                    "use_control_features": bool(run.get("use_control_features")),
                    "sentence": example["sentence"],
                    "listed_word": listed_word,
                    "twin_word": twin_word,
                    "position": position,
                    "feature_activations": activation_rows_by_run[run_id],
                    "free": {
                        "off": compact_generation(free_off),
                        "on": compact_generation(free_on),
                    },
                    "forced": {
                        "a": {
                            "off": {
                                "top1": forced_off_cells["a"]["top1"],
                                "top_tokens": forced_off_cells["a"]["top_tokens"],
                            },
                            "on": {
                                "top1": forced_on_cells["a"]["top1"],
                                "top_tokens": forced_on_cells["a"]["top_tokens"],
                            },
                        },
                        "an": {
                            "off": {
                                "top1": forced_off_cells["an"]["top1"],
                                "top_tokens": forced_off_cells["an"]["top_tokens"],
                            },
                            "on": {
                                "top1": forced_on_cells["an"]["top1"],
                                "top_tokens": forced_on_cells["an"]["top_tokens"],
                            },
                        },
                    },
                    "comparisons": comparisons,
                    "generic_article_prefix": generic_article,
                    "reproduction": reproduction,
                    "decomposition": decomposition,
                    "pre_specified_twin_effects": twin_effects,
                }
            )
            logging.info(
                "%s %d/%d off=%r on=%r selected=%s residual_tv=%s",
                run_id,
                index,
                len(examples),
                free_off["continuation"],
                free_on["continuation"],
                selected_article,
                (
                    f"{decomposition['residual_tv']:.4f}"
                    if decomposition is not None
                    else "na"
                ),
            )

    analysis = {
        run["id"]: summarize_handle(
            [row for row in all_rows if row["run_id"] == run["id"]],
            config,
        )
        for run in runs
    }
    summary = {
        "experiment": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - started,
        "model": config["model"],
        "transcoder_set": config["transcoder_set"],
        "n_prompts": len(examples),
        "runs": [
            {
                **run,
                "features": handle_features[run["id"]],
            }
            for run in runs
        ],
        "analysis": analysis,
        "intervention_equation": "z_f <- amplify_factor * z_f(prompt, P)",
    }
    write_json(RESULTS_DIR / "rows.json", all_rows)
    write_json(RESULTS_DIR / "summary.json", summary)
    write_report(summary, RESULTS_DIR / "report.md")
    logging.info("Done in %.1fs", summary["elapsed_sec"])


if __name__ == "__main__":
    main()
