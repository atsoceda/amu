#!/usr/bin/env python3
"""Exact S1 x generated-article causal decomposition.

Primary design (on the complete pre-specified E1 held-out set):

    S1 in {off, amplify 5x}
      x article in {free, force "a", force "an"}
      x attention in {recomputed, frozen}

For forced-article cells S1 remains active at the original pre-article
position during noun prediction.  The primary outcome is distributional:
compare S1-on and S1-off noun logits under the exact same visible prefix.
Pre-specified listed/twin nouns are secondary outcomes; nouns selected by
S1's free-generation output are explicitly marked post hoc.
"""
from __future__ import annotations

import json
import logging
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiments.lib.aan_protocol import (
    article_and_word,
    build_amplify_interventions,
    vowel_initial,
    write_json,
)
from experiments.lib.core import load_replacement_model, setup_file_logging, token_id_for_text


EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_config_path(config: dict[str, Any], key: str) -> Path:
    return (EXP_DIR / str(config[key])).resolve()


def load_features(config: dict[str, Any]) -> list[dict[str, Any]]:
    selection = load_json(resolve_config_path(config, "e1_selection_path"))
    selected = selection["sets"][config["feature_set"]]["selected_features"]
    if not selected:
        raise RuntimeError(f"No selected features for {config['feature_set']}")
    return [
        {
            "layer": int(item["layer"]),
            "feature_idx": int(item["feature_idx"]),
            "mean_activation": float(item.get("mean_activation", 0.0)),
            "label": str(item.get("label", "")),
        }
        for item in selected
    ]


def intervention_tuples(interventions: list[dict[str, Any]]) -> list[tuple[int, int, int, float]]:
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
    *,
    freeze_attention: bool,
) -> torch.Tensor:
    logits, _ = model.feature_intervention(
        prompt,
        interventions=intervention_tuples(interventions),
        freeze_attention=freeze_attention,
        sparse=True,
        return_activations=False,
    )
    return logits[0, -1].detach().float().cpu()


def top_tokens(tokenizer, logits: torch.Tensor, k: int) -> list[dict[str, Any]]:
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=min(k, probs.numel()))
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


def token_stats(tokenizer, logits: torch.Tensor, token_id: int) -> dict[str, Any]:
    probs = torch.softmax(logits, dim=-1)
    return {
        "token_id": int(token_id),
        "token": tokenizer.decode([int(token_id)]),
        "logit": float(logits[token_id]),
        "prob": float(probs[token_id]),
        "rank": int((logits > logits[token_id]).sum().item() + 1),
    }


def js_divergence_from_logits(left: torch.Tensor, right: torch.Tensor) -> float:
    log_p = torch.log_softmax(left, dim=-1)
    log_q = torch.log_softmax(right, dim=-1)
    p = torch.exp(log_p)
    q = torch.exp(log_q)
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp_min(torch.finfo(m.dtype).tiny))
    js = 0.5 * torch.sum(p * (log_p - log_m)) + 0.5 * torch.sum(q * (log_q - log_m))
    return float(js)


def total_variation_from_logits(left: torch.Tensor, right: torch.Tensor) -> float:
    p = torch.softmax(left, dim=-1)
    q = torch.softmax(right, dim=-1)
    return float(0.5 * torch.sum(torch.abs(p - q)))


def top_k_overlap(left: torch.Tensor, right: torch.Tensor, k: int) -> float:
    left_ids = set(torch.topk(left, k=min(k, left.numel())).indices.tolist())
    right_ids = set(torch.topk(right, k=min(k, right.numel())).indices.tolist())
    return len(left_ids & right_ids) / max(len(left_ids | right_ids), 1)


def restricted_js(left: torch.Tensor, right: torch.Tensor, token_ids: list[int]) -> float:
    ids = torch.tensor(sorted(set(token_ids)), dtype=torch.long)
    return js_divergence_from_logits(left[ids], right[ids])


def candidate_mass(logits: torch.Tensor, token_ids: list[int]) -> float:
    probs = torch.softmax(logits, dim=-1)
    ids = torch.tensor(sorted(set(token_ids)), dtype=torch.long)
    return float(probs[ids].sum())


def candidate_onset_mass(
    tokenizer,
    logits: torch.Tensor,
    candidate_ids: list[int],
) -> dict[str, float]:
    probs = torch.softmax(logits, dim=-1)
    vowel = 0.0
    consonant = 0.0
    for token_id in sorted(set(candidate_ids)):
        text = tokenizer.decode([token_id]).strip().lower()
        if not text or not text[0].isalpha():
            continue
        if vowel_initial(text):
            vowel += float(probs[token_id])
        else:
            consonant += float(probs[token_id])
    return {"vowel": vowel, "consonant": consonant}


def distribution_comparison(
    tokenizer,
    on_logits: torch.Tensor,
    off_logits: torch.Tensor,
    *,
    candidate_ids: list[int],
    top_k: int,
) -> dict[str, Any]:
    on_mass = candidate_onset_mass(tokenizer, on_logits, candidate_ids)
    off_mass = candidate_onset_mass(tokenizer, off_logits, candidate_ids)
    return {
        "js_full_vocab": js_divergence_from_logits(on_logits, off_logits),
        "tv_full_vocab": total_variation_from_logits(on_logits, off_logits),
        "top_k_jaccard": top_k_overlap(on_logits, off_logits, top_k),
        "js_candidate_tokens": restricted_js(on_logits, off_logits, candidate_ids),
        "candidate_mass_on": candidate_mass(on_logits, candidate_ids),
        "candidate_mass_off": candidate_mass(off_logits, candidate_ids),
        "candidate_vowel_mass_on": on_mass["vowel"],
        "candidate_vowel_mass_off": off_mass["vowel"],
        "candidate_consonant_mass_on": on_mass["consonant"],
        "candidate_consonant_mass_off": off_mass["consonant"],
    }


def decode_generated(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)


def greedy_generate(
    model,
    prompt: str,
    interventions: list[dict[str, Any]],
    *,
    freeze_attention: bool,
    max_new_tokens: int,
    top_k: int,
) -> dict[str, Any]:
    current = prompt
    ids: list[int] = []
    steps: list[dict[str, Any]] = []
    step_logits: list[torch.Tensor] = []
    for _ in range(max_new_tokens):
        logits = next_logits(
            model,
            current,
            interventions,
            freeze_attention=freeze_attention,
        )
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
    continuation = decode_generated(model.tokenizer, ids)
    article, word = article_and_word(continuation)
    return {
        "continuation": continuation,
        "article": article,
        "word": word,
        "ids": ids,
        "steps": steps,
        "_step_logits": step_logits,
    }


def strip_private_tensors(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not key.startswith("_")}


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
    freeze_attention: bool,
    first_logits: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    ids = word_token_ids(model.tokenizer, word)
    if not ids:
        return None
    current = prefix
    total = 0.0
    pieces: list[dict[str, Any]] = []
    logits = first_logits
    for index, token_id in enumerate(ids):
        if logits is None:
            logits = next_logits(
                model,
                current,
                interventions,
                freeze_attention=freeze_attention,
            )
        log_probs = torch.log_softmax(logits, dim=-1)
        value = float(log_probs[token_id])
        pieces.append(
            {
                "token_id": token_id,
                "token": model.tokenizer.decode([token_id]),
                "logprob": value,
                "rank": int((logits > logits[token_id]).sum().item() + 1),
            }
        )
        total += value
        current += model.tokenizer.decode([token_id])
        logits = None
    return {
        "word": word,
        "token_ids": ids,
        "n_tokens": len(ids),
        "logprob_sum": total,
        "logprob_mean": total / len(ids),
        "pieces": pieces,
    }


def forced_cell(
    model,
    prompt: str,
    article: str,
    interventions: list[dict[str, Any]],
    *,
    freeze_attention: bool,
    words: list[str],
    target_ids: list[int],
    top_k: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    article_id = token_id_for_text(model.tokenizer, f" {article}")
    prefix = prompt + model.tokenizer.decode([article_id])
    logits = next_logits(
        model,
        prefix,
        interventions,
        freeze_attention=freeze_attention,
    )
    cell = {
        "article": article,
        "article_id": article_id,
        "prefix": prefix,
        "top_tokens": top_tokens(model.tokenizer, logits, top_k),
        "word_sequences": {},
    }
    for word in words:
        result = sequence_logprob(
            model,
            prefix,
            word,
            interventions,
            freeze_attention=freeze_attention,
            first_logits=logits,
        )
        if result is not None:
            cell["word_sequences"][word] = result
    cell["first_token_stats"] = {
        str(token_id): token_stats(model.tokenizer, logits, token_id)
        for token_id in target_ids
    }
    return cell, logits


def percentile_interval(values: list[float], rng: random.Random, n_resamples: int) -> dict[str, Any]:
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
        if len(ordered) % 2
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


def summarize_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(int(config["bootstrap_seed"]))
    resamples = int(config["bootstrap_resamples"])
    summaries: dict[str, Any] = {}
    for mode in [item["id"] for item in config["attention_modes"]]:
        mode_rows = [row for row in rows if row["attention_mode"] == mode]
        summaries[mode] = {}
        for article in ("a", "an", "s1_selected"):
            comparisons = [
                row["comparisons"][article]
                for row in mode_rows
                if row["comparisons"].get(article) is not None
            ]
            summaries[mode][article] = {
                metric: percentile_interval(
                    [float(item[metric]) for item in comparisons],
                    rng,
                    resamples,
                )
                for metric in (
                    "js_full_vocab",
                    "tv_full_vocab",
                    "top_k_jaccard",
                    "js_candidate_tokens",
                )
            }

        twin_rows = [
            row["pre_specified_twin_effects"]
            for row in mode_rows
            if row["pre_specified_twin_effects"] is not None
        ]
        summaries[mode]["pre_specified_twins"] = {}
        for article in ("a", "an"):
            summaries[mode]["pre_specified_twins"][article] = {
                "delta_delta_target_minus_source": percentile_interval(
                    [
                        float(item[article]["delta_delta_target_minus_source"])
                        for item in twin_rows
                    ],
                    rng,
                    resamples,
                ),
                "delta_sequence_log_odds": percentile_interval(
                    [
                        float(item[article]["delta_sequence_log_odds"])
                        for item in twin_rows
                    ],
                    rng,
                    resamples,
                ),
            }

        summaries[mode]["rates"] = {
            "n_prompts": len(mode_rows),
            "baseline_free_article_a": sum(
                row["free"]["off"]["article"] == "a" for row in mode_rows
            )
            / max(len(mode_rows), 1),
            "s1_free_article_an": sum(
                row["free"]["on"]["article"] == "an" for row in mode_rows
            )
            / max(len(mode_rows), 1),
            "s1_free_noun_changed": sum(
                row["free"]["on"]["word"] != row["free"]["off"]["word"]
                for row in mode_rows
            )
            / max(len(mode_rows), 1),
            "article_only_reproduces_s1_top1": sum(
                row["article_only_reproduction"]["top1_match"]
                for row in mode_rows
                if row["article_only_reproduction"] is not None
            )
            / max(
                sum(
                    row["article_only_reproduction"] is not None for row in mode_rows
                ),
                1,
            ),
        }
        decompositions = [
            row["decomposition"]
            for row in mode_rows
            if row["decomposition"] is not None
        ]
        summaries[mode]["decomposition"] = {
            metric: percentile_interval(
                [float(item[metric]) for item in decompositions],
                rng,
                resamples,
            )
            for metric in (
                "total_js_full_vocab",
                "article_only_js_full_vocab",
                "residual_js_full_vocab",
                "total_tv_full_vocab",
                "article_only_tv_full_vocab",
                "residual_tv_full_vocab",
                "residual_tv_over_total_tv",
                "posthoc_total_delta_delta",
                "posthoc_article_only_delta_delta",
                "posthoc_residual_delta_delta",
            )
        }
        article_switch_decompositions = [
            row["decomposition"]
            for row in mode_rows
            if row["decomposition"] is not None
            and row["decomposition"]["baseline_article"]
            != row["decomposition"]["s1_selected_article"]
        ]
        summaries[mode]["descriptive_article_switch_subset"] = {
            "n": len(article_switch_decompositions),
            **{
                metric: percentile_interval(
                    [float(item[metric]) for item in article_switch_decompositions],
                    rng,
                    resamples,
                )
                for metric in (
                    "total_tv_full_vocab",
                    "article_only_tv_full_vocab",
                    "residual_tv_full_vocab",
                    "residual_tv_over_total_tv",
                )
            },
        }
    return summaries


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# S1 × article factorial",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        f"S1 amplify factor: {summary['amplify_factor']}",
        f"Runtime: {summary['elapsed_sec']:.1f}s",
        "",
        "Primary analysis uses all pre-specified E1 held-out prompts. "
        "The free-switching subset is descriptive only.",
        "",
    ]
    for mode, block in summary["analysis"].items():
        lines.extend([f"## Attention {mode}", ""])
        rates = block["rates"]
        lines.append(
            f"- S1 free article=`an`: {rates['s1_free_article_an']:.2f}; "
            f"S1 free noun changed: {rates['s1_free_noun_changed']:.2f}; "
            f"article-only top-1 reproduction: {rates['article_only_reproduces_s1_top1']:.2f}."
        )
        decomposition = block["decomposition"]
        lines.append(
            "- Distributional decomposition (S1-free selected article versus baseline-free "
            "article): "
            f"total TV={decomposition['total_tv_full_vocab']['mean']:.4f}, "
            f"article-only TV={decomposition['article_only_tv_full_vocab']['mean']:.4f}, "
            f"same-prefix residual TV={decomposition['residual_tv_full_vocab']['mean']:.4f}; "
            f"residual/total={decomposition['residual_tv_over_total_tv']['mean']:.3f} "
            f"[{decomposition['residual_tv_over_total_tv']['lo']:.3f}, "
            f"{decomposition['residual_tv_over_total_tv']['hi']:.3f}]."
        )
        lines.append(
            "- Post-hoc generated target/source contrast (descriptive): "
            f"total ΔΔ={decomposition['posthoc_total_delta_delta']['mean']:.3f}, "
            f"article-only={decomposition['posthoc_article_only_delta_delta']['mean']:.3f}, "
            f"same-prefix residual={decomposition['posthoc_residual_delta_delta']['mean']:.3f}."
        )
        switch_subset = block["descriptive_article_switch_subset"]
        lines.append(
            f"- Descriptive article-switch subset (N={switch_subset['n']}): "
            f"total TV={switch_subset['total_tv_full_vocab']['mean']:.4f}, "
            f"article-only TV={switch_subset['article_only_tv_full_vocab']['mean']:.4f}, "
            f"residual TV={switch_subset['residual_tv_full_vocab']['mean']:.4f}; "
            f"residual/total={switch_subset['residual_tv_over_total_tv']['mean']:.3f} "
            f"[{switch_subset['residual_tv_over_total_tv']['lo']:.3f}, "
            f"{switch_subset['residual_tv_over_total_tv']['hi']:.3f}]."
        )
        for article in ("a", "an", "s1_selected"):
            js = block[article]["js_full_vocab"]
            tv = block[article]["tv_full_vocab"]
            lines.append(
                f"- Same-prefix `{article}`: JS={js['mean']:.6f} "
                f"[{js['lo']:.6f}, {js['hi']:.6f}], "
                f"TV={tv['mean']:.6f} [{tv['lo']:.6f}, {tv['hi']:.6f}] "
                f"(N={js['n']})."
            )
        lines.append("")
        lines.append("Pre-specified twin target effects:")
        for article in ("a", "an"):
            dd = block["pre_specified_twins"][article][
                "delta_delta_target_minus_source"
            ]
            seq = block["pre_specified_twins"][article]["delta_sequence_log_odds"]
            lines.append(
                f"- force `{article}`: first-token ΔΔ={dd['mean']:.3f} "
                f"[{dd['lo']:.3f}, {dd['hi']:.3f}]; "
                f"sequence log-odds Δ={seq['mean']:.3f} "
                f"[{seq['lo']:.3f}, {seq['hi']:.3f}] (N={dd['n']})."
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation guardrails",
            "",
            "- A small same-prefix S1-on/off distance supports an article-mediated "
            "account for this intervention; it does not prove absence of other noun pathways.",
            "- A nonzero controlled effect establishes residual S1 control but does not "
            "identify its internal ontology.",
            "- Frozen-attention and recomputed-attention runs estimate different intervention "
            "semantics and are reported separately.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    config = load_json(CONFIG_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR)
    started = time.time()

    e1_config = load_json(resolve_config_path(config, "e1_config_path"))
    examples = list(e1_config["test_examples"])
    features = load_features(config)

    logging.info("Loading model for %s", config["experiment_name"])
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")

    vocabulary_words = sorted(
        {
            word
            for example in examples
            for word in (example.get("listed_word", ""), example.get("twin_word", ""))
            if word
        }
    )
    candidate_ids = sorted(
        {
            token_id
            for word in vocabulary_words
            for token_id in word_token_ids(tokenizer, word)[:1]
        }
    )
    tokenization = {
        word: {
            "token_ids": word_token_ids(tokenizer, word),
            "tokens": [
                tokenizer.decode([token_id]) for token_id in word_token_ids(tokenizer, word)
            ],
            "single_token": len(word_token_ids(tokenizer, word)) == 1,
        }
        for word in vocabulary_words
    }

    rows: list[dict[str, Any]] = []
    for mode_config in config["attention_modes"]:
        mode = str(mode_config["id"])
        freeze_attention = bool(mode_config["freeze_attention"])
        for index, example in enumerate(examples, start=1):
            prompt = f"{config['demonstration']} {example['sentence']}"
            position = len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1
            interventions, activation_rows = build_amplify_interventions(
                model,
                prompt,
                position,
                features,
                float(config["amplify_factor"]),
            )

            free_off_raw = greedy_generate(
                model,
                prompt,
                [],
                freeze_attention=freeze_attention,
                max_new_tokens=int(config["max_new_tokens"]),
                top_k=int(config["top_k"]),
            )
            free_on_raw = greedy_generate(
                model,
                prompt,
                interventions,
                freeze_attention=freeze_attention,
                max_new_tokens=int(config["max_new_tokens"]),
                top_k=int(config["top_k"]),
            )

            posthoc_words = [
                word
                for word in (free_off_raw["word"], free_on_raw["word"])
                if word
            ]
            specified_words = [
                word
                for word in (example.get("listed_word", ""), example.get("twin_word", ""))
                if word
            ]
            words = list(dict.fromkeys(specified_words + posthoc_words))
            target_ids = sorted(
                {
                    token_id
                    for word in words
                    for token_id in word_token_ids(tokenizer, word)[:1]
                }
            )

            forced: dict[str, Any] = {}
            forced_logits: dict[str, dict[str, torch.Tensor]] = {}
            for article in ("a", "an"):
                forced[article] = {}
                forced_logits[article] = {}
                for intervention_name, active_interventions in (
                    ("off", []),
                    ("on", interventions),
                ):
                    cell, logits = forced_cell(
                        model,
                        prompt,
                        article,
                        active_interventions,
                        freeze_attention=freeze_attention,
                        words=words,
                        target_ids=target_ids,
                        top_k=int(config["top_k"]),
                    )
                    forced[article][intervention_name] = cell
                    forced_logits[article][intervention_name] = logits

            comparisons = {
                article: distribution_comparison(
                    tokenizer,
                    forced_logits[article]["on"],
                    forced_logits[article]["off"],
                    candidate_ids=candidate_ids,
                    top_k=int(config["candidate_top_k"]),
                )
                for article in ("a", "an")
            }

            selected_article = free_on_raw["article"]
            if selected_article in {"a", "an"}:
                comparisons["s1_selected"] = comparisons[selected_article]
                selected_off_logits = forced_logits[selected_article]["off"]
                selected_on_logits = forced_logits[selected_article]["on"]
                free_noun_logits = (
                    free_on_raw["_step_logits"][1]
                    if len(free_on_raw["_step_logits"]) > 1
                    else None
                )
                reproduction = {
                    "selected_article": selected_article,
                    "s1_free_word": free_on_raw["word"],
                    "article_only_word": tokenizer.decode(
                        [int(torch.argmax(selected_off_logits).item())]
                    )
                    .strip()
                    .lower(),
                    "s1_forced_word": tokenizer.decode(
                        [int(torch.argmax(selected_on_logits).item())]
                    )
                    .strip()
                    .lower(),
                    "top1_match": int(torch.argmax(selected_off_logits).item())
                    == int(torch.argmax(selected_on_logits).item()),
                    "same_prefix_s1_on_off": comparisons[selected_article],
                    "free_on_vs_forced_on_max_abs_logit": (
                        float(torch.max(torch.abs(free_noun_logits - selected_on_logits)))
                        if free_noun_logits is not None
                        else None
                    ),
                }
                baseline_article = free_off_raw["article"]
                if baseline_article in {"a", "an"}:
                    total_comparison = distribution_comparison(
                        tokenizer,
                        selected_on_logits,
                        forced_logits[baseline_article]["off"],
                        candidate_ids=candidate_ids,
                        top_k=int(config["candidate_top_k"]),
                    )
                    article_only_comparison = distribution_comparison(
                        tokenizer,
                        selected_off_logits,
                        forced_logits[baseline_article]["off"],
                        candidate_ids=candidate_ids,
                        top_k=int(config["candidate_top_k"]),
                    )
                    residual_comparison = comparisons[selected_article]

                    source_word = free_off_raw["word"]
                    target_word = free_on_raw["word"]
                    source_ids = word_token_ids(tokenizer, source_word)
                    target_ids_posthoc = word_token_ids(tokenizer, target_word)
                    if source_ids and target_ids_posthoc:
                        source_id = source_ids[0]
                        target_id = target_ids_posthoc[0]

                        def contrast(logits: torch.Tensor) -> float:
                            return float(logits[target_id] - logits[source_id])

                        baseline_contrast = contrast(
                            forced_logits[baseline_article]["off"]
                        )
                        article_only_contrast = contrast(selected_off_logits)
                        s1_selected_contrast = contrast(selected_on_logits)
                    else:
                        baseline_contrast = 0.0
                        article_only_contrast = 0.0
                        s1_selected_contrast = 0.0

                    total_tv = float(total_comparison["tv_full_vocab"])
                    decomposition = {
                        "baseline_article": baseline_article,
                        "s1_selected_article": selected_article,
                        "posthoc_source_word": source_word,
                        "posthoc_target_word": target_word,
                        "total_js_full_vocab": total_comparison["js_full_vocab"],
                        "article_only_js_full_vocab": article_only_comparison[
                            "js_full_vocab"
                        ],
                        "residual_js_full_vocab": residual_comparison[
                            "js_full_vocab"
                        ],
                        "total_tv_full_vocab": total_tv,
                        "article_only_tv_full_vocab": article_only_comparison[
                            "tv_full_vocab"
                        ],
                        "residual_tv_full_vocab": residual_comparison[
                            "tv_full_vocab"
                        ],
                        "residual_tv_over_total_tv": (
                            float(residual_comparison["tv_full_vocab"]) / total_tv
                            if total_tv > 0
                            else 0.0
                        ),
                        "posthoc_baseline_contrast": baseline_contrast,
                        "posthoc_article_only_contrast": article_only_contrast,
                        "posthoc_s1_selected_contrast": s1_selected_contrast,
                        "posthoc_total_delta_delta": s1_selected_contrast
                        - baseline_contrast,
                        "posthoc_article_only_delta_delta": article_only_contrast
                        - baseline_contrast,
                        "posthoc_residual_delta_delta": s1_selected_contrast
                        - article_only_contrast,
                    }
                else:
                    decomposition = None
            else:
                comparisons["s1_selected"] = None
                reproduction = None
                decomposition = None

            twin_effects = None
            listed_word = str(example.get("listed_word", ""))
            twin_word = str(example.get("twin_word", ""))
            if listed_word and twin_word:
                listed_id = word_token_ids(tokenizer, listed_word)[0]
                twin_id = word_token_ids(tokenizer, twin_word)[0]
                twin_effects = {}
                for article in ("a", "an"):
                    off_logits = forced_logits[article]["off"]
                    on_logits = forced_logits[article]["on"]
                    first_token_off = float(off_logits[twin_id] - off_logits[listed_id])
                    first_token_on = float(on_logits[twin_id] - on_logits[listed_id])
                    off_sequences = forced[article]["off"]["word_sequences"]
                    on_sequences = forced[article]["on"]["word_sequences"]
                    sequence_odds_off = (
                        off_sequences[twin_word]["logprob_sum"]
                        - off_sequences[listed_word]["logprob_sum"]
                    )
                    sequence_odds_on = (
                        on_sequences[twin_word]["logprob_sum"]
                        - on_sequences[listed_word]["logprob_sum"]
                    )
                    twin_effects[article] = {
                        "target": twin_word,
                        "source": listed_word,
                        "target_minus_source_off": first_token_off,
                        "target_minus_source_on": first_token_on,
                        "delta_delta_target_minus_source": first_token_on
                        - first_token_off,
                        "sequence_log_odds_off": sequence_odds_off,
                        "sequence_log_odds_on": sequence_odds_on,
                        "delta_sequence_log_odds": sequence_odds_on
                        - sequence_odds_off,
                    }

            row = {
                "index": index,
                "attention_mode": mode,
                "freeze_attention": freeze_attention,
                "sentence": example["sentence"],
                "listed_word": listed_word,
                "twin_word": twin_word,
                "expected_article": example["expected_article"],
                "position": position,
                "feature_activations": activation_rows,
                "free": {
                    "off": strip_private_tensors(free_off_raw),
                    "on": strip_private_tensors(free_on_raw),
                },
                "forced": forced,
                "comparisons": comparisons,
                "article_only_reproduction": reproduction,
                "decomposition": decomposition,
                "pre_specified_twin_effects": twin_effects,
                "descriptive_switch_subset": free_off_raw["word"] != free_on_raw["word"],
            }
            rows.append(row)
            logging.info(
                "%s %d/%d off=%r on=%r selected=%s JS=%.6f",
                mode,
                index,
                len(examples),
                free_off_raw["continuation"],
                free_on_raw["continuation"],
                selected_article,
                (
                    comparisons["s1_selected"]["js_full_vocab"]
                    if comparisons["s1_selected"] is not None
                    else float("nan")
                ),
            )

    analysis = summarize_rows(rows, config)
    summary = {
        "experiment": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - started,
        "model": config["model"],
        "transcoder_set": config["transcoder_set"],
        "feature_set": config["feature_set"],
        "features": features,
        "amplify_factor": config["amplify_factor"],
        "n_pre_specified_prompts": len(examples),
        "analysis": analysis,
        "interpretation_status": "pending_result_review",
    }
    write_json(RESULTS_DIR / "rows.json", rows)
    write_json(RESULTS_DIR / "summary.json", summary)
    write_json(
        RESULTS_DIR / "tokenization.json",
        {
            "articles": {"a": a_id, "an": an_id},
            "words": tokenization,
            "candidate_first_token_ids": candidate_ids,
        },
    )
    write_report(summary, RESULTS_DIR / "report.md")
    logging.info("Done in %.1fs", summary["elapsed_sec"])


if __name__ == "__main__":
    main()
