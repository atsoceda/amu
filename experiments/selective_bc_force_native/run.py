#!/usr/bin/env python3
"""Selective b-step intervention with force-native article, then c with intervention off.

Tests editable-wrapper (modular C→B) vs packager under the protocol:
  1) intervene only while predicting article b
  2) force the native baseline article token
  3) generate content c with interventions off

Feature sets (from E1): S1 dual-effect, S2 article-only ≈ F_B, S3 content-only ≈ F_C.
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
    activation_at,
    article_and_word,
    build_amplify_interventions,
    build_zero_interventions,
    choose_control_features,
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


def mean_set_activation(
    model,
    prompt: str,
    position: int,
    features: list[dict[str, Any]],
    interventions: list[dict[str, Any]] | None = None,
) -> float:
    """Mean activation of a feature set at position, optionally under interventions."""
    if not features:
        return 0.0
    if not interventions:
        vals = [
            activation_at(
                model, prompt, int(f["layer"]), position, int(f["feature_idx"])
            )
            for f in features
        ]
        return float(sum(vals) / len(vals))
    tuples = [
        (
            int(item["layer"]),
            int(item["pos"]),
            int(item["feature_idx"]),
            float(item["value"]),
        )
        for item in interventions
    ]
    _, activations = model.feature_intervention(
        prompt,
        interventions=tuples,
        freeze_attention=True,
        sparse=True,
        return_activations=True,
    )
    if activations is None:
        return 0.0
    vals = []
    for f in features:
        vals.append(
            float(
                activations[int(f["layer"]), position, int(f["feature_idx"])]
                .detach()
                .float()
                .cpu()
            )
        )
    return float(sum(vals) / len(vals))


def first_token_id(model, prompt: str, interventions: list[dict[str, Any]]) -> int:
    tuples = [
        (
            int(item["layer"]),
            int(item["pos"]),
            int(item["feature_idx"]),
            float(item["value"]),
        )
        for item in interventions
    ]
    logits, _ = model.feature_intervention(
        prompt,
        interventions=tuples,
        freeze_attention=True,
        sparse=True,
        return_activations=False,
    )
    return int(torch.argmax(logits[0, -1]).item())


def generate_force_native_then_off(
    model,
    prompt: str,
    *,
    native_article_id: int,
    b_step_interventions: list[dict[str, Any]],
    max_new_tokens: int,
) -> dict[str, Any]:
    """Intervene only for diagnostics at b; force native article; continue with off."""
    # b-step under intervention (logits already measured elsewhere; still run for parity)
    _ = first_token_id(model, prompt, b_step_interventions)
    article_text = model.tokenizer.decode([native_article_id])
    current = prompt + article_text
    generated_ids = [native_article_id]
    for _ in range(max(0, max_new_tokens - 1)):
        logits, _ = model.feature_intervention(
            current,
            interventions=[],
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
    continuation = model.tokenizer.decode(generated_ids)
    return {
        "continuation": continuation,
        "forced_article_id": native_article_id,
        "forced_article_text": article_text,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1

    def rate(key: str) -> float:
        return sum(1 for r in rows if r.get(key)) / n

    def mean(key: str) -> float:
        return sum(float(r.get(key, 0.0)) for r in rows) / n

    return {
        "n": len(rows),
        "mean_delta_an_minus_a": mean("delta_an_minus_a"),
        "preferred_an_rate": rate("preferred_an"),
        "preferred_article_changed_rate": rate("preferred_article_changed"),
        "content_preserved_force_rate": rate("content_preserved_force"),
        "content_preserved_free_rate": rate("content_preserved_free"),
        "class_shifted_free_rate": rate("class_shifted_free"),
        "trajectory_like_free_rate": rate("trajectory_like_free"),
        "wrapper_logit_force_rate": rate("wrapper_logit_force"),
        "illicit_free_rate": rate("illicit_free"),
        "mean_act_S2_delta": mean("act_S2_delta"),
        "mean_act_S3_delta": mean("act_S3_delta"),
        "mean_act_S1_delta": mean("act_S1_delta"),
    }


def evaluate_condition(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    condition_name: str,
    features: list[dict[str, Any]] | None,
    op: str | None,
    readout_sets: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    a_id = token_id_for_text(tokenizer, " a")
    an_id = token_id_for_text(tokenizer, " an")
    target_ids = [a_id, an_id]
    factor = float(config["amplify_factor"])
    rows: list[dict[str, Any]] = []

    for index, example in enumerate(examples, start=1):
        prompt = f"{config['demonstration']} {example['sentence']}"
        position = len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1

        baseline = logits_for_prompt(
            model, prompt, target_ids, top_k=5, return_activations=False
        )
        baseline_continuation = generate_with_interventions(
            model, prompt, [], max_new_tokens=int(config["max_new_tokens"])
        )
        baseline_article, baseline_word = article_and_word(baseline_continuation)
        native_article_id = first_token_id(model, prompt, [])
        # Prefer decoded article token ids when baseline starts with a/an
        if baseline_article == "a":
            native_article_id = a_id
        elif baseline_article == "an":
            native_article_id = an_id

        interventions: list[dict[str, Any]] = []
        if features and op == "amplify":
            interventions, _ = build_amplify_interventions(
                model, prompt, position, features, factor
            )
        elif features and op == "zero":
            interventions = build_zero_interventions(features, position)

        if interventions:
            intervened = dict_intervention_result(
                model, prompt, interventions, target_ids, baseline
            )
            delta_a = intervened["targets"][str(a_id)]["delta_logit"]
            delta_an = intervened["targets"][str(an_id)]["delta_logit"]
            logit_a = intervened["targets"][str(a_id)]["logit"]
            logit_an = intervened["targets"][str(an_id)]["logit"]
        else:
            delta_a = 0.0
            delta_an = 0.0
            logit_a = baseline["targets"][str(a_id)]["logit"]
            logit_an = baseline["targets"][str(an_id)]["logit"]

        preferred_an = logit_an > logit_a
        baseline_preferred_an = (
            baseline["targets"][str(an_id)]["logit"]
            > baseline["targets"][str(a_id)]["logit"]
        )
        preferred_article = "an" if preferred_an else "a"
        baseline_preferred_article = "an" if baseline_preferred_an else "a"

        # Latent readouts at b-step
        act_base = {
            name: mean_set_activation(model, prompt, position, feats, None)
            for name, feats in readout_sets.items()
        }
        act_int = {
            name: mean_set_activation(
                model, prompt, position, feats, interventions or None
            )
            for name, feats in readout_sets.items()
        }

        force_out = generate_force_native_then_off(
            model,
            prompt,
            native_article_id=native_article_id,
            b_step_interventions=interventions,
            max_new_tokens=int(config["max_new_tokens"]),
        )
        force_article, force_word = article_and_word(force_out["continuation"])

        free_continuation = generate_with_interventions(
            model,
            prompt,
            interventions,
            max_new_tokens=int(config["max_new_tokens"]),
        )
        free_article, free_word = article_and_word(free_continuation)

        content_preserved_force = bool(baseline_word) and force_word == baseline_word
        content_preserved_free = bool(baseline_word) and free_word == baseline_word
        class_shifted_free = (
            bool(baseline_word)
            and bool(free_word)
            and vowel_initial(baseline_word) != vowel_initial(free_word)
        )
        article_moved_toward_an = (delta_an - delta_a) > 0
        preferred_article_changed = preferred_article != baseline_preferred_article
        wrapper_logit_force = (
            article_moved_toward_an
            and content_preserved_force
            and baseline_preferred_article == "a"
            and preferred_an
        )
        trajectory_like_free = (
            article_moved_toward_an and class_shifted_free and not content_preserved_free
        )
        illicit_free = False
        if free_article == "an" and free_word:
            illicit_free = not vowel_initial(free_word)
        elif free_article == "a" and free_word:
            illicit_free = vowel_initial(free_word)

        rows.append(
            {
                "index": index,
                "condition": condition_name,
                "op": op or "none",
                "target_prompt": example["sentence"],
                "listed_word": example["listed_word"],
                "expected_article": example["expected_article"],
                "twin_word": example.get("twin_word", ""),
                "baseline_continuation": baseline_continuation,
                "baseline_article": baseline_article,
                "baseline_word": baseline_word,
                "baseline_preferred_article": baseline_preferred_article,
                "delta_a": delta_a,
                "delta_an": delta_an,
                "delta_an_minus_a": delta_an - delta_a,
                "preferred_article": preferred_article,
                "preferred_an": preferred_an,
                "preferred_article_changed": preferred_article_changed,
                "force_continuation": force_out["continuation"],
                "force_article": force_article,
                "force_word": force_word,
                "content_preserved_force": content_preserved_force,
                "free_continuation": free_continuation,
                "free_article": free_article,
                "free_word": free_word,
                "content_preserved_free": content_preserved_free,
                "class_shifted_free": class_shifted_free,
                "trajectory_like_free": trajectory_like_free,
                "wrapper_logit_force": wrapper_logit_force,
                "illicit_free": illicit_free,
                "article_moved_toward_an": article_moved_toward_an,
                "act_S1_base": act_base.get("S1_dual_effect", 0.0),
                "act_S2_base": act_base.get("S2_article_only", 0.0),
                "act_S3_base": act_base.get("S3_content_only", 0.0),
                "act_S1_int": act_int.get("S1_dual_effect", 0.0),
                "act_S2_int": act_int.get("S2_article_only", 0.0),
                "act_S3_int": act_int.get("S3_content_only", 0.0),
                "act_S1_delta": act_int.get("S1_dual_effect", 0.0)
                - act_base.get("S1_dual_effect", 0.0),
                "act_S2_delta": act_int.get("S2_article_only", 0.0)
                - act_base.get("S2_article_only", 0.0),
                "act_S3_delta": act_int.get("S3_content_only", 0.0)
                - act_base.get("S3_content_only", 0.0),
            }
        )
        logging.info(
            "%s %d/%d %s force=%r free=%r Δ(an-a)=%.3f",
            condition_name,
            index,
            len(examples),
            example["sentence"][:48],
            force_out["continuation"],
            free_continuation,
            delta_an - delta_a,
        )
    return rows


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Selective b-step intervention with force-native article",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        f"Amplify factor: {summary['amplify_factor']}",
        f"Runtime seconds: {summary['runtime_seconds']:.1f}",
        "",
        "## Protocol",
        "",
        "1. Intervene only on the forward pass that scores article `b`.",
        "2. Force the native baseline article token into the string.",
        "3. Generate content `c` with interventions **off**.",
        "4. Companion: free generation with intervention left on (packager check).",
        "",
        "Feature mapping: `S3` ≈ content concept \(C\); `S2` ≈ article/licensing \(B\); "
        "`S1` = dual-effect (Latent-Planning-style joint set). No pure \(A\) set.",
        "",
        "## Condition summaries",
        "",
        "| Condition | Δ(an−a) | Pref. article changed | Content preserved (force) | Content preserved (free) | Trajectory-like (free) | Wrapper-logit+force | Illicit (free) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in summary["conditions"].items():
        s = block["summary"]
        lines.append(
            f"| {name} | {s['mean_delta_an_minus_a']:.3f} | "
            f"{s['preferred_article_changed_rate']:.2f} | "
            f"{s['content_preserved_force_rate']:.2f} | "
            f"{s['content_preserved_free_rate']:.2f} | "
            f"{s['trajectory_like_free_rate']:.2f} | "
            f"{s['wrapper_logit_force_rate']:.2f} | "
            f"{s['illicit_free_rate']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
            "",
            "## Example rows (S1 amplify)",
            "",
        ]
    )
    s1_rows = [
        r
        for r in summary.get("example_rows", [])
        if r.get("condition") == "S1_dual_effect_amplify"
    ][:8]
    for r in s1_rows:
        lines.append(
            f"- `{r['target_prompt']}` baseline `{r['baseline_continuation'].strip()}` | "
            f"force `{r['force_continuation'].strip()}` | free `{r['free_continuation'].strip()}` | "
            f"Δ(an−a)={r['delta_an_minus_a']:.3f}"
        )
    path.write_text("\n".join(lines) + "\n")


def interpret(summaries: dict[str, dict[str, Any]]) -> str:
    s1a = summaries.get("S1_dual_effect_amplify", {})
    s3a = summaries.get("S3_content_only_amplify", {})
    s3z = summaries.get("S3_content_only_zero", {})
    s2a = summaries.get("S2_article_only_amplify", {})
    parts = []
    # Crit 2-ish: C set moves only B, content preserved under force
    for label, block in [("S3 amplify", s3a), ("S3 zero", s3z)]:
        if not block:
            continue
        parts.append(
            f"{label}: Δ(an−a)={block['mean_delta_an_minus_a']:.3f}, "
            f"force content preserve={block['content_preserved_force_rate']:.2f}, "
            f"free trajectory={block['trajectory_like_free_rate']:.2f}, "
            f"wrapper_logit_force={block['wrapper_logit_force_rate']:.2f}."
        )
    if s1a:
        parts.append(
            f"S1 amplify: free trajectory={s1a['trajectory_like_free_rate']:.2f}, "
            f"force content preserve={s1a['content_preserved_force_rate']:.2f}, "
            f"wrapper_logit_force={s1a['wrapper_logit_force_rate']:.2f}, "
            f"illicit free={s1a['illicit_free_rate']:.2f}."
        )
    if s2a:
        parts.append(
            f"S2 amplify: Δ(an−a)={s2a['mean_delta_an_minus_a']:.3f}, "
            f"force preserve={s2a['content_preserved_force_rate']:.2f}."
        )

    wrapper_alive = False
    if s3a and s3a["wrapper_logit_force_rate"] >= 0.25 and s3a["mean_delta_an_minus_a"] > 0.2:
        wrapper_alive = True
    if s1a and s1a["wrapper_logit_force_rate"] >= 0.25 and s1a["illicit_free_rate"] >= 0.15:
        wrapper_alive = True

    if wrapper_alive:
        verdict = (
            "Editable-wrapper signal detected under selective force-native protocol "
            "(see wrapper_logit_force / illicit rates)."
        )
    elif s1a and s1a["trajectory_like_free_rate"] >= 0.5 and s1a["content_preserved_force_rate"] >= 0.7:
        verdict = (
            "Packager-consistent: free generation class-switches, while forcing the "
            "native article restores baseline content. No clean modular C→B wrapper."
        )
    else:
        verdict = (
            "No clear editable-wrapper recovery; inspect condition table for weak C→B handles."
        )
    return " ".join(parts + [verdict])


def main() -> None:
    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    model = load_replacement_model(config)
    tokenizer = model.tokenizer

    readout_sets = {
        set_id: load_set_features(config, set_id) for set_id in config["feature_sets"]
    }
    examples = list(config["test_examples"])

    all_rows: list[dict[str, Any]] = []
    conditions: dict[str, Any] = {}

    # Baseline
    base_rows = evaluate_condition(
        model,
        tokenizer,
        examples,
        config,
        condition_name="baseline",
        features=None,
        op=None,
        readout_sets=readout_sets,
    )
    all_rows.extend(base_rows)
    conditions["baseline"] = {
        "features": [],
        "summary": summarize_rows(base_rows),
    }

    for set_id in config["feature_sets"]:
        feats = readout_sets[set_id]
        for op in config["ops"]:
            name = f"{set_id}_{op}"
            rows = evaluate_condition(
                model,
                tokenizer,
                examples,
                config,
                condition_name=name,
                features=feats,
                op=op,
                readout_sets=readout_sets,
            )
            all_rows.extend(rows)
            conditions[name] = {
                "feature_set": set_id,
                "op": op,
                "features": [
                    {"layer": f["layer"], "feature_idx": f["feature_idx"]} for f in feats
                ],
                "summary": summarize_rows(rows),
            }

    if config.get("run_controls", True):
        # One control amplify matched to S1 on first prompt's activation profile
        demo_prompt = f"{config['demonstration']} {examples[0]['sentence']}"
        pos = len(tokenizer(demo_prompt, add_special_tokens=True).input_ids) - 1
        control_feats = choose_control_features(
            model, demo_prompt, pos, readout_sets["S1_dual_effect"], config
        )
        for op in config["ops"]:
            name = f"control_{op}"
            rows = evaluate_condition(
                model,
                tokenizer,
                examples,
                config,
                condition_name=name,
                features=control_feats,
                op=op,
                readout_sets=readout_sets,
            )
            all_rows.extend(rows)
            conditions[name] = {
                "feature_set": "control",
                "op": op,
                "features": [
                    {"layer": f["layer"], "feature_idx": f["feature_idx"]}
                    for f in control_feats
                ],
                "summary": summarize_rows(rows),
            }

    summaries = {k: v["summary"] for k, v in conditions.items()}
    interpretation = interpret(summaries)
    summary = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "amplify_factor": config["amplify_factor"],
        "runtime_seconds": time.time() - started,
        "protocol": "intervene_at_b_only__force_native_b__generate_c_off",
        "concept_mapping": {
            "S3_content_only": "approx F_C (content)",
            "S2_article_only": "approx F_B (licensing/article)",
            "S1_dual_effect": "joint dual-effect (not pure F_A)",
        },
        "conditions": conditions,
        "interpretation": interpretation,
        "example_rows": [
            r for r in all_rows if r["condition"] == "S1_dual_effect_amplify"
        ],
    }
    write_json(RESULTS_DIR / "summary.json", summary)
    write_json(RESULTS_DIR / "rows.json", all_rows)
    write_report(summary, RESULTS_DIR / "report.md")
    logging.info("Done: %s", interpretation)
    print(interpretation)


if __name__ == "__main__":
    main()
