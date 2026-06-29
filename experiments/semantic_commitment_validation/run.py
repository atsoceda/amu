#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.core import (
    dict_intervention_result,
    load_replacement_model,
    logits_for_prompt,
    setup_file_logging,
    token_id_for_text,
)

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"


def setup_logging() -> None:
    setup_file_logging(RESULTS_DIR)


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def resolve_exp_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (EXP_DIR / path).resolve()


def logits_snapshot(model, prompt: str, target_ids: list[int]) -> dict[str, Any]:
    return logits_for_prompt(model, prompt, target_ids, top_k=12, return_activations=False)


def intervention_snapshot(
    model,
    prompt: str,
    interventions: list[dict[str, Any]],
    target_ids: list[int],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    return dict_intervention_result(model, prompt, interventions, target_ids, baseline)


def select_pilot_interventions(pilot_summary: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    by_name = {item["name"]: item for item in pilot_summary["interventions"]}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise KeyError(f"Pilot summary missing interventions: {missing}")
    return [by_name[name] for name in names]


def prompt_score(
    result: dict[str, Any],
    semantic_id: int,
    competitor_ids: list[int],
    fallback_ids: list[int],
) -> dict[str, Any]:
    targets = result["targets"]
    semantic = targets[str(semantic_id)]
    competitor_delta = max(targets[str(tid)]["delta_logit"] for tid in competitor_ids)
    fallback_delta = max(targets[str(tid)]["delta_logit"] for tid in fallback_ids)
    return {
        "semantic_delta_logit": semantic["delta_logit"],
        "semantic_delta_rank": semantic["delta_rank"],
        "max_competitor_delta_logit": competitor_delta,
        "max_fallback_delta_logit": fallback_delta,
        "specificity_margin": semantic["delta_logit"] - max(competitor_delta, fallback_delta),
    }


def summarize_by_family(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    families = sorted({row["prompt_family"] for row in rows})
    for family in families:
        family_rows = [row for row in rows if row["prompt_family"] == family]
        out[family] = {
            "n": len(family_rows),
            "mean_semantic_delta_logit": mean(r["score"]["semantic_delta_logit"] for r in family_rows),
            "mean_specificity_margin": mean(r["score"]["specificity_margin"] for r in family_rows),
            "positive_semantic_count": sum(r["score"]["semantic_delta_logit"] > 0 for r in family_rows),
            "positive_specificity_count": sum(r["score"]["specificity_margin"] > 0 for r in family_rows),
        }
    return out


def write_report(summary: dict[str, Any]) -> None:
    semantic_id = str(summary["token_ids"]["semantic_target"])
    lines = [
        "# Semantic Commitment Validation Report",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['config']['model']}`",
        f"Pilot interventions reused: {', '.join(summary['config']['selected_pilot_interventions'])}",
        "",
        "## Verdict",
        "",
        summary["verdict"],
        "",
        "## Family Summary",
        "",
    ]
    for family, item in summary["family_summary"].items():
        lines.append(
            f"- `{family}` n={item['n']}; "
            f"mean Paris delta_logit={item['mean_semantic_delta_logit']:.3f}; "
            f"mean specificity_margin={item['mean_specificity_margin']:.3f}; "
            f"positive Paris={item['positive_semantic_count']}/{item['n']}; "
            f"positive specific={item['positive_specificity_count']}/{item['n']}"
        )

    lines += ["", "## Strongest Prompt-Level Effects", ""]
    ranked = sorted(
        summary["rows"],
        key=lambda row: row["score"]["semantic_delta_logit"],
        reverse=True,
    )
    for row in ranked[:16]:
        score = row["score"]
        lines.append(
            f"- `{row['intervention_name']}` on `{row['prompt_id']}` "
            f"({row['prompt_family']}): Paris delta_logit={score['semantic_delta_logit']:.3f}, "
            f"rank_delta={score['semantic_delta_rank']}, "
            f"specificity_margin={score['specificity_margin']:.3f}"
        )

    lines += ["", "## Baseline Snapshots", ""]
    for prompt in summary["config"]["prompts"]:
        base = summary["baselines"][prompt["id"]]
        paris = base["targets"][semantic_id]
        top = ", ".join(f"`{item['token']}`" for item in base["top_tokens"][:5])
        lines.append(
            f"- `{prompt['id']}`: top5 {top}; "
            f"`Paris` rank={paris['rank']} prob={paris['prob']:.6f}"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "A useful validation result would show positive `Paris` movement on near-match and France prompts while remaining weak or negative on semantic competitors and syntax/domain controls. Positive movement everywhere is evidence for a broad perturbation rather than a clean semantic-commitment handle.",
        "",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def build_verdict(family_summary: dict[str, Any]) -> str:
    near = family_summary.get("near_match", {})
    france = family_summary.get("france_semantic", {})
    competitors = family_summary.get("semantic_competitor", {})
    syntax = family_summary.get("syntax_control", {})
    domain = family_summary.get("domain_control", {})

    support = (
        near.get("mean_semantic_delta_logit", 0.0) > 0.15
        and france.get("mean_semantic_delta_logit", 0.0) > 0.15
    )
    leakage = (
        competitors.get("mean_semantic_delta_logit", 0.0) > 0.15
        or syntax.get("mean_semantic_delta_logit", 0.0) > 0.15
        or domain.get("mean_semantic_delta_logit", 0.0) > 0.15
    )
    specific = (
        near.get("mean_specificity_margin", 0.0) > 0.0
        and france.get("mean_specificity_margin", 0.0) > 0.0
    )

    if support and specific and not leakage:
        return "The reused pilot interventions validate as relatively specific Paris/France semantic-commitment handles."
    if support and leakage:
        return "The reused pilot interventions still move `Paris`, but they leak into controls; treat them as broad causal handles, not clean semantic-commitment features."
    if support:
        return "The reused pilot interventions produce directional support, but specificity is mixed and needs tighter feature selection."
    return "The reused pilot interventions do not generalize reliably beyond the original pilot prompt."


def main() -> None:
    setup_logging()
    started = time.time()
    config = load_config()
    pilot_summary_path = resolve_exp_path(config["pilot_summary_path"])
    pilot_summary = json.loads(pilot_summary_path.read_text())

    logging.info("Loading model")
    model = load_replacement_model(config)
    tokenizer = model.tokenizer

    target_text_to_id = {}
    target_ids = []
    for text in config["target_token_texts"]:
        token_id = token_id_for_text(tokenizer, text)
        target_text_to_id[text] = token_id
        if token_id not in target_ids:
            target_ids.append(token_id)

    semantic_id = target_text_to_id[config["semantic_target_text"]]
    competitor_ids = [target_text_to_id[t] for t in config["competitor_target_texts"]]
    fallback_ids = [target_text_to_id[t] for t in config["fallback_target_texts"]]
    selected = select_pilot_interventions(pilot_summary, config["selected_pilot_interventions"])

    logging.info("Computing baselines")
    baselines = {}
    for prompt in config["prompts"]:
        baselines[prompt["id"]] = logits_snapshot(model, prompt["text"], target_ids)

    rows = []
    for intervention in selected:
        logging.info("Testing intervention %s", intervention["name"])
        for prompt in config["prompts"]:
            result = intervention_snapshot(
                model,
                prompt["text"],
                intervention["interventions"],
                target_ids,
                baselines[prompt["id"]],
            )
            rows.append(
                {
                    "intervention_name": intervention["name"],
                    "intervention_kind": intervention["kind"],
                    "prompt_id": prompt["id"],
                    "prompt_family": prompt["family"],
                    "prompt_text": prompt["text"],
                    "score": prompt_score(result, semantic_id, competitor_ids, fallback_ids),
                    "result": result,
                }
            )

    family_summary = summarize_by_family(rows)
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": time.time() - started,
        "config": config,
        "pilot_summary_path": str(pilot_summary_path),
        "target_text_to_id": target_text_to_id,
        "token_ids": {
            "semantic_target": semantic_id,
            "competitor_ids": competitor_ids,
            "fallback_ids": fallback_ids,
        },
        "selected_interventions": [
            {
                "name": item["name"],
                "kind": item["kind"],
                "feature_set": item["feature_set"],
                "interventions": item["interventions"],
            }
            for item in selected
        ],
        "baselines": baselines,
        "rows": rows,
        "family_summary": family_summary,
        "verdict": build_verdict(family_summary),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary)
    logging.info("Wrote %s", RESULTS_DIR / "summary.json")
    logging.info("Wrote %s", RESULTS_DIR / "report.md")


if __name__ == "__main__":
    main()
