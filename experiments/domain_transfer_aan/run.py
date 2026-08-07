#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.lib.aan_protocol import (
    choose_control_features,
    evaluate_amplify_condition,
    summarize_condition,
    write_json,
)
from experiments.lib.core import load_replacement_model, setup_file_logging

EXP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXP_DIR / "config.json"
RESULTS_DIR = EXP_DIR / "results"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def resolve_set_ids(config: dict[str, Any]) -> list[str]:
    ids = list(config.get("feature_set_ids", ["S1_dual_effect"]))
    e3_path = (EXP_DIR / config.get("e3_summary_path", "")).resolve()
    if e3_path.exists():
        e3 = json.loads(e3_path.read_text())
        for key in ("article_set_id", "content_set_id"):
            sid = e3.get(key)
            if sid and sid not in ids:
                ids.append(sid)
    return ids


def load_features(config: dict[str, Any], set_id: str) -> list[dict[str, Any]]:
    selection = json.loads((EXP_DIR / config["e1_selection_path"]).resolve().read_text())
    feats = selection["sets"][set_id]["selected_features"]
    if not feats:
        raise RuntimeError(f"No features for {set_id}")
    return feats


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Domain Transfer a/an (E5)",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        "",
        summary["interpretation"],
        "",
        "| Set | Mean Δ(an−a) | Trajectory-like | Wrapper-like | Content preserved | vs control Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for set_id, block in summary["set_results"].items():
        s = block["summary"]
        c = block["control_summary"]
        lines.append(
            f"| {set_id} | {s['mean_delta_an_minus_a']:.3f} | "
            f"{s['trajectory_like_rate']:.3f} | {s['wrapper_like_rate']:.3f} | "
            f"{s['content_preserved_rate']:.3f} | "
            f"{s['mean_delta_an_minus_a'] - c['mean_delta_an_minus_a']:.3f} |"
        )
    lines.append("")
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    config = load_config()
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    set_ids = resolve_set_ids(config)
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    examples = config["test_examples"]
    first_prompt = f"{config['demonstration']} {examples[0]['sentence']}"
    first_pos = len(tokenizer(first_prompt, add_special_tokens=True).input_ids) - 1

    set_results: dict[str, Any] = {}
    for set_id in set_ids:
        features = load_features(config, set_id)
        control = choose_control_features(
            model, first_prompt, first_pos, features, config
        )
        rows = evaluate_amplify_condition(
            model, tokenizer, examples, features, config, f"{set_id}_amplify"
        )
        control_rows = evaluate_amplify_condition(
            model, tokenizer, examples, control, config, f"{set_id}_control"
        )
        set_results[set_id] = {
            "selected_features": features,
            "control_features": control,
            "summary": summarize_condition(rows),
            "control_summary": summarize_condition(control_rows),
            "examples": rows,
            "control_examples": control_rows,
        }

    # Locality check: transfer nonspecific if no set beats control
    any_specific = False
    for block in set_results.values():
        s, c = block["summary"], block["control_summary"]
        if abs(s["mean_delta_an_minus_a"]) > abs(c["mean_delta_an_minus_a"]) + 0.125 or (
            s["trajectory_like_rate"] > c["trajectory_like_rate"] + 0.1
            or s["wrapper_like_rate"] > c["wrapper_like_rate"] + 0.1
        ):
            any_specific = True
    interpretation = (
        "Occupation-selected features transfer above controls on non-occupation a/an prompts."
        if any_specific
        else "Transfer is nonspecific (control-like). Report locality; do not expand domains."
    )
    payload = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "runtime_seconds": time.time() - started,
        "set_ids": set_ids,
        "interpretation": interpretation,
        "transfer_specific": any_specific,
        "set_results": set_results,
    }
    write_json(RESULTS_DIR / "summary.json", payload)
    write_report(payload)
    logging.info("E5 complete: %s", interpretation)


if __name__ == "__main__":
    main()
