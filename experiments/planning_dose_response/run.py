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


def resolve_sets(config: dict[str, Any]) -> list[str]:
    summary_path = (EXP_DIR / config["e1_summary_path"]).resolve()
    sets = list(config.get("default_sets", ["S1_dual_effect"]))
    if summary_path.exists():
        e1 = json.loads(summary_path.read_text())
        for set_id in e1.get("e2_dose_sweep_sets", []):
            if set_id not in sets:
                sets.append(set_id)
        # Also pull any set with wrapper_like >= threshold
        thr = float(config.get("wrapper_like_threshold", 0.25))
        for set_id, block in e1.get("set_results", {}).items():
            if block.get("summary", {}).get("wrapper_like_rate", 0) >= thr:
                if set_id not in sets:
                    sets.append(set_id)
    return sets


def load_selected_features(config: dict[str, Any], set_id: str) -> list[dict[str, Any]]:
    selection = json.loads((EXP_DIR / config["e1_selection_path"]).resolve().read_text())
    block = selection["sets"][set_id]
    feats = block["selected_features"]
    if not feats:
        raise RuntimeError(f"No selected features for {set_id} in E1 selection.json")
    return feats


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Planning Dose–Response (E2)",
        "",
        f"Generated: {summary['generated_at']}",
        f"Model: `{summary['model']}`",
        "",
        "## Question",
        "",
        "Is there a dose window for content-preserving article movement, or does gain scale as package switching?",
        "",
        f"Sets swept: {', '.join(summary['sets_swept'])}",
        f"Factors: {summary['amplify_factors']}",
        "",
        "## Results",
        "",
    ]
    for set_id, block in summary["set_results"].items():
        lines.append(f"### {set_id}")
        lines.append("")
        lines.append(
            "| Factor | Mean Δ(an−a) | Wrapper-like | Trajectory-like | Content preserved | Class shifted | Control Δ |"
        )
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for factor_key, row in block["by_factor"].items():
            s = row["summary"]
            c = row["control_summary"]
            lines.append(
                f"| {factor_key} | {s['mean_delta_an_minus_a']:.3f} | "
                f"{s['wrapper_like_rate']:.3f} | {s['trajectory_like_rate']:.3f} | "
                f"{s['content_preserved_rate']:.3f} | {s['class_shifted_rate']:.3f} | "
                f"{c['mean_delta_an_minus_a']:.3f} |"
            )
        lines.append("")
        lines.append(f"Dose interpretation: {block['interpretation']}")
        lines.append("")
    lines += ["## Overall", "", summary["interpretation"], ""]
    (RESULTS_DIR / "report.md").write_text("\n".join(lines))


def interpret_dose(by_factor: dict[str, Any]) -> str:
    factors = sorted(by_factor.keys(), key=float)
    traj = [by_factor[f]["summary"]["trajectory_like_rate"] for f in factors]
    wrap = [by_factor[f]["summary"]["wrapper_like_rate"] for f in factors]
    if any(w >= 0.25 for w in wrap) and max(wrap) >= max(traj):
        return "Possible wrapper-like dose window; inspect peak wrapper factor."
    if traj == sorted(traj) and traj[-1] >= 0.15:
        return "Monotone (non-decreasing) trajectory-like rate with dose; no wrapper window."
    if max(traj) >= 0.15 and max(wrap) < 0.15:
        return "Trajectory-like effects dominate across doses; no wrapper window."
    return "Mixed dose pattern; see per-factor table."


def main() -> None:
    config = load_config()
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    set_ids = resolve_sets(config)
    logging.info("E2 dose-sweep sets: %s", set_ids)
    model = load_replacement_model(config)
    tokenizer = model.tokenizer
    first_prompt = (
        f"{config['demonstration']} {config['test_examples'][0]['sentence']}"
    )
    first_pos = len(tokenizer(first_prompt, add_special_tokens=True).input_ids) - 1

    set_results: dict[str, Any] = {}
    for set_id in set_ids:
        features = load_selected_features(config, set_id)
        control_features = choose_control_features(
            model, first_prompt, first_pos, features, config
        )
        by_factor: dict[str, Any] = {}
        for factor in config["amplify_factors"]:
            rows = evaluate_amplify_condition(
                model,
                tokenizer,
                config["test_examples"],
                features,
                config,
                f"{set_id}_x{factor}",
                amplify_factor=float(factor),
            )
            control_rows = evaluate_amplify_condition(
                model,
                tokenizer,
                config["test_examples"],
                control_features,
                config,
                f"{set_id}_control_x{factor}",
                amplify_factor=float(factor),
            )
            by_factor[str(factor)] = {
                "summary": summarize_condition(rows),
                "control_summary": summarize_condition(control_rows),
                "examples": rows,
                "control_examples": control_rows,
            }
            logging.info(
                "%s factor=%s traj=%.3f wrap=%.3f",
                set_id,
                factor,
                by_factor[str(factor)]["summary"]["trajectory_like_rate"],
                by_factor[str(factor)]["summary"]["wrapper_like_rate"],
            )
        set_results[set_id] = {
            "selected_features": features,
            "control_features": control_features,
            "by_factor": by_factor,
            "interpretation": interpret_dose(by_factor),
        }

    # Recommend dose for E4: peak twin match among factors for S1 if present
    recommended_factor = float(config["amplify_factors"][config["amplify_factors"].index(5.0)] if 5.0 in config["amplify_factors"] else config["amplify_factors"][-1])
    if "S1_dual_effect" in set_results:
        best = None
        for factor, row in set_results["S1_dual_effect"]["by_factor"].items():
            twin = row["summary"]["matched_twin_rate"]
            traj = row["summary"]["trajectory_like_rate"]
            score = (twin, traj)
            if best is None or score > best[0]:
                best = (score, float(factor))
        if best is not None:
            recommended_factor = best[1]

    interpretation = (
        "No wrapper dose window observed across swept sets; trajectory-like scaling dominates."
        if all(
            "wrapper" not in block["interpretation"].lower()
            or "no wrapper" in block["interpretation"].lower()
            for block in set_results.values()
        )
        else "At least one set shows a possible wrapper-like dose window; carry into E3."
    )
    # refine
    if any("Possible wrapper" in b["interpretation"] for b in set_results.values()):
        interpretation = (
            "At least one set shows a possible wrapper-like dose window; carry into E3."
        )
    elif all(
        "Monotone" in b["interpretation"] or "Trajectory-like" in b["interpretation"]
        for b in set_results.values()
    ):
        interpretation = (
            "Dose sweeps favor package switching without a content-preserving wrapper window."
        )

    payload = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "runtime_seconds": time.time() - started,
        "sets_swept": set_ids,
        "amplify_factors": config["amplify_factors"],
        "recommended_e4_amplify_factor": recommended_factor,
        "interpretation": interpretation,
        "set_results": set_results,
    }
    write_json(RESULTS_DIR / "summary.json", payload)
    write_report(payload)
    logging.info("E2 complete: %s", interpretation)


if __name__ == "__main__":
    main()
