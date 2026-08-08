#!/usr/bin/env python3
"""Experiment 1: residual / MLP-in direction patch under Stage XVI fixed-b.

Builds a dense content direction (hint_target − hint_source at mlp.hook_in)
and steers it at planning position P while pasting native b and predicting c.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from nnsight import save

from experiments.lib.aan_protocol import (
    article_and_word,
    first_content_token_text,
    vowel_initial,
    write_json,
)
from experiments.lib.core import (
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


def pre_article_pos(tokenizer, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=True).input_ids) - 1


def word_token_id(tokenizer, word: str) -> int:
    return token_id_for_text(tokenizer, first_content_token_text(tokenizer, word))


def legal_for_article(article: str, word: str) -> bool:
    if not word or article not in {"a", "an"}:
        return False
    return (article == "an") == vowel_initial(word)


def mlp_in_at(model, prompt: str, layer: int, position: int) -> torch.Tensor:
    """Return mlp-in residual vector at (layer, position), shape [d_model]."""
    with model.trace(prompt):
        loc = model.get_feature_input_loc(layer)
        act = save(loc.output)
    vec = act[0, position].detach().float().cpu()
    return vec


def build_content_direction(
    model,
    prompt: str,
    *,
    source_word: str,
    target_word: str,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Unit directions per layer: hint(target) − hint(source) at pre-article pos."""
    hint_source = f"Think of a {source_word}. {prompt}"
    hint_target = f"Think of a {target_word}. {prompt}"
    pos_s = pre_article_pos(model.tokenizer, hint_source)
    pos_t = pre_article_pos(model.tokenizer, hint_target)
    directions: dict[int, torch.Tensor] = {}
    for layer in layers:
        src = mlp_in_at(model, hint_source, layer, pos_s)
        tgt = mlp_in_at(model, hint_target, layer, pos_t)
        delta = tgt - src
        norm = float(delta.norm().item())
        if norm < 1e-8:
            directions[layer] = torch.zeros_like(delta)
        else:
            directions[layer] = delta / norm
    return directions


def build_random_direction(
    template: dict[int, torch.Tensor],
    seed: int,
) -> dict[int, torch.Tensor]:
    g = torch.Generator().manual_seed(int(seed))
    out: dict[int, torch.Tensor] = {}
    for layer, vec in template.items():
        rnd = torch.randn(vec.shape, generator=g, dtype=torch.float32)
        norm = float(rnd.norm().item())
        out[layer] = rnd / max(norm, 1e-8)
    return out


def apply_residual_patches(
    model,
    prompt: str,
    *,
    position: int,
    directions: dict[int, torch.Tensor],
    alpha: float,
) -> torch.Tensor:
    """Forward with mlp-in += alpha * direction at position; return logits [1,T,V]."""
    with model.trace(prompt):
        for layer, direction in directions.items():
            loc = model.get_feature_input_loc(int(layer))
            delta = (alpha * direction).to(device=loc.output.device, dtype=loc.output.dtype)
            loc.output[:, position, :] = loc.output[:, position, :] + delta
        logits = save(model.output.logits)
    return logits


def generate_force_native_then_c_residual(
    model,
    prompt: str,
    *,
    native_article_id: int,
    position: int,
    directions: dict[int, torch.Tensor] | None,
    alpha: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Paste native article, then generate with residual steering kept on at P."""
    article_text = model.tokenizer.decode([native_article_id])
    current = prompt + article_text
    generated_ids = [native_article_id]
    use_dirs = directions or {}
    for _ in range(max(0, max_new_tokens - 1)):
        if use_dirs and abs(alpha) > 0:
            logits = apply_residual_patches(
                model,
                current,
                position=position,
                directions=use_dirs,
                alpha=alpha,
            )
        else:
            with model.trace(current):
                logits = save(model.output.logits)
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


def noun_logits_residual(
    model,
    prompt_with_article: str,
    *,
    position: int,
    directions: dict[int, torch.Tensor] | None,
    alpha: float,
    token_ids: list[int],
) -> dict[int, float]:
    if directions and abs(alpha) > 0:
        logits = apply_residual_patches(
            model,
            prompt_with_article,
            position=position,
            directions=directions,
            alpha=alpha,
        )[0, -1].detach().float().cpu()
    else:
        with model.trace(prompt_with_article):
            logits_t = save(model.output.logits)
        logits = logits_t[0, -1].detach().float().cpu()
    return {tid: float(logits[tid]) for tid in token_ids}


def article_delta_residual(
    model,
    prompt: str,
    *,
    position: int,
    directions: dict[int, torch.Tensor] | None,
    alpha: float,
    a_id: int,
    an_id: int,
) -> float:
    base = logits_for_prompt(
        model, prompt, [a_id, an_id], top_k=5, return_activations=False
    )
    base_gap = float(base["targets"][str(an_id)]["logit"] - base["targets"][str(a_id)]["logit"])
    if not directions or abs(alpha) == 0:
        return 0.0
    logits = apply_residual_patches(
        model, prompt, position=position, directions=directions, alpha=alpha
    )[0, -1].detach().float().cpu()
    gap = float(logits[an_id] - logits[a_id])
    return gap - base_gap


def evaluate_row(
    model,
    family: dict[str, Any],
    config: dict[str, Any],
    *,
    condition_name: str,
    directions: dict[int, torch.Tensor] | None,
    alpha: float,
) -> dict[str, Any]:
    tokenizer = model.tokenizer
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

    # Free baseline (no residual patch)
    free_ids: list[int] = []
    current = prompt
    for _ in range(int(config["max_new_tokens"])):
        with model.trace(current):
            logits = save(model.output.logits)
        tid = int(torch.argmax(logits[0, -1]).item())
        free_ids.append(tid)
        piece = tokenizer.decode([tid])
        current += piece
        if piece.strip() in {".", "!", "?"}:
            break
    baseline_continuation = tokenizer.decode(free_ids)
    baseline_article, baseline_word = article_and_word(baseline_continuation)
    native_article = (
        baseline_article if baseline_article in {"a", "an"} else family["native_article"]
    )
    native_article_id = a_id if native_article == "a" else an_id

    delta_an_a = article_delta_residual(
        model,
        prompt,
        position=position,
        directions=directions,
        alpha=alpha,
        a_id=a_id,
        an_id=an_id,
    )

    prompt_plus_b = prompt + tokenizer.decode([native_article_id])
    noun_ids = [source_id, same_id, cross_id]
    logits_off = noun_logits_residual(
        model,
        prompt_plus_b,
        position=position,
        directions=None,
        alpha=0.0,
        token_ids=noun_ids,
    )
    logits_on = noun_logits_residual(
        model,
        prompt_plus_b,
        position=position,
        directions=directions,
        alpha=alpha,
        token_ids=noun_ids,
    )

    force_off = generate_force_native_then_c_residual(
        model,
        prompt,
        native_article_id=native_article_id,
        position=position,
        directions=None,
        alpha=0.0,
        max_new_tokens=int(config["max_new_tokens"]),
    )
    force_on = generate_force_native_then_c_residual(
        model,
        prompt,
        native_article_id=native_article_id,
        position=position,
        directions=directions,
        alpha=alpha,
        max_new_tokens=int(config["max_new_tokens"]),
    )

    # Free generation with residual steering from the start (package check)
    free_ids = []
    current = prompt
    for _ in range(int(config["max_new_tokens"])):
        if directions and abs(alpha) > 0:
            logits = apply_residual_patches(
                model,
                current,
                position=position,
                directions=directions,
                alpha=alpha,
            )
        else:
            with model.trace(current):
                logits = save(model.output.logits)
        tid = int(torch.argmax(logits[0, -1]).item())
        free_ids.append(tid)
        piece = tokenizer.decode([tid])
        current += piece
        if piece.strip() in {".", "!", "?"}:
            break
    free_continuation = tokenizer.decode(free_ids)

    _, off_word = article_and_word(force_off["continuation"])
    _, on_word = article_and_word(force_on["continuation"])
    free_article, free_word = article_and_word(free_continuation)

    content_changed_on = bool(baseline_word) and bool(on_word) and on_word != baseline_word
    within_class_on = legal_for_article(native_article, on_word)
    c_to_c = content_changed_on and within_class_on
    matched_same = on_word == same_word.lower()

    row = {
        "condition": condition_name,
        "alpha": alpha,
        "family_id": family["id"],
        "sentence": family["sentence"],
        "source_word": source_word,
        "same_class_word": same_word,
        "native_article_used": native_article,
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
        "delta_an_minus_a_bstep": delta_an_a,
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
        "%s %s on=%r off=%r free=%r c→c=%s Δsame-src=%.3f ΔΔ=%.3f matched=%s",
        condition_name,
        family["id"],
        force_on["continuation"].strip(),
        force_off["continuation"].strip(),
        free_continuation.strip(),
        c_to_c,
        row["delta_same_minus_source_on"],
        row["delta_same_minus_source_delta"],
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
    best_dd = -1e9
    for name, block in by.items():
        rate = float(block.get("c_to_c_signal_rate", 0.0))
        dd = float(block.get("mean_delta_same_minus_source_delta", 0.0))
        if rate > best_rate or (rate == best_rate and dd > best_dd):
            best_rate = rate
            best_dd = dd
            best_name = name
        parts.append(
            f"{name}: c→c={block['c_to_c_signal_rate']:.2f}, "
            f"match_same={block['matched_same_class_on_rate']:.2f}, "
            f"Δ(same−src)={block['mean_delta_same_minus_source_on']:.3f}, "
            f"ΔΔ(same−src)={block['mean_delta_same_minus_source_delta']:.3f}."
        )
    if best_rate >= 0.34:
        parts.append(
            f"Upset: `{best_name}` shows within-class C→c under fixed b via residual "
            "steering — sparse null was a dictionary-grain artifact."
        )
        dial = True
    elif best_dd >= 1.0 and best_rate == 0:
        parts.append(
            f"Soft/mixed: `{best_name}` moves same−source logits (ΔΔ≥1) without "
            "noun switches — dense dial is weak at execution."
        )
        dial = False
    else:
        parts.append(
            "Null: residual MLP-in content directions do not open a within-class "
            "C→c dial under fixed b. Fairer negative than sparse-only Stage XVI/XVII."
        )
        dial = False
    summary["dial_found"] = dial
    summary["best_condition"] = best_name
    return " ".join(parts)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Residual-direction fixed-b C→c (Experiment 1)",
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


def main() -> None:
    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(RESULTS_DIR)
    started = time.time()
    logging.info("Loading model for residual-direction Exp 1")
    model = load_replacement_model(config)
    layers = [int(x) for x in config["patch_layers"]]
    alphas = [float(x) for x in config["steering_alphas"]]

    directions_by_family: dict[str, dict[str, Any]] = {}
    for family in config["families"]:
        prompt = f"{config['demonstration']} {family['sentence']}"
        content = build_content_direction(
            model,
            prompt,
            source_word=family["source_word"],
            target_word=family["same_class_word"],
            layers=layers,
        )
        control = build_random_direction(content, int(config["control_seed"]) + hash(family["id"]) % 1000)
        directions_by_family[family["id"]] = {
            "content": content,
            "control": control,
            "content_norms": {
                str(layer): float(vec.norm().item()) for layer, vec in content.items()
            },
        }
        logging.info(
            "Built directions for %s layers=%s",
            family["id"],
            layers,
        )

    # Serialize direction metadata (not full vectors)
    write_json(
        RESULTS_DIR / "directions_meta.json",
        {
            fid: {
                "layers": layers,
                "content_unit_norms": block["content_norms"],
            }
            for fid, block in directions_by_family.items()
        },
    )

    all_rows: list[dict[str, Any]] = []
    by_condition: dict[str, Any] = {}

    # Baseline
    base_rows = [
        evaluate_row(
            model,
            family,
            config,
            condition_name="baseline",
            directions=None,
            alpha=0.0,
        )
        for family in config["families"]
    ]
    all_rows.extend(base_rows)
    by_condition["baseline"] = summarize(base_rows)

    for alpha in alphas:
        cond = f"content_steer_a{alpha:g}"
        rows = [
            evaluate_row(
                model,
                family,
                config,
                condition_name=cond,
                directions=directions_by_family[family["id"]]["content"],
                alpha=alpha,
            )
            for family in config["families"]
        ]
        all_rows.extend(rows)
        by_condition[cond] = summarize(rows)

        cond_c = f"control_steer_a{alpha:g}"
        rows_c = [
            evaluate_row(
                model,
                family,
                config,
                condition_name=cond_c,
                directions=directions_by_family[family["id"]]["control"],
                alpha=alpha,
            )
            for family in config["families"]
        ]
        all_rows.extend(rows_c)
        by_condition[cond_c] = summarize(rows_c)

    summary = {
        "experiment_name": config["experiment_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "runtime_seconds": time.time() - started,
        "protocol": "residual_mlp_in_direction__force_native_b__steer_on_at_c",
        "patch_site": "mlp.hook_in (feature_input / residual into MLP; affine skip path)",
        "patch_layers": layers,
        "steering_alphas": alphas,
        "families": [f["id"] for f in config["families"]],
        "by_condition": by_condition,
    }
    summary["interpretation"] = interpret(summary)
    write_json(RESULTS_DIR / "summary.json", summary)
    write_json(RESULTS_DIR / "rows.json", all_rows)
    write_report(summary, RESULTS_DIR / "report.md")
    logging.info("Done. %s", summary["interpretation"])
    print(summary["interpretation"])


if __name__ == "__main__":
    main()
